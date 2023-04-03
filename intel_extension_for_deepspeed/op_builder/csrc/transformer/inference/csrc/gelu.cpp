/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "conversion_utils.h"
#include "compatible.h"
#include "memory_access_utils.h"

#define MAX_CAP 4
#define MAX_SEQ 2048

#ifdef __SYCL_DEVICE_ONLY__
inline float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}
#endif

/*
In-place gelu(biasAdd(x)) for channels last
*/
template <typename T>
__global__ void fused_bias_gelu(T* input, const T* bias, int total_count, int intermediate_size)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    // Input restriction: intermediate_size % vals_per_access == 0
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);

    /* const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_access; */
    const int offset = (pos.get_group(0) * pos.get_local_range(0) + pos.get_local_id(0)) * values_per_access;

    if (offset < total_count) {
        T data[values_per_access];
        T data_bias[values_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(data_bias, bias + (offset % intermediate_size));

#pragma unroll
        for (int i = 0; i < values_per_access; i++) {
            float data_f = conversion::to<float>(data[i]);
            float bias_f = conversion::to<float>(data_bias[i]);
            data[i] = conversion::to<T>(gelu(data_f + bias_f));
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
}

template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));

    fused_bias_gelu(input, bias, total_count, intermediate_size);
}

template void launch_bias_gelu<float>(float*, const float*, int, int);
template void launch_bias_gelu<bf16>(bf16*,
                                     const bf16*,
                                     int,
                                     int);
template void launch_bias_gelu<half>(half*, const half*, int, int);

/*
In-place channels-last bias add
*/
template <typename T>
__global__ void fused_bias_add(T* input, const T* bias, int total_count, int intermediate_size)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    // Input restriction: intermediate_size % vals_per_access == 0
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);
    const int offset = (pos.get_group(0) * pos.get_local_range(0) + pos.get_local_id(0)) * values_per_access;

    if (offset < total_count) {
        T data[values_per_access];
        T data_bias[values_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(data_bias, bias + (offset % intermediate_size));

#pragma unroll
        for (int i = 0; i < values_per_access; i++) {
            float data_f = conversion::to<float>(data[i]);
            float bias_f = conversion::to<float>(data_bias[i]);
            data[i] = conversion::to<T>(data_f + bias_f);
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
}

template <typename T>
void launch_bias_add(T* input,
                     const T* bias,
                     int intermediate_size,
                     int batch_size)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));

    fused_bias_add(input, bias, total_count, intermediate_size);
}

template void launch_bias_add<float>(float*, const float*, int, int);
template void launch_bias_add<bf16>(bf16*,
                                    const bf16*,
                                    int,
                                    int);
template void launch_bias_add<half>(half*, const half*, int, int);

__global__ void fused_bias_residual(float* residual,
                                    const float* hidden_state,
                                    const float* attn,
                                    const float* bias,
                                    const float* attn_bias,
                                    const int total_count,
                                    const int intermediate_size,
                                    const float mp_scale,
                                    const bool preln)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    
    float4* res_fl4_ptr = reinterpret_cast<float4*>(residual);
    const float4* hs_fl4_ptr = reinterpret_cast<const float4*>(hidden_state);
    const float4* attn_fl4_ptr = reinterpret_cast<const float4*>(attn);
    const float4* bias_fl4_ptr = reinterpret_cast<const float4*>(bias);
    const float4* attn_bias_fl4_ptr = reinterpret_cast<const float4*>(attn_bias);
    /* const int offset = blockIdx.x * blockDim.x + threadIdx.x; */
    const int offset = pos.get_group(0) * pos.get_local_range(0) + pos.get_local_id(0);

    if (offset < total_count) {
        float4 res_fl4 = res_fl4_ptr[offset];
        const float4 hs_fl4 = hs_fl4_ptr[offset];
        const float4 attn_fl4 = attn_fl4_ptr[offset];
        const float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];
        const float4 attn_bias_fl4 = attn_bias_fl4_ptr[offset % intermediate_size];
        if (preln) {
            // residual = (residual + attention + bias + attention_bias) *
            // mp_scale + hidden_state
            res_fl4.x() =
                (res_fl4.x() + attn_fl4.x() + bias_fl4.x() + attn_bias_fl4.x()) * mp_scale + (hs_fl4.x());
            res_fl4.y() =
                (res_fl4.y() + attn_fl4.y() + bias_fl4.y() + attn_bias_fl4.y()) * mp_scale + (hs_fl4.y());
            res_fl4.z() =
                (res_fl4.z() + attn_fl4.z() + bias_fl4.z() + attn_bias_fl4.z()) * mp_scale + (hs_fl4.z());
            res_fl4.w() =
                (res_fl4.w() + attn_fl4.w() + bias_fl4.w() + attn_bias_fl4.w()) * mp_scale + (hs_fl4.w());
        } else {
            // residual += hidden_state + bias
            res_fl4.x() = res_fl4.x() + hs_fl4.x() + bias_fl4.x();
            res_fl4.y() = res_fl4.y() + hs_fl4.y() + bias_fl4.y();
            res_fl4.z() = res_fl4.z() + hs_fl4.z() + bias_fl4.z();
            res_fl4.w() = res_fl4.w() + hs_fl4.w() + bias_fl4.w();
        }
        res_fl4_ptr[offset] = res_fl4;
    }
}

template <typename T>
__global__ void fused_bias_residual(T* residual,
                                    const T* hidden_state,
                                    const T* attn,
                                    const T* bias,
                                    const T* attn_bias,
                                    const int total_count,
                                    const int intermediate_size,
                                    const float mp_scale,
                                    const bool preln)
{
    using T2 =
        typename std::conditional<std::is_same<T, half>::value, half2, bf162>::type;

    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    
    float2* res_fl2_ptr = reinterpret_cast<float2*>(residual);
    const float2* hs_fl2_ptr = reinterpret_cast<const float2*>(hidden_state);
    const float2* attn_fl2_ptr = reinterpret_cast<const float2*>(attn);
    const float2* bias_fl2_ptr = reinterpret_cast<const float2*>(bias);
    const float2* attn_bias_fl2_ptr = reinterpret_cast<const float2*>(attn_bias);
    const int offset = pos.get_group(0) * pos.get_local_range(0) + pos.get_local_id(0);

    if (offset < total_count) {
        float2 res_fl2 = res_fl2_ptr[offset];
        const float2 hs_fl2 = hs_fl2_ptr[offset];
        const float2 attn_fl2 = attn_fl2_ptr[offset];
        const float2 bias_fl2 = bias_fl2_ptr[offset % intermediate_size];
        const float2 attn_bias_fl2 = attn_bias_fl2_ptr[offset % intermediate_size];

        T2* res_half2 = reinterpret_cast<T2*>(&res_fl2);
        const T2* hs_half2 = reinterpret_cast<const T2*>(&hs_fl2);
        const T2* attn_half2 = reinterpret_cast<const T2*>(&attn_fl2);
        const T2* bias_half2 = reinterpret_cast<const T2*>(&bias_fl2);
        const T2* attn_bias_half2 = reinterpret_cast<const T2*>(&attn_bias_fl2);

        float2 res_low = conversion::to<float2>(res_half2[0]);
        float2 res_high = conversion::to<float2>(res_half2[1]);

        const float2 hs_low = conversion::to<float2>(hs_half2[0]);
        const float2 hs_high = conversion::to<float2>(hs_half2[1]);

        const float2 attn_low = conversion::to<float2>(attn_half2[0]);
        const float2 attn_high = conversion::to<float2>(attn_half2[1]);

        const float2 bias_low = conversion::to<float2>(bias_half2[0]);
        const float2 bias_high = conversion::to<float2>(bias_half2[1]);

        const float2 attn_bias_low = conversion::to<float2>(attn_bias_half2[0]);
        const float2 attn_bias_high = conversion::to<float2>(attn_bias_half2[1]);

        if (preln) {
            // residual = (residual + attention + bias + attention_bias) *
            // mp_scale + hidden_state
            res_low.x() =
                (res_low.x() + attn_low.x() + bias_low.x() + attn_bias_low.x()) * mp_scale + hs_low.x();
            res_low.y() =
                (res_low.y() + attn_low.y() + bias_low.y() + attn_bias_low.y()) * mp_scale + hs_low.y();
            res_high.x() =
                (res_high.x() + attn_high.x() + bias_high.x() + attn_bias_high.x()) * mp_scale + hs_high.x();
            res_high.y() =
                (res_high.y() + attn_high.y() + bias_high.y() + attn_bias_high.y()) * mp_scale + hs_high.y();
        } else {
            // residual += hidden_state + bias
            res_low.x() = (res_low.x() + hs_low.x() + bias_low.x());
            res_low.y() = (res_low.y() + hs_low.y() + bias_low.y());
            res_high.x() = (res_high.x() + hs_high.x() + bias_high.x());
            res_high.y() = (res_high.y() + hs_high.y() + bias_high.y());
        }
        res_half2[0] = conversion::to<T2>(res_low);
        res_half2[1] = conversion::to<T2>(res_high);

        res_fl2_ptr[offset] = res_fl2;
    }
}

template <typename T>
void launch_bias_residual(T* residual,
                          T* hidden_state,
                          T* attn,
                          T* bias,
                          T* attn_bias,
                          int batch,
                          int hidden_dim,
                          int mp_size,
                          bool preln)
{
    int total_count = batch * hidden_dim / 4;

    fused_bias_residual(residual,
                        hidden_state,
                        attn,
                        bias,
                        attn_bias,
                        total_count,
                        hidden_dim / 4,
                        1.0 / mp_size,
                        preln);
}

template void launch_bias_residual<
    float>(float*, float*, float*, float*, float*, int, int, int, bool);
template void launch_bias_residual<bf16>(bf16*,
                                         bf16*,
                                         bf16*,
                                         bf16*,
                                         bf16*,
                                         int,
                                         int,
                                         int,
                                         bool);
template void launch_bias_residual<
    half>(half*, half*, half*, half*, half*, int, int, int, bool);

template __global__ void fused_bias_residual(bf16* residual,
                                             const bf16* hidden_state,
                                             const bf16* attn,
                                             const bf16* bias,
                                             const bf16* attn_bias,
                                             const int total_count,
                                             const int intermediate_size,
                                             const float mp_scale,
                                             const bool preln);

template __global__ void fused_bias_residual(half* residual,
                                             const half* hidden_state,
                                             const half* attn,
                                             const half* bias,
                                             const half* attn_bias,
                                             const int total_count,
                                             const int intermediate_size,
                                             const float mp_scale,
                                             const bool preln);

__global__ void gptj_residual_add(float* residual,
                                  const float* hidden_state,
                                  const float* attn,
                                  const float* bias,
                                  const float* attn_bias,
                                  const int total_count,
                                  const int intermediate_size,
                                  const float mp_scale)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    
    float4* res_fl4_ptr = reinterpret_cast<float4*>(residual);
    const float4* hs_fl4_ptr = reinterpret_cast<const float4*>(hidden_state);
    const float4* attn_fl4_ptr = reinterpret_cast<const float4*>(attn);
    const float4* bias_fl4_ptr = reinterpret_cast<const float4*>(bias);
    const float4* attn_bias_fl4_ptr = reinterpret_cast<const float4*>(attn_bias);
    /* const int offset = blockIdx.x * blockDim.x + threadIdx.x; */
    const int offset = pos.get_group(0) * pos.get_local_range(0) + pos.get_local_id(0);

    if (offset < total_count) {
        float4 res_fl4 = res_fl4_ptr[offset];
        const float4 hs_fl4 = hs_fl4_ptr[offset];
        const float4 attn_fl4 = attn_fl4_ptr[offset];
        const float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];

        if (attn_bias) {
            float4 attn_bias_fl4 = attn_bias_fl4_ptr[offset % intermediate_size];
            // residual += attention_bias
            res_fl4.x() += attn_bias_fl4.x();
            res_fl4.y() += attn_bias_fl4.y();
            res_fl4.z() += attn_bias_fl4.z();
            res_fl4.w() += attn_bias_fl4.w();
        }
        // residual = hidden_state + attention + (residual + bias) * mp_scale
        res_fl4.x() = hs_fl4.x() + attn_fl4.x() + (res_fl4.x() + bias_fl4.x()) * mp_scale;
        res_fl4.y() = hs_fl4.y() + attn_fl4.y() + (res_fl4.y() + bias_fl4.y()) * mp_scale;
        res_fl4.z() = hs_fl4.z() + attn_fl4.z() + (res_fl4.z() + bias_fl4.z()) * mp_scale;
        res_fl4.w() = hs_fl4.w() + attn_fl4.w() + (res_fl4.w() + bias_fl4.w()) * mp_scale;

        res_fl4_ptr[offset] = res_fl4;
    }
}

template <typename T>
__global__ void gptj_residual_add(T* residual,
                                  const T* hidden_state,
                                  const T* attn,
                                  const T* bias,
                                  const T* attn_bias,
                                  const int total_count,
                                  const int intermediate_size,
                                  const float mp_scale)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    
    using T2 =
        typename std::conditional<std::is_same<T, half>::value, half2, bf162>::type;
    float2* res_fl2_ptr = reinterpret_cast<float2*>(residual);
    const float2* hs_fl2_ptr = reinterpret_cast<const float2*>(hidden_state);
    const float2* attn_fl2_ptr = reinterpret_cast<const float2*>(attn);
    const float2* bias_fl2_ptr = reinterpret_cast<const float2*>(bias);
    const float2* attn_bias_fl2_ptr = reinterpret_cast<const float2*>(attn_bias);
    /* const int offset = blockIdx.x * blockDim.x + threadIdx.x; */
    const int offset = pos.get_group(0) * pos.get_local_range(0) + pos.get_local_id(0);

    if (offset < total_count) {
        float2 res_fl2 = res_fl2_ptr[offset];
        const float2 hs_fl2 = hs_fl2_ptr[offset];
        const float2 attn_fl2 = attn_fl2_ptr[offset];
        const float2 bias_fl2 = bias_fl2_ptr[offset % intermediate_size];

        T2* res_half2 = reinterpret_cast<T2*>(&res_fl2);
        const T2* hs_half2 = reinterpret_cast<const T2*>(&hs_fl2);
        const T2* attn_half2 = reinterpret_cast<const T2*>(&attn_fl2);
        const T2* bias_half2 = reinterpret_cast<const T2*>(&bias_fl2);

        float2 res_low = conversion::to<float2>(res_half2[0]);
        float2 res_high = conversion::to<float2>(res_half2[1]);

        const float2 hs_low = conversion::to<float2>(hs_half2[0]);
        const float2 hs_high = conversion::to<float2>(hs_half2[1]);

        const float2 attn_low = conversion::to<float2>(attn_half2[0]);
        const float2 attn_high = conversion::to<float2>(attn_half2[1]);

        const float2 bias_low = conversion::to<float2>(bias_half2[0]);
        const float2 bias_high = conversion::to<float2>(bias_half2[1]);

        if (attn_bias) {
            const float2 attn_bias_fl2 = attn_bias_fl2_ptr[offset % intermediate_size];
            const T2* attn_bias_half2 = reinterpret_cast<const T2*>(&attn_bias_fl2);
            const float2 attn_bias_low = conversion::to<float2>(attn_bias_half2[0]);
            const float2 attn_bias_high = conversion::to<float2>(attn_bias_half2[1]);
            // residual += attention_bias
            res_low.x() += attn_bias_low.x();
            res_low.y() += attn_bias_low.y();
            res_high.x() += attn_bias_high.x();
            res_high.y() += attn_bias_high.y();
        }
        // residual = hidden_state + attention + (residual + bias) * mp_scale
        res_low.x() = attn_low.x() + hs_low.x() + (res_low.x() + bias_low.x()) * mp_scale;
        res_low.y() = attn_low.y() + hs_low.y() + (res_low.y() + bias_low.y()) * mp_scale;
        res_high.x() = attn_high.x() + hs_high.x() + (res_high.x() + bias_high.x()) * mp_scale;
        res_high.y() = attn_high.y() + hs_high.y() + (res_high.y() + bias_high.y()) * mp_scale;

        res_half2[0] = conversion::to<T2>(res_low);
        res_half2[1] = conversion::to<T2>(res_high);

        res_fl2_ptr[offset] = res_fl2;
    }
}

template <typename T>
void launch_gptj_residual_add(T* residual,
                              T* hidden_state,
                              T* attn,
                              T* bias,
                              T* attn_bias,
                              int hidden_dim,
                              int batch,
                              int mp_size)
{
    int total_count = batch * hidden_dim / 4;

    gptj_residual_add(
        residual, hidden_state, attn, bias, attn_bias, total_count, hidden_dim / 4, 1.0 / mp_size);
}

template void launch_gptj_residual_add<float>(float*,
                                              float*,
                                              float*,
                                              float*,
                                              float*,
                                              int,
                                              int,
                                              int);

template void launch_gptj_residual_add<bf16>(bf16*,
                                             bf16*,
                                             bf16*,
                                             bf16*,
                                             bf16*,
                                             int,
                                             int,
                                             int);

template void launch_gptj_residual_add<half>(half*,
                                               half*,
                                               half*,
                                               half*,
                                               half*,
                                               int,
                                               int,
                                               int);

template __global__ void gptj_residual_add(bf16* residual,
                                           const bf16* hidden_state,
                                           const bf16* attn,
                                           const bf16* bias,
                                           const bf16* attn_bias,
                                           const int total_count,
                                           const int intermediate_size,
                                           const float mp_scale);

template __global__ void gptj_residual_add(half* residual,
                                           const half* hidden_state,
                                           const half* attn,
                                           const half* bias,
                                           const half* attn_bias,
                                           const int total_count,
                                           const int intermediate_size,
                                           const float mp_scale);

template <typename T>
__global__ void moe_res_matmul(T* residual, T* coef, T* mlp_out, int seq_len, int hidden_dim)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(T);

    T* residual_seq = residual + pos.get_group(0) * hidden_dim;
    T* mlp_out_seq = mlp_out + pos.get_group(0) * hidden_dim;

    for (unsigned tid = pos.get_local_id(0) * vals_per_access; tid < hidden_dim;
         tid += pos.get_local_range(0) * vals_per_access) {
        T mlp[vals_per_access];
        T res[vals_per_access];
        T coef1[vals_per_access];
        T coef2[vals_per_access];

        mem_access::load_global<granularity>(mlp, mlp_out_seq + tid);
        mem_access::load_global<granularity>(res, residual_seq + tid);
        mem_access::load_global<granularity>(coef1, coef + tid);
        mem_access::load_global<granularity>(coef2, coef + tid + hidden_dim);

#pragma unroll
        for (int idx = 0; idx < vals_per_access; idx++) {
            mlp[idx] = mlp[idx] * coef2[idx] + res[idx] * coef1[idx];
        }

        mem_access::store_global<granularity>(mlp_out_seq + tid, mlp);
    }
}

template <typename T>
void launch_moe_res_matmul(T* residual,
                           T* coef,
                           T* mlp_out,
                           int seq_len,
                           int hidden_dim)
{
    moe_res_matmul(residual, coef, mlp_out, seq_len, hidden_dim);
}

template void launch_moe_res_matmul(float* residual,
                                    float* coef,
                                    float* mlp_out,
                                    int seq_len,
                                    int hidden_dim);

template void launch_moe_res_matmul(bf16* residual,
                                    bf16* coef,
                                    bf16* mlp_out,
                                    int seq_len,
                                    int hidden_dim);

template void launch_moe_res_matmul(half* residual,
                                    half* coef,
                                    half* mlp_out,
                                    int seq_len,
                                    int hidden_dim);

template <typename T>
__global__ void pad_data_kernel(T* padded_output, T* output, int head_size, int padded_head_size)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    
    using T2 =
        typename std::conditional<std::is_same<T, half>::value, half2, bf162>::type;
    float4* padded_output_cast = reinterpret_cast<float4*>(padded_output);
    float4* output_cast = reinterpret_cast<float4*>(output);
    // TODO: blockDim.y replacement
    int bid = pos.get_group(0) * (pos.get_local_range(1)) + pos.get_local_id(1);
    int idx = pos.get_local_id(0);
    padded_output_cast += (bid * padded_head_size);
    output_cast += (bid * head_size);
    float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++) ZERO_h[i] = zero_h;
    if (idx < head_size)
        padded_output_cast[idx] = output_cast[idx];
    else
        padded_output_cast[idx] = ZERO;
}

__global__ void pad_data_kernel(float* padded_output,
                                float* output,
                                int head_size,
                                int padded_head_size)
{
}

template <typename T>
void pad_data(T* padded_output,
              T* output,
              int bsz,
              int head_size,
              int padded_head_size)
{
    pad_data_kernel(padded_output, output, head_size / 8, padded_head_size / 8);
}
template void pad_data(half* padded_output,
                       half* output,
                       int bsz,
                       int head_size,
                       int padded_head_size);

template void pad_data(bf16* padded_output,
                       bf16* output,
                       int bsz,
                       int head_size,
                       int padded_head_size);

template void pad_data(float* padded_output,
                       float* output,
                       int bsz,
                       int head_size,
                       int padded_head_size);

template __global__ void pad_data_kernel(bf16* padded_output,
                                         bf16* output,
                                         int head_size,
                                         int padded_head_size);

template __global__ void pad_data_kernel(half* padded_output,
                                         half* output,
                                         int head_size,
                                         int padded_head_size);

template <typename T>
__global__ void pad_head_seq_kernel(T* padded_output,
                                    T* output,
                                    int seq_len,
                                    int padded_seq_len,
                                    int head_size,
                                    int padded_head_size)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    
    using T2 =
        typename std::conditional<std::is_same<T, half>::value, half2, bf162>::type;
    float4* padded_output_cast = reinterpret_cast<float4*>(padded_output);
    float4* output_cast = reinterpret_cast<float4*>(output);
    /* const int offset = (pos.get_group(0) * pos.get_local_range(0) + pos.get_local_id(0)) * values_per_access; */
    int bsz = pos.get_group(0);
    int bid = pos.get_group(0) * (pos.get_local_range(1)) + pos.get_local_id(1);
    int idx = pos.get_local_id(0);
    padded_output_cast += (bsz * padded_seq_len + bid) * padded_head_size;
    output_cast += (bsz * seq_len + bid) * head_size;
    float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++) ZERO_h[i] = zero_h;

    if (idx < head_size && bid < seq_len)
        padded_output_cast[idx] = output_cast[idx];
    else
        padded_output_cast[idx] = ZERO;
}

__global__ void pad_head_seq_kernel(float* padded_output,
                                    float* output,
                                    int seq_len,
                                    int padded_seq_len,
                                    int head_size,
                                    int padded_head_size)
{
}

template <typename T>
void pad_head_seq(T* padded_output,
                  T* output,
                  int bsz,
                  int seq_len,
                  int padded_seq_len,
                  int head_size,
                  int padded_head_size)
{
    pad_head_seq_kernel(
        padded_output, output, seq_len, padded_seq_len, head_size / 8, padded_head_size / 8);
}

template void pad_head_seq(half* padded_output,
                           half* output,
                           int bsz,
                           int seq_len,
                           int padded_seq_len,
                           int head_size,
                           int padded_head_size);

template void pad_head_seq(bf16* padded_output,
                           bf16* output,
                           int bsz,
                           int seq_len,
                           int padded_seq_len,
                           int head_size,
                           int padded_head_size);

template void pad_head_seq(float* padded_output,
                           float* output,
                           int bsz,
                           int seq_len,
                           int padded_seq_len,
                           int head_size,
                           int padded_head_size);

// TODO(cmikeh2): evaluate different GeLU performance
#ifdef __SYCL_DEVICE_ONLY__
inline float old_gelu(float val)
{
    // 1 / sqrt(2)
    constexpr float rsqrt_2 = 0.707106769084930419922;
    return val * 0.5f * (1.0f + erff(val * rsqrt_2));
}
#endif

namespace fused_geglu {
constexpr int threads = 256;
constexpr int steps = 2;
constexpr int granularity = 16;
}  // namespace fused_geglu

template <typename T>
__global__ void fused_bias_geglu(T* output,
                                 const T* activation,
                                 const T* bias,
                                 int base_channels,
                                 int total_elems)
{
    auto pos = sycl::ext::oneapi::experimental::this_nd_item<1>();
    
    constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
    constexpr int T_per_step = T_per_access * fused_geglu::threads;
    constexpr int T_per_block = T_per_step * fused_geglu::steps;

    const int id = pos.get_group(0) * T_per_block + pos.get_local_id(0) * T_per_access;

#pragma unroll
    for (int i = 0; i < fused_geglu::steps; i++) {
        T activation_buffer_1[T_per_access];
        T activation_buffer_2[T_per_access];
        T bias_buffer_1[T_per_access];
        T bias_buffer_2[T_per_access];

        const int iter_id = id + T_per_step * i;
        if (iter_id < total_elems) {
            const int channel_id = iter_id % base_channels;
            const int seq_id = iter_id / base_channels;
            const int seq_offset = seq_id * base_channels * 2;

            mem_access::load_global<fused_geglu::granularity>(activation_buffer_1,
                                                              activation + seq_offset + channel_id);
            mem_access::load_global<fused_geglu::granularity>(
                activation_buffer_2, activation + seq_offset + channel_id + base_channels);
            mem_access::load_global<fused_geglu::granularity>(bias_buffer_1, bias + channel_id);
            mem_access::load_global<fused_geglu::granularity>(bias_buffer_2,
                                                              bias + channel_id + base_channels);

            // Since the GeLU is going to happen at float, might as well
            // convert
#pragma unroll
            for (int v = 0; v < T_per_access; v++) {
                T hidden_state = activation_buffer_1[v] + bias_buffer_1[v];
                T pre_gate = activation_buffer_2[v] + bias_buffer_2[v];
                float gate_f = old_gelu(conversion::to<float>(pre_gate));
                T gate = conversion::to<T>(gate_f);
                activation_buffer_1[v] = hidden_state * gate;
            }

            mem_access::store_global<fused_geglu::granularity>(output + iter_id,
                                                               activation_buffer_1);
        }
    }
}

template <typename T>
void launch_fused_bias_geglu(T* output,
                             const T* activation,
                             const T* bias,
                             int rows,
                             int elems_per_row)
{
    /*
    Fused bias GEGLU is a variant of the gated activation functions.
    The input here is a matrix of [batch, seq_len, 2 * intermediate_dim]
    where the second half of the channels act as GeLU gates for the first
    half.
    */

    // Re-derive the above figures
    constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
    constexpr int T_per_step = T_per_access * fused_geglu::threads;
    constexpr int T_per_block = T_per_step * fused_geglu::steps;

    const int base_channels = elems_per_row / 2;
    const int total_elems = base_channels * rows;


    fused_bias_geglu(output, activation, bias, base_channels, total_elems);
}

template void launch_fused_bias_geglu(half*,
                                      const half*,
                                      const half*,
                                      int,
                                      int);
template void launch_fused_bias_geglu(bf16*,
                                      const bf16*,
                                      const bf16*,
                                      int,
                                      int);
template void launch_fused_bias_geglu(float*, const float*, const float*, int, int);
