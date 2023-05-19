#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
using namespace sycl;
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
using namespace cl::sycl;
#else
#error "Unsupported compiler"
#endif
#include "conversion_utils.h"

inline float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanh(sqrt_param * (x + mul_param * x * x * x)));
}

inline float d_gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;

    float x2mul = x * x * mul_param;
    float tan_h = tanh(sqrt_param * (x + x * x2mul));
    float dg1 = 0.5f * (1.0f + tan_h);
    float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
    float dg3 = dg2 * 3 * x2mul;
    return (dg1 + dg2 + dg3);
}

template <typename T>
void gelu_kernel(const T* input, T* vals, int row_stride, int iterations, nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float input_f = conversion::to<float>(input[row * row_stride + i * loop_stride + id]);
            vals[row * row_stride + i * loop_stride + id] = conversion::to<T>(gelu(input_f));
        }
    }
}

template <typename T>
void fused_bias_gelu(const T* input,
                     const T* bias,
                     T* vals,
                     int row_stride,
                     int iterations,
                     nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float input_f = conversion::to<float>(input[row * row_stride + i * loop_stride + id]);
            float bias_f = conversion::to<float>(bias[i * loop_stride + id]);
            vals[row * row_stride + i * loop_stride + id] = conversion::to<T>(gelu(input_f + bias_f));
        }
    }
}

template <typename T>
void d_gelu_func(T* d_output,
                 const T* gelu_input,
                 const T* bias,
                 int row_stride,
                 int iterations,
                 nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    int loop_stride = item_ct1.get_local_range(2);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float d_output_f = conversion::to<float>(d_output[row * row_stride + i * loop_stride + id]);
            float gelu_input_f = conversion::to<float>(gelu_input[row * row_stride + i * loop_stride + id]);
            float bias_f = conversion::to<float>(bias[i * loop_stride + id]);
            float output_f = d_output_f * d_gelu(gelu_input_f + bias_f);
            d_output[row * row_stride + i * loop_stride + id] = conversion::to<T>(output_f);
        }
    }
}

template <typename T>
void launch_bias_gelu(const T* input,
                      const T* bias,
                      T* output,
                      int intermediate_size,
                      int batch_size,
                      queue stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations) + 1;
    range<3> block_dims(1, 1, threads);
    range<3> grid_dims(1, 1, batch_size);

    stream.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dims * block_dims, block_dims), [=](nd_item<3> item_ct1) {
            fused_bias_gelu(input, bias, output, intermediate_size, iterations, item_ct1);
        });
    });
}

template <typename T>
void launch_gelu(const T* input, T* output, int intermediate_size, int batch_size, queue stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations) + 1;
    range<3> block_dims(1, 1, threads);
    range<3> grid_dims(1, 1, batch_size);

    stream.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dims * block_dims, block_dims), [=](nd_item<3> item_ct1) {
            gelu_kernel(input, output, intermediate_size, iterations, item_ct1);
        });
    });
}

template void launch_bias_gelu<float>(const float*, const float*, float*, int, int, queue);
template void launch_bias_gelu<half>(const half*, const half*, half*, int, int, queue);
template void launch_bias_gelu<bf16>(const bf16*, const bf16*, bf16*, int, int, queue);

template void launch_gelu<float>(const float*, float*, int, int, queue);
template void launch_gelu<half>(const half*, half*, int, int, queue);
template void launch_gelu<bf16>(const bf16*, bf16*, int, int, queue);

template <typename T>
void launch_d_gelu(T* d_output,
                   const T* input,
                   const T* bias,
                   int intermediate_size,
                   int batch_size,
                   queue stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations) + 1;
    range<3> block_dims(1, 1, threads);
    range<3> grid_dims(1, 1, batch_size);

    stream.submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dims * block_dims, block_dims), [=](nd_item<3> item_ct1) {
            d_gelu_func(d_output, input, bias, intermediate_size, iterations, item_ct1);
        });
    });
}

template void launch_d_gelu<float>(float*, const float*, const float*, int, int, queue);
template void launch_d_gelu<half>(half*, const half*, const half*, int, int, queue);
template void launch_d_gelu<bf16>(bf16*, const bf16*, const bf16*, int, int, queue);
