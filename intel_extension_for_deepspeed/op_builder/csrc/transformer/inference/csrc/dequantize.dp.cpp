// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversion_utils.h"
#include "inference_cuda_layers.h"

#define MAX_QUANTIZE_GROUPING 1024

#define loop_unroll 1
#define loop_unroll_bits 1

template <typename T>
void dequantize_kernel(T* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int output_size,
                                  int hidden_dim,
                                  int groups,
                                  int merge_count)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    unsigned merge_hidden = hidden_dim >> merge_count;
    unsigned quantization_stride = (merge_hidden * output_size) / groups;

    unsigned bid = item_ct1.get_group(2);
    unsigned tid = item_ct1.get_local_id(2);

    while (tid < output_size) {
        unsigned w_index = bid / merge_hidden;
        unsigned q_index = tid + bid * output_size;

        auto q = input[q_index];

        unsigned merge_hidden_total = w_index * merge_hidden;
        unsigned scale_index =
            ((((bid - merge_hidden_total) + tid * merge_hidden) / quantization_stride)
             << merge_count) +
            w_index;

        float scale_data = qscale[scale_index];

        output[q_index] = conversion::to<T>(scale_data * (float)q);
        tid += item_ct1.get_local_range(2);
    }
}

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       unsigned merge_count,
                       dpct::queue_ptr stream)
{
    unsigned threads = 1024;
    sycl::range<3> block_dims(1, 1, threads);
    sycl::range<3> grid_dims(1, 1, hidden_dim);

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64, sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(grid_dims * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                dequantize_kernel(
                    output, input, qscale, output_size, hidden_dim, groups, merge_count);
            });
    }
}

#define INSTANTIATE_DEQUANTIZE_MERGE(T) \
    template void launch_dequantize<T>( \
        T*, const int8_t*, const float*, unsigned, unsigned, unsigned, unsigned, dpct::queue_ptr);

INSTANTIATE_DEQUANTIZE_MERGE(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_DEQUANTIZE_MERGE(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_DEQUANTIZE_MERGE(sycl::half);

void dequantize_kernel(float* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
}

template <typename T>
void dequantize_kernel(T* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  unsigned hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    unsigned bid = item_ct1.get_group(2) * item_ct1.get_group_range(1) + item_ct1.get_group(1);
    unsigned tid = item_ct1.get_local_id(2);

    float local_scale = qscale[item_ct1.get_group(2)];

    const float* input_cast = reinterpret_cast<const float*>(input);
    sycl::float2* output_cast = reinterpret_cast<sycl::float2*>(output);

    input_cast += bid * merge_hidden;
    output_cast += bid * merge_hidden;

    for (int c = 0; c < cnt; c++) {
        if (tid < merge_hidden) {
            float q = input_cast[tid];
            int8_t* q_int8 = (int8_t*)&q;

            sycl::float2 q_f;
            T* q_h = (T*)&q_f;

            q_h[0] = conversion::to<T>(local_scale * (float)q_int8[0]);
            q_h[1] = conversion::to<T>(local_scale * (float)q_int8[1]);
            q_h[2] = conversion::to<T>(local_scale * (float)q_int8[2]);
            q_h[3] = conversion::to<T>(local_scale * (float)q_int8[3]);
            output_cast[tid] = q_f;
            tid += item_ct1.get_local_range(2);
        }
    }
}

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       dpct::queue_ptr stream)
{
    unsigned threads = 1024;
    hidden_dim /= 4;
    unsigned thd_cnt = (hidden_dim - 1) / threads + 1;

    assert(output_size % groups == 0);
    unsigned blocks = output_size / groups;

    sycl::range<3> block_dims(1, 1, threads);
    sycl::range<3> grid_dims(1, blocks, groups);

    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64, sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(grid_dims * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                dequantize_kernel(output, input, qscale, hidden_dim, hidden_dim, thd_cnt);
            });
    }
}

#define INSTANTIATE_DEQUANTIZE_NO_MERGE(T) \
    template void launch_dequantize<T>(    \
        T*, const int8_t*, const float*, unsigned, unsigned, unsigned, dpct::queue_ptr);

INSTANTIATE_DEQUANTIZE_NO_MERGE(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_DEQUANTIZE_NO_MERGE(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_DEQUANTIZE_NO_MERGE(sycl::half);
