// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

#define MAX_CAP 4
#define MAX_SEQ 2048

inline float relu(const float x) { return x < 0 ? 0 : x; }

/*
In-place relu(biasAdd(x)) for channels last
*/
template <typename T>
void fused_bias_relu(T* input, const T* bias, int total_count, int intermediate_size)
{
    // Input restriction: intermediate_size % vals_per_access == 0
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);
    const int offset =
        (item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2)) *
        values_per_access;

    if (offset < total_count) {
        T data[values_per_access];
        T data_bias[values_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(
            data_bias, bias + (offset % intermediate_size), bias != nullptr);

#pragma unroll
        for (int i = 0; i < values_per_access; i++) {
            float data_f = conversion::to<float>(data[i]);
            float bias_f = conversion::to<float>(data_bias[i]);
            data[i] = conversion::to<T>(relu(data_f + bias_f));
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
}

template <typename T>
void launch_bias_relu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      dpct::queue_ptr stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    sycl::range<3> block_dims(1, 1, threads);
    sycl::range<3> grid_dims(1, 1, (total_count + elems_per_block - 1) / elems_per_block);

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64, sycl::aspect::fp16});
        stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) {
                                 fused_bias_relu(input, bias, total_count, intermediate_size);
                             });
    }
}

#define INSTANTIATE_LAUNCH_BIAS_RELU(T) \
    template void launch_bias_relu<T>(T*, const T*, int, int, dpct::queue_ptr);

INSTANTIATE_LAUNCH_BIAS_RELU(float)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_RELU(sycl::ext::oneapi::bfloat16)
#endif
INSTANTIATE_LAUNCH_BIAS_RELU(sycl::half)
