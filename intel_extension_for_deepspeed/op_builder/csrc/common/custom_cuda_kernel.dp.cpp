// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "custom_cuda_layers.h"

void param_update_kernel(const float* input, sycl::half* output, int size)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if (id < size) { output[id] = (sycl::half)input[id]; }
}

void launch_param_update(const float* input, sycl::half* output, int size, dpct::queue_ptr stream)
{
    int threads = 1024;

    sycl::range<3> grid_dim(1, 1, (size - 1) / threads + 1);
    sycl::range<3> block_dim(1, 1, threads);

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](sycl::nd_item<3> item_ct1) {
                                 param_update_kernel(input, output, size);
                             });
    }
}

void param_update_kernel_half(const float* input, sycl::half* output, int size)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    sycl::half2* output_cast = reinterpret_cast<sycl::half2*>(output);
    if (id < size) {
        float input_f = input[id];
        sycl::half2* input_h = reinterpret_cast<sycl::half2*>(&input_f);
        output_cast[id] = *input_h;
    }
}

void launch_param_update_half(const float* input,
                              sycl::half* output,
                              int size,
                              dpct::queue_ptr stream)
{
    int threads = 1024;
    size /= 2;
    sycl::range<3> grid_dim(1, 1, (size - 1) / threads + 1);
    sycl::range<3> block_dim(1, 1, threads);

    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](sycl::nd_item<3> item_ct1) {
                                 param_update_kernel_half(input, output, size);
                             });
    }
}
