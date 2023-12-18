// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversion_utils.h"
#ifdef __HIP_PLATFORM_AMD__
#include "hip/hip_cooperative_groups.h"
#else
#endif
#include "ds_kernel_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

#ifndef __HIP_PLATFORM_AMD__
#endif

namespace rot_half {
constexpr int threads = 256;
}  // namespace rot_half

template <typename T, int threadsPerHead, int granularity>
/*
DPCT1110:3: The total declared local variable size in device function apply_rotary_pos_half exceeds
128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to avoid high register
pressure.
*/
void apply_rotary_pos_half(T* mixed_query,
                           T* key_layer,
                           unsigned rotary_dim,
                           unsigned seq_len,
                           unsigned seq_offset,
                           unsigned num_heads,
                           unsigned head_size,
                           unsigned total_count,
                           float rope_theta,
                           int max_out_tokens)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    constexpr int T_per_thread = granularity / sizeof(T);
    constexpr int heads_per_block = rot_half::threads / threadsPerHead;

    sycl::group<3> tb = sycl::ext::oneapi::experimental::this_group<3>();
    auto head_group =
        sycl::ext::oneapi::experimental::this_sub_group();

    const int head_idx =
        item_ct1.get_group(2) * heads_per_block + item_ct1.get_local_id(2) / threadsPerHead;
    const int cur_seq_idx = head_idx % seq_len;
    const int offset = head_idx * head_size;
    const int k_offset = (cur_seq_idx + (head_idx / seq_len) * max_out_tokens) * head_size;

    const int seq_idx = cur_seq_idx + seq_offset;
    const int half_dim = rotary_dim >> 1;
    const int half_dim_threads = half_dim / T_per_thread;

    if (head_idx < total_count) {
        /*
        DPCT1007:0: Migration of thread_rank is not supported.
        */
        const int base_neuron_idx = head_group.get_local_linear_id() * T_per_thread;

        T q[T_per_thread], k[T_per_thread];
        mem_access::load_global<granularity>(q, mixed_query + offset + base_neuron_idx);
        mem_access::load_global<granularity>(k, key_layer + k_offset + base_neuron_idx);

#pragma unroll
        for (int i = 0; i < T_per_thread; i++) {
            const int neuron_idx = base_neuron_idx + i;
            if (neuron_idx < rotary_dim) {
                float inv_freq = (float)((neuron_idx % half_dim) * 2) / (float)rotary_dim;
                inv_freq = 1.0 / dpct::pow(rope_theta, inv_freq) * (float)seq_idx;

                float rotary_sign = (neuron_idx > (half_dim - 1) ? -1.0 : 1.0);
                float q_rot = conversion::to<float>(q[i]) * rotary_sign;
                float k_rot = conversion::to<float>(k[i]) * rotary_sign;

                const int target_lane = (neuron_idx < half_dim)
                                            /*
                                            DPCT1007:1: Migration of thread_rank is not supported.
                                            */
                                            ? head_group.get_local_linear_id() + half_dim_threads
                                            /*
                                            DPCT1007:2: Migration of thread_rank is not supported.
                                            */
                                            : head_group.get_local_linear_id() - half_dim_threads;

                /*
                DPCT1007:5: Migration of cooperative_groups::thread_block_tile::shfl is not
                supported.
                */
                const float q_rot_temp = head_group.shuffle(q_rot, target_lane);
                /*
                DPCT1007:6: Migration of cooperative_groups::thread_block_tile::shfl is not
                supported.
                */
                const float k_rot_temp = head_group.shuffle(k_rot, target_lane);

                q[i] = conversion::to<T>(conversion::to<float>(q[i]) * sycl::cos(inv_freq) +
                                         q_rot_temp * sycl::sin(inv_freq));
                k[i] = conversion::to<T>(conversion::to<float>(k[i]) * sycl::cos(inv_freq) +
                                         k_rot_temp * sycl::sin(inv_freq));
            }
        }

        mem_access::store_global<granularity>(mixed_query + offset + base_neuron_idx, q);
        mem_access::store_global<granularity>(key_layer + k_offset + base_neuron_idx, k);
    }
}

/*
DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define LAUNCH_ROT_POS_EMB_HALF(HEAD_THREADS, ALIGNMENT)                                          \
  {                                                                                               \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
    stream->submit([&](sycl::handler& cgh) {                                                      \
      T* mixed_query_ct0 = mixed_query;                                                           \
      T* key_layer_ct1 = key_layer;                                                               \
      auto rotary_dim_ct2 = rotary_dim;                                                           \
      auto seq_len_ct3 = seq_len;                                                                 \
      auto offset_ct4 = offset;                                                                   \
      auto num_heads_ct5 = num_heads;                                                             \
      auto head_size_ct6 = head_size;                                                             \
      auto total_count_ct7 = total_count;                                                         \
      auto rope_theta_ct8 = rope_theta;                                                           \
      auto max_out_tokens_ct9 = max_out_tokens;                                                   \
                                                                                                  \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                    \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {        \
                         apply_rotary_pos_half<T, HEAD_THREADS, ALIGNMENT>(mixed_query_ct0,       \
                                                                           key_layer_ct1,         \
                                                                           rotary_dim_ct2,        \
                                                                           seq_len_ct3,           \
                                                                           offset_ct4,            \
                                                                           num_heads_ct5,         \
                                                                           head_size_ct6,         \
                                                                           total_count_ct7,       \
                                                                           rope_theta_ct8,        \
                                                                           max_out_tokens_ct9);   \
                       });                                                                        \
    });                                                                                           \
  }

#ifdef __HIP_PLATFORM_AMD__
#define LAUNCH_FOR_ALIGNMENT(ALIGNMENT)         \
    if (threads_per_head == 4) {                \
        LAUNCH_ROT_POS_EMB_HALF(4, ALIGNMENT);  \
    } else if (threads_per_head == 8) {         \
        LAUNCH_ROT_POS_EMB_HALF(8, ALIGNMENT);  \
    } else if (threads_per_head == 16) {        \
        LAUNCH_ROT_POS_EMB_HALF(16, ALIGNMENT); \
    } else if (threads_per_head == 32) {        \
        LAUNCH_ROT_POS_EMB_HALF(32, ALIGNMENT); \
    } else if (threads_per_head == 64) {        \
        LAUNCH_ROT_POS_EMB_HALF(64, ALIGNMENT); \
    } else {                                    \
        assert(false);                          \
    }
#else
#define LAUNCH_FOR_ALIGNMENT(ALIGNMENT)         \
    if (threads_per_head == 4) {                \
        LAUNCH_ROT_POS_EMB_HALF(4, ALIGNMENT);  \
    } else if (threads_per_head == 8) {         \
        LAUNCH_ROT_POS_EMB_HALF(8, ALIGNMENT);  \
    } else if (threads_per_head == 16) {        \
        LAUNCH_ROT_POS_EMB_HALF(16, ALIGNMENT); \
    } else if (threads_per_head == 32) {        \
        LAUNCH_ROT_POS_EMB_HALF(32, ALIGNMENT); \
    } else {                                    \
        assert(false);                          \
    }
#endif

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 float rope_theta,
                                 dpct::queue_ptr stream,
                                 int max_out_tokens)
{
    const int half_dim = rotary_dim >> 1;

    int alignment = sizeof(T);
    if (half_dim % (16 / sizeof(T)) == 0) {
        alignment = 16;
    } else if (half_dim % (8 / sizeof(T)) == 0) {
        alignment = 8;
    } else if (half_dim % (4 / sizeof(T)) == 0) {
        alignment = 4;
    } else {
        assert(false);
    }
    const int T_per_elem = alignment / sizeof(T);

    int total_count = batch * num_heads * seq_len;

    const int padded_head_size = next_pow2(head_size);

    assert(padded_head_size <= hw_warp_size * T_per_elem);

    const int threads_per_head = padded_head_size / T_per_elem;
    const int heads_per_block = rot_half::threads / threads_per_head;

    sycl::range<3> block(1, 1, rot_half::threads);
    sycl::range<3> grid(1, 1, (total_count + heads_per_block - 1) / heads_per_block);

    if (alignment == 4) {
        LAUNCH_FOR_ALIGNMENT(4);
    } else if (alignment == 8) {
        LAUNCH_FOR_ALIGNMENT(8);
    } else if (alignment == 16) {
        LAUNCH_FOR_ALIGNMENT(16);
    } else {
        assert(false);
    }
}

#define INSTANTIATE_LAUNCH_ROTARY_POS_EMB(T)                      \
    template void launch_apply_rotary_pos_emb<T>(T*,              \
                                                 T*,              \
                                                 unsigned,        \
                                                 unsigned,        \
                                                 unsigned,        \
                                                 unsigned,        \
                                                 unsigned,        \
                                                 unsigned,        \
                                                 float,           \
                                                 dpct::queue_ptr, \
                                                 int);

INSTANTIATE_LAUNCH_ROTARY_POS_EMB(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_ROTARY_POS_EMB(sycl::half);
