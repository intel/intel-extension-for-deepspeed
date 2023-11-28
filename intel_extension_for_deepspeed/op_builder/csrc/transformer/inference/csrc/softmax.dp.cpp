// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>
#include "device.hpp"
#include <limits>
#include "conversion_utils.h"
#include "inference_sycl_layers.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define MAX_REG_SIZE 8

#define minus_infinity -10000.0


template <typename T, int iterations>
void attn_softmax_v2(T* vals,
                     T* mask,
                     T* alibi,
                     float layer_scale,
                     bool triangular,
                     bool recompute,
                     bool local_attention,
                     int window_size,
                     int total_count,
                     int heads,
                     int sequence_length,
                     int num_seq,
                     int head_offset,
                     int mask_stride,
                     int mp_size,
                     int reduceWidth)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    sycl::float2 low_data[MAX_REG_SIZE];
    sycl::float2 high_data[MAX_REG_SIZE];
    const T zero_h = conversion::to<T>(0.f);

    int wid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = item_ct1.get_local_id(2) % reduceWidth;

    auto& partialSum = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MAX_WARP_NUM]>(
        sycl::ext::oneapi::experimental::this_group<3>());

    int iter_offset = item_ct1.get_group(2) * (warp_num / reduce_blocks) + (wid / reduce_blocks);
    int batch_idx = iter_offset / (num_seq * heads);
    int alibi_offset = batch_idx * heads * mp_size + head_offset;
    int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);

    if (iter_offset < total_count) {
        vals += (iter_offset * sequence_length);

        alibi_offset = (alibi_offset + ((iter_offset / num_seq) % heads)) * sequence_length;
        mask_offset = mask_offset * sequence_length;
        int seq_id = iter_offset % num_seq;

        int real_seq_id = seq_id + (num_seq == sequence_length ? 0 : sequence_length);
        int window_stride4 = (local_attention && (real_seq_id >> 2) > (window_size >> 2))
                                 ? (real_seq_id >> 2) - (window_size >> 2)
                                 : 0;
        int window_stride =
            (local_attention && real_seq_id >= window_size) ? real_seq_id - window_size : -1;

        float max_val = minus_infinity;
        // if (lane == 0) printf("%d, %d: %d \n", wid, blockIdx.x, mask_offset);
        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            bool check = (data_id >> 2) >= window_stride4;
            bool low_x_check = check && (data_id < sequence_length) &&
                               (!triangular || (data_id <= seq_id)) && (data_id > window_stride);
            bool low_y_check = check && ((data_id + reduceWidth) < sequence_length) &&
                               (!triangular || ((data_id + reduceWidth) <= seq_id)) &&
                               ((data_id + reduceWidth) > window_stride);
            bool high_x_check = check && ((data_id + reduceWidth * 2) < sequence_length) &&
                                (!triangular || ((data_id + reduceWidth * 2) <= seq_id)) &&
                                ((data_id + reduceWidth * 2) > window_stride);
            bool high_y_check = check && ((data_id + reduceWidth * 3) < sequence_length) &&
                                (!triangular || ((data_id + reduceWidth * 3) <= seq_id)) &&
                                ((data_id + reduceWidth * 3) > window_stride);

            if (mask && alibi) {
                low_data[i].x() = low_x_check
                                      ? conversion::to<float>(vals[data_id]) * layer_scale +
                                            (conversion::to<float>(alibi[data_id + alibi_offset])) +
                                            (conversion::to<float>(mask[data_id + mask_offset]))
                                      : minus_infinity;
                low_data[i].y() =
                    low_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale +
                              (conversion::to<float>(alibi[data_id + alibi_offset + reduceWidth])) +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth]))
                        : minus_infinity;
                high_data[i].x() =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 2])) +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 2]))
                        : minus_infinity;
                high_data[i].y() =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 3])) +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 3]))
                        : minus_infinity;
            } else if (mask) {
                low_data[i].x() = low_x_check
                                      ? conversion::to<float>(vals[data_id]) * layer_scale +
                                            (conversion::to<float>(mask[data_id + mask_offset]))
                                      : minus_infinity;
                low_data[i].y() =
                    low_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth]))
                        : minus_infinity;
                high_data[i].x() =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 2]))
                        : minus_infinity;
                high_data[i].y() =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 3]))
                        : minus_infinity;
            } else if (alibi) {
                low_data[i].x() = low_x_check
                                      ? conversion::to<float>(vals[data_id]) * layer_scale +
                                            (conversion::to<float>(alibi[data_id + alibi_offset]))
                                      : minus_infinity;
                low_data[i].y() =
                    low_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale +
                              (conversion::to<float>(alibi[data_id + alibi_offset + reduceWidth]))
                        : minus_infinity;
                high_data[i].x() =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 2]))
                        : minus_infinity;
                high_data[i].y() =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 3]))
                        : minus_infinity;
            } else {
                low_data[i].x() = low_x_check ? conversion::to<float>(vals[data_id]) * layer_scale
                                              : minus_infinity;
                low_data[i].y() =
                    low_y_check ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale
                                : minus_infinity;
                high_data[i].x() =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale
                        : minus_infinity;
                high_data[i].y() =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale
                        : minus_infinity;
            }

            // if(lane == 0) printf("%f , %d, %d \n", low_data[i].x, data_id, seq_id);
            max_val = (low_data[i].x() > max_val ? low_data[i].x() : max_val);
            max_val = (low_data[i].y() > max_val ? low_data[i].y() : max_val);
            max_val = (high_data[i].x() > max_val ? high_data[i].x() : max_val);
            max_val = (high_data[i].y() > max_val ? high_data[i].y() : max_val);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) {
            auto temp = sycl::permute_group_by_xor(
                sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = max_val;
            item_ct1.barrier();

            if (lane < warp_num) max_val = partialSum[lane];

            item_ct1.barrier();

            for (int i = 1; i < reduce_blocks; i *= 2) {
                auto temp = sycl::permute_group_by_xor(
                    sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            max_val = g.shuffle(max_val, item_ct1.get_local_id(2) / WARP_SIZE);
        }
        float sum = 0;
        for (int i = 0; i < iterations; i++) {
            low_data[i].x() = sycl::native::exp(low_data[i].x() - max_val);
            low_data[i].y() = sycl::native::exp(low_data[i].y() - max_val);
            high_data[i].x() = sycl::native::exp(high_data[i].x() - max_val);
            high_data[i].y() = sycl::native::exp(high_data[i].y() - max_val);

            sum += (low_data[i].x() + low_data[i].y() + high_data[i].x() + high_data[i].y());
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) sum +=
            sycl::permute_group_by_xor(sycl::ext::oneapi::experimental::this_sub_group(), sum, i);

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = sum;
            item_ct1.barrier();

            if (lane < warp_num) sum = partialSum[lane];

            item_ct1.barrier();

            for (int i = 1; i < reduce_blocks; i *= 2) {
                sum += sycl::permute_group_by_xor(
                    sycl::ext::oneapi::experimental::this_sub_group(), sum, i);
            }

            sum = g.shuffle(sum, item_ct1.get_local_id(2) / WARP_SIZE);
        }
        sum += 1e-6;
        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            if (data_id < sequence_length) {
                vals[data_id] = conversion::to<T>(low_data[i].x() / sum);
                if ((data_id + reduceWidth) < sequence_length)
                    vals[data_id + reduceWidth] = conversion::to<T>(low_data[i].y() / sum);
                if ((data_id + reduceWidth * 2) < sequence_length)
                    vals[data_id + reduceWidth * 2] = conversion::to<T>(high_data[i].x() / sum);
                if ((data_id + reduceWidth * 3) < sequence_length)
                    vals[data_id + reduceWidth * 3] = conversion::to<T>(high_data[i].y() / sum);
            }
        }
    }
}

template <int iterations>
void attn_softmax_v2(float* vals,
                     float* attn_mask,
                     float* alibi,
                     float layer_scale,
                     bool triangular,
                     bool recompute,
                     bool local_attention,
                     int window_size,
                     int total_count,
                     int heads,
                     int sequence_length,
                     int num_seq,
                     int head_offset,
                     int mask_stride,
                     int mp_size,
                     int reduceWidth)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    sycl::group<3> b = sycl::ext::oneapi::experimental::this_group<3>();
    sycl::sub_group g = sycl::ext::oneapi::experimental::this_sub_group();

    sycl::float4 data[MAX_REG_SIZE];

    int wid = item_ct1.get_local_id(2) >> 5;
    int lane = item_ct1.get_local_id(2) & 0x1f;
    int warp_num = item_ct1.get_local_range(2) >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = item_ct1.get_local_id(2) % reduceWidth;

    auto& partialSum = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MAX_WARP_NUM]>(
        sycl::ext::oneapi::experimental::this_group<3>());

    int iter_offset = item_ct1.get_group(2) * (warp_num / reduce_blocks) + (wid / reduce_blocks);
    if (iter_offset < total_count) {
        vals += (iter_offset * sequence_length);

        int batch_idx = iter_offset / (num_seq * heads);
        int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);
        mask_offset = mask_offset * sequence_length;
        int seq_id = iter_offset % num_seq;

        int real_seq_id = seq_id + (num_seq == sequence_length ? 0 : sequence_length);
        int window_stride4 = (local_attention && (real_seq_id >> 2) > (window_size >> 2))
                                 ? (real_seq_id >> 2) - (window_size >> 2)
                                 : 0;
        int window_stride =
            (local_attention && real_seq_id >= window_size) ? real_seq_id - window_size : -1;

        float max_val = minus_infinity;

        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            bool check = (data_id >> 2) >= window_stride4;
            bool x_check = check && (data_id < sequence_length) &&
                           (!triangular || (data_id <= seq_id)) && (data_id > window_stride);
            bool y_check = check && ((data_id + reduceWidth) < sequence_length) &&
                           (!triangular || ((data_id + reduceWidth) <= seq_id)) &&
                           ((data_id + reduceWidth) > window_stride);
            bool z_check = check && ((data_id + reduceWidth * 2) < sequence_length) &&
                           (!triangular || ((data_id + reduceWidth * 2) <= seq_id)) &&
                           ((data_id + reduceWidth * 2) > window_stride);
            bool w_check = check && ((data_id + reduceWidth * 3) < sequence_length) &&
                           (!triangular || ((data_id + reduceWidth * 3) <= seq_id)) &&
                           ((data_id + reduceWidth * 3) > window_stride);

            if (attn_mask) {
                data[i].x() = x_check ? vals[data_id] + attn_mask[data_id + mask_offset]
                                      : minus_infinity;
                data[i].y() = y_check ? vals[data_id + reduceWidth] +
                                            attn_mask[data_id + mask_offset + reduceWidth]
                                      : minus_infinity;
                data[i].z() = z_check ? vals[data_id + reduceWidth * 2] +
                                            attn_mask[data_id + mask_offset + reduceWidth * 2]
                                      : minus_infinity;
                data[i].w() = w_check ? vals[data_id + reduceWidth * 3] +
                                            attn_mask[data_id + mask_offset + reduceWidth * 3]
                                      : minus_infinity;
            } else {
                data[i].x() = x_check ? vals[data_id] : minus_infinity;
                data[i].y() = y_check ? vals[data_id + reduceWidth] : minus_infinity;
                data[i].z() = z_check ? vals[data_id + reduceWidth * 2] : minus_infinity;
                data[i].w() = w_check ? vals[data_id + reduceWidth * 3] : minus_infinity;
            }

            max_val = (data[i].x() > max_val ? data[i].x() : max_val);
            max_val = (data[i].y() > max_val ? data[i].y() : max_val);
            max_val = (data[i].z() > max_val ? data[i].z() : max_val);
            max_val = (data[i].w() > max_val ? data[i].w() : max_val);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) {
            auto temp = sycl::permute_group_by_xor(
                sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = max_val;
            item_ct1.barrier();

            if (lane < warp_num) max_val = partialSum[lane];

            item_ct1.barrier();

            for (int i = 1; i < reduce_blocks; i *= 2) {
                auto temp = sycl::permute_group_by_xor(
                    sycl::ext::oneapi::experimental::this_sub_group(), max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            max_val = g.shuffle(max_val, item_ct1.get_local_id(2) / WARP_SIZE);
        }

        float sum = 0;
        for (int i = 0; i < iterations; i++) {
            data[i].x() = sycl::native::exp(data[i].x() - max_val);
            data[i].y() = sycl::native::exp(data[i].y() - max_val);
            data[i].z() = sycl::native::exp(data[i].z() - max_val);
            data[i].w() = sycl::native::exp(data[i].w() - max_val);

            sum += (data[i].x() + data[i].y() + data[i].z() + data[i].w());
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) sum +=
            sycl::permute_group_by_xor(sycl::ext::oneapi::experimental::this_sub_group(), sum, i);

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = sum;
            item_ct1.barrier();

            if (lane < warp_num) sum = partialSum[lane];

            item_ct1.barrier();

            for (int i = 1; i < reduce_blocks; i *= 2) {
                sum += sycl::permute_group_by_xor(
                    sycl::ext::oneapi::experimental::this_sub_group(), sum, i);
            }

            sum = g.shuffle(sum, item_ct1.get_local_id(2) / WARP_SIZE);
        }
        sum += 1e-6;

        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            if (data_id < sequence_length) {
                vals[data_id] = data[i].x() / sum;
                if ((data_id + reduceWidth) < sequence_length)
                    vals[data_id + reduceWidth] = data[i].y() / sum;
                if ((data_id + reduceWidth * 2) < sequence_length)
                    vals[data_id + reduceWidth * 2] = data[i].z() / sum;
                if ((data_id + reduceWidth * 3) < sequence_length)
                    vals[data_id + reduceWidth * 3] = data[i].w() / sum;
            }
        }
    }
}

#define LAUNCH_ATTN_SOFTMAX_V2(iterations)                                                        \
  {                                                                                               \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16}); \
    stream->submit([&](sycl::handler& cgh) {                                                      \
      T* vals_ct0 = vals;                                                                         \
      T* mask_ct1 = mask;                                                                         \
      T* alibi_ct2 = alibi;                                                                       \
      auto layer_scale_ct3 = layer_scale;                                                         \
      auto triangular_ct4 = triangular;                                                           \
      auto recompute_ct5 = recompute;                                                             \
      auto local_attention_ct6 = local_attention;                                                 \
      auto window_size_ct7 = window_size;                                                         \
      auto total_count_ct8 = total_count;                                                         \
      auto heads_ct9 = heads;                                                                     \
      auto sequence_length_ct10 = sequence_length;                                                \
      auto num_seq_ct11 = num_seq;                                                                \
      auto head_offset_ct12 = head_offset;                                                        \
      auto mask_stride_ct13 = mask_stride;                                                        \
      auto mp_size_ct14 = mp_size;                                                                \
      auto reduce_width_ct15 = reduce_width;                                                      \
                                                                                                  \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                    \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {        \
                         attn_softmax_v2<T, iterations>(vals_ct0,                                             \
                                            mask_ct1,                                             \
                                            alibi_ct2,                                            \
                                            layer_scale_ct3,                                      \
                                            triangular_ct4,                                       \
                                            recompute_ct5,                                        \
                                            local_attention_ct6,                                  \
                                            window_size_ct7,                                      \
                                            total_count_ct8,                                      \
                                            heads_ct9,                                            \
                                            sequence_length_ct10,                                 \
                                            num_seq_ct11,                                         \
                                            head_offset_ct12,                                     \
                                            mask_stride_ct13,                                     \
                                            mp_size_ct14,                                         \
                                            reduce_width_ct15);                                   \
                       });                                                                        \
    });                                                                                           \
  }

template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            T* alibi,
                            float layer_scale,
                            bool triangular,
                            bool recompute,
                            bool local_attention,
                            int window_size,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            int head_offset,
                            int mask_stride,
                            int mp_size,
                            sycl::queue * stream)
{
    const int total_count = batch_size * heads * num_seq;

    // Scheduling Overview
    // 4 element unroll with power of 2 `reduce_width` threads to a ceiling of `attn_threads`
    // Each block should be partitioned into as many `reduce_width` blocks
    // as can be fit.
    constexpr int attn_threads = 256;
    constexpr int min_reduce_width = hw_warp_size;
    constexpr int internal_unroll = 4;

    // Handle internal unroll then round to next power of 2. Bump up to minimum granularity.
    const int thread_steps_rounded =
        next_pow2((sequence_length + internal_unroll - 1) / internal_unroll);
    const int thread_steps_schedule =
        (thread_steps_rounded < min_reduce_width) ? min_reduce_width : thread_steps_rounded;
    // Bound reduce width to the number of threads
    const int reduce_width = (thread_steps_schedule < attn_threads) ? thread_steps_schedule
                                                                    : attn_threads;
    // Scale for the excess
    const int iterations = thread_steps_schedule / reduce_width;
    // Should be safe since reduce_width is capped to attn_threads
    const int partitions = attn_threads / reduce_width;

    // Launch params
    sycl::range<3> grid(1, 1, (total_count + partitions - 1) / partitions);
    sycl::range<3> block(1, 1, attn_threads);

    if (sequence_length <= 32768) {
        if (iterations == 1) {
            LAUNCH_ATTN_SOFTMAX_V2(1);
        } else if (iterations == 2) {
            LAUNCH_ATTN_SOFTMAX_V2(2);
        } else if (iterations == 4) {
            LAUNCH_ATTN_SOFTMAX_V2(4);
        } else if (iterations == 8) {
            LAUNCH_ATTN_SOFTMAX_V2(8);
        } else if (iterations == 16) {
            LAUNCH_ATTN_SOFTMAX_V2(16);
        } else if (iterations == 32) {
            LAUNCH_ATTN_SOFTMAX_V2(32);
        } else if (iterations == 64) {
            LAUNCH_ATTN_SOFTMAX_V2(64);
        }
    } else
        throw std::runtime_error("Unsupport Seq_Length!");
}

#define INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(T)                  \
    template void launch_attn_softmax_v2(T* vals,              \
                                         T* mask,              \
                                         T* alibi,             \
                                         float layer_scale,    \
                                         bool triangular,      \
                                         bool recompute,       \
                                         bool local_attention, \
                                         int window_size,      \
                                         int batch_size,       \
                                         int heads,            \
                                         int num_seq,          \
                                         int sequence_length,  \
                                         int head_offset,      \
                                         int mask_stride,      \
                                         int mp_size,          \
                                         sycl::queue * stream);

INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(sycl::ext::oneapi::bfloat16);
#endif
INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(sycl::half);

#define DEF_ATTN_SOFTMAX_V2_HALF(_iter)                                    \
    template void attn_softmax_v2<sycl::half, _iter>(sycl::half * vals,    \
                                                     sycl::half * mask,    \
                                                     sycl::half * alibi,   \
                                                     float layer_scale,    \
                                                     bool triangular,      \
                                                     bool recompute,       \
                                                     bool local_attention, \
                                                     int window_size,      \
                                                     int total_count,      \
                                                     int heads,            \
                                                     int sequence_length,  \
                                                     int num_seq,          \
                                                     int head_offset,      \
                                                     int mask_stride,      \
                                                     int mp_size,          \
                                                     int reduceWidth)

#define DEF_ATTN_SOFTMAX_V2_BF16(_iter)                                \
    template void attn_softmax_v2<sycl::ext::oneapi::bfloat16, _iter>( \
        sycl::ext::oneapi::bfloat16 * vals,                            \
        sycl::ext::oneapi::bfloat16 * mask,                            \
        sycl::ext::oneapi::bfloat16 * alibi,                           \
        float layer_scale,                                             \
        bool triangular,                                               \
        bool recompute,                                                \
        bool local_attention,                                          \
        int window_size,                                               \
        int total_count,                                               \
        int heads,                                                     \
        int sequence_length,                                           \
        int num_seq,                                                   \
        int head_offset,                                               \
        int mask_stride,                                               \
        int mp_size,                                                   \
        int reduceWidth)

#define FOREACH_ITERATIONS(cb) \
    cb(1);                     \
    cb(2);                     \
    cb(4);                     \
    cb(8);                     \
    cb(16);                    \
    cb(32);                    \
    cb(64)

FOREACH_ITERATIONS(DEF_ATTN_SOFTMAX_V2_HALF);
#ifdef BF16_AVAILABLE
FOREACH_ITERATIONS(DEF_ATTN_SOFTMAX_V2_BF16);
#endif
