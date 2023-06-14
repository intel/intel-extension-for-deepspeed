/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once
#include "flash_attn_bwd_outer_loop.hpp"

template <
    typename T,
    typename out_T,
    typename acc_T,
    // uint32_t batch_num_,
    // uint32_t head_num_,
    uint32_t head_size_,
    uint32_t blocksize_c_>
bool flash_attn_bwd(
    sycl::queue& queue,
    void* dq, // gradient of Q, [Bs, Hn, Sl, Hs]
    void* dk, // gradient of K, [Bs, Hn, Sl, Hs]
    void* dv, // gradient of V, [Bs, Hn, Sl, Hs]
    void* grad_softmax, // temp buffer for grad_softmax output, [Bs, Hn, Sl, Sl]
    const void* out, // output, [Bs, Hn, Sl, Hs]
    const void*
        gradout, // gradient of output, has been permuted as [Bs, Hn, Sl, Hs]
    const uint32_t Bs, // saved Bs from forward
    const uint32_t Hn, // saved Hn from forward
    const uint32_t Sl, // saved Sl from forward
    const float hs_rsqrt_scale, // saved hs_rsqrt_scale from forward
    const void* q_ptr, // saved Q input from forward
    const void* k_ptr, // saved K input from forward
    const void* v_ptr, // saved V input from forward
    const void* drop_mask_ptr, // may be saved drop_mask from forward or
                               // regenrated drop mask use uint8_t as data type
    const float dropout_scale, // saved dropout_scale from forward
    const void* softmax_workspace_ptr, // saved softmax output or
                                       // row_max/row_sum from forward
    const bool softmax_out_saved =
        false, // Indicate whether softmax result has been saved and not need to
               // be re-computed
    const bool is_causal = true, // Indicate whether do mask_fill before softmax
    const void* causal_mask_ptr = nullptr) {
  using namespace cl::sycl;
  using namespace gpu::xetla;
  using namespace gpu::xetla::group;
  using namespace gpu::xetla::kernel;
  using namespace gpu::xetla::subgroup;

  uint32_t batch_num = Bs * Hn;
  // static constexpr uint32_t head_num = head_num_;
  static constexpr uint32_t d = head_size_;

  static constexpr uint32_t bc = blocksize_c_;
  static constexpr uint32_t br = std::min(d, bc);

  using block_brxbc_tile_t = gemm_block_tile_t<br, bc, d, br, bc, 16, 32, 16>;
  using block_brxd_tile_t = gemm_block_tile_t<br, d, bc, br, d, 16, 32, 16>;
  using block_bcxd_tile_t = gemm_block_tile_t<bc, d, br, bc, d, 16, 32, 16>;

  using fmha_t = flash_attention_bwd<
      T,
      out_T,
      acc_T,
      block_brxd_tile_t,
      block_bcxd_tile_t,
      block_brxbc_tile_t>;

  using fmha_args_t = typename fmha_t::arguments_t;

  int qkv_batch_offset = Sl * d;
  int lm_batch_offset = Sl;

  // constexpr int matrix_m = Sl;
  // constexpr int matrix_n = d;
  // constexpr int matrix_k = Sl;

  constexpr int wg_tile_m = 128;
  constexpr int wg_tile_n = 128;
  constexpr int sg_tile_m = 16; // 32 thread
  constexpr int sg_tile_n = 32; // 32 thread
  constexpr int sg_tile_k = 16;

  size_t group_range_m = 1; // magical number
  size_t group_range_n = 1; // magical number

  constexpr size_t subgroup_range_m = wg_tile_m / sg_tile_m;
  constexpr size_t subgroup_range_n = wg_tile_n / sg_tile_n;

  std::cout << "group_num_x: " << group_range_n
            << ", group_num_y: " << group_range_m
            << ", group_num_z: " << batch_num << "\n";
  std::cout << "group_size_x: " << subgroup_range_n
            << ", group_size_y: " << subgroup_range_m << std::endl;
  // cl::sycl::range<3> GroupRange{batch_num, group_range_m, group_range_n};
  cl::sycl::range<3> GroupRange{batch_num, 1, 1};
  cl::sycl::range<3> LocalRange{1, subgroup_range_m, subgroup_range_n};
  cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

  try {
    queue.submit([&](handler& cgh) {
      cgh.parallel_for<class Test>(
          Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
            xetla_exec_item<3> ei(item);
            static constexpr uint32_t slm_size = bc * br * sizeof(T);
            xetla_local_init<slm_size>();

            uint32_t batch_id = ei.get_group(0);
            fmha_t fmha_bwd;

            fmha_args_t args(
                (T*)q_ptr + batch_id * qkv_batch_offset,
                (T*)k_ptr + batch_id * qkv_batch_offset,
                (T*)v_ptr + batch_id * qkv_batch_offset,
                (T*)out + batch_id * qkv_batch_offset, // matO_ptr
                (acc_T*)softmax_workspace_ptr +
                    (batch_id * 2 + 1) * lm_batch_offset, // vecl_ptr,
                (acc_T*)softmax_workspace_ptr +
                    (batch_id * 2) * lm_batch_offset, // vecm_ptr,
                (T*)gradout + batch_id * qkv_batch_offset,
                (T*)dq + batch_id * qkv_batch_offset,
                (T*)dk + batch_id * qkv_batch_offset,
                (T*)dv + batch_id * qkv_batch_offset,
                Sl,
                Sl,
                hs_rsqrt_scale
                // matdP_ptr,
                // matS_ptr,
                // matP_ptr
            );
            fmha_bwd(ei, args);
          });
    });
    // gpu_event.wait();
    // prof.add_gpu_event(gpu_event);
  } catch (cl::sycl::exception const& e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    // FAIL();
    return false;
  }
  return true;
}

// bool flash_scaled_attn_bf16_bwd(
//     sycl::queue& queue,
//     void* dq, // gradient of Q, [Bs, Hn, Sl, Hs]
//     void* dk, // gradient of K, [Bs, Hn, Sl, Hs]
//     void* dv, // gradient of V, [Bs, Hn, Sl, Hs]
//     void* grad_softmax, // temp buffer for grad_softmax output, [Bs, Hn, Sl,
//     Sl] const void*
//         gradout, // gradient of output, has been permuted as [Bs, Hn, Sl, Hs]
//     const uint32_t Bs, // saved Bs from forward
//     const uint32_t Hn, // saved Hn from forward
//     const uint32_t Sl, // saved Sl from forward
//     const uint32_t Hs, // saved Hs from forward
//     const float hs_rsqrt_scale, // saved hs_rsqrt_scale from forward
//     const void* q_ptr, // saved Q input from forward
//     const void* k_ptr, // saved K input from forward
//     const void* v_ptr, // saved V input from forward
//     const void* drop_mask_ptr, // may be saved drop_mask from forward or
//                                // regenrated drop mask use uint8_t as data
//                                type
//     const float dropout_scale, // saved dropout_scale from forward
//     const void* softmax_workspace_ptr, // saved softmax output or
//                                        // row_max/row_sum from forward
//     const bool softmax_out_saved =
//         false, // Indicate whether softmax result has been saved and not need
//         to
//                // be re-computed
//     const bool is_causal = true, // Indicate whether do mask_fill before
//     softmax const void* causal_mask_ptr =
//         nullptr){  // for causal mask if has, may not need (for backward,
//         default
//                   // make right-uptringle as zero)

//   return flash_attn_bwd<bf16, bf16, float, 96, 8, 128, 128>(
//       queue,
//       dq, // gradient of Q, [Bs, Hn, Sl, Hs]
//       dk, // gradient of K, [Bs, Hn, Sl, Hs]
//       dv, // gradient of V, [Bs, Hn, Sl, Hs]
//       grad_softmax, // temp buffer for grad_softmax output, [Bs, Hn, Sl, Sl]
//       gradout, // gradient of output, has been permuted as [Bs, Hn, Sl, Hs]
//       Sl, // saved Sl from forward
//       hs_rsqrt_scale, // saved hs_rsqrt_scale from forward
//       q_ptr, // saved Q input from forward
//       k_ptr, // saved K input from forward
//       v_ptr, // saved V input from forward
//       drop_mask_ptr, // may be saved drop_mask from forward or
//                                 // regenrated drop mask use uint8_t as data
//                                 type
//       dropout_scale, // saved dropout_scale from forward
//       softmax_workspace_ptr // saved softmax output or
//                                         // row_max/row_sum from forward
//       // const bool softmax_out_saved =
//       //     false, // Indicate whether softmax result has been saved and not
//       need to
//       //            // be re-computed
//       // const bool is_causal = true, // Indicate whether do mask_fill before
//       softmax
//       // const void* causal_mask_ptr = nullptr
//   );
// }