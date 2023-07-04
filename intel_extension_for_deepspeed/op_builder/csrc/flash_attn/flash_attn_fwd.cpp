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

#include "flash_attn.hpp"
#include "flash_attn_fwd.hpp"

namespace xpu {
namespace xetla {

bool flash_scaled_attn_bf16_inf(
    sycl::queue& queue,
    // void* output, // pointer to output buffer, [Bs, Hn, Sl, Hs] ==> [Bs, Sl,
    // Hn, Hs] ==> [Sl, Bs, Hn, Hs]
    void* output, // pointer to output buffer, [Bs, Hn, Sl, Hs]
    void* out_buffer, // [Bs, Hn, Sl, Hs]
    const uint32_t Bs, // Batch sizes,
    const uint32_t Hn, // number of heads
    const uint32_t Sl, // sequence length, current assume Sq == Sk, fixed Sl
    // for current Batches
    const uint32_t Hs, // head sizes
    const float hs_rsqrt_scale, // hs_rsqrt_scale = 1 / sqrt(hs)
    const void* q_ptr, // pointer to Q data buffer, [Bs, Hn, Sl, Hs]
    const void* k_ptr, // pointer to K data buffer, [Bs, Hn, Sl, Hs]
    const void* v_ptr, // pointer to V data buffer, [Bs, Hn, Sl, Hs]
    const bool is_casual) {
  return false;
}
template <typename P>
static int flash_scaled_attn_bf16_fwd_run(
    sycl::queue& queue,
    typename P::arguments_t& args) {
  P kernel(args);
  sycl::nd_range<3> nd_range = kernel.get_nd_range();

  // auto evt = queue.submit([&](sycl::handler& cgh) {
  //   cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
  //     using namespace gpu::xetla;
  //     using namespace gpu::xetla::group;
  //     using namespace gpu::xetla::kernel;
  //     using namespace gpu::xetla::subgroup;

  //     xetla_exec_item<3> ei(item);
  //     kernel.run(ei);
  //   });
  // });
  // evt.wait();
  auto cgf = [&](sycl::handler& cgh) {
    cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      using namespace gpu::xetla;
      using namespace gpu::xetla::group;
      using namespace gpu::xetla::kernel;
      using namespace gpu::xetla::subgroup;
      xetla_exec_item<3> ei(item);
      kernel.run(ei);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);

  return 0;
}
bool flash_scaled_attn_bf16_fwd(
    sycl::queue& queue,
    void* output, // pointer to output buffer, [Bs, Hn, Sl, Hs], will consider
    // permute later
    void* out_buffer, // [Bs, Hn, Sl, Hs]
    void* softmax_workspace, // if store_sofmax_out is true,
    //   it's pointer to softmax output buffer, sizes
    //   are [Bs, Hn, Sl, Sl]
    // if store_sofmax_out is false,
    //   it's pointer to softmax row_max and row_sum
    //   buffer, sizes are [Bs*Hn, 2, Sl], row_max is
    //   stored at [Bs*Hn, 0, Sl], row_sum is stored at
    //   [Bs*Hn, 1, Sl]
    const uint32_t Bs, // Batch sizes,
    const uint32_t Hn, // number of heads
    const uint32_t Sl, // sequence length, current assume Sq == Sk, fixed Sl
    // for current Batches
    const uint32_t Hs, // head sizes
    const float hs_rsqrt_scale, // hs_rsqrt_scale = 1 / sqrt(hs)
    const void* q_ptr, // pointer to Q data buffer, [Bs, Hn, Sl, Hs]
    const void* k_ptr, // pointer to K data buffer, [Bs, Hn, Sl, Hs]
    const void* v_ptr, // pointer to V data buffer, [Bs, Hn, Sl, Hs]
    const void* drop_mask, // for dtopout mask if has, use uint8_t as data type
    const float dropout_scale, // dropout_scale = 1 / (1 - drop_p)
    const bool is_casual, // Indicate whether do mask_fill before softmax
    const bool store_softmax_out) {
  bool ret = false;
  if (Hs == 128) {
    using flash_attn_fwd_h128 =
        xpu::xetla::FLASH_ATTENTION_FWD_PARAM::tuning_parameter_t<
            gpu::xetla::bf16,
            gpu::xetla::bf16,
            gpu::xetla::bf16,
            gpu::xetla::bf16,
            float,
            float,
            float,
            float,
            float,
            true,
            true,
            xpu::xetla::FLASH_ATTENTION_FWD_PARAM::pv_buffer_type::global,
            32,
            128,
            128,
            128,
            64,
            32,
            128,
            16>;
    using P = xpu::xetla::FLASH_ATTENTION_FWD_IMPL<flash_attn_fwd_h128>;
    using arguments_t = P::arguments_t;
    P::dtype_q* ptr_q = reinterpret_cast<P::dtype_q*>(const_cast<void*>(q_ptr));
    P::dtype_k* ptr_k = reinterpret_cast<P::dtype_k*>(const_cast<void*>(k_ptr));
    P::dtype_v* ptr_v = reinterpret_cast<P::dtype_v*>(const_cast<void*>(v_ptr));
    P::dtype_o* ptr_o =
        reinterpret_cast<P::dtype_o*>(const_cast<void*>(output));
    P::dtype_m* ptr_m =
        reinterpret_cast<P::dtype_m*>(const_cast<void*>(softmax_workspace));
    P::dtype_b* ptr_b =
        reinterpret_cast<P::dtype_b*>(const_cast<void*>(out_buffer));
    arguments_t args(
        Bs, Hn, Sl, hs_rsqrt_scale, ptr_q, ptr_k, ptr_v, ptr_o, ptr_m, ptr_b);
    flash_scaled_attn_bf16_fwd_run<P>(queue, args);

    ret = true;
  } else if (Hs == 96) {
    using flash_attn_fwd_h96 =
        xpu::xetla::FLASH_ATTENTION_FWD_PARAM::tuning_parameter_t<
            gpu::xetla::bf16,
            gpu::xetla::bf16,
            gpu::xetla::bf16,
            gpu::xetla::bf16,
            float,
            float,
            float,
            float,
            float,
            true,
            true,
            xpu::xetla::FLASH_ATTENTION_FWD_PARAM::pv_buffer_type::global,
            32,
            96,
            128,
            128,
            64,
            32,
            128,
            16>;
    using P = xpu::xetla::FLASH_ATTENTION_FWD_IMPL<flash_attn_fwd_h96>;
    using arguments_t = P::arguments_t;
    P::dtype_q* ptr_q = reinterpret_cast<P::dtype_q*>(const_cast<void*>(q_ptr));
    P::dtype_k* ptr_k = reinterpret_cast<P::dtype_k*>(const_cast<void*>(k_ptr));
    P::dtype_v* ptr_v = reinterpret_cast<P::dtype_v*>(const_cast<void*>(v_ptr));
    P::dtype_o* ptr_o =
        reinterpret_cast<P::dtype_o*>(const_cast<void*>(output));
    P::dtype_m* ptr_m =
        reinterpret_cast<P::dtype_m*>(const_cast<void*>(softmax_workspace));
    P::dtype_b* ptr_b =
        reinterpret_cast<P::dtype_b*>(const_cast<void*>(out_buffer));
    arguments_t args(
        Bs, Hn, Sl, hs_rsqrt_scale, ptr_q, ptr_k, ptr_v, ptr_o, ptr_m, ptr_b);
    flash_scaled_attn_bf16_fwd_run<P>(queue, args);

    ret = true;
  }
  return ret;
}

} // namespace xetla
} // namespace xpu