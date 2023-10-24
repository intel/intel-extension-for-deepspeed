
#include "flash_attn.hpp"
#include "flash_attn_bwd.hpp"

namespace xpu {
namespace xetla {

bool flash_scaled_attn_bf16_bwd(
    sycl::queue& queue,
    void* dq,
    void* dk,
    void* dv,
    void* out, // [Bs, Hn, Sl, Hs]
    void* gradout,
    void* softmax_workspace, // [Bs*Hn, 1, Sl]: row_max + log(row_sum)
    void* d_buffer, // temp buffer for D = O pointmul dO [Bs*Hn, 1, Sl]
    const uint32_t Bs,
    const uint32_t Hn,
    const uint32_t Sl,
    const uint32_t Hs,
    const float hs_rsqrt_scale,
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const void* dropout_mask_ptr,
    const float dropout_prob,
    const float dropout_scale,
    const uint64_t dropout_seed,
    const bool is_casual,
    const bool store_softmax) {
  if (Hs == 128 && is_casual && !store_softmax) {
    return flash_attn_bwd<bwd_kernel_traits<bf16, bf16, float, 128, 128, 128>>(
        queue,
        dq,
        dk,
        dv,
        out,
        gradout,
        softmax_workspace,
        d_buffer,
        Bs,
        Hn,
        Sl,
        Hs,
        hs_rsqrt_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        dropout_mask_ptr,
        dropout_prob,
        dropout_scale,
        dropout_seed,
        is_casual,
        store_softmax);
  } else if (Hs == 96 && is_casual && !store_softmax) {
    return flash_attn_bwd<bwd_kernel_traits<bf16, bf16, float, 96, 128, 128>>(
        queue,
        dq,
        dk,
        dv,
        out,
        gradout,
        softmax_workspace,
        d_buffer,
        Bs,
        Hn,
        Sl,
        Hs,
        hs_rsqrt_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        dropout_mask_ptr,
        dropout_prob,
        dropout_scale,
        dropout_seed,
        is_casual,
        store_softmax);
  } else {
    return false;
  }
}

} // namespace xetla
} // namespace xpu