
#include "flash_attn.hpp"
#include "flash_attn_fwd.hpp"

namespace xpu {
namespace xetla {

bool flash_scaled_attn_bf16_fwd(
    sycl::queue& queue,
    void* output, // [Bs, Hn, Sl, Hs]
    // void* out_buffer, // [Bs, Hn, Sl, Hs]
    void* softmax_L, // [Bs*Hn, 1, Sl]: row_max + log(row_sum)
    void* softmax_m,
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
    const bool store_softmax){
  if (Hs == 128 && is_casual && !store_softmax) {
    return flash_attn_fwd<fwd_kernel_traits<bf16, bf16, float, 128, 128, 128>>(
        queue, // sycl queue
        output, // output
        softmax_L, // softmax_L
        softmax_m, // softmax_m
        Bs, // Bs
        Hn, // Hn
        Sl, // Sl
        Hs, // head size
        hs_rsqrt_scale, // hs_rsqrt_scale
        q_ptr, // Q input
        k_ptr, // K input
        v_ptr, // V input
        dropout_mask_ptr, // drop_mask
        dropout_prob, // dropout_prob
        dropout_scale, // dropout_scale
        dropout_seed, // rand_seed
        is_casual, // casual
        store_softmax // store_softmax
    );
  } else if (Hs == 96 && is_casual && !store_softmax) {
    return flash_attn_fwd<fwd_kernel_traits<bf16, bf16, float, 96, 128, 128>>(
        queue, // sycl queue
        output, // output
        softmax_L, // softmax_L
        softmax_m, // softmax_m
        Bs, // Bs
        Hn, // Hn
        Sl, // Sl
        Hs, // head size
        hs_rsqrt_scale, // hs_rsqrt_scale
        q_ptr, // Q input
        k_ptr, // K input
        v_ptr, // V input
        dropout_mask_ptr, // drop_mask
        dropout_prob, // dropout_prob
        dropout_scale, // dropout_scale
        dropout_seed, // rand_seed
        is_casual, // casual
        store_softmax // store_softmax
    );
  } else {
    return false;
  }
}

} // namespace xetla
} // namespace xpu