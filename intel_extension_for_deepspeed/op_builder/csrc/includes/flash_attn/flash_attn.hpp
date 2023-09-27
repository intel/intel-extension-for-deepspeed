#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <stddef.h>
#include <ipex.h>
#include <torch/extension.h>

#define DPCPP_Q_SUBMIT(q, cgf, ...)                                          \
  {                                                                          \
    auto e = (q).submit((cgf), ##__VA_ARGS__);                               \
    (q).throw_asynchronous();                                                \
    xpu::profiler_record("dpcpp_kernel", e);                                 \
  }

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
    const bool is_casual =
        true); // Indicate whether do mask_fill before softmax

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
    const bool store_softmax); // Indicate whether output softmax result

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
    const bool store_softmax);

} // namespace xetla
} // namespace xpu

class FlashAttention {
public:
    virtual ~FlashAttention() {}
    
    bool Forward(sycl::queue &stream,
                 void* output,
                 void* softmax_L,
                 void* softmax_m,
                 const uint32_t Bs,
                 const uint32_t Hn,
                 const uint32_t Sl,
                 const uint32_t Hs,
                 const float hs_rsqrt_scale,
                 const void* q_ptr,
                 const void* k_ptr,
                 const void* v_ptr,
                 const void* drop_mask = nullptr,
                 const float dropout_prob = 0.0,
                 const float dropout_scale = 1.0,
                 const uint64_t dropout_rand_seed = 0,
                 const bool is_causal = true,
                 const bool store_softmax_out = false) {
        RECORD_FUNCTION("flash_scaled_attn_bf16_fwd", c10::ArrayRef<c10::IValue>({}));
        return xpu::xetla::flash_scaled_attn_bf16_fwd(
            stream,
            output,
            softmax_L,
            softmax_m,
            Bs,
            Hn,
            Sl,
            Hs,
            hs_rsqrt_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            drop_mask,
            dropout_prob,
            dropout_scale,
            dropout_rand_seed,
            is_causal,
            store_softmax_out
        );
    }

    bool Backward(sycl::queue &stream,
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
        RECORD_FUNCTION("flash_scaled_attn_bf16_bwd", c10::ArrayRef<c10::IValue>({}));
        return xpu::xetla::flash_scaled_attn_bf16_bwd(
            stream,
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
            store_softmax
        );
    }
};
