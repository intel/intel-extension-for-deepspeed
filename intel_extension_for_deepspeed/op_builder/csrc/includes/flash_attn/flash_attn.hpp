#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <stddef.h>
#include "flash_attn_fwd_kernel.hpp"
#include "flash_attn_fwd_program.hpp"

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
    const void* drop_mask =
        nullptr, // for dtopout mask if has, use uint8_t as data type
    const float dropout_scale = 1.0, // dropout_scale = 1 / (1 - drop_p)
    const bool is_casual = true, // Indicate whether do mask_fill before softmax
    const bool store_softmax_out =
        false); // Indicate whether output softmax result

bool flash_scaled_attn_bf16_bwd(
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
    const uint32_t Hs, // saved Hs from forward
    const float hs_rsqrt_scale, // saved hs_rsqrt_scale from forward
    const void* q_ptr, // saved Q input from forward
    const void* k_ptr, // saved K input from forward
    const void* v_ptr, // saved V input from forward
    const void* drop_mask_ptr, // may be saved drop_mask from forward or
                               // regenrated drop mask use uint8_t as data type
    const float dropout_scale, // saved dropout_scale from forward
    const void* softmax_workspace_ptr, // saved softmax output or
                                       // row_max/row_sum from forward
    const bool is_casual = true, // Indicate whether do mask_fill before softmax
    const bool softmax_out_saved =
        false); // Indicate whether softmax result has been saved and not need
                // to be re-computed


class FlashAttention {
public:
    virtual ~FlashAttention() {}
    
    bool Forward(sycl::queue &stream,
                 void *output,
                 void* out_buffer,
                 void *softmax_storespace,
                 const uint32_t &Bs,
                 const uint32_t &Hn,
                 const uint32_t &Sl,
                 const uint32_t &Hs,
                 const float &hs_rsqrt_scale,
                 const void* q_ptr,
                 const void* k_ptr,
                 const void* v_ptr,
                 const void *attn_mask_ptr = nullptr,
                 const void *drop_mask = nullptr,
                 const float dropout_scale = 1.0,
                 const bool store_softmax_out = false,
                 const bool is_causal = false) {
        return flash_scaled_attn_bf16_fwd(
            stream,
            output,
            out_buffer,
            softmax_storespace,
            Bs,
            Hn,
            Sl,
            Hs,
            hs_rsqrt_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            attn_mask_ptr,
            drop_mask,
            dropout_scale,
            store_softmax_out,
            is_causal
        );
    }
};
