#pragma once

#include <mha.h>


class FlashAttention {
public:
    virtual ~FlashAttention() {}
    
    bool Forward(sycl::queue &stream,
                 void* output,
                 void* softmax_L,
                 uint32_t num_batches,
                 uint32_t num_heads,
                 uint32_t head_size,
                 uint32_t num_queries,
                 uint32_t num_keys,
                 float hs_rsqrt_scale,
                 void* q_ptr,
                 void* k_ptr,
                 void* v_ptr,
                 void* dropout_mask = nullptr,
                 float dropout_prob = 0.0,
                 uint64_t rand_seed = 0,
                 uint64_t rank_offset = 0,
                 bool is_causal = true,
                 bool is_training = true,
                 bool is_dropout = true) {
        RECORD_FUNCTION("flash_scaled_attn_bf16_fwd", c10::ArrayRef<c10::IValue>({}));
        gpu::xetla::fmha_forward_kernel(
            gpu::xetla::XetlaType::bf16,
            stream,
            q_ptr,
            k_ptr,
            v_ptr,
            dropout_mask,
            output,
            softmax_L,
            hs_rsqrt_scale,
            dropout_prob,
            num_batches,
            num_heads,
            head_size,
            num_queries,
            num_keys,
            num_keys,
            is_causal,
            is_training,
            is_dropout,
            rand_seed,
            rank_offset
        );

        return true;
    }

    bool Backward(sycl::queue &stream,
                  void* dq,
                  void* dk,
                  void* dv,
                  void* out, // [Bs, Hn, Sl, Hs]
                  void* gradout,
                  void* softmax_workspace, // [Bs*Hn, 1, Sl]: row_max + log(row_sum)
                  void* d_buffer, // temp buffer for D = O pointmul dO [Bs*Hn, 1, Sl]
                  void* dq_acc,
                  uint32_t num_batches,
                  uint32_t num_heads,
                  uint32_t head_size,
                  uint32_t num_queries,
                  uint32_t num_keys,
                  float hs_rsqrt_scale,
                  void* q_ptr,
                  void* k_ptr,
                  void* v_ptr,
                  void* dropout_mask = nullptr,
                  float dropout_prob = 0.0,
                  uint64_t rand_seed = 0,
                  uint64_t rank_offset = 0,
                  bool is_causal = true,
                  bool is_dropout = true) {
        RECORD_FUNCTION("flash_scaled_attn_bf16_bwd", c10::ArrayRef<c10::IValue>({}));
        gpu::xetla::fmha_backward_kernel(
            gpu::xetla::XetlaType::bf16,
            stream,
            gradout,
            q_ptr,
            k_ptr,
            v_ptr,
            out,
            softmax_workspace,
            d_buffer,
            dq_acc,
            hs_rsqrt_scale,
            dropout_prob,
            dq,
            dk,
            dv,
            num_batches,
            num_heads,
            head_size,
            num_queries,
            num_keys,
            num_keys,
            is_causal,
            is_dropout,
            rand_seed,
            rank_offset
        );
        return true;
    }
};
