#pragma once
#include "flash_attn_fwd_inner_loop.hpp"

template<
    typename input_T,
    typename output_T,
    typename acc_T,
    typename gemm_brxd_block_tile_t,
    typename gemm_bcxd_block_tile_t,
    typename gemm_brxbc_block_tile_t,
    uint32_t acc_stride = 16,
    uint32_t prefetch_distance = 3,
    uint32_t periodic_sync_interval = 8>
struct flash_attention_fwd {
  // mem_desc_t<dtype, layout, space>
  using mem_desc_brxd_t =
      mem_desc_t<input_T, mem_layout::row_major, mem_space::global>;
  using mem_desc_bcxd_t =
      mem_desc_t<input_T, mem_layout::row_major, mem_space::global>;
  using mem_desc_brxbc_t =
      mem_desc_t<input_T, mem_layout::row_major, mem_space::local>;
  using mem_desc_l_m_t =
      mem_desc_t<acc_T, mem_layout::row_major, mem_space::global>;

  using fmha_fwd_inner_loop_t = fmha_fwd_block_t<
      input_T,
      output_T,
      float,
      mem_desc_brxbc_t,
      mem_desc_brxd_t,
      mem_desc_bcxd_t,
      mem_desc_l_m_t,
      gemm_brxbc_block_tile_t,
      gemm_brxd_block_tile_t,
      gemm_bcxd_block_tile_t>;
  using inner_loop_arguments = typename fmha_fwd_inner_loop_t::arguments_t;

  struct arguments_t : public inner_loop_arguments {
    arguments_t(
        input_T* ptr_q,
        input_T* ptr_k,
        input_T* ptr_v,
        output_T* ptr_o,
        // input_T* ptr_o_buffer,
        acc_T* ptr_l,
        acc_T* ptr_m,
        uint32_t seq_q,
        uint32_t seq_k,
        float scale,
        const float dropout_prob,
        const float dropout_scale,
        const uint64_t rand_seed,
        uint32_t p_base,
        input_T* ptr_mask) : inner_loop_arguments (
            ptr_q,
            ptr_k,
            ptr_v,
            ptr_o,
            // ptr_o_buffer,
            ptr_l,
            ptr_m,
            seq_q,
            seq_k,
            scale,
            dropout_prob,
            dropout_scale,
            rand_seed,
            p_base,
            ptr_mask
        ){};
  };
  
  __XETLA_API KERNEL_FUNC void operator()(
      xetla_exec_item<3>& ei,
      arguments_t& args) {
    fmha_fwd_inner_loop_t fmha_inner_loop;
    xetla_nbarrier_init<
        gemm_brxbc_block_tile_t::tile_shape_t::wg_size_x +
        gemm_brxd_block_tile_t::tile_shape_t::wg_size_y +
        gemm_brxbc_block_tile_t::tile_shape_t::wg_size_y>();
    int block_br = gemm_brxbc_block_tile_t::blocked_M;
    int max_loop_steps = (args.seq_q + block_br - 1) / block_br;
    for (int loop_idx = 0; loop_idx < max_loop_steps; loop_idx++) {
      fmha_inner_loop(ei, args, loop_idx);
    }
  }
};