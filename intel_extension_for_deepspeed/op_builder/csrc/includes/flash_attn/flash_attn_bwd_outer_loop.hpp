#pragma once
#include "flash_attn_bwd_inner_loop.hpp"

template<
    typename input_T,
    typename out_T,
    typename acc_T,
    typename gemm_brxd_block_tile_t,
    typename gemm_bcxd_block_tile_t,
    typename gemm_brxbc_block_tile_t,
    uint32_t acc_stride = 16,
    uint32_t prefetch_distance = 3,
    uint32_t periodic_sync_interval = 8>
struct flash_attention_bwd {
  // mem_desc_t<dtype, layout, space>
  using mem_desc_brxd_t =
      mem_desc_t<input_T, mem_layout::row_major, mem_space::global>;
  using mem_desc_bcxd_t =
      mem_desc_t<input_T, mem_layout::row_major, mem_space::global>;
  using mem_desc_brxbc_t =
      mem_desc_t<input_T, mem_layout::row_major, mem_space::local>;
  using mem_desc_l_m_t =
      mem_desc_t<acc_T, mem_layout::row_major, mem_space::global>;

  using fmha_bwd_inner_loop_t = fmha_bwd_block_t<
      input_T,
      out_T,
      float,
      mem_desc_brxbc_t,
      mem_desc_brxd_t,
      mem_desc_bcxd_t,
      mem_desc_l_m_t,
      gemm_brxbc_block_tile_t,
      gemm_brxd_block_tile_t,
      gemm_bcxd_block_tile_t>;
  using inner_loop_arguments = typename fmha_bwd_inner_loop_t::arguments_t;

  struct arguments_t : public inner_loop_arguments {
    arguments_t(
        input_T* ptr_q,
        input_T* ptr_k,
        input_T* ptr_v,
        out_T* ptr_o,
        acc_T* ptr_L,
        input_T* ptr_dq,
        input_T* ptr_dk,
        input_T* ptr_dv,
        out_T* ptr_do,
        acc_T* ptr_D,
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
            ptr_L,
            ptr_dq,
            ptr_dk,
            ptr_dv,
            ptr_do,
            ptr_D,
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
    fmha_bwd_inner_loop_t fmha_inner_loop;
    xetla_nbarrier_init<
        gemm_brxbc_block_tile_t::tile_shape_t::wg_size_x +
        gemm_brxd_block_tile_t::tile_shape_t::wg_size_y +
        gemm_brxbc_block_tile_t::tile_shape_t::wg_size_y>();
    int block_bc = gemm_brxbc_block_tile_t::blocked_N;
    int max_loop_steps = (args.seq_k + block_bc - 1) / block_bc;
    fmha_inner_loop(ei, args, max_loop_steps);
  }
};