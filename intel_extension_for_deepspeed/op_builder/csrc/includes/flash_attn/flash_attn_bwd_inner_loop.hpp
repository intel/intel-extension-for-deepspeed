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

#include "flash_attn_bwd_utils.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <
    typename T,
    typename out_T,
    typename acc_T,
    typename mem_desc_brxbc_t_,
    typename mem_desc_brxd_t_,
    typename mem_desc_bcxd_t_,
    typename mem_desc_l_m_t_,
    typename gemm_brxbc_block_tile_t,
    typename gemm_brxd_block_tile_t,
    typename gemm_bcxd_block_tile_t,
    bool is_casual = true,
    uint32_t accum_stride = 16,
    uint32_t prefetch_distance = 3,
    uint32_t periodic_sync_interval = 0>
struct fmha_block_t {
  static constexpr int THREAD_NUM = 32;

  using mem_desc_brxd_t = mem_desc_brxd_t_;
  using mem_desc_bcxd_t = mem_desc_bcxd_t_;
  using mem_desc_brxbc_t = mem_desc_brxbc_t_;

  static constexpr mem_layout mem_k_trans_layout =
      mem_desc_bcxd_t::layout == mem_layout::row_major ? mem_layout::col_major
                                                       : mem_layout::row_major;
  using mem_desc_bcxd_trans_t = mem_desc_t<
      typename mem_desc_bcxd_t::dtype,
      mem_k_trans_layout,
      mem_desc_bcxd_t::space>;

  // scatter transposed while store to slm
  using mem_desc_brxbc_trans_t = mem_desc_brxbc_t_;
  // using mem_desc_dO_t = mem_desc_dO_t_;

  // TODO: pass by template param
  using mem_desc_l_m_t = mem_desc_l_m_t_;

  // TODO: mem_desc_c_t store out just for debug
  using mem_desc_c_t =
      mem_desc_t<out_T, mem_layout::row_major, mem_space::global>;

  using mem_desc_o_t =
      mem_desc_t<out_T, mem_layout::row_major, mem_space::global>;

  // TODO: for temp
  using mem_desc_trans_p_t =
      mem_desc_t<bf16, mem_layout::row_major, mem_space::local>;

  using tile_shape_bcxd = typename gemm_bcxd_block_tile_t::tile_shape_t;
  using tile_shape_brxbc = typename gemm_brxbc_block_tile_t::tile_shape_t;
  using tile_shape_brxd = typename gemm_brxd_block_tile_t::tile_shape_t;

  using compute_attr = compute_attr_t<T, T, acc_T>;

  using perf_tuning_knob = perf_tuning_knob_t<
      accum_stride /*sg_tile_k*/,
      prefetch_distance,
      periodic_sync_interval>;

  using compute_policy =
      compute_policy_default_xmx<compute_attr, perf_tuning_knob, gpu_arch::Xe>;

  using brgemm_brxd_t = brgemm_t<
      compute_policy,
      tile_shape_brxd,
      mem_desc_brxbc_t,
      mem_desc_bcxd_t>;
  using brgemm_bcxd_t = brgemm_t<
      compute_policy,
      tile_shape_bcxd,
      mem_desc_brxbc_trans_t,
      mem_desc_brxd_t>;
  using brgemm_brxbc_t = brgemm_t<
      compute_policy,
      tile_shape_brxbc,
      mem_desc_brxd_t,
      mem_desc_bcxd_trans_t>;

  using brgemm_brxd_args_t = typename brgemm_brxd_t::arguments_t;
  using brgemm_bcxd_args_t = typename brgemm_bcxd_t::arguments_t;
  using brgemm_brxbc_args_t = typename brgemm_brxbc_t::arguments_t;

  using matAcc_brxd_t = typename brgemm_brxd_t::matAcc_t;
  using matAcc_bcxd_t = typename brgemm_bcxd_t::matAcc_t;
  using matAcc_brxbc_t = typename brgemm_brxbc_t::matAcc_t;

  using tile_desc_brxd_t = typename brgemm_brxd_t::matAcc_tile_desc_t;
  using tile_desc_bcxd_t = typename brgemm_bcxd_t::matAcc_tile_desc_t;
  using tile_desc_brxbc_t = typename brgemm_brxbc_t::matAcc_tile_desc_t;
  using tile_desc_l_m_t = tile_desc_t<
      tile_desc_brxbc_t::tile_size_y,
      1,
      tile_desc_brxbc_t::block_size_y,
      1,
      reg_layout::tiled>;

  using tile_Tr_t = tile_t<T, tile_desc_brxd_t>;
  using tile_Tc_t = tile_t<T, tile_desc_bcxd_t>;
  using tile_rc_t = tile_t<T, tile_desc_brxbc_t>;
  using tile_l_m_t = tile_t<acc_T, tile_desc_l_m_t>;

  using tile_payload_Tr_t = mem_payload_t<
      T,
      tile_desc_brxd_t,
      msg_type_v<tile_desc_brxd_t, mem_desc_brxd_t::space>,
      mem_desc_brxd_t::layout,
      mem_desc_brxd_t::space,
      gpu_arch::Xe>;
  using tile_payload_Tc_t = mem_payload_t<
      T,
      tile_desc_bcxd_t,
      msg_type_v<tile_desc_bcxd_t, mem_desc_bcxd_t::space>,
      mem_desc_bcxd_t::layout,
      mem_desc_bcxd_t::space,
      gpu_arch::Xe>;
  using tile_payload_rc_t = mem_payload_t<
      T,
      tile_desc_brxbc_t,
      msg_type_v<tile_desc_brxbc_t, mem_desc_brxbc_t::space>,
      mem_desc_brxbc_t::layout,
      mem_desc_brxbc_t::space,
      gpu_arch::Xe>;
  using tile_payload_l_m_t = mem_payload_t<
      acc_T,
      tile_desc_l_m_t,
      msg_type_v<tile_desc_l_m_t, mem_desc_l_m_t::space>,
      mem_desc_l_m_t::layout,
      mem_desc_l_m_t::space,
      gpu_arch::Xe>;

  using epilogue_p_t = epilogue_transp_t<
      epilogue_policy_tile_op<
          chained_tile_op_t<>,
          result_overwrite,
          gpu_arch::Xe>,
      tile_shape_brxbc,
      mem_desc_brxbc_t>;

  using epilogue_global_t = epilogue_t<
      epilogue_policy_tile_op<
          chained_tile_op_t<>,
          result_overwrite,
          gpu_arch::Xe>,
      tile_shape_brxd,
      mem_desc_c_t>;

  using epilogue_local_t = epilogue_t<
      epilogue_policy_tile_op<
          chained_tile_op_t<>,
          result_overwrite,
          gpu_arch::Xe>,
      tile_shape_brxbc,
      mem_desc_brxbc_t>;

  // TODO: all brgemm's sg_tile split are not same
  using worker_scope_t = typename brgemm_brxbc_t::work_group_t;

  using wg_reduce_sum_t = customer_group_reduce_t<
      acc_T,
      1,
      tile_shape_brxd::sg_tile_size_y,
      reduce_op::sum,
      tile_shape_brxd::wg_size_x,
      true,
      gpu_arch::Xe>;

  using MASK = casual_mask<
      tile_shape_brxbc,
      gemm_brxbc_block_tile_t::blocked_M,
      gemm_brxbc_block_tile_t::blocked_N,
      matAcc_brxbc_t,
      worker_scope_t>;

  struct arguments_t {
    T* ptr_q;
    T* ptr_k;
    T* ptr_v;
    T* ptr_o;
    T* ptr_p; // debug
    T* ptr_s;
    acc_T* ptr_l;
    acc_T* ptr_m;
    T* ptr_dO;
    T* ptr_dQ;
    T* ptr_dK;
    T* ptr_dV;
    T* ptr_dP; // TODO: delete

    uint32_t matP_base = 0;

    arguments_t(){};
    arguments_t(
        T* ptr_q,
        T* ptr_k,
        T* ptr_v,
        T* ptr_o,
        acc_T* ptr_l,
        acc_T* ptr_m,
        T* ptr_dO,
        T* ptr_dQ,
        T* ptr_dK,
        T* ptr_dV,
        uint32_t seq_q,
        uint32_t seq_k,
        float scale,
        T* ptr_dP = nullptr,
        T* ptr_s = nullptr, // debug
        T* ptr_p = nullptr, // debug
        uint32_t matP_base = 0)
        : ptr_q(ptr_q),
          ptr_k(ptr_k),
          ptr_v(ptr_v),
          ptr_o(ptr_o),
          ptr_l(ptr_l),
          ptr_m(ptr_m),
          ptr_dO(ptr_dO),
          ptr_dQ(ptr_dQ),
          ptr_dK(ptr_dK),
          ptr_dV(ptr_dV),
          seq_q(seq_q),
          seq_k(seq_k),
          scale(scale),
          ptr_dP(ptr_dP) // TODO: delete , just for debug
          ,
          ptr_s(ptr_s),
          ptr_p(ptr_p),
          matP_base(matP_base){};
    uint32_t seq_q;
    uint32_t seq_k;
    float scale;
    // uint32_t seq_v;

    uint32_t head_size = 128;
  };

  __XETLA_API KERNEL_FUNC void operator()(
      xetla_exec_item<3> ei,
      arguments_t& args,
      int loop_idx /*, uint32_t slm_base = 0*/) {
    brgemm_brxd_t brgemm_brxd;
    brgemm_bcxd_t brgemm_bcxd;
    brgemm_brxbc_t brgemm_brxbc;

    brgemm_brxd_args_t brgemm_brxd_args;
    brgemm_bcxd_args_t brgemm_bcxd_args;
    brgemm_brxbc_args_t brgemm_brxbc_args;

    epilogue_p_t epilogue_p;
    epilogue_global_t epilogue;
    epilogue_local_t epilogue_local;

    mem_desc_brxd_t mem_desc_q;
    mem_desc_brxd_t mem_desc_dO;
    mem_desc_brxd_t mem_desc_o;
    mem_desc_brxd_t mem_desc_dQ;
    mem_desc_bcxd_trans_t mem_desc_trans_k;
    mem_desc_bcxd_t mem_desc_k;
    mem_desc_bcxd_trans_t mem_desc_trans_v;
    mem_desc_brxbc_t mem_desc_p;

    mem_desc_l_m_t mem_desc_l_m;

    mem_desc_brxbc_trans_t mem_desc_trans_p;

    mem_desc_c_t mem_desc_c;

    tile_Tr_t mat_dO;
    tile_Tr_t mat_O;
    tile_Tr_t mat_dQ;

    tile_l_m_t tile_l, tile_m;

    tile_payload_Tr_t tile_payload_Tr, tile_payload_Tr_1;
    tile_payload_l_m_t tile_payload_m;

    // tile_payload_Tr_t mat_payload_Tr;
    matAcc_brxbc_t matAcc_p;
    matAcc_bcxd_t matAcc_dV;
    matAcc_bcxd_t matAcc_dK;
    matAcc_brxbc_t matAcc_dP;
    matAcc_brxd_t matAcc_D;
    matAcc_brxd_t matAcc_dO;
    matAcc_brxd_t matAcc_O;
    matAcc_brxd_t matAcc_dQ;

    // matP_t matP;
    worker_scope_t g(ei.get_local_linear_id());

    static constexpr uint32_t wg_tile_n_Tr = tile_shape_brxd::wg_tile_size_x;
    static constexpr uint32_t wg_tile_m_Tr = tile_shape_brxd::wg_tile_size_y;
    static constexpr uint32_t sg_tile_n_Tr = tile_shape_brxd::sg_tile_size_x;
    static constexpr uint32_t sg_tile_m_Tr = tile_shape_brxd::sg_tile_size_y;

    static constexpr uint32_t wg_tile_n_Tc = tile_shape_bcxd::wg_tile_size_x;
    static constexpr uint32_t wg_tile_m_Tc = tile_shape_bcxd::wg_tile_size_y;
    static constexpr uint32_t sg_tile_n_Tc = tile_shape_bcxd::sg_tile_size_x;
    static constexpr uint32_t sg_tile_m_Tc = tile_shape_bcxd::sg_tile_size_y;

    static constexpr uint32_t wg_tile_n_rc = tile_shape_brxbc::wg_tile_size_x;
    static constexpr uint32_t wg_tile_m_rc = tile_shape_brxbc::wg_tile_size_y;
    static constexpr uint32_t sg_tile_n_rc = tile_shape_brxbc::sg_tile_size_x;
    static constexpr uint32_t sg_tile_m_rc = tile_shape_brxbc::sg_tile_size_y;

    static constexpr uint32_t wg_tile_k_Tr = gemm_brxd_block_tile_t::blocked_K;
    static constexpr uint32_t wg_tile_k_Tc = gemm_bcxd_block_tile_t::blocked_K;
    static constexpr uint32_t wg_tile_k_rc = gemm_brxbc_block_tile_t::blocked_K;

    // TODO: remove magical number
    const int steps = (args.seq_q + gemm_brxbc_block_tile_t::blocked_M - 1) /
        gemm_brxbc_block_tile_t::blocked_M;
    int start_n_Tr;
    int start_m_Tr;
    int start_k_Tr;

    // add loop_idx
    int start_n_Tc;
    int start_m_Tc;
    int start_k_Tc;

    int start_n_rc;
    int start_m_rc;
    int start_k_rc;

    uint32_t boundary_n_Tr;
    uint32_t boundary_m_Tr;
    uint32_t boundary_k_Tr;

    uint32_t boundary_n_Tc;
    uint32_t boundary_m_Tc;
    uint32_t boundary_k_Tc;

    uint32_t boundary_n_rc;
    uint32_t boundary_m_rc;
    uint32_t boundary_k_rc;

    // inner_loop
    // loop_idx = 0;
    matAcc_dV.init(0);
    matAcc_dK.init(0);
    // is_casual;
    // TODO: blow left do not need caculate when is_casual true
    int begin = is_casual ? loop_idx * gemm_brxbc_block_tile_t::blocked_N /
            gemm_brxbc_block_tile_t::blocked_M
                          : 0;
    // begin = 0;
    for (int i = begin; i < steps; i++) {
      start_n_Tr = ei.get_group(2) * wg_tile_n_Tr;
      start_m_Tr = ei.get_group(1) * wg_tile_m_Tr +
          i * gemm_brxd_block_tile_t::blocked_M;
      start_k_Tr = loop_idx * gemm_brxd_block_tile_t::blocked_K;

      // add loop_idx
      start_n_Tc = ei.get_group(2) * wg_tile_n_Tc;
      start_m_Tc = ei.get_group(1) * wg_tile_m_Tc +
          loop_idx * gemm_bcxd_block_tile_t::blocked_M;
      start_k_Tc = i * gemm_bcxd_block_tile_t::blocked_K;

      start_n_rc = ei.get_group(2) * wg_tile_n_rc +
          loop_idx * gemm_brxbc_block_tile_t::blocked_N;
      start_m_rc = ei.get_group(1) * wg_tile_m_rc +
          i * gemm_brxbc_block_tile_t::blocked_M;
      start_k_rc = 0;

      boundary_n_Tr = (start_n_Tr + wg_tile_n_Tr) > args.head_size
          ? args.head_size
          : start_n_Tr + wg_tile_n_Tr;
      boundary_m_Tr = (start_m_Tr + wg_tile_m_Tr) > args.seq_q
          ? args.seq_q
          : start_m_Tr + wg_tile_m_Tr;
      boundary_k_Tr = start_k_Tr + gemm_brxd_block_tile_t::blocked_K;

      boundary_n_Tc = (start_n_Tc + wg_tile_n_Tc) > args.head_size
          ? args.head_size
          : start_n_Tc + wg_tile_n_Tc;
      boundary_m_Tc = (start_m_Tc + wg_tile_m_Tc) > args.seq_k
          ? args.seq_k
          : start_m_Tc + wg_tile_m_Tc;
      boundary_k_Tc = start_k_Tc + gemm_bcxd_block_tile_t::blocked_K;

      boundary_n_rc = (start_n_rc + wg_tile_n_rc) > args.seq_k
          ? args.seq_k
          : start_n_rc + wg_tile_n_rc;
      boundary_m_rc = (start_m_rc + wg_tile_m_rc) > args.seq_q
          ? args.seq_q
          : start_m_rc + wg_tile_m_rc;
      boundary_k_rc = start_k_rc + gemm_brxbc_block_tile_t::blocked_K;

      mem_desc_trans_k.init(
          args.ptr_k,
          {boundary_n_rc, boundary_k_rc, args.head_size},
          {start_n_rc, start_k_rc});
      mem_desc_trans_v.init(
          args.ptr_v,
          {boundary_n_rc, boundary_k_rc, args.head_size},
          {start_n_rc, start_k_rc});
      mem_desc_k.init(
          args.ptr_k,
          {boundary_n_Tr, boundary_k_Tr, args.head_size},
          {start_n_Tr, start_k_Tr});

      mem_desc_o.init(
          args.ptr_o,
          {boundary_n_Tr, boundary_m_Tr, args.head_size},
          {start_n_Tr + brgemm_brxd_t::get_matC_offset_x(g),
           start_m_Tr + brgemm_brxd_t::get_matC_offset_y(g)});

      mem_desc_dO.init(
          args.ptr_dO,
          {boundary_n_Tr, boundary_m_Tr, args.head_size},
          {start_n_Tr + brgemm_brxd_t::get_matC_offset_x(g),
           start_m_Tr + brgemm_brxd_t::get_matC_offset_y(g)});

      tile_payload_Tr_1.init(mem_desc_dO);
      tile_load(mat_dO, tile_payload_Tr_1);

      tile_payload_Tr.init(mem_desc_o);
      tile_load(mat_O, tile_payload_Tr);

      // S𝑖𝑗 = 𝜏Q𝑖K𝑇𝑗
      {
        mem_desc_q.init(
            args.ptr_q,
            {boundary_k_rc, boundary_m_rc, gemm_brxbc_block_tile_t::blocked_K},
            {start_k_rc, start_m_rc});
        matAcc_p.init(0);

        brgemm_brxbc_args.init(
            mem_desc_q,
            mem_desc_trans_k,
            gemm_brxbc_block_tile_t::inner_loop_count);
        brgemm_brxbc(g, matAcc_p, brgemm_brxbc_args);
        matAcc_p.reg *= args.scale;
      }
      // debug
      /****************elemwise opertation******************/
      /****************MASK elemwise opertation******************/
      MASK::apply_mask(g, matAcc_p, i, loop_idx);
      /****************SOFTMAX elemwise opertation***************/
      { // softmax
        mem_desc_l_m.init(
            args.ptr_m,
            {boundary_m_rc, 1, boundary_m_rc},
            {start_m_rc + brgemm_brxd_t::get_matC_offset_y(g), 0});
        tile_payload_m.init(mem_desc_l_m);
        tile_load(tile_m, tile_payload_m);

        tile_broadcast_op<tile_minus, matAcc_brxbc_t>(matAcc_p, tile_m.reg);
        matAcc_p.reg = gpu::xetla::xetla_exp<acc_T>(matAcc_p.reg);
        mem_desc_l_m.init(
            args.ptr_l,
            {boundary_m_rc, 1, boundary_m_rc},
            {start_m_rc + brgemm_brxd_t::get_matC_offset_y(g), 0});
        tile_payload_m.init(mem_desc_l_m);
        tile_load(tile_l, tile_payload_m);
        tile_broadcast_op<tile_div, matAcc_brxbc_t>(matAcc_p, tile_l.reg);
      }
      /****************compute dropout mask**********************/
      /****************DROP_OUT elemwise opertation**************/

      /****************elemwise opertation******************/
      // mem_desc_c.init(args.ptr_p, {boundary_n_rc, boundary_m_rc,
      // /*gemm_p_block_tile_t::blocked_N*/uint32_t(args.seq_k)}, {start_n_rc,
      // start_m_rc}); epilogue(g, matAcc_p, mem_desc_c);
      // transpose P in slm
      {
        mem_desc_p.init(
            args.matP_base, {wg_tile_n_rc, wg_tile_m_rc, wg_tile_n_rc}, {0, 0});
        // mem_desc_p.init(args.matP_base, {wg_tile_n, wg_tile_m, wg_tile_n},
        // {0, 0});
        epilogue_p(g, matAcc_p, mem_desc_p);
        __esimd_barrier();
        SW_BARRIER();
      }

      { // acc_dV = acc_dV + P_dropped_ij_T x dOi
        mem_desc_dO.init(
            args.ptr_dO,
            {boundary_n_Tc, boundary_k_Tc, gemm_bcxd_block_tile_t::blocked_N},
            {start_n_Tc, start_k_Tc});
        mem_desc_trans_p.init(
            args.matP_base, {wg_tile_k_Tc, wg_tile_m_Tc, wg_tile_k_Tc}, {0, 0});
        brgemm_bcxd_args.init(
            mem_desc_trans_p,
            mem_desc_dO,
            gemm_bcxd_block_tile_t::inner_loop_count);
        brgemm_bcxd(g, matAcc_dV, brgemm_bcxd_args);
        SW_BARRIER();
      }
      { // dP_ij = dO_i x V_j_T
        mem_desc_dO.init(
            args.ptr_dO,
            {boundary_k_rc, boundary_m_rc, gemm_brxbc_block_tile_t::blocked_K},
            {start_k_rc, start_m_rc});

        matAcc_dP.init(0);
        brgemm_brxbc_args.init(
            mem_desc_dO,
            mem_desc_trans_v,
            gemm_brxbc_block_tile_t::inner_loop_count);
        brgemm_brxbc(g, matAcc_dP, brgemm_brxbc_args);
      }
      // mem_desc_c.init(args.ptr_s, {boundary_n_rc, boundary_m_rc,
      // /*gemm_p_block_tile_t::blocked_N*/uint32_t(args.seq_k)}, {start_n_rc,
      // start_m_rc}); epilogue(g, matAcc_dP, mem_desc_c);
      /**************Dropout bwd elemwise multiply*****************/
      { // D_i= rowsum(dO_i * O_i)
        // TODO:: just support wg_tile_n = d;
        elemwise_cvt<matAcc_brxd_t, tile_Tr_t>(matAcc_O, mat_O);
        elemwise_cvt<matAcc_brxd_t, tile_Tr_t>(matAcc_dO, mat_dO);
        matAcc_D.reg = matAcc_O.reg * matAcc_dO.reg;
        int32_t sg_idx = g.get_id() % tile_shape_brxd::wg_size_x;
        int32_t sg_idy = g.get_id() / tile_shape_brxd::wg_size_x;

        uint32_t nbarrier_id = /*nbarrier_base*/ 0 + sg_idy;
        uint32_t slm_base_addr = /*slm_base*/ 0 +
            sg_idy * tile_shape_brxbc::wg_size_x *
                tile_shape_brxbc::sg_tile_size_y * sizeof(acc_T);

        xetla_vector<acc_T, tile_shape_brxbc::sg_tile_size_y> local_sum =
            tile_reduce<reduce_op::sum, matAcc_brxd_t, acc_T, 1>(matAcc_D);
        wg_reduce_sum_t wg_reduce_sum(sg_idx, nbarrier_id, slm_base_addr);
        xetla_vector<acc_T, tile_shape_brxbc::sg_tile_size_y> group_sum =
            wg_reduce_sum(local_sum);
        // dS_ij = P_ij*(dP_ij-D_ij)
        matAcc_D.init(0);
        tile_broadcast_op<tile_minus, matAcc_brxd_t>(matAcc_D, group_sum);
        matAcc_dP.reg = (matAcc_dP.reg + matAcc_D.reg) * matAcc_p.reg;
        // nbarrier.arrive_wait();
      }
      // mem_desc_c.init({args.ptr_dP}, {boundary_n_rc, boundary_m_rc,
      // /*gemm_p_block_tile_t::blocked_N*/uint32_t(args.seq_k)}, {start_n_rc,
      // start_m_rc}); epilogue(g, matAcc_dP, mem_desc_c);

      { // store dP to local
        mem_desc_p.init(
            args.matP_base, {wg_tile_n_rc, wg_tile_m_rc, wg_tile_n_rc}, {0, 0});
        epilogue_local(g, matAcc_dP, mem_desc_p);
        __esimd_barrier();
      }

      mem_desc_p.init(
          args.matP_base, {wg_tile_k_Tr, wg_tile_m_Tr, wg_tile_k_Tr}, {0, 0});

      { // dQ_i = dQ_i + 𝜏dS_ij x K_j
        // TODO: mat_dQ.reg = /*𝜏*/matAcc_dQ.reg + mat_dQ.reg;
        mem_desc_dQ.init(
            args.ptr_dQ,
            {boundary_n_Tr, boundary_m_Tr, gemm_brxd_block_tile_t::blocked_N},
            {start_n_Tr + brgemm_brxd_t::get_matC_offset_x(g),
             start_m_Tr + brgemm_brxd_t::get_matC_offset_y(g)});
        tile_payload_Tr.init(mem_desc_dQ);
        tile_load(mat_dQ, tile_payload_Tr);

        matAcc_dQ.init(0);
        brgemm_brxd_args.init(
            mem_desc_p, mem_desc_k, gemm_brxd_block_tile_t::inner_loop_count);
        brgemm_brxd(g, matAcc_dQ, brgemm_brxd_args);
        matAcc_dQ.reg = args.scale * matAcc_dQ.reg + mat_dQ.reg;
        mem_desc_c.init(
            args.ptr_dQ,
            {boundary_n_Tr, boundary_m_Tr, gemm_brxd_block_tile_t::blocked_N},
            {start_n_Tr, start_m_Tr});
        epilogue(g, matAcc_dQ, mem_desc_c);
      }
      { // transpose dP in slm
        mem_desc_p.init(
            args.matP_base, {wg_tile_n_Tc, wg_tile_m_Tc, wg_tile_n_Tc}, {0, 0});
        epilogue_p(g, matAcc_dP, mem_desc_p);
        __esimd_barrier();
      }

      { // caculata dK
        brgemm_bcxd_args.init(
            mem_desc_trans_p,
            mem_desc_q,
            gemm_bcxd_block_tile_t::inner_loop_count);
        brgemm_bcxd(g, matAcc_dK, brgemm_bcxd_args);
        // epilogue(g, matAcc_dK, mem_desc_c);
      }
    }
    matAcc_dK.reg = args.scale * matAcc_dK.reg;
    mem_desc_c.init(
        args.ptr_dK,
        {boundary_n_Tc, boundary_m_Tc, gemm_bcxd_block_tile_t::blocked_N},
        {start_n_Tc, start_m_Tc});
    epilogue(g, matAcc_dK, mem_desc_c);
    mem_desc_c.init(
        {args.ptr_dV},
        {boundary_n_Tc, boundary_m_Tc, gemm_bcxd_block_tile_t::blocked_N},
        {start_n_Tc, start_m_Tc});
    epilogue(g, matAcc_dV, mem_desc_c);
  }
};