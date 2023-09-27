#pragma once
#include "flash_attn_utils.hpp"
#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <
    typename input_T,
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
    uint32_t periodic_sync_interval = 8>
struct fmha_fwd_block_t {
  static constexpr int THREAD_NUM = 32;

  using mem_desc_brxd_t = mem_desc_brxd_t_;
  using mem_desc_bcxd_t = mem_desc_bcxd_t_;
  using mem_desc_brxbc_t = mem_desc_brxbc_t_;

  /**************** memory descriptor ******************/
  static constexpr mem_layout mem_trans_layout =
      mem_desc_bcxd_t::layout == mem_layout::row_major ? mem_layout::col_major
                                                       : mem_layout::row_major;
  
  using mem_desc_bcxd_trans_t = mem_desc_t<
      typename mem_desc_bcxd_t::dtype,
      mem_trans_layout,
      mem_desc_bcxd_t::space>;
  using mem_desc_l_m_t = mem_desc_l_m_t_;

  using mem_desc_c_t =
      mem_desc_t<out_T, mem_layout::row_major, mem_space::global>;

  /**************** tile shape ******************/
  using tile_shape_bcxd = typename gemm_bcxd_block_tile_t::tile_shape_t;
  using tile_shape_brxbc = typename gemm_brxbc_block_tile_t::tile_shape_t;
  using tile_shape_brxd = typename gemm_brxd_block_tile_t::tile_shape_t;

  /**************** compute args ******************/
  using compute_attr = compute_attr_t<input_T, input_T, acc_T>;
  using perf_tuning_knob_S = perf_tuning_knob_t<
      accum_stride /*sg_tile_k*/,
      prefetch_distance,
      periodic_sync_interval>;
  using compute_policy_S =
      compute_policy_default_xmx<compute_attr, perf_tuning_knob_S, gpu_arch::Xe>;
  
  using perf_tuning_knob_O = perf_tuning_knob_t<
      accum_stride /*sg_tile_k*/,
      prefetch_distance,
      periodic_sync_interval>;
  using compute_policy_O =
      compute_policy_default_xmx<compute_attr, perf_tuning_knob_O, gpu_arch::Xe>;

  /**************** brgemm descriptor ******************/
  using brgemm_brxd_t = brgemm_t<
      compute_policy_O,
      tile_shape_brxd,
      mem_desc_brxbc_t,
      mem_desc_bcxd_t>;
  using brgemm_bcxd_t = brgemm_t<
      compute_policy_O,
      tile_shape_bcxd,
      mem_desc_brxbc_t,
      mem_desc_brxd_t>;
  using brgemm_brxbc_t = brgemm_t<
      compute_policy_S,
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
      tile_desc_brxbc_t::tile_size_y, // br
      1,
      tile_desc_brxbc_t::block_size_y, // br
      1,
      reg_layout::tiled>;

  using tile_Tr_t = tile_t<input_T, tile_desc_brxd_t>;
  using tile_Tc_t = tile_t<input_T, tile_desc_bcxd_t>;
  using tile_rc_t = tile_t<input_T, tile_desc_brxbc_t>;
  using tile_l_m_t = tile_t<acc_T, tile_desc_l_m_t>;

  using tile_payload_Tr_t = mem_payload_t<
      input_T,
      tile_desc_brxd_t,
      msg_type_v<tile_desc_brxd_t, mem_desc_brxd_t::space>,
      mem_desc_brxd_t::layout,
      mem_desc_brxd_t::space,
      gpu_arch::Xe>;
  using tile_payload_Tc_t = mem_payload_t<
      input_T,
      tile_desc_bcxd_t,
      msg_type_v<tile_desc_bcxd_t, mem_desc_bcxd_t::space>,
      mem_desc_bcxd_t::layout,
      mem_desc_bcxd_t::space,
      gpu_arch::Xe>;
  using tile_payload_rc_t = mem_payload_t<
      input_T,
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
  
  using epilogue_global_t = epilogue_t<
      epilogue_policy_tile_op<
          chained_tile_op_t<>,
          gpu_arch::Xe>,
      tile_shape_brxd,
      mem_desc_brxd_t>;
  
  using epilogue_local_t = epilogue_t<
      epilogue_policy_tile_op<
          chained_tile_op_t<>,
          gpu_arch::Xe>,
      tile_shape_brxbc,
      mem_desc_brxbc_t>;
  
  // for debug, save S/P
  using epilogue_debug_t = epilogue_t<
      epilogue_policy_tile_op<
          chained_tile_op_t<>,
          gpu_arch::Xe>,
      tile_shape_brxd,
      mem_desc_brxbc_t>;
  
  using worker_scope_brxbc_t = typename brgemm_brxbc_t::work_group_t;
  using worker_scope_brxd_t = typename brgemm_brxd_t::work_group_t;
  using worker_scope_bcxd_t = typename brgemm_bcxd_t::work_group_t;

  using rowmax_t = xetla_vector<acc_T, tile_shape_brxbc::sg_tile_size_y>;
  using rowsum_t = rowmax_t;

  using wg_reduce_max_t = group_reduce_t<
      acc_T,
      1,
      tile_shape_brxbc::sg_tile_size_y,
      reduce_op::max,
      tile_shape_brxbc::wg_size_x,
      true,
      gpu_arch::Xe>;
  using wg_reduce_sum_t = group_reduce_t<
      acc_T,
      1,
      tile_shape_brxbc::sg_tile_size_y,
      reduce_op::sum,
      tile_shape_brxbc::wg_size_x,
      true,
      gpu_arch::Xe>;

  using MASK = casual_mask<
      tile_shape_brxbc,
      gemm_brxbc_block_tile_t::blocked_M,
      gemm_brxbc_block_tile_t::blocked_N,
      matAcc_brxbc_t,
      worker_scope_brxbc_t>;

  using dropout_t = dropout_fwd_t<tile_desc_brxbc_t::tile_elems>;
  dropout_t dropout_op;

  xetla_nbarrier_t<tile_shape_brxbc::wg_size_y, tile_shape_brxbc::wg_size_y>
      nbarrier_y;
  xetla_nbarrier_t<tile_shape_brxbc::wg_size_x, tile_shape_brxbc::wg_size_x>
      nbarrier_x;
  
  struct arguments_t {
    input_T* ptr_q;
    input_T* ptr_k;
    input_T* ptr_v;
    out_T* ptr_o;
    // input_T* ptr_o_buffer;
    acc_T* ptr_l;
    acc_T* ptr_m;
    const float dropout_prob;
    const float dropout_scale;
    const uint64_t rand_seed;
    bool dropout_enabled;

    input_T* ptr_dropmask = nullptr;
    uint32_t matP_base = 0;

    const uint32_t seq_q;
    const uint32_t seq_k;
    const float scale;
    uint32_t head_size;

    arguments_t(){};
    arguments_t(
        input_T* ptr_q_,
        input_T* ptr_k_,
        input_T* ptr_v_,
        input_T* ptr_o_,
        // input_T* ptr_o_buffer_,
        acc_T* ptr_l_,
        acc_T* ptr_m_,
        const uint32_t seq_q_,
        const uint32_t seq_k_,
        const float scale_,
        const float dropout_prob_ = 0,
        const float dropout_scale_ = 0,
        const uint64_t rand_seed_ = 67280421310721,
        const uint32_t matP_base_ = 0,
        input_T* drop_mask_ptr_ = nullptr)
        : ptr_q(ptr_q_),
          ptr_k(ptr_k_),
          ptr_v(ptr_v_),
          ptr_o(ptr_o_),
          // ptr_o_buffer(ptr_o_buffer_),
          ptr_l(ptr_l_),
          ptr_m(ptr_m_),
          seq_q(seq_q_),
          seq_k(seq_k_),
          scale(scale_),
          dropout_prob(dropout_prob_),
          dropout_scale(dropout_scale_),
          rand_seed(rand_seed_),
          matP_base(matP_base_),
          ptr_dropmask(drop_mask_ptr_) {
      dropout_enabled = dropout_prob > 0;
    };
  };
  
  __XETLA_API KERNEL_FUNC void operator()(
      xetla_exec_item<3>& ei,
      arguments_t& args,
      int outer_loop_idx) {
    brgemm_brxbc_t brgemm_brxbc;
    brgemm_brxd_t brgemm_brxd;
    brgemm_bcxd_t brgemm_bcxd;

    brgemm_brxd_args_t brgemm_brxd_args;
    brgemm_bcxd_args_t brgemm_bcxd_args;
    brgemm_brxbc_args_t brgemm_brxbc_args;

    epilogue_global_t epilogue_global;
    epilogue_local_t epilogue_local;

    mem_desc_brxbc_t mem_desc_p;
    mem_desc_brxd_t mem_desc_o;
    mem_desc_bcxd_t mem_desc_q, mem_desc_v;
    mem_desc_bcxd_trans_t mem_desc_trans_k;
    mem_desc_l_m_t mem_desc_l_m;
    mem_desc_c_t mem_desc_c; // for debug

    tile_Tr_t mat_O;
    tile_l_m_t tile_l, tile_m;
    tile_payload_Tr_t tile_payload_Tr;
    tile_payload_l_m_t tile_payload_l_m;

    matAcc_brxbc_t matAcc_p;
    matAcc_brxd_t matAcc_o;

    worker_scope_brxbc_t g_brxbc(ei.get_local_linear_id());
    worker_scope_brxd_t g_brxd(ei.get_local_linear_id());
    worker_scope_bcxd_t g_bcxd(ei.get_local_linear_id());

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

    int start_n_Tr, start_m_Tr, start_k_Tr;
    uint32_t end_n_Tr, end_m_Tr, end_k_Tr;

    int start_n_Tc, start_m_Tc, start_k_Tc;
    uint32_t end_n_Tc, end_m_Tc, end_k_Tc;

    int start_n_rc, start_m_rc, start_k_rc;
    uint32_t end_n_rc, end_m_rc, end_k_rc;

    int32_t sg_idx = g_brxbc.get_id() % tile_shape_brxbc::wg_size_x;
    int32_t sg_idy = g_brxbc.get_id() / tile_shape_brxbc::wg_size_x;
    nbarrier_x.init_nbarrier(sg_idy, nbarrier_role::producer_consumer);
    nbarrier_y.init_nbarrier(
        tile_shape_brxd::wg_size_y + sg_idx, nbarrier_role::producer_consumer);

    start_m_Tr = outer_loop_idx * gemm_brxd_block_tile_t::blocked_M + 
        ei.get_group(1) * wg_tile_m_Tr;
    start_n_Tr = 0;
    // start_k_Tr = inner_loop_idx * gemm_brxd_block_tile_t::blocked_K +
    //     ei.get_group(2) * wg_tile_k_Tr;
    end_m_Tr = (start_m_Tr + wg_tile_m_Tr) > args.seq_q
        ? args.seq_q
        : start_m_Tr + wg_tile_m_Tr;
    end_n_Tr = (start_n_Tr + wg_tile_n_Tr) > args.head_size
        ? args.head_size
        : start_n_Tr + wg_tile_n_Tr;
    // end_k_Tr = start_k_Tr + gemm_brxd_block_tile_t::blocked_K;

    // start_m_Tc = inner_loop_idx * gemm_bcxd_block_tile_t::blocked_M +
    //     ei.get_group(1) * wg_tile_m_Tc;
    start_n_Tc = 0;
    start_k_Tc = outer_loop_idx * gemm_bcxd_block_tile_t::blocked_K;
    // end_m_Tc = (start_m_Tc + wg_tile_m_Tc) > args.seq_k
    //     ? args.seq_k
    //     : start_m_Tc + wg_tile_m_Tc;
    end_n_Tc = (start_n_Tc + wg_tile_n_Tc) > args.head_size
        ? args.head_size
        : start_n_Tc + wg_tile_n_Tc;
    end_k_Tc = (start_k_Tc + gemm_bcxd_block_tile_t::blocked_K) > args.seq_q
        ? args.seq_q
        : start_k_Tc + gemm_bcxd_block_tile_t::blocked_K;

    start_m_rc = ei.get_group(1) * wg_tile_m_rc + 
        outer_loop_idx * gemm_brxbc_block_tile_t::blocked_M;
    // start_n_rc = ei.get_group(2) * wg_tile_n_rc +
    //     inner_loop_idx * gemm_brxbc_block_tile_t::blocked_N;
    start_k_rc = 0;
    end_m_rc = (start_m_rc + wg_tile_m_rc) > args.seq_q
        ? args.seq_q
        : start_m_rc + wg_tile_m_rc;
    // end_n_rc = (start_n_rc + wg_tile_n_rc) > args.seq_k
    //     ? args.seq_k
    //     : start_n_rc + wg_tile_n_rc
    end_k_rc = (start_k_rc + gemm_brxbc_block_tile_t::blocked_K) > args.head_size
        ? args.head_size
        : start_k_rc + gemm_brxbc_block_tile_t::blocked_K;

    // for rowmax & rosum
    rowmax_t& m_vec_i = tile_m.reg;
    rowsum_t& l_vec_i = tile_l.reg;
    rowmax_t m_vec_j, m_vec_diff;
    rowsum_t l_vec_j;

    m_vec_i = negative_Inf;
    l_vec_i = 0.0f;

    uint32_t nbarrier_id = /*nbarrier_base*/ tile_shape_brxbc::wg_size_x +
        tile_shape_brxbc::wg_size_y + sg_idy;
    uint32_t slm_base_addr = /*slm_base*/ 0 +
        sg_idy * tile_shape_brxbc::wg_size_x *
            tile_shape_brxbc::sg_tile_size_y * sizeof(acc_T);

    // for dropout
    int batch_idx = ei.get_group(0);
    using sg_shape = gemm_brxbc_block_tile_t::tile_shape_t;
    int32_t j_s = g_brxbc.get_id() % sg_shape::wg_size_x;
    int32_t i_s = g_brxbc.get_id() / sg_shape::wg_size_x;
    int32_t i_w = outer_loop_idx;
    // int32_t j_w = inner_loop_idx;
    int coord_y = i_w * gemm_brxbc_block_tile_t::blocked_M +
        i_s * sg_shape::sg_tile_size_y;
    // int coord_x = j_w * gemm_brxbc_block_tile_t::blocked_N +
    //    j_s * sg_shape::sg_tile_size_x;
    // const uint64_t subseq = uint64_t(coord_y) << 32 | uint64_t(coord_x);
    const uint64_t offset = ei.get_group(0);
    const uint32_t threshold =
        uint32_t(args.dropout_prob * float(4294967296));

    matAcc_o.init(0);

    const int steps = is_casual
        ? ((outer_loop_idx + 1) * gemm_brxbc_block_tile_t::blocked_M) /
            gemm_brxbc_block_tile_t::blocked_N
        : (args.seq_k + gemm_brxbc_block_tile_t::blocked_N - 1) /
            gemm_brxbc_block_tile_t::blocked_N;

    for (int inner_loop_idx = 0; inner_loop_idx < steps; ++inner_loop_idx) {
      start_k_Tr = inner_loop_idx * gemm_brxd_block_tile_t::blocked_K +
          ei.get_group(2) * wg_tile_k_Tr;
      end_k_Tr = start_k_Tr + gemm_brxd_block_tile_t::blocked_K > args.seq_k
          ? args.seq_k
          : start_k_Tr + gemm_brxd_block_tile_t::blocked_K;

      start_m_Tc = inner_loop_idx * gemm_bcxd_block_tile_t::blocked_M +
          ei.get_group(2) * wg_tile_m_Tc;
      end_m_Tc = (start_m_Tc + wg_tile_m_Tc) > args.seq_k
          ? args.seq_k
          : start_m_Tc + wg_tile_m_Tc;

      start_n_rc = ei.get_group(2) * wg_tile_n_rc +
          inner_loop_idx * gemm_brxbc_block_tile_t::blocked_N;
      end_n_rc = (start_n_rc + wg_tile_n_rc) > args.seq_k
          ? args.seq_k
          : start_n_rc + wg_tile_n_rc;

      int32_t j_w = inner_loop_idx;
      int coord_x = j_w * gemm_brxbc_block_tile_t::blocked_N +
         j_s * sg_shape::sg_tile_size_x;
      const uint64_t subseq = uint64_t(coord_y) << 32 | uint64_t(coord_x);

      {
        // Calc Sij -> Pij = Qi x K_Tj
        mem_desc_q.init(
          {args.ptr_q},
          {end_k_rc, end_m_rc, gemm_brxbc_block_tile_t::blocked_K},
          {start_k_rc, start_m_rc}
        );
        mem_desc_trans_k.init(
          {args.ptr_k},
          {end_n_rc, end_k_rc, gemm_brxbc_block_tile_t::blocked_N},
          {start_n_rc, start_k_rc}
        );
        matAcc_p.init(0);
        
        brgemm_brxbc_args.init(
          mem_desc_q,
          mem_desc_trans_k,
          gemm_brxbc_block_tile_t::inner_loop_count
        );
        brgemm_brxbc(g_brxbc, matAcc_p, brgemm_brxbc_args);
        matAcc_p.reg *= args.scale;
      }

      {
        // Apply Causal mask on S(P)
        MASK::apply_mask(g_brxbc, matAcc_p, outer_loop_idx, inner_loop_idx);

        // // for debug
        // mem_desc_c.init(
        //     {args.ptr_o},
        //     {end_n_rc, end_m_rc, args.seq_k},
        //     {start_n_rc, start_m_rc}
        // );
        // epilogue_global(g_brxbc, matAcc_p, mem_desc_c);
      }

      {
        // Calc mi = max(mi, rowmax(Pij))
        rowmax_t local_m = tile_reduce<reduce_op::max, acc_T, acc_T, 1, matAcc_brxbc_t>(matAcc_p);
        wg_reduce_max_t wg_reduce_max(sg_idx, nbarrier_id, slm_base_addr);
        m_vec_j = wg_reduce_max(local_m);
        m_vec_j = xetla_max<acc_T, tile_shape_brxbc::sg_tile_size_y>(m_vec_i, m_vec_j);

        // Save diff mij-1 - mij
        m_vec_diff = inner_loop_idx == 0
            ? 0.0f
            : m_vec_i - m_vec_j;

        // Update mi
        m_vec_i = m_vec_j;
      }

      {
        // Calc Pij = exp(Pij - mij)
        tile_broadcast_op<tile_minus, matAcc_brxbc_t>(matAcc_p, m_vec_i);
        matAcc_p.reg = xetla_exp<acc_T>(matAcc_p.reg);
      }

      {
        // Calc lij = exp(mij-1 - mij) * lij-1 + rowsum(Pij)
        rowsum_t local_l = tile_reduce<reduce_op::sum, acc_T, acc_T, 1, matAcc_brxbc_t>(matAcc_p);
        wg_reduce_sum_t wg_reduce_sum(sg_idx, nbarrier_id, slm_base_addr);
        l_vec_j = wg_reduce_sum(local_l);
        m_vec_diff = xetla_exp<acc_T>(m_vec_diff);
        l_vec_i = m_vec_diff * l_vec_i + l_vec_j;
      }

      {
        // Apply Dropout on P
        dropout_op.init(
            args.rand_seed, subseq, offset, threshold, args.dropout_scale);
        matAcc_p.reg = dropout_op.template process<float>(matAcc_p.reg);
      }

      {
        nbarrier_x.arrive_wait();
        nbarrier_y.arrive_wait();
        SW_BARRIER();
        // Store Pij to slm
        mem_desc_p.init(
            {args.matP_base},
            {wg_tile_n_rc, wg_tile_m_rc, wg_tile_n_rc},
            {0, 0}
        );
        epilogue_local(g_brxbc, matAcc_p, mem_desc_p);
        nbarrier_x.arrive_wait();
        nbarrier_y.arrive_wait();
        SW_BARRIER();
      }

      {
        // Calc Oij = diag(exp(diff_m)) x Oij-1 + Pij x Vj
        tile_broadcast_op<tile_mul, matAcc_brxd_t>(matAcc_o, m_vec_diff);
        mem_desc_p.init(
            {args.matP_base},
            {wg_tile_k_Tr, wg_tile_m_Tr, wg_tile_k_Tr},
            {0, 0}
        );
        mem_desc_v.init(
            {args.ptr_v},
            {end_n_Tr, end_k_Tr, gemm_brxd_block_tile_t::blocked_N},
            {start_n_Tr, start_k_Tr}
        );
        brgemm_brxd_args.init(
            mem_desc_p,
            mem_desc_v,
            gemm_brxd_block_tile_t::inner_loop_count
        );
        brgemm_brxd(g_brxd, matAcc_o, brgemm_brxd_args);
      }
    } // end inner loop

    // // for debug
    // mem_desc_l_m.init(
    //     args.ptr_m,
    //     {end_m_rc, 1, gemm_brxbc_block_tile_t::blocked_M},
    //     {start_m_rc + brgemm_brxbc_t::get_matC_offset_y(g_brxbc), 0}
    // );
    // tile_payload_l_m.init(mem_desc_l_m);
    // tile_store(tile_m, tile_payload_l_m);

    // mem_desc_l_m.init(
    //     args.ptr_l,
    //     {end_m_rc, 1, gemm_brxbc_block_tile_t::blocked_M},
    //     {start_m_rc + brgemm_brxbc_t::get_matC_offset_y(g_brxbc), 0}
    // );
    // tile_payload_l_m.init(mem_desc_l_m);
    // tile_store(tile_l, tile_payload_l_m);

    // Oi = diag(li)_(-1) x Oi and save to global memory
    mem_desc_o.init(
        args.ptr_o,
        {end_n_Tr, end_m_Tr, gemm_brxd_block_tile_t::blocked_N},
        {start_n_Tr, start_m_Tr}
    );
    tile_broadcast_op<tile_div, matAcc_brxbc_t>(matAcc_o, l_vec_i);
    epilogue_global(g_brxd, matAcc_o, mem_desc_o);
    
    // // Li = mi + log(li) and save to global memory
    l_vec_i = xetla_log<acc_T>(l_vec_i);
    l_vec_i = m_vec_i + l_vec_i;

    mem_desc_l_m.init(
        args.ptr_l,
        {end_m_rc, 1, gemm_brxbc_block_tile_t::blocked_M},
        {start_m_rc + brgemm_brxbc_t::get_matC_offset_y(g_brxbc), 0}
    );
    tile_payload_l_m.init(mem_desc_l_m);
    tile_store(tile_l, tile_payload_l_m);
  }
};