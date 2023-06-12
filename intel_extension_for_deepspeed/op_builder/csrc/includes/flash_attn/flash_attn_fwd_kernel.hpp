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

#include <xetla.hpp>

namespace xpu {
namespace xetla {

class FLASH_ATTENTION_FWD_H128 {
 public:
  using dtype_q = gpu::xetla::bf16;
  using dtype_k = gpu::xetla::bf16;
  using dtype_v = gpu::xetla::bf16;
  using dtype_o = gpu::xetla::bf16;
  // using dtype_q = float;
  // using dtype_k = float;
  // using dtype_v = float;
  // using dtype_o = float;
  using dtype_m = float;
  using dtype_b = float;

  using dtype_acc = float;
  using dtype_s = dtype_acc;
  using dtype_p = dtype_acc;

  // hidden size per head
  static constexpr uint32_t H = 128;

  struct tuning_parameter {
    static constexpr int thread_num = 32;

    static constexpr int B_r = 128;
    static constexpr int B_c = 128;

    static constexpr int matS_n_s = B_c / 2;
    static constexpr int matS_k_s = 32;

    static constexpr int matO_k_s = 16;
  };

  struct arguments_t {
    const uint32_t batch_dim;
    const uint32_t head_dim;
    const uint32_t batch_size;
    const uint32_t sequence_length;
    const float hs_rsqrt_scale;
    dtype_q* ptr_q;
    dtype_k* ptr_k;
    dtype_v* ptr_v;
    dtype_o* ptr_o;
    dtype_m* ptr_m;
    dtype_b* ptr_b;

    arguments_t(
        uint32_t batch_dim_,
        const uint32_t head_dim_,
        uint32_t sequence_length_,
        float hs_rsqrt_scale_,
        dtype_q* ptr_q_,
        dtype_k* ptr_k_,
        dtype_v* ptr_v_,
        dtype_o* ptr_o_,
        dtype_m* ptr_m_,
        dtype_b* ptr_b_)
        : batch_dim(batch_dim_),
          head_dim(head_dim_),
          batch_size(batch_dim_ * head_dim_),
          sequence_length(sequence_length_),
          hs_rsqrt_scale(hs_rsqrt_scale_),
          ptr_q(ptr_q_),
          ptr_k(ptr_k_),
          ptr_v(ptr_v_),
          ptr_o(ptr_o_),
          ptr_m(ptr_m_),
          ptr_b(ptr_b_){};
  };

  static constexpr int thread_num = tuning_parameter::thread_num;
  static constexpr int B_r = tuning_parameter::B_r;
  static constexpr int B_c = tuning_parameter::B_c;
  const uint32_t batch_dim;
  const uint32_t head_dim;
  const uint32_t batch_size;
  const uint32_t seq_len;
  const float hs_rsqrt_scale;
  dtype_q* const ptr_q;
  dtype_k* const ptr_k;
  dtype_v* const ptr_v;
  dtype_o* const ptr_o;
  dtype_m* const ptr_m;
  dtype_b* const ptr_b;
  const uint32_t T_r;
  const uint32_t T_c;
  const uint32_t t_y;
  const uint32_t t_x;

  explicit FLASH_ATTENTION_FWD_H128(arguments_t& args)
      : batch_dim(args.batch_dim),
        head_dim(args.head_dim),
        batch_size(args.batch_size),
        seq_len(args.sequence_length),
        hs_rsqrt_scale(args.hs_rsqrt_scale),
        ptr_q(args.ptr_q),
        ptr_k(args.ptr_k),
        ptr_v(args.ptr_v),
        ptr_o(args.ptr_o),
        ptr_m(args.ptr_m),
        ptr_b(args.ptr_b),
        T_r(args.sequence_length / B_r),
        T_c(args.sequence_length / B_c),
        t_y(param_S::m_w / param_S::m_s),
        t_x(param_S::n_w / param_S::n_s) {}

  template <
      uint32_t m_w_,
      uint32_t n_w_,
      uint32_t k_w_,
      uint32_t n_s_,
      uint32_t k_s_>
  struct param_S_t {
    using compute_attr =
        gpu::xetla::group::compute_attr_t<dtype_q, dtype_k, dtype_acc>;
    struct fine_tuning {
      static constexpr uint32_t periodic_sync_interval = 8;
      static constexpr uint32_t prefetch_distance = 3;
    };
    // should larger than 8
    static constexpr uint32_t k_iter_num = k_s_;
    using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<
        k_iter_num,
        fine_tuning::prefetch_distance,
        fine_tuning::periodic_sync_interval>;
    using compute_policy = gpu::xetla::group::compute_policy_default_xmx<
        compute_attr,
        perf_tuning_knob,
        gpu::xetla::gpu_arch::Xe>;
    using mem_desc_input_q = gpu::xetla::mem_desc_t<
        dtype_q,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    using mem_desc_input_k = gpu::xetla::mem_desc_t<
        dtype_k,
        gpu::xetla::mem_layout::col_major,
        gpu::xetla::mem_space::global>;
    static constexpr int m_w = m_w_;
    static constexpr int n_w = n_w_;
    static constexpr int n_s = n_s_;
    static_assert(n_w % n_s == 0, "invalid configuration for n_w, n_s");
    static_assert(
        m_w * n_w % (n_s * thread_num) == 0,
        "invalid configuration for m_w");
    static constexpr int m_s = m_w * n_w / n_s / thread_num;
    using tile_shape = gpu::xetla::group::tile_shape_t<n_w, m_w, n_s, m_s>;
    using brgemm_t = gpu::xetla::group::brgemm_t<
        compute_policy,
        tile_shape,
        mem_desc_input_q,
        mem_desc_input_k>;
    using mat_tile_shape = brgemm_t::tile_shape;
    using mat_out_t = brgemm_t::matAcc_t;
    static constexpr int split_k_cnt = k_w_ / k_iter_num;
    static constexpr int barrier_count = brgemm_t::barrier_count;
    static constexpr int slm_size = brgemm_t::slm_size;
  };

  using param_S = param_S_t<
      B_r,
      B_c,
      H,
      tuning_parameter::matS_n_s,
      tuning_parameter::matS_k_s>;

  struct param_P {
    using mat_type = param_S::mat_out_t;
    using mat_in_t = mat_type;
    using mat_out_t = mat_type;
    using mat_tile_shape = param_S::mat_tile_shape;

    using rowmax_t = gpu::xetla::
        xetla_vector<typename mat_out_t::dtype, mat_in_t::tile_size_y>;
    using rowsum_t = rowmax_t;
    static constexpr int vec_length = rowmax_t::length;
    using vec_dtype = rowmax_t::element_type;

    using mem_desc_output_m = gpu::xetla::mem_desc_t<
        dtype_m,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    using factor_tile_desc_t = gpu::xetla::subgroup::tile_desc_t<
        vec_length,
        1,
        vec_length,
        1,
        gpu::xetla::reg_layout::tiled>;
    using factor_tile_t =
        gpu::xetla::subgroup::tile_t<dtype_m, factor_tile_desc_t>;
    using factor_payload_t = gpu::xetla::subgroup::mem_payload_t<
        dtype_m,
        factor_tile_desc_t,
        gpu::xetla::msg_type::block_1d,
        mem_desc_output_m::layout,
        mem_desc_output_m::space,
        param_S::brgemm_t::arch_tag>;

    static constexpr int reduce_list_ylength = mat_tile_shape::wg_size_y;
    static constexpr int reduce_list_xlength = mat_tile_shape::wg_size_x;
    static constexpr int reduce_elem_count = vec_length;

    using wg_reduce_max_t = gpu::xetla::group::group_reduce_t<
        typename mat_out_t::dtype,
        1,
        vec_length,
        gpu::xetla::reduce_op::max,
        reduce_list_xlength,
        true,
        factor_payload_t::arch_tag>;
    using wg_reduce_sum_t = gpu::xetla::group::group_reduce_t<
        typename mat_out_t::dtype,
        1,
        vec_length,
        gpu::xetla::reduce_op::sum,
        reduce_list_xlength,
        true,
        factor_payload_t::arch_tag>;

    using reduce_nbarrier_t =
        gpu::xetla::xetla_nbarrier_t<reduce_list_xlength, reduce_list_xlength>;
    static constexpr int reduce_barrier_count =
        (reduce_list_xlength > 0) ? reduce_list_ylength : 0;
    static constexpr int reduce_slm_size = (reduce_list_xlength > 0)
        ? (reduce_list_ylength * reduce_list_xlength * reduce_elem_count *
           sizeof(vec_dtype))
        : 0;

    static constexpr uint32_t slm_base_addr = 0;
    using mem_desc_output_p = gpu::xetla::mem_desc_t<
        dtype_p,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::local>;
    using epilogue_t = gpu::xetla::group::epilogue_t<
        gpu::xetla::group::epilogue_policy_default<gpu::xetla::gpu_arch::Xe>,
        mat_tile_shape,
        mem_desc_output_p>;
    using store_nbarrier_t =
        gpu::xetla::xetla_nbarrier_t<thread_num, thread_num>;
    static constexpr int local_store_barrier_count = 1;
    static constexpr int local_store_slm_size =
        mat_type::tile_elems * sizeof(mat_type::dtype);

    static constexpr int barrier_count =
        std::max({reduce_barrier_count, local_store_barrier_count});
    static constexpr int slm_size =
        std::max({reduce_slm_size, local_store_slm_size});
  };

  template <uint32_t m_w_, uint32_t n_w_, uint32_t k_w_, uint32_t k_s_>
  struct param_O_t {
    using compute_attr =
        gpu::xetla::group::compute_attr_t<dtype_p, dtype_acc, dtype_acc>;
    struct fine_tuning {
      static constexpr uint32_t periodic_sync_interval = 8;
      static constexpr uint32_t prefetch_distance = 3;
    };
    // should larger than 8
    static constexpr uint32_t k_iter_num = k_s_;
    using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<
        k_iter_num,
        fine_tuning::prefetch_distance,
        fine_tuning::periodic_sync_interval>;
    using compute_policy = gpu::xetla::group::compute_policy_default_xmx<
        compute_attr,
        perf_tuning_knob,
        gpu::xetla::gpu_arch::Xe>;
    // TODO: load P from slm
    using mem_desc_input_p = gpu::xetla::mem_desc_t<
        dtype_m,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    using mem_desc_input_v = gpu::xetla::mem_desc_t<
        dtype_v,
        gpu::xetla::mem_layout::row_major,
        gpu::xetla::mem_space::global>;
    static constexpr int m_w = m_w_;
    static constexpr int n_w = n_w_;
    static constexpr int v_s = n_w * param_S::n_s / param_S::n_w;
    static constexpr int n_s = v_s;
    static_assert(n_w % n_s == 0, "invalid configuration for n_w, n_s");
    static_assert(
        m_w * n_w % (n_s * thread_num) == 0,
        "invalid configuration for m_w");
    static constexpr int m_s = m_w * n_w / n_s / thread_num;
    using tile_shape = gpu::xetla::group::tile_shape_t<n_w, m_w, n_s, m_s>;
    using brgemm_t = gpu::xetla::group::brgemm_t<
        compute_policy,
        tile_shape,
        mem_desc_input_p,
        mem_desc_input_v>;
    using mat_out_t = brgemm_t::matAcc_t;
    static constexpr int split_k_cnt = k_w_ / k_iter_num;
    static constexpr int barrier_count = brgemm_t::barrier_count;
    static constexpr int slm_size = brgemm_t::slm_size;
  };

  using param_O =
      param_O_t<param_S::m_w, H, param_S::n_w, tuning_parameter::matO_k_s>;

  struct utils;
  struct program;

 public:
  sycl::nd_range<3> get_nd_range() const {
    // sycl::range<3> global_range {batch_size, T_r, T_c};
    sycl::range<3> global_range{batch_size, T_r, 1};
    sycl::range<3> local_range{1, t_y, t_x};
    sycl::nd_range<3> nd_range(global_range * local_range, local_range);

    return nd_range;
  }

  __XETLA_API KERNEL_FUNC void run(gpu::xetla::xetla_exec_item<3>& ei) const;
};

} // namespace xetla
} // namespace xpu