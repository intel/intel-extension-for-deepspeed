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

#include <limits>
#include "flash_attn_fwd_kernel.hpp"

namespace xpu {
namespace xetla {

template <typename tuning_parameter_>
struct FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::utils {
  static constexpr float inf = std::numeric_limits<float>::infinity();

  template <int dims = 1>
  class work_item;

  using work_item3 = work_item<3>;

  template <int dims = 1>
  static __XETLA_API KERNEL_FUNC work_item<dims> make_work_item(
      gpu::xetla::xetla_exec_item<dims>& ei);

  static __XETLA_API KERNEL_FUNC uint32_t matrix_size(uint32_t seq_len);
  static __XETLA_API KERNEL_FUNC uint32_t factor_size(uint32_t seq_len);

  template <uint32_t block_elems>
  struct calculate_new_ml_op_t {
    template <typename rowmax_t, typename rowsum_t>
    __XETLA_API KERNEL_FUNC void operator()(
        rowmax_t& m_new_vec,
        rowmax_t& m_tilde_vec,
        rowmax_t& m_vec,
        rowsum_t& l_new_vec,
        rowsum_t& l_tilde_vec,
        rowsum_t& l_vec);
  };

  struct row_exp_mul_op_t {
    template <typename rowvec_t, typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(
        rowvec_t& new_vec,
        rowvec_t& vec,
        matAcc_t& mat);
  };

  struct row_mul_op_t {
    template <typename rowvec_t, typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(rowvec_t& vec, matAcc_t& mat);
  };

  struct row_div_op_t {
    template <typename rowvec_t, typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(rowvec_t& vec, matAcc_t& mat);
  };

  template <typename tile_shape_t>
  static __XETLA_API KERNEL_FUNC int check_diag_intersection(
      utils::work_item3& ei);

  template <typename tile_shape_t, typename matAcc_t>
  static __XETLA_API KERNEL_FUNC void causal_mask(
      utils::work_item3& ei,
      uint32_t seq_len,
      matAcc_t& mask);
};

template <typename tuning_parameter_>
template <int dims>
class FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::utils::work_item {
 public:
  work_item() = default;
  explicit work_item(gpu::xetla::xetla_exec_item<dims>& ei)
      : ei_(ei), local_group_(0) {}

  inline uint32_t get_local_linear_id() const {
    return ei_.get_local_linear_id();
  }

  inline uint32_t get_local_id(int dimension) const {
    return ei_.get_local_id(dimension);
  }

  inline uint32_t get_local_range(int dimension) const {
    return ei_.get_local_range(dimension);
  }

  inline uint32_t get_group(int dimension) const {
    return ei_.get_group(dimension) + local_group_[dimension];
  }

  inline uint32_t get_global_linear_id() const {
    // TODO: imcompatible with local_group_
    uint32_t ei_id = ei_.get_global_linear_id();

    uint32_t local_id = 0;

    return ei_id + local_id;
  }

  void update_group_coord(int D, uint32_t val) {
    local_group_[D] = val;
  }

  gpu::xetla::xetla_exec_item<dims> ei_;

 private:
  gpu::xetla::xetla_vector<uint32_t, dims> local_group_;
};

template <typename tuning_parameter_>
template <int dims>
__XETLA_API KERNEL_FUNC
    FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::utils::work_item<dims>
    FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::utils::make_work_item(
        gpu::xetla::xetla_exec_item<dims>& ei) {
  return work_item<dims>(ei);
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC uint32_t
FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::utils::matrix_size(
    uint32_t seq_len) {
  return seq_len * H;
}

template <typename tuning_parameter_>
__XETLA_API KERNEL_FUNC uint32_t
FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::utils::factor_size(
    uint32_t seq_len) {
  return seq_len * 2;
}

template <typename tuning_parameter_>
template <uint32_t block_elems>
template <typename rowmax_t, typename rowsum_t>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    utils::calculate_new_ml_op_t<block_elems>::operator()(
        rowmax_t& m_new_vec,
        rowmax_t& m_tilde_vec,
        rowmax_t& m_vec,
        rowsum_t& l_new_vec,
        rowsum_t& l_tilde_vec,
        rowsum_t& l_vec) {
  using dtype = rowmax_t::element_type;
  static constexpr int N = rowmax_t::length;

  static_assert(rowmax_t::length == rowsum_t::length, "vec size mismatch");
  static_assert(
      std::is_same<
          typename rowmax_t::element_type,
          typename rowsum_t::element_type>::value,
      "dtype mismatch");
#pragma unroll
  for (int i = 0; i < N / block_elems; ++i) {
    auto m_vec_blk = m_vec.xetla_select<block_elems, 1>(i * block_elems);
    auto m_tilde_vec_blk =
        m_tilde_vec.xetla_select<block_elems, 1>(i * block_elems);
    auto m_new_vec_blk =
        m_new_vec.xetla_select<block_elems, 1>(i * block_elems);
    auto l_vec_blk = l_vec.xetla_select<block_elems, 1>(i * block_elems);
    auto l_tilde_vec_blk =
        l_tilde_vec.xetla_select<block_elems, 1>(i * block_elems);
    auto l_new_vec_blk =
        l_new_vec.xetla_select<block_elems, 1>(i * block_elems);
    // 1. m_new_i = max(m_i, m_tilde_ij)
    m_new_vec_blk =
        gpu::xetla::xetla_max<dtype, block_elems>(m_vec_blk, m_tilde_vec_blk);
    // 2. l_new_i = sum(exp(m_i - m_new_i) * l_i, exp(m_tilde_ij - m_new_i) *
    // l_tilde_ij)
    gpu::xetla::xetla_vector<dtype, block_elems> diff_m =
        m_vec_blk - m_new_vec_blk;
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      gpu::xetla::xetla_mask<block_elems> m_mask = m_vec_blk <= -utils::inf;
      m_mask |= m_new_vec_blk <= -utils::inf;
      diff_m.merge(-utils::inf, m_mask);
    }
    diff_m = gpu::xetla::xetla_exp<dtype, block_elems>(diff_m);
    gpu::xetla::xetla_vector<dtype, block_elems> diff_m_tilde =
        m_tilde_vec_blk - m_new_vec_blk;
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      gpu::xetla::xetla_mask<block_elems> m_tilde_mask =
          m_tilde_vec_blk <= -utils::inf;
      m_tilde_mask |= m_new_vec_blk <= -utils::inf;
      diff_m_tilde.merge(-utils::inf, m_tilde_mask);
    }
    diff_m_tilde = gpu::xetla::xetla_exp<dtype, block_elems>(diff_m_tilde);
    l_new_vec_blk = diff_m * l_vec_blk + diff_m_tilde * l_tilde_vec_blk;
  }
  static constexpr int remain_elems = N % block_elems;
  if constexpr (remain_elems != 0) {
    static constexpr int offset = N - remain_elems;
    auto m_vec_blk = m_vec.xetla_select<remain_elems, 1>(offset);
    auto m_tilde_vec_blk = m_tilde_vec.xetla_select<remain_elems, 1>(offset);
    auto m_new_vec_blk = m_new_vec.xetla_select<remain_elems, 1>(offset);
    auto l_vec_blk = l_vec.xetla_select<remain_elems, 1>(offset);
    auto l_tilde_vec_blk = l_tilde_vec.xetla_select<remain_elems, 1>(offset);
    auto l_new_vec_blk = l_new_vec.xetla_select<remain_elems, 1>(offset);
    // 1. m_new_i = max(m_i, m_tilde_ij)
    m_new_vec_blk =
        gpu::xetla::xetla_max<dtype, remain_elems>(m_vec_blk, m_tilde_vec_blk);
    // 2. l_new_i = sum(exp(m_i - m_new_i) * l_i, exp(m_tilde_ij - m_new_i) *
    // l_tilde_ij)
    gpu::xetla::xetla_vector<dtype, remain_elems> diff_m =
        m_vec_blk - m_new_vec_blk;
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      gpu::xetla::xetla_mask<block_elems> m_mask = m_vec_blk <= -utils::inf;
      m_mask |= m_new_vec_blk <= -utils::inf;
      diff_m.merge(-utils::inf, m_mask);
    }
    diff_m = gpu::xetla::xetla_exp<dtype, remain_elems>(diff_m);
    gpu::xetla::xetla_vector<dtype, remain_elems> diff_m_tilde =
        m_tilde_vec_blk - m_new_vec_blk;
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      gpu::xetla::xetla_mask<block_elems> m_tilde_mask =
          m_tilde_vec_blk <= -utils::inf;
      m_tilde_mask |= m_new_vec_blk <= -utils::inf;
      diff_m_tilde.merge(-utils::inf, m_tilde_mask);
    }
    diff_m_tilde = gpu::xetla::xetla_exp<dtype, remain_elems>(diff_m_tilde);
    l_new_vec_blk = diff_m * l_vec_blk + diff_m_tilde * l_tilde_vec_blk;
  }
}

template <typename tuning_parameter_>
template <typename rowvec_t, typename matAcc_t>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    utils::row_exp_mul_op_t::operator()(
        rowvec_t& new_vec,
        rowvec_t& vec,
        matAcc_t& mat) {
  // mat[i, :] *= exp(vec[i] - new_vec[i])
  static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
  static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
  static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
  static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
  static constexpr uint32_t num_block_y = matAcc_t::num_block_y;
  static constexpr uint32_t num_block_x = matAcc_t::num_block_x;
  static constexpr uint32_t block_elems = matAcc_t::block_elems;

  using dtype = matAcc_t::dtype;

  static_assert(
      matAcc_t::tile_size_y == rowvec_t::length, "row number mismatch");
  static_assert(
      std::is_same<typename matAcc_t::dtype, typename rowvec_t::element_type>::
          value,
      "dtype mismatch");

#pragma unroll
  for (int i = 0; i < tile_size_y / block_size_y; ++i) {
    auto new_vec_blk = new_vec.xetla_select<block_size_y, 1>(i * block_size_y);
    auto vec_blk = vec.xetla_select<block_size_y, 1>(i * block_size_y);
    auto diff_blk = vec_blk - new_vec_blk;
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      gpu::xetla::xetla_mask<block_size_y> v_mask = vec_blk <= -utils::inf;
      v_mask |= new_vec_blk <= -utils::inf;
      diff_blk.merge(-utils::inf, v_mask);
    }
    diff_blk = gpu::xetla::xetla_exp<dtype, block_size_y>(diff_blk);
#pragma unroll
    for (int j = 0; j < num_block_x; j++) {
      auto reg_blk_2d =
          (mat.reg)
              .xetla_select<block_elems, 1>((i * num_block_x + j) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
      for (int r = 0; r < block_size_y; ++r) {
        auto mat_reg_row = reg_blk_2d.row(r);
        mat_reg_row = mat_reg_row * diff_blk[r];
      }
    }
  }
  static constexpr int remain_block_size_y = tile_size_y % block_size_y;
  if constexpr (remain_block_size_y != 0) {
    static constexpr int remain_block_start_y =
        tile_size_y - remain_block_size_y;
    auto new_vec_blk =
        new_vec.xetla_select<remain_block_size_y, 1>(remain_block_start_y);
    auto vec_blk =
        vec.xetla_select<remain_block_size_y, 1>(remain_block_start_y);
    auto diff_blk = vec_blk - new_vec_blk;
    if constexpr (enable_mask) {
      // special handling for masking the whole row
      gpu::xetla::xetla_mask<remain_block_size_y> v_mask =
          vec_blk <= -utils::inf;
      v_mask |= new_vec_blk <= -utils::inf;
      diff_blk.merge(-utils::inf, v_mask);
    }
    diff_blk = gpu::xetla::xetla_exp<dtype, remain_block_size_y>(diff_blk);
#pragma unroll
    for (int j = 0; j < num_block_x; j++) {
      auto reg_blk_2d =
          (mat.reg)
              .xetla_select<block_elems, 1>(
                  remain_block_start_y * tile_size_x + j * block_elems)
              .xetla_format<dtype, remain_block_size_y, block_size_x>();
#pragma unroll
      for (int r = 0; r < remain_block_size_y; ++r) {
        auto mat_reg_row = reg_blk_2d.row(r);
        mat_reg_row = mat_reg_row * diff_blk[r];
      }
    }
  }
}

template <typename tuning_parameter_>
template <typename rowvec_t, typename matAcc_t>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    utils::row_mul_op_t::operator()(rowvec_t& vec, matAcc_t& mat) {
  // mat[i, :] *= vec[i]
  static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
  static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
  static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
  static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
  static constexpr uint32_t num_block_y = matAcc_t::num_block_y;
  static constexpr uint32_t num_block_x = matAcc_t::num_block_x;
  static constexpr uint32_t block_elems = matAcc_t::block_elems;

  using dtype = matAcc_t::dtype;

  static_assert(
      matAcc_t::tile_size_y == rowvec_t::length, "row number mismatch");
  static_assert(
      std::is_same<typename matAcc_t::dtype, typename rowvec_t::element_type>::
          value,
      "dtype mismatch");

#pragma unroll
  for (int i = 0; i < tile_size_y / block_size_y; ++i) {
    auto vec_blk = vec.xetla_select<block_size_y, 1>(i * block_size_y);
#pragma unroll
    for (int j = 0; j < num_block_x; j++) {
      auto reg_blk_2d =
          (mat.reg)
              .xetla_select<block_elems, 1>((i * num_block_x + j) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
      for (int r = 0; r < block_size_y; ++r) {
        auto mat_reg_row = reg_blk_2d.row(r);
        mat_reg_row = mat_reg_row * vec_blk[r];
      }
    }
  }
  static constexpr int remain_block_size_y = tile_size_y % block_size_y;
  if constexpr (remain_block_size_y != 0) {
    static constexpr int remain_block_start_y =
        tile_size_y - remain_block_size_y;
    auto vec_blk =
        vec.xetla_select<remain_block_size_y, 1>(remain_block_start_y);
#pragma unroll
    for (int j = 0; j < num_block_x; j++) {
      auto reg_blk_2d =
          (mat.reg)
              .xetla_select<block_elems, 1>(
                  remain_block_start_y * tile_size_x + j * block_elems)
              .xetla_format<dtype, remain_block_size_y, block_size_x>();
#pragma unroll
      for (int r = 0; r < remain_block_size_y; ++r) {
        auto mat_reg_row = reg_blk_2d.row(r);
        mat_reg_row = mat_reg_row * vec_blk[r];
      }
    }
  }
}

template <typename tuning_parameter_>
template <typename rowvec_t, typename matAcc_t>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    utils::row_div_op_t::operator()(rowvec_t& vec, matAcc_t& mat) {
  // mat[i, :] /= vec[i]
  static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
  static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
  static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
  static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
  static constexpr uint32_t num_block_y = matAcc_t::num_block_y;
  static constexpr uint32_t num_block_x = matAcc_t::num_block_x;
  static constexpr uint32_t block_elems = matAcc_t::block_elems;

  using dtype = matAcc_t::dtype;

  static_assert(
      matAcc_t::tile_size_y == rowvec_t::length, "row number mismatch");
  static_assert(
      std::is_same<typename matAcc_t::dtype, typename rowvec_t::element_type>::
          value,
      "dtype mismatch");

#pragma unroll
  for (int i = 0; i < tile_size_y / block_size_y; ++i) {
    auto vec_blk = vec.xetla_select<block_size_y, 1>(i * block_size_y);
#pragma unroll
    for (int j = 0; j < num_block_x; j++) {
      auto reg_blk_2d =
          (mat.reg)
              .xetla_select<block_elems, 1>((i * num_block_x + j) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
      for (int r = 0; r < block_size_y; ++r) {
        auto mat_reg_row = reg_blk_2d.row(r);
        mat_reg_row = mat_reg_row / vec_blk[r];
      }
    }
  }
  static constexpr int remain_block_size_y = tile_size_y % block_size_y;
  if constexpr (remain_block_size_y != 0) {
    static constexpr int remain_block_start_y =
        tile_size_y - remain_block_size_y;
    auto vec_blk =
        vec.xetla_select<remain_block_size_y, 1>(remain_block_start_y);
#pragma unroll
    for (int j = 0; j < num_block_x; j++) {
      auto reg_blk_2d =
          (mat.reg)
              .xetla_select<block_elems, 1>(
                  remain_block_start_y * tile_size_x + j * block_elems)
              .xetla_format<dtype, remain_block_size_y, block_size_x>();
#pragma unroll
      for (int r = 0; r < remain_block_size_y; ++r) {
        auto mat_reg_row = reg_blk_2d.row(r);
        mat_reg_row = mat_reg_row / vec_blk[r];
      }
    }
  }
}

template <typename tuning_parameter_>
template <typename tile_shape_t>
__XETLA_API KERNEL_FUNC int FLASH_ATTENTION_FWD_IMPL<
    tuning_parameter_>::utils::check_diag_intersection(utils::work_item3& ei) {
  // returns 0 if has intersection with diagonal of S
  // returns 1 if is in upper triangle
  // returns -1 if is in lower triangle
  int i_w = ei.get_group(1);
  int j_w = ei.get_group(2);
  constexpr int n_w = tile_shape_t::wg_tile_size_x;
  constexpr int m_w = tile_shape_t::wg_tile_size_y;

  {
    // check intersection with diagonal
    // 1. (j_w * B_c) <= i_w * B_r < (j_w * B_c + B_c)
    // 1. (j_w * B_c) <= i_w * B_r + B_r - 1 < (j_w * B_c + B_c)
    // 3. (i_w * B_r) <= j_w * B_c < (i_w * B_r + B_r)
    // 4. (i_w * B_r) <= j_w * B_c + B_c - 1 < (i_w * B_r + B_r)
    gpu::xetla::xetla_vector<int32_t, 4> lp(
        {j_w * n_w, j_w * n_w, i_w * m_w, i_w * m_w});
    gpu::xetla::xetla_vector<int32_t, 4> up(
        {(j_w + 1) * n_w, (j_w + 1) * n_w, (i_w + 1) * m_w, (i_w + 1) * m_w});
    gpu::xetla::xetla_vector<int32_t, 4> eg(
        {i_w * m_w, (i_w + 1) * m_w - 1, j_w * n_w, (j_w + 1) * n_w - 1});
    gpu::xetla::xetla_mask<4> lp_mask = lp <= eg;
    gpu::xetla::xetla_mask<4> up_mask = eg < up;

    // check if any condition is valid
    lp_mask &= up_mask;
    bool is_diag = lp_mask.any() > 0 ? true : false;
    if (is_diag) {
      return 0;
    }
  }
  {
    // checker lower triangle
    // min(i_w * B_r + ii) > max(j_w * B_c + jj)
    if (i_w * m_w > j_w * n_w + n_w - 1) {
      return -1;
    }
  }
  // check upper triangle
  // max(i_w * B_r + ii) < min(j_w * B_c + jj)
  if (i_w * m_w + m_w - 1 < j_w * n_w) {
    return 1;
  }
  // should not reach here
  return 0;
}

template <typename tuning_parameter_>
template <typename tile_shape_t, typename matAcc_t>
__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_IMPL<tuning_parameter_>::
    utils::causal_mask(
        utils::work_item3& ei,
        uint32_t seq_len,
        matAcc_t& matM) {
  // mask upper triangle excluding diagonal:
  // M_ij[ii, jj] += -inf, for i_g < j_g
  // where:
  //   - shape(M_ij) = [m_s, n_s]
  //   - i_g = g(f(ii, jj))[0]
  //   - j_g = g(f(ii, jj))[1]
  //   - f: (i, j) |-> x
  //   - g: x |-> (i_g, j_g) = g3(g2(g1(x)))
  //   - g1: x |-> (r_b, c_b, i_b, j_b)
  //   - g2: (r_b, c_b, i_b, j_b) |-> (i_l, j_l) = (i_b * b_y + r_b, j_b * b_x +
  //   c_b)
  //   - g3: (i_l, j_l) |-> (i_g, j_g) = (i_w * m_w + i_s * m_s + i_l, j_w * n_w
  //   + j_s * n_s + j_l)
  //   - x: register linear index
  int i_w = ei.get_group(1);
  int j_w = ei.get_group(2);
  constexpr int n_w = tile_shape_t::wg_tile_size_x;
  constexpr int m_w = tile_shape_t::wg_tile_size_y;
  typename tile_shape_t::work_group_t g(ei.get_local_linear_id());
  int i_s = g.get_id() / tile_shape_t::wg_size_x;
  int j_s = g.get_id() % tile_shape_t::wg_size_x;
  constexpr int n_s = tile_shape_t::sg_tile_size_x;
  constexpr int m_s = tile_shape_t::sg_tile_size_y;
  constexpr uint32_t b_y = matAcc_t::block_size_y;
  constexpr uint32_t b_x = matAcc_t::block_size_x;
  constexpr uint32_t n_y = matAcc_t::num_block_y;
  constexpr uint32_t n_x = matAcc_t::num_block_x;
  constexpr uint32_t block_elems = matAcc_t::block_elems;

  using dtype = matAcc_t::dtype;

  int i_o = i_w * m_w + i_s * m_s;
  int j_o = j_w * n_w + j_s * n_s;

#pragma unroll
  for (int i_b = 0; i_b < m_s / b_y; ++i_b) {
#pragma unroll
    for (int j_b = 0; j_b < n_x; j_b++) {
      auto reg_blk_2d =
          (matM.reg)
              .xetla_select<block_elems, 1>((i_b * n_x + j_b) * block_elems)
              .xetla_format<dtype, b_y, b_x>();
#pragma unroll
      for (int r_b = 0; r_b < b_y; ++r_b) {
        auto mat_reg_row = reg_blk_2d.row(r_b);
        uint32_t i_g = i_o + i_b * b_y + r_b;
        gpu::xetla::xetla_vector<uint32_t, b_x> j_g =
            gpu::xetla::xetla_vector_gen<uint32_t, b_x>(j_o + j_b * b_x, 1);
        gpu::xetla::xetla_mask<b_x> mask = i_g < j_g;
        mat_reg_row.merge(-utils::inf, mask);
      }
    }
  }
  static constexpr int b_tilde_y = m_s % b_y;
  if constexpr (b_tilde_y != 0) {
    static constexpr int remain_block_start_y = m_s - b_tilde_y;
#pragma unroll
    for (int j_b = 0; j_b < n_x; j_b++) {
      auto reg_blk_2d =
          (matM.reg)
              .xetla_select<b_tilde_y * b_x, 1>(
                  (remain_block_start_y * n_x + j_b) * block_elems)
              .xetla_format<dtype, b_tilde_y, b_x>();
#pragma unroll
      for (int r_b = 0; r_b < b_tilde_y; ++r_b) {
        auto mat_reg_row = reg_blk_2d.row(r_b);
        uint32_t i_g = i_o + remain_block_start_y + r_b;
        gpu::xetla::xetla_vector<uint32_t, b_x> j_g =
            gpu::xetla::xetla_vector_gen<uint32_t, b_x>(j_o + j_b * b_x, 1);
        gpu::xetla::xetla_mask<b_x> mask = i_g < j_g;
        mat_reg_row.merge(-utils::inf, mask);
      }
    }
  }
}

} // namespace xetla
} // namespace xpu