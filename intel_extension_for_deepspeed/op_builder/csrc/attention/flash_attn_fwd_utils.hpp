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

#include "flash_attn_fwd_kernel.hpp"

struct FLASH_ATTENTION_FWD_H128::utils {

    template <int dims = 1>
    class work_item;

    template <int dims = 1>
    static __XETLA_API KERNEL_FUNC work_item<dims> make_work_item(
            gpu::xetla::xetla_exec_item<dims> &ei);

    static __XETLA_API KERNEL_FUNC uint32_t matrix_size(uint32_t seq_len);
    static __XETLA_API KERNEL_FUNC uint32_t factor_size(uint32_t seq_len);

    template <uint32_t block_elems>
    struct calculate_new_ml_op_t {
        template <typename rowmax_t, typename rowsum_t>
        __XETLA_API KERNEL_FUNC void operator()(rowmax_t &m_new_vec,
                rowmax_t &m_tilde_vec, rowmax_t &m_vec, rowsum_t &l_new_vec,
                rowsum_t &l_tilde_vec, rowsum_t &l_vec);
    };

    struct row_exp_mul_op_t {
        template <typename rowvec_t, typename matAcc_t>
        __XETLA_API KERNEL_FUNC void operator()(
                rowvec_t &new_vec, rowvec_t &vec, matAcc_t &mat);
    };

    struct row_mul_op_t {
        template <typename rowvec_t, typename matAcc_t>
        __XETLA_API KERNEL_FUNC void operator()(rowvec_t &vec, matAcc_t &mat);
    };

    struct row_div_op_t {
        template <typename rowvec_t, typename matAcc_t>
        __XETLA_API KERNEL_FUNC void operator()(rowvec_t &vec, matAcc_t &mat);
    };
};

template <int dims>
class FLASH_ATTENTION_FWD_H128::utils::work_item {
public:
    work_item() = default;
    explicit work_item(gpu::xetla::xetla_exec_item<dims> &ei)
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

    void update_group_coord(int D, uint32_t val) { local_group_[D] = val; }

    gpu::xetla::xetla_exec_item<dims> ei_;

private:
    gpu::xetla::xetla_vector<uint32_t, dims> local_group_;
};

template <int dims>
__XETLA_API KERNEL_FUNC FLASH_ATTENTION_FWD_H128::utils::work_item<dims>
FLASH_ATTENTION_FWD_H128::utils::make_work_item(
        gpu::xetla::xetla_exec_item<dims> &ei) {
    return work_item<dims>(ei);
}

__XETLA_API KERNEL_FUNC uint32_t FLASH_ATTENTION_FWD_H128::utils::matrix_size(
        uint32_t seq_len) {
    return seq_len * H;
}

__XETLA_API KERNEL_FUNC uint32_t FLASH_ATTENTION_FWD_H128::utils::factor_size(
        uint32_t seq_len) {
    return seq_len * 2;
}

template <uint32_t block_elems>
template <typename rowmax_t, typename rowsum_t>
__XETLA_API KERNEL_FUNC void
FLASH_ATTENTION_FWD_H128::utils::calculate_new_ml_op_t<block_elems>::operator()(
        rowmax_t &m_new_vec, rowmax_t &m_tilde_vec, rowmax_t &m_vec,
        rowsum_t &l_new_vec, rowsum_t &l_tilde_vec, rowsum_t &l_vec) {
    using dtype = rowmax_t::element_type;
    static constexpr int N = rowmax_t::length;

    static_assert(rowmax_t::length == rowsum_t::length, "vec size mismatch");
    static_assert(std::is_same<typename rowmax_t::element_type,
                          typename rowsum_t::element_type>::value,
            "dtype mismatch");
#pragma unroll
    for (int i = 0; i < N / block_elems; ++i) {
        auto m_vec_blk = m_vec.xetla_select<block_elems, 1>(i * block_elems);
        auto m_tilde_vec_blk
                = m_tilde_vec.xetla_select<block_elems, 1>(i * block_elems);
        auto m_new_vec_blk
                = m_new_vec.xetla_select<block_elems, 1>(i * block_elems);
        auto l_vec_blk = l_vec.xetla_select<block_elems, 1>(i * block_elems);
        auto l_tilde_vec_blk
                = l_tilde_vec.xetla_select<block_elems, 1>(i * block_elems);
        auto l_new_vec_blk
                = l_new_vec.xetla_select<block_elems, 1>(i * block_elems);
        // 1. m_new_i = max(m_i, m_tilde_ij)
        m_new_vec_blk = gpu::xetla::xetla_max<dtype, block_elems>(
                m_vec_blk, m_tilde_vec_blk);
        // 2. l_new_i = sum(exp(m_i - m_new_i) * l_i, exp(m_tilde_ij - m_new_i) * l_tilde_ij)
        gpu::xetla::xetla_vector<dtype, block_elems> diff_m
                = m_vec_blk - m_new_vec_blk;
        diff_m = gpu::xetla::xetla_exp<dtype, block_elems>(diff_m);
        gpu::xetla::xetla_vector<dtype, block_elems> diff_m_tilde
                = m_tilde_vec_blk - m_new_vec_blk;
        diff_m_tilde = gpu::xetla::xetla_exp<dtype, block_elems>(diff_m_tilde);
        l_new_vec_blk = diff_m * l_vec_blk + diff_m_tilde * l_tilde_vec_blk;
    }
    static constexpr int remain_elems = N % block_elems;
    if constexpr (remain_elems != 0) {
        static constexpr int offset = N - remain_elems;
        auto m_vec_blk = m_vec.xetla_select<remain_elems, 1>(offset);
        auto m_tilde_vec_blk
                = m_tilde_vec.xetla_select<remain_elems, 1>(offset);
        auto m_new_vec_blk = m_new_vec.xetla_select<remain_elems, 1>(offset);
        auto l_vec_blk = l_vec.xetla_select<remain_elems, 1>(offset);
        auto l_tilde_vec_blk
                = l_tilde_vec.xetla_select<remain_elems, 1>(offset);
        auto l_new_vec_blk = l_new_vec.xetla_select<remain_elems, 1>(offset);
        // 1. m_new_i = max(m_i, m_tilde_ij)
        m_new_vec_blk = gpu::xetla::xetla_max<dtype, remain_elems>(
                m_vec_blk, m_tilde_vec_blk);
        // 2. l_new_i = sum(exp(m_i - m_new_i) * l_i, exp(m_tilde_ij - m_new_i) * l_tilde_ij)
        gpu::xetla::xetla_vector<dtype, remain_elems> diff_m
                = m_vec_blk - m_new_vec_blk;
        diff_m = gpu::xetla::xetla_exp<dtype, remain_elems>(diff_m);
        gpu::xetla::xetla_vector<dtype, remain_elems> diff_m_tilde
                = m_tilde_vec_blk - m_new_vec_blk;
        diff_m_tilde = gpu::xetla::xetla_exp<dtype, remain_elems>(diff_m_tilde);
        l_new_vec_blk = diff_m * l_vec_blk + diff_m_tilde * l_tilde_vec_blk;
    }
}

template <typename rowvec_t, typename matAcc_t>
__XETLA_API KERNEL_FUNC void
FLASH_ATTENTION_FWD_H128::utils::row_exp_mul_op_t::operator()(
        rowvec_t &new_vec, rowvec_t &vec, matAcc_t &mat) {
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
    static_assert(std::is_same<typename matAcc_t::dtype,
                          typename rowvec_t::element_type>::value,
            "dtype mismatch");

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; ++i) {
        auto new_vec_blk
                = new_vec.xetla_select<block_size_y, 1>(i * block_size_y);
        auto vec_blk = vec.xetla_select<block_size_y, 1>(i * block_size_y);
        auto diff_blk = vec_blk - new_vec_blk;
        diff_blk = gpu::xetla::xetla_exp<dtype, block_size_y>(diff_blk);
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_blk_2d
                    = (mat.reg)
                              .xetla_select<block_elems, 1>(
                                      (i * num_block_x + j) * block_elems)
                              .xetla_format<dtype, block_size_y,
                                      block_size_x>();
#pragma unroll
            for (int r = 0; r < block_size_y; ++r) {
                auto mat_reg_row = reg_blk_2d.row(r);
                mat_reg_row = mat_reg_row * diff_blk[r];
            }
        }
    }
    static constexpr int remain_block_size_y = tile_size_y % block_size_y;
    if constexpr (remain_block_size_y != 0) {
        static constexpr int remain_block_start_y
                = tile_size_y - remain_block_size_y;
        auto new_vec_blk = new_vec.xetla_select<remain_block_size_y, 1>(
                remain_block_start_y);
        auto vec_blk = vec.xetla_select<remain_block_size_y, 1>(
                remain_block_start_y);
        auto diff_blk = vec_blk - new_vec_blk;
        diff_blk = gpu::xetla::xetla_exp<dtype, remain_block_size_y>(diff_blk);
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_blk_2d = (mat.reg)
                                      .xetla_select<block_elems, 1>(
                                              remain_block_start_y * tile_size_x
                                              + j * block_elems)
                                      .xetla_format<dtype, remain_block_size_y,
                                              block_size_x>();
#pragma unroll
            for (int r = 0; r < remain_block_size_y; ++r) {
                auto mat_reg_row = reg_blk_2d.row(r);
                mat_reg_row = mat_reg_row * diff_blk[r];
            }
        }
    }
}

template <typename rowvec_t, typename matAcc_t>
__XETLA_API KERNEL_FUNC void
FLASH_ATTENTION_FWD_H128::utils::row_mul_op_t::operator()(
        rowvec_t &vec, matAcc_t &mat) {
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
    static_assert(std::is_same<typename matAcc_t::dtype,
                          typename rowvec_t::element_type>::value,
            "dtype mismatch");

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; ++i) {
        auto vec_blk = vec.xetla_select<block_size_y, 1>(i * block_size_y);
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_blk_2d
                    = (mat.reg)
                              .xetla_select<block_elems, 1>(
                                      (i * num_block_x + j) * block_elems)
                              .xetla_format<dtype, block_size_y,
                                      block_size_x>();
#pragma unroll
            for (int r = 0; r < block_size_y; ++r) {
                auto mat_reg_row = reg_blk_2d.row(r);
                mat_reg_row = mat_reg_row * vec_blk[r];
            }
        }
    }
    static constexpr int remain_block_size_y = tile_size_y % block_size_y;
    if constexpr (remain_block_size_y != 0) {
        static constexpr int remain_block_start_y
                = tile_size_y - remain_block_size_y;
        auto vec_blk = vec.xetla_select<remain_block_size_y, 1>(
                remain_block_start_y);
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_blk_2d = (mat.reg)
                                      .xetla_select<block_elems, 1>(
                                              remain_block_start_y * tile_size_x
                                              + j * block_elems)
                                      .xetla_format<dtype, remain_block_size_y,
                                              block_size_x>();
#pragma unroll
            for (int r = 0; r < remain_block_size_y; ++r) {
                auto mat_reg_row = reg_blk_2d.row(r);
                mat_reg_row = mat_reg_row * vec_blk[r];
            }
        }
    }
}

template <typename rowvec_t, typename matAcc_t>
__XETLA_API KERNEL_FUNC void
FLASH_ATTENTION_FWD_H128::utils::row_div_op_t::operator()(
        rowvec_t &vec, matAcc_t &mat) {
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
    static_assert(std::is_same<typename matAcc_t::dtype,
                          typename rowvec_t::element_type>::value,
            "dtype mismatch");

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; ++i) {
        auto vec_blk = vec.xetla_select<block_size_y, 1>(i * block_size_y);
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_blk_2d
                    = (mat.reg)
                              .xetla_select<block_elems, 1>(
                                      (i * num_block_x + j) * block_elems)
                              .xetla_format<dtype, block_size_y,
                                      block_size_x>();
#pragma unroll
            for (int r = 0; r < block_size_y; ++r) {
                auto mat_reg_row = reg_blk_2d.row(r);
                mat_reg_row = mat_reg_row / vec_blk[r];
            }
        }
    }
    static constexpr int remain_block_size_y = tile_size_y % block_size_y;
    if constexpr (remain_block_size_y != 0) {
        static constexpr int remain_block_start_y
                = tile_size_y - remain_block_size_y;
        auto vec_blk = vec.xetla_select<remain_block_size_y, 1>(
                remain_block_start_y);
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_blk_2d = (mat.reg)
                                      .xetla_select<block_elems, 1>(
                                              remain_block_start_y * tile_size_x
                                              + j * block_elems)
                                      .xetla_format<dtype, remain_block_size_y,
                                              block_size_x>();
#pragma unroll
            for (int r = 0; r < remain_block_size_y; ++r) {
                auto mat_reg_row = reg_blk_2d.row(r);
                mat_reg_row = mat_reg_row / vec_blk[r];
            }
        }
    }
}