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
#include "flash_attn_fwd_utils.hpp"

struct FLASH_ATTENTION_FWD_H128::program {
    static __XETLA_API KERNEL_FUNC void initialization(utils::work_item<3> &ei);

    static __XETLA_API KERNEL_FUNC void calculate_S_ij(utils::work_item<3> &ei,
            uint32_t seq_len, param_S::mat_out_t &matS, dtype_q *ptr_q,
            dtype_k *ptr_k, float hs_rsqrt_scale);

    static __XETLA_API KERNEL_FUNC void calculate_P_ij(utils::work_item<3> &ei,
            param_S::mat_out_t &matS, param_P::rowmax_t &m_tilde_vec,
            param_P::rowsum_t &l_tilde_vec);

    static __XETLA_API KERNEL_FUNC void load_ml(utils::work_item<3> &ei,
            dtype_m *ptr_m, param_P::rowmax_t &m_vec, param_P::rowsum_t &l_vec);

    static __XETLA_API KERNEL_FUNC void calculate_new_ml(
            utils::work_item<3> &ei, param_P::rowmax_t &m_new_vec,
            param_P::rowmax_t &m_tilde_vec, param_P::rowmax_t &m_vec,
            param_P::rowsum_t &l_new_vec, param_P::rowsum_t &l_tilde_vec,
            param_P::rowsum_t &l_vec);

    static __XETLA_API KERNEL_FUNC void calculate_PV(utils::work_item<3> &ei,
            uint32_t batch_size, uint32_t seq_len, uint32_t T_r, uint32_t T_c,
            param_O::mat_out_t &matO_new, param_P::mat_out_t &matP,
            dtype_m *ptr_m, dtype_v *ptr_v, dtype_b *ptr_b,
            param_P::rowmax_t &m_new_vec, param_P::rowmax_t &m_tilde_vec);

    static __XETLA_API KERNEL_FUNC void load_O(utils::work_item<3> &ei,
            param_O::mat_out_t &matO, param_P::rowmax_t &m_new_vec,
            param_P::rowmax_t &m_vec);

    static __XETLA_API KERNEL_FUNC void update_O(utils::work_item<3> &ei,
            param_O::mat_out_t &matO, param_O::mat_out_t &matO_new,
            param_P::rowsum_t &l_new_vec, param_P::rowsum_t &l_vec);

    static __XETLA_API KERNEL_FUNC void store_O(
            utils::work_item<3> &ei, param_O::mat_out_t &matO);

    static __XETLA_API KERNEL_FUNC void store_ml(utils::work_item<3> &ei,
            param_P::rowmax_t &m_vec, param_P::rowmax_t &m_new_vec,
            param_P::rowsum_t &l_vec, param_P::rowsum_t &l_new_vec);

    static __XETLA_API KERNEL_FUNC void store_P_ij(
            utils::work_item<3> &ei, param_P::mat_out_t &matP);

    static __XETLA_API KERNEL_FUNC void store_global_O(utils::work_item<3> &ei,
            uint32_t batch_dim, uint32_t head_dim, uint32_t seq_len,
            dtype_o *ptr_o, param_O::mat_out_t &matO);

    static __XETLA_API KERNEL_FUNC void store_global_ml(utils::work_item<3> &ei,
            dtype_m *ptr_m, param_P::factor_tile_t &m_store,
            param_P::factor_payload_t &m_payload,
            param_P::factor_tile_t &l_store,
            param_P::factor_payload_t &l_payload);

    static __XETLA_API KERNEL_FUNC void setup_ml_store(utils::work_item<3> &ei,
            uint32_t batch_size, uint32_t seq_len, uint32_t T_r, uint32_t T_c,
            uint32_t B_r, uint32_t B_c, dtype_m *ptr_m,
            param_P::factor_payload_t &m_payload,
            param_P::factor_payload_t &l_payload);
};

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::initialization(
        utils::work_item<3> &ei) {
    static constexpr uint32_t barrier_count = std::max({param_S::barrier_count,
            param_P::barrier_count, param_O::barrier_count});
    static constexpr uint32_t slm_size = std::max(
            {param_S::slm_size, param_P::slm_size, param_O::slm_size});
    gpu::xetla::xetla_nbarrier_init<barrier_count>();
    gpu::xetla::xetla_local_init<slm_size>();
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::calculate_S_ij(
        utils::work_item<3> &ei, uint32_t seq_len, param_S::mat_out_t &matS,
        dtype_q *ptr_q, dtype_k *ptr_k, float hs_rsqrt_scale) {
    // S_ij = Q_i @ K_j.T
    // where:
    //   - shape(Q_i)   = [B_r, H]
    //   - shape(K_j.T) = [H, B_c]
    //   - shape(S_ij)  = [B_r, B_c]
    int batch_idx = ei.get_group(0);
    int start_m = ei.get_group(1) * param_S::m_w;
    int start_n = ei.get_group(2) * param_S::n_w;
    int start_k = 0;
    param_S::mem_desc_input_q md_q(
            {ptr_q + batch_idx * utils::matrix_size(seq_len)}, {H, seq_len, H},
            {start_k, start_m});
    param_S::mem_desc_input_k md_k(
            {ptr_k + batch_idx * utils::matrix_size(seq_len)}, {seq_len, H, H},
            {start_n, start_k});
    param_S::brgemm_t matmul_op;
    param_S::brgemm_t::arguments_t args(md_q, md_k, param_S::split_k_cnt);
    param_S::brgemm_t::work_group_t g(ei.get_local_linear_id());
    matS.init(0);
    matmul_op(g, matS, args);
    matS.reg *= hs_rsqrt_scale;
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::calculate_P_ij(
        utils::work_item<3> &ei, param_S::mat_out_t &matS,
        param_P::rowmax_t &m_tilde_vec, param_P::rowsum_t &l_tilde_vec) {
    using mat_t = param_S::mat_out_t;
    using mat_dtype = mat_t::dtype;
    using mat_tile_shape = param_S::mat_tile_shape;
    using mat_worker_scope_t = mat_tile_shape::work_group_t;
    mat_worker_scope_t g(ei.get_local_linear_id());
    int32_t j_s = g.get_id() % mat_tile_shape::wg_size_x;
    int32_t i_s = g.get_id() / mat_tile_shape::wg_size_x;
    uint32_t nbarrier_id = i_s;
    uint32_t slm_base_addr = param_P::slm_base_addr
            + param_P::reduce_slm_size / param_P::reduce_list_ylength * i_s;
    // 1. m_tilde_ij = rowmax(S_ij)
    // where:
    //   - shape(m_tilde_ij) = [B_r,]
    {
        using rowmax_t = param_P::rowmax_t;
        static constexpr auto reduce_op_t = gpu::xetla::reduce_op::max;
        rowmax_t m_tilde_vec_thread
                = gpu::xetla::subgroup::tile_reduce<reduce_op_t, mat_t,
                        mat_dtype, 1>(matS);
        m_tilde_vec = m_tilde_vec_thread;
        param_P::wg_reduce_max_t wg_reduce_max(j_s, nbarrier_id, slm_base_addr);
        m_tilde_vec = wg_reduce_max(m_tilde_vec_thread);
    }
    // 2. P_ij = exp(S_ij - m_tilde_ij)
    // where:
    //   - shape(P_ij) = [B_r, B_c]
    {
        using broadcast_op_t = gpu::xetla::subgroup::tile_minus;
        gpu::xetla::subgroup::tile_broadcast_op<broadcast_op_t, mat_t>(
                matS, m_tilde_vec);
        matS.reg = gpu::xetla::xetla_exp<mat_dtype>(matS.reg);
    }
    auto &matP = matS;
    // 3. l_tilde_ij = rowsum(P_ij)
    // where:
    //   - shape(l_tilde_ij) = [B_r,]
    {
        using rowsum_t = param_P::rowsum_t;
        static constexpr auto reduce_op_t = gpu::xetla::reduce_op::sum;
        rowsum_t l_tilde_vec_thread
                = gpu::xetla::subgroup::tile_reduce<reduce_op_t, mat_t,
                        mat_dtype, 1>(matP);
        param_P::wg_reduce_sum_t wg_reduce_sum(j_s, nbarrier_id, slm_base_addr);
        l_tilde_vec = wg_reduce_sum(l_tilde_vec_thread);
    }
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::load_ml(
        utils::work_item<3> &ei, dtype_m *ptr_m, param_P::rowmax_t &m_vec,
        param_P::rowsum_t &l_vec) {
    // m_i, l_i is created and persists in register
    // storing the result to ptr_m is skipped to reduce memory read/write
}

__XETLA_API KERNEL_FUNC void
FLASH_ATTENTION_FWD_H128::program::calculate_new_ml(utils::work_item<3> &ei,
        param_P::rowmax_t &m_new_vec, param_P::rowmax_t &m_tilde_vec,
        param_P::rowmax_t &m_vec, param_P::rowsum_t &l_new_vec,
        param_P::rowsum_t &l_tilde_vec, param_P::rowsum_t &l_vec) {
    // 1. m_new_i = max(m_i, m_tilde_ij)
    // 2. l_new_i = sum(exp(m_i - m_new_i) * l_i, exp(m_tilde_ij - m_new_i) * l_tilde_ij)
    // where:
    //   - shape(m) = [B_r,]
    //   - shape(l) = [B_r,]
    static constexpr uint32_t block_elems = std::min({32, param_P::vec_length});
    using calculate_new_ml_op_t = utils::calculate_new_ml_op_t<block_elems>;
    calculate_new_ml_op_t max_sum_op;
    max_sum_op(m_new_vec, m_tilde_vec, m_vec, l_new_vec, l_tilde_vec, l_vec);
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::calculate_PV(
        utils::work_item<3> &ei, uint32_t batch_size, uint32_t seq_len,
        uint32_t T_r, uint32_t T_c, param_O::mat_out_t &matO_new,
        param_P::mat_out_t &matP, dtype_m *ptr_m, dtype_v *ptr_v,
        dtype_b *ptr_b, param_P::rowmax_t &m_new_vec,
        param_P::rowmax_t &m_tilde_vec) {
    // TODO: use slm for store
    // 1. store P_ij to shared local memory (slm)
    // where:
    //   - shape(P_ij) = [B_r, B_c]
    // store_P_ij(ei, matP);
    // 2. O_tilde_i = P_ij @ V_j
    // where:
    //   - shape(V_j) = [B_c, H]
    //   - mem_space(P_ij) = slm
    //   - mem_space(V_j)  = global
    {
        int batch_idx = ei.get_group(0);
        int i_w = ei.get_group(1);
        int j_w = ei.get_group(2);
        int start_v_m = j_w * param_O::m_w;
        int start_v_n = 0;
        uint32_t slm_base_addr = param_P::slm_base_addr;
        uint32_t boundary_v_m = start_v_m + param_O::m_w > seq_len
                ? seq_len
                : start_v_m + param_O::m_w;
        uint32_t boundary_v_n
                = start_v_n + param_O::n_w > H ? H : start_v_n + param_O::n_w;
        // param_O::mem_desc_input_p md_p({slm_base_addr},
        //         {param_S::n_w, param_S::m_w, param_S::n_w}, {0, 0});

        // use template buffer in global: use slm for better performance
        using mem_desc_input_tmp = gpu::xetla::mem_desc_t<dtype_b,
                gpu::xetla::mem_layout::row_major,
                gpu::xetla::mem_space::global>;
        constexpr uint32_t buf_size = param_S::m_w * param_S::n_w;
        mem_desc_input_tmp md_p(
                {ptr_b + (batch_idx * T_r * T_c + i_w * T_c + j_w) * buf_size},
                {param_S::n_w, param_S::m_w, param_S::n_w}, {0, 0});
        {
            using mat_tile_shape = param_P::mat_tile_shape;
            using mat_worker_scope_t = mat_tile_shape::work_group_t;
            mat_worker_scope_t g(ei.get_local_linear_id());
            using epilogue_t = gpu::xetla::group::epilogue_t<
                    gpu::xetla::group::epilogue_policy_default<
                            gpu::xetla::gpu_arch::Xe>,
                    mat_tile_shape, mem_desc_input_tmp>;
            epilogue_t epilogue;
            epilogue(g, matP, md_p);

            // TODO: hangs here
            // param_P::store_nbarrier_t nbarrier;
            // nbarrier.init_nbarrier(0, gpu::xetla::nbarrier_role::producer_consumer);
            gpu::xetla::xetla_fence<gpu::xetla::memory_kind::untyped_global>();
            // nbarrier.arrive_wait();
        }
        param_O::mem_desc_input_v md_v(
                {ptr_v + batch_idx * utils::matrix_size(seq_len)},
                {boundary_v_n, boundary_v_m, H}, {start_v_n, start_v_m});
        param_O::brgemm_t matmul_op;
        param_O::brgemm_t::arguments_t args(md_p, md_v, param_O::split_k_cnt);
        param_O::brgemm_t::work_group_t g(ei.get_local_linear_id());
        matO_new.init(0);
        matmul_op(g, matO_new, args);
    }
    // 3. O_tilde_i_weighted = diag(exp(m_tilde_ij - m_new_i)) @ O_tilde_i
    {
        using row_exp_mul_op_t = utils::row_exp_mul_op_t;
        row_exp_mul_op_t row_exp_mul_op;
        row_exp_mul_op(m_new_vec, m_tilde_vec, matO_new);
    }
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::load_O(
        utils::work_item<3> &ei, param_O::mat_out_t &matO,
        param_P::rowmax_t &m_new_vec, param_P::rowmax_t &m_vec) {
    // O_i_weighted = diag(exp(m_i - m_new_i)) @ O_i
    using row_exp_mul_op_t = utils::row_exp_mul_op_t;
    row_exp_mul_op_t row_exp_mul_op;
    row_exp_mul_op(m_new_vec, m_vec, matO);
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::update_O(
        utils::work_item<3> &ei, param_O::mat_out_t &matO,
        param_O::mat_out_t &matO_new, param_P::rowsum_t &l_new_vec,
        param_P::rowsum_t &l_vec) {
    // 1. O_i_weighted_unnorm = diag(l_i) @ O_i_weighted
    {
        using row_mul_op_t = utils::row_mul_op_t;
        row_mul_op_t row_mul_op;
        row_mul_op(l_vec, matO);
    }
    // 2. O_i_unnorm = O_i_weighted_unnorm + O_tilde_i_weighted
    { matO.reg = matO.reg + matO_new.reg; }
    // 3. O_i = inv(diag(l_new_i)) @ O_i_unnorm
    {
        using row_div_op_t = utils::row_div_op_t;
        row_div_op_t row_div_op;
        row_div_op(l_new_vec, matO);
    }
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::store_O(
        utils::work_item<3> &ei, param_O::mat_out_t &matO) {
    // O_i is created and persists in register
    // storing the result to shared local memory is skipped to reduce memory read/write
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::store_ml(
        utils::work_item<3> &ei, param_P::rowmax_t &m_vec,
        param_P::rowmax_t &m_new_vec, param_P::rowsum_t &l_vec,
        param_P::rowsum_t &l_new_vec) {
    // m_i, l_i are stored in register instead of shared local memory
    // 1. m_i = m_new_i
    m_vec = m_new_vec;
    // 2. l_i = l_new_i
    l_vec = l_new_vec;
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::store_P_ij(
        utils::work_item<3> &ei, param_P::mat_out_t &matP) {
    // store P_ij from register to shared local memory
    using mat_tile_shape = param_P::mat_tile_shape;
    using mat_worker_scope_t = mat_tile_shape::work_group_t;
    mat_worker_scope_t g(ei.get_local_linear_id());

    uint32_t slm_base_addr = param_P::slm_base_addr;

    param_P::mem_desc_output_p md_p({slm_base_addr},
            {param_S::n_w, param_S::m_w, param_S::n_w}, {0, 0});
    param_P::epilogue_t epilogue;
    epilogue(g, matP, md_p);

    param_P::store_nbarrier_t nbarrier;
    nbarrier.init_nbarrier(0, gpu::xetla::nbarrier_role::producer_consumer);
    gpu::xetla::xetla_fence<gpu::xetla::memory_kind::shared_local>();
    nbarrier.arrive_wait();
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::store_global_O(
        utils::work_item<3> &ei, uint32_t batch_dim, uint32_t head_dim,
        uint32_t seq_len, dtype_o *ptr_o, param_O::mat_out_t &matO) {
    const uint32_t B = batch_dim;
    const uint32_t N = head_dim;
    const uint32_t F = seq_len;
    static constexpr uint32_t tile_size_y = param_O::m_s;
    static constexpr uint32_t tile_size_x = param_O::n_s;

    gpu::xetla::xetla_tdescriptor transpose_store_tdecs;
    // Define a temprary vector as output buffer
    gpu::xetla::xetla_vector<dtype_o, tile_size_x> out_reg;

    param_O::brgemm_t::work_group_t g(ei.get_local_linear_id());
    // Calculate new coordination of each element
    uint32_t b = ei.get_group(2) / N; // dim0 coord
    uint32_t n = ei.get_group(2) % N; // dim1 coord
    uint32_t start_m = ei.get_group(0) * param_O::m_w;
    uint32_t start_n = ei.get_group(1) * param_O::n_w;
    uint32_t f
            = start_m + param_O::brgemm_t::get_matC_offset_y(g); // dim2 coord
    uint32_t h
            = start_n + param_O::brgemm_t::get_matC_offset_x(g); // dim3 coord

    // transpose 8 * 16 tile and store to global

    for (uint32_t j = 0; j < tile_size_y; ++j, ++f) {
        // TODO: vairiable align with FLASH_ATTENTION_FWD_H128
        uint32_t dst_offset = b * N * F * H + n * F * H + f * H + h;
        out_reg = matO.reg.xetla_select<tile_size_x, 1>(j * tile_size_x);
        gpu::xetla::xetla_fill_tdesc<dtype_o, tile_size_x /*, 1, 1*/>(
                transpose_store_tdecs.xetla_format<uint32_t>(),
                ptr_o + dst_offset, H, 1, H, h, 0);
        gpu::xetla::xetla_tstore_global<dtype_o, tile_size_x,
                gpu::xetla::cache_hint::write_back,
                gpu::xetla::cache_hint::write_back>(
                transpose_store_tdecs, out_reg);
    }
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::store_global_ml(
        utils::work_item<3> &ei, dtype_m *ptr_m,
        param_P::factor_tile_t &m_store, param_P::factor_payload_t &m_payload,
        param_P::factor_tile_t &l_store, param_P::factor_payload_t &l_payload) {
    using mat_tile_shape = param_S::mat_tile_shape;
    using mat_worker_scope_t = mat_tile_shape::work_group_t;
    mat_worker_scope_t g(ei.get_local_linear_id());
    int32_t j_s = g.get_id() % mat_tile_shape::wg_size_x;
    int32_t i_s = g.get_id() / mat_tile_shape::wg_size_x;
    // store m, l once, threads with coordinate j_s != 0 can skip
    if (j_s != 0) { return; }
    // store m_i, l_i to global memory
    gpu::xetla::subgroup::tile_store<gpu::xetla::cache_hint::uncached>(
            m_store, m_payload);
    gpu::xetla::subgroup::tile_store<gpu::xetla::cache_hint::uncached>(
            l_store, l_payload);
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::program::setup_ml_store(
        utils::work_item<3> &ei, uint32_t batch_size, uint32_t seq_len,
        uint32_t T_r, uint32_t T_c, uint32_t B_r, uint32_t B_c, dtype_m *ptr_m,
        param_P::factor_payload_t &m_payload,
        param_P::factor_payload_t &l_payload) {
    // M[i] = m_i
    // L[i] = l_i
    // where:
    //   - shape(M) = [T_r, B_r]
    //   - mem_space(M) = global
    //   - mem_location(M) = {base_address: ptr_m, offset: batch_idx * factor_size
    //   - shape(L) = [T_r, B_r]
    //   - mem_space(L) = global
    //   - mem_location(L) = {base_address: ptr_m, offset: batch_idx * factor_size + T_r * B_r}

    int batch_idx = ei.get_group(0);
    int i_w = ei.get_group(1);

    using mat_t = param_S::mat_out_t;
    using mat_dtype = mat_t::dtype;
    using mat_tile_shape = param_S::mat_tile_shape;
    using mat_worker_scope_t = mat_tile_shape::work_group_t;
    mat_worker_scope_t g(ei.get_local_linear_id());
    int32_t i_s = g.get_id() / mat_tile_shape::wg_size_x;
    static constexpr uint32_t vec_length = param_P::vec_length;

    uint32_t factor_size = utils::factor_size(seq_len);
    param_P::mem_desc_output_m md_m(
            {ptr_m + batch_idx * factor_size}, {B_r, T_r, B_r}, {0, i_w});

    m_payload.init(md_m);
    l_payload.init(md_m);
    m_payload.update_tdesc(i_s * vec_length);
    l_payload.update_tdesc(i_s * vec_length + T_r * B_r);
}

__XETLA_API KERNEL_FUNC void FLASH_ATTENTION_FWD_H128::run(
        gpu::xetla::xetla_exec_item<3> &ei) const {
    auto uei = utils::make_work_item(ei);

    program::initialization(uei);

    typename param_O::mat_out_t matO;
    decltype(matO) matO_new;

    typename param_P::factor_tile_t m_store;
    typename param_P::factor_tile_t l_store;
    typename param_P::factor_payload_t m_payload;
    typename param_P::factor_payload_t l_payload;

    program::setup_ml_store(uei, batch_size, seq_len, T_r, T_c, B_r, B_c, ptr_m,
            m_payload, l_payload);

    typename param_P::rowmax_t &m_vec = m_store.reg;
    typename param_P::rowsum_t &l_vec = l_store.reg;
    typename param_P::rowmax_t m_tilde_vec;
    typename param_P::rowsum_t l_tilde_vec;
    typename param_P::rowmax_t m_new_vec;
    typename param_P::rowsum_t l_new_vec;

    debug::test_reg_op(matO);

    for (int j = 0; j < T_c; ++j) {
        uei.update_group_coord(2, j);
        {
            typename param_S::mat_out_t matS;
            program::calculate_S_ij(
                    uei, seq_len, matS, ptr_q, ptr_k, hs_rsqrt_scale);
            debug::store_S_ij(uei, seq_len, matS, ptr_o);
            continue;
            auto &matP = matS;
            program::calculate_P_ij(uei, matS, m_tilde_vec, l_tilde_vec);
            // debug::store_S_ij(uei, seq_len, matS, ptr_o);

            program::load_ml(uei, ptr_m, m_vec, l_vec);

            program::calculate_new_ml(uei, m_new_vec, m_tilde_vec, m_vec,
                    l_new_vec, l_tilde_vec, l_vec);

            program::calculate_PV(uei, batch_size, seq_len, T_r, T_c, matO_new,
                    matP, ptr_m, ptr_v, ptr_b, m_new_vec, m_tilde_vec);
        }

        program::load_O(uei, matO, m_new_vec, m_vec);

        program::update_O(uei, matO, matO_new, l_new_vec, l_vec);
        program::store_O(uei, matO);

        program::store_ml(uei, m_vec, m_new_vec, l_vec, l_new_vec);
    }
    return;
    program::store_global_O(uei, batch_dim, head_dim, seq_len, ptr_o, matO);
    program::store_global_ml(
            uei, ptr_m, m_store, m_payload, l_store, l_payload);
}
