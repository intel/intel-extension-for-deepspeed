#pragma once
#include "flash_attn_fwd_outer_loop.hpp"

template <typename fwd_kernel_traits>
bool flash_attn_fwd(
    sycl::queue& queue,
    void* output, // [Bs, Hn, Sl, Hs]
    // void* out_buffer, // [Bs, Hn, Sl, Hs]
    void* softmax_L, // [Bs*Hn, 1, Sl]: row_max + log(row_sum)
    void* softmax_m,
    const uint32_t Bs,
    const uint32_t Hn,
    const uint32_t Sl,
    const uint32_t Hs,
    const float hs_rsqrt_scale,
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const void* dropout_mask_ptr,
    const float dropout_prob,
    const float dropout_scale,
    const uint64_t dropout_seed,
    const bool is_casual,
    const bool store_softmax) {
  using namespace cl::sycl;
  using namespace gpu::xetla;
  using namespace gpu::xetla::group;
  using namespace gpu::xetla::kernel;
  using namespace gpu::xetla::subgroup;

  // according to paper defination
  static constexpr uint32_t block_d = fwd_kernel_traits::head_size;
  static constexpr uint32_t block_br = fwd_kernel_traits::blocksize_r;
  static constexpr uint32_t block_bc = fwd_kernel_traits::blocksize_c;

  // dataytype
  using input_T = fwd_kernel_traits::input_T;
  using output_T = fwd_kernel_traits::out_T;
  using acc_T = fwd_kernel_traits::acc_T;

  constexpr int wg_tile_m = 128;
  constexpr int wg_tile_n = 128;
  constexpr int sg_tile_m = 16;
  constexpr int sg_tile_n = 32;
  constexpr int sg_tile_k = 16;

  using block_brxd_tile_t = gemm_block_tile_t<block_br, block_d, block_bc, wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k>;
  using block_bcxd_tile_t = gemm_block_tile_t<block_bc, block_d, block_br, wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k>;
  using block_brxbc_tile_t = gemm_block_tile_t<block_br, block_bc, block_d, wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k>;

  using fmha_t = flash_attention_fwd<
      input_T,
      output_T,
      acc_T,
      block_brxd_tile_t,
      block_bcxd_tile_t,
      block_brxbc_tile_t>;
  using fmha_args_t = typename fmha_t::arguments_t;

  int qkv_batch_offset = Sl * block_d;
  int softmax_batch_offset = Sl;

  constexpr size_t wg_range_m = block_br / wg_tile_m;
  constexpr size_t wg_range_n = block_bc / wg_tile_n;
  constexpr size_t sg_range_m = wg_tile_m / sg_tile_m;
  constexpr size_t sg_range_n = wg_tile_n / sg_tile_n;

  cl::sycl::range<3> group_range{Bs * Hn, wg_range_m, wg_range_n};
  cl::sycl::range<3> local_range{1, sg_range_m, sg_range_n};
  cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

  try {
    auto cgf = ([&](handler& cgh) {
      cgh.parallel_for<fwd_kernel_traits>(
          nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
            xetla_exec_item<3> ei(item);
            static constexpr uint32_t slm_size = wg_tile_m * wg_tile_n * sizeof(input_T);
            xetla_local_init<slm_size>();

            uint32_t batch_id = ei.get_group(0);
            fmha_t fmha_fwd;
            fmha_args_t args(
                (input_T*)q_ptr + batch_id * qkv_batch_offset,
                (input_T*)k_ptr + batch_id * qkv_batch_offset,
                (input_T*)v_ptr + batch_id * qkv_batch_offset,
                (output_T*)output + batch_id * qkv_batch_offset,
                // (input_T*)out_buffer + batch_id * qkv_batch_offset,
                (acc_T*)softmax_L +
                    (batch_id) * softmax_batch_offset, // softmax_l/L
                (acc_T*)softmax_m +
                    (batch_id) * softmax_batch_offset, // softmax_m
                Sl,
                Sl,
                hs_rsqrt_scale,
                dropout_prob,
                dropout_scale,
                dropout_seed,
                0, // slm_base
                (input_T*)dropout_mask_ptr + batch_id * Sl * Sl);
            args.head_size = block_d;

            fmha_fwd(ei, args);
          });
    });
    DPCPP_Q_SUBMIT(queue, cgf);
  } catch (cl::sycl::exception const& e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }
  return true;
}