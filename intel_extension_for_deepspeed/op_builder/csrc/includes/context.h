// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/* #include <ATen/cuda/CUDAContext.h> */
#include <cassert>
#include <iostream>
#include <vector>
#include <dpct/blas_utils.hpp>

#include <dpct/rng_utils.hpp>

#include "gemm_test.h"

#include <ipex.h>

#ifndef SYCL_CUDA_STREAM
#define SYCL_CUDA_STREAM
namespace at {
  namespace cuda {
    inline dpct::queue_ptr getCurrentCUDAStream() {
      auto device_type = c10::DeviceType::XPU;
      c10::impl::VirtualGuardImpl impl(device_type);
      c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
      auto& queue = xpu::get_queue_from_stream(c10_stream);
      return &queue;
    }

    inline dpct::queue_ptr getStreamFromPool(bool) {
      // not implemented
      return nullptr;
    }
    
    inline dpct::queue_ptr getStreamFromPool() {
      // not implemented
      return nullptr;
    }
  }
}
#endif

#define WARP_SIZE 32

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                                          \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N)
{
    return (std::max)(
        (std::min)((N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS, DS_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
}

class TrainingContext {
public:
    TrainingContext() try : _workspace(nullptr), _seed(42), _curr_offset(0) {
        _gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
        _gen->set_seed(123);
        int stat = DPCT_CHECK_ERROR(_cublasHandle = &dpct::get_in_order_queue());
        if (stat != 0) {
            // It would be nice to use cublasGetStatusName and
            // cublasGetStatusString, but they were only added in CUDA 11.4.2.
            auto message = std::string("Failed to create cublas handle: cublasStatus_t was ") +
                           std::to_string(stat);
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                << std::endl;
      std::exit(1);
    }

    virtual ~TrainingContext()
    {
        _cublasHandle = nullptr;
        sycl::free(_workspace, dpct::get_in_order_queue());
    }

    static TrainingContext& Instance()
    {
        static TrainingContext _ctx;
        return _ctx;
    }

    void SetWorkSpace(void* workspace)
    {
        if (!workspace) { throw std::runtime_error("Workspace is null."); }
        _workspace = workspace;
    }

    void* GetWorkSpace() { return _workspace; }

    dpct::rng::host_rng_ptr& GetRandGenerator() { return _gen; }

    dpct::queue_ptr GetCurrentStream()
    {
        // get current pytorch stream.
        dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
        return stream;
    }

    dpct::queue_ptr GetNewStream() { return at::cuda::getStreamFromPool(); }

    dpct::queue_ptr GetCublasHandle() { return _cublasHandle; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

    void TestGemmFP16(bool test_gemm, int batch_size, int seq_len, int head_num, int size_per_head)
    {
        // avoid rerun.
        if (_gemm_algos.size() > 0) return;

        if (test_gemm) {
            dpct::queue_ptr handle = GetCublasHandle();

            std::unique_ptr<GemmTest<sycl::half>> test_qkv_fw(
                new GemmTest<sycl::half>(batch_size * seq_len,      // M
                                     head_num * size_per_head,  // N
                                     head_num * size_per_head,  // K
                                     oneapi::mkl::transpose::trans,
                                     oneapi::mkl::transpose::nontrans,
                                     handle));

            std::unique_ptr<GemmTest<sycl::half>> test_inter(
                new GemmTest<sycl::half>(batch_size * seq_len,          // M
                                     4 * head_num * size_per_head,  // N
                                     head_num * size_per_head,      // K
                                     oneapi::mkl::transpose::trans,
                                     oneapi::mkl::transpose::nontrans,
                                     handle));

            std::unique_ptr<GemmTest<sycl::half>> test_output(
                new GemmTest<sycl::half>(batch_size * seq_len,          // M
                                     head_num * size_per_head,      // N
                                     4 * head_num * size_per_head,  // K
                                     oneapi::mkl::transpose::trans,
                                     oneapi::mkl::transpose::nontrans,
                                     handle));

            std::unique_ptr<StridedGemmTest<sycl::half>> test_attn_scores(
                new StridedGemmTest<sycl::half>(batch_size * head_num,  // batch
                                            seq_len,                // M
                                            seq_len,                // N
                                            size_per_head,          // K
                                            oneapi::mkl::transpose::trans,
                                            oneapi::mkl::transpose::nontrans,
                                            handle));

            std::unique_ptr<StridedGemmTest<sycl::half>> test_attn_context(
                new StridedGemmTest<sycl::half>(batch_size * head_num,  // batch
                                            size_per_head,          // M
                                            seq_len,                // N
                                            seq_len,                // K
                                            oneapi::mkl::transpose::nontrans,
                                            oneapi::mkl::transpose::nontrans,
                                            handle));

            _gemm_algos.push_back(test_qkv_fw->TestAlgo(100));
            _gemm_algos.push_back(test_inter->TestAlgo(100));
            _gemm_algos.push_back(test_output->TestAlgo(100));
            _gemm_algos.push_back(test_attn_scores->TestAlgo(100));
            _gemm_algos.push_back(test_attn_context->TestAlgo(100));
        } else {
            // Use default algo.
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
        }
    }

    const std::vector<std::array<int, 3>>& GetGemmAlgos() const { return _gemm_algos; }

private:
    dpct::rng::host_rng_ptr _gen;
    dpct::queue_ptr _cublasHandle;
    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;
    std::vector<std::array<int, 3>> _gemm_algos;
};
