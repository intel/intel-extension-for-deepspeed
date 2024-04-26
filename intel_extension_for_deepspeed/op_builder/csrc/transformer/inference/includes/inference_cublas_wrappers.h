// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <dpct/blas_utils.hpp>

#ifdef BF16_AVAILABLE
#endif
#ifndef __HIP_PLATFORM_AMD__
#endif
#include <stdio.h>
#include <dpct/lib_common_utils.hpp>

#ifdef __HIP_PLATFORM_AMD__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   rocblas_gemm_algo algo,
                   int b_stride = -1)
#else
int cublas_gemm_ex(dpct::queue_ptr handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   int algo,
                   int b_stride = -1)
#endif
 try {
    const int ldb = (b_stride == -1) ? ((transb == oneapi::mkl::transpose::nontrans) ? k : n)
                                     : b_stride;
#ifdef __HIP_PLATFORM_AMD__
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            (const void*)A,
                                            rocblas_datatype_f32_r,
                                            (transa == rocblas_operation_none) ? m : k,
                                            (const void*)B,
                                            rocblas_datatype_f32_r,
                                            ldb,
                                            (const void*)beta,
                                            C,
                                            rocblas_datatype_f32_r,
                                            m,
                                            C,
                                            rocblas_datatype_f32_r,
                                            m,
                                            rocblas_datatype_f32_r,
                                            algo,
                                            0,
                                            0);
#else
    int status = DPCT_CHECK_ERROR(dpct::gemm(*handle,
                                             transa,
                                             transb,
                                             m,
                                             n,
                                             k,
                                             (const void*)alpha,
                                             (const void*)A,
                                             dpct::library_data_t::real_float,
                                             (transa == oneapi::mkl::transpose::nontrans) ? m : k,
                                             (const void*)B,
                                             dpct::library_data_t::real_float,
                                             ldb,
                                             (const void*)beta,
                                             C,
                                             dpct::library_data_t::real_float,
                                             m,
                                             dpct::library_data_t::real_float));
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != 0) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}

template <typename T>
#ifdef __HIP_PLATFORM_AMD__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const T* A,
                   const T* B,
                   T* C,
                   rocblas_gemm_algo algo,
                   int b_stride = -1)
#else
int cublas_gemm_ex(dpct::queue_ptr handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const T* A,
                   const T* B,
                   T* C,
                   int algo,
                   int b_stride = -1)
#endif
 try {
    const int ldb = (b_stride == -1) ? ((transb == oneapi::mkl::transpose::nontrans) ? k : n)
                                     : b_stride;
#ifdef __HIP_PLATFORM_AMD__
    constexpr auto rocblas_dtype_16 = std::is_same<T, sycl::half>::value ? rocblas_datatype_f16_r
                                                                     : rocblas_datatype_bf16_r;
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            (const void*)A,
                                            rocblas_dtype_16,
                                            (transa == rocblas_operation_none) ? m : k,
                                            (const void*)B,
                                            rocblas_dtype_16,
                                            ldb,
                                            (const void*)beta,
                                            (void*)C,
                                            rocblas_dtype_16,
                                            m,
                                            (void*)C,
                                            rocblas_dtype_16,
                                            m,
                                            rocblas_datatype_f32_r,
                                            algo,
                                            0,
                                            0);
#else
    constexpr auto cublas_dtype_16 = std::is_same<T, sycl::half>::value
                                         ? dpct::library_data_t::real_half
                                         : dpct::library_data_t::real_bfloat16;
    int status = DPCT_CHECK_ERROR(dpct::gemm(*handle,
                                             transa,
                                             transb,
                                             m,
                                             n,
                                             k,
                                             (const void*)alpha,
                                             (const void*)A,
                                             cublas_dtype_16,
                                             (transa == oneapi::mkl::transpose::nontrans) ? m : k,
                                             (const void*)B,
                                             cublas_dtype_16,
                                             ldb,
                                             (const void*)beta,
                                             (void*)C,
                                             cublas_dtype_16,
                                             m,
                                             dpct::library_data_t::real_float));
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != 0) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}

#ifdef __HIP_PLATFORM_AMD__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo)
#else
int cublas_strided_batched_gemm(dpct::queue_ptr handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                oneapi::mkl::transpose op_A,
                                oneapi::mkl::transpose op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo)
#endif
 try {
#ifdef __HIP_PLATFORM_AMD__
    rocblas_status status =
        rocblas_gemm_strided_batched_ex(handle,
                                        op_A,
                                        op_B,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        rocblas_datatype_f32_r,
                                        (op_A == rocblas_operation_none) ? m : k,
                                        stride_A,
                                        B,
                                        rocblas_datatype_f32_r,
                                        (op_B == rocblas_operation_none) ? k : n,
                                        stride_B,
                                        beta,
                                        C,
                                        rocblas_datatype_f32_r,
                                        m,
                                        stride_C,
                                        C,
                                        rocblas_datatype_f32_r,
                                        m,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        algo,
                                        0,
                                        0);
#else
    int status =
        DPCT_CHECK_ERROR(dpct::gemm_batch(*handle,
                                          op_A,
                                          op_B,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          A,
                                          dpct::library_data_t::real_float,
                                          (op_A == oneapi::mkl::transpose::nontrans) ? m : k,
                                          stride_A,
                                          B,
                                          dpct::library_data_t::real_float,
                                          (op_B == oneapi::mkl::transpose::nontrans) ? k : n,
                                          stride_B,
                                          beta,
                                          C,
                                          dpct::library_data_t::real_float,
                                          m,
                                          stride_C,
                                          batch,
                                          dpct::library_data_t::real_float));
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != 0) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n",
                batch,
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}

template <typename T>
#ifdef __HIP_PLATFORM_AMD__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const T* A,
                                const T* B,
                                T* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo)
#else
int cublas_strided_batched_gemm(dpct::queue_ptr handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const T* A,
                                const T* B,
                                T* C,
                                oneapi::mkl::transpose op_A,
                                oneapi::mkl::transpose op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo)
#endif
 try {
#ifdef __HIP_PLATFORM_AMD__
    constexpr auto rocblas_dtype_16 = std::is_same<T, sycl::half>::value ? rocblas_datatype_f16_r
                                                                     : rocblas_datatype_bf16_r;
    rocblas_status status =
        rocblas_gemm_strided_batched_ex(handle,
                                        op_A,
                                        op_B,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        rocblas_dtype_16,
                                        (op_A == rocblas_operation_none) ? m : k,
                                        stride_A,
                                        B,
                                        rocblas_dtype_16,
                                        (op_B == rocblas_operation_none) ? k : n,
                                        stride_B,
                                        beta,
                                        C,
                                        rocblas_dtype_16,
                                        m,
                                        stride_C,
                                        C,
                                        rocblas_dtype_16,
                                        m,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        algo,
                                        0,
                                        0);
#else
    constexpr auto cublas_dtype_16 = std::is_same<T, sycl::half>::value
                                         ? dpct::library_data_t::real_half
                                         : dpct::library_data_t::real_bfloat16;
    int status =
        DPCT_CHECK_ERROR(dpct::gemm_batch(*handle,
                                          op_A,
                                          op_B,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          A,
                                          cublas_dtype_16,
                                          (op_A == oneapi::mkl::transpose::nontrans) ? m : k,
                                          stride_A,
                                          B,
                                          cublas_dtype_16,
                                          (op_B == oneapi::mkl::transpose::nontrans) ? k : n,
                                          stride_B,
                                          beta,
                                          C,
                                          cublas_dtype_16,
                                          m,
                                          stride_C,
                                          batch,
                                          dpct::library_data_t::real_float));
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != 0) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }

    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}
