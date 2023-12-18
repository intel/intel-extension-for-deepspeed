// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <assert.h>
#include <stdio.h>
#include "gemm.hpp"
#ifdef BF16_AVAILABLE
#endif

int onemkl_gemm_ex(sycl::queue * handle,
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
 try {
    const int ldb = (b_stride == -1) ? ((transb == oneapi::mkl::transpose::nontrans) ? k : n)
                                     : b_stride;
    const int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
    const int ldc = m;

    gemm_impl<float, float, float, float>(
        *handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}

template <typename T>
int onemkl_gemm_ex(sycl::queue * handle,
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
 try {
    const int ldb = (b_stride == -1) ? ((transb == oneapi::mkl::transpose::nontrans) ? k : n)
                                     : b_stride;
    const int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
    const int ldc = m;

    if (std::is_same<T, sycl::half>::value){
        float alpha_value = get_value(reinterpret_cast<const float *>(alpha), *handle);
        float beta_value = get_value(reinterpret_cast<const float *>(beta), *handle);
    sycl::half alpha_half(alpha_value);
    sycl::half beta_half(beta_value);
    gemm_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
        *handle, transa, transb, m, n, k, &alpha_half, A, lda, B, ldb, &beta_half, C, ldc);
    }else{
    gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
              oneapi::mkl::bfloat16, float>(
        *handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}

int onemkl_strided_batched_gemm(sycl::queue * handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                oneapi::mkl::transpose transa,
                                oneapi::mkl::transpose transb,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo)
 try {
       const int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
       const int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
       const int ldc = m;
       gemm_batch_impl<float, float, float, float>(*handle, transa, transb, 
           m, n, k, alpha, A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch);
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}

template <typename T>
int onemkl_strided_batched_gemm(sycl::queue * handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const T* A,
                                const T* B,
                                T* C,
                                oneapi::mkl::transpose transa,
                                oneapi::mkl::transpose transb,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo)
 try {
       int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
       int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
       int ldc = m;
       if (std::is_same<T, sycl::half>::value){
    	float alpha_value = get_value(reinterpret_cast<const float *>(alpha), *handle);
    	float beta_value = get_value(reinterpret_cast<const float *>(beta), *handle);
    	sycl::half alpha_half(alpha_value);
    	sycl::half beta_half(beta_value);
    	gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
            *handle, transa, transb, m, n, k, &alpha_half, A, lda, stride_A, B, ldb, stride_B,
            &beta_half, C, ldc, stride_C, batch);
    }else{
      gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                      oneapi::mkl::bfloat16, float>(
      *handle, transa, transb, m, n, k, alpha, A, lda, stride_A, B, ldb, stride_B,
            beta, C, ldc, stride_C, batch);
    }
    return 0;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
            << std::endl;
  std::exit(1);
}
