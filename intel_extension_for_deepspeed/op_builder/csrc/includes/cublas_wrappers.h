// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <dpct/blas_utils.hpp>

#ifndef __HIP_PLATFORM_AMD__
#endif
#ifdef __HIP_PLATFORM_AMD__
#include <rocblas/rocblas.h>
#endif
#include <stdio.h>

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
#ifdef __HIP_PLATFORM_AMD__
                   rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                   int algo = -1);
#endif

int cublas_gemm_ex(dpct::queue_ptr handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const sycl::half* A,
                   const sycl::half* B,
                   sycl::half* C,
#ifdef __HIP_PLATFORM_AMD__
                   rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                   int algo = 99);
#endif

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
#ifdef __HIP_PLATFORM_AMD__
                                rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                                int algo = -1);
#endif

int cublas_strided_batched_gemm(dpct::queue_ptr handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const sycl::half* A,
                                const sycl::half* B,
                                sycl::half* C,
                                oneapi::mkl::transpose op_A,
                                oneapi::mkl::transpose op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
#ifdef __HIP_PLATFORM_AMD__
                                rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                                int algo = 99);
#endif
