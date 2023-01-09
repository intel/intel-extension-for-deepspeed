#pragma once
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <ext/oneapi/experimental/bfloat16.hpp>

using namespace cl::sycl;
using bf16 = sycl::ext::oneapi::experimental::bfloat16;

int onednn_matmul_ex(sycl::queue* handle,
                     bool trans_src,
                     bool trans_wgt,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const bf16* src_ptr,
                     const bf16* wgt_ptr,
                     bf16* dst_ptr);

int onednn_batchgemm(sycl::queue* handle,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const bf16* src_ptr,
                     const bf16* wgt_ptr,
                     bf16* dst_ptr,
                     bool trans_src,
                     bool trans_wgt,
                     int batch);
