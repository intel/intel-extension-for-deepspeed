#pragma once
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
using namespace sycl;
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
using namespace cl::sycl;
#else
#error "Unsupported compiler"
#endif
#include <ext/oneapi/experimental/bfloat16.hpp>

template <typename T>
int onednn_matmul_ex(sycl::queue handle,
                     bool trans_src,
                     bool trans_wgt,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr);

template <typename T>
int onednn_batchgemm(sycl::queue handle,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr,
                     bool trans_src,
                     bool trans_wgt,
                     int batch);
