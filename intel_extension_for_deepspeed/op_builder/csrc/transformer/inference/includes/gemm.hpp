#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "memory.hpp"

template <typename T> struct DataType { using T2 = T; };
template <typename T> struct DataType<sycl::vec<T, 2>> {
  using T2 = std::complex<T>;
};

template <typename T> inline auto get_memory(const void *x) {
  T *new_x = reinterpret_cast<T *>(const_cast<void *>(x));
  return new_x;
}

template <typename T>
inline typename DataType<T>::T2 get_value(const T *s, sycl::queue &q) {
  using Ty = typename DataType<T>::T2;
  Ty s_h;
  if (dpct::detail::get_pointer_attribute(q, s) == dpct::detail::pointer_access_attribute::device_only)
    dpct::detail::dpct_memcpy(q, (void *)&s_h, (void *)s, sizeof(T), dpct::device_to_host)
        .wait();
  else
    s_h = *reinterpret_cast<const Ty *>(s);
  return s_h;
}

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                         oneapi::mkl::transpose b_trans, int m, int n, int k,
                         const void *alpha, const void *a, int lda, const void *b,
                         int ldb, const void *beta, void *c, int ldc) {
  Ts alpha_value = get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = get_value(reinterpret_cast<const Ts *>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm(
      q, a_trans, b_trans, m, n, k, alpha_value, data_a, lda,
      data_b, ldb, beta_value, data_c, ldc);
}

template <class Ta, class Tb, class Tc, class Ts>
inline void
gemm_batch_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                    oneapi::mkl::transpose b_trans, int m, int n,
                    int k, const void *alpha, const void *a, int lda,
                    long long int stride_a, const void *b, int ldb,
                    long long int stride_b, const void *beta, void *c,
                    int ldc, long long int stride_c, int batch_size) {
  Ts alpha_value = get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = get_value(reinterpret_cast<const Ts *>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm_batch(
      q, a_trans, b_trans, m, n, k, alpha_value, data_a, lda,
      stride_a, data_b, ldb, stride_b, beta_value,
      data_c, ldc, stride_c, batch_size);
}
