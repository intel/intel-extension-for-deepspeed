#include <ATen/ATen.h>
#include "inference_onemkl_wrappers.hpp"
#include "inference_sycl_layers.h"

struct hash_pair {
  static size_t hash_combine( size_t lhs, size_t rhs ) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }

  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2> &pair) const {
    return hash_combine(std::hash<T1>()(pair.first), std::hash<T2>()(pair.second));
  }
};

// DNNL engine and stream should be permnantly existing and binding to sycl queue
static std::pair<dnnl::engine, dnnl::stream> get_dnnl_engine_stream(sycl::queue queue) {
  static std::unordered_map<sycl::queue, dnnl::stream> dnnl_streams;
  auto it_stream = dnnl_streams.find(queue);

  static std::unordered_map<std::pair<sycl::device, sycl::context>, dnnl::engine, hash_pair> dnnl_engines;
  auto context = std::make_pair(queue.get_device(), queue.get_context());
  // if hit, we know both engine and queue are preserved
  if (it_stream != dnnl_streams.end()) {
    return std::make_pair(dnnl_engines[context], it_stream->second);
  }

  auto it = dnnl_engines.find(context);

  dnnl::engine engine;
  if (it != dnnl_engines.end()) {
    engine = it->second;
  } else {
    engine = dnnl::sycl_interop::make_engine(context.first, context.second);
    dnnl_engines.emplace(std::make_pair(context, engine));
  }

  dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, queue);
  dnnl_streams.emplace(std::make_pair(queue, stream));

  return std::make_pair(engine, stream);
}

template <typename T>
int onemkl_matmul_ex(sycl::queue handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float alpha,
                   const float beta,
                   const T* A,
                   const T* B,
                   T* C)
{
    // TODO: ldb and ldc is right? 
    try {
      int lda = (transa == oneapi::mkl::transpose::nontrans) ? k : m;
      int ldb = (transb == oneapi::mkl::transpose::nontrans) ? n : k;
      int ldc = n;
      oneapi::mkl::blas::row_major::gemm(
          handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
      std::exit(1);
    }

    return 0;
}

// TODO: if stride_A needed
template <typename T>
int onemkl_strided_batched_gemm(sycl::queue handle,
                                oneapi::mkl::transpose transa,
                                oneapi::mkl::transpose transb,
                                int m,
                                int n,
                                int k,
                                const float alpha,
                                const float beta,
                                const T* A,
                                const T* B,
                                T* C,
                                int batch)
{
    try {
      int lda = (transa == oneapi::mkl::transpose::nontrans) ? k : m;
      int ldb = (transb == oneapi::mkl::transpose::nontrans) ? n : k;
      int ldc = n;

      int stride_A = m * k;
      int stride_B = k * n;
      int stride_C = m * n;

      oneapi::mkl::blas::row_major::gemm_batch(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    A,
                                    lda,
                                    stride_A,
                                    B,
                                    ldb,
                                    stride_B,
                                    beta,
                                    C,
                                    ldc,
                                    stride_C,
                                    batch);
    } catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                << " (batch, m, n, k)" << batch << " " << m << " " << n << " " << k << std::endl;
      std::exit(1);
    }
}

template int onemkl_matmul_ex(sycl::queue handle,
                            oneapi::mkl::transpose transa,
                            oneapi::mkl::transpose transb,
                            int m,
                            int n,
                            int k,
                            const float alpha,
                            const float beta,
                            const sycl::half* A,
                            const sycl::half* B,
                            sycl::half* C);

template int onemkl_matmul_ex(sycl::queue handle,
                            oneapi::mkl::transpose transa,
                            oneapi::mkl::transpose transb,
                            int m,
                            int n,
                            int k,
                            const float alpha,
                            const float beta,
                            const float* A,
                            const float* B,
                            float* C);

template int onemkl_strided_batched_gemm(sycl::queue handle,
                                         oneapi::mkl::transpose op_A,
                                         oneapi::mkl::transpose op_B,
                                         int m,
                                         int n,
                                         int k,
                                         const float alpha,
                                         const float beta,
                                         const sycl::half* A,
                                         const sycl::half* B,
                                         sycl::half* C,
                                         int batch);

template int onemkl_strided_batched_gemm(sycl::queue handle,
                                         oneapi::mkl::transpose op_A,
                                         oneapi::mkl::transpose op_B,
                                         int m,
                                         int n,
                                         int k,
                                         const float alpha,
                                         const float beta,
                                         const float* A,
                                         const float* B,
                                         float* C,
                                         int batch);
