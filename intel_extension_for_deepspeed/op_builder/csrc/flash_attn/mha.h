#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <stddef.h>
#include <ipex.h>
#include <torch/extension.h>
#include "xetla.hpp"

#define DPCPP_Q_CGF(h) [&](sycl::handler & h)

#define DPCPP_Q_SUBMIT(q, cgf, ...)                                          \
  {                                                                          \
    auto e = (q).submit((cgf), ##__VA_ARGS__);                               \
    (q).throw_asynchronous();                                                \
    xpu::profiler_record("dpcpp_kernel", e);                                 \
  }

namespace gpu::xetla {

enum class XetlaType {
  fp16,
  bf16,
};

void fmha_forward_kernel(
    XetlaType xeType,
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* dropout,
    void* out,
    void* log_sumexp,
    float alpha,
    float dropout_prob,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t attn_mask_padded_block_size,
    bool is_causal,
    bool is_training,
    bool is_dropout,
    uint64_t seed_t,
    uint64_t offset_t);

void fmha_backward_kernel(
    XetlaType xeType,
    sycl::queue& q,
    void* grad_out,
    void* query,
    void* key,
    void* value,
    void* out,
    void* log_sumexp,
    void* workspace,
    float alpha,
    float dropout_prob,
    void* grad_query,
    void* grad_key,
    void* grad_value,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t attn_mask_padding,
    bool is_causal,
    bool is_dropout,
    uint64_t seed_t,
    uint64_t offset_t);

} // namespace gpu::xetla
