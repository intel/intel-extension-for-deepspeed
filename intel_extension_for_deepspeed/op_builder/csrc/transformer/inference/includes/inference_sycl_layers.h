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
void launch_attn_softmax_v2(T *vals, T *mask, T *alibi, float layer_scale,
                            bool triangular, bool recompute,
                            bool local_attention, int window_size,
                            int batch_size, int heads, int num_seq,
                            int sequence_length, int head_offset,
                            int mask_stride, int mp_size, sycl::queue stream);
