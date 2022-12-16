/* Taken from NVIDIA/apex commit 855808f3fc268e9715d613f3c2e56469d8c986d8 */
#include <ATen/ATen.h>
#include <CL/sycl.hpp>

// Forward/backward compatiblity hack around
// https://github.com/pytorch/pytorch/commit/3aeb78079bcd68282fe9117088e138b77318e288
// pending more future-proof guidance from upstream.
// struct TypeShim
// {
//   const at::Type& payload;
//   TypeShim(const at::Type& type) : payload(type) {}
//   // Enable trivial conversion to a const at::Type& for pre-3aeb78
//   operator const at::Type&(){ return payload; };
//   // Enable dispatch switch statements to take *this directly for post-3aeb78
//   //operator at::ScalarType(){ return payload.; };
// };

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)                          \
    switch (TYPE) {                                                              \
        case at::ScalarType::Float: {                                            \
            using scalar_t_##LEVEL = float;                                      \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Half: {                                             \
            using scalar_t_##LEVEL = at::Half;                                   \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        default: AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }

#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)                   \
    switch (TYPE) {                                                              \
        case at::ScalarType::Double: {                                           \
            using scalar_t_##LEVEL = double;                                     \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Float: {                                            \
            using scalar_t_##LEVEL = float;                                      \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Half: {                                             \
            using scalar_t_##LEVEL = at::Half;                                   \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        default: AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }

#define DISPATCH_DOUBLE_AND_FLOAT(TYPE, LEVEL, NAME, ...)                        \
    switch (TYPE) {                                                              \
        case at::ScalarType::Double: {                                           \
            using scalar_t_##LEVEL = double;                                     \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Float: {                                            \
            using scalar_t_##LEVEL = float;                                      \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        default: AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }

template <typename T>
T reduce_block_into_lanes(T* x,
                          T val,
                          sycl::nd_item<3> item_ct1,
                          int lanes = 1,
                          bool share_result = false)  // lanes is intended to be <= 32.
{
    int tid =
        item_ct1.get_local_id(2) + item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2);
    int blockSize = item_ct1.get_local_range(2) *
                    item_ct1.get_local_range(1);  // blockSize is intended to be a multiple of 32.

    if (blockSize >= 64) {
        x[tid] = val;
        item_ct1.barrier();
    }

#pragma unroll
    for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
        if (tid < i) x[tid] = x[tid] + x[tid + i];
        item_ct1.barrier();
    }

    T final;

    if (tid < 32) {
        if (blockSize >= 64)
            final = x[tid] + x[tid + 32];
        else
            final = val;
            // __SYNCWARP();

#pragma unroll
        for (int i = 16; i >= lanes; i >>= 1)
            final = final + __shfl_down_sync(0xffffffff, final, i);
    }

    if (share_result) {
        if (tid < lanes) x[tid] = final;  // EpilogueOp
        // Make sure the smem result is visible to all warps.
        item_ct1.barrier();
    }

    return final;
}
