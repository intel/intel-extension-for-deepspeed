// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Centralized header file for preprocessor macros and constants
used throughout the codebase.
*/

#pragma once

#include <sycl/sycl.hpp>

#ifdef BF16_AVAILABLE
#endif

#define DS_HD_INLINE __forceinline__
#define DS_D_INLINE __inline__
#define __dpct_align__(n) __attribute__((aligned(n)))

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 32;

#define HALF_PRECISION_AVAILABLE = 1
// #define PTX_AVAILABLE


inline int next_pow2(const int val)
{
    int rounded_val = val - 1;
    rounded_val |= rounded_val >> 1;
    rounded_val |= rounded_val >> 2;
    rounded_val |= rounded_val >> 4;
    rounded_val |= rounded_val >> 8;
    return rounded_val + 1;
}
