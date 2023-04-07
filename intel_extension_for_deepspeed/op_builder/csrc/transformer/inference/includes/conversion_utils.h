/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include <stdint.h>
#include <sycl/half_type.hpp>

#include "compatible.h"
#include <ext/oneapi/experimental/bfloat16.hpp>

namespace conversion {

// Basic primitive for constructing conversions
// sycl cannot call recursive func
template <typename TO, typename FROM>
inline TO to(FROM val)
{
    return TO(val);
}

// Specializations

/********************* Identity Conversions *********************/
/*
Identity conversions are useful in templated functions where we might have
a fixed destination type. For example, I might have a kernel that accepts
half, bf16, and float but always want to do the core computation
at floating point:

T mem_value = input[idx];
float compute_value = conversion::to<float, T>(mem_value);

In practice, we should be able to elide the second template parameter:
float compute_val = conversion::to<float>(mem_value);

In this case, we need an implementation to handle the T = float case

NOTE: The type inferencing system appears to be unable to handle inferring the first
template parameter, even in the trivial case.
*/

// Floating point types
template <>
inline double to(double val)
{
    return val;
}
template <>
inline float to(float val)
{
    return val;
}
template <>
inline half to(half val)
{
    return val;
}
#ifdef BF16_AVAILABLE
template <>
inline bf16 to(bf16 val)
{
    return val;
}
#endif

// Integer types
template <>
inline int8_t to(int8_t val)
{
    return val;
}
template <>
inline uint8_t to(uint8_t val)
{
    return val;
}
template <>
inline int16_t to(int16_t val)
{
    return val;
}
template <>
inline uint16_t to(uint16_t val)
{
    return val;
}
template <>
inline int32_t to(int32_t val)
{
    return val;
}
template <>
inline uint32_t to(uint32_t val)
{
    return val;
}
template <>
inline int64_t to(int64_t val)
{
    return val;
}
template <>
inline uint64_t to(uint64_t val)
{
    return val;
}

// TODO: evaluate if we want bools

/*********************  To Double Conversions *********************/

// * to double variants

// Would normally like to not use C cast, but this is an important enough conversion
// to keep
template <>
inline double to(float val)
{
    return double(val);
}
// Note: there is a CVT instruction for half -> double, but there's no inline interface
// for passing a single half value
template <>
inline double to(half val)
{
    return to<double>(sycl::detail::half2Float(val));
}
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline double to(int64_t val)
{
    return __imf_ll2double_rn(val);
}
template <>
inline double to(int32_t val)
{
    return __imf_int2double_rn(val);
}
template <>
inline double to(int16_t val)
{
    return __imf_int2double_rn(val);
}
template <>
inline double to(int8_t val)
{
    return __imf_int2double_rn(val);
}
template <>
inline double to(uint64_t val)
{
    return __imf_ull2double_rn(val);
}
template <>
inline double to(uint32_t val)
{
    return __imf_uint2double_rn(val);
}
template <>
inline double to(uint16_t val)
{
    return __imf_uint2double_rn(val);
}
template <>
inline double to(uint8_t val)
{
    return __imf_uint2double_rn(val);
}
#endif

// Same applies here
#ifdef BF16_AVAILABLE
template <>
inline double to(bf16 val)
{
    return to<double>(to<float>(val));
}
#endif

/*********************  To Float Conversions *********************/
template <>
inline float to(half val)
{
    return sycl::detail::half2Float(val);
}

#ifdef __SYCL_DEVICE_ONLY__
template <>
inline float to(double val)
{
    return __imf_double2float_rn(val);
}
template <>
inline float to(int64_t val)
{
    return __imf_ll2float_rn(val);
}
template <>
inline float to(int32_t val)
{
    return __imf_int2float_rn(val);
}
template <>
inline float to(int16_t val)
{
    return __imf_int2float_rn(val);
}
template <>
inline float to(int8_t val)
{
    return __imf_int2float_rn(val);
}
template <>
inline float to(uint64_t val)
{
    return __imf_ull2float_rn(val);
}
template <>
inline float to(uint32_t val)
{
    return __imf_uint2float_rn(val);
}
template <>
inline float to(uint16_t val)
{
    return __imf_uint2float_rn(val);
}
template <>
inline float to(uint8_t val)
{
    return __imf_uint2float_rn(val);
}
#endif

#ifdef BF16_AVAILABLE
template <>
inline float to(bf16 val)
{
    return __bfloat162float(val);
}
#endif

/*********************  To Float2 Conversions *********************/
template <>
inline float2 to(half2 val)
{
    return val.convert<float>();
}

// TODO: ushort as bf16 replacement for bf16 is not compatible with sycl::vec
template <>
inline float2 to(sycl::ushort2 val)
{
    float2 tmp;
    tmp[0] = bf16::to_float(val[0]);
    tmp[1] = bf16::to_float(val[1]);
    return tmp;
}


#ifdef BF16_AVAILABLE
template <>
inline float2 to(bf162 val)
{
    float2 tmp;
    tmp[0] = bf16::to_float(val[0]);
    tmp[1] = bf16::to_float(val[1]);
    return tmp;
}
#endif

/*********************  To Half Conversions *********************/
template <>
inline half to(double val)
{
    return sycl::detail::float2Half(to<float>(val));
}
template <>
inline half to(float val)
{
    return sycl::detail::float2Half(val);
}
template <>
inline half to(int64_t val)
{
    return to<half>(to<float>(val));
}
template <>
inline half to(int32_t val)
{
    return to<half>(to<float>(val));
}
template <>
inline half to(int16_t val)
{
    return to<half>(to<float>(val));
}
template <>
inline half to(int8_t val)
{
    return to<half>(to<float>(val));
}
template <>
inline half to(uint64_t val)
{
    return to<half>(to<float>(val));
}
template <>
inline half to(uint32_t val)
{
    return to<half>(to<float>(val));
}
template <>
inline half to(uint16_t val)
{
    return to<half>(to<float>(val));
}
template <>
inline half to(uint8_t val)
{
    return to<half>(to<float>(val));
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
inline half to(bf16 val)
{
    return to<half>(to<float>(val));
}
#endif

/*********************  To Half2 Conversions *********************/
template <>
inline half2 to(float2 val)
{
    return val.convert<half, rounding_mode::rtn>();
}
template <>
inline half2 to(float val)
{
    half2 tmp;
    tmp[0] = to<half>(val);
    tmp[1] = to<half>(val);
    return tmp;
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
inline half2 to(bf162 val)
{
    return to<half2>(to<float2>(val));
}
#endif

/*********************  To BF16 Conversions *********************/
#ifdef BF16_AVAILABLE
template <>
inline bf16 to(double val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(float val)
{
    return bf16::from_float(val);
}
template <>
inline bf16 to(int64_t val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(int32_t val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(int16_t val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(int8_t val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(uint64_t val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(uint32_t val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(uint16_t val)
{
    return bf16::from_float(to<float>(val));
}
template <>
inline bf16 to(uint8_t val)
{
    return bf16::from_float(to<float>(val));
}
#endif

/*********************  To BF162 Conversions *********************/
// TODO: use ushort as vec<bf16> replacement
template <>
inline sycl::ushort2 to(float2 val)
{
    sycl::ushort2 tmp;
    tmp[0] = bf16::from_float(val[0]);
    tmp[1] = bf16::from_float(val[1]);
    return tmp;
}

#ifdef BF16_AVAILABLE
template <>
inline bf162 to(float2 val)
{
    bf162 tmp;
    tmp[0] = bf16::from_float(val[0]);
    tmp[1] = bf16::from_float(val[1]);
    return tmp;
}
template <>
inline bf162 to(float val)
{
    bf162 tmp;
    tmp[0] = to<bf16>(val);
    tmp[1] = to<bf16>(val);
    return tmp;
}
template <>
inline bf162 to(half2 val)
{
    auto tmp = to<float>(val);
    return to<bf162>(tmp);
}
#endif

/*********************  To INT64_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int64_t to(double val)
{
    return __imf_double2ll_rn(val);
}
template <>
inline int64_t to(float val)
{
    return __imf_float2ll_rn(val);
}
#endif
template <>
inline int64_t to(half val)
{
    return to<int64_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int64_t to(bf16 val)
{
    return to<int64_t>(to<float>(val));
}
#endif

/*********************  To INT32_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int32_t to(double val)
{
    return __imf_double2int_rn(val);
}
template <>
inline int32_t to(float val)
{
    return __imf_float2int_rn(val);
}
#endif
template <>
inline int32_t to(half val)
{
    return to<int32_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int32_t to(bf16 val)
{
    return to<int32_t>(to<float>(val));
}
#endif

/*********************  To INT16_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int16_t to(double val)
{
    return __imf_double2int_rn(val);
}
template <>
inline int16_t to(float val)
{
    return __imf_float2int_rn(val);
}
#endif
template <>
inline int16_t to(half val)
{
    return to<int16_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int16_t to(bf16 val)
{
    return to<int16_t>(to<float>(val));
}
#endif

/*********************  To INT8_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline int8_t to(double val)
{
    return __imf_double2int_rn(val);
}
template <>
inline int8_t to(float val)
{
    return __imf_float2int_rn(val);
}
#endif
template <>
inline int8_t to(half val)
{
    return to<int8_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline int8_t to(bf16 val)
{
    return to<int8_t>(to<float>(val));
}
#endif

/*********************  To UINT64_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint64_t to(double val)
{
    return __imf_double2ull_rn(val);
}
template <>
inline uint64_t to(float val)
{
    return __imf_float2ull_rn(val);
}
#endif
template <>
inline uint64_t to(half val)
{
    return to<uint64_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint64_t to(bf16 val)
{
    return to<uint64_t>(to<float>(val));
}
#endif

/*********************  To UINT32_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint32_t to(double val)
{
    return __imf_double2uint_rn(val);
}
template <>
inline uint32_t to(float val)
{
    return __imf_float2uint_rn(val);
}
#endif
template <>
inline uint32_t to(half val)
{
    return to<uint32_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint32_t to(bf16 val)
{
    return to<uint32_t>(to<float>(val));
}
#endif

/*********************  To UINT16_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint16_t to(double val)
{
    return __imf_double2uint_rn(val);
}
template <>
inline uint16_t to(float val)
{
    return __imf_float2uint_rn(val);
}
#endif
template <>
inline uint16_t to(half val)
{
    return to<uint16_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint16_t to(bf16 val)
{
    return to<uint16_t>(to<float>(val));
}
#endif

/*********************  To UINT8_T Conversions *********************/
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline uint8_t to(double val)
{
    return __imf_double2uint_rn(val);
}
template <>
inline uint8_t to(float val)
{
    return __imf_float2uint_rn(val);
}
#endif
template <>
inline uint8_t to(half val)
{
    return to<uint8_t>(to<float>(val));
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
inline uint8_t to(bf16 val)
{
    return to<uint8_t>(to<float>(val));
}
#endif

}  // namespace conversion
