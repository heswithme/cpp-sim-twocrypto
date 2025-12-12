// Core numeric type traits for templated code
// Supports both floating types (double/float/long double) and multiprecision integers (uint256/int256)
#pragma once

#include <type_traits>
#include <boost/multiprecision/cpp_int.hpp>

namespace arb {

using uint256_t = boost::multiprecision::uint256_t;
using int256_t  = boost::multiprecision::int256_t;

// NumTraits: type-specific constants and operations.
// Primary template (fallback for unspecialized types)
template <typename T, typename = void>
struct NumTraits {
    static constexpr bool is_integer = false;
    static constexpr T zero() { return T(0); }
    static constexpr T one() { return T(1); }

    static T from_double(double v) { return static_cast<T>(v); }
    static double to_double(T v) { return static_cast<double>(v); }
};

// -----------------------------------------------------------------------------
// Floating-point specializations
// -----------------------------------------------------------------------------

// double
template <>
struct NumTraits<double> {
    static constexpr bool is_integer = false;
    static constexpr double zero() { return 0.0; }
    static constexpr double one() { return 1.0; }
    static double from_double(double v) { return v; }
    static double to_double(double v) { return v; }
};

// float
template <>
struct NumTraits<float> {
    static constexpr bool is_integer = false;
    static constexpr float zero() { return 0.0f; }
    static constexpr float one() { return 1.0f; }
    static float from_double(double v) { return static_cast<float>(v); }
    static double to_double(float v) { return static_cast<double>(v); }
};

// long double
template <>
struct NumTraits<long double> {
    static constexpr bool is_integer = false;
    static constexpr long double zero() { return 0.0L; }
    static constexpr long double one() { return 1.0L; }
    static long double from_double(double v) { return static_cast<long double>(v); }
    static double to_double(long double v) { return static_cast<double>(v); }
};

// -----------------------------------------------------------------------------
// Multiprecision integer specializations
// -----------------------------------------------------------------------------

// uint256_t
template <>
struct NumTraits<uint256_t> {
    static constexpr bool is_integer = true;
    static uint256_t zero() { return uint256_t(0); }
    static uint256_t one() { return uint256_t(1); }

    static uint256_t from_double(double v) {
        return static_cast<uint256_t>(v);
    }
    static double to_double(uint256_t v) {
        return v.convert_to<double>();
    }
};

// int256_t
template <>
struct NumTraits<int256_t> {
    static constexpr bool is_integer = true;
    static int256_t zero() { return int256_t(0); }
    static int256_t one() { return int256_t(1); }

    static int256_t from_double(double v) {
        return static_cast<int256_t>(v);
    }
    static double to_double(int256_t v) {
        return v.convert_to<double>();
    }
};

} // namespace arb
