// Core numeric type traits for templated math
#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <limits>
#include <type_traits>

namespace arb {

namespace stableswap {
using uint256 = boost::multiprecision::uint256_t;
using int256 = boost::multiprecision::int256_t;
}

// NumTraits: type-specific constants and operations
template <typename T>
struct NumTraits {
    static constexpr bool is_integer = false;
    static constexpr T zero() { return T(0); }
    static constexpr T one() { return T(1); }
    static T from_double(double v) { return static_cast<T>(v); }
    static double to_double(T v) { return static_cast<double>(v); }
};

// Specialization for uint256
template <>
struct NumTraits<stableswap::uint256> {
    static constexpr bool is_integer = true;
    static stableswap::uint256 zero() { return stableswap::uint256(0); }
    static stableswap::uint256 one() { return stableswap::uint256(1); }
    static stableswap::uint256 from_double(double v) {
        return stableswap::uint256(static_cast<long long>(v));
    }
    static double to_double(stableswap::uint256 v) {
        return v.convert_to<double>();
    }
};

// Specialization for double
template <>
struct NumTraits<double> {
    static constexpr bool is_integer = false;
    static constexpr double zero() { return 0.0; }
    static constexpr double one() { return 1.0; }
    static double from_double(double v) { return v; }
    static double to_double(double v) { return v; }
};

// Specialization for float
template <>
struct NumTraits<float> {
    static constexpr bool is_integer = false;
    static constexpr float zero() { return 0.0f; }
    static constexpr float one() { return 1.0f; }
    static float from_double(double v) { return static_cast<float>(v); }
    static double to_double(float v) { return static_cast<double>(v); }
};

// Specialization for long double
template <>
struct NumTraits<long double> {
    static constexpr bool is_integer = false;
    static constexpr long double zero() { return 0.0L; }
    static constexpr long double one() { return 1.0L; }
    static long double from_double(double v) { return static_cast<long double>(v); }
    static double to_double(long double v) { return static_cast<double>(v); }
};

} // namespace arb
