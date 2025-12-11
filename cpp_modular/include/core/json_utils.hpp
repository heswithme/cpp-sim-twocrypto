// JSON parsing utilities for pool configuration
#pragma once

#include <boost/json.hpp>
#include <string>
#include <sstream>
#include <iomanip>

namespace arb {

// Convert a value to a string representation scaled by 1e18
template <typename T>
static std::string to_str_1e18(T v) {
    long double scaled = static_cast<long double>(v) * 1e18L;
    if (scaled < 0) scaled = 0;
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(0);
    oss << scaled;
    return oss.str();
}

// Parse a JSON value as a real number, scaling down from 1e18 representation
template <typename T>
static inline T parse_scaled_1e18(const boost::json::value& v) {
    if (v.is_string()) return static_cast<T>(std::strtold(v.as_string().c_str(), nullptr) / 1e18L);
    if (v.is_double()) return static_cast<T>(v.as_double() / 1e18);
    if (v.is_int64())  return static_cast<T>(static_cast<long double>(v.as_int64()) / 1e18L);
    if (v.is_uint64()) return static_cast<T>(static_cast<long double>(v.as_uint64()) / 1e18L);
    return T(0);
}

// Parse a JSON value as a fee (scaled from 1e10)
template <typename T>
static inline T parse_fee_1e10(const boost::json::value& v) {
    if (v.is_string()) return static_cast<T>(std::strtold(v.as_string().c_str(), nullptr) / 1e10L);
    if (v.is_double()) return static_cast<T>(v.as_double() / 1e10);
    if (v.is_int64())  return static_cast<T>(static_cast<long double>(v.as_int64()) / 1e10L);
    if (v.is_uint64()) return static_cast<T>(static_cast<long double>(v.as_uint64()) / 1e10L);
    return T(0);
}

// Parse a JSON value as a plain real number (no scaling)
template <typename T>
static inline T parse_plain_real(const boost::json::value& v) {
    if (v.is_string()) return static_cast<T>(std::strtold(v.as_string().c_str(), nullptr));
    if (v.is_double()) return static_cast<T>(v.as_double());
    if (v.is_int64())  return static_cast<T>(v.as_int64());
    if (v.is_uint64()) return static_cast<T>(v.as_uint64());
    return T(0);
}

} // namespace arb
