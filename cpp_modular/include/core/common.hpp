// Common utilities: differs_rel, io_mutex, etc.
#pragma once

#include <cmath>
#include <mutex>
#include <algorithm>

namespace arb {

// Global mutex for synchronized console output
inline std::mutex io_mu;

// Relative difference check for floating-point comparison
// Returns true if values differ by more than rel * max(1, max(|a|, |b|))
template <typename T>
inline bool differs_rel(T a, T b, T rel = T(1e-12), T abs_eps = T(0)) {
    const T da    = std::abs(a - b);
    const T scale = std::max<T>(T(1), std::max(std::abs(a), std::abs(b)));
    return da > std::max(abs_eps, rel * scale);
}

} // namespace arb
