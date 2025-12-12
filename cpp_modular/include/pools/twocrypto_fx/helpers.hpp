// Helper utilities for twocrypto_fx pool
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include "pools/twocrypto_fx/stableswap_math.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"

namespace arb {
namespace pools {
namespace twocrypto_fx {

template <typename T>
using PoolT = TwoCryptoPool<T>;

// -----------------------------------------------------------------------------
// Balance/vector helpers
// -----------------------------------------------------------------------------
template <typename T>
inline std::array<T, 2> pool_xp_from(
    const PoolT<T>& pool,
    const std::array<T, 2>& balances,
    const T& price_scale
) {
    return {
        balances[0] * pool.precisions[0],
        balances[1] * pool.precisions[1] * price_scale / PoolTraits<T>::PRECISION()
    };
}

template <typename T>
inline std::array<T, 2> pool_xp_current(const PoolT<T>& pool) {
    return pool_xp_from(pool, pool.balances, pool.cached_price_scale);
}

template <typename T>
inline T xp_to_tokens_j(const PoolT<T>& pool, size_t j, T amount_xp, const T& price_scale) {
    T v = amount_xp;
    if (j == 1) {
        v = v * PoolTraits<T>::PRECISION() / price_scale;
    }
    return v / pool.precisions[j];
}

// -----------------------------------------------------------------------------
// Fee + price helpers
// -----------------------------------------------------------------------------
template <typename T>
inline T dyn_fee(
    const std::array<T, 2>& xp,
    const T& mid_fee,
    const T& out_fee,
    const T& fee_gamma
) {
    const T Bsum = xp[0] + xp[1];
    if (!(Bsum > T(0))) {
        return mid_fee;
    }

    // Matches the pool's internal _fee formula; for floating types PRECISION() == 1.
    T B = PoolTraits<T>::PRECISION() * PoolT<T>::N_COINS * PoolT<T>::N_COINS * xp[0] / Bsum * xp[1] / Bsum;
    B = fee_gamma * B /
        (fee_gamma * B / PoolTraits<T>::PRECISION() + PoolTraits<T>::PRECISION() - B);

    return (
        mid_fee * B + out_fee * (PoolTraits<T>::PRECISION() - B)
    ) / PoolTraits<T>::PRECISION();
}

template <typename T>
inline std::pair<T, T> post_trade_price_and_fee(
    const PoolT<T>& pool,
    size_t i,
    size_t j,
    T dx
) {
    const T ps = pool.cached_price_scale;

    auto balances_local = pool.balances;
    balances_local[i] += dx;
    auto xp = pool_xp_from(pool, balances_local, ps);

    auto y_out = MathOps<T>::get_y(pool.A, pool.gamma, xp, pool.D, j);
    T dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    const T fee_pool = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    const T D_new = MathOps<T>::newton_D(pool.A, pool.gamma, xp, T(0));
    const T p_new = MathOps<T>::get_p(xp, D_new, {pool.A, pool.gamma}) * ps / PoolTraits<T>::PRECISION();

    return {p_new, fee_pool};
}

// -----------------------------------------------------------------------------
// Derived signals
// -----------------------------------------------------------------------------
template <typename T>
inline bool instantaneous_dr(const PoolT<T>& pool, T& out_d, T& out_r) {
    const auto xp_now = pool_xp_current(pool);
    const T p_now = MathOps<T>::get_p(xp_now, pool.D, {pool.A, pool.gamma}) * pool.cached_price_scale;
    const T x_b   = pool.balances[1];
    if (!(p_now > T(0)) || !(x_b > T(0))) return false;

    T eps = static_cast<T>(1e-6);
    for (int attempt = 0; attempt < 3; ++attempt) {
        const T dx_tokens = std::max(eps * x_b, std::numeric_limits<T>::min());
        auto pr = post_trade_price_and_fee(pool, /*i=*/1, /*j=*/0, dx_tokens);
        const T p_new = pr.first;
        const T p_avg = (p_now + p_new) / static_cast<T>(2);
        const T dp_abs = std::abs(p_new - p_now);
        if (p_avg > T(0) && dp_abs > T(0)) {
            const T rel = dp_abs / p_avg;
            const T d = (dx_tokens / x_b) / rel;
            if (d > T(0) && std::isfinite(static_cast<double>(d))) {
                out_d = d;
                out_r = static_cast<T>(1) / d;
                return true;
            }
        }
        eps *= static_cast<T>(10);
    }
    return false;
}

template <typename T>
inline T balance_indicator(const PoolT<T>& pool) {
    const T ps = pool.cached_price_scale;
    const auto xp = pool_xp_from(pool, pool.balances, ps);
    const T denom = xp[0] + xp[1];
    if (!(denom > T(0))) {
        return T(0);
    }
    return static_cast<T>(4) * xp[0] * xp[1] / (denom * denom);
}

template <typename T>
inline T true_growth(const PoolT<T>& pool, T price_ref = static_cast<T>(-1)) {
    const T price_in = (price_ref > T(0)) ? price_ref : pool.cached_price_scale;
    const auto xp = pool_xp_from(pool, pool.balances, price_in);
    const T product = xp[0] * xp[1];
    return (product > T(0)) ? std::sqrt(product) : T(0);
}

// Standalone version without pool object (for end-state metrics)
// For floating types, precisions are 1.0
template <typename T>
inline T true_growth_from_balances(const std::array<T, 2>& balances, const T& price_scale) {
    const T xp0 = balances[0];
    const T xp1 = balances[1] * price_scale;
    const T product = xp0 * xp1;
    return (product > T(0)) ? std::sqrt(product) : T(0);
}

// -----------------------------------------------------------------------------
// Simulation helper (no state change)
// -----------------------------------------------------------------------------
template <typename T>
inline std::pair<T, T> simulate_exchange_once(
    const PoolT<T>& pool,
    size_t i,
    size_t j,
    T dx
) {
    const T ps = pool.cached_price_scale;

    auto balances_local = pool.balances;
    balances_local[i] += dx;
    auto xp = pool_xp_from(pool, balances_local, ps);

    auto y_out = MathOps<T>::get_y(pool.A, pool.gamma, xp, pool.D, j);
    T dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    T dy_tokens  = xp_to_tokens_j(pool, j, dy_xp, ps);
    T fee_pool   = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    T fee_tokens = fee_pool * dy_tokens / PoolTraits<T>::FEE_PRECISION();
    return {dy_tokens - fee_tokens, fee_tokens};
}

// -----------------------------------------------------------------------------
// Convenience conversion
// -----------------------------------------------------------------------------
template <typename T>
inline T coin0_equiv(const T& amt0, const T& amt1, const T& price_scale) {
    return amt0 + amt1 * price_scale / PoolTraits<T>::PRECISION();
}

} // namespace twocrypto_fx
} // namespace pools
} // namespace arb
