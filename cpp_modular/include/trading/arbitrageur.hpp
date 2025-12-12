// Arbitrage decision logic (floating-point only)
// Currently twocrypto_fx specific - will refactor when adding other pool types
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>

#include <boost/math/tools/roots.hpp>

#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/costs.hpp"
#include "trading/decision.hpp"

// Enable with TRACE_ARB=1 environment variable
#ifndef TRACE_ARB_ENABLED
#define TRACE_ARB_ENABLED 0
#endif

namespace arb {
namespace trading {

namespace fx = arb::pools::twocrypto_fx;

// Root finder wrapper
template <typename F>
inline bool toms748_root(
    F&& f,
    double lo, double hi,
    double Flo, double Fhi,
    double& out_root,
    unsigned max_iters = 100
) {
    if (!(hi > lo) || !(Flo * Fhi < 0.0)) return false;
    auto tol = boost::math::tools::eps_tolerance<double>(std::numeric_limits<double>::digits10 - 3);
    boost::uintmax_t it = max_iters;
    auto r = boost::math::tools::toms748_solve(std::forward<F>(f), lo, hi, Flo, Fhi, tol, it);
    out_root = (r.first + r.second) / 2.0;
    return true;
}

template <typename T, typename PoolT>
Decision<T> decide_trade(
    const PoolT& pool,
    T cex_price,
    const Costs<T>& costs,
    T notional_cap_coin0,
    T min_swap_frac,
    T max_swap_frac
) {
    static_assert(std::is_floating_point_v<T>, "decide_trade is floating-only");

    Decision<T> d{};

    if (!(cex_price > T(0))) return d;

    const T fee_cex = costs.arb_fee_bps / T(10000);

    const auto xp_now = fx::pool_xp_current(pool);
    const T p_now = fx::MathOps<T>::get_p(xp_now, pool.D, {pool.A, pool.gamma}) * pool.cached_price_scale;
    const T fee_pool = fx::dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);

    const T one_minus_fee = std::max(T(1) - fee_pool, T(1e-12));
    const T p_pool_bid = one_minus_fee * p_now;
    const T p_pool_ask = p_now / one_minus_fee;

    const T p_cex_bid = (T(1) - fee_cex) * cex_price;
    const T p_cex_ask = (T(1) + fee_cex) * cex_price;

    // Check for arb edge
    const T edge_01 = p_cex_bid - p_pool_ask;  // buy pool, sell CEX
    const T edge_10 = p_pool_bid - p_cex_ask;  // buy CEX, sell pool

#if TRACE_ARB_ENABLED
    std::cerr << std::setprecision(15)
              << "[TRACE_ARB] cex=" << cex_price
              << " p_now=" << p_now
              << " fee_pool=" << fee_pool
              << " p_pool_bid=" << p_pool_bid
              << " p_pool_ask=" << p_pool_ask
              << " p_cex_bid=" << p_cex_bid
              << " p_cex_ask=" << p_cex_ask
              << " edge_01=" << edge_01
              << " edge_10=" << edge_10
              << "\n";
#endif

    int sel_i = -1, sel_j = -1;
    if (edge_01 <= T(0) && edge_10 <= T(0)) return d;
    if (edge_01 >= edge_10) { sel_i = 0; sel_j = 1; } else { sel_i = 1; sel_j = 0; }

#if TRACE_ARB_ENABLED
    std::cerr << "[TRACE_ARB] Passed edge check: sel_i=" << sel_i << " sel_j=" << sel_j << "\n";
#endif

    const T avail = pool.balances[static_cast<size_t>(sel_i)];
    if (!(avail > T(0))) return d;

    // Sizing bounds
    T dx_lo = std::max(T(1e-18), avail * std::max(T(1e-12), min_swap_frac));
    T dx_hi = avail * max_swap_frac;

    if (std::isfinite(static_cast<double>(notional_cap_coin0)) && notional_cap_coin0 > T(0)) {
        dx_hi = (sel_i == 0)
            ? std::min(dx_hi, notional_cap_coin0)
            : std::min(dx_hi, notional_cap_coin0 / pool.cached_price_scale);
    }
    if (!(dx_hi > dx_lo)) return d;

    // Residual: post-trade pool price vs CEX price
    auto residual = [&](double dx_d) -> double {
        T dx = static_cast<T>(dx_d);
        auto pr = fx::post_trade_price_and_fee(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx);
        T p_new = pr.first;
        T fee_new = pr.second;
        T p_bid = (T(1) - fee_new) * p_new;
        T p_ask = p_new / (T(1) - fee_new);
        return (sel_i == 0)
            ? static_cast<double>(p_ask - p_cex_bid)
            : static_cast<double>(p_bid - p_cex_ask);
    };

    double F_lo = residual(static_cast<double>(dx_lo));
    double F_hi = residual(static_cast<double>(dx_hi));

    T dx_star = dx_hi;
    if (F_lo * F_hi < 0.0) {
        double root;
        if (toms748_root(residual, static_cast<double>(dx_lo), static_cast<double>(dx_hi), F_lo, F_hi, root)) {
            dx_star = std::max(static_cast<T>(root), dx_lo);
        }
#if TRACE_ARB_ENABLED
        std::cerr << "[TRACE_ARB] Root found: dx_star=" << dx_star << " F_lo=" << F_lo << " F_hi=" << F_hi << "\n";
#endif
    } else {
        // No crossing — check if edge exists at lo
#if TRACE_ARB_ENABLED
        std::cerr << "[TRACE_ARB] No crossing: F_lo=" << F_lo << " F_hi=" << F_hi << " sel_i=" << sel_i << "\n";
#endif
        if ((sel_i == 0 && !(F_lo < 0.0)) || (sel_i == 1 && !(F_lo > 0.0))) return d;
        dx_star = dx_hi;
    }

    // Simulate and compute profit
    auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx_star);
    T dy_after_fee = sim.first;

    T profit;
    if (sel_i == 0) {
        profit = dy_after_fee * cex_price * (T(1) - fee_cex) - dx_star - costs.gas_coin0;
    } else {
        profit = dy_after_fee - dx_star * cex_price * (T(1) + fee_cex) - costs.gas_coin0;
    }

    if (!(profit > T(0))) {
#if TRACE_ARB_ENABLED
        std::cerr << "[TRACE_ARB] Profit check failed: profit=" << profit << " dx_star=" << dx_star << " dy=" << dy_after_fee << "\n";
#endif
        return d;
    }

#if TRACE_ARB_ENABLED
    std::cerr << "[TRACE_ARB] TRADE: i=" << sel_i << " j=" << sel_j << " dx=" << dx_star << " profit=" << profit << "\n";
#endif

    d.do_trade = true;
    d.i = sel_i;
    d.j = sel_j;
    d.dx = dx_star;
    d.profit = profit;
    d.fee_tokens = sim.second;
    d.notional_coin0 = (sel_i == 0) ? dx_star : dx_star * cex_price;

    return d;
}

} // namespace trading
} // namespace arb
