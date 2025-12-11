// Arbitrage decision logic (templated on numeric type)
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include <boost/math/tools/roots.hpp>

#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/costs.hpp"
#include "trading/decision.hpp"

namespace arb {
namespace trading {

namespace fx = arb::pools::twocrypto_fx;

namespace detail {

// Root finder wrapper on double scalar
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

// Traits to adapt numeric types to a common double sizing scalar
template <typename T, bool IsFloat = std::is_floating_point_v<T>>
struct SizingTraits;

// Floating path: keep dx/profit in double for sizing, cast back to T
template <typename T>
struct SizingTraits<T, true> {
    using Scalar = double;
    static Scalar to_scalar(const T& v, double scale) { (void)scale; return static_cast<double>(v); }
    static T from_scalar_dx(Scalar v, double scale) { (void)scale; return static_cast<T>(v); }
    static T from_scalar_profit(Scalar v, double scale) { (void)scale; return static_cast<T>(v); }
    static Scalar infinity() { return std::numeric_limits<Scalar>::infinity(); }
};

// Integer path: downscale by precision to double, then rescale back
template <typename T>
struct SizingTraits<T, false> {
    using Scalar = double;
    static Scalar to_scalar(const T& v, double scale) { return static_cast<double>(v) / scale; }
    static T from_scalar_dx(Scalar v, double scale) {
        if (v <= 0.0) return T(0);
        return static_cast<T>(v * scale);
    }
    static T from_scalar_profit(Scalar v, double scale) {
        if (v <= 0.0) return T(0);
        return static_cast<T>(v * scale);
    }
    static Scalar infinity() { return std::numeric_limits<Scalar>::infinity(); }
};

} // namespace detail

template <typename T, typename PoolT>
Decision<T> decide_trade(
    const PoolT& pool,
    T cex_price,
    const Costs<T>& costs,
    T notional_cap_coin0,
    T min_swap_frac,
    T max_swap_frac
) {
    Decision<T> d{};

    using Traits = detail::SizingTraits<T>;
    using Scalar = typename Traits::Scalar;

    // Common scales for conversion (PRECISION / FEE_PRECISION)
    const double PREC_D  = static_cast<double>(fx::PoolTraits<T>::PRECISION());
    const double FPREC_D = static_cast<double>(fx::PoolTraits<T>::FEE_PRECISION());

    const Scalar cex_price_s = Traits::to_scalar(cex_price, PREC_D);
    if (!(cex_price_s > 0)) return d;

    const Scalar fee_cex_s = static_cast<Scalar>(Traits::to_scalar(costs.arb_fee_bps, 1.0) / 1e4);

    using Ops = fx::MathOps<T>;

    const auto xp_now = fx::pool_xp_current(pool);
    const Scalar p_now = static_cast<Scalar>(Ops::get_p(xp_now, pool.D, {pool.A, pool.gamma})) / PREC_D *
                         static_cast<Scalar>(pool.cached_price_scale) / PREC_D;

    const Scalar fee_out0 = static_cast<Scalar>(fx::dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma)) / FPREC_D;

    const Scalar one_minus_fee0 = std::max<Scalar>(static_cast<Scalar>(1) - fee_out0, static_cast<Scalar>(1e-12));
    const Scalar p_pool_bid0    = one_minus_fee0 * p_now;
    const Scalar p_pool_ask0    = p_now / one_minus_fee0;

    const Scalar p_cex_bid = (static_cast<Scalar>(1) - fee_cex_s) * cex_price_s;
    const Scalar p_cex_ask = (static_cast<Scalar>(1) + fee_cex_s) * cex_price_s;

    const Scalar edge_01 = p_cex_bid - p_pool_ask0;
    const Scalar edge_10 = p_pool_bid0 - p_cex_ask;

    int sel_i = -1, sel_j = -1;
    if (edge_01 <= 0 && edge_10 <= 0) return d;
    if (edge_01 >= edge_10) { sel_i = 0; sel_j = 1; } else { sel_i = 1; sel_j = 0; }

    const Scalar avail = Traits::to_scalar(pool.balances[static_cast<size_t>(sel_i)], PREC_D);
    if (!(avail > 0)) return d;

    const Scalar min_swap_frac_s = Traits::to_scalar(min_swap_frac, PREC_D);
    const Scalar max_swap_frac_s = Traits::to_scalar(max_swap_frac, PREC_D);

    Scalar dx_lo = std::max<Scalar>(static_cast<Scalar>(1e-18), avail * std::max<Scalar>(static_cast<Scalar>(1e-12), min_swap_frac_s));
    Scalar dx_hi = avail * max_swap_frac_s;

    const Scalar notional_cap_s = Traits::to_scalar(notional_cap_coin0, PREC_D);
    if (std::isfinite(static_cast<double>(notional_cap_s)) && notional_cap_s > 0) {
        dx_hi = (sel_i == 0)
            ? std::min(dx_hi, notional_cap_s)
            : std::min(dx_hi, notional_cap_s / Traits::to_scalar(pool.cached_price_scale, PREC_D));
    }
    if (!(dx_hi > dx_lo)) return d;

    auto residual = [&](double dx_s)->double {
        T dx_t = Traits::from_scalar_dx(dx_s, PREC_D);
        auto pr = fx::post_trade_price_and_fee(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx_t);
        double p_new = static_cast<double>(pr.first) / PREC_D;
        double fee_pool = static_cast<double>(pr.second) / FPREC_D;
        double p_pool_bid = (1.0 - fee_pool) * p_new;
        double p_pool_ask = p_new / (1.0 - fee_pool);
        double p_cex_bid2 = (1.0 - fee_cex_s) * cex_price_s;
        double p_cex_ask2 = (1.0 + fee_cex_s) * cex_price_s;
        return (sel_i == 0) ? (p_pool_ask - p_cex_bid2) : (p_pool_bid - p_cex_ask2);
    };

    double F_lo = residual(dx_lo);
    double F_hi = residual(dx_hi);
    const bool cross = (F_lo * F_hi < 0.0);

    double dx_star_s = dx_hi;
    if (cross) {
        double root;
        if (detail::toms748_root(residual, dx_lo, dx_hi, F_lo, F_hi, root)) dx_star_s = std::max(root, dx_lo);
    } else {
        if ((sel_i == 0 && !(F_lo < 0.0)) || (sel_i == 1 && !(F_lo > 0.0))) return d;
        dx_star_s = dx_hi;
    }

    T dx_star_t = Traits::from_scalar_dx(dx_star_s, PREC_D);
    auto sim = fx::simulate_exchange_once(pool, static_cast<size_t>(sel_i), static_cast<size_t>(sel_j), dx_star_t);
    double dy_after_fee_s = static_cast<double>(sim.first) / PREC_D;

    double f_sell = 1.0 - fee_cex_s;
    double f_buy  = 1.0 + fee_cex_s;

    double profit_s = 0.0;
    if (sel_i == 0) {
        profit_s = dy_after_fee_s * cex_price_s * f_sell - dx_star_s - Traits::to_scalar(costs.gas_coin0, PREC_D);
    } else {
        profit_s = dy_after_fee_s - dx_star_s * cex_price_s * f_buy - Traits::to_scalar(costs.gas_coin0, PREC_D);
    }
    if (!(profit_s > 0.0)) return d;

    d.do_trade = true;
    d.i = sel_i; d.j = sel_j;
    d.dx = dx_star_t;
    d.profit = Traits::from_scalar_profit(profit_s, PREC_D);
    d.fee_tokens = sim.second;
    double notional_s = (sel_i == 0) ? dx_star_s : dx_star_s * cex_price_s;
    d.notional_coin0 = Traits::from_scalar_profit(notional_s, PREC_D);
    return d;
}

} // namespace trading
} // namespace arb
