// Candle-Driven Arbitrage Harness (multi-pool, threaded)
// -----------------------------------------------------
// - Loads candles once, generates two price events per candle, and processes
//   multiple pools in parallel.
// - Trades are decided by an exact fee-aware pre-filter, then sized by solving
//   for the *marginal price equality* after the swap (TOMS748 root) or by
//   taking the cap when equality is unreachable within the bracket.
// - Donations: fixed-per-period amounts, with robust "catch-up by count" that
//   *never* silently drops overdue periods. A dry-run on a copy enforces caps.
//
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <boost/math/tools/roots.hpp>

#include "twocrypto.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace json = boost::json;

// ------------------------------- Numeric mode --------------------------------
#if defined(ARB_MODE_F)
using RealT = float;
#elif defined(ARB_MODE_LD)
using RealT = long double;
#else
using RealT = double;
#endif

// ------------------------------ Small utilities ------------------------------
namespace {

template <typename T>
inline bool differs_rel(T a, T b, T rel = T(1e-12), T abs_eps = T(0)) {
    const T da    = std::abs(a - b);
    const T scale = std::max<T>(T(1), std::max(std::abs(a), std::abs(b)));
    return da > std::max(abs_eps, rel * scale);
}

struct Costs {
    RealT arb_fee_bps{static_cast<RealT>(10.0)};  // aggregator / CEX fee (bps)
    RealT gas_coin0{static_cast<RealT>(0.0)};     // fixed gas in coin0 units
    bool  use_volume_cap{false};
    RealT volume_cap_mult{static_cast<RealT>(1.0)};
};

struct Metrics {
    size_t trades{0};
    RealT  notional{0};
    RealT  lp_fee_coin0{0};
    RealT  arb_pnl_coin0{0};
    size_t n_rebalances{0};
    // Donations
    size_t donations{0};
    RealT  donation_coin0_total{0};
    std::array<RealT,2> donation_amounts_total{0,0};
};

std::mutex io_mu;

static inline std::array<RealT,2> pool_xp_from(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    const std::array<RealT,2>& balances,
    RealT price_scale
) {
    return {
        balances[0] * pool.precisions[0],
        balances[1] * pool.precisions[1] * price_scale
    };
}
static inline std::array<RealT,2> pool_xp_current(const twocrypto::TwoCryptoPoolT<RealT>& pool) {
    return pool_xp_from(pool, pool.balances, pool.cached_price_scale);
}
static inline RealT xp_to_tokens_j(
    const twocrypto::TwoCryptoPoolT<RealT>& pool, size_t j,
    RealT amount_xp, RealT price_scale
) {
    RealT v = amount_xp;
    if (j == 1) v /= price_scale;
    return v / pool.precisions[j];
}

static inline RealT dyn_fee(
    const std::array<RealT,2>& xp, RealT mid_fee, RealT out_fee, RealT fee_gamma
) {
    const RealT Bsum = xp[0] + xp[1];
    if (!(Bsum > 0)) return mid_fee;
    RealT B = RealT(4) * (xp[0]/Bsum) * (xp[1]/Bsum);
    B = fee_gamma * B / (fee_gamma * B + RealT(1) - B);
    return mid_fee * B + out_fee * (RealT(1) - B);
}

struct Decision {
    bool  do_trade{false};
    int   i{0}, j{1};
    RealT dx{0};
    RealT profit{0};
    RealT fee_tokens{0};
    RealT notional_coin0{0};
};

// ---------- Lightweight pool sims used by sizing / pre-filters ---------------
static inline std::pair<RealT, RealT>
post_trade_price_and_fee(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                         size_t i, size_t j, RealT dx)
{
    using Ops = stableswap::MathOps<RealT>;
    const RealT ps = pool.cached_price_scale;

    // balances after adding dx on i
    auto balances_local = pool.balances; balances_local[i] += dx;
    auto xp = pool_xp_from(pool, balances_local, ps);

    // outflow on j; xp AFTER the trade
    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, j);
    RealT dy_xp = xp[j] - y_out.value; xp[j] -= dy_xp;

    // dynamic fee at post-trade skew
    const RealT fee_pool = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);

    // D and marginal price at post-trade state (coin0 per coin1)
    const RealT D_new = Ops::newton_D(pool.A, pool.gamma, xp, RealT(0));
    const RealT p_new = Ops::get_p(xp, D_new, {pool.A, pool.gamma}) * ps;

    return {p_new, fee_pool};
}

static inline std::pair<RealT, RealT>
simulate_exchange_once(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                       size_t i, size_t j, RealT dx)
{
    using Ops = stableswap::MathOps<RealT>;
    const RealT ps = pool.cached_price_scale;

    auto balances_local = pool.balances; balances_local[i] += dx;
    auto xp = pool_xp_from(pool, balances_local, ps);
    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, j);
    RealT dy_xp = xp[j] - y_out.value; xp[j] -= dy_xp;

    RealT dy_tokens  = xp_to_tokens_j(pool, j, dy_xp, ps);
    RealT fee_pool       = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    RealT fee_tokens = fee_pool * dy_tokens;
    return {dy_tokens - fee_tokens, fee_tokens};
}

// -------------------------- Root solver convenience --------------------------
template <typename F>
static inline bool toms748_root(F&& f,
                                RealT lo, RealT hi,
                                RealT Flo, RealT Fhi,
                                RealT& out_root,
                                unsigned max_iters = 100)
{
    if (!(hi > lo) || !(Flo * Fhi < RealT(0))) return false; // need a strict sign change
    auto tol = boost::math::tools::eps_tolerance<RealT>(
        std::numeric_limits<RealT>::digits10 - 3
    );
    boost::uintmax_t it = max_iters;
    auto r = boost::math::tools::toms748_solve(f, lo, hi, Flo, Fhi, tol, it);
    out_root = (r.first + r.second) / RealT(2);
    return true;
}
// ----------------------- Profit-maximizing trade sizing ----------------------
static Decision decide_trade_size(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    RealT cex_price,
    const Costs& costs,
    RealT notional_cap_coin0,
    RealT min_swap_frac,
    RealT max_swap_frac
) {
    Decision best{}; // default: do_trade=false
    if (!(cex_price > 0)) return best;

    // Aggregator/CEX fee factors
    const RealT fee_cex  = costs.arb_fee_bps / RealT(1e4);
    const RealT f_sell1  = (RealT(1) - fee_cex); // proceeds factor when selling coin1 on CEX
    const RealT f_buy1   = (RealT(1) + fee_cex); // cost factor when buying coin1 on CEX

    // Small guards
    const RealT EPS_DX     = static_cast<RealT>(1e-18);
    const RealT REL_TOL    = static_cast<RealT>(1e-9);
    const unsigned MAX_GS_ITERS = 64;
    const int N_COARSE     = 12; // coarse scan points per direction (log-spaced, inc. endpoints)

    // Profit function for a given direction (i -> j)
    auto profit_fn = [&](int i, int j, RealT dx)->std::pair<RealT, RealT> {
        if (!(dx > 0)) return {RealT(0), RealT(0)};
        auto [dy_after_fee, fee_tokens] =
            simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), dx);

        RealT profit_coin0 = 0;
        if (i == 0) {
            // Put coin0 in, receive coin1 from pool, sell coin1 on CEX
            profit_coin0 = dy_after_fee * cex_price * f_sell1 - dx - costs.gas_coin0;
        } else {
            // Put coin1 in, receive coin0 from pool, buy coin1 back on CEX
            profit_coin0 = dy_after_fee - dx * cex_price * f_buy1 - costs.gas_coin0;
        }
        return {profit_coin0, fee_tokens};
    };

    // Golden-section maximization on [a,c] (assumes near-unimodal in practice).
    auto golden_section_max = [&](auto&& f, RealT a, RealT c, RealT& x_best, RealT& f_best) {
        // Ensure proper order
        if (!(c > a)) { x_best = a; f_best = f(a); return; }
        const RealT invphi  = static_cast<RealT>(0.6180339887498948482L); // (sqrt(5)-1)/2
        const RealT invphi2 = static_cast<RealT>(1) - invphi;

        RealT x1 = c - invphi * (c - a);
        RealT x2 = a + invphi * (c - a);
        RealT f1 = f(x1), f2 = f(x2);

        for (unsigned it = 0; it < MAX_GS_ITERS; ++it) {
            // Stop when absolute interval small relative to scale
            const RealT width = c - a;
            const RealT scale = std::max<RealT>(RealT(1), std::max(std::abs(a), std::abs(c)));
            if (width <= REL_TOL * scale) break;

            if (f1 < f2) {
                a = x1;
                x1 = x2; f1 = f2;
                x2 = a + invphi * (c - a);
                f2 = f(x2);
            } else {
                c = x2;
                x2 = x1; f2 = f1;
                x1 = c - invphi * (c - a);
                f1 = f(x1);
            }
        }
        if (f1 > f2) { x_best = x1; f_best = f1; } else { x_best = x2; f_best = f2; }
    };

    // Build direction candidates (0->1) and (1->0), pick the globally best.
    auto try_direction = [&](int i, int j)->Decision {
        Decision d{};
        const RealT avail = pool.balances[static_cast<size_t>(i)];
        if (!(avail > 0)) return d;

        // Bounds
        RealT dx_lo = std::max(EPS_DX, avail * std::max<RealT>(RealT(1e-12), min_swap_frac));
        RealT dx_hi = avail * max_swap_frac;

        // Notional cap in coin0 units
        if (std::isfinite(static_cast<double>(notional_cap_coin0)) && notional_cap_coin0 > 0) {
            if (i == 0) dx_hi = std::min(dx_hi, notional_cap_coin0);
            else        dx_hi = std::min(dx_hi, (cex_price > 0) ? notional_cap_coin0 / cex_price : dx_hi);
        }
        if (!(dx_hi > dx_lo)) return d;

        // Quick pre-filter using current-state bid/ask to skip obviously unprofitable directions.
        // (Keeps runtime low; we still fully maximize if it passes.)
        const auto xp_now    = pool_xp_current(pool);
        const RealT p_now    = stableswap::MathOps<RealT>::get_p(
                                   xp_now, pool.D, {pool.A, pool.gamma}) * pool.cached_price_scale;
        const RealT fee_out0 = dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);
        const RealT one_m    = std::max<RealT>(RealT(1) - fee_out0, RealT(1e-12));
        const RealT p_pool_bid0 = one_m * p_now;
        const RealT p_pool_ask0 = p_now / one_m;
        const RealT p_cex_bid   = (RealT(1) - fee_cex) * cex_price;
        const RealT p_cex_ask   = (RealT(1) + fee_cex) * cex_price;
        const RealT edge_01     = p_cex_bid - p_pool_ask0; // >0 suggests 0->1 profitable at margin (ignoring gas)
        const RealT edge_10     = p_pool_bid0 - p_cex_ask; // >0 suggests 1->0 profitable at margin (ignoring gas)
        if ((i == 0 && edge_01 <= 0) && (i == 1)) { /* fallthrough */ }
        if ((i == 1 && edge_10 <= 0) && (i == 0)) { /* fallthrough */ }
        // We do NOT early-return on negative edge; gas can only hurt, but we’ll still check endpoints.

        // Coarse log-spaced scan to find a good bracket for the maximum
        std::array<RealT, 1 + 12> xs{}; // capacity guard; N_COARSE<=12 default
        std::array<RealT, 1 + 12> fs{};
        const int K = std::max(2, N_COARSE);
        const RealT log_lo = std::log(dx_lo);
        const RealT log_hi = std::log(dx_hi);
        int argmax_k = 0;

        for (int k = 0; k < K; ++k) {
            RealT t = static_cast<RealT>(k) / static_cast<RealT>(K - 1);
            RealT x = std::exp(log_lo + t * (log_hi - log_lo));
            auto [pi, _fee_tok] = profit_fn(i, j, x);
            xs[k] = x; fs[k] = pi;
            if (k == 0 || pi > fs[argmax_k]) argmax_k = k;
        }

        // If the best coarse point is negative profit, no trade in this direction.
        if (!(fs[argmax_k] > 0)) return d;

        // Try to form a unimodal bracket around the coarse max; fallback to boundary if monotone.
        RealT a = dx_lo, c = dx_hi;
        bool have_bracket = false;
        if (argmax_k > 0 && argmax_k < K - 1) {
            if (fs[argmax_k] >= fs[argmax_k - 1] && fs[argmax_k] >= fs[argmax_k + 1]) {
                a = xs[argmax_k - 1];
                c = xs[argmax_k + 1];
                have_bracket = true;
            }
        }
        if (!have_bracket) {
            // Monotone case: take the better boundary
            RealT f_lo = fs[0], f_hi = fs[K - 1];
            RealT dx_pick = (f_hi >= f_lo) ? dx_hi : dx_lo;
            auto [pi, fee_tok] = profit_fn(i, j, dx_pick);
            if (!(pi > 0)) return d;
            d.do_trade = true;
            d.i = i; d.j = j;
            d.dx = dx_pick;
            d.profit = pi;
            d.fee_tokens = fee_tok;
            d.notional_coin0 = (i == 0) ? dx_pick : dx_pick * cex_price;
            return d;
        }

        // Refine the interior maximum with golden-section search of the true profit function.
        auto f_scalar = [&](RealT x)->RealT {
            auto [pi, _] = profit_fn(i, j, x);
            return pi;
        };
        RealT x_best = xs[argmax_k];
        RealT f_best = fs[argmax_k];
        golden_section_max(f_scalar, a, c, x_best, f_best);

        // Safety: clamp to [dx_lo, dx_hi]
        if (x_best < dx_lo) x_best = dx_lo;
        if (x_best > dx_hi) x_best = dx_hi;

        auto [pi_best, fee_tok_best] = profit_fn(i, j, x_best);
        if (!(pi_best > 0)) return d;

        d.do_trade = true;
        d.i = i; d.j = j;
        d.dx = x_best;
        d.profit = pi_best;
        d.fee_tokens = fee_tok_best;
        d.notional_coin0 = (i == 0) ? x_best : x_best * cex_price;
        return d;
    };

    // Evaluate both directions and pick the better positive profit, if any.
    Decision d01 = try_direction(0, 1);
    Decision d10 = try_direction(1, 0);
    if (d01.do_trade && (!d10.do_trade || d01.profit >= d10.profit)) return d01;
    if (d10.do_trade) return d10;
    return best; // no profitable trade
}

// ----------------------- Trading decision & sizing ---------------------------
static Decision decide_trade(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    RealT cex_price,
    const Costs& costs,
    RealT notional_cap_coin0,
    RealT min_swap_frac,
    RealT max_swap_frac
) {
    Decision d{};
    if (!(cex_price > 0)) return d;

    // ----- Exact fee-aware pre-filter at current state (bid/ask form) -----
    const auto  xp_now    = pool_xp_current(pool);
    const RealT p_now = stableswap::MathOps<RealT>::get_p(xp_now, pool.D, {pool.A, pool.gamma})
                            * pool.cached_price_scale;

    // Pool outflow fee at current skew
    const RealT fee_out0  = dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    // Aggregator/CEX fee on notional
    const RealT fee_cex   = costs.arb_fee_bps / RealT(1e4);

    // Pool bid/ask around the marginal mid (outflow-fee model)
    const RealT one_minus_fee0 = std::max<RealT>(RealT(1) - fee_out0, RealT(1e-12));
    const RealT p_pool_bid0    = one_minus_fee0 * p_now;     // sell 1 coin1 to pool -> coin0 you get
    const RealT p_pool_ask0    = p_now / one_minus_fee0;     // buy 1 coin1 from pool -> coin0 you pay

    // CEX bid/ask (coin0 per coin1)
    const RealT p_cex_bid = (RealT(1) - fee_cex) * cex_price;    // sell 1 coin1 on CEX
    const RealT p_cex_ask = (RealT(1) + fee_cex) * cex_price;    // buy 1 coin1 on CEX

    // Edges at dx->0 (coin0 per coin1)
    const RealT edge_01 = p_cex_bid - p_pool_ask0; // >0 : 0->1 profitable at the margin
    const RealT edge_10 = p_pool_bid0 - p_cex_ask; // >0 : 1->0 profitable at the margin

    int i = -1, j = -1;
    // printf("cex_price: %f, edge_01: %f, edge_10: %f\n", cex_price, edge_01, edge_10);
    if (edge_01 <= 0 && edge_10 <= 0) return d;    // no profitable direction

    // Choose the direction with the larger POSITIVE edge
    if (edge_01 >= edge_10) { i = 0; j = 1; } else { i = 1; j = 0; }

    // ----- Trade bounds -----
    const RealT avail = pool.balances[static_cast<size_t>(i)];
    if (!(avail > 0)) return d;
    RealT dx_lo = std::max<RealT>(RealT(1e-18), avail * std::max<RealT>(RealT(1e-12), min_swap_frac));
    RealT dx_hi = avail * max_swap_frac;
    if (std::isfinite(static_cast<double>(notional_cap_coin0)) && notional_cap_coin0 > 0) {
        dx_hi = (i == 0) ? std::min(dx_hi, notional_cap_coin0)
                         : std::min(dx_hi, notional_cap_coin0 / pool.cached_price_scale);

    }
    if (!(dx_hi > dx_lo)) return d;

    // ----- Residual: post-trade marginal equality (pool vs cex incl. fees) -----
    auto residual = [&](RealT dx)->RealT {
        auto [p_new, fee_pool] = post_trade_price_and_fee(pool, static_cast<size_t>(i), static_cast<size_t>(j), dx);
        // Pool bid/ask around the marginal mid, accounting for outflow fee on the *output* side
        const RealT p_pool_bid = (1 - fee_pool) * p_new;      // sell 1 coin1 to pool -> coin0 you get
        const RealT p_pool_ask = p_new / (1 - fee_pool);      // buy 1 coin1 from pool -> coin0 you pay

        // CEX bid/ask (coin0 per coin1)
        const RealT p_cex_bid = (RealT(1) - fee_cex) * cex_price; // sell 1 coin1 on CEX
        const RealT p_cex_ask = (RealT(1) + fee_cex) * cex_price; // buy 1 coin1 on CEX

        return (i == 0)
             ? (p_pool_ask - p_cex_bid)                  // 0->1
             : (p_pool_bid - p_cex_ask);                 // 1->0
    };
    RealT F_lo = residual(dx_lo), F_hi = residual(dx_hi);
    const bool cross = (F_lo * F_hi < 0);

    // Solve for marginal equality inside [lo, hi], else use the cap when profitable across the whole bracket.
    RealT dx_star = dx_hi;
    if (cross) {
        RealT root;
        if (toms748_root(residual, dx_lo, dx_hi, F_lo, F_hi, root)) dx_star = std::max(root, dx_lo);
    } else {
        // Quick reject if already "beyond equality" at the left edge.
        if ((i == 0 && !(F_lo < 0)) || (i == 1 && !(F_lo > 0))) return d;
        dx_star = dx_hi; // all profitable until cap
    }
    // ----- Final profit check using a single dry simulation -----
    auto [dy_after_fee, fee_tokens] = simulate_exchange_once(pool, static_cast<size_t>(i), static_cast<size_t>(j), dx_star);
    const RealT f_sell  = (RealT(1) - fee_cex);
    const RealT f_buy = (RealT(1) + fee_cex);
    RealT profit_coin0 = 0;
    if (i == 0) { // buy 1 on pool, sell 1 on cex
        profit_coin0 = dy_after_fee * cex_price * f_sell - dx_star - costs.gas_coin0;
    } else {      // buy 0 on pool, sell 0 on cex
        profit_coin0 = dy_after_fee - dx_star * cex_price * f_buy - costs.gas_coin0;
    }
    if (!(profit_coin0 > 0)) return d;

    d.do_trade = true;
    d.i = i; d.j = j;
    d.dx = dx_star;
    d.profit = profit_coin0;
    d.fee_tokens = fee_tokens;
    d.notional_coin0 = (i == 0) ? dx_star : dx_star * cex_price;
    return d;
}

// ------------------------------- Data structs --------------------------------
struct Candle { uint64_t ts; RealT open, high, low, close, volume; };
struct Event  { uint64_t ts; RealT p_cex; RealT volume; };

template <typename T>
static std::string to_str_1e18(T v) {
    long double scaled = static_cast<long double>(v) * 1e18L;
    if (scaled < 0) scaled = 0;
    std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << scaled;
    return oss.str();
}

static json::object pool_state_json(const twocrypto::TwoCryptoPoolT<RealT>& p) {
    json::object o;
    o["balances"]       = json::array{to_str_1e18(p.balances[0]), to_str_1e18(p.balances[1])};
    o["xp"]             = json::array{to_str_1e18(p.balances[0]*p.precisions[0]), to_str_1e18(p.balances[1]*p.precisions[1]*p.cached_price_scale)};
    o["D"]              = to_str_1e18(p.D);
    o["virtual_price"]  = to_str_1e18(p.virtual_price);
    o["xcp_profit"]     = to_str_1e18(p.xcp_profit);
    o["price_scale"]    = to_str_1e18(p.cached_price_scale);
    o["price_oracle"]   = to_str_1e18(p.cached_price_oracle);
    o["last_prices"]    = to_str_1e18(p.last_prices);
    o["totalSupply"]    = to_str_1e18(p.totalSupply);
    o["donation_shares"]   = to_str_1e18(p.donation_shares);
    o["donation_unlocked"] = to_str_1e18(p.donation_unlocked());
    o["timestamp"]      = p.block_timestamp;
    return o;
}

// ----------------------------- JSON parsing helpers --------------------------
static inline RealT parse_scaled_1e18(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr) / 1e18L);
    if (v.is_double()) return static_cast<RealT>(v.as_double() / 1e18);
    if (v.is_int64())  return static_cast<RealT>(static_cast<long double>(v.as_int64()) / 1e18L);
    if (v.is_uint64()) return static_cast<RealT>(static_cast<long double>(v.as_uint64()) / 1e18L);
    return RealT(0);
}
static inline RealT parse_fee_1e10(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr) / 1e10L);
    if (v.is_double()) return static_cast<RealT>(v.as_double() / 1e10);
    if (v.is_int64())  return static_cast<RealT>(static_cast<long double>(v.as_int64()) / 1e10L);
    if (v.is_uint64()) return static_cast<RealT>(static_cast<long double>(v.as_uint64()) / 1e10L);
    return RealT(0);
}
static inline RealT parse_plain_real(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr));
    if (v.is_double()) return static_cast<RealT>(v.as_double());
    if (v.is_int64())  return static_cast<RealT>(v.as_int64());
    if (v.is_uint64()) return static_cast<RealT>(v.as_uint64());
    return RealT(0);
}

struct PoolInit {
    std::array<RealT,2> precisions{RealT(1), RealT(1)};
    RealT A{static_cast<RealT>(100000.0)};
    RealT gamma{0};
    RealT mid_fee{static_cast<RealT>(3e-4)};
    RealT out_fee{static_cast<RealT>(5e-4)};
    RealT fee_gamma{static_cast<RealT>(0.23)};
    RealT allowed_extra{static_cast<RealT>(1e-3)};
    RealT adj_step{static_cast<RealT>(1e-3)};
    RealT ma_time{static_cast<RealT>(600.0)};
    RealT initial_price{static_cast<RealT>(1.0)};
    std::array<RealT,2> initial_liq{static_cast<RealT>(1e6), static_cast<RealT>(1e6)};
    uint64_t start_ts{0};
    // Donation controls (plain units)
    RealT donation_apy{0};
    RealT donation_frequency{0};   // seconds
    RealT donation_coins_ratio{static_cast<RealT>(0.5)};
};

static void parse_pool_entry(const json::object& entry, PoolInit& out_pool, Costs& out_costs,
                             json::object& echo_pool, json::object& echo_costs)
{
    const json::object pool = entry.contains("pool") ? entry.at("pool").as_object() : entry;
    echo_pool = pool;

    if (auto* v = pool.if_contains("initial_liquidity")) {
        const auto& a = v->as_array();
        out_pool.initial_liq[0] = parse_scaled_1e18(a[0]);
        out_pool.initial_liq[1] = parse_scaled_1e18(a[1]);
    }
    if (auto* v = pool.if_contains("A")) out_pool.A = parse_plain_real(*v);
    if (auto* v = pool.if_contains("gamma")) out_pool.gamma = parse_plain_real(*v);
    if (auto* v = pool.if_contains("mid_fee")) out_pool.mid_fee = parse_fee_1e10(*v);
    if (auto* v = pool.if_contains("out_fee")) out_pool.out_fee = parse_fee_1e10(*v);
    if (auto* v = pool.if_contains("fee_gamma")) out_pool.fee_gamma = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("allowed_extra_profit")) out_pool.allowed_extra = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("adjustment_step")) out_pool.adj_step = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("ma_time")) out_pool.ma_time = parse_plain_real(*v);
    if (auto* v = pool.if_contains("initial_price")) out_pool.initial_price = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("start_timestamp")) out_pool.start_ts = static_cast<uint64_t>(parse_plain_real(*v));
    if (auto* v = pool.if_contains("donation_apy")) out_pool.donation_apy = parse_plain_real(*v);
    if (auto* v = pool.if_contains("donation_frequency")) out_pool.donation_frequency = parse_plain_real(*v);
    if (auto* v = pool.if_contains("donation_coins_ratio")) {
        RealT r = parse_plain_real(*v);
        out_pool.donation_coins_ratio = std::clamp<RealT>(r, 0, 1);
    }

    if (auto* c = entry.if_contains("costs")) {
        echo_costs = c->as_object();
        const auto co = c->as_object();
        if (auto* v = co.if_contains("arb_fee_bps")) out_costs.arb_fee_bps = parse_plain_real(*v);
        if (auto* v = co.if_contains("gas_coin0")) out_costs.gas_coin0 = parse_plain_real(*v);
        if (auto* v = co.if_contains("use_volume_cap")) out_costs.use_volume_cap = v->as_bool();
        if (auto* v = co.if_contains("volume_cap_mult")) out_costs.volume_cap_mult = parse_plain_real(*v);
    } else {
        echo_costs = json::object{};
    }
}

// --------------------------- Candles & eventization --------------------------
static std::vector<Candle>
load_candles(const std::string& path, size_t max_candles = 0, double squeeze_frac = .999)
{
    std::vector<Candle> out; out.reserve(1024);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open candles file: " + path);
    std::ostringstream oss; oss << in.rdbuf();
    const std::string s = oss.str();

    json::value val = json::parse(s);
    if (!val.is_array()) throw std::runtime_error("Candles JSON must be an array of arrays");

    const auto& arr = val.as_array();
    const size_t limit = (max_candles ? std::min(max_candles, arr.size()) : arr.size());
    out.reserve(limit);

    auto to_d = [](const json::value& v)->RealT {
        if (v.is_double()) return static_cast<RealT>(v.as_double());
        if (v.is_int64())  return static_cast<RealT>(v.as_int64());
        if (v.is_uint64()) return static_cast<RealT>(v.as_uint64());
        return RealT(0);
    };

    for (size_t idx = 0; idx < limit; ++idx) {
        const auto& a = arr[idx].as_array();
        if (a.size() < 6) continue;
        Candle c{};
        uint64_t ts = 0;
        const auto& tsv = a[0];
        if (tsv.is_uint64()) ts = tsv.as_uint64();
        else if (tsv.is_int64()) ts = static_cast<uint64_t>(tsv.as_int64());
        else if (tsv.is_double()) ts = static_cast<uint64_t>(tsv.as_double());
        if (ts > 10000000000ULL) ts /= 1000ULL; // ms->s
        c.ts = ts;

        c.open = to_d(a[1]); c.high = to_d(a[2]);
        c.low  = to_d(a[3]); c.close = to_d(a[4]);
        c.volume = to_d(a[5]);

        if (squeeze_frac > 0.0) {
            const RealT oc_mid = RealT(0.5) * (c.open + c.close);
            if (oc_mid > 0) {
                const RealT max_h = oc_mid * (1 + squeeze_frac);
                const RealT min_l = oc_mid * (1 - squeeze_frac);
                if (c.high > max_h) c.high = max_h;
                if (c.low  < min_l) c.low  = min_l;
            }
        }
        // if (c.ts < 1672643800*10) { //CANDLE HARDSTOP
        out.push_back(c);
        // }
    }
    return out;
}

// two events per candle: "low first" vs "high first" path chooser
static std::vector<Event> gen_events(const std::vector<Candle>& cs) {
    std::vector<Event> evs; evs.reserve(cs.size()*2);
    for (const auto& c : cs) {
        const RealT path1 = std::abs(c.open - c.low)  + std::abs(c.high - c.close);
        const RealT path2 = std::abs(c.open - c.high) + std::abs(c.low  - c.close);
        const bool first_low = path1 < path2;
        evs.push_back(Event{c.ts ,  first_low ? c.low  : c.high, c.volume/RealT(2)});
        evs.push_back(Event{c.ts + 10, first_low ? c.high : c.low,  c.volume/RealT(2)});
    }
    std::sort(evs.begin(), evs.end(), [](auto& a, auto& b){ return a.ts < b.ts; });
    return evs;
}

// Directly load events.
// Accepts either events [[ts, price, volume], ...] or candles [[ts,o,h,l,c,v], ...].
// For candles, we map to a single event per candle using close and volume.
static std::vector<Event> load_events(const std::string& path, size_t max_events = 0) {
    std::vector<Event> out; out.reserve(1024);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open events file: " + path);
    std::ostringstream oss; oss << in.rdbuf();
    const std::string s = oss.str();

    json::value val = json::parse(s);
    const json::array* arr = nullptr;
    if (val.is_array()) {
        arr = &val.as_array();
    } else if (val.is_object()) {
        const auto& o = val.as_object();
        if (o.contains("data") && o.at("data").is_array()) arr = &o.at("data").as_array();
        else if (o.contains("events") && o.at("events").is_array()) arr = &o.at("events").as_array();
    }
    if (!arr) throw std::runtime_error("Events JSON must be an array of [ts,price,vol] or object with 'data'/'events' array");

    const size_t limit = (max_events ? std::min(max_events, arr->size()) : arr->size());
    out.reserve(limit);

    auto to_d = [](const json::value& v)->RealT {
        if (v.is_double()) return static_cast<RealT>(v.as_double());
        if (v.is_int64())  return static_cast<RealT>(v.as_int64());
        if (v.is_uint64()) return static_cast<RealT>(v.as_uint64());
        if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr));
        return RealT(0);
    };

    for (size_t i = 0; i < limit; ++i) {
        const auto& e = (*arr)[i];
        if (e.is_array()) {
            const auto& a = e.as_array();
            if (a.size() < 3) continue;
            uint64_t ts = 0;
            const auto& tsv = a[0];
            if (tsv.is_uint64()) ts = tsv.as_uint64();
            else if (tsv.is_int64()) ts = static_cast<uint64_t>(tsv.as_int64());
            else if (tsv.is_double()) ts = static_cast<uint64_t>(tsv.as_double());
            else if (tsv.is_string()) ts = static_cast<uint64_t>(std::strtoll(tsv.as_string().c_str(), nullptr, 10));
            if (ts > 10000000000ULL) ts /= 1000ULL; // ms->s guard

            if (a.size() >= 6) {
                // Candles: [ts, o, h, l, c, v] -> use close and volume
                Event ev{ts, to_d(a[4]), to_d(a[5])};
                out.push_back(ev);
            } else {
                // Events: [ts, price, volume]
                Event ev{ts, to_d(a[1]), to_d(a[2])};
                out.push_back(ev);
            }
        } else if (e.is_object()) {
            const auto& o = e.as_object();
            uint64_t ts = 0;
            if (auto* v = o.if_contains("ts")) {
                if (v->is_uint64()) ts = v->as_uint64();
                else if (v->is_int64()) ts = static_cast<uint64_t>(v->as_int64());
                else if (v->is_double()) ts = static_cast<uint64_t>(v->as_double());
            }
            if (ts > 10000000000ULL) ts /= 1000ULL;
            RealT price = 0, vol = 0;
            if (auto* v = o.if_contains("price")) price = to_d(*v);
            else if (auto* v = o.if_contains("close")) price = to_d(*v);
            else if (auto* v = o.if_contains("c")) price = to_d(*v);
            if (auto* v = o.if_contains("volume")) vol = to_d(*v);
            else if (auto* v = o.if_contains("v")) vol = to_d(*v);
            if (ts && price > 0) out.push_back(Event{ts, price, vol});
        }
    }
    std::sort(out.begin(), out.end(), [](auto& A, auto& B){ return A.ts < B.ts; });
    return out;
}

// ----------------------------- Donation scheduler ----------------------------
struct DonationCfg {
    bool     enabled{false};
    uint64_t freq_s{0};
    uint64_t next_ts{0};
    RealT    apy{0};              // fraction per year, e.g., 0.05
    RealT    ratio1{0.5};
};

static inline RealT coin0_equiv(RealT amt0, RealT amt1, RealT ps) {
    return amt0 + amt1 * ps;
}

// Try to donate all overdue periods in one shot, subject to the pool's donation cap.
// Uses a dry-run copy and binary search on the number of periods. Advances the schedule
// by exactly the number of periods that were actually committed (never silently drops).
template <typename ActionsArray, typename StatesArray>
static void make_donation(
    twocrypto::TwoCryptoPoolT<RealT>& pool,
    DonationCfg& cfg,
    uint64_t ev_ts,
    bool save_actions,
    ActionsArray& actions,
    StatesArray& states,
    Metrics& m
) {
    if (!cfg.enabled || cfg.next_ts == 0 || ev_ts < cfg.next_ts) return;

    // Compute one-period donation from current TVL
    constexpr RealT SEC_PER_YEAR = static_cast<RealT>(365.0 * 86400.0);
    const RealT ps   = pool.cached_price_scale;
    const RealT tvl0 = pool.balances[0] + pool.balances[1] * ps;
    const RealT frac = cfg.apy * static_cast<RealT>(static_cast<long double>(cfg.freq_s) / static_cast<long double>(SEC_PER_YEAR));
    const RealT coin0_equiv_amt = tvl0 * frac;
    const RealT amt0 = (RealT(1) - cfg.ratio1) * coin0_equiv_amt;
    const RealT amt1 = (ps > 0) ? (cfg.ratio1 * coin0_equiv_amt / ps) : RealT(0);

    const uint64_t ts_due = cfg.next_ts;
    const RealT ps_before = pool.cached_price_scale;
    try {
        (void)pool.add_liquidity({amt0, amt1}, RealT(0), /*donation=*/true);
        const RealT ps_after  = pool.cached_price_scale;
        if (differs_rel(ps_after, ps_before)) m.n_rebalances += 1;
        m.donations += 1;
        m.donation_amounts_total[0] += amt0;
        m.donation_amounts_total[1] += amt1;
        m.donation_coin0_total += coin0_equiv(amt0, amt1, ps_before);

        if (save_actions) {
            json::object da;
            da["type"]            = "donation";
            da["ts"]              = ev_ts;
            da["ts_due"]          = ts_due;
            da["amounts"]         = json::array{static_cast<double>(amt0), static_cast<double>(amt1)};
            da["amounts_wei"]     = json::array{to_str_1e18(amt0), to_str_1e18(amt1)};
            da["price_scale"]     = static_cast<double>(pool.cached_price_scale);
            da["donation_ratio1"] = static_cast<double>(cfg.ratio1);
            da["apy_per_year"]     = static_cast<double>(cfg.apy);
            da["freq_s"]           = static_cast<uint64_t>(cfg.freq_s);
            actions.push_back(std::move(da));
        }
    } catch (...) {
        // Ignore failed donation
    }

    // Advance schedule by exactly one period (no catch-up)
    cfg.next_ts = ts_due + cfg.freq_s;
}

} // namespace

// =================================== MAIN ====================================
int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::unitbuf);

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <pools.json> <candles_or_events.json> <output.json>"
                  << " [--n-candles N] [--save-actions] [--events]"
                  << " [--min-swap F] [--max-swap F]"
                  << " [--threads N|-n N] [--candle-filter PCT]"
                  << " [--dustswapfreq S]\n";
        return 1;
    }

    // ---------- CLI ----------
    const std::string pools_path   = argv[1];
    const std::string candles_path = argv[2];
    const std::string out_path     = argv[3];

    size_t max_candles    = 0;
    bool   save_actions   = false;
    bool   use_events     = false;
    double min_swap_frac  = 1e-6;
    double max_swap_frac  = 1.0;
    size_t n_workers      = std::max<size_t>(1, std::thread::hardware_concurrency());
    double candle_filter_pct = 99;       // +/-X% squeeze around (O+C)/2
    uint64_t dustswapfreq_s  = 3600;       // EMA update tick cadence when idle

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        try {
            if (arg == "--n-candles"     && i+1 < argc) max_candles       = static_cast<size_t>(std::stoll(argv[++i]));
            else if (arg == "--save-actions")           save_actions       = true;
            else if (arg == "--events")                 use_events         = true;
            else if (arg == "--min-swap"   && i+1 < argc) min_swap_frac    = std::stod(argv[++i]);
            else if (arg == "--max-swap"   && i+1 < argc) max_swap_frac    = std::stod(argv[++i]);
            else if ((arg == "--threads" || arg == "-n") && i+1 < argc) n_workers = static_cast<size_t>(std::stoll(argv[++i]));
            else if (arg == "--candle-filter" && i+1 < argc) candle_filter_pct = std::stod(argv[++i]);
            else if (arg == "--dustswapfreq"  && i+1 < argc) dustswapfreq_s    = static_cast<uint64_t>(std::stoll(argv[++i]));
        } catch (...) { /* ignore bad flags */ }
    }

    try {
        using clk = std::chrono::high_resolution_clock;

        // ---------- Load & build events once ----------
        const auto t_read0 = clk::now();
        std::vector<Event> events;
        if (use_events) {
            events = load_events(candles_path, max_candles);
        } else {
            const auto candles = load_candles(candles_path, max_candles, candle_filter_pct / 100.0);
            events = gen_events(candles);
        }
        const auto t_read1 = clk::now();

        // ---------- Read pool configs ----------
        std::ifstream in(pools_path);
        if (!in) throw std::runtime_error("Cannot open pools json: " + pools_path);
        const std::string s((std::istreambuf_iterator<char>(in)), {});
        const json::value root = json::parse(s);

        std::vector<json::object> entries;
        if (root.is_object()) {
            const auto& obj = root.as_object();
            if (obj.contains("pools")) for (auto& v : obj.at("pools").as_array()) entries.push_back(v.as_object());
            else if (obj.contains("pool")) entries.push_back(obj);
            else throw std::runtime_error("Invalid pools json: expected 'pools' array or single 'pool'");
        } else if (root.is_array()) {
            for (auto& v : root.as_array()) entries.push_back(v.as_object());
        } else {
            throw std::runtime_error("Invalid pools json root type");
        }

        // ---------- Thread pool ----------
        struct Job { size_t idx; json::object entry; };
        std::vector<Job> jobs; jobs.reserve(entries.size());
        for (size_t i = 0; i < entries.size(); ++i) jobs.push_back(Job{i, entries[i]});

        std::vector<json::object> results(entries.size());
        std::atomic<size_t> next{0};

        const auto t_exec0 = clk::now();
        std::vector<std::thread> threads; threads.reserve(n_workers);

        for (size_t t = 0; t < n_workers; ++t) {
            threads.emplace_back([&]() {
                while (true) {
                    const size_t idx = next.fetch_add(1);
                    if (idx >= jobs.size()) break;

                    {
                        std::lock_guard<std::mutex> lk(io_mu);
                        std::cout << "dispatch job " << (idx + 1) << "/" << jobs.size() << "\n";
                    }

                    const auto& entry = jobs[idx].entry;

                    // ---- Parse one pool entry ----
                    json::object echo_pool, echo_costs;
                    Costs costs{}; PoolInit cfg{};
                    parse_pool_entry(entry, cfg, costs, echo_pool, echo_costs);

                    using Pool = twocrypto::TwoCryptoPoolT<RealT>;
                    Pool pool({cfg.precisions[0], cfg.precisions[1]}, cfg.A, cfg.gamma,
                              cfg.mid_fee, cfg.out_fee, cfg.fee_gamma,
                              cfg.allowed_extra, cfg.adj_step, cfg.ma_time, cfg.initial_price);

                    // Align EMA baseline before any liquidity
                    uint64_t init_ts = cfg.start_ts ? cfg.start_ts : (events.empty() ? 0 : events.front().ts);
                    if (init_ts) pool.set_block_timestamp(init_ts);

                    (void)pool.add_liquidity({cfg.initial_liq[0], cfg.initial_liq[1]}, RealT(0));

                    const RealT ps_init   = pool.cached_price_scale;
                    const RealT tvl_start = pool.balances[0] + pool.balances[1] * ps_init;

                    // Donation configuration derived from initial TVL
                    DonationCfg dcfg{};
                    if (cfg.donation_apy > 0 && cfg.donation_frequency > 0 && !events.empty()) {
                        constexpr RealT SEC_PER_YEAR = static_cast<RealT>(365.0 * 86400.0);
                        dcfg.enabled = true;
                        dcfg.freq_s  = static_cast<uint64_t>(cfg.donation_frequency);
                        const uint64_t base_ts = cfg.start_ts ? cfg.start_ts : events.front().ts;
                        dcfg.next_ts = base_ts;

                        dcfg.apy    = cfg.donation_apy;
                        dcfg.ratio1 = std::clamp<RealT>(cfg.donation_coins_ratio, 0, 1);
                    }

                    // ---- Per-job run state ----
                    Metrics m{};
                    json::array actions;
                    json::array states;
                    // auto push_state = [&]() { if (save_actions) states.push_back(pool_state_json(pool)); };
                    // temp do-nothing lambda
                    auto push_state = [&]() { if (save_actions) { }; };

                    auto count_rebalance = [&](RealT before, RealT after){
                        if (differs_rel(after, before)) m.n_rebalances += 1;
                    };

                    // Metric: sum of |p_cex - price_scale| at the beginning of each event
                    long double total_cex_diff = 0.0L;
                    long double max_cex_diff = 0.0L;
                    // Metric: fraction of time price_scale deviates more than 10% from p_cex
                    const RealT FAR_THRESH = static_cast<RealT>(0.10);
                    long double time_far_s = 0.0L;
                    uint64_t last_ts_band = 0;
                    bool have_band = false;
                    bool last_far = false;
                    const auto t_pool0 = clk::now();
                    uint64_t last_dust_ts = init_ts;
                    push_state(); // initial snapshot

                    // ---- Main event loop ----
                    for (const auto& ev : events) {
                        pool.set_block_timestamp(ev.ts);
                        // Sample at beginning of event (pre-trade, pre-tick)
                        const RealT cur_diff_abs = std::abs(ev.p_cex - pool.cached_price_scale);
                        total_cex_diff += static_cast<long double>(cur_diff_abs);
                        if (static_cast<long double>(cur_diff_abs) > max_cex_diff) max_cex_diff = static_cast<long double>(cur_diff_abs);
                        // Accumulate time when |ps/p_cex - 1| > 10%
                        if (have_band && ev.ts > last_ts_band && last_far) {
                            time_far_s += static_cast<long double>(ev.ts - last_ts_band);
                        }
                        if (ev.p_cex > 0) {
                            const RealT rel = std::abs(pool.cached_price_scale / ev.p_cex - RealT(1));
                            last_far = (rel > FAR_THRESH);
                        } else {
                            last_far = false;
                        }
                        last_ts_band = ev.ts;
                        have_band = true;
                        push_state(); // after time update

                        // Donation: single-period, based on current TVL
                        make_donation(pool, dcfg, ev.ts, save_actions, actions, states, m);

                        const RealT pre_p_pool = pool.get_p();

                        // Optional volume cap (coin0 notional)
                        RealT notional_cap_coin0 = std::numeric_limits<RealT>::infinity();
                        if (costs.use_volume_cap) {
                            notional_cap_coin0 = ev.volume * costs.volume_cap_mult;
                        }
                        // Decide and trade
                        Decision d = decide_trade(pool, ev.p_cex, costs, notional_cap_coin0,
                                                  static_cast<RealT>(min_swap_frac),
                                                  static_cast<RealT>(max_swap_frac));

                        if (!d.do_trade) {
                            // Idle: opportunistic EMA/price tweak at coarse cadence
                            const bool can_tick = (dustswapfreq_s == 0) ||
                                                  (ev.ts >= pool.last_timestamp + dustswapfreq_s);
                            if (!can_tick) continue;
                            try {
                                const RealT log_ps_before = pool.cached_price_scale;
                                const RealT log_oracle_before = pool.cached_price_oracle;
                                const RealT log_xcp_profit_before = pool.xcp_profit;
                                const RealT log_vp_before = pool.get_vp_boosted();
                                pool.tick();
                                const RealT log_ps_after  = pool.cached_price_scale;
                                const RealT log_oracle_after = pool.cached_price_oracle;
                                const RealT log_xcp_profit_after = pool.xcp_profit;
                                const RealT log_vp_after = pool.get_vp_boosted();
                                count_rebalance(log_ps_before, log_ps_after);
                                if (save_actions) {
                                    json::object tr;
                                    tr["type"] = "tick";
                                    tr["ts"]   = ev.ts;
                                    tr["p_cex"] = static_cast<double>(ev.p_cex);
                                    tr["ps_before"] = static_cast<double>(log_ps_before);
                                    tr["psafter"] = static_cast<double>(log_ps_after);
                                    tr["oracle_before"] = static_cast<double>(log_oracle_before);
                                    tr["oracle_after"] = static_cast<double>(log_oracle_after);
                                    tr["xcp_profit_before"] = static_cast<double>(log_xcp_profit_before);
                                    tr["xcp_profit_after"] = static_cast<double>(log_xcp_profit_after);
                                    tr["vp_before"] = static_cast<double>(log_vp_before);
                                    tr["vp_after"] = static_cast<double>(log_vp_after);
                                    actions.push_back(std::move(tr));
                                    push_state();
                                }
                                last_dust_ts = ev.ts;
                            } catch (...) { /* ignore */ }
                            continue;
                        }

                        try {
                            const RealT ps_before = pool.cached_price_scale;
                            const RealT oracle_before = pool.cached_price_oracle;
                            const RealT last_ts_before = pool.last_timestamp;
                            const RealT lp_before = pool.last_prices;
                            const RealT log_xcp_profit_pre = pool.xcp_profit;
                            const RealT log_vp_pre = pool.get_vp_boosted();
                            auto res = pool.exchange((RealT)d.i, (RealT)d.j, d.dx, RealT(0));
                            const RealT log_xcp_profit_after = pool.xcp_profit;
                            const RealT log_vp_after = pool.get_vp_boosted();
                            const RealT last_ts_after = pool.last_timestamp;
                            const RealT oracle_after = pool.cached_price_oracle;
                            const RealT dy_after_fee = res[0];
                            const RealT fee_tokens   = res[1];
                            const RealT ps_after     = pool.cached_price_scale;
                            const RealT p_after = pool.get_p();
                            const RealT lp_after = pool.last_prices;
                            // printf("t: %llu, vp_boosted_before=%10.12Lf, vp_boosted_after=%10.12Lf\n", ev.ts, static_cast<double>(log_vp_pre), static_cast<double>(log_vp_after));
                            m.trades   += 1;
                            m.notional += d.notional_coin0;
                            m.lp_fee_coin0 += (d.j == 1 ? fee_tokens * ev.p_cex : fee_tokens);
                            m.arb_pnl_coin0 += d.profit;
                            count_rebalance(ps_before, ps_after);

                            if (save_actions) {
                                json::object tr;
                                tr["type"] = "exchange";
                                tr["ts"] = ev.ts;
                                tr["i"]  = d.i; tr["j"] = d.j;
                                tr["dx"] = static_cast<double>(d.dx);
                                tr["dx_wei"] = to_str_1e18(d.dx);
                                tr["dy_after_fee"] = static_cast<double>(dy_after_fee);
                                tr["fee_tokens"]   = static_cast<double>(fee_tokens);
                                tr["profit_coin0"] = static_cast<double>(d.profit);
                                tr["p_cex"]        = static_cast<double>(ev.p_cex);
                                tr["p_pool_before"]= static_cast<double>(pre_p_pool);
                                tr["p_pool_after"] = static_cast<double>(p_after);
                                tr["oracle_before"] = static_cast<double>(oracle_before);
                                tr["oracle_after"] = static_cast<double>(oracle_after);
                                tr["ps_before"] = static_cast<double>(ps_before);
                                tr["ps_after"] = static_cast<double>(ps_after);
                                tr["last_ts_before"] = static_cast<double>(last_ts_before);
                                tr["last_ts_after"] = static_cast<double>(last_ts_after);
                                tr["lp_before"] = static_cast<double>(lp_before);
                                tr["lp_after"] = static_cast<double>(lp_after);
                                tr["xcp_profit_before"] = static_cast<double>(log_xcp_profit_pre);
                                tr["xcp_profit_after"] = static_cast<double>(log_xcp_profit_after);
                                tr["vp_before"] = static_cast<double>(log_vp_pre);
                                tr["vp_after"] = static_cast<double>(log_vp_after);
                                actions.push_back(std::move(tr));
                                push_state();
                            }
                        } catch (...) {
                            // Trade failed; ignore and continue
                        }
                    }

                    const auto t_pool1 = clk::now();
                    const double pool_exec_ms =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t_pool1 - t_pool0).count() / 1e6;

                    // ---- Summaries ----
                    json::object summary;
                    summary["events"]               = events.size();
                    summary["trades"]               = m.trades;
                    summary["total_notional_coin0"] = static_cast<double>(m.notional);
                    summary["lp_fee_coin0"]         = static_cast<double>(m.lp_fee_coin0);
                    summary["arb_pnl_coin0"]        = static_cast<double>(m.arb_pnl_coin0);
                    summary["n_rebalances"]         = m.n_rebalances;
                    summary["donations"]            = m.donations;
                    summary["donation_coin0_total"] = static_cast<double>(m.donation_coin0_total);
                    summary["donation_amounts_total"] =
                        json::array{static_cast<double>(m.donation_amounts_total[0]),
                                    static_cast<double>(m.donation_amounts_total[1])};
                    summary["pool_exec_ms"] = pool_exec_ms;

                    // APY metrics
                    constexpr RealT SEC_PER_YEAR = static_cast<RealT>(365.0 * 86400.0);
                    const uint64_t t_start = events.empty() ? init_ts : events.front().ts;
                    const uint64_t t_end   = events.empty() ? init_ts : events.back().ts;
                    const double duration_s = (t_end > t_start) ? double(t_end - t_start) : 0.0;

                    // Finalize time-averaged |p_cex - price_scale| (units: price)
                    double avg_cex_diff = -1.0;
                    if (duration_s > 0.0) {
                        avg_cex_diff = static_cast<double>(total_cex_diff) / duration_s;
                    }
                    summary["avg_cex_diff"] = avg_cex_diff;
                    summary["max_cex_diff"] = static_cast<double>(max_cex_diff);
                    // Fraction of time price_scale deviates >10% from CEX price
                    double cex_follow_time_frac = -1.0;
                    if (duration_s > 0.0) {
                        cex_follow_time_frac = 1 - static_cast<double>(time_far_s) / duration_s;
                    }
                    summary["cex_follow_time_frac"] = cex_follow_time_frac;

                    const RealT tvl_end = pool.balances[0] + pool.balances[1] * pool.cached_price_scale;
                    // Baseline: HODL initial balances valued at end price (coin0 units)
                    const RealT v_hold_end = cfg.initial_liq[0] + cfg.initial_liq[1] * pool.cached_price_scale;
                    double apy_coin0 = 0.0, apy_coin0_boost = 0.0, apy_vp = 0.0;
                    double apy_coin0_raw = 0.0, apy_coin0_boost_raw = 0.0; // before baseline subtraction

                    if (duration_s > 0.0 && tvl_start > 0) {
                        const double exponent = SEC_PER_YEAR / duration_s;
                        const long double vp_end = static_cast<long double>(pool.get_virtual_price());
                        apy_vp  = (vp_end > 0.0) ? std::pow(static_cast<double>(vp_end), exponent) - 1.0 : -1.0;
                        // Raw APYs against start TVL (kept for transparency)
                        apy_coin0_raw = (tvl_end > 0) ? std::pow(static_cast<double>(tvl_end / tvl_start), exponent) - 1.0 : -1.0;
                        const double tvl_end_adj_raw = static_cast<double>(tvl_end) - static_cast<double>(m.donation_coin0_total);
                        apy_coin0_boost_raw = (tvl_end_adj_raw > 0.0)
                                           ? std::pow(tvl_end_adj_raw / tvl_start, exponent) - 1.0
                                           : -1.0;
                        // Baseline-subtracted (excess) APYs: compare to HODL valued at end price
                        if (v_hold_end > 0) {
                            apy_coin0 = std::pow(static_cast<double>(tvl_end / v_hold_end), exponent) - 1.0;
                            const double tvl_end_adj = static_cast<double>(tvl_end) - static_cast<double>(m.donation_coin0_total);
                            apy_coin0_boost = (tvl_end_adj > 0.0)
                                               ? std::pow(tvl_end_adj / static_cast<double>(v_hold_end), exponent) - 1.0
                                               : -1.0;
                        } else {
                            apy_coin0 = -1.0;
                            apy_coin0_boost = -1.0;
                        }
                    }
                    summary["t_start"]           = t_start;
                    summary["t_end"]             = t_end;
                    summary["duration_s"]        = duration_s;
                    summary["tvl_coin0_start"]   = static_cast<double>(tvl_start);
                    summary["tvl_coin0_end"]     = static_cast<double>(tvl_end);
                    summary["baseline_hold_end_coin0"] = static_cast<double>(v_hold_end);
                    summary["apy"]               = apy_vp;
                    // Baseline-subtracted APYs (excess over HODL in coin0)
                    summary["apy_coin0"]         = apy_coin0;
                    summary["apy_coin0_boost"]= apy_coin0_boost;
                    // For reference, include raw (non-baseline) APYs
                    summary["apy_coin0_raw"]         = apy_coin0_raw;
                    summary["apy_coin0_boost_raw"]= apy_coin0_boost_raw;
                    summary["vp"]                = static_cast<double>(pool.get_virtual_price());
                    summary["vp_boosted"]        = static_cast<double>(pool.get_vp_boosted());
                    summary["vpminusone"]        = static_cast<double>(pool.get_virtual_price() - 1.0);
                    json::object params;
                    params["pool"] = echo_pool;
                    if (!echo_costs.empty()) params["costs"] = echo_costs;

                    json::object out;
                    out["result"]      = summary;
                    out["params"]      = params;
                    out["final_state"] = pool_state_json(pool);
                    if (save_actions) {
                        out["actions"] = actions;
                        // out["states"]  = states;
                    }
                    results[idx] = std::move(out);

                    {
                        std::lock_guard<std::mutex> lk(io_mu);
                        std::cout << "finished job " << (idx + 1) << "/" << jobs.size()
                                  << ", time: " << std::fixed << std::setprecision(4)
                                  << (pool_exec_ms / 1000.0) << " s\n";
                    }
                }
            });
        }
        for (auto& th : threads) th.join();

        const auto t_exec1 = clk::now();

        // ---------- Output ----------
        json::object meta;
        meta[ use_events ? "events_file" : "candles_file" ] = candles_path;
        meta["events"]          = static_cast<uint64_t>(events.size());
        meta["threads"]         = static_cast<uint64_t>(n_workers);
        meta["candles_read_ms"] = std::chrono::duration_cast<std::chrono::nanoseconds>(t_read1 - t_read0).count() / 1e6;
        meta["exec_ms"]         = std::chrono::duration_cast<std::chrono::nanoseconds>(t_exec1 - t_exec0).count() / 1e6;

        json::object O;
        O["metadata"] = meta;
        json::array runs; runs.reserve(results.size());
        for (auto& r : results) runs.push_back(r);
        O["runs"] = runs;

        std::ofstream of(out_path);
        of << json::serialize(O) << '\n';
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Arb error: " << e.what() << std::endl;
        return 1;
    }
}
