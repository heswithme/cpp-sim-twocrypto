// Candle-Driven Arbitrage Harness (multi-pool, threaded)
//
// Accepts a JSON file with multiple pool entries and a single candles file.
// Loads and parses the candles once, generates events once, and processes pools
// in parallel using a thread pool. Outputs an aggregated JSON with one result
// per pool entry. Includes per-pool n_rebalances counting and optional action
// capture.

#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include "twocrypto.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <iomanip>
#include <mutex>
#include <vector>
#include <limits>

namespace json = boost::json;

// Select numeric type for pool math and harness at compile time
#if defined(ARB_MODE_F)
using RealT = float;
#elif defined(ARB_MODE_LD)
using RealT = long double;
#else
using RealT = double;
#endif

namespace {
template <typename T>
inline bool differs_rel(T a, T b, T rel = T(1e-12), T abs_eps = T(0)) {
    T da = std::abs(a - b);
    T scale = std::max<T>(T(1), std::max(std::abs(a), std::abs(b)));
    return da > std::max(abs_eps, rel * scale);
}
struct Costs {
    RealT arb_fee_bps{static_cast<RealT>(10.0)};  // basis points
    RealT gas_coin0{static_cast<RealT>(0.0)};     // fixed gas cost in coin0 units (plain units, not 1e18-scaled)
    bool  use_volume_cap{false};
    RealT volume_cap_mult{static_cast<RealT>(1.0)};
};

struct Metrics {
    size_t trades{0};
    RealT  notional{static_cast<RealT>(0)};
    RealT  lp_fee_coin0{static_cast<RealT>(0)};
    RealT  arb_pnl_coin0{static_cast<RealT>(0)};
    size_t n_rebalances{0};
    // Donation stats
    size_t donations{0};
    RealT  donation_coin0_total{static_cast<RealT>(0)};
    std::array<RealT,2> donation_amounts_total{static_cast<RealT>(0), static_cast<RealT>(0)};
};

// Simple stdout guard for cleaner multi-threaded logs
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

static inline std::array<RealT,2> pool_xp_current(
    const twocrypto::TwoCryptoPoolT<RealT>& pool
) {
    return pool_xp_from(pool, pool.balances, pool.cached_price_scale);
}

static inline RealT xp_to_tokens_j(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    size_t j,
    RealT amount_xp,
    RealT price_scale
) {
    RealT v = amount_xp;
    if (j == 1) v = v / price_scale;
    return v / pool.precisions[j];
}

static RealT dyn_fee(const std::array<RealT,2>& xp, RealT mid_fee, RealT out_fee, RealT fee_gamma) {
    RealT Bsum = xp[0] + xp[1];
    if (Bsum <= 0) return mid_fee;
    RealT B = RealT(4) * (xp[0]/Bsum) * (xp[1]/Bsum);
    B = fee_gamma * B / (fee_gamma * B + RealT(1) - B);
    return mid_fee * B + out_fee * (1.0 - B);
}


struct Decision {
    bool do_trade{false};
    int i{0};
    int j{1};
    RealT dx{0.0};
    RealT profit{0.0};
    RealT fee_tokens{0.0};
    RealT notional_coin0{0.0};
};

// --- Decide trade v2: exact ratio pre-filter + Brent sizing on marginal equality ---

// Helper: compute post-trade marginal pool price p_new (coin0 per coin1) and LP fee at post-trade skew
static inline std::pair<RealT, RealT>
post_trade_price_fee(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                     size_t i, size_t j, RealT dx)
{
    using Ops = stableswap::MathOps<RealT>;
    const RealT ps = pool.cached_price_scale;

    // balances after adding dx on i
    auto balances_local = pool.balances;
    balances_local[i] += dx;

    // xp after adding dx (before taking dy from j)
    auto xp = pool_xp_from(pool, balances_local, ps);

    // solve outflow on j and form xp AFTER the trade
    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, j);
    RealT dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    // dynamic LP fee at post-trade skew
    RealT f_lp = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);

    // D and marginal price at post-trade state
    RealT D_new = Ops::newton_D(pool.A, pool.gamma, xp, RealT(0));
    RealT p_new = Ops::get_p(xp, D_new, {pool.A, pool.gamma}) * ps; // coin0 per coin1

    return {p_new, f_lp};
}

// Evaluate dy_after_fee and fee_tokens (no commit)
static inline std::pair<RealT, RealT>
simulate_exchange_once(const twocrypto::TwoCryptoPoolT<RealT>& pool,
                       size_t i, size_t j, RealT dx)
{
    using Ops = stableswap::MathOps<RealT>;
    const RealT ps = pool.cached_price_scale;

    auto balances_local = pool.balances; balances_local[i] += dx;
    auto xp = pool_xp_from(pool, balances_local, ps);
    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, j);
    RealT dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    RealT dy_tokens = xp_to_tokens_j(pool, j, dy_xp, ps);
    RealT f_lp      = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    RealT fee_tokens = f_lp * dy_tokens;
    RealT dy_after   = dy_tokens - fee_tokens;
    return {dy_after, fee_tokens};
}

// Root finding helper: linear-space bisection on [lo, hi] with a sign change.
template <typename F>
static inline bool lin_bisect_root(F&& f,
                                   RealT lo,
                                   RealT hi,
                                   RealT Flo,
                                   RealT Fhi,
                                   int iters,
                                   RealT& out_root) {
    if (!(hi > lo)) return false;
    if (!(Flo * Fhi <= RealT(0))) return false;
    RealT a = lo, b = hi;
    for (int it = 0; it < iters; ++it) {
        RealT m = (a + b) / RealT(2);
        RealT Fm = f(m);
        if (Flo * Fm <= RealT(0)) { b = m; Fhi = Fm; }
        else { a = m; Flo = Fm; }
    }
    out_root = (a + b) / RealT(2);
    return true;
}

// Arb decision routine
static Decision decide_trade(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    RealT cex_price,
    const Costs& costs,
    RealT notional_cap_coin0,
    RealT min_swap_frac,
    RealT max_swap_frac
) {
    Decision d{};
    d.do_trade = false;

    // Sanity checks
    if (!(cex_price > RealT(0))) return d;

    // ---- Exact ratio pre-filter at current state ----
    auto xp_now = pool_xp_current(pool);
    const RealT p_now = stableswap::MathOps<RealT>::get_p(xp_now, pool.D, {pool.A, pool.gamma}) * pool.cached_price_scale;
    const RealT f_lp0 = dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    const RealT f_cex = costs.arb_fee_bps / RealT(1e4);

    // Two "edges" (positive => potentially profitable near dx->0)
    const RealT edge_01 = (RealT(1) - f_cex) * (RealT(1) - f_lp0) * cex_price - p_now;            // 0->1 candidate
    const RealT edge_10 = (RealT(1) - f_lp0) * p_now - (RealT(1) + f_cex) * cex_price;            // 1->0 candidate

    int dir_i = -1, dir_j = -1;
    if (edge_01 <= RealT(0) && edge_10 <= RealT(0)) {
        return d; // no profitable direction even at the margin
    } else if (edge_01 > edge_10) {
        dir_i = 0; dir_j = 1;
    } else {
        dir_i = 1; dir_j = 0;
    }

    // ---- Bounds for dx ----
    const RealT avail = pool.balances[dir_i];
    if (!(avail > RealT(0))) return d;

    RealT dx_lo = std::max<RealT>(RealT(1e-18), avail * std::max<RealT>(RealT(1e-12), min_swap_frac));
    RealT dx_hi = avail * max_swap_frac;
    if (notional_cap_coin0 > RealT(0) && std::isfinite(static_cast<double>(notional_cap_coin0))) {
        if (dir_i == 0) dx_hi = std::min(dx_hi, notional_cap_coin0);
        else            dx_hi = (cex_price > RealT(0)) ? std::min(dx_hi, notional_cap_coin0 / cex_price) : dx_hi;
    }
    if (!(dx_hi > dx_lo)) return d;

    // ---- Residual for the marginal equality ----
    auto residual = [&](RealT dx)->RealT {
        auto [p_new, f_lp] = post_trade_price_fee(pool, static_cast<size_t>(dir_i), static_cast<size_t>(dir_j), dx);
        if (dir_i == 0) {
            // 0->1: p_new = (1 - f_lp) * (1 - f_cex) * p_cex
            RealT rhs = (RealT(1) - f_lp) * (RealT(1) - f_cex) * cex_price;
            return p_new - rhs;
        } else {
            // 1->0: (1 - f_lp) * p_new = (1 + f_cex) * p_cex
            return (RealT(1) - f_lp) * p_new - (RealT(1) + f_cex) * cex_price;
        }
    };

    // Evaluate residual at the bracket
    RealT F_lo = residual(dx_lo);
    RealT F_hi = residual(dx_hi);

    // Expected signs:
    //  - For 0->1 we usually have F_lo < 0 and F_hi may cross to > 0 as dx increases.
    //  - For 1->0 we usually have F_lo > 0 and F_hi may cross to < 0 as dx increases.
    const bool has_root = (F_lo * F_hi < RealT(0));

    RealT dx_star = dx_hi;

    if (has_root) {
        RealT root;
        if (lin_bisect_root(residual, dx_lo, dx_hi, F_lo, F_hi, 20, root)) dx_star = std::max<RealT>(root, dx_lo);
    } else {
        // No sign change. Two cases:
        //  * Entire interval profitable (cannot equalize within caps): pick dx_hi.
        //  * Interval already beyond equality (numerical edge): quick reject.
        // Heuristic: check the "right" sign at dx_lo against expected direction
        if (dir_i == 0) {
            // 0->1 expects F_lo < 0 to be profitable near zero
            if (!(F_lo < RealT(0))) return d; // already beyond; don't trade
        } else {
            // 1->0 expects F_lo > 0 to be profitable near zero
            if (!(F_lo > RealT(0))) return d; // already beyond; don't trade
        }
        // Use cap dx_hi
        dx_star = dx_hi;
    }

    // Final sanity: simulate once to compute dy_after_fee, fees and profit
    auto [dy_after_fee, fee_tokens] = simulate_exchange_once(pool, static_cast<size_t>(dir_i), static_cast<size_t>(dir_j), dx_star);

    RealT profit_coin0 = RealT(0);
    if (dir_i == 0 && dir_j == 1) {
        RealT coin0_from_cex = dy_after_fee * cex_price * (RealT(1) - f_cex);
        profit_coin0 = coin0_from_cex - dx_star - costs.gas_coin0;
    } else {
        RealT coin0_spent_cex = dx_star * cex_price * (RealT(1) + f_cex);
        profit_coin0 = dy_after_fee - coin0_spent_cex - costs.gas_coin0;
    }
    if (!(profit_coin0 > RealT(0))) return d;

    // Populate decision
    d.do_trade = true;
    d.i = dir_i; d.j = dir_j;
    d.dx = dx_star;
    d.fee_tokens = fee_tokens;
    d.profit = profit_coin0;
    d.notional_coin0 = (dir_i == 0) ? dx_star : dx_star * cex_price;
    return d;
}


struct Candle { uint64_t ts; RealT open, high, low, close, volume; };
struct Event  { uint64_t ts; RealT p_cex; RealT volume; };

// Simplified candles loading using Boost.JSON (parses full file once)

static std::vector<Candle> load_candles(const std::string& path, size_t max_candles = 0, double candle_filter_frac = 0.10) {
    std::vector<Candle> out; out.reserve(1024);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open candles file: " + path);
    std::ostringstream oss; oss << in.rdbuf();
    std::string s = oss.str();
    json::value val = json::parse(s);
    if (!val.is_array()) throw std::runtime_error("Candles JSON must be an array of arrays");
    const auto& arr = val.as_array();
    size_t limit = (max_candles > 0) ? std::min<size_t>(arr.size(), max_candles) : arr.size();
    out.reserve(limit);
    for (size_t idx = 0; idx < limit; ++idx) {
        const auto& el = arr[idx];
        if (!el.is_array()) continue;
        const auto& a = el.as_array();
        if (a.size() < 6) continue;
        Candle c{};
        uint64_t ts = 0;
        const auto& tsv = a[0];
        if (tsv.is_uint64()) ts = tsv.as_uint64();
        else if (tsv.is_int64()) ts = static_cast<uint64_t>(tsv.as_int64());
        else if (tsv.is_double()) ts = static_cast<uint64_t>(tsv.as_double());
        if (ts > 10000000000ULL) ts /= 1000ULL;
        c.ts = ts;
        auto to_d = [](const json::value& v)->RealT { return v.is_double()? static_cast<RealT>(v.as_double()) : (v.is_int64()? static_cast<RealT>(v.as_int64()) : RealT(0)); };
        c.open = to_d(a[1]); c.high = to_d(a[2]); c.low = to_d(a[3]); c.close = to_d(a[4]); c.volume = to_d(a[5]);
        // Candle squeeze filter: clamp H/L to within +/-x% of midpoint between O and C
        if (candle_filter_frac > 0.0) {
            RealT oc_mid = static_cast<RealT>(0.5) * (c.open + c.close);
            if (oc_mid > 0) {
                RealT max_h = oc_mid * static_cast<RealT>(1.0 + candle_filter_frac);
                RealT min_l = oc_mid * static_cast<RealT>(1.0 - candle_filter_frac);
                if (c.high > max_h) c.high = max_h;
                if (c.low  < min_l) c.low  = min_l;
            }
        }
        out.push_back(c);
    }
    return out;
}

static std::vector<Event> gen_events(const std::vector<Candle>& cs) {
    std::vector<Event> evs; evs.reserve(cs.size()*2);
    for (const auto& c : cs) {
        RealT path1 = std::abs(c.open - c.low) + std::abs(c.high - c.close);
        RealT path2 = std::abs(c.open - c.high) + std::abs(c.low - c.close);
        bool first_low = path1 < path2;
        evs.push_back(Event{c.ts + 5,  first_low ? c.low  : c.high, c.volume/RealT(2)});
        evs.push_back(Event{c.ts + 15, first_low ? c.high : c.low,  c.volume/RealT(2)});
    }
    std::sort(evs.begin(), evs.end(), [](const Event& a, const Event& b){ return a.ts < b.ts; });
    return evs;
}

template <typename T>
static std::string to_str_1e18(T v) {
    // Use long double for string scaling to avoid precision loss in the textual output only.
    long double scaled = static_cast<long double>(v) * 1e18L;
    if (scaled < 0) scaled = 0;
    std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << scaled; return oss.str();
}

static json::object pool_state_json(const twocrypto::TwoCryptoPoolT<RealT>& p) {
    json::object o;
    o["balances"]     = json::array{to_str_1e18(p.balances[0]), to_str_1e18(p.balances[1])};
    o["xp"]           = json::array{to_str_1e18(p.balances[0] * p.precisions[0]), to_str_1e18(p.balances[1] * p.precisions[1] * p.cached_price_scale)};
    o["D"]            = to_str_1e18(p.D);
    o["virtual_price"] = to_str_1e18(p.virtual_price);
    o["xcp_profit"]    = to_str_1e18(p.xcp_profit);
    o["price_scale"]    = to_str_1e18(p.cached_price_scale);
    o["price_oracle"]   = to_str_1e18(p.cached_price_oracle);
    o["last_prices"]    = to_str_1e18(p.last_prices);
    o["totalSupply"]    = to_str_1e18(p.totalSupply);
    o["timestamp"]      = p.block_timestamp;
    return o;
}

static RealT parse_scaled_1e18(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr) / 1e18L);
    if (v.is_double()) return static_cast<RealT>(v.as_double() / 1e18);
    if (v.is_int64())  return static_cast<RealT>(static_cast<long double>(v.as_int64()) / 1e18L);
    if (v.is_uint64()) return static_cast<RealT>(static_cast<long double>(v.as_uint64()) / 1e18L);
    return RealT(0);
}
static RealT parse_fee_1e10(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr) / 1e10L);
    if (v.is_double()) return static_cast<RealT>(v.as_double() / 1e10);
    if (v.is_int64())  return static_cast<RealT>(static_cast<long double>(v.as_int64()) / 1e10L);
    if (v.is_uint64()) return static_cast<RealT>(static_cast<long double>(v.as_uint64()) / 1e10L);
    return RealT(0);
}
static RealT parse_plain_real(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr));
    if (v.is_double()) return static_cast<RealT>(v.as_double());
    if (v.is_int64())  return static_cast<RealT>(v.as_int64());
    if (v.is_uint64()) return static_cast<RealT>(v.as_uint64());
    return RealT(0);
}

struct PoolInit {
    std::array<RealT,2> precisions{RealT(1),RealT(1)};
    RealT A{static_cast<RealT>(100000.0)};
    RealT gamma{static_cast<RealT>(0.0)};
    RealT mid_fee{static_cast<RealT>(3e-4)};
    RealT out_fee{static_cast<RealT>(5e-4)};
    RealT fee_gamma{static_cast<RealT>(0.23)};
    RealT allowed_extra{static_cast<RealT>(1e-3)};
    RealT adj_step{static_cast<RealT>(1e-3)};
    RealT ma_time{static_cast<RealT>(600.0)};
    RealT initial_price{static_cast<RealT>(1.0)};
    std::array<RealT,2> initial_liq{static_cast<RealT>(1e6), static_cast<RealT>(1e6)};
    uint64_t start_ts{0};
    // Donation controls (harness-only)
    RealT donation_apy{static_cast<RealT>(0.0)};        // plain fraction per year, e.g., 0.05 = 5%
    RealT donation_frequency{static_cast<RealT>(0.0)};  // seconds between donations
    RealT donation_coins_ratio{static_cast<RealT>(0.5)}; // fraction of donation in coin1 (0=all coin0, 1=all coin1)
};

static void parse_pool_entry(const json::object& entry, PoolInit& out_pool, Costs& out_costs, json::object& echo_pool, json::object& echo_costs) {
    json::object pool = entry.contains("pool") ? entry.at("pool").as_object() : entry;
    echo_pool = pool;
    if (auto* v = pool.if_contains("initial_liquidity")) {
        auto arr = v->as_array();
        out_pool.initial_liq[0] = parse_scaled_1e18(arr[0]);
        out_pool.initial_liq[1] = parse_scaled_1e18(arr[1]);
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
    // Donation params (plain fraction for APY; frequency in seconds)
    if (auto* v = pool.if_contains("donation_apy")) out_pool.donation_apy = parse_plain_real(*v);
    if (auto* v = pool.if_contains("donation_frequency")) out_pool.donation_frequency = parse_plain_real(*v);
    if (auto* v = pool.if_contains("donation_coins_ratio")) {
        RealT r = parse_plain_real(*v);
        if (!(r >= RealT(0))) r = RealT(0);
        if (r > RealT(1)) r = RealT(1);
        out_pool.donation_coins_ratio = r;
    }
    if (auto* c = entry.if_contains("costs")) {
        echo_costs = c->as_object();
        auto co = c->as_object();
        if (auto* v = co.if_contains("arb_fee_bps")) out_costs.arb_fee_bps = parse_plain_real(*v);
        // gas_coin0 is now expected in plain coin0 units (not 1e18-scaled)
        if (auto* v = co.if_contains("gas_coin0")) out_costs.gas_coin0 = parse_plain_real(*v);
        if (auto* v = co.if_contains("use_volume_cap")) out_costs.use_volume_cap = v->as_bool();
        if (auto* v = co.if_contains("volume_cap_mult")) out_costs.volume_cap_mult = parse_plain_real(*v);
    } else {
        echo_costs = json::object{};
    }
}

} // namespace

int main(int argc, char* argv[]) {
    // Ensure immediate flushing of stdout for progress logs
    std::cout.setf(std::ios::unitbuf);
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <candles.json> <output.json> [--n-candles N] [--save-actions] [--min-swap F] [--max-swap F] [--threads N|-n N] [--candle-filter PCT]" << std::endl; return 1;
    }
    std::string pools_path = argv[1];
    std::string candles_path = argv[2];
    std::string out_path = argv[3];
    size_t max_candles = 0;
    bool save_actions = false;
    double min_swap_frac = 1e-6;
    double max_swap_frac = 1.0;
    size_t n_workers = std::max<size_t>(1, std::thread::hardware_concurrency());
    double candle_filter_pct = 10.0; // default 10% around (O+C)/2
    uint64_t dustswapfreq_s = 3600; // seconds between dust swaps when no arb trade
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n-candles" && (i+1) < argc) { try { long long v = std::stoll(argv[++i]); if (v > 0) max_candles = static_cast<size_t>(v);} catch(...){} }
        else if (arg == "--save-actions") save_actions = true;
        else if (arg == "--min-swap" && (i+1) < argc) { try { double f = std::stod(argv[++i]); if (f > 0.0 && f <= 1.0) min_swap_frac = f; } catch(...){} }
        else if (arg == "--max-swap" && (i+1) < argc) { try { double f = std::stod(argv[++i]); if (f > 0.0 && f <= 1.0) max_swap_frac = f; } catch(...){} }
        else if ((arg == "--threads" || arg == "-n") && (i+1) < argc) { try { long long v = std::stoll(argv[++i]); if (v > 0) n_workers = static_cast<size_t>(v); } catch(...){} }
        else if (arg == "--dustswapfreq" && (i+1) < argc) { try { long long v = std::stoll(argv[++i]); if (v >= 0) dustswapfreq_s = static_cast<uint64_t>(v); } catch(...){} }
        else if (arg == "--candle-filter" && (i+1) < argc) { try { double p = std::stod(argv[++i]); if (p >= 0.0) candle_filter_pct = p; } catch(...){} }
    }

    try {
        using clk = std::chrono::high_resolution_clock;
        auto t_read0 = clk::now();
        auto candles = load_candles(candles_path, max_candles, candle_filter_pct / 100.0);
        auto events  = gen_events(candles);
        auto t_read1 = clk::now();

        std::ifstream in(pools_path); if (!in) throw std::runtime_error("Cannot open pools json: " + pools_path);
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto root = json::parse(s);
        std::vector<json::object> pool_entries;
        if (root.is_object()) {
            auto obj = root.as_object();
            if (obj.contains("pools")) {
                for (auto& v : obj.at("pools").as_array()) pool_entries.push_back(v.as_object());
            } else if (obj.contains("pool")) {
                pool_entries.push_back(obj);
            } else {
                throw std::runtime_error("Invalid pools json: expected 'pools' array or single 'pool'");
            }
        } else if (root.is_array()) {
            for (auto& v : root.as_array()) pool_entries.push_back(v.as_object());
        } else {
            throw std::runtime_error("Invalid pools json root type");
        }

        struct Job { size_t idx; json::object entry; };
        std::vector<Job> jobs; jobs.reserve(pool_entries.size());
        for (size_t i = 0; i < pool_entries.size(); ++i) jobs.push_back(Job{i, pool_entries[i]});

        std::vector<json::object> results(pool_entries.size());
        std::atomic<size_t> next{0};
        auto t_exec0 = clk::now();
        std::vector<std::thread> threads;
        threads.reserve(n_workers);

        for (size_t t = 0; t < n_workers; ++t) {
            threads.emplace_back([&]() {
                while (true) {
                    size_t idx = next.fetch_add(1);
                    if (idx >= jobs.size()) break;
                    {
                        std::lock_guard<std::mutex> lk(io_mu);
                        std::cout << "dispatch job " << (idx + 1) << "/" << jobs.size() << std::endl;
                    }
                    const auto& entry = jobs[idx].entry;

                    json::object echo_pool, echo_costs;
                    Costs costs{}; PoolInit cfg{};
                    parse_pool_entry(entry, cfg, costs, echo_pool, echo_costs);

                    using Pool = twocrypto::TwoCryptoPoolT<RealT>;
                    Pool pool({static_cast<RealT>(cfg.precisions[0]), static_cast<RealT>(cfg.precisions[1])},
                              static_cast<RealT>(cfg.A), static_cast<RealT>(cfg.gamma),
                              static_cast<RealT>(cfg.mid_fee), static_cast<RealT>(cfg.out_fee), static_cast<RealT>(cfg.fee_gamma),
                              static_cast<RealT>(cfg.allowed_extra), static_cast<RealT>(cfg.adj_step), static_cast<RealT>(cfg.ma_time),
                              static_cast<RealT>(cfg.initial_price));
                    // Initialize timestamp baseline so EMA can progress (before any liquidity is added)
                    uint64_t init_ts = 0;
                    if (cfg.start_ts != 0) init_ts = cfg.start_ts;
                    else if (!events.empty()) init_ts = events.front().ts;
                    if (init_ts != 0) pool.set_block_timestamp(init_ts);
                    (void)pool.add_liquidity({static_cast<RealT>(cfg.initial_liq[0]), static_cast<RealT>(cfg.initial_liq[1])}, RealT(0));
                    // Initial TVL in coin0 and initial price scale snapshot
                    const RealT ps_init   = pool.cached_price_scale;
                    const RealT tvl_start = pool.balances[0] + pool.balances[1] * ps_init;

                    Metrics m{};
                    json::array actions;
                    // Optional per-action state snapshots to aid parity debugging
                    json::array states;
                    // Donation scheduler
                    const RealT SECONDS_PER_YEAR = static_cast<RealT>(365.0 * 86400.0);
                    bool donations_enabled = (cfg.donation_apy > RealT(0)) && (cfg.donation_frequency > RealT(0));
                    uint64_t next_donation_ts = 0;
                    if (donations_enabled && !events.empty()) {
                        uint64_t base_ts = (cfg.start_ts != 0) ? cfg.start_ts : events.front().ts;
                        next_donation_ts = base_ts + static_cast<uint64_t>(cfg.donation_frequency);
                    }
                    // Pre-compute fixed donation amounts per period based on initial TVL and coin ratio
                    RealT donate_amt0_per_period = RealT(0);
                    RealT donate_amt1_per_period = RealT(0);
                    if (donations_enabled) {
                        RealT per_period_coin0 = tvl_start * cfg.donation_frequency * cfg.donation_apy / SECONDS_PER_YEAR;       // coin0 units per donation
                        RealT r1 = cfg.donation_coins_ratio; // fraction in coin1
                        if (r1 < RealT(0)) r1 = RealT(0);
                        if (r1 > RealT(1)) r1 = RealT(1);
                        RealT r0 = RealT(1) - r1;             // fraction in coin0
                        donate_amt0_per_period = r0 * per_period_coin0;                                  // token0 amount
                        donate_amt1_per_period = (ps_init > RealT(0)) ? (r1 * per_period_coin0 / ps_init) // token1 amount
                                                                      : RealT(0);
                    }
                    auto t_pool0 = clk::now();
                    uint64_t last_dust_ts = 0;
                    if (save_actions) {
                        // Capture initial state snapshot
                        states.push_back(pool_state_json(pool));
                    }
                    for (const auto& ev : events) {
                        pool.set_block_timestamp(ev.ts);
                        if (save_actions) {
                            // Snapshot after time update (before any donations/trades at this timestamp)
                            states.push_back(pool_state_json(pool));
                        }
                        // Apply any due donations before trading
                        if (donations_enabled) {
                            while (next_donation_ts != 0 && ev.ts >= next_donation_ts) {
                                // Use fixed per-period amounts based on initial TVL and chosen ratio
                                RealT ps = pool.cached_price_scale; // current price for coin0-equivalent accounting
                                RealT tvl_coin0 = pool.balances[0] + pool.balances[1] * ps;
                                // Approximate minted share ratio by coin0-equivalent / TVL (cap precheck)
                                RealT donate_coin0_equiv = donate_amt0_per_period + donate_amt1_per_period * ps;
                                RealT approx_ratio = (tvl_coin0 > RealT(0)) ? (donate_coin0_equiv / tvl_coin0) : RealT(0);
                                if (approx_ratio > pool.donation_shares_max_ratio) {
                                    next_donation_ts += static_cast<uint64_t>(cfg.donation_frequency);
                                    if (next_donation_ts <= ev.ts) next_donation_ts = ev.ts + 1;
                                    continue;
                                }
                                if (donate_amt0_per_period > RealT(0) || donate_amt1_per_period > RealT(0)) {
                                    RealT amt0 = donate_amt0_per_period;
                                    RealT amt1 = donate_amt1_per_period;
                                    try {
                                        RealT price_scale_before = pool.cached_price_scale;
                                        RealT minted = pool.add_liquidity({amt0, amt1}, RealT(0), true);
                                        RealT price_scale_after = pool.cached_price_scale;
                                        m.donations += 1;
                                        m.donation_coin0_total += donate_coin0_equiv;
                                        m.donation_amounts_total[0] += amt0;
                                        m.donation_amounts_total[1] += amt1;
                                        if (differs_rel(price_scale_after, price_scale_before)) m.n_rebalances += 1;
                                        if (save_actions) {
                                            json::object da;
                                            da["type"] = "donation";
                                            // Record both the pool timestamp (when donation executed)
                                            // and the scheduled due time for reference
                                            da["ts"] = ev.ts;               // actual pool timestamp
                                            da["ts_due"] = next_donation_ts; // scheduled donation time
                                            da["amounts"] = json::array{static_cast<double>(amt0), static_cast<double>(amt1)};
                                            da["amounts_wei"] = json::array{to_str_1e18(amt0), to_str_1e18(amt1)};
                                            da["minted"] = static_cast<double>(minted);
                                            da["tvl_coin0_before"] = static_cast<double>(tvl_coin0);
                                            da["price_scale"] = static_cast<double>(ps);
                                            da["donation_coins_ratio"] = static_cast<double>(cfg.donation_coins_ratio);
                                            actions.push_back(da);
                                        }
                                        if (save_actions) {
                                            // Snapshot state after donation commit
                                            states.push_back(pool_state_json(pool));
                                        }
                                    } catch (...) {
                                        // donation failed (e.g., cap). Skip silently.
                                    }
                                }
                                next_donation_ts += static_cast<uint64_t>(cfg.donation_frequency);
                                if (next_donation_ts <= ev.ts) {
                                    // avoid stuck if frequency ridiculously small
                                    next_donation_ts = ev.ts + 1;
                                }
                            }
                        }
                        RealT pre_p_pool = pool.get_p();
                        RealT notional_cap_coin0 = std::numeric_limits<RealT>::infinity();
                        if (costs.use_volume_cap) notional_cap_coin0 = ev.volume * ev.p_cex * static_cast<RealT>(costs.volume_cap_mult);
                        Decision d = decide_trade(pool, ev.p_cex, costs, notional_cap_coin0, static_cast<RealT>(min_swap_frac), static_cast<RealT>(max_swap_frac));
                        if (!d.do_trade) {
                            // Respect cooldown for tick update (zero-cost EMA/price tweak)
                            bool can_tick = (dustswapfreq_s > 0) && (last_dust_ts == 0 || ev.ts >= last_dust_ts + dustswapfreq_s);
                            if (!can_tick) continue;
                            try {
                                RealT price_scale_before = pool.cached_price_scale;
                                RealT p_pool_before = pre_p_pool;
                                pool.tick();
                                RealT price_scale_after = pool.cached_price_scale;
                                if (differs_rel(price_scale_after, price_scale_before)) m.n_rebalances += 1;
                                if (save_actions) {
                                    json::object tr;
                                    tr["type"] = "tick";
                                    tr["ts"] = ev.ts;
                                    tr["p_cex"] = static_cast<double>(ev.p_cex);
                                    tr["p_pool_before"] = static_cast<double>(p_pool_before);
                                    actions.push_back(tr);
                                    states.push_back(pool_state_json(pool));
                                }
                                last_dust_ts = ev.ts;
                            } catch (...) { /* ignore */ }
                            continue;
                        }
                        try {
                            RealT price_scale_before = pool.cached_price_scale;
                            auto res = pool.exchange((RealT)d.i, (RealT)d.j, d.dx, RealT(0));
                            RealT dy_after_fee = res[0];
                            RealT fee_tokens   = res[1];
                            // Read the committed value directly from the pool after exchange
                            RealT price_scale_after = pool.cached_price_scale;
                            m.trades   += 1;
                            m.notional += d.notional_coin0;
                            m.lp_fee_coin0 += (d.j==1 ? fee_tokens * ev.p_cex : fee_tokens);
                            if (differs_rel(price_scale_after, price_scale_before)) m.n_rebalances += 1;
                            RealT profit_coin0 = d.profit;
                            m.arb_pnl_coin0 += profit_coin0;
                            if (save_actions) {
                                json::object tr;
                                tr["type"] = "exchange";
                                tr["ts"] = ev.ts; tr["i"] = d.i; tr["j"] = d.j; tr["dx"] = static_cast<double>(d.dx); tr["dx_wei"] = to_str_1e18(d.dx); tr["dx_wei"] = to_str_1e18(d.dx);
                                tr["dy_after_fee"] = static_cast<double>(dy_after_fee); tr["fee_tokens"] = static_cast<double>(fee_tokens);
                                tr["profit_coin0"] = static_cast<double>(profit_coin0); tr["p_cex"] = static_cast<double>(ev.p_cex); tr["p_pool_before"] = static_cast<double>(pre_p_pool);
                                actions.push_back(tr);
                                // Snapshot state after trade
                                states.push_back(pool_state_json(pool));
                            }
                        } catch (...) {}
                    }
                    auto t_pool1 = clk::now();
                    double pool_exec_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t_pool1 - t_pool0).count() / 1e6;

                    json::object summary;
                    summary["events"] = events.size();
                    summary["trades"] = m.trades;
                    summary["total_notional_coin0"] = static_cast<double>(m.notional);
                    summary["lp_fee_coin0"] = static_cast<double>(m.lp_fee_coin0);
                    summary["arb_pnl_coin0"] = static_cast<double>(m.arb_pnl_coin0);
                    summary["n_rebalances"] = m.n_rebalances;
                    summary["donations"] = m.donations;
                    summary["donation_coin0_total"] = static_cast<double>(m.donation_coin0_total);
                    summary["donation_amounts_total"] = json::array{static_cast<double>(m.donation_amounts_total[0]), static_cast<double>(m.donation_amounts_total[1])};
                    summary["pool_exec_ms"] = pool_exec_ms;
                    // APY over the run (TVL-based, compounded) and traditional VP-based APY
                    uint64_t t_start = events.empty() ? init_ts : events.front().ts;
                    uint64_t t_end   = events.empty() ? init_ts : events.back().ts;
                    double duration_s = (t_end > t_start) ? double(t_end - t_start) : 0.0;
                    const RealT tvl_end = pool.balances[0] + pool.balances[1] * pool.cached_price_scale;
                    double apy_coin0 = 0.0;
                    double apy_donation_coin0 = 0.0;
                    double apy_vp = 0.0;
                    if (duration_s > 0.0 && tvl_start > RealT(0)) {
                        double exponent = SECONDS_PER_YEAR / duration_s;
                        // Traditional VP-based APY (vp_start = 1.0)
                        long double vp_end = static_cast<long double>(pool.get_virtual_price());
                        if (vp_end > 0.0) apy_vp = std::pow(vp_end, exponent) - 1.0; else apy_vp = -1.0;

                        // Coin0 TVL-based APY
                        if (tvl_end > RealT(0)) apy_coin0 = std::pow(static_cast<double>(tvl_end / tvl_start), exponent) - 1.0; else apy_coin0 = -1.0;
                        double tvl_end_adj = static_cast<double>(tvl_end) - static_cast<double>(m.donation_coin0_total);
                        if (tvl_end_adj > 0.0) apy_donation_coin0 = std::pow(tvl_end_adj / tvl_start, exponent) - 1.0; else apy_donation_coin0 = -1.0;
                    }
                    summary["t_start"] = t_start;
                    summary["t_end"] = t_end;
                    summary["duration_s"] = duration_s;
                    summary["tvl_coin0_start"] = static_cast<double>(tvl_start);
                    summary["tvl_coin0_end"] = static_cast<double>(tvl_end);
                    summary["apy"] = apy_vp;
                    summary["apy_coin0"] = apy_coin0;
                    summary["apy_donation_coin0"] = apy_donation_coin0;

                    json::object params;
                    params["pool"] = echo_pool;
                    if (!echo_costs.empty()) params["costs"] = echo_costs;

                    json::object out;
                    out["result"] = summary;
                    out["params"] = params;
                    out["final_state"] = pool_state_json(pool);
                    if (save_actions) {
                        out["actions"] = actions;
                        out["states"] = states;
                    }
                    results[idx] = std::move(out);
                    {
                        std::lock_guard<std::mutex> lk(io_mu);
                        std::cout << "finished job " << (idx + 1) << "/" << jobs.size()
                                  << ", time: " << std::fixed << std::setprecision(4)
                                  << (pool_exec_ms / 1000.0) << " s" << std::endl;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        auto t_exec1 = clk::now();

        json::object meta;
        meta["candles_file"] = candles_path;
        meta["events"] = events.size();
        meta["threads"] = static_cast<uint64_t>(n_workers);
        meta["candles_read_ms"] = std::chrono::duration_cast<std::chrono::nanoseconds>(t_read1 - t_read0).count() / 1e6;
        meta["exec_ms"] = std::chrono::duration_cast<std::chrono::nanoseconds>(t_exec1 - t_exec0).count() / 1e6;

        json::object O;
        O["metadata"] = meta;
        json::array runs;
        runs.reserve(results.size());
        for (auto& r : results) runs.push_back(r);
        O["runs"] = runs;

        std::ofstream of(out_path);
        of << json::serialize(O) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Arb error: " << e.what() << std::endl;
        return 1;
    }
}
