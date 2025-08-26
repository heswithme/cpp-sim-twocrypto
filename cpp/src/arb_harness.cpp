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
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <iomanip>
#include <mutex>
#include <unistd.h>
#include <vector>

namespace json = boost::json;

// Select numeric type for pool math at compile time
#if defined(ARB_MODE_F)
using RealT = float;
#elif defined(ARB_MODE_LD)
using RealT = long double;
#else
using RealT = double;
#endif

namespace {
template <typename T>
inline bool differs_rel(T a, T b, long double rel = 1e-15L, long double abs_eps = 0.0L) {
    long double da = std::fabsl(static_cast<long double>(a - b));
    long double scale = std::max< long double >(1.0L, std::max(std::fabsl(static_cast<long double>(a)), std::fabsl(static_cast<long double>(b))));
    return da > std::max(abs_eps, rel * scale);
}
struct Costs {
    double arb_fee_bps{10.0};
    double gas_coin0{0.0};
    bool   use_volume_cap{false};
    double volume_cap_mult{1.0};
};

struct Metrics {
    size_t trades{0};
    double notional{0};
    double lp_fee_coin0{0};
    double arb_pnl_coin0{0};
    size_t n_rebalances{0};
    // Donation stats
    size_t donations{0};
    double donation_coin0_total{0.0};
    std::array<long double,2> donation_amounts_total{0.0L, 0.0L};
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

static RealT spot_price(const twocrypto::TwoCryptoPoolT<RealT>& pool) {
    std::array<RealT,2> xp = pool_xp_current(pool);
    std::array<RealT,2> A_gamma{ pool.A, pool.gamma };
    return stableswap::MathOps<RealT>::get_p(xp, pool.D, A_gamma) * pool.cached_price_scale;
}

struct Decision {
    bool do_trade{false};
    int i{0};
    int j{1};
    RealT dx{0.0};
    long double profit{0.0};
    RealT fee_tokens{0.0};
    RealT notional_coin0{0.0};
};

static std::optional<std::pair<RealT,long double>> simulate_profit(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    int i, int j,
    RealT dx,
    RealT cex_price,
    const Costs& costs,
    RealT& out_fee_tokens
) {
    using Ops = stableswap::MathOps<RealT>;
    const RealT price_scale = pool.cached_price_scale;

    std::array<RealT,2> balances = pool.balances; balances[i] += dx;
    std::array<RealT,2> xp = pool_xp_from(pool, balances, price_scale);

    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, static_cast<size_t>(j));
    RealT dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    RealT dy_tokens = xp_to_tokens_j(pool, static_cast<size_t>(j), dy_xp, price_scale);
    RealT fee_rate = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    RealT fee_tokens = fee_rate * dy_tokens;
    RealT dy_after_fee = dy_tokens - fee_tokens;

    long double profit_coin0 = 0.0L;
    if (i == 0 && j == 1) {
        long double coin0_out_cex = static_cast<long double>(dy_after_fee * cex_price);
        long double arb_fee = static_cast<long double>((costs.arb_fee_bps/1e4) * coin0_out_cex);
        profit_coin0 = coin0_out_cex - static_cast<long double>(dx) - arb_fee - static_cast<long double>(costs.gas_coin0);
    } else {
        long double coin0_spent_cex = static_cast<long double>(dx * cex_price);
        long double arb_fee = static_cast<long double>((costs.arb_fee_bps/1e4) * coin0_spent_cex);
        profit_coin0 = static_cast<long double>(dy_after_fee) - coin0_spent_cex - arb_fee - static_cast<long double>(costs.gas_coin0);
    }
    if (profit_coin0 <= 0.0) return std::nullopt;
    out_fee_tokens = fee_tokens;
    return std::make_pair(dy_after_fee, profit_coin0);
}

// Decide direction by probing both sides with a fixed small fraction of balance.
static bool decide_trade_direction(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    RealT cex_price,
    const Costs& costs,
    RealT probe_frac,
    int& out_i,
    int& out_j,
    RealT& out_fee_tokens,
    long double& out_profit
) {
    bool found = false;
    out_profit = 0.0L; out_fee_tokens = 0.0; out_i = 0; out_j = 1;
    for (int i = 0; i < 2; ++i) {
        int j = 1 - i;
        RealT avail = pool.balances[i];
        if (!(avail > 0)) continue;
        RealT dx = avail * probe_frac;
        if (!(dx > 0)) continue;
        RealT fee_tmp = 0.0;
        auto sim = simulate_profit(pool, i, j, dx, cex_price, costs, fee_tmp);
        if (!sim) continue;
        long double profit = sim->second;
        if (!found || profit > out_profit) {
            found = true;
            out_profit = profit;
            out_fee_tokens = fee_tmp;
            out_i = i; out_j = j;
        }
    }
    return found;
}

// Decide trade size via binary search within min/max swap fraction bounds (and optional notional cap).
static bool decide_trade_size(
    const twocrypto::TwoCryptoPoolT<RealT>& pool,
    RealT cex_price,
    const Costs& costs,
    int i, int j,
    RealT notional_cap_coin0,
    RealT min_swap_frac,
    RealT max_swap_frac,
    RealT& out_dx,
    RealT& out_fee_tokens,
    long double& out_profit
) {
    RealT available = pool.balances[i];
    if (!(available > 0)) return false;

    RealT dx_lo = std::max<RealT>(RealT(1e-18), available * std::max<RealT>(RealT(1e-12), min_swap_frac));
    RealT dx_hi = available * max_swap_frac;
    if (costs.use_volume_cap) {
        if (i == 0) dx_hi = std::min(dx_hi, notional_cap_coin0);
        else if (cex_price > 0) dx_hi = std::min(dx_hi, notional_cap_coin0 / cex_price);
    }
    if (!(dx_hi > 0)) return false;
    if (dx_lo > dx_hi) dx_lo = dx_hi;

    RealT best_dx = 0.0, best_fee = 0.0; long double best_profit = -1e100L;

    // Evaluate lo and hi
    RealT fee_lo = 0.0, fee_hi = 0.0;
    auto lo = simulate_profit(pool, i, j, dx_lo, cex_price, costs, fee_lo);
    auto hi = simulate_profit(pool, i, j, dx_hi, cex_price, costs, fee_hi);
    if (lo && lo->second > best_profit) { best_profit = lo->second; best_dx = dx_lo; best_fee = fee_lo; }
    if (hi && hi->second > best_profit) { best_profit = hi->second; best_dx = dx_hi; best_fee = fee_hi; }

    RealT L = dx_lo, R = dx_hi;
    for (int it = 0; it < 40; ++it) {
        RealT mid = RealT(0.5) * (L + R);
        RealT fee_mid = 0.0;
        auto m = simulate_profit(pool, i, j, mid, cex_price, costs, fee_mid);
        if (m) {
            L = mid;
            if (m->second > best_profit) { best_profit = m->second; best_dx = mid; best_fee = fee_mid; }
        } else {
            R = mid;
        }
        if (R - L <= std::max<RealT>(RealT(1e-12), available * RealT(1e-12))) break;
    }

    if (!(best_profit > 0)) return false;
    out_dx = best_dx;
    out_fee_tokens = best_fee;
    out_profit = best_profit;
    return true;
}

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

    // 1) Decide direction by probing both sides with a fixed small fraction
    const RealT PROBE_FRAC = RealT(1e-9);
    int i = 0, j = 1; RealT probe_fee = 0.0; long double probe_profit = 0.0L;
    if (!decide_trade_direction(pool, cex_price, costs, PROBE_FRAC, i, j, probe_fee, probe_profit)) {
        return d; // no profitable direction
    }

    // 2) Decide size using binary search within bounds
    RealT dx = 0.0, fee_tokens = 0.0; long double profit = 0.0L;
    if (!decide_trade_size(pool, cex_price, costs, i, j, notional_cap_coin0, min_swap_frac, max_swap_frac, dx, fee_tokens, profit)) {
        return d; // size selection found no profitable dx within bounds
    }

    d.do_trade = true;
    d.i = i; d.j = j;
    d.dx = dx;
    d.fee_tokens = fee_tokens;
    d.profit = profit;
    d.notional_coin0 = (i == 0) ? dx : dx * cex_price;
    return d;
}

struct Candle { uint64_t ts; RealT open, high, low, close, volume; };
struct Event  { uint64_t ts; RealT p_cex; RealT volume; };

// Simplified candles loading using Boost.JSON (parses full file once)

static std::vector<Candle> load_candles(const std::string& path, size_t max_candles = 0) {
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

static long double parse_scaled_1e18(const json::value& v) {
    if (v.is_string()) return std::strtold(v.as_string().c_str(), nullptr) / 1e18L;
    if (v.is_double()) return static_cast<long double>(v.as_double()) / 1e18L;
    if (v.is_int64())  return static_cast<long double>(v.as_int64()) / 1e18L;
    if (v.is_uint64()) return static_cast<long double>(v.as_uint64()) / 1e18L;
    return 0.0L;
}
static long double parse_fee_1e10(const json::value& v) {
    if (v.is_string()) return std::strtold(v.as_string().c_str(), nullptr) / 1e10L;
    if (v.is_double()) return static_cast<long double>(v.as_double()) / 1e10L;
    if (v.is_int64())  return static_cast<long double>(v.as_int64()) / 1e10L;
    if (v.is_uint64()) return static_cast<long double>(v.as_uint64()) / 1e10L;
    return 0.0L;
}
static long double parse_plain_double(const json::value& v) {
    if (v.is_string()) return std::strtold(v.as_string().c_str(), nullptr);
    if (v.is_double()) return static_cast<long double>(v.as_double());
    if (v.is_int64())  return static_cast<long double>(v.as_int64());
    if (v.is_uint64()) return static_cast<long double>(v.as_uint64());
    return 0.0L;
}

struct PoolInit {
    std::array<long double,2> precisions{1.0L,1.0L};
    long double A{100000.0L};
    long double gamma{0.0L};
    long double mid_fee{3e-4L};
    long double out_fee{5e-4L};
    long double fee_gamma{0.23L};
    long double allowed_extra{1e-3L};
    long double adj_step{1e-3L};
    long double ma_time{600.0L};
    long double initial_price{1.0L};
    std::array<long double,2> initial_liq{1e6L,1e6L};
    uint64_t start_ts{0};
    // Donation controls (harness-only)
    long double donation_apy{0.0L};              // plain fraction per year, e.g., 0.05 = 5%
    long double donation_frequency{0.0L};        // seconds between donations
};

static void parse_pool_entry(const json::object& entry, PoolInit& out_pool, Costs& out_costs, json::object& echo_pool, json::object& echo_costs) {
    json::object pool = entry.contains("pool") ? entry.at("pool").as_object() : entry;
    echo_pool = pool;
    if (auto* v = pool.if_contains("initial_liquidity")) {
        auto arr = v->as_array();
        out_pool.initial_liq[0] = parse_scaled_1e18(arr[0]);
        out_pool.initial_liq[1] = parse_scaled_1e18(arr[1]);
    }
    if (auto* v = pool.if_contains("A")) out_pool.A = parse_plain_double(*v);
    if (auto* v = pool.if_contains("gamma")) out_pool.gamma = parse_plain_double(*v);
    if (auto* v = pool.if_contains("mid_fee")) out_pool.mid_fee = parse_fee_1e10(*v);
    if (auto* v = pool.if_contains("out_fee")) out_pool.out_fee = parse_fee_1e10(*v);
    if (auto* v = pool.if_contains("fee_gamma")) out_pool.fee_gamma = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("allowed_extra_profit")) out_pool.allowed_extra = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("adjustment_step")) out_pool.adj_step = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("ma_time")) out_pool.ma_time = parse_plain_double(*v);
    if (auto* v = pool.if_contains("initial_price")) out_pool.initial_price = parse_scaled_1e18(*v);
    if (auto* v = pool.if_contains("start_timestamp")) out_pool.start_ts = static_cast<uint64_t>(parse_plain_double(*v));
    // Donation params (plain fraction for APY; frequency in seconds)
    if (auto* v = pool.if_contains("donation_apy")) out_pool.donation_apy = parse_plain_double(*v);
    if (auto* v = pool.if_contains("donation_frequency")) out_pool.donation_frequency = parse_plain_double(*v);
    if (auto* c = entry.if_contains("costs")) {
        echo_costs = c->as_object();
        auto co = c->as_object();
        if (auto* v = co.if_contains("arb_fee_bps")) out_costs.arb_fee_bps = parse_plain_double(*v);
        if (auto* v = co.if_contains("gas_coin0")) out_costs.gas_coin0 = parse_scaled_1e18(*v);
        if (auto* v = co.if_contains("use_volume_cap")) out_costs.use_volume_cap = v->as_bool();
        if (auto* v = co.if_contains("volume_cap_mult")) out_costs.volume_cap_mult = parse_plain_double(*v);
    } else {
        echo_costs = json::object{};
    }
}

} // namespace

int main(int argc, char* argv[]) {
    // Ensure immediate flushing of stdout for progress logs
    std::cout.setf(std::ios::unitbuf);
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <candles.json> <output.json> [--n-candles N] [--save-actions] [--min-swap F] [--max-swap F] [--threads N|-n N]" << std::endl; return 1;
    }
    std::string pools_path = argv[1];
    std::string candles_path = argv[2];
    std::string out_path = argv[3];
    size_t max_candles = 0;
    bool save_actions = false;
    double min_swap_frac = 1e-6;
    double max_swap_frac = 1.0;
    size_t n_workers = std::max<size_t>(1, std::thread::hardware_concurrency());
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n-candles" && (i+1) < argc) { try { long long v = std::stoll(argv[++i]); if (v > 0) max_candles = static_cast<size_t>(v);} catch(...){} }
        else if (arg == "--save-actions") save_actions = true;
        else if (arg == "--min-swap" && (i+1) < argc) { try { double f = std::stod(argv[++i]); if (f > 0.0 && f <= 1.0) min_swap_frac = f; } catch(...){} }
        else if (arg == "--max-swap" && (i+1) < argc) { try { double f = std::stod(argv[++i]); if (f > 0.0 && f <= 1.0) max_swap_frac = f; } catch(...){} }
        else if ((arg == "--threads" || arg == "-n") && (i+1) < argc) { try { long long v = std::stoll(argv[++i]); if (v > 0) n_workers = static_cast<size_t>(v); } catch(...){} }
    }

    try {
        using clk = std::chrono::high_resolution_clock;
        auto t_read0 = clk::now();
        auto candles = load_candles(candles_path, max_candles);
        auto events  = gen_events(candles);
        // // print first 10 candles and events
        // std::cout << "candles:" << std::endl;
        // for (size_t i = 0; i < 10; ++i) {
        //     std::cout << "candle " << i << ": " << candles[i].ts << " " << candles[i].open << " " << candles[i].high << " " << candles[i].low << " " << candles[i].close << " " << candles[i].volume << std::endl;
        // }
        // std::cout << "events:" << std::endl;
        // for (size_t i = 0; i < 10; ++i) {
        //     std::cout << "event " << i << ": " << events[i].ts << " " << events[i].p_cex << " " << events[i].volume << std::endl;
        // }
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
                    // Initial TVL in coin0
                    const double tvl_start = pool.balances[0] + pool.balances[1] * pool.cached_price_scale;

                    Metrics m{};
                    json::array actions;
                    // Optional per-action state snapshots to aid parity debugging
                    json::array states;
                    // Donation scheduler
                    const long double SECONDS_PER_YEAR = 365.0L * 86400.0L;
                    bool donations_enabled = (cfg.donation_apy > 0.0L) && (cfg.donation_frequency > 0.0L);
                    uint64_t next_donation_ts = 0;
                    if (donations_enabled && !events.empty()) {
                        uint64_t base_ts = (cfg.start_ts != 0) ? cfg.start_ts : events.front().ts;
                        next_donation_ts = base_ts + static_cast<uint64_t>(cfg.donation_frequency);
                    }
                    auto t_pool0 = clk::now();
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
                                // Compute donation sized as fraction of current TVL in coin0
                                RealT ps = pool.cached_price_scale;
                                RealT tvl_coin0 = pool.balances[0] + pool.balances[1] * ps;
                                long double frac = static_cast<long double>(cfg.donation_apy * (cfg.donation_frequency / SECONDS_PER_YEAR));
                                // Skip if fraction exceeds cap approximation (avoid revert/noise)
                                if (frac > pool.donation_shares_max_ratio) {
                                    next_donation_ts += static_cast<uint64_t>(cfg.donation_frequency);
                                    if (next_donation_ts <= ev.ts) next_donation_ts = ev.ts + 1;
                                    continue;
                                }
                                RealT donate_coin0 = tvl_coin0 * static_cast<RealT>(frac);
                                if (donate_coin0 > 0.0) {
                                    RealT amt0 = RealT(0.5) * donate_coin0;
                                    RealT amt1 = (RealT(0.5) * donate_coin0) / (ps > 0.0 ? ps : RealT(1));
                                    try {
                                        RealT price_scale_before = pool.cached_price_scale;
                                        RealT minted = pool.add_liquidity({amt0, amt1}, RealT(0), true);
                                        RealT price_scale_after = pool.cached_price_scale;
                                        m.donations += 1;
                                        m.donation_coin0_total += static_cast<double>(donate_coin0);
                                        m.donation_amounts_total[0] += static_cast<long double>(amt0);
                                        m.donation_amounts_total[1] += static_cast<long double>(amt1);
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
                        RealT pre_p_pool = spot_price(pool);
                        RealT notional_cap_coin0 = std::numeric_limits<RealT>::infinity();
                        if (costs.use_volume_cap) notional_cap_coin0 = ev.volume * ev.p_cex * static_cast<RealT>(costs.volume_cap_mult);
                        Decision d = decide_trade(pool, ev.p_cex, costs, notional_cap_coin0, static_cast<RealT>(min_swap_frac), static_cast<RealT>(max_swap_frac));
                        if (!d.do_trade) continue;
                        try {
                            RealT price_scale_before = pool.cached_price_scale;
                            auto res = pool.exchange((RealT)d.i, (RealT)d.j, d.dx, RealT(0));
                            RealT dy_after_fee = res[0];
                            RealT fee_tokens   = res[1];
                            // Read the committed value directly from the pool after exchange
                            RealT price_scale_after = pool.cached_price_scale;
                            m.trades   += 1;
                            m.notional += static_cast<double>(d.notional_coin0);
                            m.lp_fee_coin0 += static_cast<double>((d.j==1 ? fee_tokens * ev.p_cex : fee_tokens));
                            if (differs_rel(price_scale_after, price_scale_before)) m.n_rebalances += 1;
                            long double profit_coin0 = 0.0L;
                            if (d.i == 0 && d.j == 1) {
                                long double coin0_out_cex = static_cast<long double>(dy_after_fee * ev.p_cex);
                                long double arb_fee = static_cast<long double>((costs.arb_fee_bps/1e4) * coin0_out_cex);
                                profit_coin0 = coin0_out_cex - static_cast<long double>(d.dx) - arb_fee - static_cast<long double>(costs.gas_coin0);
                            } else {
                                long double coin0_spent_cex = static_cast<long double>(d.dx * ev.p_cex);
                                long double arb_fee = static_cast<long double>((costs.arb_fee_bps/1e4) * coin0_spent_cex);
                                profit_coin0 = static_cast<long double>(dy_after_fee) - coin0_spent_cex - arb_fee - static_cast<long double>(costs.gas_coin0);
                            }
                            m.arb_pnl_coin0 += static_cast<double>(profit_coin0);
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
                    summary["total_notional_coin0"] = m.notional;
                    summary["lp_fee_coin0"] = m.lp_fee_coin0;
                    summary["arb_pnl_coin0"] = m.arb_pnl_coin0;
                    summary["n_rebalances"] = m.n_rebalances;
                    summary["donations"] = m.donations;
                    summary["donation_coin0_total"] = m.donation_coin0_total;
                    summary["donation_amounts_total"] = json::array{static_cast<double>(m.donation_amounts_total[0]), static_cast<double>(m.donation_amounts_total[1])};
                    summary["pool_exec_ms"] = pool_exec_ms;
                    // APY over the run (TVL-based, compounded) and traditional VP-based APY
                    uint64_t t_start = events.empty() ? init_ts : events.front().ts;
                    uint64_t t_end   = events.empty() ? init_ts : events.back().ts;
                    double duration_s = (t_end > t_start) ? double(t_end - t_start) : 0.0;
                    const long double tvl_end = static_cast<long double>(pool.balances[0] + pool.balances[1] * pool.cached_price_scale);
                    double apy_coin0 = 0.0;
                    double apy_donation_coin0 = 0.0;
                    double apy_vp = 0.0;
                    if (duration_s > 0.0 && tvl_start > 0.0) {
                        double exponent = SECONDS_PER_YEAR / duration_s;
                        // Traditional VP-based APY (vp_start = 1.0)
                        long double vp_end = static_cast<long double>(pool.get_virtual_price());
                        if (vp_end > 0.0) apy_vp = std::pow(vp_end, exponent) - 1.0; else apy_vp = -1.0;

                        // Coin0 TVL-based APY
                        if (tvl_end > 0.0) apy_coin0 = std::pow(static_cast<double>(tvl_end / tvl_start), exponent) - 1.0; else apy_coin0 = -1.0;
                        double tvl_end_adj = static_cast<double>(tvl_end) - m.donation_coin0_total;
                        if (tvl_end_adj > 0.0) apy_donation_coin0 = std::pow(tvl_end_adj / tvl_start, exponent) - 1.0; else apy_donation_coin0 = -1.0;
                    }
                    summary["t_start"] = t_start;
                    summary["t_end"] = t_end;
                    summary["duration_s"] = duration_s;
                    summary["tvl_coin0_start"] = tvl_start;
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
