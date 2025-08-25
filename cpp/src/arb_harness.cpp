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

namespace {
inline bool differs_rel(double a, double b, double rel = 1e-15, double abs_eps = 0.0) {
    double da = std::abs(a - b);
    double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return da > std::max(abs_eps, rel * scale);
}
struct Costs {
    double arb_fee_bps{10.0};
    double gas_coin0{0.0};
    double max_trade_frac{0.1};
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
    std::array<double,2> donation_amounts_total{0.0, 0.0};
};

// Simple stdout guard for cleaner multi-threaded logs
std::mutex io_mu;

static inline std::array<double,2> pool_xp_from(
    const twocrypto::TwoCryptoPoolT<double>& pool,
    const std::array<double,2>& balances,
    double price_scale
) {
    return {
        balances[0] * pool.precisions[0],
        balances[1] * pool.precisions[1] * price_scale
    };
}

static inline std::array<double,2> pool_xp_current(
    const twocrypto::TwoCryptoPoolT<double>& pool
) {
    return pool_xp_from(pool, pool.balances, pool.cached_price_scale);
}

static inline double xp_to_tokens_j(
    const twocrypto::TwoCryptoPoolT<double>& pool,
    size_t j,
    double amount_xp,
    double price_scale
) {
    double v = amount_xp;
    if (j == 1) v = v / price_scale;
    return v / pool.precisions[j];
}

static double dyn_fee(const std::array<double,2>& xp, double mid_fee, double out_fee, double fee_gamma) {
    double Bsum = xp[0] + xp[1];
    if (Bsum <= 0) return mid_fee;
    double B = 4.0 * (xp[0]/Bsum) * (xp[1]/Bsum);
    B = fee_gamma * B / (fee_gamma * B + 1.0 - B);
    return mid_fee * B + out_fee * (1.0 - B);
}

static double spot_price(const twocrypto::TwoCryptoPoolT<double>& pool) {
    std::array<double,2> xp = pool_xp_current(pool);
    std::array<double,2> A_gamma{ pool.A, pool.gamma };
    return stableswap::MathOps<double>::get_p(xp, pool.D, A_gamma) * pool.cached_price_scale;
}

struct Decision {
    bool do_trade{false};
    int i{0};
    int j{1};
    double dx{0.0};
    double profit{0.0};
    double fee_tokens{0.0};
    double notional_coin0{0.0};
};

static std::optional<std::pair<double,double>> simulate_profit(
    const twocrypto::TwoCryptoPoolT<double>& pool,
    int i, int j,
    double dx,
    double cex_price,
    const Costs& costs,
    double& out_fee_tokens
) {
    using Ops = stableswap::MathOps<double>;
    const double price_scale = pool.cached_price_scale;

    std::array<double,2> balances = pool.balances; balances[i] += dx;
    std::array<double,2> xp = pool_xp_from(pool, balances, price_scale);

    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, static_cast<size_t>(j));
    double dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp;

    double dy_tokens = xp_to_tokens_j(pool, static_cast<size_t>(j), dy_xp, price_scale);
    double fee_rate = dyn_fee(xp, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    double fee_tokens = fee_rate * dy_tokens;
    double dy_after_fee = dy_tokens - fee_tokens;

    double profit_coin0 = 0.0;
    if (i == 0 && j == 1) {
        double coin0_out_cex = dy_after_fee * cex_price;
        double arb_fee = (costs.arb_fee_bps/1e4) * coin0_out_cex;
        profit_coin0 = coin0_out_cex - dx - arb_fee - costs.gas_coin0;
    } else {
        double coin0_spent_cex = dx * cex_price;
        double arb_fee = (costs.arb_fee_bps/1e4) * coin0_spent_cex;
        profit_coin0 = dy_after_fee - coin0_spent_cex - arb_fee - costs.gas_coin0;
    }
    if (profit_coin0 <= 0.0) return std::nullopt;
    out_fee_tokens = fee_tokens;
    return std::make_pair(dy_after_fee, profit_coin0);
}

static Decision decide_trade(
    const twocrypto::TwoCryptoPoolT<double>& pool,
    double cex_price,
    const Costs& costs,
    double notional_cap_coin0,
    double min_swap_frac,
    double max_swap_frac
) {
    Decision d{};
    const double pool_price = spot_price(pool);
    if (pool_price == cex_price) return d;
    d.i = (pool_price < cex_price) ? 0 : 1;
    d.j = 1 - d.i;

    double available = pool.balances[d.i];
    double dx0  = std::max(1e-18, available * std::max(1e-12, min_swap_frac));
    double dxHi = available * std::min(max_swap_frac, costs.max_trade_frac);
    if (costs.use_volume_cap) {
        if (d.i == 0) dxHi = std::min(dxHi, notional_cap_coin0);
        else if (cex_price > 0) dxHi = std::min(dxHi, notional_cap_coin0 / cex_price);
        if (dxHi <= 0) return d;
    }

    double fee_lo = 0.0, fee_hi = 0.0;
    auto lo = simulate_profit(pool, d.i, d.j, dx0, cex_price, costs, fee_lo);
    if (!lo) return d;
    auto hi = simulate_profit(pool, d.i, d.j, dxHi, cex_price, costs, fee_hi);

    double best_dx = dx0;
    double best_profit = lo->second;
    double best_fee   = fee_lo;

    if (hi && hi->second >= best_profit) {
        best_dx     = dxHi;
        best_profit = hi->second;
        best_fee    = fee_hi;
    } else {
        double L = dx0, R = dxHi;
        for (int it = 0; it < 40; ++it) {
            double mid = 0.5 * (L + R);
            double fee_mid = 0.0;
            auto m = simulate_profit(pool, d.i, d.j, mid, cex_price, costs, fee_mid);
            if (m) {
                L = mid;
                if (m->second >= best_profit) {
                    best_profit = m->second;
                    best_dx     = mid;
                    best_fee    = fee_mid;
                }
            } else {
                R = mid;
            }
            if (R - L <= std::max(1e-12, available * 1e-12)) break;
        }
    }

    d.do_trade       = true;
    d.dx             = best_dx;
    d.profit         = best_profit;
    d.fee_tokens     = best_fee;
    d.notional_coin0 = (d.i==0 && d.j==1) ? best_dx : best_dx * cex_price;
    return d;
}

struct Candle { uint64_t ts; double open, high, low, close, volume; };
struct Event  { uint64_t ts; double p_cex; double volume; };

inline const unsigned char* skip_ws(const unsigned char* p, const unsigned char* end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) ++p; return p;
}
inline const unsigned char* scan_u64(const unsigned char* p, const unsigned char* end, uint64_t& v) {
    v = 0; p = skip_ws(p,end); while (p < end && *p >= '0' && *p <= '9') { v = v*10 + (*p - '0'); ++p; } return skip_ws(p,end);
}
inline const unsigned char* scan_num(const unsigned char* p, const unsigned char* end, double& d) {
    p = skip_ws(p,end); const char* c = reinterpret_cast<const char*>(p); char* ce=nullptr; d = std::strtod(c,&ce); if (ce==c) return p; p = reinterpret_cast<const unsigned char*>(ce); return skip_ws(p,end);
}

static std::vector<Candle> load_candles(const std::string& path, size_t max_candles = 0) {
    std::vector<Candle> out; out.reserve(1024);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        struct stat st{};
        if (::fstat(fd, &st) == 0 && st.st_size > 0) {
            size_t sz = static_cast<size_t>(st.st_size);
            void* map = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
            if (map != MAP_FAILED) {
                const unsigned char* p = reinterpret_cast<const unsigned char*>(map);
                const unsigned char* end = p + sz;
                p = skip_ws(p,end); if (p<end && *p=='[') ++p;
                while (p < end) {
                    p = skip_ws(p,end); if (p>=end || *p==']') break; if (*p!='[') { ++p; continue; }
                    ++p; Candle c{}; uint64_t ts=0; double o=0,h=0,l=0,cl=0,v=0;
                    p=scan_u64(p,end,ts); if (ts>10000000000ULL) ts/=1000ULL;
                    p=scan_num(p,end,o); p=scan_num(p,end,h); p=scan_num(p,end,l); p=scan_num(p,end,cl); p=scan_num(p,end,v);
                    while (p<end && *p!=']') ++p; if (p<end && *p==']') ++p;
                    c.ts=ts; c.open=o; c.high=h; c.low=l; c.close=cl; c.volume=v; out.push_back(c);
                    if (max_candles > 0 && out.size() >= max_candles) break;
                }
                ::munmap(map, sz);
                ::close(fd);
                return out;
            }
        }
        ::close(fd);
    }
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("Cannot open candles file: " + path);
    std::ostringstream oss; oss << in.rdbuf(); std::string buf = oss.str();
    const unsigned char* p = reinterpret_cast<const unsigned char*>(buf.data()); const unsigned char* end = p + buf.size();
    p = skip_ws(p,end); if (p<end && *p=='[') ++p;
    while (p < end) {
        p = skip_ws(p,end); if (p>=end || *p==']') break; if (*p!='[') { ++p; continue; }
        ++p; Candle c{}; uint64_t ts=0; double o=0,h=0,l=0,cl=0,v=0; p=scan_u64(p,end,ts); if (ts>10000000000ULL) ts/=1000ULL;
        p=scan_num(p,end,o); p=scan_num(p,end,h); p=scan_num(p,end,l); p=scan_num(p,end,cl); p=scan_num(p,end,v);
        while (p<end && *p!=']') ++p; if (p<end && *p==']') ++p;
        c.ts=ts; c.open=o; c.high=h; c.low=l; c.close=cl; c.volume=v; out.push_back(c);
        if (max_candles > 0 && out.size() >= max_candles) break;
    }
    return out;
}

static std::vector<Event> gen_events(const std::vector<Candle>& cs) {
    std::vector<Event> evs; evs.reserve(cs.size()*2);
    for (const auto& c : cs) {
        double path1 = std::abs(c.open - c.low) + std::abs(c.high - c.close);
        double path2 = std::abs(c.open - c.high) + std::abs(c.low - c.close);
        bool first_low = path1 < path2;
        evs.push_back(Event{c.ts + 5,  first_low ? c.low  : c.high, c.volume/2.0});
        evs.push_back(Event{c.ts + 15, first_low ? c.high : c.low,  c.volume/2.0});
    }
    std::sort(evs.begin(), evs.end(), [](const Event& a, const Event& b){ return a.ts < b.ts; });
    return evs;
}

static std::string to_str_1e18(double v) {
    long double scaled = static_cast<long double>(v) * 1e18L;
    if (scaled < 0) scaled = 0;
    std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << scaled; return oss.str();
}

static json::object pool_state_json(const twocrypto::TwoCryptoPoolT<double>& p) {
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

static double parse_scaled_1e18(const json::value& v) {
    if (v.is_string()) return std::strtod(v.as_string().c_str(), nullptr) / 1e18;
    if (v.is_double()) return v.as_double() / 1e18;
    if (v.is_int64())  return static_cast<double>(v.as_int64()) / 1e18;
    if (v.is_uint64()) return static_cast<double>(v.as_uint64()) / 1e18;
    return 0.0;
}
static double parse_fee_1e10(const json::value& v) {
    if (v.is_string()) return std::strtod(v.as_string().c_str(), nullptr) / 1e10;
    if (v.is_double()) return v.as_double() / 1e10;
    if (v.is_int64())  return static_cast<double>(v.as_int64()) / 1e10;
    if (v.is_uint64()) return static_cast<double>(v.as_uint64()) / 1e10;
    return 0.0;
}
static double parse_plain_double(const json::value& v) {
    if (v.is_string()) return std::strtod(v.as_string().c_str(), nullptr);
    if (v.is_double()) return v.as_double();
    if (v.is_int64())  return static_cast<double>(v.as_int64());
    if (v.is_uint64()) return static_cast<double>(v.as_uint64());
    return 0.0;
}

struct PoolInit {
    std::array<double,2> precisions{1.0,1.0};
    double A{100000.0};
    double gamma{0.0};
    double mid_fee{3e-4};
    double out_fee{5e-4};
    double fee_gamma{0.23};
    double allowed_extra{1e-3};
    double adj_step{1e-3};
    double ma_time{600.0};
    double initial_price{1.0};
    std::array<double,2> initial_liq{1e6,1e6};
    uint64_t start_ts{0};
    // Donation controls (harness-only)
    double donation_apy{0.0};              // plain fraction per year, e.g., 0.05 = 5%
    double donation_frequency{0.0};        // seconds between donations
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
        if (auto* v = co.if_contains("max_trade_frac")) out_costs.max_trade_frac = parse_plain_double(*v);
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

                    using Pool = twocrypto::TwoCryptoPoolT<double>;
                    Pool pool(cfg.precisions, cfg.A, cfg.gamma, cfg.mid_fee, cfg.out_fee, cfg.fee_gamma, cfg.allowed_extra, cfg.adj_step, cfg.ma_time, cfg.initial_price);
                    // Initialize timestamp baseline so EMA can progress (before any liquidity is added)
                    uint64_t init_ts = 0;
                    if (cfg.start_ts != 0) init_ts = cfg.start_ts;
                    else if (!events.empty()) init_ts = events.front().ts;
                    if (init_ts != 0) pool.set_block_timestamp(init_ts);
                    (void)pool.add_liquidity(cfg.initial_liq, 0.0);

                    Metrics m{};
                    json::array actions;
                    // Donation scheduler
                    const double SECONDS_PER_YEAR = 365.0 * 86400.0;
                    bool donations_enabled = (cfg.donation_apy > 0.0) && (cfg.donation_frequency > 0.0);
                    uint64_t next_donation_ts = 0;
                    if (donations_enabled && !events.empty()) {
                        uint64_t base_ts = (cfg.start_ts != 0) ? cfg.start_ts : events.front().ts;
                        next_donation_ts = base_ts + static_cast<uint64_t>(cfg.donation_frequency);
                    }
                    auto t_pool0 = clk::now();
                    for (const auto& ev : events) {
                        pool.set_block_timestamp(ev.ts);
                        // Apply any due donations before trading
                        if (donations_enabled) {
                            while (next_donation_ts != 0 && ev.ts >= next_donation_ts) {
                                // Compute donation sized as fraction of current TVL in coin0
                                double ps = pool.cached_price_scale;
                                double tvl_coin0 = pool.balances[0] + pool.balances[1] * ps;
                                double frac = cfg.donation_apy * (cfg.donation_frequency / SECONDS_PER_YEAR);
                                // Skip if fraction exceeds cap approximation (avoid revert/noise)
                                if (frac > pool.donation_shares_max_ratio) {
                                    next_donation_ts += static_cast<uint64_t>(cfg.donation_frequency);
                                    if (next_donation_ts <= ev.ts) next_donation_ts = ev.ts + 1;
                                    continue;
                                }
                                double donate_coin0 = tvl_coin0 * frac;
                                if (donate_coin0 > 0.0) {
                                    double amt0 = 0.5 * donate_coin0;
                                    double amt1 = (0.5 * donate_coin0) / (ps > 0.0 ? ps : 1.0);
                                    try {
                                        double price_scale_before = pool.cached_price_scale;
                                        double minted = pool.add_liquidity({amt0, amt1}, 0.0, true);
                                        double price_scale_after = pool.cached_price_scale;
                                        m.donations += 1;
                                        m.donation_coin0_total += donate_coin0;
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
                                            da["amounts"] = json::array{amt0, amt1};
                                            da["minted"] = minted;
                                            da["tvl_coin0_before"] = tvl_coin0;
                                            da["price_scale"] = ps;
                                            actions.push_back(da);
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
                        double pre_p_pool = spot_price(pool);
                        double notional_cap_coin0 = std::numeric_limits<double>::infinity();
                        if (costs.use_volume_cap) notional_cap_coin0 = ev.volume * ev.p_cex * costs.volume_cap_mult;
                        Decision d = decide_trade(pool, ev.p_cex, costs, notional_cap_coin0, min_swap_frac, max_swap_frac);
                        if (!d.do_trade) continue;
                        try {
                            double price_scale_before = pool.cached_price_scale;
                            auto res = pool.exchange((double)d.i, (double)d.j, d.dx, 0.0);
                            double dy_after_fee = res[0];
                            double fee_tokens   = res[1];
                            // Read the committed value directly from the pool after exchange
                            double price_scale_after = pool.cached_price_scale;
                            m.trades   += 1;
                            m.notional += d.notional_coin0;
                            m.lp_fee_coin0 += (d.j==1 ? fee_tokens * ev.p_cex : fee_tokens);
                            if (differs_rel(price_scale_after, price_scale_before)) m.n_rebalances += 1;
                            double profit_coin0 = 0.0;
                            if (d.i == 0 && d.j == 1) {
                                double coin0_out_cex = dy_after_fee * ev.p_cex;
                                double arb_fee = (costs.arb_fee_bps/1e4) * coin0_out_cex;
                                profit_coin0 = coin0_out_cex - d.dx - arb_fee - costs.gas_coin0;
                            } else {
                                double coin0_spent_cex = d.dx * ev.p_cex;
                                double arb_fee = (costs.arb_fee_bps/1e4) * coin0_spent_cex;
                                profit_coin0 = dy_after_fee - coin0_spent_cex - arb_fee - costs.gas_coin0;
                            }
                            m.arb_pnl_coin0 += profit_coin0;
                            if (save_actions) {
                                json::object tr;
                                tr["type"] = "exchange";
                                tr["ts"] = ev.ts; tr["i"] = d.i; tr["j"] = d.j; tr["dx"] = d.dx;
                                tr["dy_after_fee"] = dy_after_fee; tr["fee_tokens"] = fee_tokens;
                                tr["profit_coin0"] = profit_coin0; tr["p_cex"] = ev.p_cex; tr["p_pool_before"] = pre_p_pool;
                                actions.push_back(tr);
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
                    summary["donation_amounts_total"] = json::array{m.donation_amounts_total[0], m.donation_amounts_total[1]};
                    summary["pool_exec_ms"] = pool_exec_ms;

                    json::object params;
                    params["pool"] = echo_pool;
                    if (!echo_costs.empty()) params["costs"] = echo_costs;

                    json::object out;
                    out["result"] = summary;
                    out["params"] = params;
                    out["final_state"] = pool_state_json(pool);
                    if (save_actions) out["actions"] = actions;
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
