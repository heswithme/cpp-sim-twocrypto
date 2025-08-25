// Candle-Driven Arbitrage Harness
//
// Simulates a single TwoCrypto pool against a CEX price stream derived from
// candles. For each candle we create two sub-events (L->H or H->L ordering)
// and execute at most one optimal arbitrage trade per sub-event.
//
// Units and scales (double variant):
// - Token amounts are in raw token units (1.0 == 1e18 in JSON inputs).
// - Fees: mid_fee/out_fee are parsed from 1e10-scaled JSON integers.
// - Prices are coin1 in coin0 units; coin1 uses price_scale inside the pool.
// - Costs: gas_coin0 is in coin0 units; arb_fee_bps is in bps (1/10000).
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include "twocrypto.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <optional>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace json = boost::json;

namespace {
// ---- Cost & Metrics ---------------------------------------------------------
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
};

// ---- Pool helpers (read/estimate only) -------------------------------------
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

// ---- Price/Fee helpers ------------------------------------------------------
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
// ---- Simulation & Decision --------------------------------------------------
struct SimulationResult {
    double dy_after_fee{0.0};
    double fee_tokens{0.0};
    double profit_coin0{0.0};
    double notional_coin0{0.0};
};

static std::optional<SimulationResult> simulate_exchange(
    const twocrypto::TwoCryptoPoolT<double>& pool,
    int i, int j,
    double dx,
    double cex_price,
    const Costs& costs
) {
    using Ops = stableswap::MathOps<double>;
    const double price_scale = pool.cached_price_scale;

    // Build local xp with dx injected on side i
    std::array<double,2> balances = pool.balances; balances[i] += dx;
    std::array<double,2> xp = pool_xp_from(pool, balances, price_scale);

    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, static_cast<size_t>(j));
    double dy_xp = xp[j] - y_out.value;
    xp[j] -= dy_xp; // post-trade xp for fee computation

    // Convert xp delta to token units (apply price_scale for coin1)
    double dy_tokens = xp_to_tokens_j(pool, static_cast<size_t>(j), dy_xp, price_scale);

    // Dynamic fee in token units of coin j
    std::array<double,2> xp_fee{ xp[0], xp[1] };
    double fee_rate = dyn_fee(xp_fee, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    double fee_tokens = fee_rate * dy_tokens;
    double dy_after_fee = dy_tokens - fee_tokens;

    // Compute round-trip profit in coin0
    double profit_coin0 = 0.0;
    if (i == 0 && j == 1) {
        // Buy coin1 from pool using coin0; sell on CEX back to coin0
        double coin0_out_cex = dy_after_fee * cex_price;
        double arb_fee = (costs.arb_fee_bps/1e4) * coin0_out_cex;
        profit_coin0 = coin0_out_cex - dx - arb_fee - costs.gas_coin0;
    } else {
        // Sell coin1 to pool for coin0; buy coin1 on CEX using coin0
        double coin0_spent_cex = dx * cex_price;
        double arb_fee = (costs.arb_fee_bps/1e4) * coin0_spent_cex;
        profit_coin0 = dy_after_fee - coin0_spent_cex - arb_fee - costs.gas_coin0;
    }
    if (profit_coin0 <= 0.0) return std::nullopt;

    SimulationResult res{};
    res.dy_after_fee = dy_after_fee;
    res.fee_tokens   = fee_tokens;
    res.profit_coin0 = profit_coin0;
    res.notional_coin0 = (i==0 && j==1) ? dx : dx * cex_price;
    return res;
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

    // Direction: pool underprices coin1 => buy from pool (i=0->j=1), else sell (i=1->j=0)
    d.i = (pool_price < cex_price) ? 0 : 1;
    d.j = 1 - d.i;

    // Sizing bounds (fractions of available balance)
    double available = pool.balances[d.i];
    double dx0  = std::max(1e-18, available * std::max(1e-12, min_swap_frac));
    double dxHi = available * std::min(max_swap_frac, costs.max_trade_frac);
    if (costs.use_volume_cap) {
        if (d.i == 0) dxHi = std::min(dxHi, notional_cap_coin0);
        else if (cex_price > 0) dxHi = std::min(dxHi, notional_cap_coin0 / cex_price);
        if (dxHi <= 0) return d;
    }

    // Evaluate profitability at bounds
    auto sim_lo = simulate_exchange(pool, d.i, d.j, dx0, cex_price, costs);
    if (!sim_lo) return d; // not profitable even at minimum size
    auto sim_hi = simulate_exchange(pool, d.i, d.j, dxHi, cex_price, costs);

    double best_dx = dx0;
    double best_profit = sim_lo->profit_coin0;
    double best_fee_tokens = sim_lo->fee_tokens;

    if (sim_hi && sim_hi->profit_coin0 >= best_profit) {
        // Entire bracket profitable; choose upper bound
        best_dx = dxHi;
        best_profit = sim_hi->profit_coin0;
        best_fee_tokens = sim_hi->fee_tokens;
    } else {
        // Binary search for largest profitable dx in [dx0, dxHi]
        double lo = dx0;
        double hi = dxHi;
        for (int it = 0; it < 40; ++it) {
            double mid = 0.5 * (lo + hi);
            auto sim_mid = simulate_exchange(pool, d.i, d.j, mid, cex_price, costs);
            if (sim_mid) {
                lo = mid;
                if (sim_mid->profit_coin0 >= best_profit) {
                    best_profit = sim_mid->profit_coin0;
                    best_dx = mid;
                    best_fee_tokens = sim_mid->fee_tokens;
                }
            } else {
                hi = mid;
            }
            if (hi - lo <= std::max(1e-12, available * 1e-12)) break;
        }
    }

    d.do_trade       = true;
    d.dx             = best_dx;
    d.profit         = best_profit;
    d.fee_tokens     = best_fee_tokens;
    d.notional_coin0 = (d.i==0 && d.j==1) ? best_dx : best_dx * cex_price;
    return d;
}

static bool simulate_exchange(const twocrypto::TwoCryptoPoolT<double>& pool, int i, int j, double dx, double p_cex, const Costs& costs, double& dy_after_fee, double& profit_coin0, double& fee_tokens) {
    using Ops = stableswap::MathOps<double>;
    const double price_scale = pool.cached_price_scale;
    std::array<double,2> balances = pool.balances; balances[i] += dx;
    std::array<double,2> xp{ balances[0] * pool.precisions[0], balances[1] * pool.precisions[1] * price_scale };
    auto y_out = Ops::get_y(pool.A, pool.gamma, xp, pool.D, static_cast<size_t>(j));
    double dy_xp = xp[j] - y_out.value; xp[j] -= dy_xp;
    double dy_tokens = dy_xp; if (j == 1) dy_tokens = dy_tokens / price_scale; dy_tokens = dy_tokens / pool.precisions[j];
    std::array<double,2> xp_fee{ xp[0], xp[1] }; double fee_rate = dyn_fee(xp_fee, pool.mid_fee, pool.out_fee, pool.fee_gamma);
    fee_tokens = fee_rate * dy_tokens; dy_after_fee = dy_tokens - fee_tokens;
    double arb_fee = 0.0; // kept for readability; profit computed below
    // Compute profit in coin0
    if (i == 0 && j == 1) {
        double coin0_out_cex = dy_after_fee * p_cex;
        profit_coin0 = coin0_out_cex - dx - (costs.arb_fee_bps/1e4) * coin0_out_cex - costs.gas_coin0;
    } else {
        double coin0_spent_cex = dx * p_cex;
        profit_coin0 = dy_after_fee - coin0_spent_cex - (costs.arb_fee_bps/1e4) * coin0_spent_cex - costs.gas_coin0;
    }
    return profit_coin0 > 0.0;
}
// ---- IO & Event generation --------------------------------------------------
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

// Memory-map and byte-scan a JSON array of arrays. Each row:
// [timestamp, open, high, low, close, volume]
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
                p = skip_ws(p,end); if (p<end && *p=='[') ++p; // opening `[` of array
                while (p < end) {
                    p = skip_ws(p,end); if (p>=end || *p==']') break; if (*p!='[') { ++p; continue; }
                    ++p; Candle c{}; uint64_t ts=0; double o=0,h=0,l=0,cl=0,v=0;
                    p=scan_u64(p,end,ts); if (ts>10000000000ULL) ts/=1000ULL; // ms -> s
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
    // Fallback to stream read if mmap failed
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("Cannot open candles file: " + path);
    std::ostringstream oss; oss << in.rdbuf(); std::string buf = oss.str();
    const unsigned char* p = reinterpret_cast<const unsigned char*>(buf.data()); const unsigned char* end = p + buf.size();
    p = skip_ws(p,end); if (p<end && *p=='[') ++p; // opening `[` of array
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

// ---- JSON helpers for final state ------------------------------------------
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

std::vector<Event> gen_events(const std::vector<Candle>& cs) {
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

// ---------- Pool+Costs parsing (expects a single JSON file) ----------
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
};

static std::string as_string(const json::value& v) {
    if (v.is_string()) return std::string(v.as_string().c_str());
    if (v.is_int64()) { return std::to_string(v.as_int64()); }
    if (v.is_uint64()) { return std::to_string(v.as_uint64()); }
    if (v.is_double()) { std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << v.as_double(); return oss.str(); }
    return "0";
}
static double parse_scaled_1e18(const json::value& v) {
    // Accept string or number assumed to be 1e18-scaled integer
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

static void parse_pool_and_costs(const std::string& pool_json_path, PoolInit& out_pool, Costs& out_costs) {
    std::ifstream in(pool_json_path);
    if (!in) throw std::runtime_error("Cannot open pool json: " + pool_json_path);
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto root = json::parse(s);
    json::object obj = root.is_object() ? root.as_object() : json::object{};
    json::object pool = obj.contains("pool") ? obj.at("pool").as_object() : obj;
    // initial_liquidity
    if (auto* il = pool.if_contains("initial_liquidity")) {
        auto arr = il->as_array();
        out_pool.initial_liq[0] = parse_scaled_1e18(arr[0]);
        out_pool.initial_liq[1] = parse_scaled_1e18(arr[1]);
    }
    // Params
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

    // Costs (optional)
    if (auto* c = obj.if_contains("costs")) {
        auto co = c->as_object();
        if (auto* v = co.if_contains("arb_fee_bps")) out_costs.arb_fee_bps = parse_plain_double(*v);
        if (auto* v = co.if_contains("gas_coin0")) out_costs.gas_coin0 = parse_scaled_1e18(*v);
        if (auto* v = co.if_contains("max_trade_frac")) out_costs.max_trade_frac = parse_plain_double(*v);
        if (auto* v = co.if_contains("use_volume_cap")) out_costs.use_volume_cap = v->as_bool();
        if (auto* v = co.if_contains("volume_cap_mult")) out_costs.volume_cap_mult = parse_plain_double(*v);
    }
}
} // anonymous namespace

// ---- Main entrypoint --------------------------------------------------------
int run_arb_mode(const std::string& pool_json, const std::string& candles_path, const std::string& out_json,
                 size_t max_candles = 0, bool save_actions = false,
                 double min_swap_frac = 1e-6, double max_swap_frac = 1.0) {
    try {
        using clk = std::chrono::high_resolution_clock;
        auto t_read0 = clk::now();
        auto candles = load_candles(candles_path, max_candles);
        auto t_read1 = clk::now();
        auto events  = gen_events(candles);
        auto t_exec0 = clk::now();
        using Pool = twocrypto::TwoCryptoPoolT<double>;
        PoolInit cfg; Costs costs;
        if (pool_json != "-" && !pool_json.empty()) {
            parse_pool_and_costs(pool_json, cfg, costs);
        }
        Pool pool(cfg.precisions, cfg.A, cfg.gamma, cfg.mid_fee, cfg.out_fee, cfg.fee_gamma, cfg.allowed_extra, cfg.adj_step, cfg.ma_time, cfg.initial_price);
        if (cfg.start_ts != 0) pool.set_block_timestamp(cfg.start_ts);
        (void)pool.add_liquidity(cfg.initial_liq, 0.0);
        Metrics m{};
        json::array actions;
        for (const auto& ev : events) {
            pool.set_block_timestamp(ev.ts);
            double pre_p_pool = spot_price(pool);
            double notional_cap_coin0 = std::numeric_limits<double>::infinity();
            if (costs.use_volume_cap) {
                notional_cap_coin0 = ev.volume * ev.p_cex * costs.volume_cap_mult;
            }
            Decision d = decide_trade(pool, ev.p_cex, costs, notional_cap_coin0, min_swap_frac, max_swap_frac);
            if (!d.do_trade) continue;
            try {
                auto res = pool.exchange((double)d.i, (double)d.j, d.dx, 0.0);
                double dy_after_fee = res[0];
                double fee_tokens   = res[1];
                // Update metrics from committed trade
                m.trades   += 1;
                m.notional += d.notional_coin0;
                m.lp_fee_coin0 += (d.j==1 ? fee_tokens * ev.p_cex : fee_tokens);
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
                    tr["ts"] = ev.ts;
                    tr["i"]  = d.i;
                    tr["j"]  = d.j;
                    tr["dx"] = d.dx;
                    tr["dy_after_fee"] = dy_after_fee;
                    tr["fee_tokens"] = fee_tokens;
                    tr["profit_coin0"] = profit_coin0;
                    tr["p_cex"] = ev.p_cex;
                    tr["p_pool_before"] = pre_p_pool;
                    actions.push_back(tr);
                }
            } catch (...) {}
        }
        auto t_exec1 = clk::now();
        double candles_read_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t_read1 - t_read0).count() / 1e6;
        double exec_ms         = std::chrono::duration_cast<std::chrono::nanoseconds>(t_exec1 - t_exec0).count() / 1e6;
        json::object summary; summary["events"]=events.size(); summary["trades"]=m.trades; summary["total_notional_coin0"]=m.notional; summary["lp_fee_coin0"]=m.lp_fee_coin0; summary["arb_pnl_coin0"]=m.arb_pnl_coin0;
        summary["candles_read_ms"] = candles_read_ms;
        summary["exec_ms"] = exec_ms;
        json::object params;
        // Echo pool + costs used
        json::object pool_params;
        pool_params["initial_liquidity"] = json::array{to_str_1e18(cfg.initial_liq[0]), to_str_1e18(cfg.initial_liq[1])};
        pool_params["A"] = cfg.A; pool_params["gamma"] = cfg.gamma; pool_params["mid_fee"] = to_str_1e18(cfg.mid_fee); pool_params["out_fee"] = to_str_1e18(cfg.out_fee);
        pool_params["fee_gamma"] = to_str_1e18(cfg.fee_gamma); pool_params["allowed_extra_profit"] = to_str_1e18(cfg.allowed_extra); pool_params["adjustment_step"] = to_str_1e18(cfg.adj_step);
        pool_params["ma_time"] = cfg.ma_time; pool_params["initial_price"] = to_str_1e18(cfg.initial_price);
        json::object costs_params; costs_params["arb_fee_bps"] = costs.arb_fee_bps; costs_params["gas_coin0"] = to_str_1e18(costs.gas_coin0);
        costs_params["max_trade_frac"] = costs.max_trade_frac; costs_params["use_volume_cap"] = costs.use_volume_cap; costs_params["volume_cap_mult"] = costs.volume_cap_mult;
        params["pool"] = pool_params; params["costs"] = costs_params;

        json::object meta{{"candles_file", candles_path}}; if (pool_json != "-") meta["pool_config_file"] = pool_json;
        json::object O; O["result"]=summary; O["metadata"]=meta; O["params"] = params; O["final_state"] = pool_state_json(pool);
        if (save_actions) O["actions"] = actions;
        std::ofstream of(out_json); of << json::serialize(O) << std::endl; return 0;
    } catch (const std::exception& e) { std::cerr << "Arb error: " << e.what() << std::endl; return 1; }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pool.json> <candles.json> <output.json> [--n-candles N] [--save-actions] [--min-swap F] [--max-swap F]" << std::endl; return 1;
    }
    std::string pool = argv[1]; std::string candles = argv[2]; std::string out = argv[3];
    size_t max_candles = 0;
    bool save_actions = false;
    double min_swap_frac = 1e-6;
    double max_swap_frac = 1.0;
    // Parse optional flags
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n-candles" && (i+1) < argc) {
            try {
                long long v = std::stoll(argv[i+1]);
                if (v > 0) max_candles = static_cast<size_t>(v);
            } catch (...) { /* ignore invalid */ }
            ++i; // skip value
        } else if (arg == "--save-actions") {
            save_actions = true;
        } else if (arg == "--min-swap" && (i+1) < argc) {
            try {
                double f = std::stod(argv[i+1]);
                if (f > 0.0 && f <= 1.0) min_swap_frac = f;
            } catch (...) {}
            ++i;
        } else if (arg == "--max-swap" && (i+1) < argc) {
            try {
                double f = std::stod(argv[i+1]);
                if (f > 0.0 && f <= 1.0) max_swap_frac = f;
            } catch (...) {}
            ++i;
        }
    }
    if (max_swap_frac < min_swap_frac) max_swap_frac = min_swap_frac;
    return run_arb_mode(pool, candles, out, max_candles, save_actions, min_swap_frac, max_swap_frac);
}
