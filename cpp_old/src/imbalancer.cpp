// TwoCrypto pool imbalancer (from-scratch, focused, readable)
// -----------------------------------------------------------
// Goal
// - Initialize a TwoCrypto pool from a simple pool.json (next to cpp/)
//   containing only:
//     - initial_balances: ["<wei>", "<wei>"]
//     - price_scale:      "<1e18-scaled>"
//     - mid_fee:          "<1e10-scaled>"
//     - out_fee:          "<1e10-scaled>"
// - Then either:
//   1) Precisely solve a single swap that drives coin <target_coin> down to
//      exactly <target_wei> (or as close as possible), or
//   2) Run a simple fuzzer that repeatedly swaps a lot of coin i into coin j
//      to unbalance the pool toward the same target.
//
// CLI
//   imbalancer_i <pool.json> <output.json> --target-coin <0|1> --target-wei <N>
//                [--mode solve|fuzz] [--max-steps K] [--seed S]
//
// Notes
// - Uses boost::multiprecision uint256 via TwoCryptoPoolT<uint256> by default.
// - All pool parameters are required in pool.json (no defaults hidden here).
// - Price oracle/EMA and dynamic fee paths are active per provided params.

#include "twocrypto.hpp"

#include <boost/json.hpp>
#include <boost/json/src.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace twocrypto;
namespace json = boost::json;

namespace {

#if defined(IMBALANCER_MODE_D)
using NumT = double;
#else
using NumT = stableswap::uint256; // default to big-int
#endif

// -------------------------- Small parsing utilities --------------------------
static inline std::string as_string(const json::value& v) {
    if (v.is_string()) return std::string(v.as_string().c_str());
    if (v.is_int64())  return std::to_string(v.as_int64());
    if (v.is_uint64()) return std::to_string(v.as_uint64());
    if (v.is_double()) {
        std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0);
        oss << v.as_double(); return oss.str();
    }
    throw std::runtime_error("expected string/number in JSON");
}

// Parse decimal string s and scale by 10^scale, returning uint256 integer
static stableswap::uint256 parse_scaled_u256(std::string s, unsigned scale) {
    // remove spaces
    s.erase(std::remove_if(s.begin(), s.end(), [](char c){ return c==' ' || c=='\t' || c=='\n' || c=='_'; }), s.end());
    if (s.empty()) return stableswap::uint256(0);
    // handle leading +
    if (s[0] == '+') s.erase(s.begin());
    // find dot
    auto pos = s.find('.');
    std::string ip = s;
    std::string fp;
    if (pos != std::string::npos) {
        ip = s.substr(0, pos);
        fp = s.substr(pos + 1);
    }
    // strip non-digits (basic guard)
    auto only_digits = [](std::string x){
        std::string y; y.reserve(x.size());
        for (char c : x) if (c >= '0' && c <= '9') y.push_back(c);
        if (y.empty()) y = "0";
        // remove leading zeros
        size_t i = 0; while (i + 1 < y.size() && y[i] == '0') ++i; return y.substr(i);
    };
    ip = only_digits(ip);
    fp = only_digits(fp);
    // build integer string: ip + fp (truncated/padded to 'scale')
    if (fp.size() > scale) fp = fp.substr(0, scale);
    else if (fp.size() < scale) fp.append(scale - fp.size(), '0');
    std::string digits = ip + fp;
    if (digits.empty()) digits = "0";
    return stableswap::uint256(digits);
}

// Generic converters honoring the requested internal scales
static inline NumT parse_balance(const json::value& v) {
    if constexpr (std::is_same_v<NumT, stableswap::uint256>) return parse_scaled_u256(as_string(v), 18);
    else return static_cast<NumT>(std::strtold(as_string(v).c_str(), nullptr));
}
static inline NumT parse_price_scale(const json::value& v) {
    if constexpr (std::is_same_v<NumT, stableswap::uint256>) return parse_scaled_u256(as_string(v), 18);
    else return static_cast<NumT>(std::strtold(as_string(v).c_str(), nullptr));
}
static inline NumT parse_fee_1e10(const json::value& v) {
    if constexpr (std::is_same_v<NumT, stableswap::uint256>) return parse_scaled_u256(as_string(v), 10);
    else return static_cast<NumT>(std::strtold(as_string(v).c_str(), nullptr));
}
static inline NumT parse_A_times_1e4(const json::value& v) {
    if constexpr (std::is_same_v<NumT, stableswap::uint256>) return parse_scaled_u256(as_string(v), 4);
    else return static_cast<NumT>(std::strtold(as_string(v).c_str(), nullptr) * 10000.0L);
}
static inline NumT parse_scaled_1e18(const json::value& v) {
    if constexpr (std::is_same_v<NumT, stableswap::uint256>) return parse_scaled_u256(as_string(v), 18);
    else return static_cast<NumT>(std::strtold(as_string(v).c_str(), nullptr));
}
static inline NumT parse_plain(const json::value& v) {
    if constexpr (std::is_same_v<NumT, stableswap::uint256>) return stableswap::uint256(as_string(v));
    else return static_cast<NumT>(std::strtold(as_string(v).c_str(), nullptr));
}

static inline std::string to_wei_str(const NumT& x) {
    if constexpr (std::is_same_v<NumT, stableswap::uint256>) return x.str();
    long double s = static_cast<long double>(x) * 1e18L;
    if (s < 0) s = 0;
    std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << s; return oss.str();
}

// ------------------------- Pool init from pool.json --------------------------
struct PoolConfig {
    std::array<NumT,2> initial_balances{NumTraits<NumT>::ZERO(), NumTraits<NumT>::ZERO()};
    NumT price_scale{NumTraits<NumT>::PRECISION()};
    NumT mid_fee{NumTraits<NumT>::ZERO()};
    NumT out_fee{NumTraits<NumT>::ZERO()};
    // Required additional params (all from JSON, scaled internally here)
    // A is provided as plain (e.g., 9) and we multiply by 10_000 internally
    NumT A{NumTraits<NumT>::ZERO()};
    // gamma is provided as plain decimal, scaled by 1e18 internally for uint256
    NumT gamma{NumTraits<NumT>::ZERO()};
    // The following use PRECISION (1e18) scale internally for uint256; doubles keep decimals
    NumT fee_gamma{NumTraits<NumT>::ZERO()};
    NumT allowed_extra_profit{NumTraits<NumT>::ZERO()};
    NumT adjustment_step{NumTraits<NumT>::ZERO()};
    // ma_time is seconds (plain)
    NumT ma_time{NumTraits<NumT>::ZERO()};
};

static PoolConfig load_pool_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open pool.json: " + path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto j = json::parse(s).as_object();

    PoolConfig cfg;
    auto init = j.at("initial_balances").as_array();
    cfg.initial_balances[0] = parse_balance(init[0]);            // *1e18 for uint256
    cfg.initial_balances[1] = parse_balance(init[1]);
    cfg.price_scale         = parse_price_scale(j.at("price_scale")); // *1e18 for uint256
    cfg.mid_fee             = parse_fee_1e10(j.at("mid_fee"));       // *1e10 for uint256
    cfg.out_fee             = parse_fee_1e10(j.at("out_fee"));

    // All remaining params required
    cfg.A                    = parse_A_times_1e4(j.at("A"));
    cfg.gamma                = parse_scaled_1e18(j.at("gamma"));
    cfg.fee_gamma            = parse_scaled_1e18(j.at("fee_gamma"));
    cfg.allowed_extra_profit = parse_scaled_1e18(j.at("allowed_extra_profit"));
    cfg.adjustment_step      = parse_scaled_1e18(j.at("adjustment_step"));
    cfg.ma_time              = parse_plain(j.at("ma_time")); // seconds
    return cfg;
}

static TwoCryptoPoolT<NumT> make_pool(const PoolConfig& cfg) {
    std::array<NumT,2> precisions{ NumTraits<NumT>::ONE(), NumTraits<NumT>::ONE() };
    TwoCryptoPoolT<NumT> pool(
        precisions,
        cfg.A,
        cfg.gamma,
        cfg.mid_fee,
        cfg.out_fee,
        cfg.fee_gamma,
        cfg.allowed_extra_profit,
        cfg.adjustment_step,
        cfg.ma_time,
        cfg.price_scale // initial_price
    );
    (void)pool.add_liquidity(cfg.initial_balances, NumTraits<NumT>::ZERO());
    return pool;
}

// ------------------------------ Solver helpers ------------------------------
static inline NumT simulate_dy_after_fee(const TwoCryptoPoolT<NumT>& pool, size_t i, size_t j, const NumT& dx) {
    if (dx == NumTraits<NumT>::ZERO()) return NumTraits<NumT>::ZERO();
    try {
        TwoCryptoPoolT<NumT> p2 = pool;
        auto res = p2.exchange(NumT(i), NumT(j), dx, NumTraits<NumT>::ZERO());
        return res[0];
    } catch (...) {
        return NumTraits<NumT>::ZERO();
    }
}

static inline NumT find_dx_for_target_j(const TwoCryptoPoolT<NumT>& pool, size_t i, size_t j, const NumT& target_j) {
    using Traits = NumTraits<NumT>;
    const NumT bj = pool.balances[j];
    if (bj <= target_j) return Traits::ZERO();
    const NumT need = bj - target_j;

    NumT lo = Traits::ZERO();
    NumT hi = Traits::ONE();
    NumT dy_prev = Traits::ZERO();
    NumT dy_hi = simulate_dy_after_fee(pool, i, j, hi);
    int guard = 0;
    while (dy_hi < need && guard++ < 512) {
        hi = hi + hi; // *= 2
        NumT dy_new = simulate_dy_after_fee(pool, i, j, hi);
        dy_prev = dy_hi;
        dy_hi = dy_new;
    }
    if (!(dy_hi >= need)) return hi;

    for (int it = 0; it < 256; ++it) {
        NumT mid = (lo + hi) / NumT(2);
        if (mid == lo || mid == hi) break;
        NumT dy_mid = simulate_dy_after_fee(pool, i, j, mid);
        if (dy_mid >= need) { hi = mid; } else { lo = mid; }
        if (hi == lo + Traits::ONE()) break;
    }
    return hi;
}

// --------------------------------- Runner -----------------------------------
struct Args {
    std::string pool_json;
    std::string output_json;
    int         target_coin{1};
    stableswap::uint256 target_wei_u256{8};
    std::string mode{"solve"}; // or "fuzz"
    int max_steps{64};
    uint64_t seed{0};
};

static Args parse_args(int argc, char* argv[]) {
    if (argc < 5) {
        throw std::runtime_error(
            "Usage: " + std::string(argv[0]) +
            " <pool.json> <output.json> --target-coin <0|1> --target-wei <N> [--mode solve|fuzz] [--max-steps K] [--seed S]"
        );
    }
    Args a; a.pool_json = argv[1]; a.output_json = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](int n){ if (i + 1 >= argc) throw std::runtime_error("missing value for " + k); return argv[++i]; };
        if (k == std::string("--target-coin")) {
            a.target_coin = std::stoi(need(1));
            if (a.target_coin != 0 && a.target_coin != 1) throw std::runtime_error("target_coin must be 0 or 1");
        } else if (k == std::string("--target-wei")) {
            a.target_wei_u256 = stableswap::uint256(std::string(need(1)));
        } else if (k == std::string("--mode")) {
            a.mode = need(1);
        } else if (k == std::string("--max-steps")) {
            a.max_steps = std::stoi(need(1));
        } else if (k == std::string("--seed")) {
            a.seed = static_cast<uint64_t>(std::strtoull(need(1), nullptr, 10));
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

static int run(const Args& args) {
    try {
        auto cfg  = load_pool_config(args.pool_json);
        auto pool = make_pool(cfg);

        const size_t j = static_cast<size_t>(args.target_coin);
        const size_t i = 1 - j;

        json::object report;
        report["mode"] = args.mode;
        report["target_coin"] = static_cast<int>(j);
        report["target_wei"]  = args.target_wei_u256.str();
        report["initial_balances"] = json::array{ to_wei_str(pool.balances[0]), to_wei_str(pool.balances[1]) };
        report["initial_price_scale"] = to_wei_str(pool.cached_price_scale);

        if (args.mode == "solve") {
            // Iterative solver: repeatedly shrink toward target to avoid extreme brackets
            NumT target_j = std::is_same_v<NumT, stableswap::uint256>
                ? NumT(args.target_wei_u256)
                : NumT(static_cast<long double>(args.target_wei_u256.convert_to<long double>()) / 1e18L);

            TwoCryptoPoolT<NumT> cur = pool;
            json::array steps;
            int iters = 0;
            while (iters++ < 128 && cur.balances[j] > target_j) {
                // Aim halfway to target each iteration
                NumT j_now = cur.balances[j];
                NumT step_target;
                if constexpr (std::is_same_v<NumT, stableswap::uint256>) {
                    stableswap::uint256 gap = j_now - target_j;
                    step_target = j_now - (gap / stableswap::uint256(2));
                    if (step_target < target_j) step_target = target_j;
                } else {
                    NumT gap = j_now - target_j;
                    step_target = std::max(target_j, j_now - gap / NumT(2));
                }
                NumT dx = find_dx_for_target_j(cur, i, j, step_target);
                if (dx == NumTraits<NumT>::ZERO()) break;
                try {
                    (void)cur.exchange(NumT(i), NumT(j), dx, NumTraits<NumT>::ZERO());
                } catch (...) {
                    break;
                }
                json::object stp; stp["dx"] = to_wei_str(dx);
                stp["post_balances"] = json::array{ to_wei_str(cur.balances[0]), to_wei_str(cur.balances[1]) };
                steps.push_back(stp);
                // If just above target, try one last direct jump
                if (cur.balances[j] <= target_j) break;
            }
            // Optional final nudge toward exact target (may be no-op)
            if (cur.balances[j] > target_j) {
                NumT dx_last = find_dx_for_target_j(cur, i, j, target_j);
                if (dx_last != NumTraits<NumT>::ZERO()) {
                    (void)cur.exchange(NumT(i), NumT(j), dx_last, NumTraits<NumT>::ZERO());
                    json::object stp; stp["dx"] = to_wei_str(dx_last);
                    stp["post_balances"] = json::array{ to_wei_str(cur.balances[0]), to_wei_str(cur.balances[1]) };
                    steps.push_back(stp);
                }
            }
            report["steps"] = steps;
            report["final_balances"] = json::array{ to_wei_str(cur.balances[0]), to_wei_str(cur.balances[1]) };
            report["final_price_scale"] = to_wei_str(cur.cached_price_scale);
        } else if (args.mode == "fuzz") {
            TwoCryptoPoolT<NumT> cur = pool;
            json::array steps;
            int n = 0;
            // Convert target to NumT
            NumT target_j = std::is_same_v<NumT, stableswap::uint256>
                ? NumT(args.target_wei_u256)
                : NumT(static_cast<long double>(args.target_wei_u256.convert_to<long double>()) / 1e18L);

            while (n++ < args.max_steps) {
                if (cur.balances[j] <= target_j) break;

                // Aim to remove a fraction of the remaining gap to target
                // Similar spirit to pool_imbalance_2.py (dy fraction shrink)
                // Here we compute dx that would land us at: new_j = max(target, j - frac*(j-target))
                // Start large and adaptively reduce if needed (guaranteed monotone)
                NumT j_now = cur.balances[j];
                // Use decreasing fraction as we get closer
                long double frac = 0.25L;
                if constexpr (!std::is_same_v<NumT, stableswap::uint256>) {
                    frac = 0.25L;
                }
                // target for this step
                NumT step_target = target_j;
                if constexpr (std::is_same_v<NumT, stableswap::uint256>) {
                    stableswap::uint256 gap = (j_now > target_j) ? (j_now - target_j) : stableswap::uint256(0);
                    stableswap::uint256 remove = (gap * stableswap::uint256(25)) / stableswap::uint256(100);
                    step_target = (j_now - remove < target_j) ? target_j : (j_now - remove);
                } else {
                    NumT gap = j_now - target_j;
                    NumT remove = gap * static_cast<NumT>(frac);
                    step_target = std::max(target_j, j_now - remove);
                }

                NumT dx = find_dx_for_target_j(cur, i, j, step_target);
                auto before_j = cur.balances[j];
                if (dx == NumTraits<NumT>::ZERO()) {
                    // fallback: try large geometric dx with backoff
                    NumT trial = (cur.balances[i] > NumTraits<NumT>::ZERO()) ? (cur.balances[i] / NumT(4) + NumT(1)) : NumTraits<NumT>::ONE();
                    bool ok = false;
                    for (int k = 0; k < 32; ++k) {
                        if (trial == NumTraits<NumT>::ZERO()) break;
                        try {
                            TwoCryptoPoolT<NumT> tmp = cur;
                            (void)tmp.exchange(NumT(i), NumT(j), trial, NumTraits<NumT>::ZERO());
                            cur = tmp; dx = trial; ok = true; break;
                        } catch (...) {
                            trial = trial / NumT(2);
                        }
                    }
                    if (!ok) break;
                } else {
                    try { (void)cur.exchange(NumT(i), NumT(j), dx, NumTraits<NumT>::ZERO()); }
                    catch (...) { break; }
                }

                json::object stp;
                stp["dx"] = to_wei_str(dx);
                stp["post_balances"] = json::array{ to_wei_str(cur.balances[0]), to_wei_str(cur.balances[1]) };
                steps.push_back(stp);
                if (cur.balances[j] >= before_j) break; // safety
            }
            report["steps"] = steps;
            report["final_balances"] = json::array{ to_wei_str(cur.balances[0]), to_wei_str(cur.balances[1]) };
            report["final_price_scale"] = to_wei_str(cur.cached_price_scale);
        } else {
            throw std::runtime_error("unknown mode: " + args.mode);
        }

        std::ofstream of(args.output_json);
        of << json::serialize(report) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace

// -------------------------------- Entrypoints --------------------------------
int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);
    return run(args);
}
