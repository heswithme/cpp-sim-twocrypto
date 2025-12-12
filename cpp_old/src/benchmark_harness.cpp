// Unified C++ benchmark harness (mode: i|d)
#include "twocrypto.hpp"
#include <iostream>
#include <fstream>
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>

using namespace twocrypto;
namespace json = boost::json;

// Typed scaling helpers

template <typename T>
typename std::enable_if<std::is_same<T, stableswap::uint256>::value, T>::type
from_string_scaled_T(const std::string& s) { return stableswap::uint256(s); }

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
from_string_scaled_T(const std::string& s) { return static_cast<T>(std::strtold(s.c_str(), nullptr) / 1e18L); }

template <typename T>
typename std::enable_if<std::is_same<T, stableswap::uint256>::value, T>::type
from_fee_scaled_T(const std::string& s) { return stableswap::uint256(s); }

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
from_fee_scaled_T(const std::string& s) { return static_cast<T>(std::strtold(s.c_str(), nullptr) / 1e10L); }

template <typename T>
struct Report;

template <>
struct Report<stableswap::uint256> {
    static json::object to_json(const TwoCryptoPoolT<stableswap::uint256>& p) {
        auto to_str = [](const stableswap::uint256& v){ return v.str(); };
        json::object o;
        o["balances"] = json::array{to_str(p.balances[0]), to_str(p.balances[1])};
        auto xp = json::array{to_str(p.balances[0] * p.precisions[0]), to_str(p.balances[1] * p.precisions[1] * p.cached_price_scale / NumTraits<stableswap::uint256>::PRECISION())};
        o["xp"] = xp;
        o["D"] = to_str(p.D);
        o["virtual_price"] = to_str(p.virtual_price);
        o["xcp_profit"] = to_str(p.xcp_profit);
        o["price_scale"] = to_str(p.cached_price_scale);
        o["price_oracle"] = to_str(p.cached_price_oracle);
        o["last_prices"] = to_str(p.last_prices);
        o["totalSupply"] = to_str(p.totalSupply);
        o["timestamp"] = p.block_timestamp;
        return o;
    }
};

template <>
struct Report<double> {
    static std::string to_str(double v) {
        long double scaled = static_cast<long double>(v) * 1e18L; if (scaled < 0) scaled = 0; std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << scaled; return oss.str();
    }
    static json::object to_json(const TwoCryptoPoolT<double>& p) {
        json::object o;
        o["balances"] = json::array{to_str(p.balances[0]), to_str(p.balances[1])};
        o["xp"] = json::array{to_str(p.balances[0] * p.precisions[0]), to_str(p.balances[1] * p.precisions[1] * p.cached_price_scale)};
        o["D"] = to_str(p.D);
        o["virtual_price"] = to_str(p.virtual_price);
        o["xcp_profit"] = to_str(p.xcp_profit);
        o["price_scale"] = to_str(p.cached_price_scale);
        o["price_oracle"] = to_str(p.cached_price_oracle);
        o["last_prices"] = to_str(p.last_prices);
        o["totalSupply"] = to_str(p.totalSupply);
        o["timestamp"] = p.block_timestamp;
        return o;
    }
};

template <>
struct Report<float> {
    static std::string to_str(float v) {
        long double scaled = static_cast<long double>(v) * 1e18L; if (scaled < 0) scaled = 0; std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << scaled; return oss.str();
    }
    static json::object to_json(const TwoCryptoPoolT<float>& p) {
        json::object o;
        o["balances"] = json::array{to_str(p.balances[0]), to_str(p.balances[1])};
        o["xp"] = json::array{to_str(p.balances[0] * p.precisions[0]), to_str(p.balances[1] * p.precisions[1] * p.cached_price_scale)};
        o["D"] = to_str(p.D);
        o["virtual_price"] = to_str(p.virtual_price);
        o["xcp_profit"] = to_str(p.xcp_profit);
        o["price_scale"] = to_str(p.cached_price_scale);
        o["price_oracle"] = to_str(p.cached_price_oracle);
        o["last_prices"] = to_str(p.last_prices);
        o["totalSupply"] = to_str(p.totalSupply);
        o["timestamp"] = p.block_timestamp;
        return o;
    }
};

template <>
struct Report<long double> {
    static std::string to_str(long double v) {
        long double scaled = v * 1e18L; if (scaled < 0) scaled = 0; std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << scaled; return oss.str();
    }
    static json::object to_json(const TwoCryptoPoolT<long double>& p) {
        json::object o;
        o["balances"] = json::array{to_str(p.balances[0]), to_str(p.balances[1])};
        o["xp"] = json::array{to_str(p.balances[0] * p.precisions[0]), to_str(p.balances[1] * p.precisions[1] * p.cached_price_scale)};
        o["D"] = to_str(p.D);
        o["virtual_price"] = to_str(p.virtual_price);
        o["xcp_profit"] = to_str(p.xcp_profit);
        o["price_scale"] = to_str(p.cached_price_scale);
        o["price_oracle"] = to_str(p.cached_price_oracle);
        o["last_prices"] = to_str(p.last_prices);
        o["totalSupply"] = to_str(p.totalSupply);
        o["timestamp"] = p.block_timestamp;
        return o;
    }
};

template <typename T>
int run_harness(const std::string& pools_file, const std::string& sequences_file, const std::string& output_file) {
    try {
        // Load pools
        std::ifstream pf(pools_file); if (!pf) throw std::runtime_error("Cannot open pools file"); std::string s1((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        json::array pools = json::parse(s1).as_object().at("pools").as_array();
        // Load sequence
        std::ifstream sf(sequences_file); if (!sf) throw std::runtime_error("Cannot open sequences file"); std::string s2((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
        json::array seqs = json::parse(s2).as_object().at("sequences").as_array(); if (seqs.empty()) throw std::runtime_error("No sequences found");
        auto sequence = seqs[0].as_object();

        // Snapshot controls via env
        bool save_last_only = false;
        size_t snapshot_every = 1; // 1 = snapshot every action (default)
        if (const char* slo = std::getenv("SAVE_LAST_ONLY")) {
            if (std::string(slo) == "1") save_last_only = true;
        }
        if (const char* se = std::getenv("SNAPSHOT_EVERY")) {
            try {
                long v = std::stol(se);
                if (v <= 0) { snapshot_every = 0; save_last_only = true; }
                else snapshot_every = static_cast<size_t>(v);
            } catch (...) { /* keep default */ }
        } else if (save_last_only) {
            snapshot_every = 0;
        }

        struct Task { size_t pi; };
        std::vector<Task> tasks; tasks.reserve(pools.size()); for (size_t i = 0; i < pools.size(); ++i) tasks.push_back({i});
        size_t threads = std::max<size_t>(1, std::thread::hardware_concurrency());
        if (const char* thr = std::getenv("CPP_THREADS")) { try { threads = std::max<size_t>(1, std::stoul(thr)); } catch (...) {} }
        std::vector<json::object> results(tasks.size()); std::atomic<size_t> next{0}; std::mutex io_mu;

        auto worker = [&]() {
            for (;;) {
                size_t idx = next.fetch_add(1); if (idx >= tasks.size()) break; auto pool_obj = pools[tasks[idx].pi].as_object();
                std::string pool_name;
                try {
                    pool_name = std::string(pool_obj.at("name").as_string().c_str());
                    {
                        std::lock_guard<std::mutex> lk(io_mu); std::cout << "Processing " << pool_name << "..." << std::endl;
                    }
                    // Normalize params
                    auto init_liq = pool_obj.at("initial_liquidity").as_array();
                    std::array<T,2> precisions = {NumTraits<T>::ONE(), NumTraits<T>::ONE()};
                    // A uses A_MULTIPLIER scale (not 1e18). gamma is unused in math here; keep unscaled.
                    T A;
                    T gamma;
                    if constexpr (std::is_same_v<T, stableswap::uint256>) {
                        A = from_string_scaled_T<stableswap::uint256>(std::string(pool_obj.at("A").as_string().c_str()));
                        gamma = from_string_scaled_T<stableswap::uint256>(std::string(pool_obj.at("gamma").as_string().c_str()));
                    } else {
                        A = static_cast<T>(std::strtold(std::string(pool_obj.at("A").as_string().c_str()).c_str(), nullptr));
                        gamma = static_cast<T>(std::strtold(std::string(pool_obj.at("gamma").as_string().c_str()).c_str(), nullptr));
                    }
                    T mid_fee = from_fee_scaled_T<T>(std::string(pool_obj.at("mid_fee").as_string().c_str()));
                    T out_fee = from_fee_scaled_T<T>(std::string(pool_obj.at("out_fee").as_string().c_str()));
                    // fee_gamma uses PRECISION scale
                    T fee_gamma;
                    if constexpr (std::is_same_v<T, stableswap::uint256>) fee_gamma = from_string_scaled_T<stableswap::uint256>(std::string(pool_obj.at("fee_gamma").as_string().c_str()));
                    else fee_gamma = static_cast<T>(std::strtold(std::string(pool_obj.at("fee_gamma").as_string().c_str()).c_str(), nullptr) / 1e18L);
                    T allowed_extra_profit = from_string_scaled_T<T>(std::string(pool_obj.at("allowed_extra_profit").as_string().c_str()));
                    if constexpr (!std::is_same_v<T, stableswap::uint256>) allowed_extra_profit = static_cast<T>(std::strtold(std::string(pool_obj.at("allowed_extra_profit").as_string().c_str()).c_str(), nullptr) / 1e18L);
                    T adjustment_step = from_string_scaled_T<T>(std::string(pool_obj.at("adjustment_step").as_string().c_str()));
                    if constexpr (!std::is_same_v<T, stableswap::uint256>) adjustment_step = static_cast<T>(std::strtold(std::string(pool_obj.at("adjustment_step").as_string().c_str()).c_str(), nullptr) / 1e18L);
                    // ma_time is in seconds (unscaled)
                    T ma_time;
                    if constexpr (std::is_same_v<T, stableswap::uint256>) ma_time = from_string_scaled_T<stableswap::uint256>(std::string(pool_obj.at("ma_time").as_string().c_str()));
                    else ma_time = static_cast<T>(std::strtold(std::string(pool_obj.at("ma_time").as_string().c_str()).c_str(), nullptr));
                    T initial_price = from_string_scaled_T<T>(std::string(pool_obj.at("initial_price").as_string().c_str()));
                    std::array<T,2> initial_amounts = {from_string_scaled_T<T>(std::string(init_liq[0].as_string().c_str())), from_string_scaled_T<T>(std::string(init_liq[1].as_string().c_str()))};

                    TwoCryptoPoolT<T> pool(precisions, A, gamma, mid_fee, out_fee, fee_gamma, allowed_extra_profit, adjustment_step, ma_time, initial_price);
                    // start timestamp
                    if (sequence.if_contains("start_timestamp")) pool.set_block_timestamp(static_cast<uint64_t>(sequence.at("start_timestamp").as_int64()));
                    (void)pool.add_liquidity(initial_amounts, NumTraits<T>::ZERO());

                json::array states;
                json::object last_state;
                bool have_last_state = false;
                // Initial snapshot unless final-only
                if (snapshot_every != 0) {
                    states.push_back(Report<T>::to_json(pool));
                }
                auto actions = sequence.at("actions").as_array();
                size_t action_idx = 0;
                for (const auto& a : actions) {
                    auto act = a.as_object(); bool success = true; std::string error;
                    try {
                        auto type = act.at("type").as_string();
                        if (type == "exchange") {
                            T i = T(static_cast<int>(act.at("i").as_int64())); T j = T(static_cast<int>(act.at("j").as_int64())); T dx = from_string_scaled_T<T>(std::string(act.at("dx").as_string().c_str()));
                            (void)pool.exchange(i, j, dx, NumTraits<T>::ZERO());
                        } else if (type == "add_liquidity") {
                            auto arr = act.at("amounts").as_array(); std::array<T,2> amts = {from_string_scaled_T<T>(std::string(arr[0].as_string().c_str())), from_string_scaled_T<T>(std::string(arr[1].as_string().c_str()))}; bool donation = act.if_contains("donation") ? act.at("donation").as_bool() : false; (void)pool.add_liquidity(amts, NumTraits<T>::ZERO(), donation);
                        } else if (type == "time_travel") {
                            if (act.if_contains("seconds")) { uint64_t secs = static_cast<uint64_t>(act.at("seconds").as_int64()); if (secs > 0) pool.advance_time(secs); } else if (act.if_contains("timestamp")) { pool.set_block_timestamp(static_cast<uint64_t>(act.at("timestamp").as_int64())); }
                        }
                    } catch (const std::exception& e) { success = false; error = e.what(); }
                    if (snapshot_every == 0) {
                        // Final-only: avoid per-step JSON materialization for speed
                        have_last_state = true;
                    } else {
                        auto st = Report<T>::to_json(pool); st["action_success"] = success; if (!success) st["error"] = error;
                        if (snapshot_every == 1) {
                            states.push_back(st);
                        } else {
                            // push at interval boundaries; always ensure final pushed later
                            if (((action_idx + 1) % snapshot_every) == 0) {
                                states.push_back(st);
                            }
                            last_state = st; have_last_state = true;
                        }
                    }
                    ++action_idx;
                }
                json::object tr; tr["pool_config"] = pool_name; tr["sequence"] = sequence.at("name").as_string();
                json::object res; res["success"] = true;
                if (snapshot_every == 0) {
                    // final-only
                    // Build final snapshot once
                    json::object st = Report<T>::to_json(pool);
                    if (st.if_contains("action_success")) st.erase("action_success");
                    res["final_state"] = st;
                } else {
                    // ensure final step is included for interval mode
                    if (snapshot_every > 1 && (!states.empty()) && have_last_state) {
                        // If last state in list is not the most recent action, append it
                        const auto& back = states.back().as_object();
                        // Heuristic: compare timestamp if present; else always append
                        bool same = false;
                        if (back.if_contains("timestamp") && last_state.if_contains("timestamp")) {
                            same = back.at("timestamp") == last_state.at("timestamp");
                        }
                        if (!same) states.push_back(last_state);
                    }
                    res["states"] = states;
                }
                tr["result"] = res; results[idx] = std::move(tr);
                } catch (const std::exception& e) {
                    // Per-pool failure should not bring down the whole harness
                    json::object tr; tr["pool_config"] = pool_name; tr["sequence"] = sequence.at("name").as_string();
                    json::object res; res["success"] = false; res["error"] = e.what();
                    tr["result"] = res; results[idx] = std::move(tr);
                }
            }
        };
        std::vector<std::thread> ws; ws.reserve(threads); for (size_t t=0;t<threads;++t) ws.emplace_back(worker); for (auto& th:ws) th.join();
        json::array out; for (auto& r : results) out.push_back(r);
        json::object O; O["results"] = out; O["metadata"] = json::object{{"pool_configs_file", pools_file}, {"action_sequences_file", sequences_file},{"total_tests", pools.size()}};
        std::ofstream of(output_file); of << json::serialize(O) << std::endl; return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
}

#if defined(HARNESS_MODE_I)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>" << std::endl; return 1;
    }
    std::string pools = argv[1]; std::string seq = argv[2]; std::string out = argv[3];
    return run_harness<stableswap::uint256>(pools, seq, out);
}
#elif defined(HARNESS_MODE_D)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>" << std::endl; return 1;
    }
    std::string pools = argv[1]; std::string seq = argv[2]; std::string out = argv[3];
    return run_harness<double>(pools, seq, out);
}
#elif defined(HARNESS_MODE_F)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>" << std::endl; return 1;
    }
    std::string pools = argv[1]; std::string seq = argv[2]; std::string out = argv[3];
    return run_harness<float>(pools, seq, out);
}
#elif defined(HARNESS_MODE_LD)
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pools.json> <sequences.json> <output.json>" << std::endl; return 1;
    }
    std::string pools = argv[1]; std::string seq = argv[2]; std::string out = argv[3];
    return run_harness<long double>(pools, seq, out);
}
#else
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <mode:i|d|f|ld> <pools.json> <sequences.json> <output.json>" << std::endl; return 1;
    }
    std::string mode = argv[1]; std::string pools = argv[2]; std::string seq = argv[3]; std::string out = argv[4];
    if (mode == "i") return run_harness<stableswap::uint256>(pools, seq, out);
    if (mode == "d") return run_harness<double>(pools, seq, out);
    if (mode == "f") return run_harness<float>(pools, seq, out);
    if (mode == "ld") return run_harness<long double>(pools, seq, out);
    std::cerr << "Unknown mode: " << mode << std::endl; return 1;
}
#endif
