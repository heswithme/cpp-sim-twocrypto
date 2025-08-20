#include "twocrypto_d.hpp"
#include <iostream>
#include <fstream>
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <cstdlib>
#include <sstream>

// This harness uses double precision for fast approximations.

using twocrypto_d::TwoCryptoPoolD;
namespace json = boost::json;

static constexpr double UNIT = 1e18; // input amounts/virtual price scale
static constexpr double FEE_UNIT = 1e10; // fee params scale

static double to_f(const std::string& s) { return std::strtod(s.c_str(), nullptr) / UNIT; }
static std::string from_f(double v) {
    long double scaled = static_cast<long double>(v) * static_cast<long double>(UNIT);
    if (scaled < 0) scaled = 0;
    std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(0); oss << scaled; return oss.str();
}

struct PoolStateReportF {
    std::array<double, 2> balances;
    std::array<double, 2> xp;
    double D;
    double virtual_price;
    double xcp_profit;
    double price_scale;
    double price_oracle;
    double last_prices;
    double totalSupply;
    double donation_shares;
    double donation_shares_unlocked;
    double donation_protection_expiry_ts;
    double last_donation_release_ts;
    uint64_t timestamp;

    json::object to_json() const {
        json::object obj;
        obj["balances"] = json::array{from_f(balances[0]), from_f(balances[1])};
        obj["xp"] = json::array{from_f(xp[0]), from_f(xp[1])};
        obj["D"] = from_f(D);
        obj["virtual_price"] = from_f(virtual_price);
        obj["xcp_profit"] = from_f(xcp_profit);
        obj["price_scale"] = from_f(price_scale);
        obj["price_oracle"] = from_f(price_oracle);
        obj["last_prices"] = from_f(last_prices);
        obj["totalSupply"] = from_f(totalSupply);
        obj["donation_shares"] = from_f(donation_shares);
        obj["donation_shares_unlocked"] = from_f(donation_shares_unlocked);
        obj["donation_protection_expiry_ts"] = from_f(donation_protection_expiry_ts);
        obj["last_donation_release_ts"] = from_f(last_donation_release_ts);
        obj["timestamp"] = timestamp;  // Let boost::json handle conversion
        return obj;
    }
};

static PoolStateReportF get_pool_state(TwoCryptoPoolD& pool) {
    PoolStateReportF r;
    r.balances = pool.balances;
    r.D = pool.D;
    r.virtual_price = pool.get_virtual_price();
    r.xcp_profit = pool.xcp_profit;
    r.price_scale = pool.cached_price_scale;
    // Use actual cached oracle value (not calculated view value)
    r.price_oracle = pool.cached_price_oracle;
    r.last_prices = pool.last_prices;
    r.totalSupply = pool.totalSupply;
    r.donation_shares = pool.donation_shares;
    r.donation_protection_expiry_ts = pool.donation_protection_expiry_ts;
    r.last_donation_release_ts = pool.last_donation_release_ts;
    r.timestamp = pool.block_timestamp;
    r.xp[0] = pool.balances[0] * pool.precisions[0];
    r.xp[1] = pool.balances[1] * pool.precisions[1] * pool.cached_price_scale;

    if (pool.donation_shares <= 0.0) {
        r.donation_shares_unlocked = 0.0;
    } else {
        double elapsed = static_cast<double>(pool.block_timestamp) - pool.last_donation_release_ts;
        double unlocked = pool.donation_shares * elapsed / pool.donation_duration;
        if (unlocked > pool.donation_shares) unlocked = pool.donation_shares;
        double protection_factor = 0.0;
        if (pool.donation_protection_expiry_ts > static_cast<double>(pool.block_timestamp)) {
            protection_factor = (pool.donation_protection_expiry_ts - static_cast<double>(pool.block_timestamp)) / pool.donation_protection_period;
            if (protection_factor > 1.0) protection_factor = 1.0;
        }
        r.donation_shares_unlocked = unlocked * (1.0 - protection_factor);
    }
    return r;
}

static json::object process_pool_sequence(const json::object& pool_config, const json::object& sequence) {
    json::object result; result["pool_name"] = pool_config.at("name");
    try {
        double A = std::strtod(pool_config.at("A").as_string().c_str(), nullptr);
        double gamma = std::strtod(pool_config.at("gamma").as_string().c_str(), nullptr);
        double mid_fee = std::strtod(pool_config.at("mid_fee").as_string().c_str(), nullptr) / FEE_UNIT;
        double out_fee = std::strtod(pool_config.at("out_fee").as_string().c_str(), nullptr) / FEE_UNIT;
        double fee_gamma = std::strtod(pool_config.at("fee_gamma").as_string().c_str(), nullptr) / UNIT;
        double allowed_extra_profit = std::strtod(pool_config.at("allowed_extra_profit").as_string().c_str(), nullptr) / UNIT;
        double adjustment_step = std::strtod(pool_config.at("adjustment_step").as_string().c_str(), nullptr) / UNIT;
        double ma_time = std::strtod(pool_config.at("ma_time").as_string().c_str(), nullptr);
        double initial_price = to_f(std::string(pool_config.at("initial_price").as_string().c_str()));

        auto init_liq = pool_config.at("initial_liquidity").as_array();
        std::array<double, 2> initial_amounts = { to_f(std::string(init_liq[0].as_string().c_str())),
                                                  to_f(std::string(init_liq[1].as_string().c_str())) };
        std::array<double, 2> precisions = {1.0, 1.0};
        TwoCryptoPoolD pool(precisions, A, gamma, mid_fee, out_fee, fee_gamma, allowed_extra_profit, adjustment_step, ma_time, initial_price);

        bool save_last_only = false;
        if (const char* env = std::getenv("SAVE_LAST_ONLY")) {
            save_last_only = std::string(env) == "1";
        }
        json::array states;
        json::object last_state;
        bool all_success = true;
        if (sequence.if_contains("start_timestamp")) {
            uint64_t start_ts = static_cast<uint64_t>(sequence.at("start_timestamp").as_int64());
            pool.set_block_timestamp(start_ts); pool.last_timestamp = start_ts;
        }
        (void)pool.add_liquidity(initial_amounts, 0.0);
        if (!save_last_only) {
            states.push_back(get_pool_state(pool).to_json());
        }

        long snapshot_every = 1; if (const char* s = std::getenv("SNAPSHOT_EVERY")) { try { snapshot_every = std::stol(s); } catch (...) {} }
        if (std::getenv("SAVE_LAST_ONLY") && std::string(std::getenv("SAVE_LAST_ONLY")) == "1") snapshot_every = 0;
        auto actions = sequence.at("actions").as_array();
        size_t action_idx = 0;
        for (const auto& action : actions) {
            auto act = action.as_object(); bool success = true; std::string error;
            try {
                auto type = act.at("type").as_string();
                if (type == "exchange") {
                    double i = static_cast<double>(act.at("i").as_int64());
                    double j = static_cast<double>(act.at("j").as_int64());
                    double dx = to_f(std::string(act.at("dx").as_string().c_str()));
                    (void)pool.exchange(i, j, dx, 0.0);
                } else if (type == "add_liquidity") {
                    auto arr = act.at("amounts").as_array();
                    std::array<double,2> amts = { to_f(std::string(arr[0].as_string().c_str())),
                                                  to_f(std::string(arr[1].as_string().c_str())) };
                    bool donation = act.contains("donation") && act.at("donation").as_bool();
                    (void)pool.add_liquidity(amts, 0.0, donation);
                } else if (type == "time_travel") {
                    if (act.if_contains("seconds")) {
                        uint64_t secs = static_cast<uint64_t>(act.at("seconds").as_int64());
                        if (secs > 0) pool.advance_time(secs);
                    } else if (act.if_contains("timestamp")) {
                        uint64_t ts = static_cast<uint64_t>(act.at("timestamp").as_int64());
                        pool.set_block_timestamp(ts);
                    }
                }
            } catch (const std::exception& e) { success = false; error = e.what(); }
            if (!success) all_success = false;
            
            // Determine if we should take a snapshot
            bool do_snap = false;
            if (snapshot_every == 1) {
                // Capture every state
                do_snap = true;
            } else if (snapshot_every > 1 && ((action_idx + 1) % snapshot_every == 0)) {
                // Capture every N states
                do_snap = true;
            }
            // Note: if snapshot_every == 0, we don't capture intermediate states
            
            if (do_snap) {
                auto st = get_pool_state(pool).to_json();
                st["action_success"] = success; if (!success) st["error"] = error; states.push_back(st);
            }
            action_idx++;
        }

        // Always capture final state when snapshot_every == 0 or is the last action
        if (snapshot_every == 0 || (snapshot_every > 1 && (action_idx % snapshot_every != 0))) {
            auto final_state = get_pool_state(pool).to_json();
            final_state["action_success"] = all_success;
            states.push_back(final_state);
        }
        
        if (save_last_only) {
            // For backwards compatibility with SAVE_LAST_ONLY
            result["final_state"] = states.empty() ? get_pool_state(pool).to_json() : states.back();
        } else {
            result["states"] = states;
        }
        result["success"] = all_success;
    } catch (const std::exception& e) {
        result["success"] = false; result["error"] = e.what();
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 4) { std::cerr << "Usage: " << argv[0] << " <pool_configs.json> <action_sequences.json> <output_results.json>" << std::endl; return 1; }
    std::string pool_configs_file = argv[1]; std::string action_sequences_file = argv[2]; std::string output_file = argv[3];
    try {
        std::ifstream pool_configs_stream(pool_configs_file); if (!pool_configs_stream) throw std::runtime_error("Cannot open pool configs file: " + pool_configs_file);
        std::string pool_configs_str((std::istreambuf_iterator<char>(pool_configs_stream)), std::istreambuf_iterator<char>());
        json::value pool_configs_data = json::parse(pool_configs_str);
        json::array pools = pool_configs_data.as_object().at("pools").as_array();
        std::ifstream action_sequences_stream(action_sequences_file); if (!action_sequences_stream) throw std::runtime_error("Cannot open action sequences file: " + action_sequences_file);
        std::string action_sequences_str((std::istreambuf_iterator<char>(action_sequences_stream)), std::istreambuf_iterator<char>());
        json::value action_sequences_data = json::parse(action_sequences_str);
        json::array sequences = action_sequences_data.as_object().at("sequences").as_array();
        if (sequences.empty()) throw std::runtime_error("No sequences found in sequences.json");

        const char* only_pool = std::getenv("FILTER_POOL");
        struct Task { size_t pi; size_t si; };
        std::vector<Task> tasks; for (size_t pi = 0; pi < pools.size(); ++pi) { std::string pool_name = pools[pi].as_object().at("name").as_string().c_str(); if (only_pool && pool_name != std::string(only_pool)) continue; tasks.push_back({pi, 0}); }
        size_t threads = std::thread::hardware_concurrency(); if (threads == 0) threads = 4; if (const char* thr = std::getenv("CPP_THREADS")) { try { threads = std::max<size_t>(1, std::stoul(thr)); } catch (...) {} }
        std::cout << "Running DOUBLE harness with " << threads << " worker threads (" << tasks.size() << " tasks)" << std::endl;

        std::vector<json::object> results_vec(tasks.size()); std::atomic<size_t> next{0}; std::mutex io_mu;
        auto worker = [&]() {
            for (;;) {
                size_t idx = next.fetch_add(1); if (idx >= tasks.size()) break; auto [pi, si] = tasks[idx]; auto pool_obj = pools[pi].as_object(); auto seq_obj = sequences[0].as_object(); std::string pool_name = pool_obj.at("name").as_string().c_str(); std::string seq_name = seq_obj.at("name").as_string().c_str();
                { std::lock_guard<std::mutex> lk(io_mu); std::cout << "Processing " << pool_name << " with " << seq_name << "..." << std::endl; }
                json::object test_result; test_result["pool_config"] = pool_name; test_result["sequence"] = seq_name; test_result["result"] = process_pool_sequence(pool_obj, seq_obj); results_vec[idx] = std::move(test_result);
            }
        };
        std::vector<std::thread> workers; for (size_t t = 0; t < threads; ++t) workers.emplace_back(worker); for (auto& th : workers) th.join();
        json::array results; for (auto& obj : results_vec) results.push_back(obj);
        json::object output; output["results"] = results; output["metadata"] = {{"pool_configs_file", pool_configs_file},{"action_sequences_file", action_sequences_file},{"num_pools", pools.size()},{"num_sequences", 1},{"total_tests", pools.size()}};
        std::ofstream out_file(output_file); if (!out_file) throw std::runtime_error("Cannot open output file: " + output_file); out_file << json::serialize(output) << std::endl;
        std::cout << "\n✓ Processed " << (pools.size() * sequences.size()) << " pool-sequence combinations" << std::endl; std::cout << "✓ Results written to " << output_file << std::endl;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
    return 0;
}
