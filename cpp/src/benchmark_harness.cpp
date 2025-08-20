#include "twocrypto.hpp"
#include <iostream>
#include <fstream>
#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

using namespace twocrypto;
namespace json = boost::json;

// Helper to convert string to uint256
uint256 str_to_uint256(const std::string& s) {
    return uint256(s);
}

// Helper to convert uint256 to string
std::string uint256_to_str(const uint256& value) {
    return value.str();
}

// Pack parameters as done in Vyper factory
uint256 pack_2(const uint256& a, const uint256& b) {
    return a + (b << 128);
}

uint256 pack_3(const uint256& a, const uint256& b, const uint256& c) {
    // Matches Vyper: (x0 << 128) | (x1 << 64) | x2
    return (a << 128) + (b << 64) + c;
}

// Structure to hold pool state for reporting
// NOTE: Includes extra donation + timestamp fields to aid parity debugging.
struct PoolStateReport {
    std::array<uint256, 2> balances;
    std::array<uint256, 2> xp;
    uint256 D;
    uint256 virtual_price;
    uint256 xcp_profit;
    uint256 price_scale;
    uint256 price_oracle;
    uint256 last_prices;
    uint256 totalSupply;
    uint256 donation_shares;
    uint256 donation_shares_unlocked;
    uint256 donation_protection_expiry_ts;
    uint256 last_donation_release_ts;
    uint64_t timestamp;
    
    json::object to_json() const {
        json::object obj;
        obj["balances"] = json::array{uint256_to_str(balances[0]), uint256_to_str(balances[1])};
        obj["xp"] = json::array{uint256_to_str(xp[0]), uint256_to_str(xp[1])};
        obj["D"] = uint256_to_str(D);
        obj["virtual_price"] = uint256_to_str(virtual_price);
        obj["xcp_profit"] = uint256_to_str(xcp_profit);
        obj["price_scale"] = uint256_to_str(price_scale);
        obj["price_oracle"] = uint256_to_str(price_oracle);
        obj["last_prices"] = uint256_to_str(last_prices);
        obj["totalSupply"] = uint256_to_str(totalSupply);
        obj["donation_shares"] = uint256_to_str(donation_shares);
        obj["donation_shares_unlocked"] = uint256_to_str(donation_shares_unlocked);
        obj["donation_protection_expiry_ts"] = uint256_to_str(donation_protection_expiry_ts);
        obj["last_donation_release_ts"] = uint256_to_str(last_donation_release_ts);
        obj["timestamp"] = std::to_string(timestamp);
        return obj;
    }
};

// Get pool state report
// NOTE: We fetch a minimal but sufficient set of fields to compare parity
// with the Vyper runner. If more fields are needed, add here and in the
// Vyper runner snapshot for like-for-like comparisons.
PoolStateReport get_pool_state(TwoCryptoPool& pool) {
    PoolStateReport report;
    
    report.balances = pool.balances;
    report.D = pool.D;
    report.virtual_price = pool.get_virtual_price();
    report.xcp_profit = pool.xcp_profit;
    report.price_scale = pool.cached_price_scale;
    // Match Vyper's price_oracle() view: compute EMA lazily on read
    {
        uint256 price_oracle = pool.cached_price_oracle;
        uint256 price_scale = pool.cached_price_scale;
        uint256 last_prices = pool.last_prices;
        uint256 last_ts = pool.last_timestamp;
        if (last_ts < pool.block_timestamp) {
            auto rebal = pool.packed_rebalancing_params;
            // unpack ma_time from packed_rebalancing_params (x2)
            uint256 mask64 = (uint256(1) << 64) - 1;
            uint256 ma_time = rebal & mask64; // x2
            auto neg = stableswap::int256(- (stableswap::int256(pool.block_timestamp - last_ts) * stableswap::int256(twocrypto::PRECISION()) / stableswap::int256(ma_time)));
            uint256 alpha = stableswap::StableswapMath::wad_exp(neg);
            uint256 capped = last_prices;
            if (capped > 2 * price_scale) capped = 2 * price_scale;
            price_oracle = (capped * (twocrypto::PRECISION() - alpha) + price_oracle * alpha) / twocrypto::PRECISION();
        }
        report.price_oracle = price_oracle;
    }
    report.last_prices = pool.last_prices;
    report.totalSupply = pool.totalSupply;
    report.donation_shares = pool.donation_shares;
    report.donation_protection_expiry_ts = pool.donation_protection_expiry_ts;
    report.last_donation_release_ts = pool.last_donation_release_ts;
    report.timestamp = pool.block_timestamp;
    
    // Calculate xp
    report.xp[0] = pool.balances[0] * pool.precisions[0];
    report.xp[1] = pool.balances[1] * pool.precisions[1] * pool.cached_price_scale / PRECISION();
    
    // Unlocked donation shares (with protection)
    // replicate _donation_shares(True)
    if (pool.donation_shares == 0) {
        report.donation_shares_unlocked = 0;
    } else {
        uint256 elapsed = pool.block_timestamp - pool.last_donation_release_ts;
        uint256 unlocked = pool.donation_shares * elapsed / pool.donation_duration;
        if (unlocked > pool.donation_shares) unlocked = pool.donation_shares;
        uint256 protection_factor = 0;
        if (pool.donation_protection_expiry_ts > pool.block_timestamp) {
            protection_factor = (pool.donation_protection_expiry_ts - pool.block_timestamp) * PRECISION() / pool.donation_protection_period;
            if (protection_factor > PRECISION()) protection_factor = PRECISION();
        }
        report.donation_shares_unlocked = unlocked * (PRECISION() - protection_factor) / PRECISION();
    }
    
    return report;
}

// Process a single pool configuration with one action sequence.
// Elegance guidelines:
// - Keep this a straightforward state machine over JSON-defined actions.
// - Avoid adding abstractions: sequences are small and clarity is preferred.
// - Use explicit comments where behavior may be non-obvious (e.g., timestamps).
json::object process_pool_sequence(
    const json::object& pool_config,
    const json::object& sequence
) {
    json::object result;
    result["pool_name"] = pool_config.at("name");
    
    try {
        // Extract pool parameters
        uint256 A = str_to_uint256(pool_config.at("A").as_string().c_str());
        uint256 gamma = str_to_uint256(pool_config.at("gamma").as_string().c_str());
        uint256 mid_fee = str_to_uint256(pool_config.at("mid_fee").as_string().c_str());
        uint256 out_fee = str_to_uint256(pool_config.at("out_fee").as_string().c_str());
        uint256 fee_gamma = str_to_uint256(pool_config.at("fee_gamma").as_string().c_str());
        uint256 allowed_extra_profit = str_to_uint256(pool_config.at("allowed_extra_profit").as_string().c_str());
        uint256 adjustment_step = str_to_uint256(pool_config.at("adjustment_step").as_string().c_str());
        uint256 ma_time = str_to_uint256(pool_config.at("ma_time").as_string().c_str());
        uint256 initial_price = str_to_uint256(pool_config.at("initial_price").as_string().c_str());
        
        // Initial liquidity amounts
        auto init_liq = pool_config.at("initial_liquidity").as_array();
        std::array<uint256, 2> initial_amounts = {
            str_to_uint256(init_liq[0].as_string().c_str()),
            str_to_uint256(init_liq[1].as_string().c_str())
        };
        
        // Pack parameters
        uint256 packed_gamma_A = pack_2(gamma, A);
        uint256 packed_fee_params = pack_3(mid_fee, out_fee, fee_gamma);
        uint256 packed_rebalancing_params = pack_3(allowed_extra_profit, adjustment_step, ma_time);
        
        // Assume 18-decimal tokens (precision = 1)
        std::array<uint256, 2> precisions = {uint256(1), uint256(1)};
        
        // Create pool
        TwoCryptoPool pool(precisions, packed_gamma_A, packed_fee_params, 
                          packed_rebalancing_params, initial_price);
        // Enable tracing via env var
        const char* trace_env = std::getenv("TRACE");
        bool trace_enabled = trace_env && std::string(trace_env) == "1";
        if (trace_enabled) {
            pool.set_trace(true);
            std::cout << "TRACE enabled for "
                      << pool_config.at("name").as_string().c_str() << " / "
                      << sequence.at("name").as_string().c_str() << std::endl;
        }
        
        // Array to store states after each action
        json::array states;

        // Sync start timestamp if provided.
        // IMPORTANT: We set both the pool's block time and last_timestamp to
        // the same value before any operation to align oracle timing across
        // harnesses, since the Vyper runner also sets env.timestamp pre-deploy.
        if (sequence.if_contains("start_timestamp")) {
            uint64_t start_ts = static_cast<uint64_t>(sequence.at("start_timestamp").as_int64());
            pool.set_block_timestamp(start_ts);
            pool.last_timestamp = start_ts;
        }

        // Add initial liquidity
        uint256 lp_tokens = pool.add_liquidity(initial_amounts, 0);
        
        // Store state after initial liquidity
        states.push_back(get_pool_state(pool).to_json());
        
        // Process each action
        auto actions = sequence.at("actions").as_array();
        for (const auto& action : actions) {
            auto act = action.as_object();
            if (trace_enabled) {
                std::cout << "TRACE action ts=" << pool.block_timestamp << " -> ";
                auto type = act.at("type").as_string();
                if (type == "exchange") {
                    std::cout << "exchange i=" << act.at("i").as_int64()
                              << " j=" << act.at("j").as_int64()
                              << " dx=" << act.at("dx").as_string().c_str();
                } else if (type == "add_liquidity") {
                    std::cout << "add_liquidity donation="
                              << (act.contains("donation") && act.at("donation").as_bool());
                } else if (type == "time_travel") {
                    std::cout << "time_travel to=" << act.at("timestamp").as_int64();
                }
                std::cout << std::endl;
            }
            
            // Apply time delta if present
            if (act.contains("time_delta")) {
                uint64_t time_delta = act.at("time_delta").as_int64();
                if (time_delta > 0) {
                    pool.advance_time(time_delta);
                }
            }
            
            bool success = true;
            std::string error;
            
            try {
                auto type = act.at("type").as_string();
                if (type == "exchange") {
                    uint256 i = act.at("i").as_int64();
                    uint256 j = act.at("j").as_int64();
                    uint256 dx = str_to_uint256(act.at("dx").as_string().c_str());
                    
                    auto [dy, fee, price_scale] = pool.exchange(i, j, dx, 0);
                } else if (type == "add_liquidity") {
                    auto arr = act.at("amounts").as_array();
                    std::array<uint256,2> amts = {
                        str_to_uint256(arr[0].as_string().c_str()),
                        str_to_uint256(arr[1].as_string().c_str())
                    };
                    bool donation = false;
                    if (act.contains("donation")) {
                        donation = act.at("donation").as_bool();
                    }
                    (void)pool.add_liquidity(amts, 0, donation);
                } else if (type == "time_travel") {
                    // Absolute timestamp jump.
                    // IMPORTANT: Do NOT update last_timestamp here. The EMA
                    // logic in the pool should decide when to use the time
                    // delta, just like Vyper's last_timestamp is only moved
                    // inside tweak_price.
                    uint64_t ts = static_cast<uint64_t>(act.at("timestamp").as_int64());
                    pool.set_block_timestamp(ts);
                }
            } catch (const std::exception& e) {
                success = false;
                error = e.what();
            }
            
            // Store state after action
            auto state_json = get_pool_state(pool).to_json();
            state_json["action_success"] = success;
            if (!success) {
                state_json["error"] = error;
            }
            states.push_back(state_json);
        }
        
        result["states"] = states;
        result["success"] = true;
        
    } catch (const std::exception& e) {
        result["success"] = false;
        result["error"] = e.what();
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <pool_configs.json> <action_sequences.json> <output_results.json>" << std::endl;
        return 1;
    }
    
    std::string pool_configs_file = argv[1];
    std::string action_sequences_file = argv[2];
    std::string output_file = argv[3];
    
    try {
        // Load pool configurations
        std::ifstream pool_configs_stream(pool_configs_file);
        if (!pool_configs_stream) {
            throw std::runtime_error("Cannot open pool configs file: " + pool_configs_file);
        }
        std::string pool_configs_str((std::istreambuf_iterator<char>(pool_configs_stream)),
                                    std::istreambuf_iterator<char>());
        json::value pool_configs_data = json::parse(pool_configs_str);
        json::array pools = pool_configs_data.as_object().at("pools").as_array();
        
        // Load action sequences
        std::ifstream action_sequences_stream(action_sequences_file);
        if (!action_sequences_stream) {
            throw std::runtime_error("Cannot open action sequences file: " + action_sequences_file);
        }
        std::string action_sequences_str((std::istreambuf_iterator<char>(action_sequences_stream)),
                                        std::istreambuf_iterator<char>());
        json::value action_sequences_data = json::parse(action_sequences_str);
        json::array sequences = action_sequences_data.as_object().at("sequences").as_array();
        
        // Build tasks and run with a thread pool
        const char* only_pool = std::getenv("TRACE_ONLY_POOL");
        const char* only_seq = std::getenv("TRACE_ONLY_SEQUENCE");

        struct Task { size_t pi; size_t si; };
        std::vector<Task> tasks;
        tasks.reserve(pools.size() * sequences.size());
        for (size_t pi = 0; pi < pools.size(); ++pi) {
            std::string pool_name = pools[pi].as_object().at("name").as_string().c_str();
            if (only_pool && pool_name != std::string(only_pool)) continue;
            for (size_t si = 0; si < sequences.size(); ++si) {
                std::string seq_name = sequences[si].as_object().at("name").as_string().c_str();
                if (only_seq && seq_name != std::string(only_seq)) continue;
                tasks.push_back({pi, si});
            }
        }

        size_t threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4;
        if (const char* thr = std::getenv("CPP_THREADS")) {
            try { threads = std::max<size_t>(1, std::stoul(thr)); } catch (...) {}
        }
        std::cout << "Running with " << threads << " worker threads (" << tasks.size() << " tasks)" << std::endl;

        std::vector<json::object> results_vec(tasks.size());
        std::atomic<size_t> next{0};
        std::mutex io_mu;

        auto worker = [&]() {
            for (;;) {
                size_t idx = next.fetch_add(1);
                if (idx >= tasks.size()) break;
                auto [pi, si] = tasks[idx];
                auto pool_obj = pools[pi].as_object();
                auto seq_obj = sequences[si].as_object();
                std::string pool_name = pool_obj.at("name").as_string().c_str();
                std::string seq_name = seq_obj.at("name").as_string().c_str();
                {
                    std::lock_guard<std::mutex> lk(io_mu);
                    std::cout << "Processing " << pool_name << " with " << seq_name << "..." << std::endl;
                }
                json::object test_result;
                test_result["pool_config"] = pool_name;
                test_result["sequence"] = seq_name;
                test_result["result"] = process_pool_sequence(pool_obj, seq_obj);
                results_vec[idx] = std::move(test_result);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (size_t t = 0; t < threads; ++t) workers.emplace_back(worker);
        for (auto& th : workers) th.join();

        json::array results;
        results.reserve(results_vec.size());
        for (auto& obj : results_vec) results.push_back(obj);
        
        // Prepare output
        json::object output;
        output["results"] = results;
        output["metadata"] = {
            {"pool_configs_file", pool_configs_file},
            {"action_sequences_file", action_sequences_file},
            {"num_pools", pools.size()},
            {"num_sequences", sequences.size()},
            {"total_tests", pools.size() * sequences.size()}
        };
        
        // Write output
        std::ofstream out_file(output_file);
        if (!out_file) {
            throw std::runtime_error("Cannot open output file: " + output_file);
        }
        
        // Boost.JSON doesn't have built-in pretty printing in the version we're using
        // Just write the compact JSON
        out_file << json::serialize(output) << std::endl;
        
        std::cout << "\n✓ Processed " << (pools.size() * sequences.size()) 
                  << " pool-sequence combinations" << std::endl;
        std::cout << "✓ Results written to " << output_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
