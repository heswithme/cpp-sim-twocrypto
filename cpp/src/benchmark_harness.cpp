#include "twocrypto.hpp"
#include <iostream>
#include <fstream>
#include <boost/json.hpp>
#include <boost/json/src.hpp>

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
    return a + (b << 85) + (c << 170);
}

// Structure to hold pool state for reporting
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
        return obj;
    }
};

// Get pool state report
PoolStateReport get_pool_state(TwoCryptoPool& pool) {
    PoolStateReport report;
    
    report.balances = pool.balances;
    report.D = pool.D;
    report.virtual_price = pool.get_virtual_price();
    report.xcp_profit = pool.xcp_profit;
    report.price_scale = pool.cached_price_scale;
    report.price_oracle = pool.cached_price_oracle;
    report.last_prices = pool.last_prices;
    report.totalSupply = pool.totalSupply;
    
    // Calculate xp
    report.xp[0] = pool.balances[0] * pool.precisions[0];
    report.xp[1] = pool.balances[1] * pool.precisions[1] * pool.cached_price_scale / PRECISION();
    
    return report;
}

// Process a pool configuration with action sequence
json::object process_pool_sequence(
    const json::object& pool_config,
    const json::array& actions
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
        
        // Array to store states after each action
        json::array states;
        
        // Add initial liquidity
        uint256 lp_tokens = pool.add_liquidity(initial_amounts, 0);
        
        // Store state after initial liquidity
        states.push_back(get_pool_state(pool).to_json());
        
        // Process each action
        for (const auto& action : actions) {
            auto act = action.as_object();
            
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
                if (act.at("type").as_string() == "exchange") {
                    uint256 i = act.at("i").as_int64();
                    uint256 j = act.at("j").as_int64();
                    uint256 dx = str_to_uint256(act.at("dx").as_string().c_str());
                    
                    auto [dy, fee, price_scale] = pool.exchange(i, j, dx, 0);
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
        
        // Process each pool with each sequence
        json::array results;
        
        for (const auto& pool_config : pools) {
            for (const auto& sequence : sequences) {
                auto seq_obj = sequence.as_object();
                std::string pool_name = pool_config.as_object().at("name").as_string().c_str();
                std::string seq_name = seq_obj.at("name").as_string().c_str();
                
                std::cout << "Processing " << pool_name << " with " << seq_name << "..." << std::endl;
                
                json::object test_result;
                test_result["pool_config"] = pool_name;
                test_result["sequence"] = seq_name;
                test_result["result"] = process_pool_sequence(
                    pool_config.as_object(),
                    seq_obj.at("actions").as_array()
                );
                
                results.push_back(test_result);
            }
        }
        
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