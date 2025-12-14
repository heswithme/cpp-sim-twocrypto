// CLI argument parsing
#pragma once

#include <cstdint>
#include <string>
#include <thread>

namespace arb {
namespace harness {

struct CliArgs {
    // Positional arguments
    std::string pools_path;
    std::string candles_path;
    std::string out_path;

    // Options
    size_t max_candles{0};           // 0 = all
    bool save_actions{false};
    bool use_events{false};          // --events: load events directly instead of candles
    double min_swap_frac{1e-6};
    double max_swap_frac{1.0};
    size_t n_threads{std::thread::hardware_concurrency()};
    double candle_filter_pct{99.0};  // squeeze high/low within +/-X% of (O+C)/2
    uint64_t dustswap_freq_s{3600};  // EMA tick cadence when idle
    
    // User swap settings
    uint64_t user_swap_freq_s{0};    // 0 = disabled
    double user_swap_size_frac{0.01};
    double user_swap_thresh{0.05};   // max relative deviation vs CEX
    
    // APY window settings
    double apy_period_days{7.0};
    int apy_period_cap_pct{100};
    
    // Detailed logging
    bool detailed_log{false};  // write detailed_log.json next to output
    
    // Validation
    bool valid{false};
    std::string error_msg;
};

// Parse command line arguments
// Returns CliArgs with valid=true on success, valid=false with error_msg on failure
CliArgs parse_cli(int argc, char* argv[]);

// Print usage message
void print_usage(const char* prog_name);

} // namespace harness
} // namespace arb
