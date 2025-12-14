// CLI argument parsing implementation

#include "harness/cli.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace arb {
namespace harness {

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name
              << " <pools.json> <candles_or_events.json> <output.json>\n"
              << "       [--n-candles N] [--save-actions] [--events]\n"
              << "       [--min-swap F] [--max-swap F]\n"
              << "       [--threads N | -n N] [--candle-filter PCT]\n"
              << "       [--dustswapfreq S]\n"
              << "       [--userswapfreq S] [--userswapsize F] [--userswapthresh F]\n"
              << "       [--apy-period-days D] [--apy-period-cap PCT]\n"
              << "       [--detailed-log]\n";
}

CliArgs parse_cli(int argc, char* argv[]) {
    CliArgs args{};

    if (argc < 4) {
        args.valid = false;
        args.error_msg = "Not enough arguments (need pools.json, candles.json, output.json)";
        return args;
    }

    args.pools_path = argv[1];
    args.candles_path = argv[2];
    args.out_path = argv[3];

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        try {
            if (arg == "--n-candles" && i + 1 < argc) {
                args.max_candles = static_cast<size_t>(std::stoll(argv[++i]));
            } else if (arg == "--save-actions") {
                args.save_actions = true;
            } else if (arg == "--events") {
                args.use_events = true;
            } else if (arg == "--min-swap" && i + 1 < argc) {
                args.min_swap_frac = std::stod(argv[++i]);
            } else if (arg == "--max-swap" && i + 1 < argc) {
                args.max_swap_frac = std::stod(argv[++i]);
            } else if ((arg == "--threads" || arg == "-n") && i + 1 < argc) {
                args.n_threads = static_cast<size_t>(std::stoll(argv[++i]));
            } else if (arg == "--candle-filter" && i + 1 < argc) {
                args.candle_filter_pct = std::stod(argv[++i]);
            } else if (arg == "--dustswapfreq" && i + 1 < argc) {
                args.dustswap_freq_s = static_cast<uint64_t>(std::stoll(argv[++i]));
            } else if (arg == "--userswapfreq" && i + 1 < argc) {
                args.user_swap_freq_s = static_cast<uint64_t>(std::stoll(argv[++i]));
            } else if (arg == "--userswapsize" && i + 1 < argc) {
                args.user_swap_size_frac = std::stod(argv[++i]);
            } else if (arg == "--userswapthresh" && i + 1 < argc) {
                args.user_swap_thresh = std::stod(argv[++i]);
            } else if (arg == "--apy-period-days" && i + 1 < argc) {
                args.apy_period_days = std::stod(argv[++i]);
            } else if (arg == "--apy-period-cap" && i + 1 < argc) {
                args.apy_period_cap_pct = std::stoi(argv[++i]);
            } else if (arg == "--detailed-log") {
                args.detailed_log = true;
            }
            // Unknown flags are silently ignored (matches original behavior)
        } catch (...) {
            // Ignore parse errors for individual flags (matches original behavior)
        }
    }

    // Clamp negative values
    if (args.user_swap_size_frac < 0) args.user_swap_size_frac = 0;
    if (args.user_swap_thresh < 0) args.user_swap_thresh = 0;
    if (args.n_threads == 0) args.n_threads = 1;

    args.valid = true;
    return args;
}

} // namespace harness
} // namespace arb
