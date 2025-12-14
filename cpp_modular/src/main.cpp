// Modular arb_harness - Entry point with compile-time numeric type selection
//
// Build targets:
//   arb_harness    - double (default)
//   arb_harness_f  - float
//   arb_harness_ld - long double

#include <iostream>
#include <iomanip>
#include <limits>

#include "harness/cli.hpp"
#include "harness/runner.hpp"
#include "harness/output.hpp"
#include "harness/detailed_output.hpp"
#include "events/loader.hpp"
#include "pools/config.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"
#include "pools/twocrypto_fx/helpers.hpp"
#include "trading/costs.hpp"
#include "trading/arbitrageur.hpp"

// Compile-time numeric type selection (floating-only)
#if defined(ARB_MODE_F)
using RealT = float;
static constexpr const char* TYPE_NAME = "float";
#elif defined(ARB_MODE_LD)
using RealT = long double;
static constexpr const char* TYPE_NAME = "long double";
#else
using RealT = double;
static constexpr const char* TYPE_NAME = "double";
#endif

// Helper to print a value
template <typename T>
void print_value(const char* name, const T& val) {
    std::cout << "  " << name << " = " << std::setprecision(12) << val << "\n";
}

// Pool test (floating types)
template <typename T>
void test_pool(T cex_price = T(-1)) {
    using Pool = arb::pools::twocrypto_fx::TwoCryptoPool<T>;
    using Traits = arb::pools::twocrypto_fx::PoolTraits<T>;

    std::array<T, 2> precisions = {Traits::ONE(), Traits::ONE()};

    T A = T(10000.0);
    T gamma = T(1e-5);
    T mid_fee = T(0.0001);
    T out_fee = T(0.0006);
    T fee_gamma = T(0.00023);
    T allowed_extra_profit = T(1e-8);
    T adjustment_step = T(0.0001);
    T ma_time = T(600.0);
    T initial_price = T(1.08);

    Pool pool(
        precisions,
        A, gamma,
        mid_fee, out_fee, fee_gamma,
        allowed_extra_profit, adjustment_step, ma_time,
        initial_price
    );

    pool.set_block_timestamp(1700000000);

    std::cout << "Pool created with initial_price:\n";
    print_value("cached_price_scale", pool.cached_price_scale);
    print_value("cached_price_oracle", pool.cached_price_oracle);

    T amount0 = T(10000.0);
    T amount1 = T(10000.0 / 1.08);
    std::array<T, 2> amounts = {amount0, amount1};
    T min_mint = Traits::ZERO();

    T lp_tokens = pool.add_liquidity(amounts, min_mint);

    std::cout << "\nAfter add_liquidity:\n";
    print_value("LP tokens minted", lp_tokens);
    print_value("totalSupply", pool.totalSupply);
    print_value("D", pool.D);
    print_value("balances[0]", pool.balances[0]);
    print_value("balances[1]", pool.balances[1]);
    print_value("virtual_price", pool.get_virtual_price());

    // Simulate a small exchange (no state change)
    T dx = T(100.0);
    auto sim = arb::pools::twocrypto_fx::simulate_exchange_once(pool, /*i=*/0, /*j=*/1, dx);
    std::cout << "\nSimulated exchange (0 -> 1):\n";
    print_value("dx", dx);
    print_value("dy_after_fee", sim.first);
    print_value("fee_tokens", sim.second);

    // Arbitrage decision
    if (cex_price > T(0)) {
        arb::trading::Costs<T> costs{};
        auto dec = arb::trading::decide_trade(
            pool, cex_price, costs,
            std::numeric_limits<T>::infinity(),
            T(0.001), T(0.1)
        );
        std::cout << "\nArb decision:\n";
        std::cout << "  do_trade = " << dec.do_trade << "\n";
        print_value("  dx", dec.dx);
        print_value("  profit_coin0", dec.profit);
        print_value("  fee_tokens", dec.fee_tokens);
    }

    if (pool.D > Traits::ZERO()) {
        std::cout << "\nPool test: PASSED (D > 0)\n";
    } else {
        std::cout << "\nPool test: FAILED (D <= 0)\n";
    }
}

int main(int argc, char* argv[]) {
    // Parse CLI arguments
    auto args = arb::harness::parse_cli(argc, argv);
    
    if (!args.valid) {
        // For now, if not enough args, run pool test
        if (argc < 2) {
            std::cout << "arb_harness_mod: " << TYPE_NAME << "\n";
            std::cout << "\n--- Pool Test ---\n";
            test_pool<RealT>();
            arb::harness::print_usage(argv[0]);
            return 0;
        }
        std::cerr << "Error: " << args.error_msg << "\n";
        arb::harness::print_usage(argv[0]);
        return 1;
    }

    try {
        auto t_read0 = std::chrono::high_resolution_clock::now();
        
        // Load candles+generate events, or load events directly if --events flag
        std::vector<arb::Event> events;
        if (args.use_events) {
            events = arb::load_events(args.candles_path, args.max_candles);
            std::cout << "loaded " << events.size() << " events from " << args.candles_path << "\n";
        } else {
            auto candles = arb::load_candles(args.candles_path, args.max_candles, args.candle_filter_pct / 100.0);
            events = arb::gen_events(candles);
            std::cout << "loaded " << candles.size() << " candles -> "
                      << events.size() << " events from " << args.candles_path << "\n";
        }
        
        auto t_read1 = std::chrono::high_resolution_clock::now();
        double candles_read_ms = std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();

        // Load pool configs from JSON
        auto pool_configs = arb::pools::load_pool_configs<RealT>(args.pools_path);
        if (pool_configs.empty()) {
            throw std::runtime_error("No pool configurations found in " + args.pools_path);
        }
        
        // Build run configuration from CLI args
        arb::harness::RunConfig<RealT> run_cfg{};
        run_cfg.min_swap_frac = static_cast<RealT>(args.min_swap_frac);
        run_cfg.max_swap_frac = static_cast<RealT>(args.max_swap_frac);
        run_cfg.dustswap_freq_s = args.dustswap_freq_s;
        run_cfg.user_swap_freq_s = args.user_swap_freq_s;
        run_cfg.user_swap_size_frac = static_cast<RealT>(args.user_swap_size_frac);
        run_cfg.user_swap_thresh = static_cast<RealT>(args.user_swap_thresh);
        
        // Wire APY window flags (matches old harness rounding: 7.0 days -> 604800 seconds)
        run_cfg.apy_period_s = static_cast<uint64_t>(std::max(0.0, args.apy_period_days) * 86400.0 + 0.5);
        run_cfg.apy_cap_pct = args.apy_period_cap_pct;
        
        // Wire save_actions flag
        run_cfg.save_actions = args.save_actions;
        
        // Wire detailed_log flag
        run_cfg.detailed_log = args.detailed_log;
        
        auto t_exec0 = std::chrono::high_resolution_clock::now();
        
        // Run all pools in parallel
        auto results = arb::harness::run_pools_parallel(
            pool_configs, events, run_cfg,
            args.n_threads,
            true  // verbose - prints progress per pool
        );
        
        auto t_exec1 = std::chrono::high_resolution_clock::now();
        double exec_ms = std::chrono::duration<double, std::milli>(t_exec1 - t_exec0).count();
        
        // Write JSON output
        if (!args.out_path.empty()) {
            bool ok = arb::harness::write_results_json(
                args.out_path,
                results,
                events.size(),
                args.candles_path,
                args.use_events,
                args.n_threads,
                candles_read_ms,
                exec_ms
            );
            if (!ok) {
                std::cerr << "Warning: Failed to write output to " << args.out_path << "\n";
            }
        }
        
        // Write detailed log if requested (uses first pool's detailed entries)
        // Place detailed_log.json next to the output file
        if (args.detailed_log && !results.empty()) {
            // Compute detailed_log.json path: same directory as output, fixed name
            std::string detailed_log_path;
            {
                auto pos = args.out_path.find_last_of("/\\");
                if (pos != std::string::npos) {
                    detailed_log_path = args.out_path.substr(0, pos + 1) + "detailed-output.json";
                } else {
                    detailed_log_path = "detailed-output.json";
                }
            }
            
            // Find first successful result with detailed entries
            for (const auto& res : results) {
                if (res.success && !res.detailed_entries.empty()) {
                    bool ok = arb::harness::write_detailed_log(
                        detailed_log_path,
                        res.detailed_entries
                    );
                    if (!ok) {
                        std::cerr << "Warning: Failed to write detailed log to "
                                  << detailed_log_path << "\n";
                    } else {
                        std::cout << "Wrote detailed log (" << res.detailed_entries.size()
                                  << " entries) to " << detailed_log_path << "\n";
                    }
                    break;  // Only write first pool's detailed log
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
