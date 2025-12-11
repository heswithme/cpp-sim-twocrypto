// Modular arb_harness - Entry point with compile-time numeric type selection
//
// Build targets:
//   arb_harness    - double (default)
//   arb_harness_f  - float
//   arb_harness_ld - long double
//   arb_harness_i  - uint256

#include <iostream>
#include <string>
#include <iomanip>
#include <boost/multiprecision/cpp_int.hpp>

#include "events/loader.hpp"
#include "core/common.hpp"
#include "pools/twocrypto_fx/twocrypto.hpp"

namespace stableswap {
using uint256 = boost::multiprecision::uint256_t;
}

// Compile-time numeric type selection
#if defined(ARB_MODE_F)
using RealT = float;
static constexpr const char* TYPE_NAME = "float";
#elif defined(ARB_MODE_LD)
using RealT = long double;
static constexpr const char* TYPE_NAME = "long double";
#elif defined(ARB_MODE_I)
using RealT = stableswap::uint256;
static constexpr const char* TYPE_NAME = "uint256";
#else
using RealT = double;
static constexpr const char* TYPE_NAME = "double";
#endif

// Helper to print a value (handles both numeric and uint256)
template <typename T>
void print_value(const char* name, const T& val) {
    if constexpr (std::is_same_v<T, stableswap::uint256>) {
        std::cout << "  " << name << " = " << val.template convert_to<std::string>() << "\n";
    } else {
        std::cout << "  " << name << " = " << std::setprecision(12) << val << "\n";
    }
}

// Pool test for floating-point types
template <typename T>
typename std::enable_if<!std::is_same_v<T, stableswap::uint256>, void>::type
test_pool() {
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

    if (pool.D > Traits::ZERO()) {
        std::cout << "\nPool test: PASSED (D > 0)\n";
    } else {
        std::cout << "\nPool test: FAILED (D <= 0)\n";
    }
}

// Pool test for uint256
template <typename T>
typename std::enable_if<std::is_same_v<T, stableswap::uint256>, void>::type
test_pool() {
    using Pool = arb::pools::twocrypto_fx::TwoCryptoPool<T>;
    using Traits = arb::pools::twocrypto_fx::PoolTraits<T>;

    std::array<T, 2> precisions = {Traits::ONE(), Traits::ONE()};

    T A("100000000");  // 10000 * 10000 (A_MULTIPLIER)
    T gamma("10000000000000");  // 1e-5 * 1e18
    T mid_fee("1000000");  // 0.01% in fee precision (1e10)
    T out_fee("6000000");  // 0.06%
    T fee_gamma("230000000000000");  // 0.00023 * 1e18
    T allowed_extra_profit("10000000000");  // 1e-8 * 1e18
    T adjustment_step("100000000000000");  // 0.0001 * 1e18
    T ma_time("600");  // 600 seconds
    T initial_price("1080000000000000000");  // 1.08 * 1e18

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

    T amount0("10000000000000000000000");  // 10,000 * 1e18
    T amount1("9259259259259259259259");   // 10,000 / 1.08 * 1e18
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

    if (pool.D > Traits::ZERO()) {
        std::cout << "\nPool test: PASSED (D > 0)\n";
    } else {
        std::cout << "\nPool test: FAILED (D <= 0)\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "arb_harness_mod: " << TYPE_NAME << std::endl;

    // Test differs_rel
    {
        bool test1 = arb::differs_rel<double>(1.0, 1.0 + 1e-13);  // should be false
        bool test2 = arb::differs_rel<double>(1.0, 1.1);          // should be true

        std::cout << "differs_rel tests: "
                  << ((!test1 && test2) ? "PASSED" : "FAILED") << "\n";
    }

    // Test pool creation
    {
        std::cout << "\n--- Pool Test ---\n";
        test_pool<RealT>();
    }

    if (argc < 2) {
        std::cout << "\nUsage: " << argv[0] << " <candles.json>\n";
        return 0;
    }

    try {
        const std::string candles_path = argv[1];

        // Load candles and generate events
        auto candles = arb::load_candles(candles_path);
        auto events = arb::gen_events(candles);

        std::cout << "\nLoaded " << candles.size() << " candles -> "
                  << events.size() << " events from " << candles_path << "\n";

        if (!events.empty()) {
            std::cout << "First event: ts=" << events.front().ts
                      << ", p_cex=" << events.front().p_cex
                      << ", volume=" << events.front().volume << "\n";
            std::cout << "Last event:  ts=" << events.back().ts
                      << ", p_cex=" << events.back().p_cex
                      << ", volume=" << events.back().volume << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
