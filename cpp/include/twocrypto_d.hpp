#pragma once
/**
 * @title TwoCrypto C++ Implementation (double precision)
 * @notice A Curve AMM pool for 2 unpegged assets using double precision for speed
 */

#include "stableswap_math_d.hpp"
#include <array>
#include <stdexcept>
#include <chrono>
#include <cmath>

namespace twocrypto_d {

using namespace stableswap_d;

constexpr int N_COINS = 2;

inline double PRECISION_R() { return 1.0; }
inline double FEE_ONE_R() { return 1.0; }

class TwoCryptoPoolD {
public:
    // State variables (double)
    std::array<double, 2> balances{0.0, 0.0};
    double D = 0.0;
    double totalSupply = 0.0;

    // Price variables
    double cached_price_scale = 1.0;
    double cached_price_oracle = 1.0;
    double last_prices = 1.0;
    uint64_t last_timestamp = 0;

    // Parameters (unpacked doubles)
    double A = 0.0;     // already multiplied by A_MULTIPLIER
    double gamma = 0.0; // unused in current math

    // Fees in [0,1]
    double mid_fee = 0.0;
    double out_fee = 0.0;
    double fee_gamma = 0.0;

    // Rebalancing params (doubles, base 1.0)
    double allowed_extra_profit = 0.0; // ~1e-? relative to 1.0
    double adjustment_step = 0.0;
    double ma_time = 600.0; // seconds

    // Profit tracking
    double xcp_profit = 1.0;
    double xcp_profit_a = 1.0;
    double virtual_price = 1.0;

    // Token precisions (usually 1.0)
    std::array<double, 2> precisions{1.0, 1.0};

    // Time for testing
    uint64_t block_timestamp = 0;

    // Donation & admin-fee like vars (doubles)
    double donation_shares = 0.0;
    double donation_shares_max_ratio = 0.10; // 10%
    double donation_duration = 7.0 * 86400.0;
    double last_donation_release_ts = 0.0;
    double donation_protection_expiry_ts = 0.0;
    double donation_protection_period = 10.0 * 60.0;
    double donation_protection_lp_threshold = 0.20; // 20%

    double admin_fee = 0.5; // 50%
    uint64_t last_admin_fee_claim_timestamp = 0;

public:
    TwoCryptoPoolD(
        const std::array<double, 2>& _precisions,
        double _A,
        double _gamma,
        double _mid_fee,
        double _out_fee,
        double _fee_gamma,
        double _allowed_extra_profit,
        double _adjustment_step,
        double _ma_time,
        double initial_price
    ) {
        precisions = _precisions;
        A = _A; gamma = _gamma;
        mid_fee = _mid_fee; out_fee = _out_fee; fee_gamma = _fee_gamma;
        allowed_extra_profit = _allowed_extra_profit;
        adjustment_step = _adjustment_step;
        ma_time = _ma_time;
        cached_price_scale = initial_price;
        cached_price_oracle = initial_price;
        last_prices = initial_price;

        xcp_profit = 1.0;
        xcp_profit_a = 1.0;
        virtual_price = 1.0;

        block_timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        last_timestamp = block_timestamp;
    }

private:
    std::array<double, 2> _xp(const std::array<double, 2>& _balances, double price_scale) const {
        return {
            _balances[0] * precisions[0],
            _balances[1] * precisions[1] * price_scale / PRECISION_R()
        };
    }

    double _fee(const std::array<double, 2>& xp) const {
        if (fee_gamma == 0.0) return mid_fee;
        double Bsum = xp[0] + xp[1];
        if (Bsum <= 0.0) return mid_fee;
        double B = static_cast<double>(N_COINS * N_COINS) * xp[0] * xp[1] / (Bsum * Bsum);
        B = fee_gamma * B / (fee_gamma * B + (1.0 - B));
        return mid_fee * B + out_fee * (1.0 - B);
    }

    double _xcp(double _D, double price_scale) const {
        double sqrt_price = std::sqrt(PRECISION_R() * price_scale);
        return _D * PRECISION_R() / static_cast<double>(N_COINS) / sqrt_price;
    }

    double _donation_shares(bool donation_protection = true) const {
        if (donation_shares <= 0.0) return 0.0;
        double elapsed = static_cast<double>(block_timestamp) - last_donation_release_ts;
        double unlocked = donation_shares * elapsed / donation_duration;
        if (unlocked > donation_shares) unlocked = donation_shares;
        if (!donation_protection) return unlocked;
        double protection_factor = 0.0;
        if (donation_protection_expiry_ts > static_cast<double>(block_timestamp)) {
            protection_factor = (donation_protection_expiry_ts - static_cast<double>(block_timestamp)) / donation_protection_period;
            if (protection_factor > 1.0) protection_factor = 1.0;
        }
        return unlocked * (1.0 - protection_factor);
    }

public:
    double add_liquidity(const std::array<double, 2>& amounts, double min_mint_amount, bool donation = false);
    std::array<double, 2> remove_liquidity(double amount, const std::array<double, 2>& min_amounts);
    std::array<double, 3> exchange(double i, double j, double dx, double min_dy);
    double tweak_price(const std::array<double, 2>& _A_gamma, const std::array<double, 2>& xp, double _D);

    // Views
    double get_virtual_price() const { return virtual_price <= 0.0 ? 1.0 : virtual_price; }
    double get_p() const { return (balances[0] <= 0.0 || balances[1] <= 0.0) ? cached_price_scale : last_prices; }

    // Testing helpers
    void set_block_timestamp(uint64_t ts) { block_timestamp = ts; }
    void advance_time(uint64_t seconds) { block_timestamp += seconds; }
};

} // namespace twocrypto_d
