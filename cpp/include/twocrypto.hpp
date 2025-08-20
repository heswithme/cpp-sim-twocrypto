#pragma once
/**
 * @title TwoCrypto C++ Implementation
 * @notice A Curve AMM pool for 2 unpegged assets (e.g. WETH, USD)
 * @dev Mirrors the Vyper implementation for easy maintenance
 */

#include "stableswap_math.hpp"
#include <array>
#include <stdexcept>
#include <chrono>

namespace twocrypto {

using namespace stableswap;

// ----------------------------- Constants ------------------------------------

constexpr int N_COINS = 2;

// Constants as functions to match Vyper style
inline uint256 PRECISION() { 
    static uint256 v("1000000000000000000"); // 10**18
    return v; 
}

inline uint256 A_MULTIPLIER() { 
    static uint256 v(10000);
    return v;
}

inline uint256 MIN_GAMMA() {
    static uint256 v("10000000000"); // 10**10
    return v;
}

inline uint256 MAX_GAMMA() {
    static uint256 v("199000000000000000"); // 199 * 10**15
    return v;
}

inline uint256 MIN_A() {
    static uint256 v = N_COINS * A_MULTIPLIER();
    return v;
}

inline uint256 MAX_A() {
    static uint256 v = 10000 * A_MULTIPLIER();
    return v;
}

inline uint256 NOISE_FEE() {
    static uint256 v("100000"); // 10**5 (matches Vyper)
    return v;
}

inline uint256 FEE_PRECISION() {
    static uint256 v("10000000000"); // 10**10
    return v;
}

// ----------------------------- Pool State -----------------------------------

class TwoCryptoPool {
public:
    // State variables (matching Vyper contract)
    std::array<uint256, 2> balances{0, 0};
    uint256 D = 0;
    uint256 totalSupply = 0;
    
    // Price variables
    uint256 cached_price_scale;
    uint256 cached_price_oracle;
    uint256 last_prices;
    uint256 last_timestamp;
    
    // Parameters (packed for efficiency like Vyper)
    uint256 A_gamma;  // packed A and gamma
    uint256 packed_fee_params;  // mid_fee, out_fee, fee_gamma
    uint256 packed_rebalancing_params;  // allowed_extra_profit, adjustment_step, ma_time
    
    // Profit tracking
    uint256 xcp_profit;
    uint256 xcp_profit_a;
    uint256 virtual_price;
    
    // Token precisions
    std::array<uint256, 2> precisions;
    
    // Time for testing
    uint64_t block_timestamp;
    bool trace_enabled = false;

    // --------------------- Donation & Admin Fee State ----------------------
    uint256 donation_shares = 0;
    uint256 donation_shares_max_ratio = PRECISION() * 10 / 100; // 10%
    uint256 donation_duration = uint256(7 * 86400);
    uint256 last_donation_release_ts = 0;
    uint256 donation_protection_expiry_ts = 0;
    uint256 donation_protection_period = uint256(10 * 60); // 10 minutes
    uint256 donation_protection_lp_threshold = PRECISION() * 20 / 100; // 20%

    uint256 admin_fee = uint256("5000000000"); // 5e9 (50% of 1e10)
    uint256 last_admin_fee_claim_timestamp = 0;

private:
    // ----------------------- Internal Functions -----------------------------
    
    /**
     * @notice Unpack 2 values from uint256
     */
    std::array<uint256, 2> _unpack_2(const uint256& packed) const {
        uint256 mask = (uint256(1) << 128) - 1;
        return {packed & mask, packed >> 128};
    }
    
    /**
     * @notice Unpack 3 values from uint256 using [x0<<128 | x1<<64 | x2]
     * Matches Vyper's _pack_3 in Twocrypto.
     */
    std::array<uint256, 3> _unpack_3(const uint256& packed) const {
        uint256 mask64 = (uint256(1) << 64) - 1;
        return {
            (packed >> 128) & mask64, // x0
            (packed >> 64) & mask64,  // x1
            packed & mask64           // x2
        };
    }
    
    /**
     * @notice Get A and gamma parameters in normalized order [A, gamma]
     * Mirrors Vyper's _A_gamma(), which unpacks packed_gamma_A (gamma low, A high)
     * and returns [A, gamma].
     */
    std::array<uint256, 2> _A_gamma() const {
        // packed layout: [ high: A | low: gamma ]
        uint256 mask = (uint256(1) << 128) - 1;
        uint256 gamma = A_gamma & mask;
        uint256 A = A_gamma >> 128;
        return {A, gamma};
    }
    
    /**
     * @notice Internal function to calculate xp (invariant)
     */
    std::array<uint256, 2> _xp(
        const std::array<uint256, 2>& _balances,
        const uint256& price_scale
    ) const {
        return {
            _balances[0] * precisions[0],
            _balances[1] * precisions[1] * price_scale / PRECISION()
        };
    }
    
    /**
     * @notice Calculate fee based on pool imbalance
     */
    uint256 _fee(const std::array<uint256, 2>& xp) const {
        auto fee_params = _unpack_3(packed_fee_params);
        uint256 mid_fee = fee_params[0];
        uint256 out_fee = fee_params[1];
        uint256 fee_gamma = fee_params[2];
        
        if (fee_gamma == 0) {
            return mid_fee;
        }
        
        uint256 B = xp[0] + xp[1];
        // Balance indicator that goes from 1e18 (perfect balance) to 0 (very imbalanced)
        // B = PRECISION * N^N * xp[0] / (sum^2) * xp[1]
        B = PRECISION() * N_COINS * N_COINS * xp[0] / B * xp[1] / B;

        // Regulate slope using fee_gamma
        // fee_gamma * B / (fee_gamma * B + 1 - B)
        B = fee_gamma * B / (fee_gamma * B / PRECISION() + PRECISION() - B);

        // fee = mid_fee * B + out_fee * (1 - B)
        return (mid_fee * B + out_fee * (PRECISION() - B)) / PRECISION();
    }
    
    /**
     * @notice xcp = D / N / sqrt(price_scale) = D0 / sqrt(p0 * p1)
     */
    uint256 _xcp(const uint256& _D, const uint256& price_scale) const {
        uint256 sqrt_price = boost::multiprecision::sqrt(PRECISION() * price_scale);
        return _D * PRECISION() / N_COINS / sqrt_price;
    }
    
    /**
     * @notice Update profit variables
     */
    uint256 _calc_profit(const uint256& new_xcp, const uint256& old_xcp) const {
        return new_xcp * PRECISION() / old_xcp;
    }

    /**
     * @notice Compute unlocked donation shares, optionally with protection damping
     */
    uint256 _donation_shares(bool donation_protection = true) const {
        if (donation_shares == 0) return 0;
        uint256 elapsed = uint256(block_timestamp) - last_donation_release_ts;
        uint256 unlocked_shares = donation_shares * elapsed / donation_duration;
        if (unlocked_shares > donation_shares) unlocked_shares = donation_shares;
        if (!donation_protection) return unlocked_shares;
        uint256 protection_factor = 0;
        if (donation_protection_expiry_ts > uint256(block_timestamp)) {
            protection_factor = (donation_protection_expiry_ts - uint256(block_timestamp)) * PRECISION() / donation_protection_period;
            if (protection_factor > PRECISION()) protection_factor = PRECISION();
        }
        return unlocked_shares * (PRECISION() - protection_factor) / PRECISION();
    }

    /**
     * @notice Calculate liquidity action fee approximation (for add/remove)
     * Mirrors Vyper _calc_token_fee for N_COINS=2
     */
    uint256 _calc_token_fee(
        const std::array<uint256, 2>& amounts,
        const std::array<uint256, 2>& xp,
        bool donation,
        bool deposit
    ) const {
        if (donation) {
            // Donation fees are 0, but NOISE_FEE is required for numerical stability
            return NOISE_FEE();
        }

        // balances ratio before liquidity op (scaled by precisions)
        uint256 denom = (balances[1] - amounts[1]) * precisions[1];
        uint256 balances_ratio = 0;
        if (denom > 0) {
            balances_ratio = (balances[0] - amounts[0]) * precisions[0] * PRECISION() / denom;
        }
        // amounts scaled using balances_ratio instead of price_scale
        std::array<uint256, 2> amounts_scaled = {
            amounts[0] * precisions[0],
            amounts[1] * precisions[1] * balances_ratio / PRECISION()
        };

        // fee' = _fee(xp) * N / (4 * (N-1)) = _fee/2 for N=2
        uint256 fee_prime = _fee(xp) * N_COINS / (4 * (N_COINS - 1));

        uint256 S = amounts_scaled[0] + amounts_scaled[1];
        uint256 avg = S / N_COINS;
        uint256 Sdiff = (amounts_scaled[0] > avg ? amounts_scaled[0] - avg : avg - amounts_scaled[0]) +
                        (amounts_scaled[1] > avg ? amounts_scaled[1] - avg : avg - amounts_scaled[1]);

        uint256 lp_spam_penalty_fee = 0;
        if (deposit && donation_protection_expiry_ts > block_timestamp) {
            uint256 protection_factor = (donation_protection_expiry_ts - block_timestamp) * PRECISION() / donation_protection_period;
            if (protection_factor > PRECISION()) protection_factor = PRECISION();
            lp_spam_penalty_fee = protection_factor * fee_prime / PRECISION();
        }

        return fee_prime * Sdiff / S + NOISE_FEE() + lp_spam_penalty_fee;
    }

public:
    // -------------------------- Constructor ---------------------------------
    
    TwoCryptoPool(
        const std::array<uint256, 2>& _precisions,
        const uint256& _A_gamma,
        const uint256& _packed_fee_params,
        const uint256& _packed_rebalancing_params,
        const uint256& initial_price
    ) {
        // Validate parameters
        auto gamma_A = _unpack_2(_A_gamma);
        uint256 gamma = gamma_A[0];
        uint256 A = gamma_A[1];
        
        if (gamma < MIN_GAMMA() || gamma > MAX_GAMMA()) {
            throw std::invalid_argument("gamma out of range");
        }
        if (A < MIN_A() || A > MAX_A()) {
            throw std::invalid_argument("A out of range");
        }
        
        // Initialize state
        precisions = _precisions;
        A_gamma = _A_gamma;
        packed_fee_params = _packed_fee_params;
        packed_rebalancing_params = _packed_rebalancing_params;
        
        cached_price_scale = initial_price;
        cached_price_oracle = initial_price;
        last_prices = initial_price;
        
        xcp_profit = PRECISION();
        xcp_profit_a = PRECISION();
        virtual_price = PRECISION();
        
        // Initialize timestamp
        block_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        last_timestamp = block_timestamp;
    }
    
    // ----------------------- External Functions -----------------------------
    
    /**
     * @notice Add liquidity to the pool
     * @param amounts Amounts of each coin to add
     * @param min_mint_amount Minimum LP tokens to mint
     * @return Amount of LP tokens minted
     */
    uint256 add_liquidity(
        const std::array<uint256, 2>& amounts,
        uint256 min_mint_amount,
        bool donation = false
    );
    
    /**
     * @notice Remove liquidity from the pool (balanced)
     * @param amount Amount of LP tokens to burn
     * @param min_amounts Minimum amounts of each coin to receive
     * @return Amounts of each coin withdrawn
     */
    std::array<uint256, 2> remove_liquidity(
        uint256 amount,
        const std::array<uint256, 2>& min_amounts
    );
    
    /**
     * @notice Exchange coins
     * @param i Index of input coin
     * @param j Index of output coin  
     * @param dx Amount of input coin
     * @param min_dy Minimum amount of output coin
     * @return {dy, fee, new_price_scale}
     */
    std::array<uint256, 3> exchange(
        uint256 i,
        uint256 j,
        uint256 dx,
        uint256 min_dy
    );
    
    /**
     * @notice Update price oracle and optionally adjust price_scale
     * @param A_gamma Current A and gamma parameters
     * @param xp Current scaled balances
     * @param D Current invariant
     * @return New price scale
     */
    uint256 tweak_price(
        const std::array<uint256, 2>& _A_gamma,
        const std::array<uint256, 2>& xp,
        uint256 _D
    );
    
    // ------------------------ View Functions --------------------------------
    
    /**
     * @notice Get current virtual price
     */
    uint256 get_virtual_price() const {
        return virtual_price == 0 ? PRECISION() : virtual_price;
    }
    
    /**
     * @notice Get current price (dy/dx)
     */
    uint256 get_p() const {
        if (balances[0] == 0 || balances[1] == 0) {
            return cached_price_scale;
        }
        return last_prices;
    }
    
    // ----------------------- Testing Functions ------------------------------
    
    void set_block_timestamp(uint64_t timestamp) { 
        block_timestamp = timestamp; 
    }
    
    void advance_time(uint64_t seconds) { 
        block_timestamp += seconds; 
    }

    void set_trace(bool enabled) {
        trace_enabled = enabled;
    }
};

} // namespace twocrypto
