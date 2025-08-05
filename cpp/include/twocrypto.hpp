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
    static uint256 v("100000000000"); // 10**11
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
     * @notice Unpack 3 values from uint256
     */
    std::array<uint256, 3> _unpack_3(const uint256& packed) const {
        uint256 mask = (uint256(1) << 85) - 1;
        return {
            packed & mask,
            (packed >> 85) & mask,
            packed >> 170
        };
    }
    
    /**
     * @notice Get A and gamma parameters
     */
    std::array<uint256, 2> _A_gamma() const {
        return _unpack_2(A_gamma);
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
        
        uint256 sum_xp = xp[0] + xp[1];
        
        // g = x[0] * x[1] * N_COINS^2 / sum_xp^2
        uint256 g = PRECISION() * N_COINS * N_COINS * xp[0] / sum_xp * xp[1] / sum_xp;
        
        // g = fee_gamma * g^2 / (g + (1 - g) * fee_gamma)
        g = fee_gamma * g / (fee_gamma * g / PRECISION() + PRECISION() - g);
        
        // fee = mid_fee * g + out_fee * (1 - g)
        return (mid_fee * g + out_fee * (PRECISION() - g)) / PRECISION();
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
     * @notice Add liquidity to the pool
     * @param amounts Amounts of each coin to add
     * @param min_mint_amount Minimum LP tokens to mint
     * @return Amount of LP tokens minted
     */
    uint256 add_liquidity(
        const std::array<uint256, 2>& amounts,
        uint256 min_mint_amount
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
        if (totalSupply == 0) return PRECISION();
        
        uint256 xcp = _xcp(D, cached_price_scale);
        return xcp * PRECISION() / totalSupply;
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
};

} // namespace twocrypto