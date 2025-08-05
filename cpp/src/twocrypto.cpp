#include "twocrypto.hpp"
#include <iostream>

namespace twocrypto {

// ----------------------------- exchange -------------------------------------

std::array<uint256, 3> TwoCryptoPool::exchange(
    uint256 i,
    uint256 j,
    uint256 dx,
    uint256 min_dy
) {
    // Convert uint256 indices to size_t
    size_t idx_i = static_cast<size_t>(i.convert_to<unsigned>());
    size_t idx_j = static_cast<size_t>(j.convert_to<unsigned>());
    
    // Validate inputs
    if (idx_i == idx_j) {
        throw std::invalid_argument("coin index out of range");
    }
    if (idx_i >= N_COINS || idx_j >= N_COINS) {
        throw std::invalid_argument("coin index out of range");
    }
    if (dx == 0) {
        throw std::invalid_argument("dx cannot be 0");
    }
    
    // Current state
    auto A_gamma_current = _A_gamma();
    uint256 price_scale = cached_price_scale;
    auto xp = _xp(balances, price_scale);
    
    // Add dx to balance
    xp[idx_i] += dx * precisions[idx_i];
    
    // Calculate dy using newton_y
    auto [dy, K0] = StableswapMath::get_y(
        A_gamma_current[0],  // A
        A_gamma_current[1],  // gamma
        xp,
        D,
        idx_j
    );
    
    // Subtract dy from xp
    dy = xp[idx_j] - dy - 1;  // -1 for rounding
    xp[idx_j] -= dy;
    
    // Convert dy to token amount
    if (idx_j > 0) {
        dy = dy * PRECISION() / price_scale;
    }
    dy = dy / precisions[idx_j];
    
    // Check slippage
    if (dy < min_dy) {
        throw std::runtime_error("slippage");
    }
    
    // Calculate fee
    uint256 fee = _fee(xp) * dy / PRECISION();
    dy -= fee;
    
    // Update balances
    balances[idx_i] += dx;
    balances[idx_j] -= dy;
    
    // Calculate new D
    auto xp_new = _xp(balances, price_scale);
    uint256 D_new = StableswapMath::newton_D(
        A_gamma_current[0],
        A_gamma_current[1],
        xp_new
    );
    
    // Update state
    D = D_new;
    
    // Update price (similar to Vyper's tweak_price logic)
    uint256 new_price_scale = tweak_price(A_gamma_current, xp_new, D_new);
    
    return {dy, fee, new_price_scale};
}

// ---------------------------- add_liquidity ----------------------------------

uint256 TwoCryptoPool::add_liquidity(
    const std::array<uint256, 2>& amounts,
    uint256 min_mint_amount
) {
    // Check if any tokens are being added
    if (amounts[0] + amounts[1] == 0) {
        throw std::invalid_argument("no coins to add");
    }
    
    auto A_gamma_current = _A_gamma();
    uint256 price_scale = cached_price_scale;
    
    // Store old state
    auto old_balances = balances;
    uint256 old_D = D;
    
    // Update balances
    for (size_t i = 0; i < N_COINS; ++i) {
        balances[i] += amounts[i];
    }
    
    // Calculate new D
    auto xp = _xp(balances, price_scale);
    uint256 D_new = StableswapMath::newton_D(
        A_gamma_current[0],
        A_gamma_current[1],
        xp
    );
    
    // Calculate tokens to mint
    uint256 d_token = 0;
    
    if (old_D == 0) {
        // Initial deposit
        d_token = _xcp(D_new, price_scale);
        
        // Initialize xcp_profit_a for tracking
        xcp_profit_a = PRECISION();
    } else {
        // Subsequent deposits
        d_token = totalSupply * D_new / old_D - totalSupply;
    }
    
    // Check slippage
    if (d_token < min_mint_amount) {
        throw std::runtime_error("slippage");
    }
    
    // Update state
    D = D_new;
    totalSupply += d_token;
    
    // Check if we need to update prices (imbalanced add)
    bool update_prices = false;
    if (old_D > 0) {
        auto old_xp = _xp(old_balances, price_scale);
        
        // Check if the add is imbalanced
        // Simple heuristic: if ratio of amounts differs from ratio of balances
        if (old_balances[0] > 0 && old_balances[1] > 0) {
            uint256 ratio_amounts = amounts[0] * PRECISION() / amounts[1];
            uint256 ratio_balances = old_balances[0] * PRECISION() / old_balances[1];
            
            // If ratios differ by more than 1%, update prices
            uint256 diff = ratio_amounts > ratio_balances ? 
                ratio_amounts - ratio_balances : ratio_balances - ratio_amounts;
            if (diff > PRECISION() / 100) {
                update_prices = true;
            }
        }
    }
    
    if (update_prices) {
        tweak_price(A_gamma_current, xp, D_new);
    }
    
    // Update virtual price
    if (totalSupply > 0) {
        uint256 xcp = _xcp(D_new, cached_price_scale);
        virtual_price = xcp * PRECISION() / totalSupply;
        
        // Update profit tracking
        if (old_D > 0) {
            uint256 old_xcp = _xcp(old_D, cached_price_scale);
            uint256 profit_ratio = xcp * PRECISION() / old_xcp;
            xcp_profit = xcp_profit * profit_ratio / PRECISION();
        }
    }
    
    return d_token;
}

// -------------------------- remove_liquidity ---------------------------------

std::array<uint256, 2> TwoCryptoPool::remove_liquidity(
    uint256 amount,
    const std::array<uint256, 2>& min_amounts
) {
    // This is balanced withdrawal - no complex math
    
    if (amount > totalSupply) {
        throw std::invalid_argument("insufficient LP tokens");
    }
    
    std::array<uint256, 2> withdrawn;
    
    // Calculate proportional amounts
    for (size_t i = 0; i < N_COINS; ++i) {
        withdrawn[i] = balances[i] * amount / totalSupply;
        
        // Check minimum amounts
        if (withdrawn[i] < min_amounts[i]) {
            throw std::runtime_error("withdrawal resulted in fewer coins than expected");
        }
        
        // Update balances
        balances[i] -= withdrawn[i];
    }
    
    // Update supply
    totalSupply -= amount;
    
    // Update D proportionally (no fees on balanced withdrawal)
    if (totalSupply > 0) {
        D = D * (totalSupply + amount) / totalSupply;
    } else {
        D = 0;
    }
    
    // Update virtual price
    if (totalSupply > 0) {
        uint256 xcp = _xcp(D, cached_price_scale);
        virtual_price = xcp * PRECISION() / totalSupply;
    }
    
    return withdrawn;
}

// ---------------------------- tweak_price ------------------------------------

uint256 TwoCryptoPool::tweak_price(
    const std::array<uint256, 2>& _A_gamma,
    const std::array<uint256, 2>& xp,
    uint256 _D
) {
    /**
     * @notice Updates price_oracle, last_price and conditionally adjusts
     *         price_scale. This is called whenever there is an unbalanced
     *         liquidity operation: exchange, add_liquidity, or
     *         remove_liquidity_one_coin.
     */
    
    uint256 price_scale = cached_price_scale;
    uint256 last_prices_timestamp = last_timestamp;
    uint256 current_timestamp = block_timestamp;
    
    // Update last_prices (price measured by the AMM)
    last_prices = StableswapMath::get_p(xp, _D, _A_gamma);
    
    // Skip price oracle update if not enough time has passed
    if (current_timestamp <= last_prices_timestamp) {
        return price_scale;
    }
    
    // Calculate alpha for EMA (exponential moving average)
    auto rebalancing_params = _unpack_3(packed_rebalancing_params);
    uint256 ma_time = rebalancing_params[2];
    
    uint256 dt = current_timestamp - last_prices_timestamp;
    last_timestamp = current_timestamp;
    
    // EMA of price oracle
    if (dt > 0) {
        // alpha = exp(-dt/ma_time)
        // For simplicity, use linear approximation for small dt
        // price_oracle = price_oracle * alpha + last_prices * (1 - alpha)
        
        if (dt < ma_time) {
            // Linear approximation: alpha ≈ 1 - dt/ma_time
            uint256 alpha = PRECISION() - (PRECISION() * dt / ma_time);
            cached_price_oracle = (cached_price_oracle * alpha + 
                                  last_prices * (PRECISION() - alpha)) / PRECISION();
        } else {
            // If dt is large, just use last price
            cached_price_oracle = last_prices;
        }
    }
    
    // Decide whether to adjust price_scale
    uint256 allowed_extra_profit = rebalancing_params[0];
    uint256 adjustment_step = rebalancing_params[1];
    
    // Calculate current profit
    uint256 old_xcp = _xcp(_D, price_scale);
    
    // Calculate profit at oracle price
    uint256 new_price_scale = cached_price_oracle;
    auto xp_new = _xp(balances, new_price_scale);
    uint256 D_at_oracle = StableswapMath::newton_D(_A_gamma[0], _A_gamma[1], xp_new);
    uint256 xcp_at_oracle = _xcp(D_at_oracle, new_price_scale);
    
    // Check if adjusting price would increase profit
    if (xcp_at_oracle > old_xcp) {
        // Calculate profit ratio
        uint256 profit_ratio = xcp_at_oracle * PRECISION() / old_xcp;
        
        // Only adjust if profit is significant
        if (profit_ratio > PRECISION() + allowed_extra_profit) {
            // Adjust price_scale towards oracle price
            // new_price = old_price * (1 - adjustment_step) + oracle_price * adjustment_step
            
            uint256 new_scale = (price_scale * (PRECISION() - adjustment_step) + 
                                cached_price_oracle * adjustment_step) / PRECISION();
            
            // Validate the adjustment doesn't break invariants
            auto xp_test = _xp(balances, new_scale);
            try {
                uint256 D_test = StableswapMath::newton_D(_A_gamma[0], _A_gamma[1], xp_test);
                
                // Only update if newton converged and D didn't decrease too much
                if (D_test > _D * 999 / 1000) {  // Allow max 0.1% decrease
                    cached_price_scale = new_scale;
                    D = D_test;
                    
                    // Update profit tracking
                    uint256 new_xcp = _xcp(D_test, new_scale);
                    xcp_profit = xcp_profit * new_xcp / old_xcp;
                }
            } catch (...) {
                // If newton fails, keep old price_scale
            }
        }
    }
    
    return cached_price_scale;
}

} // namespace twocrypto