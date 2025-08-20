#include "twocrypto.hpp"
#include <iostream>

namespace twocrypto {

// ---------------------------- add_liquidity ----------------------------------

uint256 TwoCryptoPool::add_liquidity(
    const std::array<uint256, 2>& amounts,
    uint256 min_mint_amount,
    bool donation
) {
    if (amounts[0] + amounts[1] == 0) {
        throw std::invalid_argument("no coins to add");
    }

    auto A_gamma_current = _A_gamma();
    uint256 price_scale = cached_price_scale;

    auto old_balances = balances;
    std::array<uint256,2> amounts_received = amounts;
    std::array<uint256,2> new_balances = {balances[0] + amounts_received[0], balances[1] + amounts_received[1]};

    auto xp = _xp(new_balances, price_scale);
    auto old_xp = _xp(old_balances, price_scale);

    uint256 old_D = D; // not ramping in this harness
    uint256 D_new = StableswapMath::newton_D(A_gamma_current[0], A_gamma_current[1], xp, 0);

    uint256 token_supply = totalSupply;
    uint256 d_token = 0;
    if (old_D > 0) {
        d_token = token_supply * D_new / old_D - token_supply;
    } else {
        d_token = _xcp(D_new, price_scale);
    }
    if (d_token == 0) {
        throw std::runtime_error("nothing minted");
    }

    uint256 d_token_fee = 0;
    if (old_D > 0) {
        uint256 approx_fee = _calc_token_fee(amounts_received, xp, donation, true);
        d_token_fee = approx_fee * d_token / FEE_PRECISION() + 1;
        d_token -= d_token_fee;
    }

    // Update balances and state, handling donation or regular LP mint
    balances = new_balances;
    if (old_D > 0) {
        D = D_new;
        if (donation) {
            // Donation path
            // token_supply already cached above
            uint256 new_donation_shares = donation_shares + d_token;
            // Cap: new_donation_shares * PRECISION / (token_supply + d_token) <= donation_shares_max_ratio
            if ((new_donation_shares * PRECISION()) / (token_supply + d_token) > donation_shares_max_ratio) {
                throw std::runtime_error("donation above cap");
            }
            // Preserve currently unlocked donations proportionally
            uint256 unlocked = _donation_shares(false);
            uint256 new_elapsed = 0;
            if (new_donation_shares > 0) {
                new_elapsed = (unlocked * donation_duration) / new_donation_shares;
            }
            last_donation_release_ts = uint256(block_timestamp) - new_elapsed;

            donation_shares = new_donation_shares;
            totalSupply += d_token; // credit donation shares to supply
        } else {
            // Extend donation protection if any donations exist
            uint256 relative_lp_add = d_token * PRECISION() / (token_supply + d_token);
            if (relative_lp_add > 0 && donation_shares > 0) {
                uint256 protection_period = donation_protection_period;
                uint256 extension_seconds = (relative_lp_add * protection_period) / donation_protection_lp_threshold;
                if (extension_seconds > protection_period) extension_seconds = protection_period;
                uint256 current_expiry = donation_protection_expiry_ts > block_timestamp ? donation_protection_expiry_ts : block_timestamp;
                uint256 new_expiry = current_expiry + extension_seconds;
                uint256 max_expiry = block_timestamp + protection_period;
                if (new_expiry > max_expiry) new_expiry = max_expiry;
                donation_protection_expiry_ts = new_expiry;
            }
            totalSupply += d_token; // mint LP
        }
        // Tweak price for both donation and regular adds
        cached_price_scale = tweak_price(A_gamma_current, xp, D_new);
    } else {
        // Instantiate empty pool
        D = D_new;
        virtual_price = PRECISION();
        xcp_profit = PRECISION();
        xcp_profit_a = PRECISION();
        totalSupply += d_token; // mint initial LP
    }

    if (d_token < min_mint_amount) {
        throw std::runtime_error("slippage");
    }

    return d_token;
}

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
    if (idx_i == idx_j || idx_i >= N_COINS || idx_j >= N_COINS) {
        throw std::invalid_argument("coin index out of range");
    }
    if (dx == 0) {
        throw std::invalid_argument("zero dx");
    }

    auto A_gamma_current = _A_gamma();
    uint256 price_scale = cached_price_scale;

    // Simulate transfer in
    auto balances_local = balances;
    balances_local[idx_i] += dx;

    // Compute xp from updated balances
    auto xp = _xp(balances_local, price_scale);

    // Calculate dy using get_y
    auto y_out = StableswapMath::get_y(
        A_gamma_current[0],
        A_gamma_current[1],
        xp,
        D,
        idx_j
    );

    uint256 dy_xp = xp[idx_j] - y_out.value;
    xp[idx_j] -= dy_xp;
    uint256 dy_tokens = dy_xp - 1; // rounding

    if (idx_j > 0) {
        dy_tokens = dy_tokens * PRECISION() / price_scale;
    }
    dy_tokens = dy_tokens / precisions[idx_j];

    // Fee
    uint256 fee = _fee(xp) * dy_tokens / FEE_PRECISION();
    uint256 dy_after_fee = dy_tokens - fee;
    if (dy_after_fee < min_dy) {
        throw std::runtime_error("slippage");
    }

    // Update storage balances
    balances[idx_i] += dx;
    balances[idx_j] -= dy_after_fee;

    // Update D and price
    auto xp_new = _xp(balances, price_scale);
    uint256 D_new = StableswapMath::newton_D(
        A_gamma_current[0],
        A_gamma_current[1],
        xp_new,
        y_out.unused
    );
    D = D_new;
    uint256 new_price_scale = tweak_price(A_gamma_current, xp_new, D_new);

    return {dy_after_fee, fee, new_price_scale};
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
    {
        uint256 old_total_supply = totalSupply + amount; // supply before burn
        if (old_total_supply > 0) {
            D = D - (D * amount / old_total_supply);
        } else {
            D = 0;
        }
        // Note: Vyper remove_liquidity does not update virtual_price here.
    }
    
    return withdrawn;
}

// ---------------------------- tweak_price ------------------------------------

uint256 TwoCryptoPool::tweak_price(
    const std::array<uint256, 2>& _A_gamma,
    const std::array<uint256, 2>& xp,
    uint256 _D
) {
    // Read storage
    uint256 price_oracle = cached_price_oracle;
    uint256 price_scale = cached_price_scale;
    auto rebalancing_params = _unpack_3(packed_rebalancing_params); // [allowed_extra_profit, adjustment_step, ma_time]

    // Update price oracle if time advanced
    uint256 last_ts = last_timestamp;
    if (last_ts < block_timestamp) {
        // alpha = exp(- (block_timestamp - last_ts)/ma_time) in wad
        uint256 dt = block_timestamp - last_ts;
        uint256 ma_time = rebalancing_params[2];
        auto neg = stableswap::int256(- (stableswap::int256(dt) * stableswap::int256(PRECISION()) / stableswap::int256(ma_time)));
        uint256 alpha = StableswapMath::wad_exp(neg);

        // Use stored last_prices for EMA update, capped at 2 * price_scale
        uint256 capped = last_prices;
        if (capped > 2 * price_scale) capped = 2 * price_scale;
        price_oracle = (capped * (PRECISION() - alpha) + price_oracle * alpha) / PRECISION();
        cached_price_oracle = price_oracle;
        last_timestamp = block_timestamp;
        if (trace_enabled) {
            std::cout << "TRACE tp_ema ts=" << block_timestamp
                      << " dt=" << dt
                      << " alpha=" << alpha
                      << " capped_last_prices=" << capped
                      << " price_oracle=" << price_oracle << std::endl;
        }
    }
    // Update spot price after EMA step
    last_prices = StableswapMath::get_p(xp, _D, _A_gamma) * price_scale / PRECISION();

    // Donation shares (0 in this harness) and supply
    uint256 total_supply = totalSupply;
    uint256 donation_unlocked = _donation_shares();
    uint256 locked_supply = total_supply - donation_unlocked;

    // Update virtual price without price adjustment first
    uint256 old_virtual_price = virtual_price;
    uint256 xcp = _xcp(_D, price_scale);
    uint256 vp = (total_supply > 0) ? (PRECISION() * xcp / total_supply) : PRECISION();
    // No ramping modeled; enforce non-decrease only matters during ramping in vyper
    // Update xcp_profit following change in virtual price
    xcp_profit = xcp_profit + vp - old_virtual_price;

    // Rebalancing condition
    uint256 threshold_vp = std::max(PRECISION(), (xcp_profit + PRECISION()) / 2);
    uint256 vp_boosted = (locked_supply > 0) ? (PRECISION() * xcp / locked_supply) : vp;
    if (vp_boosted < vp) {
        throw std::runtime_error("negative donation");
    }

    if (trace_enabled) {
        std::cout << "TRACE tp_gating ts=" << block_timestamp
                  << " vp=" << vp
                  << " xcp_profit=" << xcp_profit
                  << " threshold=" << threshold_vp
                  << " locked_supply=" << locked_supply
                  << " vp_boosted=" << vp_boosted
                  << " price_oracle=" << price_oracle
                  << " price_scale=" << price_scale
                  << std::endl;
    }

    if (vp_boosted > threshold_vp + rebalancing_params[0]) {
        uint256 norm = price_oracle * PRECISION() / price_scale;
        if (norm > PRECISION()) norm = norm - PRECISION(); else norm = PRECISION() - norm;
        uint256 adjustment_step = std::max(rebalancing_params[1], norm / 5);

        if (norm > adjustment_step) {
            uint256 p_new = (price_scale * (norm - adjustment_step) + adjustment_step * price_oracle) / norm;

            // Update xp with p_new
            auto xp_new = xp;
            // xp[1] scales with price
            xp_new[1] = xp[1] * p_new / price_scale;

            uint256 D_new = StableswapMath::newton_D(_A_gamma[0], _A_gamma[1], xp_new, 0);
            uint256 new_xcp = _xcp(D_new, p_new);
            uint256 new_vp = (total_supply > 0) ? (PRECISION() * new_xcp / total_supply) : PRECISION();

            if (trace_enabled) {
                std::cout << "TRACE tp_candidate ts=" << block_timestamp
                          << " norm=" << norm
                          << " step=" << adjustment_step
                          << " p_new=" << p_new
                          << " xp1_new=" << xp_new[1]
                          << " D_new=" << D_new
                          << " new_xcp=" << new_xcp
                          << " new_vp_pre_burn=" << new_vp
                          << std::endl;
            }

            uint256 donation_shares_to_burn = 0;
            uint256 goal_vp = std::max(threshold_vp, vp);
            if (new_vp < goal_vp) {
                // what would be total supply with goal_vp and new_xcp
                uint256 tweaked_supply = (PRECISION() * new_xcp) / goal_vp;
                if (!(tweaked_supply < total_supply)) {
                    throw std::runtime_error("tweaked supply must shrink");
                }
                uint256 diff = total_supply - tweaked_supply;
                // Only unlocked donation shares can be burned (matches Vyper local var semantics)
                uint256 donation_unlocked = _donation_shares();
                donation_shares_to_burn = diff < donation_unlocked ? diff : donation_unlocked;
                if (total_supply > donation_shares_to_burn) {
                    new_vp = (PRECISION() * new_xcp) / (total_supply - donation_shares_to_burn);
                }
                if (trace_enabled) {
                    std::cout << "TRACE tp_burn ts=" << block_timestamp
                              << " goal_vp=" << goal_vp
                              << " tweaked_supply=" << tweaked_supply
                              << " burn=" << donation_shares_to_burn
                              << " new_vp_post_burn=" << new_vp
                              << std::endl;
                }
            }

            if (new_vp > PRECISION() && new_vp >= threshold_vp) {
                D = D_new;
                virtual_price = new_vp;
                cached_price_scale = p_new;
                if (donation_shares_to_burn > 0) {
                    donation_shares -= donation_shares_to_burn;
                    totalSupply -= donation_shares_to_burn;
                    last_donation_release_ts = block_timestamp;
                }
                if (trace_enabled) {
                    std::cout << "TRACE tp_commit ts=" << block_timestamp
                              << " new_price_scale=" << p_new
                              << " donation_burnt=" << donation_shares_to_burn
                              << " new_vp=" << new_vp
                              << std::endl;
                }
                return p_new;
            }
        }
    }

    // No price_scale adjustment
    D = _D;
    virtual_price = vp;
    return price_scale;
}

} // namespace twocrypto
