#include "twocrypto_d.hpp"
#include <iostream>

namespace twocrypto_d {


double TwoCryptoPoolD::add_liquidity(
    const std::array<double, 2>& amounts,
    double min_mint_amount,
    bool donation
) {
    if (amounts[0] + amounts[1] <= 0.0) throw std::invalid_argument("no coins to add");

    std::array<double, 2> old_balances = balances;
    std::array<double, 2> new_balances = {balances[0] + amounts[0], balances[1] + amounts[1]};
    double price_scale = cached_price_scale;
    auto xp = _xp(new_balances, price_scale);
    auto old_xp = _xp(old_balances, price_scale);

    std::array<double, 2> A_gamma = {A, gamma};
    double old_D = D;
    double D_new = StableswapMathD::newton_D(A_gamma[0], A_gamma[1], xp, 0.0);

    double token_supply = totalSupply;
    double d_token = 0.0;
    if (old_D > 0.0) {
        d_token = token_supply * D_new / old_D - token_supply;
    } else {
        d_token = _xcp(D_new, price_scale);
    }

    if (old_D > 0.0) {
        // approximate token fee
        // balances ratio before op (scaled by precisions)
        double denom = (balances[1] - amounts[1]) * precisions[1];
        double balances_ratio = 0.0;
        if (denom > 0.0) balances_ratio = (balances[0] - amounts[0]) * precisions[0] / denom;
        std::array<double, 2> amounts_scaled = {
            amounts[0] * precisions[0],
            amounts[1] * precisions[1] * balances_ratio
        };
        double fee_prime = _fee(xp) * N_COINS / (4.0 * (N_COINS - 1.0));
        double S = amounts_scaled[0] + amounts_scaled[1];
        double avg = S / N_COINS;
        double Sdiff = std::fabs(amounts_scaled[0] - avg) + std::fabs(amounts_scaled[1] - avg);
        double lp_spam_penalty_fee = 0.0;
        if ((balances[0] + balances[1]) > 0.0 && donation_protection_expiry_ts > block_timestamp) {
            double protection_factor = (donation_protection_expiry_ts - block_timestamp) / donation_protection_period;
            if (protection_factor > 1.0) protection_factor = 1.0;
            lp_spam_penalty_fee = protection_factor * fee_prime;
        }
        double approx_fee = fee_prime * Sdiff / std::max(1e-18, S) + lp_spam_penalty_fee;
        double d_token_fee = approx_fee * d_token;
        d_token -= d_token_fee;
    }

    // Constraints before commit
    if (old_D > 0.0 && donation) {
        double new_donation_shares = donation_shares + d_token;
        double denom = (token_supply + d_token);
        double ratio = denom > 0.0 ? (new_donation_shares / denom) : 0.0;
        if (ratio > donation_shares_max_ratio) throw std::runtime_error("donation above cap");
    }
    if (d_token < min_mint_amount) throw std::runtime_error("slippage");

    // Commit
    balances = new_balances;
    if (old_D > 0.0) {
        D = D_new;
        if (donation) {
            double new_donation_shares = donation_shares + d_token;
            double unlocked = _donation_shares(false);
            double new_elapsed = new_donation_shares > 0.0 ? (unlocked * donation_duration) / new_donation_shares : 0.0;
            last_donation_release_ts = block_timestamp - new_elapsed;
            donation_shares = new_donation_shares;
            totalSupply += d_token;
        } else {
            double relative_lp_add = (token_supply + d_token) > 0.0 ? (d_token / (token_supply + d_token)) : 0.0;
            if (relative_lp_add > 0.0 && donation_shares > 0.0) {
                double extension_seconds = (relative_lp_add / donation_protection_lp_threshold) * donation_protection_period;
                if (extension_seconds > donation_protection_period) extension_seconds = donation_protection_period;
                double current_expiry = donation_protection_expiry_ts > block_timestamp ? donation_protection_expiry_ts : block_timestamp;
                double new_expiry = current_expiry + extension_seconds;
                double max_expiry = block_timestamp + donation_protection_period;
                if (new_expiry > max_expiry) new_expiry = max_expiry;
                donation_protection_expiry_ts = new_expiry;
            }
            totalSupply += d_token;
        }
        cached_price_scale = tweak_price(A_gamma, xp, D_new);
    } else {
        D = D_new;
        virtual_price = 1.0;
        xcp_profit = 1.0;
        xcp_profit_a = 1.0;
        totalSupply += d_token;
    }

    return d_token;
}


std::array<double, 2> TwoCryptoPoolD::remove_liquidity(
    double amount,
    const std::array<double, 2>& min_amounts
) {
    if (amount > totalSupply) throw std::invalid_argument("insufficient LP tokens");
    std::array<double, 2> withdrawn;
    for (size_t i = 0; i < N_COINS; ++i) {
        withdrawn[i] = balances[i] * amount / totalSupply;
        if (withdrawn[i] < min_amounts[i]) throw std::runtime_error("withdrawal resulted in fewer coins than expected");
        balances[i] -= withdrawn[i];
    }
    double old_total_supply = totalSupply;
    totalSupply -= amount;
    if (old_total_supply > 0.0) D = D - (D * amount / old_total_supply); else D = 0.0;
    return withdrawn;
}


std::array<double, 3> TwoCryptoPoolD::exchange(
    double i,
    double j,
    double dx,
    double min_dy
) {
    size_t idx_i = i;  // implicit conversion is fine
    size_t idx_j = j;
    if (idx_i == idx_j || idx_i >= N_COINS || idx_j >= N_COINS) throw std::invalid_argument("coin index out of range");
    if (dx <= 0.0) throw std::invalid_argument("zero dx");

    double price_scale = cached_price_scale;
    auto balances_local = balances;
    balances_local[idx_i] += dx;
    auto xp = _xp(balances_local, price_scale);

    std::array<double, 2> A_gamma = {A, gamma};
    auto y_out = StableswapMathD::get_y(A_gamma[0], A_gamma[1], xp, D, idx_j);
    double dy_xp = xp[idx_j] - y_out.value;
    xp[idx_j] -= dy_xp;
    double dy_tokens = dy_xp;
    if (idx_j > 0) dy_tokens = dy_tokens / price_scale;
    dy_tokens = dy_tokens / precisions[idx_j];

    double fee = _fee(xp) * dy_tokens;
    double dy_after_fee = dy_tokens - fee;
    if (dy_after_fee < min_dy) throw std::runtime_error("slippage");

    balances[idx_i] += dx;
    balances[idx_j] -= dy_after_fee;

    auto xp_new = _xp(balances, price_scale);
    double D_new = StableswapMathD::newton_D(A_gamma[0], A_gamma[1], xp_new, 0.0);
    D = D_new;
    double new_price_scale = tweak_price(A_gamma, xp_new, D_new);
    return {dy_after_fee, fee, new_price_scale};
}


double TwoCryptoPoolD::tweak_price(
    const std::array<double, 2>& _A_gamma,
    const std::array<double, 2>& xp,
    double _D
) {
    double price_oracle = cached_price_oracle;
    double price_scale = cached_price_scale;

    uint64_t last_ts = last_timestamp;
    if (last_ts < block_timestamp) {
        double dt = block_timestamp - last_ts;
        double alpha = std::exp(-dt / std::max(1.0, ma_time));
        double capped = last_prices;
        if (capped > 2.0 * price_scale) capped = 2.0 * price_scale;
        price_oracle = capped * (1.0 - alpha) + price_oracle * alpha;
        cached_price_oracle = price_oracle;
        last_timestamp = block_timestamp;
    }

    last_prices = StableswapMathD::get_p(xp, _D, _A_gamma) * price_scale / PRECISION_R();

    double total_supply = totalSupply;
    double donation_unlocked = _donation_shares();
    double locked_supply = total_supply - donation_unlocked;

    double old_virtual_price = virtual_price;
    double xcp = _xcp(_D, price_scale);
    double vp = (total_supply > 0.0) ? (PRECISION_R() * xcp / total_supply) : 1.0;
    xcp_profit = xcp_profit + vp - old_virtual_price;

    double threshold_vp = std::max(1.0, (xcp_profit + 1.0) / 2.0);
    double vp_boosted = (locked_supply > 0.0) ? (PRECISION_R() * xcp / locked_supply) : vp;

    if (vp_boosted > threshold_vp + allowed_extra_profit) {
        double norm = price_oracle * PRECISION_R() / price_scale;
        norm = (norm > 1.0) ? (norm - 1.0) : (1.0 - norm);
        double step = std::max(adjustment_step, norm / 5.0);
        if (norm > step) {
            double p_new = (price_scale * (norm - step) + step * price_oracle) / norm;
            auto xp_new = xp;
            xp_new[1] = xp[1] * p_new / price_scale;
            double D_new = StableswapMathD::newton_D(_A_gamma[0], _A_gamma[1], xp_new, 0.0);
            double new_xcp = _xcp(D_new, p_new);
            double new_vp = (total_supply > 0.0) ? (PRECISION_R() * new_xcp / total_supply) : 1.0;

            double burn = 0.0;
            double goal_vp = std::max(threshold_vp, vp);
            if (new_vp < goal_vp) {
                double tweaked_supply = (PRECISION_R() * new_xcp) / goal_vp;
                if (tweaked_supply < total_supply) {
                    double diff = total_supply - tweaked_supply;
                    double unlocked2 = _donation_shares();
                    burn = std::min(diff, unlocked2);
                    if (total_supply > burn) new_vp = (PRECISION_R() * new_xcp) / (total_supply - burn);
                }
            }

            if (new_vp > 1.0 && new_vp >= threshold_vp) {
                D = D_new;
                virtual_price = new_vp;
                cached_price_scale = p_new;
                if (burn > 0.0) {
                    donation_shares -= burn;
                    totalSupply -= burn;
                    last_donation_release_ts = block_timestamp;
                }
                return p_new;
            }
        }
    }

    D = _D;
    virtual_price = vp;
    return price_scale;
}

// Explicit instantiation for double only
// No template instantiation needed anymore

} // namespace twocrypto_d
