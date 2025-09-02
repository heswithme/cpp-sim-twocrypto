// Unified TwoCrypto pool (templated on numeric type)
//
// Readability goal: mirror the structure and clarity of the Vyper implementation
// (Twocrypto.vy), keeping function order and behavior aligned for parity.
#pragma once

#include <array>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "stableswap_math.hpp"

namespace twocrypto {

// Forward declare traits
template <typename T>
struct NumTraits;

template <>
struct NumTraits<stableswap::uint256> {
    using T = stableswap::uint256;
    static T PRECISION() { static T v("1000000000000000000"); return v; }
    static T FEE_PRECISION() { static T v("10000000000"); return v; }
    static T A_MULTIPLIER() { return T(10000); }
    static T NOISE_FEE() { static T v("100000"); return v; }
    static T ZERO() { return T(0); }
    static T ONE() { return T(1); }
    static T ROUNDING_UNIT_XP() { return T(1); }
    static bool is_zero(const T& x) { return x == 0; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

template <>
struct NumTraits<double> {
    using T = double;
    static T PRECISION() { return 1.0; }
    static T FEE_PRECISION() { return 1.0; }
    static T A_MULTIPLIER() { return 10000.0; }
    static T NOISE_FEE() { return 1e-5; }
    static T ZERO() { return 0.0; }
    static T ONE() { return 1.0; }
    static T ROUNDING_UNIT_XP() { return 0.0; }
    static bool is_zero(const T& x) { return std::abs(x) <= 0.0; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

template <>
struct NumTraits<float> {
    using T = float;
    static T PRECISION() { return 1.0f; }
    static T FEE_PRECISION() { return 1.0f; }
    static T A_MULTIPLIER() { return 10000.0f; }
    static T NOISE_FEE() { return 1e-5f; }
    static T ZERO() { return 0.0f; }
    static T ONE() { return 1.0f; }
    static T ROUNDING_UNIT_XP() { return 0.0f; }
    static bool is_zero(const T& x) { return std::abs(x) <= 0.0f; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

template <>
struct NumTraits<long double> {
    using T = long double;
    static T PRECISION() { return 1.0L; }
    static T FEE_PRECISION() { return 1.0L; }
    static T A_MULTIPLIER() { return 10000.0L; }
    static T NOISE_FEE() { return 1e-12L; }
    static T ZERO() { return 0.0L; }
    static T ONE() { return 1.0L; }
    static T ROUNDING_UNIT_XP() { return 0.0L; }
    static bool is_zero(const T& x) { return fabsl(x) <= 0.0L; }
    static T max(const T& a, const T& b) { return a > b ? a : b; }
    static T min(const T& a, const T& b) { return a < b ? a : b; }
};

template <typename T>
class TwoCryptoPoolT {
public:
    using Ops = stableswap::MathOps<T>;
    using Traits = NumTraits<T>;
    static constexpr int N_COINS = 2;

    // State variables
    std::array<T, 2> balances{Traits::ZERO(), Traits::ZERO()};
    T D = Traits::ZERO();
    T totalSupply = Traits::ZERO();

    // Price variables
    T cached_price_scale = Traits::PRECISION();
    T cached_price_oracle = Traits::PRECISION();
    T last_prices = Traits::PRECISION();
    uint64_t last_timestamp = 0;

    // Parameters (normalized)
    T A = Traits::ZERO();
    T gamma = Traits::ZERO();
    T mid_fee = Traits::ZERO();
    T out_fee = Traits::ZERO();
    T fee_gamma = Traits::ZERO();
    T allowed_extra_profit = Traits::ZERO();
    T adjustment_step = Traits::ZERO();
    T ma_time = Traits::ONE();

    // Profit tracking
    T xcp_profit = Traits::PRECISION();
    T xcp_profit_a = Traits::PRECISION();
    T virtual_price = Traits::PRECISION();

    // Token precisions
    std::array<T, 2> precisions{Traits::ONE(), Traits::ONE()};

    // Time for testing
    uint64_t block_timestamp = 0;

    // Donations
    T donation_shares = Traits::ZERO();
    T donation_shares_max_ratio = Traits::PRECISION() * 10 / 100; // 10%
    T donation_duration = T(7 * 86400);
    T last_donation_release_ts = Traits::ZERO();
    T donation_protection_expiry_ts = Traits::ZERO();
    T donation_protection_period = T(10 * 60);
    T donation_protection_lp_threshold = Traits::PRECISION() * 20 / 100; // 20%

    T admin_fee = NumTraits<T>::FEE_PRECISION() / 2; // 50%
    uint64_t last_admin_fee_claim_timestamp = 0;

public:
    TwoCryptoPoolT(
        const std::array<T, 2>& _precisions,
        const T& _A,
        const T& _gamma,
        const T& _mid_fee,
        const T& _out_fee,
        const T& _fee_gamma,
        const T& _allowed_extra_profit,
        const T& _adjustment_step,
        const T& _ma_time,
        const T& initial_price
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

        xcp_profit = Traits::PRECISION();
        xcp_profit_a = Traits::PRECISION();
        virtual_price = Traits::PRECISION();

        block_timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        last_timestamp = block_timestamp;
    }

private:
    // ------------------------ Internal helpers (parity with Vyper internal) ------------------------

    // _xp: balances scaled to common precision with price_scale for coin1
    std::array<T, 2> _xp(
        const std::array<T, 2>& _balances,
        const T& price_scale
    ) const {
        return {
            _balances[0] * precisions[0],
            _balances[1] * precisions[1] * price_scale / Traits::PRECISION()
        };
    }

    // _fee: dynamic fee between mid_fee and out_fee based on balance skew
    T _fee(const std::array<T, 2>& xp) const {
        if (fee_gamma == Traits::ZERO()) {
            return mid_fee;
        }

        T Bsum = xp[0] + xp[1];
        if (Bsum == Traits::ZERO()) {
            return mid_fee;
        }

        // B = sum(x)^2 / (4*x0*x1) in Vyper form transformed; we keep identical algebra
        T B = Traits::PRECISION() * N_COINS * N_COINS * xp[0] / Bsum * xp[1] / Bsum;

        // fee_gamma * B / (fee_gamma * B + 1 - B)
        B = fee_gamma * B /
            (fee_gamma * B / Traits::PRECISION() + Traits::PRECISION() - B);

        // Blend between mid_fee and out_fee
        return (
            mid_fee * B + out_fee * (Traits::PRECISION() - B)
        ) / Traits::PRECISION();
    }

    // _xcp: cross-product invariant in xcp units (parity with Vyper virtual price math)
    T _xcp(const T& _D, const T& price_scale) const {
        if constexpr (std::is_same_v<T, stableswap::uint256>) {
            auto sqrt_price = boost::multiprecision::sqrt(
                Traits::PRECISION() * price_scale
            );
            return _D * Traits::PRECISION() / N_COINS / sqrt_price;
        } else if constexpr (std::is_same_v<T, long double>) {
            long double sp = std::sqrt(static_cast<long double>(Traits::PRECISION() * price_scale));
            return _D * Traits::PRECISION() / T(N_COINS) / T(sp);
        } else {
            double sp = std::sqrt(static_cast<double>(Traits::PRECISION() * price_scale));
            return _D * Traits::PRECISION() / T(N_COINS) / T(sp);
        }
    }

    // _donation_shares: unlocked donation supply (optionally protected)
    T _donation_shares(bool donation_protection = true) const {
        if (donation_shares == Traits::ZERO()) {
            return Traits::ZERO();
        }

        T elapsed  = T(block_timestamp) - last_donation_release_ts;
        T unlocked = donation_shares * elapsed / donation_duration;
        if (unlocked > donation_shares) {
            unlocked = donation_shares;
        }

        if (!donation_protection) {
            return unlocked;
        }

        T protection_factor = Traits::ZERO();
        if (donation_protection_expiry_ts > T(block_timestamp)) {
            protection_factor = (
                (donation_protection_expiry_ts - T(block_timestamp)) *
                Traits::PRECISION() /
                donation_protection_period
            );
            if (protection_factor > Traits::PRECISION()) {
                protection_factor = Traits::PRECISION();
            }
        }

        return (
            unlocked * (Traits::PRECISION() - protection_factor)
        ) / Traits::PRECISION();
    }

    // _calc_token_fee: liquidity op fee approximation
    T _calc_token_fee(
        const std::array<T, 2>& amounts,
        const std::array<T, 2>& xp,
        bool donation,
        bool deposit
    ) const {
        if (donation) {
            return Traits::NOISE_FEE();
        }

        // balances ratio before liquidity op
        T denom = (balances[1] - amounts[1]) * precisions[1];
        T balances_ratio = Traits::ZERO();
        if (denom > Traits::ZERO()) {
            balances_ratio = (
                (balances[0] - amounts[0]) * precisions[0] * Traits::PRECISION()
            ) / denom;
        }

        std::array<T, 2> amounts_scaled = {
            amounts[0] * precisions[0],
            amounts[1] * precisions[1] * balances_ratio / Traits::PRECISION()
        };

        T fee_prime = _fee(xp) * N_COINS / (4 * (N_COINS - 1));

        T S = amounts_scaled[0] + amounts_scaled[1];
        if (S == Traits::ZERO()) {
            return Traits::NOISE_FEE();
        }

        T avg   = S / N_COINS;
        T diff0 = (amounts_scaled[0] > avg) ? (amounts_scaled[0] - avg)
                                            : (avg - amounts_scaled[0]);
        T diff1 = (amounts_scaled[1] > avg) ? (amounts_scaled[1] - avg)
                                            : (avg - amounts_scaled[1]);
        T Sdiff = diff0 + diff1;

        T lp_spam_penalty_fee = Traits::ZERO();
        if (deposit && donation_protection_expiry_ts > T(block_timestamp)) {
            T protection_factor = (
                (donation_protection_expiry_ts - T(block_timestamp)) *
                Traits::PRECISION() /
                donation_protection_period
            );
            if (protection_factor > Traits::PRECISION()) {
                protection_factor = Traits::PRECISION();
            }
            lp_spam_penalty_fee = protection_factor * fee_prime / Traits::PRECISION();
        }

        return fee_prime * Sdiff / S + Traits::NOISE_FEE() + lp_spam_penalty_fee;
    }

public:
    // Cheap tick to update EMA/oracle and possibly adjust price_scale without a swap
    void tick() {
        auto A_gamma = std::array<T, 2>{ A, gamma };
        const auto xp = _xp(balances, cached_price_scale);
        cached_price_scale = tweak_price(A_gamma, xp, D);
    }

    // API
    // add_liquidity: deposit into the pool; supports donation mode with cap semantics
    T add_liquidity(
        const std::array<T, 2>& amounts,
        T min_mint_amount,
        bool donation = false
    ) {
        if (amounts[0] + amounts[1] == Traits::ZERO()) {
            throw std::invalid_argument("no coins to add");
        }

        auto old_balances = balances;
        auto new_balances = std::array<T, 2>{
            balances[0] + amounts[0],
            balances[1] + amounts[1]
        };

        T price_scale = cached_price_scale;
        auto xp     = _xp(new_balances, price_scale);
        auto old_xp = _xp(old_balances, price_scale);
        (void)old_xp; // maintained for parity/trace; not used below

        auto A_gamma = std::array<T, 2>{ A, gamma };

        T old_D = D;
        T D_new = Ops::newton_D(A_gamma[0], A_gamma[1], xp, T(0));
        T token_supply = totalSupply;

        T d_token = Traits::ZERO();
        if (old_D > Traits::ZERO()) {
            d_token = token_supply * D_new / old_D - token_supply;
        } else {
            d_token = _xcp(D_new, price_scale);
        }

        if (old_D > Traits::ZERO()) {
            T approx_fee  = _calc_token_fee(amounts, xp, donation, /*deposit=*/true);
            T d_token_fee = approx_fee * d_token / NumTraits<T>::FEE_PRECISION();
            if constexpr (std::is_same_v<T, stableswap::uint256>) {
                // Match Vyper: add +1 after division to avoid rounding undercharge
                d_token_fee += Traits::ONE();
            }
            d_token -= d_token_fee;
        }

        // Constraints before commit
        if (old_D > Traits::ZERO() && donation) {
            T new_donation_shares = donation_shares + d_token;
            T ratio = (
                new_donation_shares * Traits::PRECISION()
            ) / (token_supply + d_token);
            if (ratio > donation_shares_max_ratio) {
                throw std::runtime_error("donation above cap");
            }
        }
        if (d_token < min_mint_amount) {
            throw std::runtime_error("slippage");
        }

        // Commit
        balances = new_balances;
        if (old_D > Traits::ZERO()) {
            D = D_new;

            if (donation) {
                T new_donation_shares = donation_shares + d_token;
                T unlocked = _donation_shares(false);
                T new_elapsed = Traits::ZERO();
                if (new_donation_shares > Traits::ZERO()) {
                    new_elapsed = (unlocked * donation_duration) / new_donation_shares;
                }
                last_donation_release_ts = T(block_timestamp) - new_elapsed;
                donation_shares = new_donation_shares;
                totalSupply += d_token;
            } else {
                T relative_lp_add = (
                    d_token * Traits::PRECISION()
                ) / (token_supply + d_token);

                if (relative_lp_add > Traits::ZERO() && donation_shares > Traits::ZERO()) {
                    T extension_seconds = (
                        relative_lp_add * donation_protection_period
                    ) / donation_protection_lp_threshold;

                    if (extension_seconds > donation_protection_period) {
                        extension_seconds = donation_protection_period;
                    }

                    T current_expiry = (
                        donation_protection_expiry_ts > T(block_timestamp)
                    ) ? donation_protection_expiry_ts : T(block_timestamp);

                    T new_expiry = current_expiry + extension_seconds;
                    T max_expiry = T(block_timestamp) + donation_protection_period;

                    if (new_expiry > max_expiry) {
                        new_expiry = max_expiry;
                    }
                    donation_protection_expiry_ts = new_expiry;
                }
                totalSupply += d_token;
            }

            cached_price_scale = tweak_price(A_gamma, xp, D_new);
        } else {
            D = D_new;
            virtual_price = Traits::PRECISION();
            xcp_profit    = Traits::PRECISION();
            xcp_profit_a  = Traits::PRECISION();
            totalSupply  += d_token;
        }

        return d_token;
    }

    // remove_liquidity: burn LP to withdraw proportionally
    std::array<T, 2> remove_liquidity(
        T amount,
        const std::array<T, 2>& min_amounts
    ) {
        if (amount > totalSupply) {
            throw std::invalid_argument("insufficient LP tokens");
        }

        std::array<T, 2> withdrawn{};
        for (size_t i = 0; i < N_COINS; ++i) {
            withdrawn[i] = balances[i] * amount / totalSupply;
            if (withdrawn[i] < min_amounts[i]) {
                throw std::runtime_error("withdrawal resulted in fewer coins than expected");
            }
            balances[i] -= withdrawn[i];
        }

        T old_total_supply = totalSupply;
        totalSupply -= amount;

        if (old_total_supply > Traits::ZERO()) {
            D = D - (D * amount / old_total_supply);
        } else {
            D = Traits::ZERO();
        }

        return withdrawn;
    }

    // exchange: swap coin i for coin j (dx in, dy out; returns [dy_after_fee, fee, new_price_scale])
    std::array<T, 3> exchange(
        T i,
        T j,
        T dx,
        T min_dy
    ) {
        size_t idx_i = static_cast<size_t>(i);
        size_t idx_j = static_cast<size_t>(j);

        if (idx_i == idx_j || idx_i >= N_COINS || idx_j >= N_COINS) {
            throw std::invalid_argument("coin index out of range");
        }
        if (dx == Traits::ZERO()) {
            throw std::invalid_argument("zero dx");
        }

        T price_scale = cached_price_scale;

        auto balances_local = balances;
        balances_local[idx_i] += dx;
        auto xp = _xp(balances_local, price_scale);

        auto A_gamma = std::array<T, 2>{ A, gamma };

        auto y_out = Ops::get_y(A_gamma[0], A_gamma[1], xp, D, idx_j);
        T dy_xp = xp[idx_j] - y_out.value;
        xp[idx_j] -= dy_xp;

        T dy_tokens = dy_xp - NumTraits<T>::ROUNDING_UNIT_XP();
        if (idx_j > 0) {
            // coin1 scaling
            dy_tokens = dy_tokens * Traits::PRECISION() / price_scale;
        }
        dy_tokens = dy_tokens / precisions[idx_j];

        T fee = _fee(xp) * dy_tokens / NumTraits<T>::FEE_PRECISION();
        T dy_after_fee = dy_tokens - fee;
        if (dy_after_fee < min_dy) {
            throw std::runtime_error("slippage");
        }

        balances[idx_i] += dx;
        balances[idx_j] -= dy_after_fee;
        auto xp_new = _xp(balances, price_scale);
        T D_new = Ops::newton_D(A_gamma[0], A_gamma[1], xp_new, 0);
        D = D_new;
        T new_price_scale = tweak_price(A_gamma, xp_new, D_new);
        return { dy_after_fee, fee, new_price_scale };
    }

    T tweak_price(
        const std::array<T, 2>& _A_gamma,
        const std::array<T, 2>& xp,
        T _D
    ) {
        const bool trace = (std::getenv("TRACE") && std::string(std::getenv("TRACE")) == "1");
        T price_oracle = cached_price_oracle;
        T price_scale  = cached_price_scale;

        // EMA update (only update last_timestamp on this path)
        uint64_t last_ts = last_timestamp;
        if (last_ts < block_timestamp) {
            T dt = T(block_timestamp - last_ts);
            if constexpr (std::is_same_v<T, stableswap::uint256>) {
                auto neg = stableswap::int256(
                    -(
                        stableswap::int256(dt) *
                        stableswap::int256(NumTraits<T>::PRECISION()) /
                        stableswap::int256(ma_time)
                    )
                );
                T alpha  = stableswap::MathOps<T>::wad_exp(neg);
                T capped = last_prices;
                if (capped > 2 * price_scale) capped = 2 * price_scale;
                price_oracle = (
                    capped * (NumTraits<T>::PRECISION() - alpha) + price_oracle * alpha
                ) / NumTraits<T>::PRECISION();
            } else {
                auto alpha = std::exp(
                    - static_cast<double>(dt) / static_cast<double>(ma_time)
                );
                T capped = last_prices;
                if (capped > 2 * price_scale) capped = 2 * price_scale;
                price_oracle = capped * (T(1) - T(alpha)) + price_oracle * T(alpha);
            }
            cached_price_oracle = price_oracle;
            last_timestamp      = block_timestamp;
        }

        // Update last_prices from current state
        last_prices = (
            stableswap::MathOps<T>::get_p(xp, _D, _A_gamma) * price_scale
        ) / NumTraits<T>::PRECISION();

        // Compute current virtual price and profits
        T total_supply      = totalSupply;
        T donation_unlocked = _donation_shares();
        T locked_supply     = total_supply - donation_unlocked;

        T old_virtual_price = virtual_price;
        T xcp               = _xcp(_D, price_scale);
        T vp = (total_supply > NumTraits<T>::ZERO())
            ? (NumTraits<T>::PRECISION() * xcp / total_supply)
            : NumTraits<T>::PRECISION();

        xcp_profit = xcp_profit + vp - old_virtual_price;
        if (trace) {
            if constexpr (std::is_same_v<T, stableswap::uint256>) {
                    std::cout << "TRACE tp_ema price_oracle=" << price_oracle.template convert_to<std::string>()
                          << " last_prices=" << last_prices.template convert_to<std::string>()
                          << " price_scale=" << price_scale.template convert_to<std::string>()
                          << "\n";
            } else {
                std::cout << "TRACE tp_ema price_oracle=" << price_oracle
                          << " last_prices=" << last_prices
                          << " price_scale=" << price_scale
                          << "\n";
            }
        }
        T threshold_vp = NumTraits<T>::max(
            NumTraits<T>::PRECISION(),
            (xcp_profit + NumTraits<T>::PRECISION()) / 2
        );

        T vp_boosted = (locked_supply > NumTraits<T>::ZERO())
            ? (NumTraits<T>::PRECISION() * xcp / locked_supply)
            : vp;
        if (trace) {
            if constexpr (std::is_same_v<T, stableswap::uint256>) {
                std::cout << "TRACE tp_gating vp=" << vp.template convert_to<std::string>()
                          << " threshold=" << threshold_vp.template convert_to<std::string>()
                          << " vp_boosted=" << vp_boosted.template convert_to<std::string>()
                          << " allowed_extra=" << allowed_extra_profit.template convert_to<std::string>()
                          << "\n";
            } else {
                std::cout << "TRACE tp_gating vp=" << vp
                          << " threshold=" << threshold_vp
                          << " vp_boosted=" << vp_boosted
                          << " allowed_extra=" << allowed_extra_profit
                          << "\n";
            }
        }

        // Price adjustment path
        if (vp_boosted > threshold_vp + allowed_extra_profit) {
            T norm = price_oracle * NumTraits<T>::PRECISION() / price_scale;
            if (norm > NumTraits<T>::PRECISION()) {
                norm = norm - NumTraits<T>::PRECISION();
            } else {
                norm = NumTraits<T>::PRECISION() - norm;
            }

            T step = NumTraits<T>::max(adjustment_step, norm / 5);
            if (trace) {
                if constexpr (std::is_same_v<T, stableswap::uint256>) {
                    std::cout << "TRACE tp_norm norm=" << norm.template convert_to<std::string>()
                              << " step=" << step.template convert_to<std::string>()
                              << "\n";
                } else {
                    std::cout << "TRACE tp_norm norm=" << norm
                              << " step=" << step
                              << "\n";
                }
            }
            if (norm > step) {
                T p_new = (price_scale * (norm - step) + step * price_oracle) / norm;

                auto xp_new = xp;
                xp_new[1] = xp[1] * p_new / price_scale;

                T D_new   = stableswap::MathOps<T>::newton_D(
                    _A_gamma[0], _A_gamma[1], xp_new, 0
                );
                T new_xcp = _xcp(D_new, p_new);
                T new_vp  = (total_supply > NumTraits<T>::ZERO())
                    ? (NumTraits<T>::PRECISION() * new_xcp / total_supply)
                    : NumTraits<T>::PRECISION();

                // Optional burn to hit goal virtual_price
                T burn    = NumTraits<T>::ZERO();
                T goal_vp = NumTraits<T>::max(threshold_vp, vp);
                if (new_vp < goal_vp) {
                    T tweaked_supply = (NumTraits<T>::PRECISION() * new_xcp) / goal_vp;
                    if (tweaked_supply < total_supply) {
                        T diff      = total_supply - tweaked_supply;
                        T unlocked2 = _donation_shares();
                        burn        = (diff < unlocked2) ? diff : unlocked2;

                        if (total_supply > burn) {
                            new_vp = (
                                NumTraits<T>::PRECISION() * new_xcp
                            ) / (total_supply - burn);
                        }
                    }
                }

                // Commit if within allowed region
                if (trace) {
                    if constexpr (std::is_same_v<T, stableswap::uint256>) {
                        std::cout << "TRACE tp_candidate p_new=" << p_new.template convert_to<std::string>()
                                  << " new_vp=" << new_vp.template convert_to<std::string>()
                                  << " burn=" << burn.template convert_to<std::string>()
                                  << "\n";
                    } else {
                        std::cout << "TRACE tp_candidate p_new=" << p_new
                                  << " new_vp=" << new_vp
                                  << " burn=" << burn
                                  << "\n";
                    }
                }
                if (new_vp > NumTraits<T>::PRECISION() && new_vp >= threshold_vp) {
                    D = D_new;
                    virtual_price      = new_vp;
                    cached_price_scale = p_new;
                    if (burn > NumTraits<T>::ZERO()) {
                        donation_shares          -= burn;
                        totalSupply              -= burn;
                        last_donation_release_ts  = T(block_timestamp);
                    }
                    if (trace) {
                        std::cout << "TRACE tp_commit price_scale=";
                        if constexpr (std::is_same_v<T, stableswap::uint256>) {
                            std::cout << cached_price_scale.template convert_to<std::string>();
                        } else {
                            std::cout << cached_price_scale;
                        }
                        std::cout << "\n";
                    }
                    return p_new;
                }
            }
        }

        D = _D;
        virtual_price = vp;
        return price_scale;
    }

    // ------------------------ Views ------------------------
    // Expose unlocked donation shares (optionally protected) for observers
    T donation_unlocked(bool donation_protection = true) const {
        return _donation_shares(donation_protection);
    }

    T get_virtual_price() const {
        return (virtual_price == Traits::ZERO()) ? Traits::PRECISION() : virtual_price;
    }

    T get_p() const {
        if (balances[0] == Traits::ZERO() || balances[1] == Traits::ZERO()) {
            return cached_price_scale;
        }
        return last_prices;
    }

    // ------------------------ Testing helpers ------------------------
    void set_block_timestamp(uint64_t ts) {
        block_timestamp = ts;
        // Align EMA baseline at initialization to Vyper deploy semantics.
        // Only do this before any liquidity has been added.
        if (D == Traits::ZERO() && totalSupply == Traits::ZERO()) {
            last_timestamp = ts;
        }
    }

    void advance_time(uint64_t seconds) {
        block_timestamp += seconds;
    }
};

using TwoCryptoPoolI = TwoCryptoPoolT<stableswap::uint256>;
using TwoCryptoPoolD = TwoCryptoPoolT<double>;

} // namespace twocrypto
