// Donation scheduler for arbitrage harness
#pragma once

#include <cstdint>
#include <array>

#include "core/common.hpp"
#include "harness/metrics.hpp"

namespace arb {
namespace harness {

// Donation configuration and scheduling state
template <typename T>
struct DonationCfg {
    bool     enabled{false};
    uint64_t freq_s{0};
    uint64_t next_ts{0};
    T        apy{0};              // fraction per year, e.g., 0.05
    T        ratio1{T(0.5)};      // fraction of donation in coin1
    
    // Initialize from pool config parameters
    void init(T donation_apy, T donation_frequency, T donation_coins_ratio, uint64_t start_ts) {
        if (donation_apy > T(0) && donation_frequency > T(0)) {
            enabled = true;
            apy = donation_apy;
            freq_s = static_cast<uint64_t>(donation_frequency);
            ratio1 = std::clamp<T>(donation_coins_ratio, T(0), T(1));
            next_ts = start_ts;
        }
    }
};

// Compute coin0-equivalent value of amounts at given price_scale
template <typename T>
inline T coin0_equiv(T amt0, T amt1, T ps) {
    return amt0 + amt1 * ps;
}

// Try to donate one period's worth if due.
// Advances schedule by exactly one period (no catch-up for missed periods).
// Updates metrics on success.
template <typename T, typename Pool>
void make_donation(
    Pool& pool,
    DonationCfg<T>& cfg,
    uint64_t ev_ts,
    Metrics<T>& m
) {
    if (!cfg.enabled || cfg.next_ts == 0 || ev_ts < cfg.next_ts) return;

    // Compute one-period donation from current TVL
    constexpr T SEC_PER_YEAR = static_cast<T>(365.0 * 86400.0);
    const T ps   = pool.cached_price_scale;
    const T tvl0 = pool.balances[0] + pool.balances[1] * ps;
    const T frac = cfg.apy * static_cast<T>(static_cast<long double>(cfg.freq_s) / static_cast<long double>(SEC_PER_YEAR));
    const T coin0_equiv_amt = tvl0 * frac;
    const T amt0 = (T(1) - cfg.ratio1) * coin0_equiv_amt;
    const T amt1 = (ps > T(0)) ? (cfg.ratio1 * coin0_equiv_amt / ps) : T(0);

    const uint64_t ts_due = cfg.next_ts;
    const T ps_before = pool.cached_price_scale;
    
    try {
        (void)pool.add_liquidity({amt0, amt1}, T(0), /*donation=*/true);
        const T ps_after = pool.cached_price_scale;
        
        if (differs_rel(ps_after, ps_before)) {
            m.n_rebalances += 1;
        }
        m.donations += 1;
        m.donation_amounts_total[0] += amt0;
        m.donation_amounts_total[1] += amt1;
        m.donation_coin0_total += coin0_equiv(amt0, amt1, ps_before);
        
    } catch (...) {
        // Ignore failed donation (e.g., donation cap exceeded)
    }

    // Advance schedule by exactly one period (no catch-up)
    cfg.next_ts = ts_due + cfg.freq_s;
}

} // namespace harness
} // namespace arb
