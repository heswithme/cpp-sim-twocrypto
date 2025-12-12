// Synthetic user swap simulation
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

namespace arb {
namespace harness {

// User swap configuration
template <typename T>
struct UserSwapCfg {
    uint64_t freq_s{0};           // seconds between user swaps (0 = disabled)
    T        size_frac{T(0.01)};  // fraction of from-side balance per swap
    T        thresh{T(0.05)};     // max relative deviation from CEX price allowed
    
    // Runtime state
    uint64_t next_ts{0};          // next scheduled user swap timestamp
    size_t   next_dir{0};         // 0 = coin0->coin1, 1 = coin1->coin0
    
    bool enabled() const { return freq_s > 0 && size_frac > T(0); }
    
    void init(uint64_t start_ts) {
        if (enabled()) {
            next_ts = start_ts + freq_s;
        }
    }
};

// Try to perform a synthetic user swap if due.
// Alternates direction each swap, only executes if pool price is close to CEX.
// Returns true if swap was executed.
template <typename T, typename Pool>
bool try_user_swap(
    Pool& pool,
    UserSwapCfg<T>& cfg,
    uint64_t ev_ts,
    T p_cex
) {
    if (!cfg.enabled()) return false;
    if (cfg.next_ts == 0 || ev_ts < cfg.next_ts) return false;
    
    // Advance schedule regardless of whether swap succeeds
    cfg.next_ts += cfg.freq_s;
    
    // Validate CEX price
    if (!(p_cex > T(0))) return false;
    
    // Check pool spot price is close enough to CEX
    const T spot = pool.get_p();
    if (!(spot > T(0))) return false;
    
    const T rel = std::abs(spot / p_cex - T(1));
    if (rel > cfg.thresh) return false;
    
    // Determine swap direction and amount
    const size_t i_from = cfg.next_dir & 1;
    const size_t j_to = i_from ^ 1;
    
    const T bal_from = pool.balances[i_from];
    if (!(bal_from > T(0))) return false;
    
    T frac = cfg.size_frac;
    if (frac > T(1)) frac = T(1);
    if (!(frac > T(0))) return false;
    
    const T dx = bal_from * frac;
    if (!(dx > T(0))) return false;
    
    try {
        (void)pool.exchange(static_cast<T>(i_from), static_cast<T>(j_to), dx, T(0));
        cfg.next_dir ^= 1;  // alternate direction for next swap
        return true;
        
    } catch (...) {
        // Ignore failed user swap
        return false;
    }
}

} // namespace harness
} // namespace arb
