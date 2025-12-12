// Idle tick (EMA update) when no trade occurs
#pragma once

#include <cstdint>

#include "core/common.hpp"
#include "harness/metrics.hpp"

namespace arb {
namespace harness {

// Idle tick configuration
template <typename T>
struct IdleTickCfg {
    uint64_t freq_s{3600};  // seconds between idle ticks (0 = disabled)
    
    bool enabled() const { return freq_s > 0; }
};

// Try to perform an idle tick (EMA/oracle update) if enough time has passed.
// Called when no arb trade was executed for this event.
// Returns true if tick was performed.
template <typename T, typename Pool>
bool try_idle_tick(
    Pool& pool,
    const IdleTickCfg<T>& cfg,
    uint64_t ev_ts,
    Metrics<T>& m
) {
    if (!cfg.enabled()) return false;
    
    // Check if enough time has passed since last pool update
    if (ev_ts < pool.last_timestamp + cfg.freq_s) return false;
    
    const T ps_before = pool.cached_price_scale;
    
    try {
        pool.tick();
        
        const T ps_after = pool.cached_price_scale;
        if (differs_rel(ps_after, ps_before)) {
            m.n_rebalances += 1;
        }
        return true;
        
    } catch (...) {
        // Ignore failed tick
        return false;
    }
}

} // namespace harness
} // namespace arb
