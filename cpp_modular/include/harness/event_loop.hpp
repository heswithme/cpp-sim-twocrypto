// Event loop for processing price events and executing arb trades
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "core/common.hpp"
#include "events/types.hpp"
#include "harness/metrics.hpp"
#include "harness/actions.hpp"
#include "harness/detailed_output.hpp"
#include "harness/donation.hpp"
#include "harness/idle_tick.hpp"
#include "harness/user_swap.hpp"
#include "trading/costs.hpp"
#include "trading/decision.hpp"
#include "trading/arbitrageur.hpp"
#include "pools/twocrypto_fx/helpers.hpp"

namespace arb {
namespace harness {

// Configuration for APY tracking
template <typename T>
struct ApyConfig {
    uint64_t period_s{0};    // 0 = disabled
    int cap_pct{100};        // Cap per-window APY at this percent
};

// Process events and execute trades with full metrics sampling.
// Returns EventLoopResult with all metrics.
template <typename T, typename Pool>
EventLoopResult<T> run_event_loop(
    Pool& pool,
    const std::vector<Event>& events,
    const trading::Costs<T>& costs,
    DonationCfg<T>& dcfg,
    IdleTickCfg<T>& icfg,
    UserSwapCfg<T>& ucfg,
    T min_swap_frac = T(1e-6),
    T max_swap_frac = T(1.0),
    size_t max_events = 0,  // 0 = all events
    const ApyConfig<T>& apy_cfg = ApyConfig<T>{},
    bool save_actions = false,
    bool detailed_log = false
) {
    EventLoopResult<T> result{};
    Metrics<T>& m = result.metrics;
    TimeWeightedMetrics<T>& tw = result.tw_metrics;
    SlippageProbes<T>& sp = result.slippage_probes;
    
    const size_t n_events = max_events > 0 ? std::min(max_events, events.size()) : events.size();
    if (n_events == 0) return result;
    
    // Record timestamps
    result.t_start = events.front().ts;
    result.t_end = events.back().ts;
    
    // Record initial state
    result.tvl_start = pool.balances[0] + pool.balances[1] * pool.cached_price_scale;
    result.true_growth_initial = pools::twocrypto_fx::true_growth(pool);
    result.initial_liq = pool.balances;
    result.donation_apy = dcfg.apy;
    
    // Probe sizes for slippage: 1%, 5%, 10% of initial TVL (coin0 terms)
    std::array<T, SlippageProbes<T>::N_SIZES> probe_sizes_coin0{};
    for (size_t k = 0; k < SlippageProbes<T>::N_SIZES; ++k) {
        probe_sizes_coin0[k] = result.tvl_start * static_cast<T>(SlippageProbes<T>::SIZE_FRACS[k]);
    }
    
    // APY tracker
    ApyTracker<T> apy_tracker{};
    if (apy_cfg.period_s > 0) {
        // Use (xcp_profit + 1) / 2 to match old harness exactly
        T lb_initial = (pool.xcp_profit + T(1)) / T(2);
        apy_tracker.init(result.t_start, lb_initial, result.true_growth_initial, 
                         apy_cfg.period_s, apy_cfg.cap_pct);
    }
    
    // Detailed logging: track last candle timestamp to detect boundaries
    uint64_t last_candle_ts = 0;
    Candle last_candle{};
    
    // Helper to log detailed entry at candle boundary
    auto log_detailed_entry = [&](const Candle& candle) {
        if (!detailed_log) return;
        DetailedEntry<T> entry;
        entry.t = candle.ts;
        entry.token0 = pool.balances[0];
        entry.token1 = pool.balances[1];
        entry.price_oracle = pool.cached_price_oracle;
        entry.price_scale = pool.cached_price_scale;
        entry.profit = pool.get_virtual_price() - T(1);
        entry.xcp = pool.xcp_profit;
        entry.open = static_cast<T>(candle.open);
        entry.high = static_cast<T>(candle.high);
        entry.low = static_cast<T>(candle.low);
        entry.close = static_cast<T>(candle.close);
        // Compute dynamic fee at current pool state
        const auto xp_now = pools::twocrypto_fx::pool_xp_current(pool);
        entry.fee = pools::twocrypto_fx::dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);
        result.detailed_entries.push_back(entry);
    };
    
    // Helper to sample slippage probes
    auto sample_slippage_probes = [&](uint64_t ts, T p_cex) {
        if (!(p_cex > T(0))) return;
        for (size_t k = 0; k < SlippageProbes<T>::N_SIZES; ++k) {
            sp.accumulate_previous(k, ts);
            
            const T S = probe_sizes_coin0[k];
            // 0 -> 1
            {
                auto pr = pools::twocrypto_fx::simulate_exchange_once(pool, 0, 1, S);
                const T dy1 = pr.first;  // coin1 after fee
                const T ideal1 = S / p_cex;
                T s01 = T(0);
                if (ideal1 > T(0)) {
                    s01 = T(1) - (dy1 / ideal1);
                }
                // 1 -> 0
                const T dx1 = S / p_cex;
                auto pr10 = pools::twocrypto_fx::simulate_exchange_once(pool, 1, 0, dx1);
                const T dy0 = pr10.first;  // coin0 after fee
                T s10 = T(0);
                if (S > T(0)) {
                    s10 = T(1) - (dy0 / S);
                }
                sp.sample(k, ts, s01, s10);
            }
        }
    };
    
    // Helper to sample instantaneous d,r
    auto sample_latent = [&](uint64_t ts) {
        T cur_d = T(0), cur_r = T(0);
        if (pools::twocrypto_fx::instantaneous_dr(pool, cur_d, cur_r)) {
            tw.sample_latent(ts, cur_d, cur_r);
        }
    };
    
    for (size_t ev_idx = 0; ev_idx < n_events; ++ev_idx) {
        const auto& ev = events[ev_idx];
        
        // Update pool timestamp
        pool.set_block_timestamp(ev.ts);
        
        const T cex_price = static_cast<T>(ev.p_cex);
        
        // APY window tracking
        if (apy_cfg.period_s > 0) {
            // Use (xcp_profit + 1) / 2 to match old harness exactly
            T lb_curr = (pool.xcp_profit + T(1)) / T(2);
            T tg_curr = pools::twocrypto_fx::true_growth(pool);
            apy_tracker.update(ev.ts, lb_curr, tg_curr);
        }
        
        // ---- Time-weighted sampling at start of event (pre-trade) ----
        
        // Price error sampling
        tw.sample_price_error(ev.ts, pool.cached_price_scale, cex_price);
        
        // Fee sampling
        {
            const auto xp_now = pools::twocrypto_fx::pool_xp_current(pool);
            const T cur_fee = pools::twocrypto_fx::dyn_fee(xp_now, pool.mid_fee, pool.out_fee, pool.fee_gamma);
            tw.sample_fee(ev.ts, cur_fee);
        }
        
        // Band tracking (>10% deviation from CEX)
        tw.sample_band(ev.ts, pool.cached_price_scale, cex_price);
        
        // Try donation before trading
        auto don_res = make_donation_ex(pool, dcfg, ev.ts, m);
        if (save_actions && don_res.success) {
            DonationAction<T> act;
            act.ts = ev.ts;
            act.ts_due = don_res.ts_due;
            act.amounts = don_res.amounts;
            act.price_scale = don_res.price_scale;
            act.donation_ratio1 = dcfg.ratio1;
            act.apy_per_year = dcfg.apy;
            act.freq_s = dcfg.freq_s;
            result.actions.push_back(std::move(act));
        }
        
        if (!(cex_price > T(0))) {
            // Detailed logging: log at candle boundary even when skipping
            if (detailed_log) {
                const bool is_last = (ev_idx == n_events - 1);
                const bool candle_changed = (last_candle_ts > 0 && ev.candle.ts != last_candle_ts);
                if (candle_changed) {
                    log_detailed_entry(last_candle);
                }
                last_candle_ts = ev.candle.ts;
                last_candle = ev.candle;
                if (is_last) {
                    log_detailed_entry(last_candle);
                }
            }
            continue;
        }
        
        // Try user swap before arb decision
        try_user_swap(pool, ucfg, ev.ts, cex_price);
        
        // Decide trade
        T notional_cap = std::numeric_limits<T>::infinity();
        if (costs.use_volume_cap) {
            notional_cap = static_cast<T>(ev.volume) * costs.volume_cap_mult;
        }
        
        auto dec = trading::decide_trade(
            pool, cex_price, costs,
            notional_cap,
            min_swap_frac, max_swap_frac
        );
        
        if (!dec.do_trade) {
            // No profitable trade - try idle tick for EMA update
            // Capture pre-tick state for action recording
            const T ps_before = pool.cached_price_scale;
            const T oracle_before = pool.cached_price_oracle;
            const T xcp_profit_before = pool.xcp_profit;
            const T vp_before = pool.get_vp_boosted();
            
            const bool did_tick = try_idle_tick(pool, icfg, ev.ts, m);
            
            if (did_tick) {
                // Sample after tick
                sample_latent(ev.ts);
                sample_slippage_probes(ev.ts, cex_price);
                
                // Record tick action
                if (save_actions) {
                    TickAction<T> act;
                    act.ts = ev.ts;
                    act.p_cex = cex_price;
                    act.ps_before = ps_before;
                    act.ps_after = pool.cached_price_scale;
                    act.oracle_before = oracle_before;
                    act.oracle_after = pool.cached_price_oracle;
                    act.xcp_profit_before = xcp_profit_before;
                    act.xcp_profit_after = pool.xcp_profit;
                    act.vp_before = vp_before;
                    act.vp_after = pool.get_vp_boosted();
                    result.actions.push_back(std::move(act));
                }
            }
            // Detailed logging: log at candle boundary even when no trade
            if (detailed_log) {
                const bool is_last = (ev_idx == n_events - 1);
                const bool candle_changed = (last_candle_ts > 0 && ev.candle.ts != last_candle_ts);
                if (candle_changed) {
                    log_detailed_entry(last_candle);
                }
                last_candle_ts = ev.candle.ts;
                last_candle = ev.candle;
                if (is_last) {
                    log_detailed_entry(last_candle);
                }
            }
            continue;
        }
        
        // Execute trade
        try {
            // Capture pre-trade state for action recording
            const T ps_before = pool.cached_price_scale;
            const T oracle_before = pool.cached_price_oracle;
            const T xcp_profit_before = pool.xcp_profit;
            const T vp_before = pool.get_vp_boosted();
            const T p_pool_before = pool.get_p();
            const uint64_t last_ts_before = pool.last_timestamp;
            // Note: "lp" in old harness means last_prices (instantaneous price), not LP tokens
            const T lp_before = pool.last_prices;
            
            auto res = pool.exchange(
                static_cast<T>(dec.i),
                static_cast<T>(dec.j),
                dec.dx,
                T(0)  // min_dy
            );
            
            // res[0] = dy_after_fee, res[1] = fee_tokens
            const T dy_after_fee = res[0];
            const T fee_tokens = res[1];
            const T ps_after = pool.cached_price_scale;
            
            // Update metrics
            m.trades += 1;
            m.notional += dec.notional_coin0;
            m.lp_fee_coin0 += (dec.j == 1 ? fee_tokens * cex_price : fee_tokens);
            m.arb_pnl_coin0 += dec.profit;
            
            // Track effective dynamic pool fee (fraction), size-weighted
            const T gross_dy_tokens = dy_after_fee + fee_tokens;
            if (gross_dy_tokens > T(0) && dec.notional_coin0 > T(0)) {
                const T fee_frac = fee_tokens / gross_dy_tokens;
                m.fee_wsum += fee_frac * dec.notional_coin0;
                m.fee_w += dec.notional_coin0;
            }
            
            // Count rebalance if price_scale changed
            if (differs_rel(ps_after, ps_before)) {
                m.n_rebalances += 1;
            }
            
            // Sample after trade
            sample_latent(ev.ts);
            sample_slippage_probes(ev.ts, cex_price);
            
            // Record exchange action
            if (save_actions) {
                ExchangeAction<T> act;
                act.ts = ev.ts;
                act.i = dec.i;
                act.j = dec.j;
                act.dx = dec.dx;
                act.dy_after_fee = dy_after_fee;
                act.fee_tokens = fee_tokens;
                act.profit_coin0 = dec.profit;
                act.p_cex = cex_price;
                act.p_pool_before = p_pool_before;
                act.p_pool_after = pool.get_p();
                act.oracle_before = oracle_before;
                act.oracle_after = pool.cached_price_oracle;
                act.ps_before = ps_before;
                act.ps_after = ps_after;
                act.last_ts_before = last_ts_before;
                act.last_ts_after = pool.last_timestamp;
                act.lp_before = lp_before;
                act.lp_after = pool.last_prices;  // Note: "lp" means last_prices in old harness
                act.xcp_profit_before = xcp_profit_before;
                act.xcp_profit_after = pool.xcp_profit;
                act.vp_before = vp_before;
                act.vp_after = pool.get_vp_boosted();
                act.slippage = tw.last_r_inst;
                act.liq_density = tw.last_d_inst;
                act.balance_indicator = pools::twocrypto_fx::balance_indicator(pool);
                result.actions.push_back(std::move(act));
            }
            
        } catch (...) {
            // Trade failed; ignore and continue
        }
        
        // Detailed logging: log at candle boundary (when candle.ts changes or last event)
        if (detailed_log) {
            const bool is_last = (ev_idx == n_events - 1);
            const bool candle_changed = (last_candle_ts > 0 && ev.candle.ts != last_candle_ts);
            
            // Log the *previous* candle when we see a new one
            if (candle_changed) {
                log_detailed_entry(last_candle);
            }
            
            // Track current candle
            last_candle_ts = ev.candle.ts;
            last_candle = ev.candle;
            
            // Log final candle at end of simulation
            if (is_last) {
                log_detailed_entry(last_candle);
            }
        }
    }
    
    // Copy out APY tracker results
    result.tw_capped_apy = apy_tracker.tw_capped_apy();
    result.tw_apy_geom_mean = apy_tracker.tw_geom_mean_apy();
    if (result.tw_capped_apy >= 0.0) {
        result.tw_capped_apy_net = result.tw_capped_apy - static_cast<double>(dcfg.apy);
    }
    // tw_apy_geom_mean_net: always compute if tw_apy_geom_mean was set (can be negative)
    if (result.tw_apy_geom_mean > -1.0) {
        result.tw_apy_geom_mean_net = result.tw_apy_geom_mean - static_cast<double>(dcfg.apy);
    }
    
    return result;
}

// Backward-compatible overload returning just Metrics<T>
template <typename T, typename Pool>
Metrics<T> run_event_loop_simple(
    Pool& pool,
    const std::vector<Event>& events,
    const trading::Costs<T>& costs,
    DonationCfg<T>& dcfg,
    IdleTickCfg<T>& icfg,
    UserSwapCfg<T>& ucfg,
    T min_swap_frac = T(1e-6),
    T max_swap_frac = T(1.0),
    size_t max_events = 0
) {
    auto result = run_event_loop(pool, events, costs, dcfg, icfg, ucfg,
                                  min_swap_frac, max_swap_frac, max_events);
    return result.metrics;
}

} // namespace harness
} // namespace arb
