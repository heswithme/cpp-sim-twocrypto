// JSON output writer for harness results
#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <boost/json.hpp>

#include "harness/runner.hpp"
#include "pools/twocrypto_fx/helpers.hpp"

namespace json = boost::json;

namespace arb {
namespace harness {

// Convert a floating-point value to a string scaled by 1e18 (matching Vyper's uint256 format)
template <typename T>
std::string to_str_1e18(T val) {
    auto scaled = static_cast<long double>(val) * 1e18L;
    if (scaled < 0) scaled = 0;
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(0);
    oss << scaled;
    return oss.str();
}

// Convert pool result's final state to JSON object
template <typename T>
json::object pool_state_json(const PoolResult<T>& r) {
    json::object o;
    o["balances"] = json::array{to_str_1e18(r.balances[0]), to_str_1e18(r.balances[1])};
    // xp = balances scaled with price_scale (matches old harness)
    o["xp"] = json::array{
        to_str_1e18(r.balances[0]),  // precisions[0] = 1 for floating types
        to_str_1e18(r.balances[1] * r.price_scale)  // precisions[1] = 1 for floating types
    };
    o["D"] = to_str_1e18(r.D);
    o["virtual_price"] = to_str_1e18(r.virtual_price);
    o["xcp_profit"] = to_str_1e18(r.xcp_profit);
    o["price_scale"] = to_str_1e18(r.price_scale);
    o["price_oracle"] = to_str_1e18(r.price_oracle);
    o["last_prices"] = to_str_1e18(r.last_prices);
    o["totalSupply"] = to_str_1e18(r.totalSupply);
    o["donation_shares"] = to_str_1e18(r.donation_shares);
    o["donation_unlocked"] = to_str_1e18(r.donation_unlocked);
    o["timestamp"] = r.timestamp;
    return o;
}

// Compute end-state APY metrics from PoolResult
template <typename T>
json::object compute_apy_metrics(const PoolResult<T>& r) {
    json::object apy;
    
    constexpr double SEC_PER_YEAR = 365.0 * 86400.0;
    const double duration_s = r.duration_s();
    
    if (duration_s <= 0.0 || r.tvl_start <= T(0)) {
        // Can't compute APYs
        apy["apy"] = -1.0;
        apy["apy_xcp"] = -1.0;
        apy["apy_coin0"] = -1.0;
        apy["apy_coin0_boost"] = -1.0;
        apy["apy_coin0_raw"] = -1.0;
        apy["apy_coin0_boost_raw"] = -1.0;
        apy["apy_geom_mean"] = -1.0;
        apy["apy_geom_mean_net"] = -1.0;
        return apy;
    }
    
    const double exponent = SEC_PER_YEAR / duration_s;
    
    // Virtual price based APY
    const double vp_end = static_cast<double>(r.virtual_price);
    double apy_vp = (vp_end > 0.0) ? std::pow(vp_end, exponent) - 1.0 : -1.0;
    
    // xcp_profit based APY
    const double xcp_end = static_cast<double>((r.xcp_profit + T(1)) / T(2));
    double apy_xcp = (xcp_end > 0.0) ? std::pow(xcp_end, exponent) - 1.0 : -1.0;
    
    // TVL based APYs
    const T tvl_end = r.balances[0] + r.balances[1] * r.price_scale;
    const T v_hold_end = r.initial_liq[0] + r.initial_liq[1] * r.price_scale;
    
    // Raw (non-baseline) APYs
    double apy_coin0_raw = (tvl_end > T(0)) 
        ? std::pow(static_cast<double>(tvl_end / r.tvl_start), exponent) - 1.0 
        : -1.0;
    
    const double tvl_end_adj_raw = static_cast<double>(tvl_end) - static_cast<double>(r.metrics.donation_coin0_total);
    double apy_coin0_boost_raw = (tvl_end_adj_raw > 0.0)
        ? std::pow(tvl_end_adj_raw / static_cast<double>(r.tvl_start), exponent) - 1.0
        : -1.0;
    
    // Baseline-subtracted (excess) APYs: compare to HODL valued at end price
    double apy_coin0 = -1.0;
    double apy_coin0_boost = -1.0;
    if (v_hold_end > T(0)) {
        apy_coin0 = std::pow(static_cast<double>(tvl_end / v_hold_end), exponent) - 1.0;
        const double tvl_end_adj = static_cast<double>(tvl_end) - static_cast<double>(r.metrics.donation_coin0_total);
        apy_coin0_boost = (tvl_end_adj > 0.0)
            ? std::pow(tvl_end_adj / static_cast<double>(v_hold_end), exponent) - 1.0
            : -1.0;
    }
    
    // True growth based APY
    const T true_growth_end = pools::twocrypto_fx::true_growth_from_balances<T>(
        r.balances, r.price_scale);
    double apy_geom_mean = -1.0;
    double apy_geom_mean_net = -1.0;
    if (r.true_growth_initial > T(0) && true_growth_end > T(0)) {
        const double ratio = static_cast<double>(true_growth_end / r.true_growth_initial);
        apy_geom_mean = std::pow(ratio, exponent) - 1.0;
        apy_geom_mean_net = apy_geom_mean - static_cast<double>(r.donation_apy);
    }
    
    apy["apy"] = apy_vp;
    apy["apy_xcp"] = apy_xcp;
    apy["apy_coin0"] = apy_coin0;
    apy["apy_coin0_boost"] = apy_coin0_boost;
    apy["apy_coin0_raw"] = apy_coin0_raw;
    apy["apy_coin0_boost_raw"] = apy_coin0_boost_raw;
    apy["apy_geom_mean"] = apy_geom_mean;
    apy["apy_geom_mean_net"] = apy_geom_mean_net;
    
    return apy;
}

// Convert all metrics to summary JSON object (matches original harness format)
template <typename T>
json::object metrics_to_summary(const PoolResult<T>& r, size_t n_events) {
    json::object summary;
    const auto& m = r.metrics;
    const auto& tw = r.tw_metrics;
    const auto& sp = r.slippage_probes;
    
    // Core metrics
    summary["events"] = static_cast<uint64_t>(n_events);
    summary["trades"] = static_cast<uint64_t>(m.trades);
    summary["total_notional_coin0"] = static_cast<double>(m.notional);
    summary["lp_fee_coin0"] = static_cast<double>(m.lp_fee_coin0);
    summary["avg_pool_fee"] = static_cast<double>(m.avg_pool_fee());
    summary["tw_avg_pool_fee"] = tw.tw_avg_pool_fee();
    summary["arb_pnl_coin0"] = static_cast<double>(m.arb_pnl_coin0);
    summary["n_rebalances"] = static_cast<uint64_t>(m.n_rebalances);
    summary["donations"] = static_cast<uint64_t>(m.donations);
    summary["donation_coin0_total"] = static_cast<double>(m.donation_coin0_total);
    summary["donation_amounts_total"] = json::array{
        static_cast<double>(m.donation_amounts_total[0]),
        static_cast<double>(m.donation_amounts_total[1])
    };
    summary["pool_exec_ms"] = r.elapsed_ms;
    
    // Time-weighted latent slippage/liquidity density (normalized vs xy=k)
    summary["tw_slippage"] = tw.tw_slippage();
    summary["tw_liq_density"] = tw.tw_liq_density();
    
    // Real slippage probes
    summary["tw_real_slippage_1pct_01"] = sp.tw_slippage_01(0);
    summary["tw_real_slippage_1pct_10"] = sp.tw_slippage_10(0);
    summary["tw_real_slippage_5pct_01"] = sp.tw_slippage_01(1);
    summary["tw_real_slippage_5pct_10"] = sp.tw_slippage_10(1);
    summary["tw_real_slippage_10pct_01"] = sp.tw_slippage_01(2);
    summary["tw_real_slippage_10pct_10"] = sp.tw_slippage_10(2);
    summary["tw_real_slippage_1pct"] = sp.tw_slippage(0);
    summary["tw_real_slippage_5pct"] = sp.tw_slippage(1);
    summary["tw_real_slippage_10pct"] = sp.tw_slippage(2);
    
    // Price follow metrics
    summary["avg_rel_bps"] = tw.avg_rel_bps();
    summary["max_rel_bps"] = tw.max_rel_bps();
    summary["cex_follow_time_frac"] = tw.cex_follow_time_frac(r.duration_s());
    
    // Time-weighted APY from tracker
    summary["tw_capped_apy"] = r.tw_capped_apy;
    summary["tw_capped_apy_net"] = r.tw_capped_apy_net;
    summary["tw_apy_geom_mean"] = r.tw_apy_geom_mean;
    summary["tw_apy_geom_mean_net"] = r.tw_apy_geom_mean_net;
    
    // Timestamps and duration
    summary["t_start"] = r.t_start;
    summary["t_end"] = r.t_end;
    summary["duration_s"] = r.duration_s();
    summary["tvl_coin0_start"] = static_cast<double>(r.tvl_start);
    
    // End-state TVL
    const T tvl_end = r.balances[0] + r.balances[1] * r.price_scale;
    summary["tvl_coin0_end"] = static_cast<double>(tvl_end);
    
    // Baseline HODL value at end price
    const T v_hold_end = r.initial_liq[0] + r.initial_liq[1] * r.price_scale;
    summary["baseline_hold_end_coin0"] = static_cast<double>(v_hold_end);
    
    // Compute and merge APY metrics
    auto apy_metrics = compute_apy_metrics(r);
    for (const auto& kv : apy_metrics) {
        summary[kv.key()] = kv.value();
    }
    
    // Additional end-state metrics
    summary["vp"] = static_cast<double>(r.virtual_price);
    summary["vp_boosted"] = static_cast<double>(r.vp_boosted);
    summary["vpminusone"] = static_cast<double>(r.virtual_price - T(1));
    
    // True growth
    const T true_growth_end = pools::twocrypto_fx::true_growth_from_balances<T>(
        r.balances, r.price_scale);
    if (r.true_growth_initial > T(0)) {
        summary["true_growth"] = static_cast<double>(true_growth_end / r.true_growth_initial);
    } else {
        summary["true_growth"] = -1.0;
    }
    
    return summary;
}

// Output format for the entire run
template <typename T>
json::object build_output_json(
    const std::vector<PoolResult<T>>& results,
    size_t n_events,
    const std::string& data_path,
    bool use_events,   // true => events_file key, false => candles_file key
    size_t n_threads,
    double candles_read_ms,
    double exec_ms
) {
    // Metadata
    json::object meta;
    if (use_events) {
        meta["events_file"] = data_path;
    } else {
        meta["candles_file"] = data_path;
    }
    meta["events"] = static_cast<uint64_t>(n_events);
    meta["threads"] = static_cast<uint64_t>(n_threads);
    meta["candles_read_ms"] = candles_read_ms;
    meta["exec_ms"] = exec_ms;
    
    // Build runs array
    json::array runs;
    runs.reserve(results.size());
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        
        json::object run;
        
        // Result summary (includes all metrics now)
        run["result"] = metrics_to_summary(r, n_events);
        
        // Params block (echoes back original pool/costs JSON)
        json::object params;
        params["pool"] = r.echo_pool;
        if (!r.echo_costs.empty()) {
            params["costs"] = r.echo_costs;
        }
        run["params"] = params;
        
        // Final state
        run["final_state"] = pool_state_json(r);
        
        // Success flag
        run["success"] = r.success;
        if (!r.success) {
            run["error"] = r.error_msg;
        }
        
        // Actions array (only if save_actions was enabled and we have actions)
        if (!r.actions.empty()) {
            run["actions"] = actions_to_json(r.actions);
        }
        
        runs.push_back(std::move(run));
    }
    
    // Build output object
    json::object O;
    O["metadata"] = meta;
    O["runs"] = runs;
    
    return O;
}

// Write results to JSON file
template <typename T>
bool write_results_json(
    const std::string& output_path,
    const std::vector<PoolResult<T>>& results,
    size_t n_events,
    const std::string& data_path,
    bool use_events,   // true => events_file key, false => candles_file key
    size_t n_threads,
    double candles_read_ms,
    double exec_ms
) {
    auto O = build_output_json(
        results, n_events, data_path, use_events, n_threads,
        candles_read_ms, exec_ms
    );
    
    std::ofstream of(output_path);
    if (!of) {
        return false;
    }
    
    of << json::serialize(O) << '\n';
    return of.good();
}

} // namespace harness
} // namespace arb
