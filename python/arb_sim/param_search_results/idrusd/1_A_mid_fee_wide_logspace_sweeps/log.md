# IDRUSD Parameter Optimization Log

## 1. Wide Logspace Sweep A vs mid_fee - October 19, 2024 3:43 AM PT

IDRUSD showed similar volatility profile to BRLUSD, so ran a first pass logspace sweep of A versus mid_fee

### Commands
```bash
uv run python/arb_sim/generate_pools_idr.py
uv run python/arb_sim/arb_sim.py python/arb_sim/trade_data/idrusd/2_idrusd_json_converted.json --dustswapfreq 600 -n 10 --candle-filter 10.0
uv run python/arb_sim/plot_heatmap.py --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,apy_geom_mean,apy_geom_mean_net,avg_rel_bps,tw_slippage,tw_liq_density --ncol 5 --font-size 28 --out python/arb_sim/param_search_results/idrusd/1_A_mid_fee_wide_logspace_sweeps/idrusd_heatmap.png
```

### Parameters

**Dynamic:**
- A: logspace sweep 10K - 2MM
- mid_fee: logspace sweep 1 - 100 bps

**Fixed:**
- donation_apy: 0.03
- out_fee: 20 bps
- gamma: 1e-4
- fee_gamma: 0.01
- adjustment_step: 0.00015
- ma_time: 866
- dustswapfreq: 600
