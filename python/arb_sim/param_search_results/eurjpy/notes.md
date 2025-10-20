Sweeping eurjpy
Commands:
uv run python/arb_sim/generate_pools_generic.py && uv run python/arb_sim/arb_sim.py  --dustswapfreq 600 -n 10
uv run python/arb_sim/plot_heatmap.py  --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,apy_geom_mean,apy_geom_mean_net,avg_rel_bps,tw_slippage,tw_liq_density,tw_real_slippage_1pct,tw_real_slippage_5pct,tw_real_slippage_10pct --ncol 5 --font-size 28

1. A-fee_logweep - seems that as with other FX pairs - not enough volatility to exeed donations spendings. (don rate here 0.05)
2. A-don_rate - with larger donations it's optimal to have larger A (fee here 10-20 bps)
3. A-fee_gamma - don rate 0.05, fees 10-20 bps, A zoomed in. Optimal fee_gamma (to keep avg fee at most 15bps) and A (to have best slippage/geom_mean_apy) - seems A~30, and fee_gamma~0.0001
3a. Zooming in. A = 30,  fee_gamma = 0.0001. 
4. A-ma_time. Seing if increase in ma_time helps with apy/slippage. It significantly increases slippage compared to cex_swp, so we should keep it 10 min (866).
5. A-fee_gamma-user: added small organic user flow (--userswapfreq 3600 --userswapsize 0.001 --userswapthresh 0.01) to see benefits. Slightly improves APY.

Resulting pool parameters for donation rate 0.05 and arb_fee 10bps:
A = 30
mid_fee = 10bps
out_fee = 20bps
fee_gamma = 0.0001
allowed_extra_profit = 1e-12
adjustment_step = 1e-7
ma_time = 866
initial_price = 1
start_timestamp = 1
donation_apy = 0.05
donation_frequency = 7*86400
donation_coins_ratio = 0.5