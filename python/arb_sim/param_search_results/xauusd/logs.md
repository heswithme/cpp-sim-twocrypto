XAU/USD Parameter Scan — Experiment Log

We study the gold–USD pair. The only hard constraint is an arbitrage redemption fee of 25 bps; all other knobs (A, mid_fee, oracle MA and auxiliaries) are free to optimize for profitability and slippage.

Commands used:

uv run python/arb_sim/generate_pools_xau.py && uv run python/arb_sim/arb_sim.py --dustswapfreq 600 -n 10
uv run python/arb_sim/plot_heatmap.py --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,arb_pnl_coin0,tw_slippage,tw_liq_density,avg_rel_bps --ncol 5 --font-size 28

We start with a wide logarithmic sweep over amplification A and mid_fee (folder 1. A-mid_fee). In 1a.wide_log_sweep-don0.03.png, A ranges from 1 to 1000 and mid_fee from 1 to 100 bps with a 3% donation rate. At higher A the number of rebalances drops but price deviation rises sharply; beyond a threshold the pool loses the peg during volatility episodes and fails to track the external price. High A is therefore unsuitable for this volatility profile.

To test whether additional subsidy expands the safe region, we repeat the sweep at a 5% donation rate. In 1b.wide_log_sweep-don0.05.png the permissible A range grows slightly, but price tracking still detaches at approximately A ≈ 40.

We then switch to linear axes. 2. zoom_linear-don0.05.png scans A = 1…200 with the same mid_fee range and 5% donation. The detachment observation persists: around A ≈ 40 the pool increasingly fails to follow the external price, with the exact threshold depending on mid_fee (roughly 20–50 bps).

A second linear zoom, 3. zoom2_linear-don0.05.png, narrows A = 1…100 and confirms the same pattern in higher resolution.

For a direct view, 4. price_comparisonA10vs70.png overlays pool price scale against the CEX price for A = 10 and A = 70 using:

uv run python/arb_sim/compare_plots.py --metrics slippage,balance_indicator

The A = 70 configuration fails to track and drifts for extended periods, while A = 10 remains aligned. High A does not earn enough fees to self-correct; imbalance accumulates and the peg degrades.

Finally, 5. zoom_lin-don0.png  repeats the zoomed sweep with no donation. Without the donation mechanism the peg is materially worse and slippage degrades, reinforcing that this asset requires conservative amplification and subsidy to maintain stability.

6. 6.zoom2_lin-don0.05-user3600-0.001-0.01.png:
uv run python/arb_sim/generate_pools_xau.py && uv run python/arb_sim/arb_sim.py  --dustswapfreq 600 -n 10 --userswapfreq 3600 --userswapsize 0.001 --userswapthresh 0.01
Users swap every 3600 seconds 0.001 of alternating side balances (i.e. ~$500 notional per hour on either side). We see that some meaningful profitability is now acquired by LPs (up to 7% yearly, or 3% net, i.e. donations subtracted). Additionally A can be a bit higher for lower slippages, thus if organic volumes persist, pool can vote for A ramp up to 70 (assuming mid_fee is high enough).

Overall, the gold pair demands relatively low A; beyond ~40 the pool detaches under realistic volatility. Donations and organic volumes improve stability; without it, both tracking and slippage worsen. Further tuning of mid_fee and oracle MA should be performed within the low-A corridor identified above.