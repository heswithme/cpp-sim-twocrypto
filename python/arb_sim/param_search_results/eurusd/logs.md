# 1. A-mid_fee sweeps (1. A-mid_fee folder)

0.wide_sweep.png:
uv run python/arb_sim/generate_pools_eur.py && uv run python/arb_sim/arb_sim.py  --dustswapfreq 600 -n 10
uv run python/arb_sim/plot_heatmap.py  --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,arb_pnl_coin0,tw_slippage,tw_liq_density,avg_rel_bps --ncol 5 --font-size 28

We begin the study with a wide parameter sweep (on log scale), summarized in "0. wide_sweep.png". We scan A from 1 to 1000 and mid_fee from 1 bps (0.01%) to 100 bps (1%). The donation rate is fixed at 3% per year and the arbitrage redemption fee at 10 bps. The issuer requests deployment with mid_fee = 3 bps and A = 50; we still run the unconstrained sweep to map the landscape before fixing fee and tuning A, oracle MA, and initialization knobs.

From 0.wide_sweep.png, LP APY first turns visibly positive around mid_fee ≈ 20 bps and A ≈ 70, but net profitability sits near 0–0.1% because donations offset fee income. For parameter choice we do not target profitability; we target slippage minimization and elimination of price deviation versus the external oracle.

The tw_slippage panel shows a broad low-slippage band for A ≈ 20–250, with the minimum shifting with fees: higher fees (~1%) push the optimum to A ≥ 150–200, while lower fees pull it toward A ≈ 50. The avg_rel_bps panel displays a sharp blue front of low deviation followed by a quick rise; at larger A the pool struggles to track, and the price scale drifts from the oracle. We therefore should keep A low enough to preserve tracking while retaining low slippage. 

1. zoom_linear.png:
uv run python/arb_sim/generate_pools_eur.py && uv run python/arb_sim/arb_sim.py  --dustswapfreq 600 -n 10
uv run python/arb_sim/plot_heatmap.py  --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,arb_pnl_coin0,tw_slippage,tw_liq_density,avg_rel_bps --ncol 5 --font-size 28

We switch to linear scale and Zoom in the region of A from 10 to 100 and for mid-fee from 1 to approximately 30. We see that on the low-fee region of uh sub-5bps, to remain profitable and to keep pool pegged better, we should have Lower parameter A not fifty, but rather around thirty. To adequately provide good price tracking and low slippage at A equals fifty the pool should have fee of approximately 10 Bps. 

1a. zoom_linear_userswap.png:
uv run python/arb_sim/generate_pools_eur.py && uv run python/arb_sim/arb_sim.py  --dustswapfreq 600 -n 10 --userswapfreq 3600 --userswapsize 0.001 --userswapthresh 0.01
uv run python/arb_sim/plot_heatmap.py  --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,arb_pnl_coin0,tw_slippage,tw_liq_density,avg_rel_bps --ncol 5 --font-size 28

We extend the zoomed linear sweep by adding conservative user flow to the simulator. The user module executes a swap every 3600 seconds with size equal to 0.1% of the current pool's balance of the selected coin, alternating directions each event, and only if the pool spot deviates by at most 1% from the external price. For a balanced pool with TVL ≈ $1M this corresponds to ~$500 notional per hour on either side.

With this flow present, the figure shows a broader low-deviation region and improved tracking at higher A for the same fee band. Relative to the pure-arb case, the stable corridor shifts upward in A and slightly leftward in fee, allowing tighter price following at lower fees. Net LP profitability becomes sustainably positive in a meaningful part of the plane, while tw_slippage remains low where avg_rel_bps is dark blue. This confirms that even modest, price-aware organic volume materially improves peg quality and LP outcomes without requiring aggressive fees.


# 2. A-ma_time sweeps (2. A-ma_time folder)

1. linear_sweep,out=mid.png:
We run linear, two-parameter sweeps over A and ma_time with fees fixed, then repeat with a higher out fee. The x-axis is ma_time on a linear grid from 600/ln 2 to 7*86400/ln 2 seconds; the y-axis is amplification A on a linear grid from 10 to 100. In the first pass I fix mid_fee = 3 bps and out_fee = 3 bps. In this configuration, the lowest time-weighted slippage at A = 50 occurs around ma_time ≈ 2.5 days (~310,169). Pushing A to 70 at the same ma_time reduces slippage a bit further, but it comes with lower APY and noticeably worse price following (higher avg_rel_bps). Given the issuer’s fee constraint, the A = 50, ma_time=310200 point is the cleanest balance between slippage and tracking.

2. linear_sweep,out>mid.png:
In the second pass we raise the imbalanced fee to out_fee = 5 bps while keeping mid_fee = 3 bps. Results improve slightly across the relevant band, but the qualitative picture is unchanged: for A = 50 the best region still centers around ma_time=310200, with only marginal gains from the higher out fee. 