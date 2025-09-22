This file explains how we optimized parameters for the usdchf fxpair.

Note: study was done for usdchf pair (price inverted from pool), but it was checked later that for chfusd pair (normal) results are the same (yearly crossvalidation contains correct pair and same patterns are seen).

Idea: run backtests for refuel (donation/boost)–enabled twocrypto pools with various parameters, using historical candlestick data, and assuming arbitrageurs behave optimally — i.e., if a profitable trade exists, they will execute it with optimal size.

Trade optimality depends on the pool execution price (a function of pool historical state, current liquidity, and invariant/fee parameters), the CEX price (assumed to allow infinite volume at each candle), and pool/CEX fees. If arbitrageurs can buy cheaper in the pool than on the CEX, they will, pushing the pool price up (and continuously equalizing pool and CEX prices).

Pool parameters (A, mid/out fee, oracle smoothing time, and some internal knobs) determine slippage behavior. Low A → closer to xy=k (bad slippage). High A → closer to x+y=k (good slippage). The invariant used in twocrypto-ng-boost is Curve’s standard stableswap (see whitepaper for math).

We began by scanning two key parameters (A / mid_fee) across a wide range, based on donation APY (1.A-midfee don sweep folder). No miracles were found: resulting APY at any donation rate was net-negative for LPs. This assumes a single LP donating to the pool — their fee income is counted minus donations. If donations come from the asset issuer and LPs are organic users, then LPs earn proportionally to donations (at the issuer’s cost). Importantly, these backtests are only arbitrage-based, aiming to conservatively simulate pool performance. On-chain FX assumes organic volume exists and brings additional fees. Being conservative, we allow LPs’ net APY to be negative.

We chose to fix donation APY at 5% yearly TVL, as this is the maximum achievable with current parameters and strikes a balance between minimizing slippage (see lowest values in don0-2.5-5-7.5-10.png) and overcompensating LPs. LPs external fee (cex fee+redemption costs is set to be 10bps which is realistic conservative value)

The next folder (2.A-mid_fee don0.05) contains sweeps for mid_fee and A, with donation APY fixed at 5%, to find optimal A and mid_fee.
We took a large initial range of A (1–2000) and mid_fee (0.01%–5%, i.e., 1–500 bps), scanning on a log scale to locate the best region for zoom-in. MA smoothing was tested but made little difference on such a large range, we investigate it on narrower set later.

The key metric is time-weighted slippage (tw_slippage, lower is better). It correlates with pool price tracking ability (avg_rel_bps, lower is better).

Zooming into A=10–1000 and mid_fee=10–100 bps (log scale, see 3a, 3b, 3c), and further into A=50–500 (same mid_fee range, linear scale, see 4,a,b,c), we find that optimal slippage occurs near A≈100 (dark blue front on tw_slippage in image 4). The final heatmap scans A=40–200 with mid_fee=10–100 bps. Notably, we can see that MA=866 (10 min) and MA=1h introduce way more noise to the results, while Ma=1d is smooth and easier to analyze. 

5a,b,c,d contrast region of interest for MA=10min, MA=1h, MA=12h and MA=1day. While not free from noise, lower MA values (10min, 1h) seem to provide slightly better minimal tw_slippage, whereas higher MA values (12h, 1day) have less noise and more stable regions of lower tw_slippage. Notably, MA=24h (5d) enables whole region of higher-A, lower-slippage values, so we chose MA=1d as best smooting time for now (we can optimize MA with another pass later on).

5d2 and 5d2 provide  investigation of different external fee levels (2bps and 1bps) to see if this enables more profit for LPs. Result: it does not really, organic voluem is key for LP net profits.

From 5d, two candidate parameter sets emerge:
	•	A=120, mid_fee≈35 bps
	•	A=150, mid_fee=75 bps

The first provides better price peg (lower avg_rel_bps in image 5d), while the second grants higher LP APY.

To assess off-peg risk, we plotted pool price_scale vs CEX price for these candidates, adding a contrast case (A=200, mid_fee=10 bps). Image 6.price_comparison.png shows strong detachments in the contrast case (green line diverging from CEX for weeks), so we discarded it.

6a.price_comparison_clean.png shows both candidate setups track price reasonably well. The orange line (A=150, fee=75 bps) follows high-frequency CEX moves less, potentially saving rebalances. The blue line (A=120, fee=35 bps) hugs the CEX price more closely, offering slightly better tracking. The main trade-off between the two candidates is fee level (35 bps vs 75 bps), which directly affects LP profitability versus trading cost.

Commands used:

uv run python/arb_sim/generate_pools_chf.py && uv run python/arb_sim/arb_sim.py python/arb_sim/trade_data/chfusd/chfusd-2019-2024.json --min-swap 1e-10 --max-swap 1 --dustswapfreq 600 -n 10

uv run python/arb_sim/plot_heatmap.py  --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,arb_pnl_coin0,tw_slippage,tw_liq_density,avg_rel_bps --ncol 5 --font-size 32

Update:
A mode simulating organic volumes was added to arb_harness.cpp. User behavior is modelled as discrete swaps with given frequency in alternating directions. Example command:
uv run python/arb_sim/generate_pools_chf.py && uv run python/arb_sim/arb_sim.py python/arb_sim/trade_data/chfusd/chfusd-2019-2024.json --min-swap 1e-10 --max-swap 1 --dustswapfreq 600 -n 10 --userswapfreq 3600 --userswapsize 0.001 --userswapthresh 0.01
Such command will add user swapping 0.1% of pool i'th coin balance every 3600 seconds (one hour) but only if pool spot price is no more than 1% away from CEX price. Such usage can be considered as conservative approximation of real-world user behavior. For example, if pool TVL is 1M$, that would mean that each hour 500$ of coin 0 or 1 will be swapped irrespective of whether this swap could give better result on cex/fx market, but only oif price is not far from it. 

7a,b demonstrate that such user behavior will significantly improve LP yields from net-zero/negative (when donations are subtracted) to 3-5% APY with best parameters. This tells that any even insignificant organic volume will disbalance the pool bringing fees, and arbitrageurs balancing the pool will bring some more fees as well. We see (from 7b) that lowest slippage can be achieved at A~=220 and mid_fee~=75bps, with ~4% LP net APY. While this result is very promising, advisable initial pool parameters are still those that don't consider any organic volume. If such volume is consistent, pool parameters can be changed by a DAO vote backed with more rigorous simulations.
Additionally we investigated larger pool scale (10m$ tvl instead of 1m$ tvl) and result is such that same user behavior (~500$ swaps per h) isnt sufficient for LP profitability, larger organic swaps (5000$ per hour, or roughly 0.1% of tvl) are needed (figures 7c,7d)