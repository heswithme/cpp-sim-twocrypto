This file explains how we optimized parameters for the usdchf fxpair.

Idea: run backtests for refuel (donation/boost)–enabled twocrypto pools with various parameters, using historical candlestick data, and assuming arbitrageurs behave optimally — i.e., if a profitable trade exists, they will execute it with optimal size.

Trade optimality depends on the pool execution price (a function of pool historical state, current liquidity, and invariant/fee parameters), the CEX price (assumed to allow infinite volume at each candle), and pool/CEX fees. If arbitrageurs can buy cheaper in the pool than on the CEX, they will, pushing the pool price up (and continuously equalizing pool and CEX prices).

Pool parameters (A, mid/out fee, oracle smoothing time, and some internal knobs) determine slippage behavior. Low A → closer to xy=k (bad slippage). High A → closer to x+y=k (good slippage). The invariant used in twocrypto-ng-boost is Curve’s standard stableswap (see whitepaper for math).

We began by scanning two key parameters (A / mid_fee) across a wide range, based on donation APY (1.A-midfee don sweep folder). No miracles were found: resulting APY at any donation rate was net-negative for LPs. This assumes a single LP donating to the pool — their fee income is counted minus donations. If donations come from the asset issuer and LPs are organic users, then LPs earn proportionally to donations (at the issuer’s cost). Importantly, these backtests are only arbitrage-based, aiming to conservatively simulate pool performance. On-chain FX assumes organic volume exists and brings additional fees. Being conservative, we allow LPs’ net APY to be negative.

We chose to fix donation APY at 5% yearly TVL, as this is the maximum achievable with current parameters and strikes a balance between minimizing slippage (see lowest values in don0-2.5-5-7.5-10.png) and overcompensating LPs.

The next folder (2.A-mid_fee don0.05) contains sweeps for mid_fee and A, with donation APY fixed at 5%, to find optimal A and mid_fee.
We took a large initial range of A (1–2000) and mid_fee (0.01%–5%, i.e., 1–500 bps), scanning on a log scale to locate the best region for zoom-in. MA smoothing was tested but made little difference on such a large range, we investigate it on narrower set later.

The key metric is time-weighted slippage (tw_slippage, lower is better). It correlates with pool price tracking ability (avg_rel_bps, lower is better).

Zooming into A=10–1000 and mid_fee=10–100 bps (log scale, see 3a, 3b, 3c), and further into A=50–500 (same mid_fee range, linear scale, see 4,a,b,c), we find that optimal slippage occurs near A≈100 (dark blue front on tw_slippage in image 4). The final heatmap scans A=40–200 with mid_fee=10–100 bps. Notably, we can see that MA=866 (10 min) and MA=1h introduce way more noise to the results, while Ma=1d is smooth and easier to analyze. 

5a & 5b contrast region of interest for MA=10min and MA=1day. While not free from noise, MA=10min seem to provide slightly better tw_slippage, so we use MA=866 for further investigations.

From 4c->5, two candidate parameter sets emerge:
	•	A=120, mid_fee≈35 bps
	•	A=150, mid_fee=75 bps

The first provides better price peg (lower avg_rel_bps in image 5), while the second grants higher LP APY.

To assess off-peg risk, we plotted pool price_scale vs CEX price for these candidates, adding a contrast case (A=200, mid_fee=10 bps). Image 6.price_comparison.png shows strong detachments in the contrast case (green line diverging from CEX for weeks), so we discarded it.

6a.price_comparison_clean.png shows both candidate setups track price reasonably well. The orange line (A=150, fee=75 bps) follows high-frequency CEX moves less, potentially saving rebalances. The blue line (A=120, fee=35 bps) hugs the CEX price more closely, offering slightly better tracking. The main trade-off between the two candidates is fee level (35 bps vs 75 bps), which directly affects LP profitability versus trading cost.
