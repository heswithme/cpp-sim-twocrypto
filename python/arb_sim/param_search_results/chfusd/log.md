New backtesting round of CHFUSD pair.
Commands used:

uv run python/arb_sim/generate_pools_chf.py && uv run python/arb_sim/arb_sim.py  --dustswapfreq 600 -n 10

uv run python/arb_sim/plot_heatmap.py  --metrics apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,apy_geom_mean,apy_geom_mean_net,avg_rel_bps,tw_slippage,tw_liq_density --ncol 5 --font-size 28 --clamp


Initial assumptions:
Arbitrageur costs are 10bps for whole roundtrip including gas/external fees.
Preferred pool fee is as low as possible, but 10bps is a reasonable max to keep pool alive. We set mid_fee 10bps and out_fee 20bps to stabilize the pool when unbalanced.
Donation rate is 5% yearly, but can be higher as ultimate goal is to provide good swapping venue with low slippage and nice price tracking.


Values to optimize:

1) A: higher is better, but very high values risk stucking the pool.
2) donation_apy - must investigate if higher values bring significant improvements.
3) ma_time - oracle smoothing - preferrable to have 10min, but must see if reasonably high value (1h, 4h) stabilize the picture.
4) allowed_extra_profit & adjustment_step - internal tweak_price parameters, must find best.
5) fee_gamma: 0 means fee=out_fee, 1 means fee=mid_fee. Must optimize so that avg_fee is closer to mid_fee.

Optimization journal:

1. A-mid_fee
1a. birdeye view, logscale, A 1-250, mid_fee 1-100bps
Obviously higher fee is better, and even allow for higher A without losing price. Net apy (if subtract donations) never gets positive (assuming arb-only trades). LPs will get up to 5% apy (due to donations acting as compensation), but donations alone don't print money. avg_rel_bps shows price detachment at higher As and lower fees. apy_geom_mean acts nicely filtering out price detachment zone. Since we have fees set at 0.1-0.2, we can now scan for best A + best donation_apy.

2. A-donation_apy
fees fixed at 10 bps (mid=out=10bps), we check donation_apy from 1-10% yearly. We see that higher donation rates don't bring net profits (except for A=1, donation=1%, but that has very bad slippage). There's a linear dependency between optimal A and donation_apy. More we donate to pool, less slippage we can achieve. Let's keep donation at 5% as agreed with asset issuer earlier.

3. A-ma_time
a. We scan linear scale A 1-100 and ma_time 10min-7 days. We see that higher ma_time allow for higher A value. However as expected we see that avg_rel_bps (how far pool price scale differs from external price) grows as we increase ma_time. It's best to fix ma_time to lower values. Let's proceed with ma_time=1h.
b. we zoom in to determina optimal A with ma_time=1h. We see that optimal A is ~32. We use this value for further optimization.

4. allowed_extra_profit & adjustment_step
X_name = "allowed_extra_profit" 
xmin = int(1e-16 * 10**18)
xmax = int(1e-6 * 10**18)
xlogspace = True

Y_name = "adjustment_step"
ymin = int(1e-14 * 10**18)
ymax =  int(1e-8 * 10**18)
ylogspace = True

allowed_extra_profit - gating the tweak_price, adjustment_step - minimal step of tweak_price.

values dont seem to affect much, so we will use to some safe defaults
    "allowed_extra_profit": int(1e-12 * 10**18),
    "adjustment_step": int(1e-7 * 10**18),

5. fee_gamma vs out_fee
a takeaway from a.scan.png. There's some optimal fee_gamma that minimizes slippage, i.e. pool spends more time in balanced state. That is not very important since colorbar shows that difference is negligible.
b.scan: zooming in we see that optimal apy is for lower values of fee_gamma, and that also achieves average pool swap_fee between mid_fee and out_fee. So, out_fee 20bps and fee_gamma 1e-4 are good defaults.

6. we repeat A-donation_apy scan with latest allowed_extra_profit, adjustment_step, fee_gamma and out_fee. 
we confirm that optimal A is ~30 for donation_apy 5%.

Resulting pool parameters:
A= 320000 #32 * 10_000
mid_fee=10000000 # 10/10_000 * 10**10, 10bps
out_fee=20000000 # 20 / 10_000 * 10**10, 20bps
fee_gamma=1000000000000000 # 0.001 * 10**18
allowed_extra_profit=1000000 # 1e-12 * 10**18
adjustment_step=100000000000 # 1e-7 * 10**18
ma_time=5200 # 3600/math.log(2), 1h
donation_apy=0.05 # 5% yearly
