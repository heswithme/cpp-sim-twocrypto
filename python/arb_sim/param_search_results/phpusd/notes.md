We simulate phpusd pair with candlestick data from DEFAULT_DATAFILE = "python/arb_sim/trade_data/usdphp/phpusd-1m.json"

Donation rate is set to 0.05, and arb_fee is set to 15bps.

1. A-mid_fee and _flipped (initially we used wrong dataset but can see that it doesn't matter at all):
We see how pool works really well on high A (100+) and high fee (up to 100bps) in terms of profitability, but price tracking and slippage aren't best in that case.

2. A-don_rate.png
For ~5% donation rate, with mid_fee and out_fee set to 5bps and 10bps respectively, we see that pool can perform well even with higher values of A, like A~50.

3. A-ma_time.png
Last scan to find if pool goes better with lower MA (prev used 1 day).
Seems that higher ma_time allows for slightly higher A, so we keep it at 1 day.


Final pool parameters:
A = 500000
mid_fee = 5bps
out_fee = 10bps
fee_gamma = 0.001
allowed_extra_profit = 1e-12
adjustment_step = 1e-7
ma_time = 124650 # 3600*24/math.log(2)

donation_apy = 0.05
arb_fee_bps = 15