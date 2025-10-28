arb_fee 75 bps, donation_apy 5%.

1. A-mid_fee.png: seems to ge well with higher A and low fees.

2. A-ma_time.png
A=50, ma_time = 12h looks good.

Final pool parameters:
A = 500000
mid_fee = 5bps
out_fee = 50bps
fee_gamma = 0.001
allowed_extra_profit = 1e-12
adjustment_step = 1e-7
ma_time = 62_000 # 3600*12/math.log(2)

donation_apy = 0.05
arb_fee_bps = 75