1. Asset issues provided desired data:
Arb fee 10bps
desired swap fee 5 bps

We set wide scan A:1-200, don_rate 1-10%. mid_fee=out_fee=5bps.

1. A-don_rate_wide.png
We see that we can have A from 5 without donations to A ~50 with donations of 10%. Let's zoom in and allow out_fee = 10bps.

2. A-don_rate_zoomed.png
For donation rate of 5% highest possible A is ~15, due to very low fees.

3. We fix donation rate to 5% and see if increasing MA will help. It seems that bumping MA to ~4hrs allows to safely have A ~20.

Final pool parameters:
A = 200000
mid_fee = 5bps
out_fee = 10bps
fee_gamma = 0.0001
allowed_extra_profit = 1e-12
adjustment_step = 1e-7
ma_time = 20775 # 3600*4/math.log(2)
donation_apy = 0.05
