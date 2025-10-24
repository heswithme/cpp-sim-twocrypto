Asset issuer required a pool with A=30, and 5bps swap fee. Let's scan for optimals of fee_gamma and ma_time. Arb fee is 50 bps, and donation is ~5%.

1. A-mid_fee_wide.png: Birdview of A-mid_fee, actually A=30 and 5bps doesn't look bad!

2. A-don_rate.png
We zoom in and see how pool performs with different donation rates. It seems that pool performance isn't great, and A is simply too high for such low fees. 

3. A-ma_time.png
The only sane way to keep pool from getting stuck on some volatile price action is to bump MA. 1 day seems to work fine for higher A values, providing lower slippage and better apy.

4. A-fee_gamma.png
To bump avg fee a bit closer to desired 5bps fee, we also scan fee_gamma. 0.005 seems to bring average pool fee closer to 5bps.

Final pool parameters:
A = 300000
mid_fee = 5bps
out_fee = 10bps
fee_gamma = 0.005
allowed_extra_profit = 1e-12
adjustment_step = 1e-7
ma_time = 124650 # 3600*24/math.log(2)

donation_apy = 0.05
arb_fee_bps = 50