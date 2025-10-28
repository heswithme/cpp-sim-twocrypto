Another look at brl/usd, this time with no imposed mid_fee and out_fee.
Donation rate can be up to ~13%, and arb_fee is conservative 50 bps (can be as low as 30!)

1. A-mid_fee5don.png: Birdview of A-mid_fee, at ma=866 don_rate 5%. Seems that A can't go higher than 50. For fees - best APY for LPs is with higher fees (100bps), but that obviously significntly affects real slippage.

2. A-mid_fee10don.png: With increased donation rate to 10%, A can get higher (up to ~30). Since asset issuer can afford that, we fix this pool to 10% donation rate.

Asset issuer wants to compete with another pool that has very high liq concentration, so we need to investigate if fee structures liek 1-20bps have a chance. Fixing A=20 we see

3. mid_fee-out_fee.png: 
Seems that pool can work well with 10-100bps fee structure. 


4. fee_gamma-out_fee.png:
Must find balance between APY, slippage and avg_fee.
Seems that mid_fee=5bps, out_fee=50bps, and fee_gamma = 0.001 provide good balance. Now fixing these params and find best A and ma_time.

5. A-ma_time.png:
A=25, ma_time=12h looks good.

Final pool parameters:
A = 250000
mid_fee = 5bps
out_fee = 50bps
fee_gamma = 0.001
allowed_extra_profit = 1e-12
adjustment_step = 1e-7
ma_time = 62_000 # 3600*12/math.log(2)

donation_apy = 0.10
arb_fee_bps = 50