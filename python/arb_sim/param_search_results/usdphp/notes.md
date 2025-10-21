We simulate phpusd pair with candlestick data from DEFAULT_DATAFILE = "python/arb_sim/trade_data/usdphp/usdphp-1m.flipped.json"

Donation rate is set to 0.05, and arb_fee is set to 10bps.

1. A-mid_fee and _flipped (initially we used wrong dataset but can see that it doesn't matter at all):
We see how pool works really well on high A (100+) and high fee (up to 100bps) in terms of profitability, but price tracking and slippage aren't best in that case.
Lower A(~40) and fee (~20bps) provide better slippages with comparable APYs.
2. Zooming in we confirm that values A~40 and mid_fee 20, out_fee 40bps provide a compromise btween low slippage and higher apy. Asset issuer's data on arb_fee and donation rate is required.