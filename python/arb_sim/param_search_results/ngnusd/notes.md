Issuer info:
cNGN: stablecoin <> USDC cost 15bps, desired swap fee in the pool will be 0.05%
So arb_fee 15bps, and swap-fee 5-10 bps (mid-out).
Donation rate aim 5% (to standardize).

Data is from ngnusd-1m.json - 2023-2025 flipped and filtered:
uv run python/arb_sim/trade_data/usdngn/csv_to_json.py
uv run python/arb_sim/plot_candles.py --file python/arb_sim/trade_data/usdngn/usdngn-1m.raw.json
uv run python/arb_sim/trade_data/process_series.py python/arb_sim/trade_data/usdngn/usdngn-1m.raw.json --cut --start 01082023 --end 01012025
uv run python/arb_sim/trade_data/process_series.py python/arb_sim/trade_data/usdngn/usdngn-1m.raw.cut.json --filter
uv run python/arb_sim/trade_data/process_series.py python/arb_sim/trade_data/usdngn/usdngn-1m.raw.cut.filtered.json --flip

1. A-mid_fee_wide.png
We see that high A values are not very good fir this data, as price can have large jumps. Lower A ensure better price tracking and slippage.

2. A-don_rate_zoom.png
With fees constraned at 5-10 bps we see that pool struggles to track price at A>5. For lower donation rates (i.e. 5%), threshold A is around 3.

3. A-ma_time.png
Lower values of ma_time decrease pool performance, so it's recommended to remain at ma_time=1day.

Final pool parameters:
A = 30000
mid_fee = 5bps
out_fee = 10bps
fee_gamma = 0.001
allowed_extra_profit = 1e-12
adjustment_step = 1e-7
ma_time = 124650 # 3600*24/math.log(2)

donation_apy = 0.05
arb_fee_bps = 15