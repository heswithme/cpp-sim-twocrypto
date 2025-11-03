ETHUSD minute candles (Binance)

- Script: `python/arb_sim/trade_data/ethusd/fetch_binance.py`
- Default pair: `ETHUSDT`
- Interval: 1 minute
- Default range: 2023-01-01 through end of 2025 (clipped to now)
- Output: `python/arb_sim/trade_data/ethusd/ethusdt-1m.json`

Run:

```
uv run python/arb_sim/trade_data/ethusd/fetch_binance.py
```

Optionally override the range by editing `START_YEAR`/`END_YEAR` in the script
or by setting both `START_OVERRIDE` and `END_OVERRIDE` to ISO8601 timestamps
like `"2024-01-01T00:00:00Z"`.

