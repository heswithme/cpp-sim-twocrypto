#!/usr/bin/env python3
"""
Synthetic 1m OHLCV generator (scripty and minimal).

Pattern:
- Start: 2025-01-01T00:00:00Z
- Price: start at 100_000, go up +1_000 per minute for 100 minutes (reaches 200_000),
         then descend back to 100_000 over 100 minutes.
- Volume: fixed 1_000_000 per candle.

Output: python/arb_sim/trade_data/synth-1m.json
Row schema: [ts, O, H, L, C, volume] with O==H==L==C.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


# Hardcoded params (tweak here if needed)
START_ISO = "2025-01-01T10:00:00Z"
START_PRICE = 100_000.0
STEP = 1000.0
UP_MINUTES = 1000
DOWN_MINUTES = 1
INTERVAL_S = 200
VOLUME = 1_000_000.0
OUT_PATH = Path(__file__).resolve().parent / "trade_data" / "btcusd" / "synth-1m.json"


def ts_from_iso(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def main() -> int:
    t0 = ts_from_iso(START_ISO)
    rows = []

    # Up
    for i in range(0, UP_MINUTES + 1):
        ts = t0 + (i - 1) * INTERVAL_S
        price = START_PRICE + STEP * (i)
        rows.append([ts, price, price, price, price, VOLUME])

    # Down
    for j in range(UP_MINUTES - 1, -1, -1):
        ts = t0 + (UP_MINUTES + (UP_MINUTES - 1 - j)) * INTERVAL_S
        price = START_PRICE + STEP * j
        rows.append([ts, price, price, price, price, VOLUME])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} candles -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
