#!/usr/bin/env python3
"""
Flip prices in usdphp-1m.json by inverting OHLC: x -> 1/x

Input (hardcoded):  arb_sim/trade_data/usdphp/usdphp-1m.json
Output:             arb_sim/trade_data/usdphp/usdphp-1m.flipped.json

Notes
- Volume is preserved as-is.
- Candles with non-positive O/H/L/C are skipped (counted and reported).
- High/Low are mapped as: H' = 1/min(OHLC), L' = 1/max(OHLC) which is
  equivalent to H' = 1/Low, L' = 1/High for positive inputs.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List


INPUT = Path(__file__).parent / "usdphp-1m.filtered.json"
OUTPUT = Path(__file__).parent / "usdphp-1m.flipped.json"


def invert_row(row: List[float]) -> List[float] | None:
    # row = [ts, open, high, low, close, volume]
    ts = int(row[0])
    o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
    v = float(row[5]) if len(row) > 5 else 0.0
    if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
        return None

    o_inv = 1.0 / o
    c_inv = 1.0 / c
    # For strictly positive inputs, 1/x reverses order: high' = 1/low, low' = 1/high
    h_inv = 1.0 / min(o, h, l, c)  # equals 1/low among OHLC
    l_inv = 1.0 / max(o, h, l, c)  # equals 1/high among OHLC
    # Ensure invariants explicitly
    if h_inv < max(o_inv, c_inv):
        h_inv = max(o_inv, c_inv)
    if l_inv > min(o_inv, c_inv):
        l_inv = min(o_inv, c_inv)
    return [ts, o_inv, h_inv, l_inv, c_inv, v]


def main() -> None:
    with INPUT.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Unexpected JSON structure; expected list of rows")

    out: List[List[float]] = []
    skipped = 0
    for row in data:
        try:
            r = invert_row(row)
            if r is None:
                skipped += 1
                continue
            out.append(r)
        except Exception:
            skipped += 1
            continue

    with OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"Wrote {len(out)} rows to {OUTPUT} (skipped {skipped})")


if __name__ == "__main__":
    main()

