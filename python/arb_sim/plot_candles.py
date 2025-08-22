#!/usr/bin/env python3
"""
Plot OHLCV JSON (array-of-arrays) using matplotlib.

Input format (single line JSON, large files supported):
  [ [timestamp, open, high, low, close, volume], ... ]  # OHLCV order

Defaults to python/backtest_pool/data/brlusd/brlusd-1m.json

Usage examples:
  uv run python/backtest_pool/plot_candles.py
  uv run python/backtest_pool/plot_candles.py --file path/to/data.json --start 2023-01-01 --end 2023-02-01
  uv run python/backtest_pool/plot_candles.py --candles --max-candles 20000 --save brlusd.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional
import random

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
# x-axis uses raw UNIX timestamps; no date conversion/formatting
import numpy as np


def _parse_ts(v: int | float | str) -> int:
    try:
        return int(v)
    except Exception:
        return int(float(v))


def _parse_date(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    # Accept unix seconds
    if s.isdigit():
        return int(s)
    # Accept common date formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            pass
    raise ValueError(f"Unrecognized date/time format: {s}")


def load_ohlcv(path: Path, t0: Optional[int], t1: Optional[int]) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
    with path.open("r") as f:
        data = json.load(f)
    ts: List[int] = []
    o: List[float] = []
    h: List[float] = []
    l: List[float] = []
    c: List[float] = []
    v: List[float] = []
    for row in data:
        # Expect [ts, open, high, low, close, volume]
        t = _parse_ts(row[0])
        if (t0 is not None and t < t0) or (t1 is not None and t > t1):
            continue
        ts.append(t)
        o.append(float(row[1]))
        h.append(float(row[2]))
        l.append(float(row[3]))
        c.append(float(row[4]))
        v.append(float(row[5]))
    # Ensure chronological order
    if ts and any(ts[i] > ts[i+1] for i in range(len(ts)-1)):
        order = sorted(range(len(ts)), key=lambda i: ts[i])
        ts = [ts[i] for i in order]
        o  = [o[i]  for i in order]
        h  = [h[i]  for i in order]
        l  = [l[i]  for i in order]
        c  = [c[i]  for i in order]
        v  = [v[i]  for i in order]
    return ts, o, h, l, c, v


def print_stats(ts: List[int], o: List[float], h: List[float], l: List[float], c: List[float], v: List[float] | None = None, k_examples: int = 5):
    n = len(ts)
    if n == 0:
        print("No data loaded; skipping stats")
        return
    ts_arr = np.asarray(ts, dtype=np.float64)
    o_arr = np.asarray(o, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    l_arr = np.asarray(l, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64) if v is not None else None

    start_dt = datetime.utcfromtimestamp(float(ts_arr[0])).strftime('%Y-%m-%d %H:%M:%S UTC')
    end_dt   = datetime.utcfromtimestamp(float(ts_arr[-1])).strftime('%Y-%m-%d %H:%M:%S UTC')

    def m_s(arr: np.ndarray) -> Tuple[float, float]:
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    om, os = m_s(o_arr)
    hm, hs = m_s(h_arr)
    lm, ls = m_s(l_arr)
    cm, cs = m_s(c_arr)
    if v_arr is not None:
        vm, vs = m_s(v_arr)

    print("\n=== Stats ===")
    print(f"Points: {n}")
    print(f"Start : {start_dt}")
    print(f"End   : {end_dt}")
    print(f"Open  : mean={om:.6f} std={os:.6f}")
    print(f"High  : mean={hm:.6f} std={hs:.6f}")
    print(f"Low   : mean={lm:.6f} std={ls:.6f}")
    print(f"Close : mean={cm:.6f} std={cs:.6f}")
    if v_arr is not None:
        print(f"Volume: mean={vm:.6f} std={vs:.6f}")

    # Random examples per parameter
    k = min(k_examples, n)
    idxs = random.sample(range(n), k)
    def fmt_pair(i: int, val: float) -> str:
        ts_h = datetime.utcfromtimestamp(float(ts_arr[i])).strftime('%Y-%m-%d %H:%M:%S')
        return f"({ts_h}, {val:.6f})"
    print("\nExamples (UTC time, value):")
    print("  Open  : [" + ", ".join(fmt_pair(i, o_arr[i]) for i in idxs) + "]")
    print("  High  : [" + ", ".join(fmt_pair(i, h_arr[i]) for i in idxs) + "]")
    print("  Low   : [" + ", ".join(fmt_pair(i, l_arr[i]) for i in idxs) + "]")
    print("  Close : [" + ", ".join(fmt_pair(i, c_arr[i]) for i in idxs) + "]")
    if v_arr is not None:
        print("  Volume: [" + ", ".join(fmt_pair(i, float(v_arr[i])) for i in idxs) + "]")


def plot_candles(ts: List[int], o: List[float], h: List[float], l: List[float], c: List[float], v: List[float],
                 use_candles: bool, max_candles: int, stride: int,
                 title: str, save: Optional[Path]):
    n = len(ts)
    if n == 0:
        print("No data in selected window")
        return

    # Determine stride
    if stride <= 0:
        stride = 1
    if use_candles and max_candles > 0 and n // stride > max_candles:
        stride = max(1, n // max_candles)

    # Downsample
    if stride > 1:
        ts = ts[::stride]; o = o[::stride]; h = h[::stride]; l = l[::stride]; c = c[::stride]; v = v[::stride]
        n = len(ts)

    # X as raw UNIX timestamps (seconds)
    x = [float(t) for t in ts]

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(2, 1, 1)
    ax_vol = fig.add_subplot(2, 1, 2, sharex=ax)

    # Common x width (in seconds) for bars
    width = (x[1] - x[0]) * 0.6 if n > 1 else 1.0

    if use_candles:
        # Wicks via LineCollection (vectorized)
        segs_up = []
        segs_dn = []
        bodies: List[Rectangle] = []
        for xi, oi, hi, li, ci in zip(x, o, h, l, c):
            up = ci >= oi
            seg = [(xi, li), (xi, hi)]
            (segs_up if up else segs_dn).append(seg)
            y0 = min(oi, ci)
            height = abs(ci - oi)
            if height == 0:
                height = 1e-12
            rect = Rectangle((xi - width / 2, y0), width, height,
                             facecolor=(0.1, 0.7, 0.1, 0.8) if up else (0.8, 0.2, 0.2, 0.8),
                             edgecolor='black', linewidth=0.2)
            bodies.append(rect)
        if segs_up:
            lc_up = LineCollection(segs_up, colors=(0.1, 0.7, 0.1, 0.8), linewidths=0.5)
            ax.add_collection(lc_up)
        if segs_dn:
            lc_dn = LineCollection(segs_dn, colors=(0.8, 0.2, 0.2, 0.8), linewidths=0.5)
            ax.add_collection(lc_dn)
        for r in bodies:
            ax.add_patch(r)
        ax.set_ylabel("Price")
    else:
        ax.plot(x, c, color='tab:blue', linewidth=0.8)
        ax.set_ylabel("Close")

    # Volume (bar chart, color by up/down)
    colors = [(0.1, 0.7, 0.1, 0.5) if ci >= oi else (0.8, 0.2, 0.2, 0.5) for oi, ci in zip(o, c)]
    ax_vol.bar(x, v, width=width, color=colors, align='center', linewidth=0)
    ax_vol.set_ylabel("Volume")

    # Formatting
    ax.grid(True, linestyle=':', alpha=0.3)
    ax_vol.grid(True, linestyle=':', alpha=0.3)
    # Keep numeric timestamp axis; no date formatting
    ax.set_title(title)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"Saved plot to {save}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot OHLCV JSON as candles/line")
    default_file = Path(__file__).parent / "data" / "brlusd" / "brlusd-1m.json"
    ap.add_argument("--file", type=Path, default=default_file, help="Path to JSON file")
    ap.add_argument("--start", type=str, default=None, help="Start (YYYY-MM-DD or unix seconds)")
    ap.add_argument("--end", type=str, default=None, help="End (YYYY-MM-DD or unix seconds)")
    ap.add_argument("--candles", action="store_true", help="Use candlesticks (default: line if too many points)")
    ap.add_argument("--max-candles", type=int, default=20000, help="Max candles to render (auto stride)")
    ap.add_argument("--stride", type=int, default=100, help="Plot every Nth point (overrides auto stride if >1)")
    ap.add_argument("--save", type=Path, default=None, help="Save to file instead of showing GUI")
    args = ap.parse_args()

    t0 = _parse_date(args.start)
    t1 = _parse_date(args.end)
    ts, o, h, l, c, v = load_ohlcv(args.file, t0, t1)
    # Print summary stats upon successful load
    print_stats(ts, o, h, l, c, v)

    title = f"{args.file.name} ({len(ts)} points)"
    use_candles = bool(args.candles)
    # Heuristic: if too many points and not forced candles -> use line
    if not use_candles and len(ts) <= args.max_candles:
        use_candles = True

    plot_candles(ts, o, h, l, c, v, use_candles, args.max_candles, args.stride, title, args.save)


if __name__ == "__main__":
    main()
