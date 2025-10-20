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
# x-axis uses UTC dates via matplotlib date converters
import numpy as np
from matplotlib import dates as mdates


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


def print_stats(ts: List[int], o: List[float], h: List[float], l: List[float], c: List[float], v: 'Optional[List[float]]' = None, k_examples: int = 5):
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

    # Gap analysis (>10 minutes between consecutive timestamps)
    if n >= 2:
        deltas_s = np.diff(ts_arr)
        gap_mask = deltas_s > 600  # strictly greater than 10 minutes
        gap_sizes_min = deltas_s[gap_mask] / 60.0
        gap_idx = np.nonzero(gap_mask)[0]  # index i means gap between i and i+1
        gap_count = int(gap_sizes_min.size)
        print("\n--- Gap Analysis (>10 min) ---")
        print(f"Gaps: count={gap_count}")
        if gap_count > 0:
            g_mean = float(np.mean(gap_sizes_min))
            g_std = float(np.std(gap_sizes_min, ddof=0))
            g_max = float(np.max(gap_sizes_min))
            print(f"Gap size (min): mean={g_mean:.2f} std={g_std:.2f} max={g_max:.2f}")
            # Top 5 largest gaps with their time positions
            order = np.argsort(-gap_sizes_min)  # descending
            top_k = min(5, gap_count)
            print("Top gaps:")
            for rank in range(top_k):
                gi = int(order[rank])
                i = int(gap_idx[gi])
                start_ts = float(ts_arr[i])
                end_ts = float(ts_arr[i + 1])
                start_h = datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S UTC')
                end_h = datetime.utcfromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M:%S UTC')
                size_min = float(gap_sizes_min[gi])
                print(f"  #{rank+1}: {start_h} -> {end_h} | +{size_min:.2f} min")

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

    # X axis as matplotlib date numbers (days since 0001-01-01)
    x_sec = np.asarray(ts, dtype=np.float64)
    epoch_days = mdates.date2num(datetime(1970, 1, 1))
    x = x_sec / 86400.0 + epoch_days  # convert POSIX seconds to matplotlib date units (days)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Common width based on original seconds, then convert to days
    if n > 1:
        width_sec = (x_sec[1] - x_sec[0]) * 0.6
    else:
        width_sec = 60.0  # default ~1 minute if only one point
    width_days = float(width_sec) / 86400.0

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
            rect = Rectangle((xi - width_days / 2, y0), width_days, height,
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

    # Formatting
    ax.grid(True, linestyle=':', alpha=0.3)
    # Date formatting on x-axis (UTC) — uniform 30 labels from start to end
    formatter = mdates.DateFormatter('%d-%m-%Y %H:%M', tz=timezone.utc)
    ax.xaxis.set_major_formatter(formatter)
    if n > 1:
        xticks = np.linspace(float(x[0]), float(x[-1]), 30)
    else:
        xticks = [float(x[0])]
    ax.set_xticks(xticks)
    ax.set_title(title)

    # Auto-rotate date labels for readability
    fig.autofmt_xdate()

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"Saved plot to {save}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot OHLCV JSON as candles/line")
    default_file = Path(__file__).parent / "trade_data" / "brlusd" / "brlusd-1m.json"
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
