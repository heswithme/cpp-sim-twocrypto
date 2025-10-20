#!/usr/bin/env python3
"""
Analyze absolute candle length (max(OHLC) - min(OHLC)) for usdphp-1m.json

Outputs:
- Row count
- Mean and std (population) of absolute candle length
- Top 10 largest candles with UTC timestamps and OHLC values
 - Absolute top-10 smallest values across all O/H/L/C and top-10 largest (with timestamps and field)

Hardcoded input (no args):
- arb_sim/trade_data/usdphp/usdphp-1m.json
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
import heapq
from typing import List, Tuple


INPUT_JSON = Path(__file__).parent / "usdphp-1m.raw.json"


def candle_length(row: List[float]) -> float:
    # row = [ts, open, high, low, close, volume]
    o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
    return max(o, h, l, c) - min(o, h, l, c)


def main() -> None:
    with INPUT_JSON.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or not data:
        print("No data loaded or unexpected format")
        return

    # Compute absolute lengths aligned with data indices
    entries: List[Tuple[float, int]] = []  # (length, data_index)
    # Track absolute top-10 smallest and largest across all O/H/L/C
    # For smallest we keep a max-heap of size K (store (-value, ...))
    # For largest we keep a min-heap of size K (store (value, ...))
    K = 10
    smallest_heap: List[tuple] = []  # (-value, ts, field, o,h,l,c)
    largest_heap: List[tuple] = []   # (value, ts, field, o,h,l,c)
    for i, row in enumerate(data):
        try:
            L = max(0.0, candle_length(row))
            entries.append((L, i))
            ts = int(row[0])
            o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
            for field, val in (("O", o), ("H", h), ("L", l), ("C", c)):
                # smallest (max-heap via negative key)
                heapq.heappush(smallest_heap, (-val, ts, field, o, h, l, c))
                if len(smallest_heap) > K:
                    heapq.heappop(smallest_heap)  # pop largest negative -> removes smallest magnitude? actually removes current largest (least small)
                # largest (min-heap)
                if len(largest_heap) < K:
                    heapq.heappush(largest_heap, (val, ts, field, o, h, l, c))
                else:
                    if val > largest_heap[0][0]:
                        heapq.heapreplace(largest_heap, (val, ts, field, o, h, l, c))
        except Exception:
            continue

    n = len(entries)
    if n == 0:
        print("No valid candles found")
        return

    lengths_only = [L for (L, _) in entries]
    mu = float(mean(lengths_only))
    sigma = float(pstdev(lengths_only)) if n > 1 else 0.0
    print("=== Absolute Candle Length Stats ===")
    print(f"Rows: {n}")
    print(f"Mean: {mu:.6f}")
    print(f"Std : {sigma:.6f} (population)")

    # Top 10 largest candles
    # Keep tuples of (length, index)
    entries.sort(key=lambda t: t[0], reverse=True)
    top_k = entries[:10]

    print("\nTop 10 largest candles (UTC):")
    for rank, (L, idx) in enumerate(top_k, 1):
        row = data[idx]
        ts = int(row[0])
        ts_h = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
        print(f"#{rank:02d} {ts_h} | len={L:.6f} | O={o:.6f} H={h:.6f} L={l:.6f} C={c:.6f}")

    # Absolute top-10 smallest and largest O/H/L/C values across the dataset
    # Recover sorted lists from heaps
    smallest = sorted([(-val, ts, field, o, h, l, c) for (val, ts, field, o, h, l, c) in smallest_heap], key=lambda t: t[0])
    largest = sorted(largest_heap, key=lambda t: t[0], reverse=True)

    print("\nTop 10 smallest O/H/L/C values (UTC):")
    for rank, (val, ts, field, o, h, l, c) in enumerate(smallest[:K], 1):
        ts_h = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"#{rank:02d} {ts_h} | {field}={val:.6f} | O={o:.6f} H={h:.6f} L={l:.6f} C={c:.6f}")

    print("\nTop 10 largest O/H/L/C values (UTC):")
    for rank, (val, ts, field, o, h, l, c) in enumerate(largest[:K], 1):
        ts_h = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"#{rank:02d} {ts_h} | {field}={val:.6f} | O={o:.6f} H={h:.6f} L={l:.6f} C={c:.6f}")

    # ------------------------------------------------------------
    # Filter step 1: drop 3-sigma long candles (length > mean + 3*std)
    # ------------------------------------------------------------
    thr_len = mu + 3.0 * sigma
    drop_len = {idx for (L, idx) in entries if L > thr_len}
    print(f"\nLength filter: thr={thr_len:.6f} (mean+3σ) | remove={len(drop_len)}")

    # Build arrays for step 2 from remaining data
    keep_mask = [True] * len(data)
    for idx in drop_len:
        keep_mask[idx] = False
    kept_indices = [i for i, k in enumerate(keep_mask) if k]
    ts_all = [int(row[0]) for row in data]
    o_all = [float(row[1]) for row in data]
    h_all = [float(row[2]) for row in data]
    l_all = [float(row[3]) for row in data]
    c_all = [float(row[4]) for row in data]

    # ------------------------------------------------------------
    # Filter step 2: 1-day trailing MA baseline; remove if any of O/H/L/C deviates >10%
    # ------------------------------------------------------------
    window_seconds = 86400
    j = 0
    sum_close = 0.0
    removed_ma = set()
    # Maintain a small deque-like structure using indices
    window_idxs: List[int] = []
    for i_pos, idx in enumerate(kept_indices):
        # Add current
        # Evict old entries beyond 1 day
        ts_now = ts_all[idx]
        # Append current idx to window data structures
        window_idxs.append(idx)
        sum_close += c_all[idx]
        # Evict while too old
        while window_idxs and ts_now - ts_all[window_idxs[0]] > window_seconds:
            old_idx = window_idxs.pop(0)
            sum_close -= c_all[old_idx]
        # Compute MA
        win_len = len(window_idxs)
        if win_len <= 0:
            continue
        baseline = sum_close / win_len
        if baseline <= 0:
            continue
        # Check deviations
        max_dev = max(
            abs(o_all[idx] - baseline) / baseline,
            abs(h_all[idx] - baseline) / baseline,
            abs(l_all[idx] - baseline) / baseline,
            abs(c_all[idx] - baseline) / baseline,
        )
        if max_dev > 0.025:
            removed_ma.add(idx)

    print(f"1d-MA filter: remove={len(removed_ma)} (>|2.5%| from trailing 1d average)")

    # ------------------------------------------------------------
    # Filter step 3: drop whole days where the day's first Open
    # differs by more than 5% from the previous day's last Close.
    # ------------------------------------------------------------
    day_to_indices: dict[str, list[int]] = {}
    for idx in kept_indices:
        d = datetime.fromtimestamp(ts_all[idx], tz=timezone.utc).strftime('%Y-%m-%d')
        day_to_indices.setdefault(d, []).append(idx)

    # Iterate kept indices in time order to detect day boundaries
    removed_day_indices: set[int] = set()
    removed_day_names: set[str] = set()
    prev_day_name: str | None = None
    prev_day_last_close: float | None = None
    seen_day_first_idx: set[str] = set()
    for idx in kept_indices:
        d = datetime.fromtimestamp(ts_all[idx], tz=timezone.utc).strftime('%Y-%m-%d')
        if d not in seen_day_first_idx:
            # First candle of day d
            if prev_day_last_close is not None and prev_day_last_close > 0:
                o_now = o_all[idx]
                dev = abs(o_now - prev_day_last_close) / prev_day_last_close
                if dev > 0.05:
                    # Mark entire day for removal
                    removed_day_names.add(d)
                    removed_day_indices.update(day_to_indices.get(d, []))
            seen_day_first_idx.add(d)
        # Update previous day's last close regardless (carries forward latest close)
        prev_day_last_close = c_all[idx]
        prev_day_name = d

    print(f"Prev-close day filter: remove_days={len(removed_day_names)} remove_rows={len(removed_day_indices)} (>|5%| O vs prev C)")

    # Compose final filtered dataset
    drop_all = drop_len.union(removed_ma).union(removed_day_indices)
    filtered = [row for i, row in enumerate(data) if i not in drop_all]
    out_path = INPUT_JSON.with_name("usdphp-1m.filtered.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(filtered, fh)
    print(f"Wrote filtered dataset: {out_path} ({len(filtered)} rows, removed {len(drop_all)})")


if __name__ == "__main__":
    main()
