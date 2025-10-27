#!/usr/bin/env python3
"""
Convert OHLC CSV (Datetime,Open,High,Low,Close[,Volume]) to array-of-arrays JSON:

  [ [ts, open, high, low, close, volume], ... ]

Notes
- Timestamps are parsed as UTC and written as UNIX seconds (int).
- If no Volume column is present, a default is used (10e6).

Hardcoded paths (no CLI args):
- Input:  arb_sim/trade_data/usdngn/usdngn.csv
- Output: arb_sim/trade_data/usdngn/usdngn-1m.json
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional


def parse_dt_utc(value: str) -> int:
    s = value.strip()
    # Accept ISO-like "YYYY-MM-DD HH:MM:SS" and variants; assume UTC
    # Try common formats
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            pass
    # Fallback: attempt fromisoformat (may ignore timezone); force UTC
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unrecognized datetime format: {value}") from exc


def convert_csv(
    in_csv: Path,
    out_json: Path,
    default_volume: float = 10_000_000.0,
) -> int:
    required = {"datetime", "open", "high", "low", "close"}
    rows_out: List[List[float]] = []

    with in_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")
        # Normalize headers to lowercase for matching
        headers_lc = [h.strip().lower() for h in reader.fieldnames]
        # Build a mapping original->lower for lookup (not used further)
        # name_map = {orig: orig.strip().lower() for orig in reader.fieldnames}

        missing = sorted(required - set(headers_lc))
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}; found {reader.fieldnames}")

        has_volume = any(h in {"volume", "vol", "v"} for h in headers_lc)
        vol_key: Optional[str] = None
        if has_volume:
            for orig in reader.fieldnames:
                if orig is None:
                    continue
                key = orig.strip().lower()
                if key in {"volume", "vol", "v"}:
                    vol_key = orig
                    break

        # Resolve canonical keys as they appear in file (preserve original case for DictReader)
        def find_key(want: str) -> str:
            for orig in reader.fieldnames or []:
                if orig is None:
                    continue
                if orig.strip().lower() == want:
                    return orig
            raise KeyError(want)

        k_dt = find_key("datetime")
        k_o = find_key("open")
        k_h = find_key("high")
        k_l = find_key("low")
        k_c = find_key("close")

        for row in reader:
            if row is None:
                continue
            try:
                ts = parse_dt_utc(str(row[k_dt]))
                o = float(row[k_o])
                h = float(row[k_h])
                l = float(row[k_l])
                c = float(row[k_c])
                if vol_key is not None and row.get(vol_key) not in (None, ""):
                    v = float(row[vol_key])
                else:
                    v = float(default_volume)
                rows_out.append([int(ts), float(o), float(h), float(l), float(c), float(v)])
            except Exception:
                # Skip malformed lines
                continue

    # Ensure chronological order and deduplicate identical timestamps (keep first occurrence)
    rows_out.sort(key=lambda r: r[0])
    deduped: List[List[float]] = []
    last_ts: Optional[int] = None
    for r in rows_out:
        ts_i = int(r[0])
        if last_ts is not None and ts_i == last_ts:
            continue
        deduped.append(r)
        last_ts = ts_i

    # No outlier filtering; emit data as-is (after dedup ordering)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(deduped, fh)
    return len(deduped)


DEFAULT_VOLUME = 10_000_000.0
INPUT_CSV = Path(__file__).parent / "usdngn.csv"
OUTPUT_JSON = Path(__file__).parent / "usdngn-1m.raw.json"


def main() -> None:
    count = convert_csv(INPUT_CSV, OUTPUT_JSON, DEFAULT_VOLUME)
    print(f"Wrote {count} rows to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
