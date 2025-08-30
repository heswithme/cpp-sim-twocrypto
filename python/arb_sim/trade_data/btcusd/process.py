#!/usr/bin/env python3
# Convert raw_fetched.json into compact events: [[ts, price, volume], ...]
# - ts: UNIX seconds from ISO8601 'time'
# - price: raw 'price' from API
# - volume: reference token amount (USDT here)

import json
import os
import sys
from datetime import datetime, timezone


def load_rows(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    return obj.get('data', [])


def to_ts(s):
    if not s:
        return None
    # Accept '...Z' and naive timestamps (assume UTC)
    try:
        if s.endswith('Z'):
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def main():
    folder = os.path.dirname(__file__)
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(folder, 'raw_fetched.json')
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(folder, 'btcusd-events.json')

    try:
        rows = load_rows(src)
    except Exception as e:
        print(f"failed to load {src}: {e}")
        return 2

    events = []
    for r in rows:
        ts = to_ts(r.get('time'))
        price = 1/r.get('price')
        if ts is None or price is None:
            continue

        # Reference token is USDT at id 0 in this dataset
        vol = None
        sid = r.get('sold_id')
        bid = r.get('bought_id')
        if sid == 0:
            vol = r.get('tokens_sold')
        elif bid == 0:
            vol = r.get('tokens_bought')
        else:
            # Fallbacks (rare): try USD volume, then tokens_sold
            vol = r.get('tokens_sold_usd') or r.get('tokens_bought_usd') or r.get('tokens_sold')

        try:
            p = float(price)
            v = float(vol) if vol is not None else 0.0
        except Exception:
            continue

        events.append([ts, p, v])

    events.sort(key=lambda x: x[0])

    with open(dst, 'w') as f:
        json.dump(events, f)
    print(f"wrote {len(events)} events -> {dst}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

