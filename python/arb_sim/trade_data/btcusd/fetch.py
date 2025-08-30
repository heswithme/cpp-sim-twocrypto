#!/usr/bin/env python3
# Minimal concurrent fetcher for Curve prices API
# - Finds the last page via exponential + binary search
# - Shows simple progress (% of pages fetched)
# - Saves intermediate results and can resume
# - Outputs raw_fetched.json (flat list) for later OHLC processing

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen

# Token addresses (Ethereum mainnet)
wbtc = '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'
usdt = '0xdac17f958d2ee523a2206206994597c13d831ec7'
pool = '0xD51a44d3FaE010294C616388b506AcdA1bfAAE46'  # tricrypto2

per_page = 100  # API max
chain = 'ethereum'


def make_url(page: int) -> str:
    return (
        f"https://prices.curve.finance/v1/trades/{chain}/{pool}"
        f"?main_token={wbtc}&reference_token={usdt}&page={page}&per_page={per_page}"
    )


def fetch_page(page: int):
    url = make_url(page)
    with urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode('utf-8'))


def page_rows(resp):
    if isinstance(resp, dict):
        return resp.get('data', [])
    if isinstance(resp, list):
        return resp
    return []


def find_last_page():
    cache = {}

    def has_data(p):
        if p in cache:
            rows = cache[p]
        else:
            try:
                resp = fetch_page(p)
            except Exception as e:
                print(f"search: page {p} error: {e}", file=sys.stderr)
                cache[p] = []
                return False
            rows = page_rows(resp)
            cache[p] = rows
        print(f"search: page {p} -> {len(rows)} rows", file=sys.stderr)
        return len(rows) > 0

    if not has_data(1):
        return 0, {}, None

    lo = 1
    hi = 2
    while has_data(hi):
        lo = hi
        hi *= 2
        if hi > 1_000_000:
            break

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if has_data(mid):
            lo = mid
        else:
            hi = mid

    last = lo
    last_rows = cache.get(last, [])
    earliest = None
    if last_rows:
        try:
            earliest = min((r.get('time') for r in last_rows if 'time' in r), default=None)
        except Exception:
            earliest = None

    return last, cache, earliest


def load_existing(path):
    if not os.path.exists(path):
        return set(), [], {}
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
    except Exception:
        return set(), [], {}

    if isinstance(obj, list):
        return set(), obj, {}

    pages = set(obj.get('pages', []))
    data = obj.get('data', [])
    meta = obj.get('meta', {})
    try:
        pages = set(int(p) for p in pages)
    except Exception:
        pages = set()
    return pages, data if isinstance(data, list) else [], meta if isinstance(meta, dict) else {}


def save_intermediate(path, rows, pages_done, meta):
    key = 'time'
    if rows and key not in rows[0]:
        key = 'timestamp' if 'timestamp' in rows[0] else 'block_number'
    try:
        rows_sorted = sorted(rows, key=lambda r: r.get(key, 0))
    except Exception:
        rows_sorted = rows

    obj = {
        'meta': meta,
        'pages': sorted(pages_done),
        'data': rows_sorted,
    }
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    # Output defaults to raw_fetched.json in this folder
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'raw_fetched.json')
    workers = int(os.environ.get('CONCURRENCY', '8'))
    save_every = int(os.environ.get('SAVE_EVERY', '25'))

    # Resume if present
    pages_done, rows, meta = load_existing(out)
    pages_done = set(pages_done)

    print("Finding last page via binary search...", file=sys.stderr)
    last_page, cached_pages, earliest_time = find_last_page()
    if last_page == 0:
        print("No data found.", file=sys.stderr)
        save_intermediate(out, rows, pages_done, {
            'chain': chain, 'pool': pool, 'main_token': wbtc, 'reference_token': usdt,
            'per_page': per_page, 'last_page': 0, 'earliest_time': None,
        })
        print(f"wrote {len(rows)} rows -> {out}")
        return

    if earliest_time:
        print(f"Last page: {last_page}, earliest time: {earliest_time}", file=sys.stderr)
    else:
        print(f"Last page: {last_page}", file=sys.stderr)

    # Seed with pages seen during search
    for p, data in cached_pages.items():
        if data and p not in pages_done:
            pages_done.add(p)
            rows.extend(data)

    total_pages = last_page
    remaining = [p for p in range(1, total_pages + 1) if p not in pages_done]
    completed = len(pages_done)
    print(f"Fetching {len(remaining)} remaining pages out of {total_pages}...", file=sys.stderr)

    meta = {
        'chain': chain,
        'pool': pool,
        'main_token': wbtc,
        'reference_token': usdt,
        'per_page': per_page,
        'last_page': last_page,
        'earliest_time': earliest_time,
    }

    # Fetch remaining pages concurrently
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Prime up to workers
        futs = {}
        idx = 0
        while idx < len(remaining) and len(futs) < workers:
            p = remaining[idx]
            futs[ex.submit(fetch_page, p)] = p
            idx += 1

        while futs:
            for fut in as_completed(list(futs.keys())):
                p = futs.pop(fut)
                try:
                    resp = fut.result()
                except Exception as e:
                    print(f"page {p} error: {e}", file=sys.stderr)
                    resp = {"data": []}
                data = page_rows(resp)
                pages_done.add(p)
                rows.extend(data)
                completed += 1

                # Progress
                pct = (completed / total_pages) * 100.0
                print(f"progress: {completed}/{total_pages} ({pct:.1f}%)", file=sys.stderr)

                # Schedule next pending page
                if idx < len(remaining):
                    np = remaining[idx]
                    futs[ex.submit(fetch_page, np)] = np
                    idx += 1

                # Periodic save
                if completed % save_every == 0 or completed == total_pages:
                    save_intermediate(out, rows, pages_done, meta)

    # Final save
    save_intermediate(out, rows, pages_done, meta)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == '__main__':
    main()
