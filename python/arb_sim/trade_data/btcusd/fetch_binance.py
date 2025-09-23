#!/usr/bin/env python3.11

"""Fetch Binance 1m candlesticks for a configurable multi-year range."""

import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration (edit these values if you need a different dataset)
# ---------------------------------------------------------------------------
PAIR = "BTCUSDT"  # Binance trading pair symbol
START_YEAR = 2020  # First calendar year (inclusive)
END_YEAR = 2024  # Last calendar year (inclusive)
START_OVERRIDE = None  # Optional explicit ISO8601 start, e.g. "2021-01-01T00:00:00Z"
END_OVERRIDE = None  # Optional explicit ISO8601 end, e.g. "2024-06-01T00:00:00Z"
LIMIT = 1000  # Candles per API request (max 1000)
RETRIES = 5  # Retry attempts per request on failure
REQUEST_COOLDOWN = 0.05  # Seconds to pause between API calls (tune for rate limits)
OUTPUT_FILENAME = "binance_candles.json"  # Output JSON file name

# ---------------------------------------------------------------------------
API_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
INTERVAL_DELTA = dt.timedelta(minutes=1)
INTERVAL_MS = int(INTERVAL_DELTA.total_seconds() * 1000)


def parse_iso8601(value: str) -> dt.datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def resolve_timerange() -> Tuple[dt.datetime, dt.datetime]:
    if START_OVERRIDE or END_OVERRIDE:
        if not (START_OVERRIDE and END_OVERRIDE):
            raise ValueError("Both START_OVERRIDE and END_OVERRIDE must be set together")
        start = parse_iso8601(START_OVERRIDE)
        end = parse_iso8601(END_OVERRIDE)
    else:
        if END_YEAR < START_YEAR:
            raise ValueError("END_YEAR must be greater than or equal to START_YEAR")
        start = dt.datetime(START_YEAR, 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(END_YEAR + 1, 1, 1, tzinfo=dt.timezone.utc)

    if end <= start:
        raise ValueError("End timestamp must be after start timestamp")

    now = dt.datetime.now(dt.timezone.utc)
    now_floor = now.replace(second=0, microsecond=0)
    if now_floor <= start:
        raise ValueError("Requested range ends in the future or start is beyond available data")
    if end > now_floor:
        print(f"Clipping end timestamp to latest available candle: {now_floor.isoformat()}")
        end = now_floor
    return start, end


def fetch_candles(session: requests.Session, start_ms: int) -> List[Sequence[object]]:
    params = {
        "symbol": PAIR,
        "interval": INTERVAL,
        "limit": LIMIT,
        "startTime": start_ms,
    }
    for attempt in range(RETRIES):
        try:
            response = session.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                raise RuntimeError(f"Binance error: {payload}")
            return payload
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRIES - 1:
                raise
            backoff = min(2 ** attempt, 30)
            print(
                f"Retrying window starting {dt.datetime.utcfromtimestamp(start_ms / 1000)} after error: {exc}",
                file=sys.stderr,
            )
            time.sleep(backoff)
    return []


def iterate_candles(start: dt.datetime, end: dt.datetime) -> Iterator[Sequence[object]]:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    total_expected = (end_ms - start_ms) // INTERVAL_MS
    print(
        f"Fetching {INTERVAL} candles for {PAIR} from {start.isoformat()} to {end.isoformat()}"
    )
    print(
        f"Estimated candles: ~{total_expected:,} (limit {LIMIT} per request, cooldown {REQUEST_COOLDOWN}s)"
    )

    session = requests.Session()
    fetched = 0
    while start_ms < end_ms:
        chunk = fetch_candles(session, start_ms)
        row_count = len(chunk)
        if row_count == 0:
            print(
                f"No data returned for window starting {dt.datetime.utcfromtimestamp(start_ms / 1000)}, stopping"
            )
            break

        for row in chunk:
            open_time = int(row[0])
            if open_time >= end_ms:
                return
            if open_time < start_ms:
                continue
            fetched += 1
            yield row

        last_open = int(chunk[-1][0])
        next_start = last_open + INTERVAL_MS
        if next_start <= start_ms:
            print("Received non-forward progress from Binance, aborting to avoid infinite loop")
            break
        start_ms = next_start

        if REQUEST_COOLDOWN > 0:
            time.sleep(REQUEST_COOLDOWN)

    print(f"Fetched {fetched:,} candles")


def transform_rows(rows: List[Sequence[object]]) -> List[List[float]]:
    transformed: List[List[float]] = []
    for row in rows:
        open_time_ms = int(row[0])
        transformed.append(
            [
                open_time_ms // 1000,
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ]
        )
    return transformed


def write_output(rows: List[Sequence[object]], output_path: Path) -> None:
    candles = transform_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(candles, fh)
    print(f"Saved {len(candles)} candles to {output_path}")


def main() -> None:
    start, end = resolve_timerange()
    raw_rows = list(iterate_candles(start, end))
    if not raw_rows:
        print("No candles fetched; nothing to write", file=sys.stderr)
        sys.exit(1)

    output_path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    write_output(raw_rows, output_path)


if __name__ == "__main__":
    main()
