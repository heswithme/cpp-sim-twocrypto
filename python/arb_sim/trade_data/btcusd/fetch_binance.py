#!/usr/bin/env python3.11

"""Fetch Binance 1m candlesticks for a configurable multi-year range."""

import datetime as dt
import json
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Iterator, List, MutableMapping, Sequence, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration (edit these values if you need a different dataset)
# ---------------------------------------------------------------------------
PAIR = "BTCUSDT"  # Binance trading pair symbol
START_YEAR = 2025  # First calendar year (inclusive)
END_YEAR = 2025  # Last calendar year (inclusive)
START_OVERRIDE = None  # Optional explicit ISO8601 start, e.g. "2021-01-01T00:00:00Z"
END_OVERRIDE = None  # Optional explicit ISO8601 end, e.g. "2024-06-01T00:00:00Z"
LIMIT = 1000  # Candles per API request (max 1000)
WORKERS = 8  # Number of concurrent requests
RETRIES = 5  # Retry attempts per request on failure
REQUEST_COOLDOWN = 0.05  # Seconds to pause between API calls per worker
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
    est_requests = max(1, (end_ms - start_ms + (LIMIT * INTERVAL_MS) - 1) // (LIMIT * INTERVAL_MS))
    print(
        f"Fetching {INTERVAL} candles for {PAIR} from {start.isoformat()} to {end.isoformat()}"
    )
    print(
        f"Estimated requests: ~{est_requests:,} (limit {LIMIT} per call, {WORKERS} workers, cooldown {REQUEST_COOLDOWN}s)"
    )

    sessions = [requests.Session() for _ in range(WORKERS)]
    task_queue: Queue[int] = Queue()
    task_queue.put(start_ms)

    cutoff_lock = Lock()
    cutoff_ms = end_ms
    results_lock = Lock()
    collected: MutableMapping[int, Sequence[object]] = {}
    chunks_done = 0
    chunks_lock = Lock()
    stop_event = Event()

    def worker(worker_id: int) -> None:
        nonlocal cutoff_ms, chunks_done
        session = sessions[worker_id]
        while True:
            if stop_event.is_set() and task_queue.empty():
                break
            try:
                window_start = task_queue.get(timeout=0.2)
            except Empty:
                if stop_event.is_set():
                    break
                continue

            with cutoff_lock:
                current_cutoff = cutoff_ms
            if window_start >= current_cutoff:
                task_queue.task_done()
                continue

            try:
                chunk = fetch_candles(session, window_start)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Request failed for window starting {dt.datetime.utcfromtimestamp(window_start / 1000)}: {exc}",
                    file=sys.stderr,
                )
                stop_event.set()
                task_queue.task_done()
                break

            row_count = len(chunk)
            with chunks_lock:
                chunks_done += 1
                print(
                    f"[{chunks_done}] {PAIR} {dt.datetime.utcfromtimestamp(window_start / 1000)} → {row_count} rows"
                )

            if row_count == 0:
                with cutoff_lock:
                    if window_start < cutoff_ms:
                        cutoff_ms = window_start
                stop_event.set()
                task_queue.task_done()
                continue

            local_last_open = int(chunk[-1][0])
            local_next_start = local_last_open + INTERVAL_MS

            with cutoff_lock:
                if row_count < LIMIT and local_next_start < cutoff_ms:
                    cutoff_ms = local_next_start
                    stop_event.set()
                current_cutoff = cutoff_ms

            with results_lock:
                for row in chunk:
                    open_time = int(row[0])
                    if open_time < window_start:
                        continue
                    if open_time >= current_cutoff or open_time >= end_ms:
                        continue
                    collected.setdefault(open_time, row)

            if not stop_event.is_set():
                if local_next_start < current_cutoff and local_next_start < end_ms:
                    task_queue.put(local_next_start)

            task_queue.task_done()
            if REQUEST_COOLDOWN > 0:
                time.sleep(REQUEST_COOLDOWN)

    threads = [Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS)]
    for thread in threads:
        thread.start()

    task_queue.join()
    stop_event.set()
    for thread in threads:
        thread.join()

    if not collected:
        print("No candles fetched; aborting", file=sys.stderr)
        return iter(() )

    ordered_times = sorted(collected)
    print(f"Fetched {len(ordered_times):,} candles")
    return (collected[ts] for ts in ordered_times)


def summarize_rows(rows: List[Sequence[object]], start: dt.datetime, end: dt.datetime) -> None:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    times = [int(row[0]) for row in rows]
    if not times:
        return
    first = times[0]
    last = times[-1]
    expected = max(0, (min(end_ms, last + INTERVAL_MS) - start_ms) // INTERVAL_MS)
    missing = 0
    prev = first
    for current in times[1:]:
        gap = current - prev
        if gap > INTERVAL_MS:
            missing += gap // INTERVAL_MS - 1
        prev = current
    print(
        f"First candle: {dt.datetime.utcfromtimestamp(first / 1000)} | "
        f"Last candle: {dt.datetime.utcfromtimestamp(last / 1000)}"
    )
    print(
        f"Collected {len(times):,} candles (expected ≈ {expected:,}); gaps detected: {missing:,}"
    )

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
        sys.exit(1)

    summarize_rows(raw_rows, start, end)

    output_path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    write_output(raw_rows, output_path)


if __name__ == "__main__":
    main()
