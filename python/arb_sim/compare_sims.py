#!/usr/bin/env python3
"""
compare_v2: Align and compare action flow between arb_run JSON and trades JSONL.

Mappings:
- Exchange (arb_run 'exchange') vs JSONL trade + its following tweak_price (post-swap):
  - dx ↔ dx
  - dy_after_fee ↔ dy
  - p_cex ↔ cex_price
  - pool_price_before/after ↔ pool_price_before/after
  - profit_coin0 ↔ profit_coin0
  - ps_before/after ↔ tweak_price.ps_pre/ps_post
  - oracle_before/after ↔ tweak_price.oracle_pre/oracle_post

- Scheduled tweak (arb_run 'tick') vs JSONL standalone tweak_price (trade_happened=0):
  - p_cex ↔ p_cex (if present)
  - ps_before/after ↔ ps_pre/ps_post
  - oracle_before/after ↔ oracle_pre/oracle_post

Defaults:
- --arb omitted: picks latest python/arb_sim/run_data/arb_run_*.json
- --jsonl omitted: picks ../cryptopool-simulator/trades-0.jsonl, then comparison/trades-0.jsonl

Usage:
  uv run python comparison/compare_v2.py [--jsonl path] [--arb path]
    [--rtol 1e-6 --atol 1e-9]
    [--star-rtol <same as rtol>] [--star-atol <same as atol>]
    [--limit 30]   # number of events to print per stream (<=0 prints all)
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # Walk up to find the git repo root (contains .git) or recognizable markers
    for d in [here.parent] + list(here.parents):
        if (d / ".git").exists():
            return d
        # Fallback markers when .git isn't available in the sandbox
        if (d / "README.md").exists() and (d / "python").exists():
            return d
    # Last resort: go up two levels
    return here.parents[2]


def _latest_arb_run(repo_root: Path) -> Path:
    rd = repo_root / "python" / "arb_sim" / "run_data"
    files = sorted([p for p in rd.glob("arb_run_*.json")])
    if not files:
        raise SystemExit(f"No arb_run_*.json found under {rd}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class Exchange:
    ts: int
    i: Optional[int]
    j: Optional[int]
    dx: Optional[float]
    dy: Optional[float]
    p_cex: Optional[float]
    spot_pre: Optional[float]
    spot_post: Optional[float]
    profit: Optional[float]
    ps_pre: Optional[float] = None
    ps_post: Optional[float] = None
    oracle_pre: Optional[float] = None
    oracle_post: Optional[float] = None


@dataclass
class Tick:
    ts: int
    p_cex: Optional[float]
    ps_pre: Optional[float]
    ps_post: Optional[float]
    oracle_pre: Optional[float]
    oracle_post: Optional[float]


def load_arb_actions(path: Path) -> Tuple[List[Exchange], List[Tick]]:
    data = json.loads(path.read_text())
    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("arb_run JSON has no runs[]")
    r = [x for x in runs if x.get("actions")][-1]
    ex: List[Exchange] = []
    tk: List[Tick] = []

    for a in r.get("actions", []):
        ty = a.get("type")
        ts = int(a.get("ts"))
        if ty == "exchange":
            ex.append(
                Exchange(
                    ts=ts,
                    i=int(a.get("i")),
                    j=int(a.get("j")),
                    dx=_float(a.get("dx")),
                    dy=_float(a.get("dy_after_fee")),
                    p_cex=_float(a.get("p_cex")),
                    spot_pre=_float(a.get("p_pool_before", a.get("pool_price_before"))),
                    spot_post=_float(a.get("p_pool_after", a.get("pool_price_after"))),
                    profit=_float(a.get("profit_coin0")),
                    ps_pre=_float(a.get("ps_before")),
                    ps_post=_float(a.get("ps_after", a.get("psafter"))),
                    oracle_pre=_float(a.get("oracle_before")),
                    oracle_post=_float(a.get("oracle_after")),
                )
            )
        elif ty == "tick":
            tk.append(
                Tick(
                    ts=ts,
                    p_cex=_float(a.get("p_cex")),
                    ps_pre=_float(a.get("ps_before")),
                    ps_post=_float(a.get("ps_after", a.get("psafter"))),
                    oracle_pre=_float(a.get("oracle_before")),
                    oracle_post=_float(a.get("oracle_after")),
                )
            )
    return ex, tk


def load_jsonl(path: Path) -> Tuple[List[Exchange], List[Tick]]:
    ex: List[Exchange] = []
    tk: List[Tick] = []
    pending_by_ts: Dict[int, int] = {}  # ts -> index of last exchange at ts
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except Exception:
            # Handle trailing comma before closing brace: ", }"
            sanitized = re.sub(r",\s*}\s*$", "}", raw)
            try:
                ev = json.loads(sanitized)
            except Exception:
                # Last resort: strip trailing comma chars
                try:
                    ev = json.loads(raw.rstrip(","))
                except Exception:
                    continue
        ts = int(ev.get("t"))
        ty = ev.get("type")
        if ty == "tweak_price":
            th = ev.get("trade_happened")
            merge_candidate = (th in (1, True)) or (th is None and ts in pending_by_ts)
            if merge_candidate:
                idx = pending_by_ts.get(ts)
                if idx is not None:
                    ex[idx].ps_pre = _float(ev.get("ps_pre"))
                    ex[idx].ps_post = _float(ev.get("ps_post"))
                    ex[idx].oracle_pre = _float(ev.get("oracle_pre"))
                    ex[idx].oracle_post = _float(ev.get("oracle_post"))
                    pending_by_ts.pop(ts, None)
                else:
                    # Flag says post-trade but no matching trade at ts; treat as scheduled
                    tk.append(
                        Tick(
                            ts=ts,
                            p_cex=_float(ev.get("p_cex")),
                            ps_pre=_float(ev.get("ps_pre")),
                            ps_post=_float(ev.get("ps_post")),
                            oracle_pre=_float(ev.get("oracle_pre")),
                            oracle_post=_float(ev.get("oracle_post")),
                        )
                    )
            else:
                tk.append(
                    Tick(
                        ts=ts,
                        p_cex=_float(ev.get("p_cex")),
                        ps_pre=_float(ev.get("ps_pre")),
                        ps_post=_float(ev.get("ps_post")),
                        oracle_pre=_float(ev.get("oracle_pre")),
                        oracle_post=_float(ev.get("oracle_post")),
                    )
                )
            continue

        # Trade line
        e = Exchange(
            ts=ts,
            i=int(ev.get("from")) if ev.get("from") is not None else None,
            j=int(ev.get("to")) if ev.get("to") is not None else None,
            dx=_float(ev.get("dx")),
            dy=_float(ev.get("dy")),
            p_cex=_float(ev.get("cex_price")),
            spot_pre=_float(ev.get("pool_price_before")),
            spot_post=_float(ev.get("pool_price_after")),
            profit=_float(ev.get("profit_coin0")),
        )
        ex.append(e)
        pending_by_ts[ts] = len(ex) - 1
    return ex, tk


def isclose(a: Optional[float], b: Optional[float], rtol: float, atol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if abs(a - b) <= atol:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= rtol * denom


def _delta_str(av: Optional[float], bv: Optional[float]) -> str:
    if av is None or bv is None:
        return "abs: n/a, rel: n/a"
    absd = abs(av - bv)
    denom = max(abs(av), abs(bv))
    if denom == 0:
        denom = 1.0
    reld = absd / denom
    return f"abs: {absd:.12g}, rel: {reld:.12g}"


def cmp_exchanges(
    a: List[Exchange],
    b: List[Exchange],
    rtol: float,
    atol: float,
    limit: int,
    star_rtol: Optional[float],
    star_atol: Optional[float],
) -> Tuple[int, int, List[str]]:
    n = min(len(a), len(b))
    mism = 0
    lines: List[str] = []
    events_budget = None if limit <= 0 else max(0, limit)
    last_ts_out: Optional[int] = None
    for k in range(n):
        x, y = a[k], b[k]
        # Count mismatches even if not printed
        if x.ts != y.ts:
            mism += 1
        # Respect event printing budget
        if events_budget is not None and events_budget == 0:
            continue
        # Add separator between different timestamps
        if lines and last_ts_out != x.ts:
            lines.append("")
        last_ts_out = x.ts
        # Print ts line with star if mismatch
        ts_marker = " *" if x.ts != y.ts else ""
        lines.append(f"[exchange #{k}] ts: arb={x.ts} jsonl={y.ts}{ts_marker}")
        checks = [
            ("dx", x.dx, y.dx),
            ("dy", x.dy, y.dy),
            ("p_cex", x.p_cex, y.p_cex),
            ("spot_pre", x.spot_pre, y.spot_pre),
            ("spot_post", x.spot_post, y.spot_post),
            ("profit", x.profit, y.profit),
            ("ps_pre", x.ps_pre, y.ps_pre),
            ("ps_post", x.ps_post, y.ps_post),
            ("oracle_pre", x.oracle_pre, y.oracle_pre),
            ("oracle_post", x.oracle_post, y.oracle_post),
        ]
        failing = False
        for name, av, bv in checks:
            ok = isclose(av, bv, rtol, atol)
            if not ok:
                failing = True
            # star threshold uses star_* if provided, else rtol/atol
            srtol = star_rtol if star_rtol is not None else rtol
            satol = star_atol if star_atol is not None else atol
            star = " *" if not isclose(av, bv, srtol, satol) else ""
            lines.append(
                f"[exchange #{k} @ {x.ts}] {name}: arb={av} jsonl={bv} ({_delta_str(av, bv)}){star}"
            )
        if failing:
            mism += 1
        if events_budget is not None:
            events_budget -= 1
    return n, mism, lines


def cmp_ticks(
    a: List[Tick],
    b: List[Tick],
    rtol: float,
    atol: float,
    limit: int,
    star_rtol: Optional[float],
    star_atol: Optional[float],
) -> Tuple[int, int, List[str]]:
    n = min(len(a), len(b))
    mism = 0
    lines: List[str] = []
    events_budget = None if limit <= 0 else max(0, limit)
    last_ts_out: Optional[int] = None
    for k in range(n):
        x, y = a[k], b[k]
        if x.ts != y.ts:
            mism += 1
        if events_budget is not None and events_budget == 0:
            continue
        if lines and last_ts_out != x.ts:
            lines.append("")
        last_ts_out = x.ts
        ts_marker = " *" if x.ts != y.ts else ""
        lines.append(f"[tick #{k}] ts: arb={x.ts} jsonl={y.ts}{ts_marker}")
        checks = [
            ("p_cex", x.p_cex, y.p_cex),
            ("ps_pre", x.ps_pre, y.ps_pre),
            ("ps_post", x.ps_post, y.ps_post),
            ("oracle_pre", x.oracle_pre, y.oracle_pre),
            ("oracle_post", x.oracle_post, y.oracle_post),
        ]
        failing = False
        for name, av, bv in checks:
            ok = isclose(av, bv, rtol, atol)
            if not ok:
                failing = True
            srtol = star_rtol if star_rtol is not None else rtol
            satol = star_atol if star_atol is not None else atol
            star = " *" if not isclose(av, bv, srtol, satol) else ""
            lines.append(
                f"[tick #{k} @ {x.ts}] {name}: arb={av} jsonl={bv} ({_delta_str(av, bv)}){star}"
            )
        if failing:
            mism += 1
        if events_budget is not None:
            events_budget -= 1
    return n, mism, lines


def _group_by_ts_ex(events: List[Exchange]) -> Dict[int, List[Exchange]]:
    m: Dict[int, List[Exchange]] = defaultdict(list)
    for e in events:
        m[e.ts].append(e)
    return m


def _group_by_ts_tk(events: List[Tick]) -> Dict[int, List[Tick]]:
    m: Dict[int, List[Tick]] = defaultdict(list)
    for e in events:
        m[e.ts].append(e)
    return m


def _print_exchange_event(
    lines: List[str],
    ts: int,
    event_id: int,
    a: Optional[Exchange],
    b: Optional[Exchange],
    *,
    rtol: float,
    atol: float,
    star_rtol: Optional[float],
    star_atol: Optional[float],
) -> int:
    mism = 0
    if a is None or b is None:
        # Missing counterpart
        lines.append(f"[exchange #{event_id}] ts: arb={a.ts if a else '—'} jsonl={b.ts if b else '—'} *")
        src = a if a is not None else b
        side = "arb" if a is not None else "jsonl"
        lines.append(f"  present={side}, missing={'jsonl' if side=='arb' else 'arb'}")
        # Print available metrics for context
        if src is not None:
            lines.append(f"  dx: {getattr(src, 'dx', None)}")
            lines.append(f"  dy: {getattr(src, 'dy', None)}")
            lines.append(f"  p_cex: {getattr(src, 'p_cex', None)}")
            lines.append(f"  spot_pre: {getattr(src, 'spot_pre', None)}")
            lines.append(f"  spot_post: {getattr(src, 'spot_post', None)}")
            lines.append(f"  profit: {getattr(src, 'profit', None)}")
            lines.append(f"  ps_pre: {getattr(src, 'ps_pre', None)}")
            lines.append(f"  ps_post: {getattr(src, 'ps_post', None)}")
            lines.append(f"  oracle_pre: {getattr(src, 'oracle_pre', None)}")
            lines.append(f"  oracle_post: {getattr(src, 'oracle_post', None)}")
        return 1  # count as mismatch

    # Both present: compare and print all fields
    ts_marker = " *" if a.ts != b.ts else ""
    lines.append(f"[exchange #{event_id}] ts: arb={a.ts} jsonl={b.ts}{ts_marker}")
    checks = [
        ("dx", a.dx, b.dx),
        ("dy", a.dy, b.dy),
        ("p_cex", a.p_cex, b.p_cex),
        ("spot_pre", a.spot_pre, b.spot_pre),
        ("spot_post", a.spot_post, b.spot_post),
        ("profit", a.profit, b.profit),
        ("ps_pre", a.ps_pre, b.ps_pre),
        ("ps_post", a.ps_post, b.ps_post),
        ("oracle_pre", a.oracle_pre, b.oracle_pre),
        ("oracle_post", a.oracle_post, b.oracle_post),
    ]
    failing = False
    srtol = star_rtol if star_rtol is not None else rtol
    satol = star_atol if star_atol is not None else atol
    for name, av, bv in checks:
        ok = isclose(av, bv, rtol, atol)
        if not ok:
            failing = True
        star = " *" if not isclose(av, bv, srtol, satol) else ""
        lines.append(f"  {name}: arb={av} jsonl={bv} ({_delta_str(av, bv)}){star}")
    if failing:
        mism += 1
    return mism


def _print_tick_event(
    lines: List[str],
    ts: int,
    event_id: int,
    a: Optional[Tick],
    b: Optional[Tick],
    *,
    rtol: float,
    atol: float,
    star_rtol: Optional[float],
    star_atol: Optional[float],
) -> int:
    mism = 0
    if a is None or b is None:
        lines.append(f"[tweak #{event_id}] ts: arb={a.ts if a else '—'} jsonl={b.ts if b else '—'} *")
        src = a if a is not None else b
        side = "arb" if a is not None else "jsonl"
        lines.append(f"  present={side}, missing={'jsonl' if side=='arb' else 'arb'}")
        if src is not None:
            lines.append(f"  p_cex: {getattr(src, 'p_cex', None)}")
            lines.append(f"  ps_pre: {getattr(src, 'ps_pre', None)}")
            lines.append(f"  ps_post: {getattr(src, 'ps_post', None)}")
            lines.append(f"  oracle_pre: {getattr(src, 'oracle_pre', None)}")
            lines.append(f"  oracle_post: {getattr(src, 'oracle_post', None)}")
        return 1

    ts_marker = " *" if a.ts != b.ts else ""
    lines.append(f"[tweak #{event_id}] ts: arb={a.ts} jsonl={b.ts}{ts_marker}")
    checks = [
        ("p_cex", a.p_cex, b.p_cex),
        ("ps_pre", a.ps_pre, b.ps_pre),
        ("ps_post", a.ps_post, b.ps_post),
        ("oracle_pre", a.oracle_pre, b.oracle_pre),
        ("oracle_post", a.oracle_post, b.oracle_post),
    ]
    failing = False
    srtol = star_rtol if star_rtol is not None else rtol
    satol = star_atol if star_atol is not None else atol
    for name, av, bv in checks:
        ok = isclose(av, bv, rtol, atol)
        if not ok:
            failing = True
        star = " *" if not isclose(av, bv, srtol, satol) else ""
        lines.append(f"  {name}: arb={av} jsonl={bv} ({_delta_str(av, bv)}){star}")
    if failing:
        mism += 1
    return mism


def unified_flow(
    arb_ex: List[Exchange],
    arb_tk: List[Tick],
    jsonl_ex: List[Exchange],
    jsonl_tk: List[Tick],
    *,
    rtol: float,
    atol: float,
    star_rtol: Optional[float],
    star_atol: Optional[float],
    limit: int,
    dynamic: bool,
) -> Tuple[List[str], Dict[str, int]]:
    lines: List[str] = []
    mismatches = 0
    printed = 0

    a_ex = _group_by_ts_ex(arb_ex)
    a_tk = _group_by_ts_tk(arb_tk)
    b_ex = _group_by_ts_ex(jsonl_ex)
    b_tk = _group_by_ts_tk(jsonl_tk)

    all_ts = sorted(set(a_ex.keys()) | set(a_tk.keys()) | set(b_ex.keys()) | set(b_tk.keys()))
    events_budget = None if limit <= 0 else max(0, limit)

    event_id = 0
    for ts in all_ts:
        if events_budget is not None and events_budget == 0:
            break
        mism_before = mismatches
        # Exchanges first at this ts
        la = a_ex.get(ts, [])
        lb = b_ex.get(ts, [])
        m = max(len(la), len(lb))
        for i in range(m):
            if lines:
                lines.append("")
            e_a = la[i] if i < len(la) else None
            e_b = lb[i] if i < len(lb) else None
            mismatches += _print_exchange_event(
                lines,
                ts,
                event_id,
                e_a,
                e_b,
                rtol=rtol,
                atol=atol,
                star_rtol=star_rtol,
                star_atol=star_atol,
            )
            printed += 1
            event_id += 1
            if events_budget is not None:
                events_budget -= 1
                if events_budget == 0:
                    break
        if events_budget is not None and events_budget == 0:
            break

        # Ticks at this ts
        la = a_tk.get(ts, [])
        lb = b_tk.get(ts, [])
        m = max(len(la), len(lb))
        for i in range(m):
            if lines:
                lines.append("")
            e_a = la[i] if i < len(la) else None
            e_b = lb[i] if i < len(lb) else None
            mismatches += _print_tick_event(
                lines,
                ts,
                event_id,
                e_a,
                e_b,
                rtol=rtol,
                atol=atol,
                star_rtol=star_rtol,
                star_atol=star_atol,
            )
            printed += 1
            event_id += 1
            if events_budget is not None:
                events_budget -= 1
                if events_budget == 0:
                    break

        # In dynamic mode, stop right after the first timestamp that produced any mismatch
        if dynamic and mismatches > mism_before:
            break

    stats = {"timestamps": len(all_ts), "printed": printed, "mismatches": mismatches}
    return lines, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare action flows (v2)")
    ap.add_argument("--arb", type=Path, default=None, help="Path to arb_run_*.json (default: latest)")
    ap.add_argument("--jsonl", type=Path, default=None, help="Path to trades-*.jsonl (default: ../cryptopool-simulator/trades-0.jsonl or comparison/trades-0.jsonl)")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--star-rtol", type=float, default=None, help="Relative threshold for marking with * (default: rtol)")
    ap.add_argument("--star-atol", type=float, default=None, help="Absolute threshold for marking with * (default: atol)")
    ap.add_argument(
        "--limit",
        type=str,
        default="30",
        help="Number of events to print (int) or 'dyn' to print until first differing timestamp",
    )
    args = ap.parse_args()

    root = _repo_root()
    arb_path = args.arb if args.arb else _latest_arb_run(root)

    if args.jsonl:
        jsonl_path = args.jsonl
    else:
        candidates = [
            (root.parent / "cryptopool-simulator" / "trades-0.jsonl").resolve(),  # sibling project
            (root / "cryptopool-simulator" / "trades-0.jsonl").resolve(),          # vendored subfolder
            (root / "comparison" / "trades-0.jsonl").resolve(),                     # fallback in-repo
        ]
        jsonl_path = next((p for p in candidates if p.exists()), None)
        if jsonl_path is None:
            raise SystemExit("Could not locate JSONL (trades-0.jsonl) via defaults and none provided.")

    arb_ex, arb_tk = load_arb_actions(arb_path)
    jsonl_ex, jsonl_tk = load_jsonl(jsonl_path)

    # Unified flow comparison
    # Interpret limit
    dyn = False
    try:
        limit_val = int(args.limit)
    except ValueError:
        if str(args.limit).lower() == "dyn":
            dyn = True
            limit_val = 0
        else:
            raise SystemExit("--limit must be an integer or 'dyn'")

    uni_lines, uni_stats = unified_flow(
        arb_ex,
        arb_tk,
        jsonl_ex,
        jsonl_tk,
        rtol=args.rtol,
        atol=args.atol,
        star_rtol=args.star_rtol,
        star_atol=args.star_atol,
        limit=limit_val,
        dynamic=dyn,
    )

    print("Summary:")
    print(f"  exchanges: arb={len(arb_ex)} jsonl={len(jsonl_ex)}")
    print(f"  tweaks   : arb={len(arb_tk)} jsonl={len(jsonl_tk)}")
    print(f"  unified  : timestamps={uni_stats['timestamps']} events_printed={uni_stats['printed']} mismatches={uni_stats['mismatches']}")

    if uni_lines:
        title = "Unified flow (until first differing timestamp):" if dyn else "Unified flow (first N events):"
        print("\n" + title)
        for line in uni_lines:
            print("  " + line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
