#!/usr/bin/env python3
"""
compare_sims: Align and compare action flow between arb_run JSON and trades JSONL.

- Merges post-trade tweak_price into trades in JSONL (using trade_happened or same-ts heuristic).
- Prints a single chronological flow (exchanges, then scheduled tweaks) with per-field deltas.
- Stars large differences and can stop at the first timestamp with structural differences (limit=dyn).

Defaults:
- --arb omitted: latest python/arb_sim/run_data/arb_run_*.json
- --jsonl omitted: ../cryptopool-simulator/trades-0.jsonl then comparison/trades-0.jsonl

Usage:
  uv run python python/arb_sim/compare_sims.py [--jsonl path] [--arb path]
    [--rtol 1e-6 --atol 1e-9]
    [--limit 30 | --limit dyn]
    [--ex-only]
"""

import argparse, json, os, re
from pathlib import Path
from collections import defaultdict


def _repo_root():
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        if (d / ".git").exists():
            return d
        if (d / "README.md").exists() and (d / "python").exists():
            return d
    return here.parents[2]


def _latest_arb_run(repo_root):
    rd = repo_root / "python" / "arb_sim" / "run_data"
    files = sorted([p for p in rd.glob("arb_run_*.json")])
    if not files:
        raise SystemExit(f"No arb_run_*.json found under {rd}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _f(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def load_arb_actions(path):
    data = json.loads(path.read_text())
    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("arb_run JSON has no runs[]")
    r = [x for x in runs if x.get("actions")][-1]
    ex, tk = [], []

    for a in r.get("actions", []):
        ts = int(a.get("ts"))
        if a.get("type") == "exchange":
            ex.append({
                "ts": ts,
                "dx": _f(a.get("dx")),
                "dy": _f(a.get("dy_after_fee")),
                "p_cex": _f(a.get("p_cex")),
                "spot_pre": _f(a.get("p_pool_before", a.get("pool_price_before"))),
                "spot_post": _f(a.get("p_pool_after", a.get("pool_price_after"))),
                "profit": _f(a.get("profit_coin0")),
                "ps_pre": _f(a.get("ps_before")),
                "ps_post": _f(a.get("ps_after", a.get("psafter"))),
                "oracle_pre": _f(a.get("oracle_before")),
                "oracle_post": _f(a.get("oracle_after")),
            })
        elif a.get("type") == "tick":
            tk.append({
                "ts": ts,
                "p_cex": _f(a.get("p_cex")),
                "ps_pre": _f(a.get("ps_before")),
                "ps_post": _f(a.get("ps_after", a.get("psafter"))),
                "oracle_pre": _f(a.get("oracle_before")),
                "oracle_post": _f(a.get("oracle_after")),
            })
    return ex, tk


def load_jsonl(path):
    ex, tk = [], []
    pending_by_ts = {}  # ts -> index of last exchange at ts
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
                    ex[idx]["ps_pre"] = _f(ev.get("ps_pre"))
                    ex[idx]["ps_post"] = _f(ev.get("ps_post"))
                    ex[idx]["oracle_pre"] = _f(ev.get("oracle_pre"))
                    ex[idx]["oracle_post"] = _f(ev.get("oracle_post"))
                    pending_by_ts.pop(ts, None)
                else:
                    tk.append({
                        "ts": ts,
                        "p_cex": _f(ev.get("p_cex")),
                        "ps_pre": _f(ev.get("ps_pre")),
                        "ps_post": _f(ev.get("ps_post")),
                        "oracle_pre": _f(ev.get("oracle_pre")),
                        "oracle_post": _f(ev.get("oracle_post")),
                    })
            else:
                tk.append({
                    "ts": ts,
                    "p_cex": _f(ev.get("p_cex")),
                    "ps_pre": _f(ev.get("ps_pre")),
                    "ps_post": _f(ev.get("ps_post")),
                    "oracle_pre": _f(ev.get("oracle_pre")),
                    "oracle_post": _f(ev.get("oracle_post")),
                })
            continue

        # Trade line
        # Support both legacy and new field names for spot price
        sp_pre = ev.get("pool_price_before")
        if sp_pre is None:
            sp_pre = ev.get("pool_spot_before")
        sp_post = ev.get("pool_price_after")
        if sp_post is None:
            sp_post = ev.get("pool_spot_after")
        ex.append({
            "ts": ts,
            "dx": _f(ev.get("dx")),
            "dy": _f(ev.get("dy")),
            "p_cex": _f(ev.get("cex_price")),
            "spot_pre": _f(sp_pre),
            "spot_post": _f(sp_post),
            "profit": _f(ev.get("profit_coin0")),
        })
        pending_by_ts[ts] = len(ex) - 1
    return ex, tk


def isclose(a, b, rtol, atol):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if abs(a - b) <= atol:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= rtol * denom


def _delta_str(a, b):
    if a is None or b is None:
        return "abs: n/a, rel: n/a"
    d = abs(a - b)
    denom = max(abs(a), abs(b)) or 1.0
    return f"abs: {d:.12g}, rel: {d/denom:.12g}"


def _rebalance_flag(e, rtol, atol):
    ps0, ps1 = e.get("ps_pre"), e.get("ps_post")
    if ps0 is None or ps1 is None:
        return 0
    return 0 if isclose(ps0, ps1, rtol, atol) else 1


# (legacy cmp_* helpers removed for simplicity)


def _group_by_ts(events):
    m = defaultdict(list)
    for e in events:
        m[e["ts"]].append(e)
    return m


def _print_exchange_event(lines, ts, event_id, a, b, rtol, atol):
    mism = 0
    if a is None or b is None:
        # Missing counterpart
        arb_ts = a.get('ts') if a else '—'
        js_ts = b.get('ts') if b else '—'
        lines.append(f"[exchange #{event_id}] ts: arb={arb_ts} jsonl={js_ts} *")
        src = a if a is not None else b
        side = "arb" if a is not None else "jsonl"
        lines.append(f"  present={side}, missing={'jsonl' if side=='arb' else 'arb'}")
        # Print available metrics for context (rebalance first)
        if src is not None:
            lines.append(f"  rebalance: {_rebalance_flag(src, rtol, atol)}")
            for k in ["dx","dy","p_cex","spot_pre","spot_post","profit","ps_pre","ps_post","oracle_pre","oracle_post"]:
                lines.append(f"  {k}: {src.get(k)}")
        return 1  # count as mismatch

    # Both present: compare and print all fields
    ts_marker = " *" if a["ts"] != b["ts"] else ""
    lines.append(f"[exchange #{event_id}] ts: arb={a['ts']} jsonl={b['ts']}{ts_marker}")
    checks = [("rebalance", _rebalance_flag(a, rtol, atol), _rebalance_flag(b, rtol, atol))]
    checks += [(k, a.get(k), b.get(k)) for k in [
        "dx","dy","p_cex","spot_pre","spot_post","profit","ps_pre","ps_post","oracle_pre","oracle_post"
    ]]
    failing = False
    for name, av, bv in checks:
        ok = isclose(av, bv, rtol, atol)
        if not ok:
            failing = True
        star = not ok
        prefix = "* " if star else ""
        lines.append(f"  {prefix}{name}: arb={av} jsonl={bv} ({_delta_str(av, bv)})")
    if failing:
        mism += 1
    return mism


def _print_tick_event(lines, ts, event_id, a, b, rtol, atol):
    mism = 0
    if a is None or b is None:
        arb_ts = a.get('ts') if a else '—'
        js_ts = b.get('ts') if b else '—'
        lines.append(f"[tweak #{event_id}] ts: arb={arb_ts} jsonl={js_ts} *")
        src = a if a is not None else b
        side = "arb" if a is not None else "jsonl"
        lines.append(f"  present={side}, missing={'jsonl' if side=='arb' else 'arb'}")
        if src is not None:
            for k in ["p_cex","ps_pre","ps_post","oracle_pre","oracle_post"]:
                lines.append(f"  {k}: {src.get(k)}")
        return 1

    ts_marker = " *" if a["ts"] != b["ts"] else ""
    lines.append(f"[tweak #{event_id}] ts: arb={a['ts']} jsonl={b['ts']}{ts_marker}")
    checks = [(k, a.get(k), b.get(k)) for k in ["p_cex","ps_pre","ps_post","oracle_pre","oracle_post"]]
    failing = False
    for name, av, bv in checks:
        ok = isclose(av, bv, rtol, atol)
        if not ok:
            failing = True
        star = not ok
        prefix = "* " if star else ""
        lines.append(f"  {prefix}{name}: arb={av} jsonl={bv} ({_delta_str(av, bv)})")
    if failing:
        mism += 1
    return mism


def unified_flow(arb_ex, arb_tk, jsonl_ex, jsonl_tk, rtol, atol, limit, dynamic, ex_only=False):
    lines: List[str] = []
    mismatches = 0
    printed = 0

    a_ex = _group_by_ts(arb_ex)
    a_tk = _group_by_ts(arb_tk)
    b_ex = _group_by_ts(jsonl_ex)
    b_tk = _group_by_ts(jsonl_tk)

    all_ts = sorted(set(a_ex.keys()) | set(a_tk.keys()) | set(b_ex.keys()) | set(b_tk.keys()))
    events_budget = None if limit <= 0 else max(0, limit)

    event_id = 0
    for ts in all_ts:
        if events_budget is not None and events_budget == 0:
            break
        # Detect structural differences at this timestamp:
        # event present on one side but not the other (by kind or count)
        la_ex = a_ex.get(ts, [])
        lb_ex = b_ex.get(ts, [])
        la_tk = a_tk.get(ts, [])
        lb_tk = b_tk.get(ts, [])
        # Structural difference for this timestamp. If ex_only, ignore ticks.
        structural_diff_ex = (len(la_ex) != len(lb_ex))
        structural_diff_tk = (len(la_tk) != len(lb_tk))
        structural_diff = structural_diff_ex or (not ex_only and structural_diff_tk)
        # Exchanges first at this ts
        la = la_ex
        lb = lb_ex
        m = max(len(la), len(lb))
        for i in range(m):
            if lines:
                lines.append("")
            e_a = la[i] if i < len(la) else None
            e_b = lb[i] if i < len(lb) else None
            mismatches += _print_exchange_event(lines, ts, event_id, e_a, e_b, rtol, atol)
            printed += 1
            event_id += 1
            if events_budget is not None:
                events_budget -= 1
                if events_budget == 0:
                    break
        if events_budget is not None and events_budget == 0:
            break

        # Ticks at this ts (suppressed if ex_only)
        if not ex_only:
            la = la_tk
            lb = lb_tk
            m = max(len(la), len(lb))
            for i in range(m):
                if lines:
                    lines.append("")
                e_a = la[i] if i < len(la) else None
                e_b = lb[i] if i < len(lb) else None
                mismatches += _print_tick_event(lines, ts, event_id, e_a, e_b, rtol, atol)
                printed += 1
                event_id += 1
                if events_budget is not None:
                    events_budget -= 1
                    if events_budget == 0:
                        break

        # In dynamic mode, stop right after the first timestamp with structural difference
        if dynamic and structural_diff:
            break

    stats = {"timestamps": len(all_ts), "printed": printed, "mismatches": mismatches}
    return lines, stats


def main():
    ap = argparse.ArgumentParser(description="Compare action flows (simplified)")
    ap.add_argument("--arb", default=None, help="Path to arb_run_*.json (default: latest)")
    ap.add_argument("--jsonl", default=None, help="Path to trades-*.jsonl (default search: ../cryptopool-simulator/trades-0.jsonl, comparison/trades-0.jsonl)")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--limit", default="30", help="N events or 'dyn' to stop at first structural timestamp difference")
    ap.add_argument("--ex-only", action="store_true", help="Only output exchange events; suppress tweak_price events")
    args = ap.parse_args()

    root = _repo_root()
    arb_path = Path(args.arb) if args.arb else _latest_arb_run(root)

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        candidates = [
            (root.parent / "cryptopool-simulator" / "trades-0.jsonl").resolve(),
            (root / "cryptopool-simulator" / "trades-0.jsonl").resolve(),
            (root / "comparison" / "trades-0.jsonl").resolve(),
        ]
        jsonl_path = next((p for p in candidates if p.exists()), None)
        if jsonl_path is None:
            raise SystemExit("Could not locate JSONL (trades-0.jsonl) via defaults and none provided.")

    arb_ex, arb_tk = load_arb_actions(arb_path)
    jsonl_ex, jsonl_tk = load_jsonl(jsonl_path)

    # Unified flow comparison
    dyn = False
    try:
        limit_val = int(args.limit)
    except ValueError:
        if str(args.limit).lower() == "dyn":
            dyn = True
            limit_val = 0
        else:
            raise SystemExit("--limit must be an integer or 'dyn'")

    uni_lines, uni_stats = unified_flow(arb_ex, arb_tk, jsonl_ex, jsonl_tk, args.rtol, args.atol, limit_val, dyn, ex_only=args.ex_only)

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
