#!/usr/bin/env python3
"""
compare_sims: Align and compare action flow between new_sim (arb_run JSON) and mich_sim (trades JSONL).

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
from typing import List
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
                # New: virtual price and xcp_profit before/after (arb/new_sim fields)
                "vp_pre": _f(a.get("vp_before", a.get("vp_pre"))),
                "vp_post": _f(a.get("vp_after", a.get("vp_post"))),
                "xcp_profit_pre": _f(a.get("xcp_profit_before", a.get("xcp_profit_pre"))),
                "xcp_profit_post": _f(a.get("xcp_profit_after", a.get("xcp_profit_post"))),
            })
        elif a.get("type") == "tick":
            tk.append({
                "ts": ts,
                "p_cex": _f(a.get("p_cex")),
                "ps_pre": _f(a.get("ps_before")),
                "ps_post": _f(a.get("ps_after", a.get("psafter"))),
                "oracle_pre": _f(a.get("oracle_before")),
                "oracle_post": _f(a.get("oracle_after")),
                # New: virtual price and xcp_profit before/after if provided
                "vp_pre": _f(a.get("vp_before", a.get("vp_pre"))),
                "vp_post": _f(a.get("vp_after", a.get("vp_post"))),
                "xcp_profit_pre": _f(a.get("xcp_profit_before", a.get("xcp_profit_pre"))),
                "xcp_profit_post": _f(a.get("xcp_profit_after", a.get("xcp_profit_post"))),
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
                    # Also capture vp/xcp_profit if present, but prefer values from exchange line if already set
                    if ex[idx].get("vp_pre") is None:
                        ex[idx]["vp_pre"] = _f(ev.get("vp_pre", ev.get("vp_before")))
                    if ex[idx].get("vp_post") is None:
                        ex[idx]["vp_post"] = _f(ev.get("vp_post", ev.get("vp_after")))
                    if ex[idx].get("xcp_profit_pre") is None:
                        ex[idx]["xcp_profit_pre"] = _f(ev.get("xcp_profit_pre", ev.get("xcp_profit_before")))
                    if ex[idx].get("xcp_profit_post") is None:
                        ex[idx]["xcp_profit_post"] = _f(ev.get("xcp_profit_post", ev.get("xcp_profit_after")))
                    pending_by_ts.pop(ts, None)
                else:
                    tk.append({
                        "ts": ts,
                        "p_cex": _f(ev.get("p_cex")),
                        "ps_pre": _f(ev.get("ps_pre")),
                        "ps_post": _f(ev.get("ps_post")),
                        "oracle_pre": _f(ev.get("oracle_pre")),
                        "oracle_post": _f(ev.get("oracle_post")),
                        # New: vp/xcp_profit if present on tweak
                        "vp_pre": _f(ev.get("vp_pre", ev.get("vp_before"))),
                        "vp_post": _f(ev.get("vp_post", ev.get("vp_after"))),
                        "xcp_profit_pre": _f(ev.get("xcp_profit_pre", ev.get("xcp_profit_before"))),
                        "xcp_profit_post": _f(ev.get("xcp_profit_post", ev.get("xcp_profit_after"))),
                    })
            else:
                tk.append({
                    "ts": ts,
                    "p_cex": _f(ev.get("p_cex")),
                    "ps_pre": _f(ev.get("ps_pre")),
                    "ps_post": _f(ev.get("ps_post")),
                    "oracle_pre": _f(ev.get("oracle_pre")),
                    "oracle_post": _f(ev.get("oracle_post")),
                    # New: vp/xcp_profit if present on tweak
                    "vp_pre": _f(ev.get("vp_pre", ev.get("vp_before"))),
                    "vp_post": _f(ev.get("vp_post", ev.get("vp_after"))),
                    "xcp_profit_pre": _f(ev.get("xcp_profit_pre", ev.get("xcp_profit_before"))),
                    "xcp_profit_post": _f(ev.get("xcp_profit_post", ev.get("xcp_profit_after"))),
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
            # New: vp/xcp_profit if present on trade lines
            "vp_pre": _f(ev.get("vp_pre", ev.get("vp_before"))),
            "vp_post": _f(ev.get("vp_post", ev.get("vp_after"))),
            "xcp_profit_pre": _f(ev.get("xcp_profit_pre", ev.get("xcp_profit_before"))),
            "xcp_profit_post": _f(ev.get("xcp_profit_post", ev.get("xcp_profit_after"))),
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


def _format_value(name, v):
    if v is None:
        return "—"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if name in ("vp_pre","vp_post","xcp_profit_pre","xcp_profit_post"):
        return f"{x:0.12f}"
    return f"{x:0.4f}"


def _print_table(lines, kind, event_id, a, b, metrics, rtol, atol):
    ts_a = a.get('ts') if a else '—'
    ts_b = b.get('ts') if b else '—'
    lines.append(f"[{kind} #{event_id}] ts: new_sim={ts_a} mich_sim={ts_b}")
    rows = [("new_sim", a or {}), ("mich_sim", b or {})]
    label_w = max(len("simulator"), *(len(r[0]) for r in rows))
    col_ws = []
    for m in metrics:
        w = len(m)
        for _, row in rows:
            w = max(w, len(_format_value(m, row.get(m))))
        col_ws.append(w)
    header = ["simulator".ljust(label_w)] + [metrics[i].rjust(col_ws[i]) for i in range(len(metrics))]
    lines.append(" | ".join(header))
    sep = ["-" * label_w] + ["-" * col_ws[i] for i in range(len(metrics))]
    lines.append("-+-".join(sep))
    for name, row in rows:
        vals = [name.ljust(label_w)] + [_format_value(metrics[i], row.get(metrics[i])).rjust(col_ws[i]) for i in range(len(metrics))]
        lines.append(" | ".join(vals))
    # Add per-metric absolute and relative difference rows
    abs_vals: List[str] = ["abs_diff".ljust(label_w)]
    rel_vals: List[str] = ["rel_diff".ljust(label_w)]
    for i, mname in enumerate(metrics):
        av = (a or {}).get(mname)
        bv = (b or {}).get(mname)
        try:
            if av is None or bv is None:
                abs_s = rel_s = "—"
            else:
                fa = float(av); fb = float(bv)
                abs_d = abs(fa - fb)
                denom = max(abs(fa), abs(fb), 1.0)
                rel_d = abs_d / denom
                abs_s = f"{abs_d:.3e}"
                rel_s = f"{rel_d:.3e}"
        except Exception:
            abs_s = rel_s = "—"
        abs_vals.append(abs_s.rjust(col_ws[i]))
        rel_vals.append(rel_s.rjust(col_ws[i]))
    lines.append(" | ".join(abs_vals))
    lines.append(" | ".join(rel_vals))


def _print_exchange_event(lines, ts, event_id, a, b, rtol, atol):
    metrics = [
        "dx","dy","p_cex","spot_pre","spot_post","profit",
        "ps_pre","ps_post","oracle_pre","oracle_post",
        "vp_pre","vp_post","xcp_profit_pre","xcp_profit_post",
    ]
    mism = 0
    if a is None or b is None:
        _print_table(lines, "exchange", event_id, a, b, metrics, rtol, atol)
        return 1
    failing = False
    for m in metrics:
        if not isclose(a.get(m), b.get(m), rtol, atol):
            failing = True
    _print_table(lines, "exchange", event_id, a, b, metrics, rtol, atol)
    if failing:
        mism += 1
    return mism


def _print_tick_event(lines, ts, event_id, a, b, rtol, atol):
    metrics = [
        "p_cex","ps_pre","ps_post","oracle_pre","oracle_post",
        "vp_pre","vp_post","xcp_profit_pre","xcp_profit_post",
    ]
    mism = 0
    if a is None or b is None:
        _print_table(lines, "tweak", event_id, a, b, metrics, rtol, atol)
        return 1
    failing = False
    for m in metrics:
        if not isclose(a.get(m), b.get(m), rtol, atol):
            failing = True
    _print_table(lines, "tweak", event_id, a, b, metrics, rtol, atol)
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
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--limit", default="30", help="N events or 'dyn' to stop at first structural timestamp difference")
    ap.add_argument("--ex-only", action="store_true", help="Only output exchange events; suppress tweak_price events")
    ap.add_argument("--plot-price", action="store_true", help="Show price trajectories (new_sim/mich_sim price_scale and p_cex); no save")
    ap.add_argument("--plot-stride", type=int, default=0, help="Downsample stride for plotting (0=auto ~2000 points)")
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
    print(f"  exchanges: new_sim={len(arb_ex)} mich_sim={len(jsonl_ex)}")
    print(f"  tweaks   : new_sim={len(arb_tk)} mich_sim={len(jsonl_tk)}")
    print(f"  unified  : timestamps={uni_stats['timestamps']} events_printed={uni_stats['printed']} mismatches={uni_stats['mismatches']}")

    if uni_lines:
        title = "Unified flow (until first differing timestamp):" if dyn else "Unified flow (first N events):"
        print("\n" + title)
        for line in uni_lines:
            print("  " + line)

    if args.plot_price:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"\nPlotting unavailable (matplotlib import failed): {e}")
            return 0
        # Determine plotting cutoff based on --limit and --limit dyn semantics
        ts_cutoff = None
        try:
            a_ex_g = _group_by_ts(arb_ex)
            a_tk_g = _group_by_ts(arb_tk)
            b_ex_g = _group_by_ts(jsonl_ex)
            b_tk_g = _group_by_ts(jsonl_tk)
            all_ts = sorted(set(a_ex_g.keys()) | set(a_tk_g.keys()) | set(b_ex_g.keys()) | set(b_tk_g.keys()))
            if dyn:
                for ts in all_ts:
                    la_ex = a_ex_g.get(ts, [])
                    lb_ex = b_ex_g.get(ts, [])
                    la_tk = a_tk_g.get(ts, [])
                    lb_tk = b_tk_g.get(ts, [])
                    structural_diff_ex = (len(la_ex) != len(lb_ex))
                    structural_diff_tk = (len(la_tk) != len(lb_tk))
                    if structural_diff_ex or (not args.ex_only and structural_diff_tk):
                        ts_cutoff = ts
                        break
            elif limit_val > 0:
                remaining = limit_val
                for ts in all_ts:
                    ex_n = max(len(a_ex_g.get(ts, [])), len(b_ex_g.get(ts, [])))
                    tk_n = 0 if args.ex_only else max(len(a_tk_g.get(ts, [])), len(b_tk_g.get(ts, [])))
                    need = ex_n + tk_n
                    if need <= 0:
                        continue
                    remaining -= need
                    ts_cutoff = ts
                    if remaining <= 0:
                        break
        except Exception:
            ts_cutoff = None

        def _ps_series(ex, tk):
            m = {}
            for e in ex:
                v = e.get("ps_post") if e.get("ps_post") is not None else e.get("ps_pre")
                if v is not None:
                    t = int(e["ts"])
                    if ts_cutoff is None or t <= ts_cutoff:
                        m[t] = float(v)
            for e in tk:
                v = e.get("ps_post") if e.get("ps_post") is not None else e.get("ps_pre")
                if v is not None:
                    t = int(e["ts"])
                    if ts_cutoff is None or t <= ts_cutoff:
                        m[t] = float(v)
            ts_sorted = sorted(m.keys())
            return ts_sorted, [m[t] for t in ts_sorted]

        def _pcex_series(ex, tk):
            m = {}
            for e in ex:
                if e.get("p_cex") is not None:
                    t = int(e["ts"])
                    if ts_cutoff is None or t <= ts_cutoff:
                        m[t] = float(e["p_cex"])
            for e in tk:
                if e.get("p_cex") is not None:
                    t = int(e["ts"])
                    if ts_cutoff is None or t <= ts_cutoff:
                        m[t] = float(e["p_cex"])
            ts_sorted = sorted(m.keys())
            return ts_sorted, [m[t] for t in ts_sorted]

        arb_ts, arb_ps = _ps_series(arb_ex, arb_tk)
        js_ts, js_ps   = _ps_series(jsonl_ex, jsonl_tk)
        pc_ts, pc_vals = _pcex_series(arb_ex if arb_ex or arb_tk else jsonl_ex, arb_tk if arb_ex or arb_tk else jsonl_tk)

        def _stride(ts, vs, stride):
            if not ts:
                return ts, vs
            if stride <= 0:
                # Auto target ~2000 points
                stride = max(1, len(ts) // 2000)
            return ts[::stride], vs[::stride]

        stride = int(args.plot_stride)
        arb_ts, arb_ps = _stride(arb_ts, arb_ps, stride)
        js_ts, js_ps   = _stride(js_ts, js_ps, stride)
        pc_ts, pc_vals = _stride(pc_ts, pc_vals, stride)

        # Single plot for price_scale and p_cex
        plt.figure(figsize=(10, 5))
        if arb_ts:
            plt.plot(arb_ts, arb_ps, label="new_sim price_scale", linewidth=1.2)
        if js_ts:
            plt.plot(js_ts, js_ps, label="mich_sim price_scale", linewidth=1.2)
        if pc_ts:
            plt.plot(pc_ts, pc_vals, label="p_cex", linewidth=1.0, alpha=0.8)
        plt.xlabel("timestamp (s)")
        plt.ylabel("price")
        plt.title("Price trajectories")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
