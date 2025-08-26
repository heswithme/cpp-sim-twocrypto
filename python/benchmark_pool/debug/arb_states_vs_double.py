#!/usr/bin/env python3
"""
Compare per-action states from arb_sim (if present) vs cpp-double replay.

Prereq: Run arb_sim with --save-actions after updating the harness to emit `states`.

Usage:
  python3 python/benchmark_pool/debug/arb_states_vs_double.py \
    [--arb <arb_run.json>] [--run-dir <run_cpp_variants_*>]
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List

KEYS = [
    "balances", "xp", "D", "virtual_price", "xcp_profit",
    "price_scale", "price_oracle", "last_prices", "totalSupply",
    "timestamp",
]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _latest_arb_run(repo_root: Path) -> Path:
    rd = repo_root / "python" / "arb_sim" / "run_data"
    files = sorted([p for p in rd.glob("arb_run_*.json")])
    if not files:
        raise SystemExit(f"No arb_run_*.json found under {rd}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _latest_run_dir(repo_root: Path) -> Path:
    rd = repo_root / "python" / "benchmark_pool" / "data" / "results"
    runs = sorted([p for p in rd.iterdir() if p.is_dir() and p.name.startswith("run_cpp_variants_")])
    if not runs:
        raise SystemExit(f"No run_cpp_variants_* folders found under {rd}")
    return runs[-1]


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def _norm(v: Any) -> Any:
    if isinstance(v, list):
        return [_to_int(x) for x in v]
    return _to_int(v)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Diff per-action arb states vs cpp-double states")
    ap.add_argument("--arb", type=str, default=None, help="Path to arb_run_*.json (default: latest)")
    ap.add_argument("--run-dir", type=str, default=None, help="Benchmark run dir (default: latest run_cpp_variants_*)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    arb_path = Path(args.arb) if args.arb else _latest_arb_run(repo_root)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(repo_root)

    arb = _load(arb_path)
    cpp = _load(run_dir / "cpp_d_combined.json")

    # Selected run (with states)
    runs = arb.get("runs", [])
    if not runs:
        raise SystemExit("No runs[] in arb_run JSON")
    rr = runs[-1]
    a_states: List[Dict[str, Any]] = rr.get("states") or []
    if not a_states:
        raise SystemExit("Selected run has no states[]; rebuild arb_harness and re-run arb_sim with --save-actions")

    # Find matching cpp states
    inputs = _load(run_dir / "inputs_pools.json")
    pool_name = inputs.get("pools", [{}])[0].get("name")
    cpp_results = cpp.get("results", [])
    d_states = None
    for t in cpp_results:
        key = t.get("pool_config") or t.get("pool_name")
        if pool_name and key != pool_name:
            continue
        res = t.get("result", {})
        st = res.get("states")
        if st:
            d_states = st
            break
    if d_states is None:
        raise SystemExit("No cpp-double states[] found (ensure run_cpp_variants was not final-only)")

    print(f"arb_run: {arb_path}")
    print(f"run_dir: {run_dir}")
    print(f"pool: {pool_name}")
    # Build an arb state stream aligned to the action list
    seq = json.loads((repo_root / "python" / "benchmark_pool" / "data" / "sequences.json").read_text())
    actions = seq["sequences"][0]["actions"]

    aligned: List[Dict[str, Any]] = []
    ai = 0  # index into arb states
    # push initial state
    if a_states:
        aligned.append(a_states[0])
    # helper to get ts
    def ts_of(st: Dict[str, Any]) -> int:
        return int(st.get("timestamp") or 0)
    def balances_of(st: Dict[str, Any]):
        return _norm(st.get("balances"))

    last = a_states[0] if a_states else {}
    for act in actions:
        t = act.get("timestamp") if act.get("type") == "time_travel" else None
        # advance to the timestamp boundary for this action (if provided)
        if t is not None:
            t = int(t)
            while ai < len(a_states) and ts_of(a_states[ai]) < t:
                ai += 1
            # pick first state at or after t
            while ai < len(a_states) and ts_of(a_states[ai]) == t and balances_of(a_states[ai]) == balances_of(last):
                # prefer the earliest snapshot at this timestamp (time update), not after donation/trade
                aligned.append(a_states[ai])
                last = a_states[ai]
                ai += 1
                break
            else:
                # if none found exactly at t, accept the next state >= t
                if ai < len(a_states):
                    aligned.append(a_states[ai])
                    last = a_states[ai]
                    ai += 1
            continue
        # For mutation actions, pick the next state with a different balance at current timestamp
        cur_ts = ts_of(last)
        while ai < len(a_states) and ts_of(a_states[ai]) == cur_ts and balances_of(a_states[ai]) == balances_of(last):
            ai += 1
        if ai < len(a_states):
            aligned.append(a_states[ai])
            last = a_states[ai]
            ai += 1

    print(f"lengths: arb={len(aligned)} cppd={len(d_states)} (aligned vs cpp)")

    n = min(len(aligned), len(d_states))
    for idx in range(n):
        diffs = []
        for k in KEYS:
            if k in aligned[idx] and k in d_states[idx]:
                av = _norm(aligned[idx][k])
                dv = _norm(d_states[idx][k])
                if av != dv:
                    diffs.append((k, av, dv))
        if diffs:
            print(f"\nFirst divergence at step {idx}")
            for k, av, dv in diffs:
                print(f"  {k}:")
                print(f"    arb : {av}")
                print(f"    cppd: {dv}")
            # Action context
            if 0 <= idx-1 < len(actions):
                print(f"  action[{idx-1}]: {actions[idx-1]}")
            return 1

    if len(aligned) != len(d_states):
        print(f"\nAligned length differs: arb_aligned={len(aligned)} cppd={len(d_states)}")
    else:
        print("\nNo divergence across tracked keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
