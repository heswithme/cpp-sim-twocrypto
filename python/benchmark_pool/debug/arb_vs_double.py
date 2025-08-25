#!/usr/bin/env python3
"""
Compare final state of arb_sim (arb_harness) vs cpp-double benchmark replay.

Defaults:
- Picks latest arb_run_*.json under python/arb_sim/run_data.
- Picks latest run_* under python/benchmark_pool/data/results and uses cpp_d_combined.json.

Usage:
  uv run python/benchmark_pool/debug/arb_vs_double.py            # defaults
  uv run python/benchmark_pool/debug/arb_vs_double.py --arb <arb_run.json> --run-dir <run_dir>
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict

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
    runs = sorted([p for p in rd.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not runs:
        raise SystemExit(f"No run_* folders found under {rd}")
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
    ap = argparse.ArgumentParser(description="Compare arb_sim final state vs cpp-double final state")
    ap.add_argument("--arb", type=str, default=None, help="Path to arb_run_*.json (default: latest)")
    ap.add_argument("--run-dir", type=str, default=None, help="Benchmark run dir (default: latest run_*)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    arb_path = Path(args.arb) if args.arb else _latest_arb_run(repo_root)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(repo_root)

    arb = _load(arb_path)
    cpp = _load(run_dir / "cpp_d_combined.json")

    # Expect single actions-carrying run; pick the last with actions, else the last
    runs = arb.get("runs", [])
    if not runs:
        raise SystemExit("No runs[] in arb_run JSON")
    with_actions = [r for r in runs if r.get("actions")]
    rr = with_actions[-1] if with_actions else runs[-1]

    a_final = rr.get("final_state", {})
    pool_name = None
    # Try to infer pool_name from benchmark pools.json
    inputs = _load(run_dir / "inputs_pools.json")
    if inputs.get("pools"):
        pool_name = inputs["pools"][0].get("name")

    # Find the matching cpp entry
    cpp_results = cpp.get("results", [])
    cpp_final = None
    for t in cpp_results:
        key = t.get("pool_config") or t.get("pool_name")
        if pool_name and key != pool_name:
            continue
        res = t.get("result", {})
        states = res.get("states")
        if states:
            cpp_final = states[-1]
        else:
            cpp_final = res.get("final_state")
        if cpp_final is not None:
            break

    if cpp_final is None:
        raise SystemExit("Could not locate cpp-double final state in run dir")

    print(f"arb_run: {arb_path}")
    print(f"run_dir: {run_dir}")
    print(f"pool: {pool_name}")

    mismatches = []
    for k in KEYS:
        if k in a_final and k in cpp_final:
            av = _norm(a_final[k])
            cv = _norm(cpp_final[k])
            if av != cv:
                mismatches.append((k, av, cv))

    if not mismatches:
        print("\n✓ Final states match across tracked keys.")
        return 0

    print("\nFinal state mismatches:")
    for k, av, cv in mismatches:
        print(f"  {k}:")
        print(f"    arb : {av}")
        print(f"    cppd: {cv}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

