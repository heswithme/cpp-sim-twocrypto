#!/usr/bin/env python3
"""
Parse the latest (or given) run directory and diff C++ vs Vyper states pool-by-pool.

Usage:
  uv run python/benchmark_pool/debug/parse_and_diff.py            # auto-detect latest run_*
  uv run python/benchmark_pool/debug/parse_and_diff.py <run_dir>  # specific run folder
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


KEYS_TO_CHECK = [
    "balances", "D", "virtual_price", "totalSupply", "price_scale",
    "price_oracle", "donation_shares", "donation_shares_unlocked",
    "last_donation_release_ts", "timestamp",
]


def find_latest_run_dir(results_dir: Path) -> Path:
    runs = sorted([p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not runs:
        raise SystemExit(f"No run_* folders found under {results_dir}")
    return runs[-1]


def load_combined(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load combined outputs, supporting both legacy and unified filenames.

    C++: prefer cpp_combined.json, fallback to cpp_i_combined.json, cpp_benchmark_results.json
    Vyper: prefer vyper_combined.json, fallback to vyper_benchmark_results.json
    """
    candidates_cpp = [
        run_dir / "cpp_combined.json",
        run_dir / "cpp_i_combined.json",
        run_dir / "cpp_benchmark_results.json",
    ]
    candidates_vy = [
        run_dir / "vyper_combined.json",
        run_dir / "vyper_benchmark_results.json",
    ]

    cpp_path = next((p for p in candidates_cpp if p.exists()), None)
    vy_path = next((p for p in candidates_vy if p.exists()), None)
    if not cpp_path or not vy_path:
        raise SystemExit(
            f"Missing combined outputs in {run_dir}. "
            f"Tried C++: {[p.name for p in candidates_cpp]} | "
            f"Vyper: {[p.name for p in candidates_vy]}"
        )

    cpp = json.loads(cpp_path.read_text())
    vy = json.loads(vy_path.read_text())
    return cpp, vy


def extract_states(results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for test in results.get("results", []):
        key = test.get("pool_config")
        out[key] = test.get("result", {})
    return out


def normalize(v: Any) -> Any:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, (int, float)):
        return str(v)
    return v


def first_divergence(cpp_states: List[Dict[str, Any]], vy_states: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    n = min(len(cpp_states), len(vy_states))
    for i in range(n):
        diffs: Dict[str, Dict[str, Any]] = {}
        for k in KEYS_TO_CHECK:
            cv = normalize(cpp_states[i].get(k))
            vv = normalize(vy_states[i].get(k))
            if cv != vv:
                diffs[k] = {"cpp": cv, "vy": vv}
        if diffs:
            return i, diffs
    return -1, {}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    default_results = repo_root / "python" / "benchmark_pool" / "data" / "results"

    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        run_dir = find_latest_run_dir(default_results)

    cpp_combined, vy_combined = load_combined(run_dir)
    cpp = extract_states(cpp_combined)
    vy = extract_states(vy_combined)

    cpp_keys = set(cpp.keys())
    vy_keys = set(vy.keys())
    all_keys = sorted(cpp_keys | vy_keys)

    print(f"Inspecting run: {run_dir}")
    print(f"Pools (C++): {len(cpp_keys)} | Pools (VY): {len(vy_keys)}")

    mismatches = 0
    missing = 0

    for key in all_keys:
        if key not in cpp:
            print(f"- Missing in C++: {key}")
            missing += 1
            continue
        if key not in vy:
            print(f"- Missing in Vyper: {key}")
            missing += 1
            continue

        c = cpp[key]
        v = vy[key]

        if c.get("error") or v.get("error"):
            print(f"- Error in results for {key}: cpp={c.get('error')} vy={v.get('error')}")
            mismatches += 1
            continue

        c_states = c.get("states", [])
        v_states = v.get("states", [])
        if not isinstance(c_states, list) or not isinstance(v_states, list):
            print(f"- Invalid states shape for {key}")
            mismatches += 1
            continue

        if len(c_states) != len(v_states):
            print(f"- Length mismatch for {key}: cpp={len(c_states)} vy={len(v_states)}")
            mismatches += 1
            continue

        idx, diffs = first_divergence(c_states, v_states)
        if idx >= 0:
            print(f"\nPool: {key}")
            print(f"First divergence at step {idx}")
            for k, dv in diffs.items():
                print(f"  {k}:")
                print(f"    cpp: {dv['cpp']}")
                print(f"    vy : {dv['vy']}")
            mismatches += 1

    if mismatches == 0 and missing == 0:
        print("\nAll pools match across checked keys.")
        return 0

    print(f"\nSummary: mismatches={mismatches}, missing_keys={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
