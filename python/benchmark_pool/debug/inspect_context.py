#!/usr/bin/env python3
"""
Inspect context around the first divergence between C++ and Vyper for a given run.

Usage:
  uv run python/benchmark_pool/debug/inspect_context.py <run_dir>
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

KEYS = [
    "balances", "xp", "D", "virtual_price", "xcp_profit", "price_scale",
    "price_oracle", "last_prices", "totalSupply", "donation_shares",
    "donation_shares_unlocked", "donation_protection_expiry_ts",
    "last_donation_release_ts", "timestamp",
]


def normalize(v: Any) -> Any:
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, (int, float)):
        return str(v)
    return v


def first_divergence(cs: List[Dict[str, Any]], vs: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    n = min(len(cs), len(vs))
    for i in range(n):
        diffs: Dict[str, Dict[str, Any]] = {}
        for k in KEYS:
            cv = normalize(cs[i].get(k))
            vv = normalize(vs[i].get(k))
            if cv != vv:
                diffs[k] = {"cpp": cv, "vy": vv}
        if diffs:
            return i, diffs
    return -1, {}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: inspect_context.py <run_dir>")
        return 1
    run_dir = Path(sys.argv[1]).resolve()
    # Support both unified and legacy naming
    cpp_candidates = [run_dir / "cpp_combined.json", run_dir / "cpp_i_combined.json", run_dir / "cpp_benchmark_results.json"]
    vy_candidates = [run_dir / "vyper_combined.json", run_dir / "vyper_benchmark_results.json"]
    cpp_path = next((p for p in cpp_candidates if p.exists()), None)
    vy_path = next((p for p in vy_candidates if p.exists()), None)
    # sequences.json lives under data/, which is two levels up from run_dir
    seqs_path = run_dir.parents[1] / "sequences.json"

    if not cpp_path or not vy_path:
        tried_cpp = ", ".join(p.name for p in cpp_candidates)
        tried_vy = ", ".join(p.name for p in vy_candidates)
        print(f"Missing combined outputs in {run_dir}. Tried C++: [{tried_cpp}] Vyper: [{tried_vy}]")
        return 1

    cpp = json.loads(cpp_path.read_text())
    vy = json.loads(vy_path.read_text())
    cpp_results = {t["pool_config"]: t["result"] for t in cpp.get("results", [])}
    vy_results = {t["pool_config"]: t["result"] for t in vy.get("results", [])}
    if not cpp_results:
        print("No C++ results found")
        return 1
    pool = next(iter(cpp_results.keys()))
    if pool not in vy_results:
        print(f"Pool {pool} missing on Vyper side")
        return 1

    cs = cpp_results[pool].get("states", [])
    vs = vy_results[pool].get("states", [])
    if not cs or not vs:
        print("Empty states")
        return 1

    step, diffs = first_divergence(cs, vs)
    if step < 0:
        print("No divergence found")
        return 0

    print(f"Pool: {pool}")
    print(f"First divergence at step {step} (action index {step-1})")

    if seqs_path.exists():
        try:
            seqs = json.loads(seqs_path.read_text())["sequences"]
            act = seqs[0]["actions"][step-1]
            print(f"Action[{step-1}]: {act}")
        except Exception:
            pass

    def dump(label: str, state: Dict[str, Any]):
        print(f"\n{label}:")
        for k in KEYS:
            if k in state:
                print(f"  {k}: {state[k]}")

    before = max(step - 1, 0)
    dump("C++ before", cs[before])
    dump("VY before", vs[before])
    dump("C++ at", cs[step])
    dump("VY at", vs[step])

    print("\nDiff keys at step:")
    for k in diffs:
        print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
