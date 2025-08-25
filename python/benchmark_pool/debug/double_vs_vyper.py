#!/usr/bin/env python3
"""
Compare cpp-double vs vyper snapshots for the latest (or given) full benchmark run.

Usage:
  uv run python/benchmark_pool/debug/double_vs_vyper.py            # latest run_*
  uv run python/benchmark_pool/debug/double_vs_vyper.py <run_dir>  # specific run folder
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

KEYS = [
    "balances", "xp", "D", "virtual_price", "xcp_profit",
    "price_scale", "price_oracle", "last_prices", "totalSupply",
    "timestamp",
]


def _find_latest_run(results_dir: Path) -> Path:
    runs = sorted([p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not runs:
        raise SystemExit(f"No run_* folders found under {results_dir}")
    return runs[-1]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _extract_states(results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for test in results.get("results", []):
        key = test.get("pool_config") or test.get("pool_name")
        res = test.get("result", {})
        st = res.get("states") or []
        if not st:
            final = res.get("final_state")
            st = [final] if final is not None else []
        out[key] = st
    return out


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def _first_divergence(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    n = min(len(a), len(b))
    for idx in range(n):
        diffs: Dict[str, Dict[str, Any]] = {}
        for k in KEYS:
            if k in a[idx] and k in b[idx]:
                av = a[idx][k]
                bv = b[idx][k]
                if isinstance(av, list) and isinstance(bv, list):
                    ai = [_to_int(x) for x in av]
                    bi = [_to_int(x) for x in bv]
                    if ai != bi:
                        diffs[k] = {"cpp_d": ai, "vyper": bi}
                else:
                    ai = _to_int(av)
                    bi = _to_int(bv)
                    if ai != bi:
                        diffs[k] = {"cpp_d": ai, "vyper": bi}
        if diffs:
            return idx, diffs
    return -1, {}


def main() -> int:
    import sys
    repo_root = Path(__file__).resolve().parents[3]
    results_root = repo_root / "python" / "benchmark_pool" / "data" / "results"
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _find_latest_run(results_root)

    cpp_path = run_dir / "cpp_d_combined.json"
    vy_path = run_dir / "vyper_combined.json"
    if not cpp_path.exists() or not vy_path.exists():
        raise SystemExit(f"Missing cpp_d_combined.json or vyper_combined.json in {run_dir}")

    C = _extract_states(_load(cpp_path))
    V = _extract_states(_load(vy_path))

    pools = sorted(set(C.keys()) & set(V.keys()))
    print(f"Inspecting run: {run_dir}")
    print(f"Pools: {len(pools)}")

    mismatches = 0
    for p in pools:
        cs = C[p]
        vs = V[p]
        step, diffs = _first_divergence(cs, vs)
        if step >= 0:
            mismatches += 1
            print(f"\nPool: {p}")
            print(f"First divergence at step {step}")
            for k in ("virtual_price", "price_scale", "last_prices", "D", "balances"):
                if k in diffs:
                    print(f"  {k}: cpp_d={diffs[k]['cpp_d']} vyper={diffs[k]['vyper']}")

    if mismatches == 0:
        print("\nAll pools match across tracked keys (cpp-double vs vyper).")
    else:
        print(f"\nSummary: pools with mismatches = {mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

