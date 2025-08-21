#!/usr/bin/env python3
"""
Run C++ variants (integer vs double) on the same dataset and summarize timings.

Usage:
  uv run benchmark_pool/run_cpp_variants.py [--pools-file FILE] [--sequences-file FILE]
                                           [--n-cpp N] [--final-only | --snapshot-every N]

Writes a timestamped folder under python/benchmark_pool/data/results/run_cpp_variants_<UTC> containing:
  - cpp_i_combined.json
  - cpp_d_combined.json
  - summary.json (timings and basic counts)
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Repo-relative imports
HERE = Path(__file__).resolve()
PYTHON_DIR = HERE.parent.parent  # repo/python
REPO_ROOT = PYTHON_DIR.parent

# Ensure parent (python/) is importable
import sys as _sys, os as _os
_sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from cpp_pool.cpp_pool_runner import run_cpp_pool as run_cpp


def _write(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _extract_states(results: Dict[str, Any]) -> Dict[str, Any]:
    states: Dict[str, Any] = {}
    for test in results.get("results", []):
        key = test.get("pool_config") or test.get("pool_name")
        res = test.get("result", {})
        s = res.get("states")
        if not s:
            final = res.get("final_state")
            s = [final] if final is not None else []
        states[key] = s
    return states


def main() -> int:
    ap = argparse.ArgumentParser(description="Run C++ variants (i vs d) over the same dataset")
    ap.add_argument("--pools-file", default=str(PYTHON_DIR / "benchmark_pool" / "data" / "pools.json"), help="Path to pools.json")
    ap.add_argument("--sequences-file", default=str(PYTHON_DIR / "benchmark_pool" / "data" / "sequences.json"), help="Path to sequences.json")
    ap.add_argument("--n-cpp", type=int, default=0, help="C++ threads per process (CPP_THREADS)")
    ap.add_argument("--final-only", action="store_true", help="Only save final state per test (set SAVE_LAST_ONLY=1)")
    ap.add_argument("--snapshot-every", type=int, default=None, help="Snapshot every N actions (0=final only, 1=every, N=interval). Overrides --final-only")
    args = ap.parse_args()

    pools_file = Path(args.pools_file).resolve()
    sequences_file = Path(args.sequences_file).resolve()
    if not pools_file.exists() or not sequences_file.exists():
        print("❌ Input not found. Generate data with: uv run benchmark_pool/generate_data.py")
        return 1

    # Prepare run dir
    results_dir = PYTHON_DIR / "benchmark_pool" / "data" / "results"
    run_dir = results_dir / f"run_cpp_variants_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure env
    prev_threads = os.environ.get("CPP_THREADS")
    prev_last = os.environ.get("SAVE_LAST_ONLY")
    prev_every = os.environ.get("SNAPSHOT_EVERY")
    try:
        if args.n_cpp > 0:
            os.environ["CPP_THREADS"] = str(args.n_cpp)
        if args.snapshot_every is not None:
            os.environ["SNAPSHOT_EVERY"] = str(args.snapshot_every)
            os.environ.pop("SAVE_LAST_ONLY", None)
        elif args.final_only:
            os.environ["SAVE_LAST_ONLY"] = "1"

        # Run integer
        print("\n=== C++ integer (uint256) ===")
        out_i = run_dir / "cpp_i_combined.json"
        res_i = run_cpp("i", str(pools_file), str(sequences_file), str(out_i))
        i_time = res_i.get("metadata", {}).get("harness_time_s")

        # Run double
        print("\n=== C++ double ===")
        out_d = run_dir / "cpp_d_combined.json"
        res_d = run_cpp("d", str(pools_file), str(sequences_file), str(out_d))
        d_time = res_d.get("metadata", {}).get("harness_time_s")

        # Compute final-state differences for key metrics (percent and absolute wei)
        def _to_int(x: Any) -> int:
            try:
                return int(x)
            except Exception:
                try:
                    return int(float(x))
                except Exception:
                    return 0

        def _rel_err_pct(a: int, b: int) -> float:
            if b == 0:
                return 0.0 if a == 0 else float('inf')
            return abs(a - b) * 100.0 / abs(b)

        I_states = _extract_states(res_i)
        D_states = _extract_states(res_d)
        metrics = ["balances", "D", "virtual_price", "totalSupply", "price_scale"]
        final_rel_errors: Dict[str, Dict[str, Any]] = {}
        final_abs_errors: Dict[str, Dict[str, Any]] = {}
        agg_stats: Dict[str, Dict[str, float]] = {m: {"count": 0, "max_rel_pct": 0.0, "sum_rel_pct": 0.0} for m in metrics}

        for pool, i_states in I_states.items():
            d_states = D_states.get(pool)
            if not d_states:
                continue
            i_final = i_states[-1] if isinstance(i_states, list) else i_states
            d_final = d_states[-1] if isinstance(d_states, list) else d_states
            per_metric: Dict[str, Any] = {}
            for m in metrics:
                if m not in i_final or m not in d_final:
                    continue
                iv = i_final[m]
                dv = d_final[m]
                if isinstance(iv, list) and isinstance(dv, list):
                    ival = [_to_int(x) for x in iv]
                    dval = [_to_int(x) for x in dv]
                    errs_rel = [_rel_err_pct(dval[k], ival[k]) for k in range(min(len(ival), len(dval)))]
                    errs_abs = [abs(dval[k] - ival[k]) for k in range(min(len(ival), len(dval)))]
                    per_metric[m] = errs_rel
                    final_abs_errors.setdefault(pool, {})[m] = errs_abs
                    # aggregate across elements
                    for e in errs_rel:
                        agg_stats[m]["count"] += 1
                        agg_stats[m]["sum_rel_pct"] += (0.0 if e == float('inf') else e)
                        if e > agg_stats[m]["max_rel_pct"]:
                            agg_stats[m]["max_rel_pct"] = e
                else:
                    ivn = _to_int(iv)
                    dvn = _to_int(dv)
                    err_rel = _rel_err_pct(dvn, ivn)
                    err_abs = abs(dvn - ivn)
                    per_metric[m] = err_rel
                    final_abs_errors.setdefault(pool, {})[m] = err_abs
                    agg_stats[m]["count"] += 1
                    agg_stats[m]["sum_rel_pct"] += (0.0 if err_rel == float('inf') else err_rel)
                    if err_rel > agg_stats[m]["max_rel_pct"]:
                        agg_stats[m]["max_rel_pct"] = err_rel
            final_rel_errors[pool] = per_metric

        # Finalize aggregated stats
        for m in metrics:
            st = agg_stats[m]
            cnt = max(st["count"], 1)
            st["mean_rel_pct"] = st["sum_rel_pct"] / cnt
            st.pop("sum_rel_pct", None)

        _write(run_dir / "final_rel_errors.json", final_rel_errors)
        _write(run_dir / "final_abs_errors.json", final_abs_errors)
        _write(run_dir / "final_rel_stats.json", agg_stats)

        # Summary
        summary = {
            "i_time_s": i_time,
            "d_time_s": d_time,
            "speedup_x": (i_time / d_time) if (i_time and d_time and d_time > 0) else None,
            "tests": len(res_i.get("results", [])),
        }
        _write(run_dir / "summary.json", summary)
        print("\n=== Summary ===")
        print(json.dumps(summary, indent=2))

        # Print concise per-pool final-state relative and absolute errors
        print("\n=== Final-state differences (double vs integer) ===")
        for pool in sorted(final_rel_errors.keys()):
            per_metric = final_rel_errors[pool]
            abs_metric = final_abs_errors.get(pool, {})
            vp = per_metric.get("virtual_price"); vp_abs = abs_metric.get("virtual_price")
            ps = per_metric.get("price_scale"); ps_abs = abs_metric.get("price_scale")
            Drel = per_metric.get("D"); Dabs = abs_metric.get("D")
            ts = per_metric.get("totalSupply"); ts_abs = abs_metric.get("totalSupply")
            bal = per_metric.get("balances"); bal_abs = abs_metric.get("balances")
            # Format relative errors as fraction (ratio) in scientific notation (not tied to specific unit)
            def _to_ratio(val: float) -> float:
                if val == float('inf'):
                    return float('inf')
                # final_rel_errors stores percent; convert to ratio
                return val / 100.0
            def fmt_ratio(x: Any) -> str:
                if isinstance(x, list):
                    return "[" + ", ".join("inf" if v == float('inf') else f"{_to_ratio(v):.3e}" for v in x) + "]"
                if isinstance(x, float):
                    return "inf" if x == float('inf') else f"{_to_ratio(x):.3e}"
                return str(x)
            def fmt_abs(x: Any) -> str:
                if isinstance(x, list):
                    return "[" + ", ".join(f"{v:.3e}" for v in x) + "]"
                try:
                    return f"{float(x):.3e}"
                except Exception:
                    return str(x)
            print(f"- {pool}:")
            print(f"    virtual_price: rel={fmt_ratio(vp)} abs={fmt_abs(vp_abs)}")
            print(f"    price_scale  : rel={fmt_ratio(ps)} abs={fmt_abs(ps_abs)}")
            print(f"    D            : rel={fmt_ratio(Drel)} abs={fmt_abs(Dabs)}")
            print(f"    totalSupply  : rel={fmt_ratio(ts)} abs={fmt_abs(ts_abs)}")
            print(f"    balances     : rel={fmt_ratio(bal)} abs={fmt_abs(bal_abs)}")

        # (No aggregated stats printed; focus on last final pool state only.)
        print(f"\n✓ Results saved to {run_dir}")
        return 0
    finally:
        # Restore env
        if prev_threads is None:
            os.environ.pop("CPP_THREADS", None)
        else:
            os.environ["CPP_THREADS"] = prev_threads
        if prev_every is None:
            os.environ.pop("SNAPSHOT_EVERY", None)
        else:
            os.environ["SNAPSHOT_EVERY"] = prev_every
        if prev_last is None:
            os.environ.pop("SAVE_LAST_ONLY", None)
        else:
            os.environ["SAVE_LAST_ONLY"] = prev_last


if __name__ == "__main__":
    raise SystemExit(main())
