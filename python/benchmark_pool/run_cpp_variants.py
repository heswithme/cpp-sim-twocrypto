#!/usr/bin/env python3
"""
Run C++ variants (uint256, float, double, long double) on the same dataset and summarize timings.

Usage:
  uv run benchmark_pool/run_cpp_variants.py [--pools-file FILE] [--sequences-file FILE]
                                           [--n-cpp N] [--final-only | --snapshot-every N]

Writes a timestamped folder under python/benchmark_pool/data/results/run_cpp_variants_<UTC> containing:
  - cpp_i_combined.json
  - cpp_f_combined.json
  - cpp_d_combined.json
  - cpp_ld_combined.json
  - summary.json (timings and basic counts)
  - final_rel_errors_vs_i.json (per-variant vs integer baseline)
  - final_abs_errors_vs_i.json
  - final_rel_stats_vs_i.json
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


def _ensure_built_harness():
    """Build typed C++ harnesses once to avoid rebuilds when switching modes."""
    build_dir = (REPO_ROOT / "cpp" / "build").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    # Configure if needed (Release)
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        import subprocess
        subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=str(build_dir), check=True)
    # Build both typed harnesses
    import subprocess
    subprocess.run(["cmake", "--build", ".", "--target", "benchmark_harness_i", "benchmark_harness_d", "benchmark_harness_f", "benchmark_harness_ld"], cwd=str(build_dir), check=True)


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

        # Save copies of inputs for downstream tools (e.g., arb_vs_double)
        try:
            with pools_file.open("r") as f:
                pools_obj = json.load(f)
            with sequences_file.open("r") as f:
                seqs_obj = json.load(f)
            with (run_dir / "inputs_pools.json").open("w") as f:
                json.dump(pools_obj, f, indent=2)
            with (run_dir / "inputs_sequences.json").open("w") as f:
                json.dump(seqs_obj, f, indent=2)
        except Exception as e:
            print(f"⚠ Failed to copy inputs into run dir: {e}")

        # Pre-build typed harnesses to avoid sequential rebuilds
        try:
            _ensure_built_harness()
        except Exception as e:
            print(f"⚠ Failed to prebuild harnesses: {e}. Proceeding anyway.")

        # Run integer
        print("\n=== C++ integer (uint256) ===")
        out_i = run_dir / "cpp_i_combined.json"
        res_i = run_cpp("i", str(pools_file), str(sequences_file), str(out_i))
        i_time = res_i.get("metadata", {}).get("harness_time_s")

        # Run float
        print("\n=== C++ float ===")
        out_f = run_dir / "cpp_f_combined.json"
        res_f = run_cpp("f", str(pools_file), str(sequences_file), str(out_f))
        f_time = res_f.get("metadata", {}).get("harness_time_s")

        # Run double
        print("\n=== C++ double ===")
        out_d = run_dir / "cpp_d_combined.json"
        res_d = run_cpp("d", str(pools_file), str(sequences_file), str(out_d))
        d_time = res_d.get("metadata", {}).get("harness_time_s")

        # Run long double
        print("\n=== C++ long double ===")
        out_ld = run_dir / "cpp_ld_combined.json"
        res_ld = run_cpp("ld", str(pools_file), str(sequences_file), str(out_ld))
        ld_time = res_ld.get("metadata", {}).get("harness_time_s")

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
        F_states = _extract_states(res_f)
        D_states = _extract_states(res_d)
        LD_states = _extract_states(res_ld)
        metrics = ["balances", "D", "virtual_price", "totalSupply", "price_scale"]

        def _diffs_vs_i(V_states: Dict[str, Any]):
            final_rel_errors: Dict[str, Dict[str, Any]] = {}
            final_abs_errors: Dict[str, Dict[str, Any]] = {}
            agg_stats: Dict[str, Dict[str, float]] = {m: {"count": 0, "max_rel_pct": 0.0, "sum_rel_pct": 0.0} for m in metrics}
            for pool, i_states in I_states.items():
                v_states = V_states.get(pool)
                if not v_states:
                    continue
                i_final = i_states[-1] if isinstance(i_states, list) else i_states
                v_final = v_states[-1] if isinstance(v_states, list) else v_states
                per_metric: Dict[str, Any] = {}
                for m in metrics:
                    if m not in i_final or m not in v_final:
                        continue
                    iv = i_final[m]
                    vv = v_final[m]
                    if isinstance(iv, list) and isinstance(vv, list):
                        ival = [_to_int(x) for x in iv]
                        vval = [_to_int(x) for x in vv]
                        errs_rel = [_rel_err_pct(vval[k], ival[k]) for k in range(min(len(ival), len(vval)))]
                        errs_abs = [abs(vval[k] - ival[k]) for k in range(min(len(ival), len(vval)))]
                        per_metric[m] = errs_rel
                        final_abs_errors.setdefault(pool, {})[m] = errs_abs
                        for e in errs_rel:
                            agg_stats[m]["count"] += 1
                            agg_stats[m]["sum_rel_pct"] += (0.0 if e == float('inf') else e)
                            if e > agg_stats[m]["max_rel_pct"]:
                                agg_stats[m]["max_rel_pct"] = e
                    else:
                        ivn = _to_int(iv)
                        vvn = _to_int(vv)
                        err_rel = _rel_err_pct(vvn, ivn)
                        err_abs = abs(vvn - ivn)
                        per_metric[m] = err_rel
                        final_abs_errors.setdefault(pool, {})[m] = err_abs
                        agg_stats[m]["count"] += 1
                        agg_stats[m]["sum_rel_pct"] += (0.0 if err_rel == float('inf') else err_rel)
                        if err_rel > agg_stats[m]["max_rel_pct"]:
                            agg_stats[m]["max_rel_pct"] = err_rel
                final_rel_errors[pool] = per_metric
            for m in metrics:
                st = agg_stats[m]
                cnt = max(st["count"], 1)
                st["mean_rel_pct"] = st["sum_rel_pct"] / cnt
                st.pop("sum_rel_pct", None)
            return final_rel_errors, final_abs_errors, agg_stats

        rel_f, abs_f, stats_f = _diffs_vs_i(F_states)
        rel_d, abs_d, stats_d = _diffs_vs_i(D_states)
        rel_ld, abs_ld, stats_ld = _diffs_vs_i(LD_states)

        _write(run_dir / "final_rel_errors_vs_i.json", {"f": rel_f, "d": rel_d, "ld": rel_ld})
        _write(run_dir / "final_abs_errors_vs_i.json", {"f": abs_f, "d": abs_d, "ld": abs_ld})
        _write(run_dir / "final_rel_stats_vs_i.json", {"f": stats_f, "d": stats_d, "ld": stats_ld})

        # Summary
        summary = {
            "i_time_s": i_time,
            "f_time_s": f_time,
            "d_time_s": d_time,
            "ld_time_s": ld_time,
            "speedup_vs_i": {
                "f": (i_time / f_time) if (i_time and f_time and f_time > 0) else None,
                "d": (i_time / d_time) if (i_time and d_time and d_time > 0) else None,
                "ld": (i_time / ld_time) if (i_time and ld_time and ld_time > 0) else None,
            },
            "tests": len(res_i.get("results", [])),
        }
        _write(run_dir / "summary.json", summary)
        print("\n=== Summary ===")
        print(json.dumps(summary, indent=2))

        # Print concise per-pool final-state relative errors vs integer baseline
        def _to_ratio(val: float) -> float:
            if val == float('inf'):
                return float('inf')
            return val / 100.0
        def fmt_ratio(x: Any) -> str:
            if isinstance(x, list):
                return "[" + ", ".join("inf" if v == float('inf') else f"{_to_ratio(v):.3e}" for v in x) + "]"
            if isinstance(x, float):
                return "inf" if x == float('inf') else f"{_to_ratio(x):.3e}"
            try:
                return f"{_to_ratio(float(x)):.3e}"
            except Exception:
                return str(x)
        def fmt_abs(x: Any) -> str:
            if isinstance(x, list):
                return "[" + ", ".join(f"{float(v):.3e}" for v in x) + "]"
            try:
                return f"{float(x):.3e}"
            except Exception:
                return str(x)

        def print_block(title: str, rel: Dict[str, Any], absd: Dict[str, Any]):
            print(f"\n=== Final-state differences vs integer ({title}) ===")
            for pool in sorted(rel.keys()):
                per_metric = rel[pool]
                abs_metric = absd.get(pool, {})
                print(f"- {pool}:")
                print(f"    virtual_price: rel={fmt_ratio(per_metric.get('virtual_price'))} abs={fmt_abs(abs_metric.get('virtual_price'))}")
                print(f"    price_scale  : rel={fmt_ratio(per_metric.get('price_scale'))} abs={fmt_abs(abs_metric.get('price_scale'))}")
                print(f"    D            : rel={fmt_ratio(per_metric.get('D'))} abs={fmt_abs(abs_metric.get('D'))}")
                print(f"    totalSupply  : rel={fmt_ratio(per_metric.get('totalSupply'))} abs={fmt_abs(abs_metric.get('totalSupply'))}")
                print(f"    balances     : rel={fmt_ratio(per_metric.get('balances'))} abs={fmt_abs(abs_metric.get('balances'))}")

        print_block("float", rel_f, abs_f)
        print_block("double", rel_d, abs_d)
        print_block("long double", rel_ld, abs_ld)

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
