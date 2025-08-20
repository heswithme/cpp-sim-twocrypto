#!/usr/bin/env python3
"""
Run C++ variants benchmark (uint256 vs float vs double) on the same dataset,
report timing and final-state precision deltas.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpp_pool.cpp_pool_runner_i import run_cpp_pool
from cpp_pool.cpp_pool_runner_d import run_cpp_pool_double


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _extract_states(results: Dict[str, Any]) -> Dict[str, Any]:
    states = {}
    for test in results.get("results", []):
        key = test.get("pool_config") or test.get("pool_name") or "unknown"
        res = test.get("result", {})
        s = res.get("states")
        if not s:
            final = res.get("final_state")
            s = [final] if final is not None else []
        states[key] = s
    return states


def _run_one(harness: str, pools_file: str, sequences_file: str, out_path: str, cpp_threads: int, final_only: bool, snapshot_every: int | None) -> Tuple[float, Dict[str, Any]]:
    # Configure worker threads via env
    prev_threads = os.environ.get("CPP_THREADS")
    try:
        if cpp_threads > 0:
            os.environ["CPP_THREADS"] = str(cpp_threads)
        # Optionally save only final state
        prev_final = os.environ.get("SAVE_LAST_ONLY")
        if snapshot_every is not None:
            os.environ["SNAPSHOT_EVERY"] = str(snapshot_every)
        elif final_only:
            os.environ["SAVE_LAST_ONLY"] = "1"
        # Measure harness time as reported by runner metadata
        if harness == "i":
            results = run_cpp_pool(pools_file, sequences_file, out_path)
        elif harness == "d":
            results = run_cpp_pool_double(pools_file, sequences_file, out_path)
        else:
            raise ValueError(f"Unknown harness: {harness}")
        elapsed = 0.0
        try:
            elapsed = float(results.get("metadata", {}).get("harness_time_s", 0.0))
        except Exception:
            pass
        return elapsed, results
    finally:
        if prev_threads is None:
            os.environ.pop("CPP_THREADS", None)
        else:
            os.environ["CPP_THREADS"] = prev_threads
        if prev_final is None:
            os.environ.pop("SAVE_LAST_ONLY", None)
        else:
            os.environ["SAVE_LAST_ONLY"] = prev_final
        # clean snapshot interval
        if snapshot_every is not None:
            os.environ.pop("SNAPSHOT_EVERY", None)


def _abs_int(v: Any) -> int:
    try:
        return abs(int(v))
    except Exception:
        return 0


def _diff_final_states(baseline: Dict[str, Any], other: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculate both absolute and relative differences."""
    abs_diffs: Dict[str, Any] = {}
    rel_diffs: Dict[str, Any] = {}
    for key, base_states in baseline.items():
        o_states = other.get(key)
        if not o_states:
            continue
        b_final = base_states[-1] if isinstance(base_states, list) else base_states
        o_final = o_states[-1] if isinstance(o_states, list) else o_states
        metrics = ["balances", "D", "virtual_price", "totalSupply", "price_scale"]
        abs_d = {}
        rel_d = {}
        for m in metrics:
            if m in b_final and m in o_final:
                bv = b_final[m]
                ov = o_final[m]
                if isinstance(bv, list) and isinstance(ov, list):
                    abs_d[m] = [abs(int(ov[i]) - int(bv[i])) for i in range(min(len(bv), len(ov)))]
                    rel_d[m] = []
                    for i in range(min(len(bv), len(ov))):
                        base_val = int(bv[i])
                        other_val = int(ov[i])
                        if base_val != 0:
                            rel_pct = abs(other_val - base_val) * 100.0 / abs(base_val)
                            rel_d[m].append(rel_pct)
                        else:
                            rel_d[m].append(0.0 if other_val == 0 else float('inf'))
                else:
                    try:
                        base_val = int(bv)
                        other_val = int(ov)
                        abs_d[m] = abs(other_val - base_val)
                        if base_val != 0:
                            rel_d[m] = abs(other_val - base_val) * 100.0 / abs(base_val)
                        else:
                            rel_d[m] = 0.0 if other_val == 0 else float('inf')
                    except Exception:
                        abs_d[m] = None
                        rel_d[m] = None
        abs_diffs[key] = abs_d
        rel_diffs[key] = rel_d
    return abs_diffs, rel_diffs


def _summarize_diffs(diffs: Dict[str, Any]) -> Dict[str, Any]:
    # Aggregate max over tests per metric
    agg: Dict[str, Any] = {}
    for test, d in diffs.items():
        for m, val in d.items():
            if isinstance(val, list):
                cur = agg.get(m, [0] * len(val))
                if len(cur) < len(val):
                    cur = cur + [0] * (len(val) - len(cur))
                agg[m] = [max(cur[i], val[i]) if val[i] != float('inf') else float('inf') for i in range(len(val))]
            else:
                if val == float('inf'):
                    agg[m] = float('inf')
                else:
                    agg[m] = max(agg.get(m, 0), val or 0)
    return agg


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare C++ integer vs double harnesses")
    ap.add_argument("--pools-file", default=str(Path(__file__).with_name("data") / "pools.json"))
    ap.add_argument("--sequences-file", default=str(Path(__file__).with_name("data") / "sequences.json"))
    ap.add_argument("--n-cpp", type=int, default=0, help="CPP_THREADS per process (0 = auto)")
    ap.add_argument("--final-only", action="store_true", help="Only save final state per test (set SAVE_LAST_ONLY=1)")
    ap.add_argument("--snapshot-every", type=int, default=None, help="Snapshot every N actions (0=final only, 1=every, N=interval). Overrides --final-only")
    ap.add_argument("--show-absolute", action="store_true", help="Show absolute differences instead of relative")
    args = ap.parse_args()

    pools_file = args.pools_file
    sequences_file = args.sequences_file
    if not os.path.exists(pools_file) or not os.path.exists(sequences_file):
        print("❌ Input not found. Generate data with: uv run python/benchmark_pool/generate_data.py")
        return 1

    # Run directory
    base_results_dir = Path(__file__).with_name("data").joinpath("results")
    base_results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("run_cpp_variants_%Y%m%dT%H%M%SZ")
    run_dir = base_results_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional pool filter via env (honored by harnesses)
    try:
        timing: Dict[str, float] = {}
        outputs: Dict[str, Dict[str, Any]] = {}

        # integer baseline
        out_i = run_dir / "cpp_i.json"
        t_i, R_i = _run_one("i", pools_file, sequences_file, str(out_i), args.n_cpp, args.final_only, args.snapshot_every)
        timing["i"] = t_i
        outputs["i"] = R_i

        # double
        out_d = run_dir / "cpp_d.json"
        t_d, R_d = _run_one("d", pools_file, sequences_file, str(out_d), args.n_cpp, args.final_only, args.snapshot_every)
        timing["d"] = t_d
        outputs["d"] = R_d

        # Extract states per variant
        states_i = _extract_states(R_i)
        states_d = _extract_states(R_d)

        # Final diffs relative to integer baseline
        abs_diff_d, rel_diff_d = _diff_final_states(states_i, states_d)
        abs_agg_d = _summarize_diffs(abs_diff_d)
        rel_agg_d = _summarize_diffs(rel_diff_d)

        summary = {
            "timing_s": timing,
            "speedup_vs_i": {
                "d": (timing["i"] / timing["d"]) if timing["d"] > 0 else None,
            },
            "final_abs_diffs_vs_i": {
                "per_pool": {
                    "d": abs_diff_d,
                },
                "aggregate_max": {
                    "d": abs_agg_d,
                },
            },
            "final_rel_diffs_vs_i": {
                "per_pool": {
                    "d": rel_diff_d,
                },
                "aggregate_max": {
                    "d": rel_agg_d,
                },
            },
        }

        _write_json(str(run_dir / "summary.json"), summary)
        _write_json(str(run_dir / "cpp_i_combined.json"), outputs["i"]) 
        _write_json(str(run_dir / "cpp_d_combined.json"), outputs["d"]) 

        # Console summary
        print("\n=== C++ Variants Timing (s) ===")
        for k in ("i", "d"):
            print(f"  {k:6s}: {timing[k]:.3f}s")
        print("\nSpeedup vs integer:")
        print(f"  double: {summary['speedup_vs_i']['d']:.2f}x")

        if args.show_absolute:
            print("\nMax absolute differences vs integer:")
            for k, v in summary["final_abs_diffs_vs_i"]["aggregate_max"]["d"].items():
                print(f"  double {k}: {v}")
        else:
            print("\nMax relative differences vs integer (% error):")
            for k, v in summary["final_rel_diffs_vs_i"]["aggregate_max"]["d"].items():
                if isinstance(v, list):
                    formatted = [f"{x:.6f}%" if x != float('inf') else "inf" for x in v]
                    print(f"  double {k}: {formatted}")
                else:
                    if v == float('inf'):
                        print(f"  double {k}: inf")
                    else:
                        print(f"  double {k}: {v:.6f}%")

        print(f"\n✓ Results saved to {run_dir}")
        return 0
    finally:
        pass


if __name__ == "__main__":
    sys.exit(main())
