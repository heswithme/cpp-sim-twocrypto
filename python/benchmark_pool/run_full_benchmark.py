#!/usr/bin/env python3
"""
Run full benchmark comparing C++ and Vyper implementations (single pass per side).
"""
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpp_pool.cpp_pool_runner import run_cpp_pool as run_cpp_pool_mode
from vyper_pool.vyper_pool_runner import run_vyper_pool


def run_cpp_benchmark(pool_configs_file: str, sequences_file: str, output_dir: str) -> Dict:
    """Run C++ benchmark and save results."""
    print("\n=== Running C++ Benchmark ===")
    
    # Run benchmark
    cpp_output = os.path.join(output_dir, "cpp_benchmark_results.json")
    
    start_time = time.time()
    results = run_cpp_pool_mode("i", pool_configs_file, sequences_file, cpp_output)
    cpp_time = time.time() - start_time
    
    print(f"✓ C++ benchmark completed in {cpp_time:.2f}s")
    
    # Extract states for each test
    cpp_states = {}
    for test in results["results"]:
        key = f"{test['pool_config']}_{test['sequence']}"
        if test["result"]["success"]:
            st = test["result"].get("states")
            cpp_states[key] = st if st is not None else [test["result"].get("final_state")]
        else:
            cpp_states[key] = {"error": test["result"].get("error", "Failed")}
    
    return {
        "states": cpp_states,
        "time": cpp_time,
        "output_file": cpp_output
    }


def run_cpp_double_benchmark(pool_configs_file: str, sequences_file: str, output_dir: str) -> Dict:
    """Run C++ double benchmark and save results."""
    print("\n=== Running C++ Double Benchmark ===")

    cpp_output = os.path.join(output_dir, "cpp_double_benchmark_results.json")
    start_time = time.time()
    results = run_cpp_pool_mode("d", pool_configs_file, sequences_file, cpp_output)
    cpp_time = time.time() - start_time
    print(f"✓ C++ double benchmark completed in {cpp_time:.2f}s")

    cpp_states = {}
    for test in results["results"]:
        key = f"{test['pool_config']}_{test['sequence']}"
        if test["result"]["success"]:
            # accept either states or final_state
            st = test["result"].get("states")
            cpp_states[key] = st if st is not None else [test["result"].get("final_state")]
        else:
            cpp_states[key] = {"error": test["result"].get("error", "Failed")}

    return {
        "states": cpp_states,
        "time": cpp_time,
        "output_file": cpp_output
    }


def run_vyper_benchmark(pool_configs_file: str, sequences_file: str, output_dir: str, n_py: int = 1) -> Dict:
    """Run Vyper benchmark possibly with multiple worker processes and save results."""
    print("\n=== Running Vyper Benchmark ===")

    vyper_output = os.path.join(output_dir, "vyper_benchmark_results.json")
    start_time = time.time()

    if n_py <= 1:
        # Single-process path
        results = run_vyper_pool(pool_configs_file, sequences_file, vyper_output)
    else:
        # Multi-process sharded by pool names
        # Load pool names
        with open(pool_configs_file, "r") as f:
            pools = [p["name"] for p in json.load(f)["pools"]]
        # Build shards (round-robin)
        shards = [pools[i::n_py] for i in range(n_py)]
        # Remove empty shards
        shards = [s for s in shards if s]
        procs: List[subprocess.Popen] = []
        shard_files: List[str] = []
        runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vyper_pool", "vyper_pool_runner.py")
        runner_path = os.path.abspath(runner_path)
        for idx, names in enumerate(shards):
            out_path = os.path.join(output_dir, f"vyper_shard_{idx:02d}.json")
            shard_files.append(out_path)
            cmd = [
                sys.executable,
                runner_path,
                pool_configs_file,
                sequences_file,
                out_path,
                "--pools",
                ",".join(names),
            ]
            procs.append(subprocess.Popen(cmd))
        # Wait and check
        exit_codes = [p.wait() for p in procs]
        if any(code != 0 for code in exit_codes):
            failed = [i for i, c in enumerate(exit_codes) if c != 0]
            raise RuntimeError(f"Vyper shard(s) failed: {failed}")
        # Merge results
        combined: Dict[str, Any] = {"results": []}
        for fp in shard_files:
            with open(fp, "r") as f:
                shard_res = json.load(f)
            combined["results"].extend(shard_res.get("results", []))
        with open(vyper_output, "w") as f:
            json.dump(combined, f, indent=2)
        results = combined

    vyper_time = time.time() - start_time
    print(f"✓ Vyper benchmark completed in {vyper_time:.2f}s")

    # Extract states for each test (Vyper runner may not include 'sequence')
    vyper_states = {}
    for test in results.get("results", []):
        key = test.get("pool_config") or test.get("pool_name")
        if not key:
            continue
        if test.get("result", {}).get("success"):
            vyper_states[key] = test["result"].get("states") or [test["result"].get("final_state")]
        else:
            vyper_states[key] = {"error": test.get("result", {}).get("error", "Failed")}

    return {"states": vyper_states, "time": vyper_time, "output_file": vyper_output}


def compare_results(cpp_results: Dict, vyper_results: Dict = None) -> Dict:
    """Compare C++ results (and optionally Vyper results)."""
    comparison = {
        "cpp_time": cpp_results["time"],
        "tests_run": len(cpp_results["states"]),
        "tests_succeeded": sum(1 for v in cpp_results["states"].values() if "error" not in v),
        "tests_failed": sum(1 for v in cpp_results["states"].values() if "error" in v)
    }
    
    if vyper_results:
        comparison["vyper_time"] = vyper_results["time"]
        comparison["speedup"] = vyper_results["time"] / cpp_results["time"] if cpp_results["time"] > 0 else 0

        def norm(v):
            if isinstance(v, list):
                return [str(x) for x in v]
            if isinstance(v, (int, float)):
                return str(v)
            return v

        matches = 0
        mismatches = []
        for key, c_states in cpp_results["states"].items():
            v_states = vyper_results["states"].get(key)
            if v_states is None:
                mismatches.append({"test": key, "metric": "missing", "cpp": "present", "vyper": "absent"})
                continue
            # Compare final snapshot
            c_final = c_states[-1] if isinstance(c_states, list) else c_states
            v_final = v_states[-1] if isinstance(v_states, list) else v_states
            metric_list = ["balances", "D", "virtual_price", "totalSupply", "price_scale"]
            any_diff = False
            for m in metric_list:
                if m in c_final and m in v_final:
                    if norm(c_final[m]) != norm(v_final[m]):
                        any_diff = True
                        mismatches.append({"test": key, "metric": m, "cpp": c_final[m], "vyper": v_final[m]})
            if not any_diff:
                matches += 1

        comparison["matches"] = matches
        comparison["mismatches"] = mismatches
        total = comparison.get("tests_run", 0)
        comparison["exact_parity"] = (matches == total and total > 0)
    
    return comparison


def compare_final_precision(baseline: Dict, approx: Dict) -> Dict:
    """Per-pool absolute diffs (final state) for basic metrics. Kept for compatibility."""
    diffs = {}
    for key, b_states in baseline["states"].items():
        a_states = approx["states"].get(key)
        if not a_states:
            continue
        b_final = b_states[-1] if isinstance(b_states, list) else b_states
        a_final = a_states[-1] if isinstance(a_states, list) else a_states
        metrics = ["balances", "D", "virtual_price", "totalSupply", "price_scale"]
        diffs[key] = {}
        for m in metrics:
            if m in b_final and m in a_final:
                b = b_final[m]
                a = a_final[m]
                if isinstance(b, list):
                    try:
                        b_list = [int(x) for x in b]
                        a_list = [int(x) for x in a]
                        diff = [abs(ai - bi) for ai, bi in zip(a_list, b_list)]
                    except Exception:
                        diff = None
                    diffs[key][m] = diff
                else:
                    try:
                        bi = int(b)
                        ai = int(a)
                        diffs[key][m] = abs(ai - bi)
                    except Exception:
                        diffs[key][m] = None
    return diffs


def compute_precision_stats(baseline_states: Dict[str, Any], approx_states: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate precision loss metrics across pools for final-state values.

    Returns per-metric stats: count, max_abs, mean_abs, max_rel_pct, mean_rel_pct.
    For list metrics (e.g., balances), computes stats across all elements.
    """
    metrics = ["balances", "D", "virtual_price", "totalSupply", "price_scale"]
    stats: Dict[str, Dict[str, float]] = {m: {
        "count": 0,
        "max_abs": 0,
        "sum_abs": 0,
        "max_rel_pct": 0.0,
        "sum_rel_pct": 0.0,
    } for m in metrics}

    def to_int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return 0

    for key, b_states in baseline_states.items():
        a_states = approx_states.get(key)
        if not a_states:
            continue
        b_final = b_states[-1] if isinstance(b_states, list) else b_states
        a_final = a_states[-1] if isinstance(a_states, list) else a_states
        for m in metrics:
            if m not in b_final or m not in a_final:
                continue
            b = b_final[m]; a = a_final[m]
            if isinstance(b, list) and isinstance(a, list):
                b_list = [to_int(x) for x in b]
                a_list = [to_int(x) for x in a]
                for bi, ai in zip(b_list, a_list):
                    abs_err = abs(ai - bi)
                    rel_pct = (abs_err * 100.0 / abs(bi)) if bi != 0 else (0.0 if ai == 0 else float('inf'))
                    st = stats[m]
                    st["count"] += 1
                    st["sum_abs"] += abs_err
                    if abs_err > st["max_abs"]:
                        st["max_abs"] = abs_err
                    if rel_pct > st["max_rel_pct"]:
                        st["max_rel_pct"] = rel_pct
                    if rel_pct != float('inf'):
                        st["sum_rel_pct"] += rel_pct
            else:
                bi = to_int(b); ai = to_int(a)
                abs_err = abs(ai - bi)
                rel_pct = (abs_err * 100.0 / abs(bi)) if bi != 0 else (0.0 if ai == 0 else float('inf'))
                st = stats[m]
                st["count"] += 1
                st["sum_abs"] += abs_err
                if abs_err > st["max_abs"]:
                    st["max_abs"] = abs_err
                if rel_pct > st["max_rel_pct"]:
                    st["max_rel_pct"] = rel_pct
                if rel_pct != float('inf'):
                    st["sum_rel_pct"] += rel_pct

    # Finalize means
    for m in metrics:
        st = stats[m]
        cnt = max(int(st["count"]), 1)
        st["mean_abs"] = st["sum_abs"] / cnt
        st["mean_rel_pct"] = st["sum_rel_pct"] / cnt
        # Drop sums from output
        st.pop("sum_abs", None)
        st.pop("sum_rel_pct", None)
    return stats


def print_precision_stats(title: str, stats: Dict[str, Any]):
    print(f"\n{title}")
    for m in ("virtual_price", "price_scale", "D", "totalSupply", "balances"):
        if m not in stats:
            continue
        st = stats[m]
        # Show relative in ppm and absolute in wei for a quick read
        mean_ppm = st["mean_rel_pct"] * 1e4  # 1% = 10,000 ppm
        max_ppm = st["max_rel_pct"] * 1e4
        print(
            f"  {m}: max={st['max_abs']} wei ({max_ppm:.2f} ppm), "
            f"mean={st['mean_abs']:.1f} wei ({mean_ppm:.2f} ppm)"
        )


def print_summary(parity_cpp_vs_vy: Dict, times: Dict[str, float], d_vs_vy_stats: Dict[str, Any], d_vs_i_stats: Dict[str, Any]):
    """Print improved, concise full summary."""
    print("\n" + "="*64)
    print("FULL BENCHMARK SUMMARY")
    print("="*64)

    # Performance
    print("\nPerformance:")
    print(f"  cpp-uint: {times['cpp_i']:.2f}s  | vyper: {times['vyper']:.2f}s  | cpp-double: {times['cpp_d']:.2f}s")
    if times['cpp_i'] > 0:
        print(f"  Speedups: vyper/cpp-uint = {times['vyper']/times['cpp_i']:.2f}x, cpp-double/cpp-uint = {times['cpp_d']/times['cpp_i']:.2f}x")

    # Parity
    total = parity_cpp_vs_vy.get('tests_run', 0)
    matches = parity_cpp_vs_vy.get('matches', 0)
    exact = parity_cpp_vs_vy.get('exact_parity', False)
    print("\nParity (cpp-uint vs vyper):")
    print(f"  Exact parity: {'YES' if exact else 'NO'}  ({matches}/{total} pools)")
    if not exact and parity_cpp_vs_vy.get('mismatches'):
        print("  First mismatches:")
        for mm in parity_cpp_vs_vy['mismatches'][:5]:
            print(f"    {mm['test']}.{mm['metric']}")

    # Precision loss
    print_precision_stats("Precision loss (cpp-double vs vyper):", d_vs_vy_stats)
    print_precision_stats("Precision loss (cpp-double vs cpp-uint):", d_vs_i_stats)


def _ensure_built_harness():
    """Build typed C++ harnesses once to avoid rebuilds when switching modes."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(repo_root, "../cpp/build")
    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    # Configure if needed (Release)
    if not os.path.exists(os.path.join(build_dir, "CMakeCache.txt")):
        subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True)
    # Build both typed harnesses
    subprocess.run(["cmake", "--build", ".", "--target", "benchmark_harness_i", "benchmark_harness_d"], cwd=build_dir, check=True)


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _run_uv(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    # Deprecated helper (kept for compatibility if needed)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run TwoCrypto full pool benchmarks (single pass; parallelism via --n-cpp/--n-py)")
    parser.add_argument("--workers", type=int, default=0, help="Deprecated. Ignored.")
    parser.add_argument("--n-py", type=int, default=1, help="Vyper worker processes (>=1)")
    parser.add_argument("--n-cpp", type=int, default=0, help="C++ threads per harness process (0 = auto)")
    parser.add_argument("--save-per-pool", action="store_true", help="Keep per-pool result files (cpp/<pool>.json, vyper/<pool>.json)")
    parser.add_argument("--final-only", action="store_true", help="Only save final state per test (set SAVE_LAST_ONLY=1)")
    parser.add_argument("--snapshot-every", type=int, default=None, help="Snapshot every N actions (0=final only, 1=every, N=interval). Overrides --final-only")
    parser.add_argument("--limit-actions", type=int, default=0, help="Use only the first N actions from the sequence (0 = no limit)")
    args = parser.parse_args()

    # Paths
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(script_dir, "data")
    base_results_dir = os.path.join(data_dir, "results")
    os.makedirs(base_results_dir, exist_ok=True)

    # Timestamped run dir
    run_stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(base_results_dir, run_stamp)
    os.makedirs(run_dir, exist_ok=True)

    pool_configs_path = os.path.join(data_dir, "pools.json")
    sequences_path = os.path.join(data_dir, "sequences.json")

    if not os.path.exists(pool_configs_path) or not os.path.exists(sequences_path):
        print("❌ Input not found. Generate data with: uv run benchmark_pool/generate_data.py")
        return 1

    with open(pool_configs_path, "r") as f:
        pools = json.load(f)["pools"]
    with open(sequences_path, "r") as f:
        sequences = json.load(f)["sequences"]
    if not sequences:
        print("❌ No sequences found")
        return 1
    sequence = sequences[0]

    # Optionally limit the number of actions
    original_actions = list(sequence.get("actions", []))
    original_count = len(original_actions)
    limit = max(0, int(getattr(args, "limit_actions", 0)))
    if limit > 0 and original_count > limit:
        seq_limited = dict(sequence)
        seq_limited["actions"] = original_actions[:limit]
        # Write limited sequence file into run dir and use it for this run
        limited_seq_path = os.path.join(run_dir, "inputs_sequences_limited.json")
        _write_json(limited_seq_path, {"sequences": [seq_limited]})
        sequences_for_run = limited_seq_path
        actions_used = limit
        print(f"Limiting sequence actions: using first {limit} of {original_count}")
    else:
        sequences_for_run = sequences_path
        actions_used = original_count

    # Save a copy of inputs in the run dir
    _write_json(os.path.join(run_dir, "inputs_pools.json"), {"pools": pools})
    # Always save the full/original sequence for reproducibility
    _write_json(os.path.join(run_dir, "inputs_sequences_full.json"), {"sequences": [sequence]})
    # Also save the sequence actually used (may be the same as full)
    if sequences_for_run.endswith("inputs_sequences_limited.json"):
        pass  # already saved above
    else:
        _write_json(os.path.join(run_dir, "inputs_sequences.json"), {"sequences": [sequence]})

    print(f"Testing {len(pools)} pools")
    if actions_used:
        if actions_used != original_count:
            print(f"Sequence actions: {original_count} (using {actions_used}; progress every 1%)")
        else:
            print(f"Sequence actions: {actions_used} (progress logs every 1%)")
    if args.workers not in (0, 1):
        print("Note: --workers is deprecated and ignored.")

    # Pre-build C++ harness once to avoid rebuild under contention
    try:
        _ensure_built_harness()
    except Exception as e:
        print(f"⚠ Failed to prebuild harness: {e}. Proceeding anyway.")

    # Configure CPP_THREADS for single-run harnesses
    cpp_threads = args.n_cpp  # 0 means let harness auto-detect
    prev_cpp_threads = os.environ.get("CPP_THREADS")
    try:
        if cpp_threads > 0:
            os.environ["CPP_THREADS"] = str(cpp_threads)
        if args.snapshot_every is not None:
            os.environ["SNAPSHOT_EVERY"] = str(args.snapshot_every)
        elif args.final_only:
            os.environ["SAVE_LAST_ONLY"] = "1"

        # Run each side once over all pools
        cpp_info = run_cpp_benchmark(pool_configs_path, sequences_for_run, run_dir)
        cpp_time = cpp_info["time"]

        cppf_info = run_cpp_double_benchmark(pool_configs_path, sequences_for_run, run_dir)
        cppf_time = cppf_info["time"]

        vy_info = run_vyper_benchmark(pool_configs_path, sequences_for_run, run_dir, n_py=args.n_py)
        vy_time = vy_info["time"]
    finally:
        # Restore env
        if prev_cpp_threads is None:
            os.environ.pop("CPP_THREADS", None)
        else:
            os.environ["CPP_THREADS"] = prev_cpp_threads
        if args.snapshot_every is not None:
            os.environ.pop("SNAPSHOT_EVERY", None)
        if args.final_only:
            os.environ.pop("SAVE_LAST_ONLY", None)

    # Extract states for comparison
    def extract_states(results: Dict[str, Any]) -> Dict[str, Any]:
        states = {}
        for test in results.get("results", []):
            key = test.get('pool_config') or test.get('pool_name')
            res = test.get("result", {})
            s = res.get("states")
            if not s:
                final = res.get("final_state")
                s = [final] if final is not None else []
            states[key] = s
        return states

    # Load combined outputs from files written by the runners
    with open(os.path.join(run_dir, "cpp_benchmark_results.json"), "r") as f:
        cpp_combined = json.load(f)
    with open(os.path.join(run_dir, "cpp_double_benchmark_results.json"), "r") as f:
        cppf_combined = json.load(f)
    with open(os.path.join(run_dir, "vyper_benchmark_results.json"), "r") as f:
        vy_combined = json.load(f)

    cpp_states = extract_states(cpp_combined)
    cppf_states = extract_states(cppf_combined)
    vy_states = extract_states(vy_combined)

    # Compare
    comparison_cpp = compare_results({"states": cpp_states, "time": cpp_time}, {"states": vy_states, "time": vy_time})
    comparison_cppf = compare_results({"states": cppf_states, "time": cppf_time}, {"states": vy_states, "time": vy_time})
    precision_loss = compare_final_precision({"states": vy_states}, {"states": cppf_states})  # legacy detailed diffs (per-pool)

    # Aggregate precision stats for concise reporting
    d_vs_vy_stats = compute_precision_stats(vy_states, cppf_states)
    d_vs_i_stats = compute_precision_stats(cpp_states, cppf_states)

    print_summary(
        parity_cpp_vs_vy=comparison_cpp,
        times={"cpp_i": cpp_time, "cpp_d": cppf_time, "vyper": vy_time},
        d_vs_vy_stats=d_vs_vy_stats,
        d_vs_i_stats=d_vs_i_stats,
    )

    # Save combined outputs under run dir
    # Also save combined under legacy-friendly names
    _write_json(os.path.join(run_dir, "cpp_i_combined.json"), cpp_combined)
    _write_json(os.path.join(run_dir, "cpp_d_combined.json"), cppf_combined)
    _write_json(os.path.join(run_dir, "vyper_combined.json"), vy_combined)
    _write_json(os.path.join(run_dir, "benchmark_comparison_i_vs_vyper.json"), comparison_cpp)
    _write_json(os.path.join(run_dir, "benchmark_comparison_d_vs_vyper.json"), comparison_cppf)
    _write_json(os.path.join(run_dir, "d_precision_loss.json"), precision_loss)
    _write_json(os.path.join(run_dir, "d_vs_vyper_precision_stats.json"), d_vs_vy_stats)
    _write_json(os.path.join(run_dir, "d_vs_cpp_i_precision_stats.json"), d_vs_i_stats)

    print(f"\n✓ Results saved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
