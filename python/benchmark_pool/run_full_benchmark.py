#!/usr/bin/env python3
"""
Run full benchmark comparing C++ and Vyper implementations, with per-pool parallelism.
"""
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpp_pool.cpp_pool_runner import run_cpp_pool
from vyper_pool.vyper_pool_runner import run_vyper_pool


def run_cpp_benchmark(pool_configs_file: str, sequences_file: str, output_dir: str) -> Dict:
    """Run C++ benchmark and save results."""
    print("\n=== Running C++ Benchmark ===")
    
    # Run benchmark
    cpp_output = os.path.join(output_dir, "cpp_benchmark_results.json")
    
    start_time = time.time()
    results = run_cpp_pool(pool_configs_file, sequences_file, cpp_output)
    cpp_time = time.time() - start_time
    
    print(f"✓ C++ benchmark completed in {cpp_time:.2f}s")
    
    # Extract states for each test
    cpp_states = {}
    for test in results["results"]:
        key = f"{test['pool_config']}_{test['sequence']}"
        if test["result"]["success"]:
            cpp_states[key] = test["result"]["states"]
        else:
            cpp_states[key] = {"error": test["result"].get("error", "Failed")}
    
    return {
        "states": cpp_states,
        "time": cpp_time,
        "output_file": cpp_output
    }


def run_vyper_benchmark(pool_configs_file: str, sequences_file: str, output_dir: str) -> Dict:
    """Run Vyper benchmark and save results."""
    print("\n=== Running Vyper Benchmark ===")
    
    # Run benchmark
    vyper_output = os.path.join(output_dir, "vyper_benchmark_results.json")
    
    start_time = time.time()
    results = run_vyper_pool(pool_configs_file, sequences_file, vyper_output)
    vyper_time = time.time() - start_time
    
    print(f"✓ Vyper benchmark completed in {vyper_time:.2f}s")
    
    # Extract states for each test
    vyper_states = {}
    for test in results["results"]:
        key = f"{test['pool_config']}_{test['sequence']}"
        if test["result"]["success"]:
            vyper_states[key] = test["result"]["states"]
        else:
            vyper_states[key] = {"error": test["result"].get("error", "Failed")}
    
    return {
        "states": vyper_states,
        "time": vyper_time,
        "output_file": vyper_output
    }


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
    
    return comparison


def print_summary(comparison: Dict):
    """Print benchmark summary."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"\nC++ Performance:")
    print(f"  Tests run: {comparison['tests_run']}")
    print(f"  Succeeded: {comparison['tests_succeeded']}")
    print(f"  Failed: {comparison['tests_failed']}")
    print(f"  Time: {comparison['cpp_time']:.2f}s")
    
    if "vyper_time" in comparison:
        print(f"\nVyper Performance:")
        print(f"  Time: {comparison['vyper_time']:.2f}s")
        print(f"  Speedup: {comparison['speedup']:.1f}x")
        
        print(f"\nAccuracy:")
        print(f"  Matches: {comparison['matches']}")
        print(f"  Mismatches: {len(comparison['mismatches'])}")
        
        if comparison['mismatches']:
            print("\n  Sample mismatches:")
            for mm in comparison['mismatches'][:5]:
                print(f"    {mm['test']}.{mm['metric']}")


def _ensure_built_harness():
    """Build C++ harness once to avoid parallel rebuild races."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(repo_root, "../cpp/build")
    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    # Configure if needed
    if not os.path.exists(os.path.join(build_dir, "CMakeCache.txt")):
        subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    # Build harness
    subprocess.run(["cmake", "--build", ".", "--target", "benchmark_harness"], cwd=build_dir, check=True)


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _run_uv(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run TwoCrypto full pool benchmarks with per-pool parallelism and C++ thread control")
    parser.add_argument("--workers", type=int, default=0, help="Deprecated alias for --n-py")
    parser.add_argument("--n-py", type=int, default=1, help="Python per-pool workers (processes) for each phase")
    parser.add_argument("--n-cpp", type=int, default=0, help="C++ threads per harness process (0 = auto)")
    parser.add_argument("--save-per-pool", action="store_true", help="Keep per-pool result files (cpp/<pool>.json, vyper/<pool>.json)")
    args = parser.parse_args()

    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
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

    # Save a copy of inputs in the run dir
    _write_json(os.path.join(run_dir, "inputs_pools.json"), {"pools": pools})
    _write_json(os.path.join(run_dir, "inputs_sequences.json"), {"sequences": [sequence]})

    print(f"Testing {len(pools)} pools")

    # Pre-build C++ harness once to avoid rebuild under contention
    try:
        _ensure_built_harness()
    except Exception as e:
        print(f"⚠ Failed to prebuild harness: {e}. Proceeding anyway.")

    # Prepare per-pool tasks
    cpp_dir = os.path.join(run_dir, "cpp")
    vy_dir = os.path.join(run_dir, "vyper")
    os.makedirs(cpp_dir, exist_ok=True)
    os.makedirs(vy_dir, exist_ok=True)

    # Resolve worker and thread counts
    py_workers = args.n_py if args.workers == 0 else (args.workers or args.n_py)
    if py_workers <= 0:
        py_workers = len(pools)
    cpp_threads = args.n_cpp  # 0 means let harness auto-detect

    # Informative logs
    print(f"\n=== Running C++ phase ===")
    print(f"  Python workers: {py_workers}")
    print(f"  C++ threads per process: {'auto' if cpp_threads == 0 else cpp_threads}")

    # Phase 1: C++ per-pool parallel runs
    start_cpp = time.time()
    with ThreadPoolExecutor(max_workers=py_workers) as ex:
        futures_cpp = []
        for pool in pools:
            name = pool["name"]
            cpp_out = os.path.join(cpp_dir, f"{name}.json")
            # Prepare env with optional CPP_THREADS
            env = os.environ.copy()
            if cpp_threads > 0:
                env["CPP_THREADS"] = str(cpp_threads)
            env["TRACE_ONLY_POOL"] = name  # filter inside harness
            futures_cpp.append((name, ex.submit(_run_uv, [
                "uv", "run", "python/cpp_pool/cpp_pool_runner.py", pool_configs_path, sequences_path, cpp_out
            ], env)))
        done = 0
        total = len(futures_cpp)
        for name, fut in futures_cpp:
            rc, out, err = fut.result()
            done += 1
            print(f"  [C++] {done}/{total} finished: {name} ({'OK' if rc == 0 else 'FAIL'})")
            if rc != 0:
                print(f"    stderr: {err.strip()}")
    cpp_time = time.time() - start_cpp

    # Phase 2: Vyper per-pool parallel runs
    print(f"\n=== Running Vyper phase ===")
    print(f"  Python workers: {py_workers}")
    start_vy = time.time()
    with ThreadPoolExecutor(max_workers=py_workers) as ex:
        futures_vy = []
        for pool in pools:
            name = pool["name"]
            vy_out = os.path.join(vy_dir, f"{name}.json")
            env = os.environ.copy()
            env["TRACE_ONLY_POOL"] = name  # filter inside vyper runner
            futures_vy.append((name, ex.submit(_run_uv, [
                "uv", "run", "python/vyper_pool/vyper_pool_runner.py", pool_configs_path, sequences_path, vy_out
            ], env)))
        done = 0
        total = len(futures_vy)
        for name, fut in futures_vy:
            rc, out, err = fut.result()
            done += 1
            print(f"  [VY]  {done}/{total} finished: {name} ({'OK' if rc == 0 else 'FAIL'})")
            if rc != 0:
                print(f"    stderr: {err.strip()}")
    vy_time = time.time() - start_vy

    # Aggregate per-pool outputs into combined structures
    def load_side(side_dir: str) -> Dict[str, Any]:
        combined = []
        for pool in pools:
            path = os.path.join(side_dir, f"{pool['name']}.json")
            if not os.path.exists(path):
                continue
            data = json.loads(open(path).read())
            # data["results"] is an array of one entry (single pool × N sequences)
            combined.extend(data.get("results", []))
        return {"results": combined}

    cpp_combined = load_side(cpp_dir)
    vy_combined = load_side(vy_dir)

    # Optionally remove per-pool files after aggregation
    if not args.save_per_pool:
        removed = 0
        for pool in pools:
            for side_dir in (cpp_dir, vy_dir):
                path = os.path.join(side_dir, f"{pool['name']}.json")
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        removed += 1
                    except Exception:
                        pass
        # Remove empty side directories if they are empty
        for side_dir in (cpp_dir, vy_dir):
            try:
                if os.path.isdir(side_dir) and not os.listdir(side_dir):
                    os.rmdir(side_dir)
            except Exception:
                pass
        print(f"  Cleaned {removed} per-pool result files")

    # Extract states for comparison
    def extract_states(results: Dict[str, Any]) -> Dict[str, Any]:
        states = {}
        for test in results.get("results", []):
            key = test['pool_config']
            states[key] = test.get("result", {}).get("states", [])
        return states

    cpp_states = extract_states(cpp_combined)
    vy_states = extract_states(vy_combined)

    # Compare
    comparison = compare_results({"states": cpp_states, "time": cpp_time}, {"states": vy_states, "time": vy_time})

    print_summary(comparison)

    # Save combined outputs under run dir
    _write_json(os.path.join(run_dir, "cpp_combined.json"), cpp_combined)
    _write_json(os.path.join(run_dir, "vyper_combined.json"), vy_combined)
    _write_json(os.path.join(run_dir, "benchmark_comparison.json"), comparison)

    print(f"\n✓ Results saved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
