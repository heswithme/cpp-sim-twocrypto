#!/usr/bin/env python3
"""
Run full benchmark comparing C++ and Vyper implementations
"""
import json
import os
import sys
import time
from typing import Dict

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
        
        # Compare individual states
        matches = 0
        mismatches = []
        
        for key in cpp_results["states"]:
            if key in vyper_results["states"]:
                cpp_state = cpp_results["states"][key]
                vyper_state = vyper_results["states"][key]
                
                if "error" not in cpp_state and "error" not in vyper_state:
                    # Compare final states
                    cpp_final = cpp_state[-1] if isinstance(cpp_state, list) else cpp_state
                    vyper_final = vyper_state[-1] if isinstance(vyper_state, list) else vyper_state
                    
                    # Check key metrics
                    metrics_match = True
                    for metric in ["balances", "D", "virtual_price", "totalSupply"]:
                        if metric in cpp_final and metric in vyper_final:
                            if cpp_final[metric] != vyper_final[metric]:
                                metrics_match = False
                                mismatches.append({
                                    "test": key,
                                    "metric": metric,
                                    "cpp": cpp_final[metric],
                                    "vyper": vyper_final[metric]
                                })
                    
                    if metrics_match:
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


def main():
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    output_dir = os.path.join(data_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    pool_configs = os.path.join(data_dir, "benchmark_pools.json")
    sequences = os.path.join(data_dir, "benchmark_sequences.json")
    
    # Check files exist
    if not os.path.exists(pool_configs):
        print(f"❌ Pool configs not found: {pool_configs}")
        print("Run: python datagen/generate_benchmark_data.py")
        return 1
        
    if not os.path.exists(sequences):
        print(f"❌ Sequences not found: {sequences}")
        print("Run: python datagen/generate_benchmark_data.py")
        return 1
    
    # Load configurations for summary
    with open(pool_configs, 'r') as f:
        pools = json.load(f)["pools"]
    with open(sequences, 'r') as f:
        seqs = json.load(f)["sequences"]
    
    print(f"Testing {len(pools)} pools × {len(seqs)} sequences = {len(pools)*len(seqs)} combinations")
    
    # Run C++ benchmark
    cpp_results = run_cpp_benchmark(pool_configs, sequences, output_dir)
    
    # Run Vyper benchmark
    vyper_results = run_vyper_benchmark(pool_configs, sequences, output_dir)
    
    # Compare results
    comparison = compare_results(cpp_results, vyper_results)
    
    # Print summary
    print_summary(comparison)
    
    # Save comparison
    comparison_file = os.path.join(output_dir, "benchmark_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())