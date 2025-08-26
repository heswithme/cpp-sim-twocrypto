#!/usr/bin/env python3
"""
Unified C++ pool benchmark runner (mode: i | d).
Builds the templated harness once (Release) and runs with the selected mode.
"""
import json
import os
import subprocess
import sys
from typing import Dict
from pathlib import Path


class CppPoolRunner:
    def __init__(self, cpp_project_path: str):
        self.cpp_project_path = cpp_project_path
        self.build_dir = os.path.join(cpp_project_path, "build")
        # Unified harness (legacy) and typed binaries
        self.harness_path = os.path.join(self.build_dir, "benchmark_harness")
        self.harness_i_path = os.path.join(self.build_dir, "benchmark_harness_i")
        self.harness_d_path = os.path.join(self.build_dir, "benchmark_harness_d")
        self.harness_f_path = os.path.join(self.build_dir, "benchmark_harness_f")
        self.harness_ld_path = os.path.join(self.build_dir, "benchmark_harness_ld")

    def configure_build(self):
        os.makedirs(self.build_dir, exist_ok=True)
        print("Configuring C++ build (Release)...")
        result = subprocess.run([
            "cmake", "..", "-DCMAKE_BUILD_TYPE=Release"
        ], cwd=self.build_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CMake configuration failed: {result.stderr}")

    def build_harness(self):
        # Always reconfigure to pick up target/name changes cleanly
        self.configure_build()
        print("Building typed C++ harnesses (i, d, f, ld)...")
        # Build both typed harnesses so switching modes requires no rebuild
        result = subprocess.run([
            "cmake", "--build", ".", "--target", "benchmark_harness_i", "benchmark_harness_d", "benchmark_harness_f", "benchmark_harness_ld"
        ], cwd=self.build_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Build failed: {result.stderr}")
        missing = []
        if not os.path.exists(self.harness_i_path):
            missing.append(self.harness_i_path)
        if not os.path.exists(self.harness_d_path):
            missing.append(self.harness_d_path)
        if not os.path.exists(self.harness_f_path):
            missing.append(self.harness_f_path)
        if not os.path.exists(self.harness_ld_path):
            missing.append(self.harness_ld_path)
        if missing:
            raise RuntimeError(f"Missing built harness(es): {missing}")
        print(f"✓ Built C++ harnesses at {self.harness_i_path} and {self.harness_d_path}")
        return self.harness_i_path, self.harness_d_path

    def run_benchmark(self, mode: str, pool_configs_file: str, sequences_file: str, output_file: str) -> Dict:
        if mode not in ("i", "d", "f", "ld"):
            raise ValueError("mode must be 'i' or 'd' or 'f' or 'ld'")
        # Ensure typed harnesses exist; build if needed
        if not (os.path.exists(self.harness_i_path) and os.path.exists(self.harness_d_path) and os.path.exists(self.harness_f_path) and os.path.exists(self.harness_ld_path)):
            self.build_harness()
        if mode == "i":
            harness_bin = self.harness_i_path
        elif mode == "d":
            harness_bin = self.harness_d_path
        elif mode == "f":
            harness_bin = self.harness_f_path
        else:
            harness_bin = self.harness_ld_path
        print(f"Running C++ harness binary: {os.path.basename(harness_bin)} ...")
        import time as _time
        t0 = _time.perf_counter()
        # Typed binaries accept 3 args (no mode)
        result = subprocess.run([
            harness_bin, pool_configs_file, sequences_file, output_file
        ], capture_output=True, text=True)
        t1 = _time.perf_counter()
        harness_time = t1 - t0
        if result.returncode != 0:
            raise RuntimeError(f"Harness execution failed: {result.stderr}")
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
        with open(output_file, 'r') as f:
            results = json.load(f)
        total_tests = len(results.get("results", []))
        print(f"\n✓ Processed {total_tests} pool-sequence combinations")
        print(f"✓ Results written to {output_file}")
        md = results.get("metadata", {})
        md["harness_time_s"] = harness_time
        results["metadata"] = md
        return results

    def format_json_output(self, json_file: str):
        with open(json_file, 'r') as f:
            data = json.load(f)
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("✓ Formatted output JSON")


def run_cpp_pool(mode: str, pool_configs_file: str, sequences_file: str, output_file: str) -> Dict:
    repo_root = Path(__file__).resolve().parents[2]
    cpp_project_path = str(repo_root / "cpp")
    runner = CppPoolRunner(cpp_project_path)
    runner.build_harness()
    results = runner.run_benchmark(mode, pool_configs_file, sequences_file, output_file)
    runner.format_json_output(output_file)
    return results


def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ("i", "d"):
        print("Usage: python cpp_pool_runner.py <i|d> <pool_configs.json> <sequences.json> <output.json>")
        return 1
    mode = sys.argv[1]
    pool_configs = sys.argv[2]
    sequences = sys.argv[3]
    output = sys.argv[4]
    if not os.path.exists(pool_configs):
        print(f"❌ Pool configs not found: {pool_configs}")
        return 1
    if not os.path.exists(sequences):
        print(f"❌ Sequences not found: {sequences}")
        return 1
    try:
        run_cpp_pool(mode, pool_configs, sequences, output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
