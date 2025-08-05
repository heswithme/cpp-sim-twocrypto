#!/usr/bin/env python3
"""
C++ pool benchmark runner - all-in-one module for building and testing C++ pools
"""
import json
import os
import subprocess
import sys
from typing import Dict


class CppPoolRunner:
    """Complete C++ pool building and testing infrastructure."""
    
    def __init__(self, cpp_project_path: str):
        """Initialize with path to C++ project."""
        self.cpp_project_path = cpp_project_path
        self.build_dir = os.path.join(cpp_project_path, "build")
        self.harness_path = os.path.join(self.build_dir, "benchmark_harness")
        
    def configure_build(self):
        """Configure CMake build."""
        os.makedirs(self.build_dir, exist_ok=True)
        
        print("Configuring C++ build...")
        result = subprocess.run(
            ["cmake", ".."],
            cwd=self.build_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"CMake configuration failed: {result.stderr}")
            
    def build_harness(self):
        """Build the C++ benchmark harness."""
        # Configure if needed
        if not os.path.exists(os.path.join(self.build_dir, "CMakeCache.txt")):
            self.configure_build()
        
        print("Building benchmark harness...")
        result = subprocess.run(
            ["cmake", "--build", ".", "--target", "benchmark_harness"],
            cwd=self.build_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Build failed: {result.stderr}")
            
        if not os.path.exists(self.harness_path):
            raise RuntimeError(f"Harness not found at {self.harness_path}")
            
        print(f"✓ Built C++ harness at {self.harness_path}")
        return self.harness_path
    
    def run_benchmark(self, pool_configs_file: str, sequences_file: str, output_file: str) -> Dict:
        """Run C++ benchmark with given configurations."""
        # Ensure harness is built
        if not os.path.exists(self.harness_path):
            self.build_harness()
        
        print("Running C++ harness...")
        
        # Run the harness
        result = subprocess.run(
            [self.harness_path, pool_configs_file, sequences_file, output_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Harness execution failed: {result.stderr}")
        
        # Print harness output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
        
        # Load and return results
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        # Count successful tests
        total_tests = len(results["results"])
        successful = sum(1 for r in results["results"] if r["result"]["success"])
        
        print(f"\n✓ Processed {total_tests} pool-sequence combinations")
        if successful < total_tests:
            print(f"  ⚠ {total_tests - successful} tests failed")
        
        print(f"✓ Results written to {output_file}")
        
        return results
    
    def process_results(self, results: Dict) -> Dict:
        """Process raw C++ results into structured format."""
        processed = {}
        
        for test in results["results"]:
            key = f"{test['pool_config']}_{test['sequence']}"
            
            if test["result"]["success"]:
                # Extract states from the result
                states = test["result"].get("states", [])
                if not states and "final_state" in test["result"]:
                    # If only final state is provided
                    states = [test["result"]["final_state"]]
                processed[key] = states
            else:
                # Store error information
                processed[key] = {"error": test["result"].get("error", "Failed")}
        
        return processed
    
    def format_json_output(self, json_file: str):
        """Format JSON file with proper indentation."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("✓ Formatted output JSON")


def run_cpp_pool(pool_configs_file: str, sequences_file: str, output_file: str) -> Dict:
    """Main entry point for running C++ pool benchmark."""
    # Path to C++ project
    cpp_project_path = "/Users/michael/Documents/projects/cpp-twocrypto/cpp"
    
    # Create runner and execute benchmark
    runner = CppPoolRunner(cpp_project_path)
    
    # Build harness if needed
    runner.build_harness()
    
    # Run benchmark
    results = runner.run_benchmark(pool_configs_file, sequences_file, output_file)
    
    # Format output
    runner.format_json_output(output_file)
    
    return results


def main():
    """Command-line interface."""
    if len(sys.argv) != 4:
        print("Usage: python cpp_pool_runner.py <pool_configs.json> <sequences.json> <output.json>")
        return 1
    
    pool_configs = sys.argv[1]
    sequences = sys.argv[2]
    output = sys.argv[3]
    
    # Check files exist
    if not os.path.exists(pool_configs):
        print(f"❌ Pool configs not found: {pool_configs}")
        return 1
    
    if not os.path.exists(sequences):
        print(f"❌ Sequences not found: {sequences}")
        return 1
    
    # Run benchmark
    try:
        run_cpp_pool(pool_configs, sequences, output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())