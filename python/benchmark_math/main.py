"""TwoCrypto Benchmark - Compare C++ vs Vyper implementation."""

import json
import time
import ctypes
import random
import subprocess
import sys
from pathlib import Path

import boa


def generate_test_cases():
    """Generate test cases with realistic and edge case values."""
    cases = []
    
    # Realistic cases
    for _ in range(100):
        cases.append({
            "A": str(random.randint(10, 20000) * 10000),
            "gamma": str(random.randint(10**10, 10**16)),
            "x0": str(random.randint(10**18, 10**24)),
            "x1": str(random.randint(10**18, 10**24))
        })
    
    # Edge cases
    edge_values = ["1000000000000000000", "2000000000000000000", "1000000000000000000000000000"]
    for v1 in edge_values[:2]:
        for v2 in edge_values[:2]:
            cases.append({
                "A": "10000000",
                "gamma": "145000000000000",
                "x0": v1,
                "x1": v2
            })
    
    # Save test cases
    output_path = Path(__file__).parent / "test_cases.json"
    with open(output_path, 'w') as f:
        json.dump(cases, f, indent=2)
    
    print(f"Generated {len(cases)} test cases")
    return cases


def build_cpp_library():
    """Build the C++ library using CMake."""
    root_dir = Path(__file__).parent.parent.parent
    cpp_dir = root_dir / "cpp"
    build_dir = cpp_dir / "build"
    
    print("Building C++ library...")
    
    # Configure
    cmd = ["cmake", "-B", str(build_dir), str(cpp_dir), "-DCMAKE_BUILD_TYPE=Release"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CMake configure failed:\n{result.stderr}")
        sys.exit(1)
    
    # Build
    cmd = ["cmake", "--build", str(build_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    
    # Find the library (different extensions on different platforms)
    import platform
    system = platform.system()
    if system == "Darwin":  # macOS
        lib_path = build_dir / "lib" / "libstableswap_math_i.dylib"
    elif system == "Linux":
        lib_path = build_dir / "lib" / "libstableswap_math_i.so"
    else:
        # Fallback: try common extensions
        lib_path = build_dir / "lib" / "libstableswap_math_i.dylib"
        if not lib_path.exists():
            lib_path = build_dir / "lib" / "libstableswap_math_i.so"
    
    if not lib_path.exists():
        print("Library not found after build")
        sys.exit(1)
    
    print(f"C++ library built successfully: {lib_path}")
    return lib_path


class VyperBenchmark:
    """Vyper contract benchmark wrapper."""
    
    def __init__(self):
        # Deploy contract
        contract_path = Path(__file__).parent.parent.parent / "twocrypto-ng" / "contracts" / "main" / "StableswapMath.vy"
        if not contract_path.exists():
            raise FileNotFoundError(f"Vyper contract not found at {contract_path}")
        self.contract = boa.load(str(contract_path))
    
    def newton_D(self, A, gamma, x0, x1):
        return self.contract.newton_D(int(A), int(gamma), [int(x0), int(x1)])
    
    def get_y(self, A, gamma, x0, x1, D, i):
        result = self.contract.get_y(int(A), int(gamma), [int(x0), int(x1)], int(D), i)
        return result[0]  # Returns (y, k)
    
    def get_p(self, x0, x1, D, A):
        return self.contract.get_p([int(x0), int(x1)], int(D), [int(A), 145000000000000])


class CppBenchmark:
    """C++ library benchmark wrapper."""
    
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Setup function signatures
        self.lib.newton_D.argtypes = [ctypes.c_char_p] * 4
        self.lib.newton_D.restype = ctypes.POINTER(ctypes.c_char)
        
        self.lib.get_y.argtypes = [ctypes.c_char_p] * 5 + [ctypes.c_int]
        self.lib.get_y.restype = ctypes.POINTER(ctypes.c_char_p)
        
        self.lib.get_p.argtypes = [ctypes.c_char_p] * 4
        self.lib.get_p.restype = ctypes.POINTER(ctypes.c_char)
        
        self.lib.free_string.argtypes = [ctypes.c_void_p]
        self.lib.free_string_array.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
    
    def newton_D(self, A, gamma, x0, x1):
        result = self.lib.newton_D(A.encode(), gamma.encode(), x0.encode(), x1.encode())
        value = ctypes.cast(result, ctypes.c_char_p).value.decode()
        self.lib.free_string(result)
        return value
    
    def get_y(self, A, gamma, x0, x1, D, i):
        result = self.lib.get_y(A.encode(), gamma.encode(), x0.encode(), x1.encode(), D.encode(), i)
        value = result[0].decode()
        self.lib.free_string_array(result, 2)
        return value
    
    def get_p(self, x0, x1, D, A):
        result = self.lib.get_p(x0.encode(), x1.encode(), D.encode(), A.encode())
        value = ctypes.cast(result, ctypes.c_char_p).value.decode()
        self.lib.free_string(result)
        return value


def benchmark_function(func, *args, iterations=10):
    """Benchmark a function with warm-up."""
    # Warm-up
    for _ in range(3):
        func(*args)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    elapsed = time.perf_counter() - start
    
    return result, elapsed / iterations


def main():
    """Run the complete benchmark."""
    print("=== TwoCrypto Benchmark ===\n")
    
    # Step 1: Generate test cases
    print("1. Generating test cases...")
    test_cases = generate_test_cases()
    
    # Step 2: Build C++ library
    print("\n2. Building C++ library...")
    lib_path = build_cpp_library()
    
    # Step 3: Initialize benchmarks
    print("\n3. Initializing benchmarks...")
    vyper = VyperBenchmark()
    cpp = CppBenchmark(lib_path)
    
    # Step 4: Run benchmarks
    print("\n4. Running benchmarks...\n")
    
    results = {
        'newton_D': {'vyper_time': 0, 'cpp_time': 0, 'matches': 0, 'total': 0},
        'get_y': {'vyper_time': 0, 'cpp_time': 0, 'matches': 0, 'total': 0},
        'get_p': {'vyper_time': 0, 'cpp_time': 0, 'matches': 0, 'total': 0}
    }
    
    # Sample 20 cases for benchmarking
    sample_cases = test_cases
    
    for i, case in enumerate(sample_cases):
        A, gamma, x0, x1 = case['A'], case.get('gamma', '145000000000000'), case['x0'], case['x1']
        
        # newton_D
        vyper_D, vyper_time = benchmark_function(vyper.newton_D, A, gamma, x0, x1)
        cpp_D, cpp_time = benchmark_function(cpp.newton_D, A, gamma, x0, x1)
        
        results['newton_D']['vyper_time'] += vyper_time
        results['newton_D']['cpp_time'] += cpp_time
        results['newton_D']['total'] += 1
        if str(vyper_D) == cpp_D:
            results['newton_D']['matches'] += 1
        
        # get_y
        D = str(vyper_D)
        vyper_y, vyper_time = benchmark_function(vyper.get_y, A, gamma, x0, x1, D, 0)
        cpp_y, cpp_time = benchmark_function(cpp.get_y, A, gamma, x0, x1, D, 0)
        
        results['get_y']['vyper_time'] += vyper_time
        results['get_y']['cpp_time'] += cpp_time
        results['get_y']['total'] += 1
        if str(vyper_y) == cpp_y:
            results['get_y']['matches'] += 1
        
        # get_p
        vyper_p, vyper_time = benchmark_function(vyper.get_p, x0, x1, D, A)
        cpp_p, cpp_time = benchmark_function(cpp.get_p, x0, x1, D, A)
        
        results['get_p']['vyper_time'] += vyper_time
        results['get_p']['cpp_time'] += cpp_time
        results['get_p']['total'] += 1
        if str(vyper_p) == cpp_p:
            results['get_p']['matches'] += 1
        
        print(f"\rProgress: {i+1}/{len(sample_cases)}", end="")
    
    # Step 5: Display results
    print("\n\n=== Results ===\n")
    
    print(f"{'Function':<15} {'Vyper (ms)':<12} {'C++ (ms)':<12} {'Speedup':<10} {'Precision'}")
    print("-" * 60)
    
    total_vyper = 0
    total_cpp = 0
    
    for func in ['newton_D', 'get_y', 'get_p']:
        data = results[func]
        vyper_ms = data['vyper_time'] * 1000 / data['total']
        cpp_ms = data['cpp_time'] * 1000 / data['total']
        speedup = vyper_ms / cpp_ms if cpp_ms > 0 else 0
        precision = data['matches'] / data['total'] * 100
        
        print(f"{func:<15} {vyper_ms:<12.3f} {cpp_ms:<12.3f} {speedup:<10.1f}x {precision:.1f}%")
        
        total_vyper += data['vyper_time']
        total_cpp += data['cpp_time']
    
    print("-" * 60)
    total_vyper_ms = total_vyper * 1000 / len(sample_cases)
    total_cpp_ms = total_cpp * 1000 / len(sample_cases)
    total_speedup = total_vyper_ms / total_cpp_ms if total_cpp_ms > 0 else 0
    print(f"{'TOTAL':<15} {total_vyper_ms:<12.3f} {total_cpp_ms:<12.3f} {total_speedup:<10.1f}x")
    
    # Save results
    output = {
        'test_cases': len(sample_cases),
        'results': results,
        'summary': {
            'avg_speedup': total_speedup,
            'precision_match': all(results[f]['matches'] == results[f]['total'] for f in results)
        }
    }
    
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Results saved to results.json")


if __name__ == "__main__":
    main()