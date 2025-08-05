#!/usr/bin/env python3
"""
Simple benchmark runner for C++ vs Vyper StableswapMath.
Usage: uv run benchmark/run_benchmark.py <test_file.json>
"""

import json
import sys
import time
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.vyper_benchmark import VyperBenchmark
from benchmark.cpp_benchmark import CppBenchmark
from benchmark.comparison import ResultComparator

console = Console()


def run_benchmark_with_retry(benchmark_func, test_cases, name=""):
    """Run benchmark function with error handling for non-convergent cases."""
    results = []
    times = []
    skipped = []
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Running {name}...", total=len(test_cases))
        
        for i, case in enumerate(test_cases):
            try:
                start = time.perf_counter_ns()
                result = benchmark_func(case)
                elapsed = time.perf_counter_ns() - start
                
                results.append(result)
                times.append(elapsed)
            except Exception as e:
                # Skip cases that don't converge
                if "Did not converge" in str(e) or "convergence" in str(e).lower():
                    results.append("0")  # Default value for failed cases
                    times.append(0)
                    skipped.append(i)
                else:
                    # Re-raise other errors
                    raise
            
            progress.update(task, advance=1)
    
    if skipped:
        console.print(f"[yellow]⚠ Skipped {len(skipped)} non-convergent cases[/yellow]")
    
    return results, times, skipped


@click.command()
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='benchmark/benchmark_results.json', help='Output file for results')
@click.option('--skip-build', is_flag=True, help='Skip building C++ library')
def main(test_file, output, skip_build):
    """Run benchmark comparison between C++ and Vyper implementations."""
    
    console.print("[bold blue]StableswapMath Benchmark[/bold blue]")
    console.print(f"Test file: {test_file}\n")
    
    # Load test cases
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Flatten all test cases
    all_cases = []
    if 'realistic' in test_data:
        all_cases.extend(test_data['realistic'])
    if 'edge_cases' in test_data:
        all_cases.extend(test_data['edge_cases'])
    if 'stress' in test_data:
        all_cases.extend(test_data['stress'])
    
    # If no categorized cases, assume flat list
    if not all_cases and isinstance(test_data, list):
        all_cases = test_data
    
    console.print(f"[cyan]Loaded {len(all_cases)} test cases[/cyan]\n")
    
    # Build C library if needed
    if not skip_build:
        console.print("Building C library...")
        import subprocess
        result = subprocess.run(["cmake", "--build", "build"], capture_output=True)
        if result.returncode != 0:
            console.print("[red]Failed to build C library[/red]")
            console.print(result.stderr.decode())
            sys.exit(1)
    
    # Find library
    lib_path = Path("build/lib/libstableswap_math.so")
    if not lib_path.exists():
        lib_path = Path("build/lib/libstableswap_math.dylib")
    if not lib_path.exists():
        console.print("[red]C library not found. Run: cmake -B build . && cmake --build build[/red]")
        sys.exit(1)
    
    # Initialize benchmarks
    console.print("\n[yellow]Deploying Vyper contract...[/yellow]")
    vyper_bench = VyperBenchmark()
    
    console.print("[yellow]Loading C library...[/yellow]")
    cpp_bench = CppBenchmark(lib_path)
    
    # Run Vyper benchmarks with error handling
    console.print("\n[bold]Running Vyper benchmarks[/bold]")
    
    vyper_results = {
        'newton_D': {'results': [], 'times_ns': [], 'skipped': []},
        'get_y': {'results': [], 'times_ns': [], 'skipped': []},
        'get_p': {'results': [], 'times_ns': [], 'skipped': []}
    }
    
    # Benchmark newton_D
    def vyper_newton_d(case):
        A = int(case['A'])
        gamma = int(case.get('gamma', '145000000000000'))
        xp = [int(case['x0']), int(case['x1'])]
        return str(vyper_bench.contract.newton_D(A, gamma, xp))
    
    results, times, skipped = run_benchmark_with_retry(
        vyper_newton_d, all_cases, "Vyper newton_D"
    )
    vyper_results['newton_D']['results'] = results
    vyper_results['newton_D']['times_ns'] = times
    vyper_results['newton_D']['skipped'] = skipped
    
    # Use computed D values for get_y and get_p
    D_values = results
    
    # Benchmark get_y
    def vyper_get_y(args):
        case, D, i = args
        if D == "0":  # Skip if newton_D failed
            return "0"
        A = int(case['A'])
        gamma = int(case.get('gamma', '145000000000000'))
        xp = [int(case['x0']), int(case['x1'])]
        result = vyper_bench.contract.get_y(A, gamma, xp, int(D), i)
        return str(result[0])  # get_y returns [y, k]
    
    get_y_cases = [(case, D, i % 2) for i, (case, D) in enumerate(zip(all_cases, D_values))]
    results, times, skipped = run_benchmark_with_retry(
        vyper_get_y, get_y_cases, "Vyper get_y"
    )
    vyper_results['get_y']['results'] = results
    vyper_results['get_y']['times_ns'] = times
    vyper_results['get_y']['skipped'] = skipped
    
    # Benchmark get_p
    def vyper_get_p(args):
        case, D = args
        if D == "0":  # Skip if newton_D failed
            return "0"
        xp = [int(case['x0']), int(case['x1'])]
        A_gamma = [int(case['A']), int(case.get('gamma', '145000000000000'))]
        return str(vyper_bench.contract.get_p(xp, int(D), A_gamma))
    
    get_p_cases = list(zip(all_cases, D_values))
    results, times, skipped = run_benchmark_with_retry(
        vyper_get_p, get_p_cases, "Vyper get_p"
    )
    vyper_results['get_p']['results'] = results
    vyper_results['get_p']['times_ns'] = times
    vyper_results['get_p']['skipped'] = skipped
    
    # Calculate Vyper totals
    vyper_total_time = sum(sum(vyper_results[f]['times_ns']) for f in ['newton_D', 'get_y', 'get_p'])
    console.print(f"[green]✓ Vyper complete: {vyper_total_time/1_000_000:.2f}ms[/green]\n")
    
    # Run C++ benchmarks
    console.print("[bold]Running C++ benchmarks[/bold]")
    
    cpp_results = {
        'newton_D': {'results': [], 'times_ns': []},
        'get_y': {'results': [], 'times_ns': []},
        'get_p': {'results': [], 'times_ns': []}
    }
    
    # Benchmark newton_D
    def cpp_newton_d(case):
        amp = case['A']
        gamma = case.get('gamma', '145000000000000')
        x0 = case['x0']
        x1 = case['x1']
        return cpp_bench.newton_D(amp, gamma, x0, x1)
    
    results, times, _ = run_benchmark_with_retry(
        cpp_newton_d, all_cases, "C++ newton_D"
    )
    cpp_results['newton_D']['results'] = results
    cpp_results['newton_D']['times_ns'] = times
    
    # Benchmark get_y (using same D values as Vyper for fair comparison)
    def cpp_get_y(args):
        case, D, i = args
        if D == "0":  # Skip if newton_D failed
            return "0"
        amp = case['A']
        gamma = case.get('gamma', '145000000000000')
        x0 = case['x0']
        x1 = case['x1']
        result, _ = cpp_bench.get_y(amp, gamma, x0, x1, D, i)
        return result
    
    get_y_cases = [(case, D, i % 2) for i, (case, D) in enumerate(zip(all_cases, D_values))]
    results, times, _ = run_benchmark_with_retry(
        cpp_get_y, get_y_cases, "C++ get_y"
    )
    cpp_results['get_y']['results'] = results
    cpp_results['get_y']['times_ns'] = times
    
    # Benchmark get_p
    def cpp_get_p(args):
        case, D = args
        if D == "0":  # Skip if newton_D failed
            return "0"
        x0 = case['x0']
        x1 = case['x1']
        amp = case['A']
        return cpp_bench.get_p(x0, x1, D, amp)
    
    get_p_cases = list(zip(all_cases, D_values))
    results, times, _ = run_benchmark_with_retry(
        cpp_get_p, get_p_cases, "C++ get_p"
    )
    cpp_results['get_p']['results'] = results
    cpp_results['get_p']['times_ns'] = times
    
    # Calculate C++ totals
    cpp_total_time = sum(sum(cpp_results[f]['times_ns']) for f in ['newton_D', 'get_y', 'get_p'])
    console.print(f"[green]✓ C++ complete: {cpp_total_time/1_000_000:.2f}ms[/green]\n")
    
    # Compare results
    console.print("[bold]Results Comparison[/bold]\n")
    
    # Accuracy check
    accuracy_table = Table(title="Accuracy", show_header=True)
    accuracy_table.add_column("Function", style="cyan")
    accuracy_table.add_column("Total Cases", justify="right")
    accuracy_table.add_column("Valid Cases", justify="right")
    accuracy_table.add_column("Match Rate", justify="right", style="green")
    
    for func in ['newton_D', 'get_y', 'get_p']:
        vyper_vals = vyper_results[func]['results']
        cpp_vals = cpp_results[func]['results']
        skipped = vyper_results[func].get('skipped', [])
        
        valid_cases = [(v, c) for i, (v, c) in enumerate(zip(vyper_vals, cpp_vals)) 
                      if i not in skipped and v != "0"]
        
        matches = sum(1 for v, c in valid_cases if v == c)
        match_rate = (matches / len(valid_cases) * 100) if valid_cases else 0
        
        accuracy_table.add_row(
            func,
            str(len(vyper_vals)),
            str(len(valid_cases)),
            f"{match_rate:.1f}%"
        )
    
    console.print(accuracy_table)
    console.print()
    
    # Performance comparison
    perf_table = Table(title="Performance", show_header=True)
    perf_table.add_column("Function", style="cyan")
    perf_table.add_column("Vyper (ms)", justify="right")
    perf_table.add_column("C++ (ms)", justify="right")
    perf_table.add_column("Speedup", justify="right", style="bold green")
    
    for func in ['newton_D', 'get_y', 'get_p']:
        vyper_time = sum(vyper_results[func]['times_ns']) / 1_000_000
        cpp_time = sum(cpp_results[func]['times_ns']) / 1_000_000
        speedup = vyper_time / cpp_time if cpp_time > 0 else 0
        
        perf_table.add_row(
            func,
            f"{vyper_time:.2f}",
            f"{cpp_time:.2f}",
            f"{speedup:.1f}x"
        )
    
    # Add overall row
    vyper_total_ms = vyper_total_time / 1_000_000
    cpp_total_ms = cpp_total_time / 1_000_000
    overall_speedup = vyper_total_ms / cpp_total_ms if cpp_total_ms > 0 else 0
    
    perf_table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{vyper_total_ms:.2f}[/bold]",
        f"[bold]{cpp_total_ms:.2f}[/bold]",
        f"[bold yellow]{overall_speedup:.1f}x[/bold yellow]"
    )
    
    console.print(perf_table)
    
    # Save results
    results_data = {
        'test_file': str(test_file),
        'num_cases': len(all_cases),
        'vyper': vyper_results,
        'cpp': cpp_results,
        'performance': {
            'vyper_total_ms': vyper_total_ms,
            'cpp_total_ms': cpp_total_ms,
            'speedup': overall_speedup
        }
    }
    
    with open(output, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"\n[green]✓ Results saved to {output}[/green]")


if __name__ == "__main__":
    main()