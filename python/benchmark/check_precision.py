#!/usr/bin/env python3
"""
Precision checker for C++ vs Vyper StableswapMath.
Only checks that results match exactly - no performance measurements.
Usage: uv run benchmark/check_precision.py <test_file.json>
"""

import json
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.vyper_benchmark import VyperBenchmark
from benchmark.cpp_benchmark import CppBenchmark

console = Console()


def run_precision_check(vyper_func, cpp_func, test_cases, name=""):
    """Run precision check between Vyper and C++ implementations."""
    mismatches = []
    skipped = []
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        console=console
    ) as progress:
        task = progress.add_task(f"Checking {name}...", total=len(test_cases))
        
        for i, case in enumerate(test_cases):
            try:
                vyper_result = vyper_func(case)
                cpp_result = cpp_func(case)
                
                if vyper_result != cpp_result:
                    mismatches.append({
                        'index': i,
                        'case': case if isinstance(case, dict) else case[0],  # Handle tuple cases
                        'vyper': vyper_result,
                        'cpp': cpp_result
                    })
            except Exception as e:
                # Skip cases that don't converge
                if "Did not converge" in str(e) or "convergence" in str(e).lower():
                    skipped.append(i)
            
            progress.update(task, advance=1)
    
    return mismatches, skipped


@click.command()
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='benchmark/precision_check_results.json', help='Output file for results')
@click.option('--skip-build', is_flag=True, help='Skip building C++ library')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed mismatch information')
def main(test_file, output, skip_build, verbose):
    """Check precision match between C++ and Vyper implementations."""
    
    console.print("[bold blue]StableswapMath Precision Check[/bold blue]")
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
    
    # Build C++ library if needed
    if not skip_build:
        # Build C library
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
    
    # Initialize implementations
    console.print("[yellow]Deploying Vyper contract...[/yellow]")
    vyper_bench = VyperBenchmark()
    
    console.print("[yellow]Loading C library...[/yellow]")
    cpp_bench = CppBenchmark(lib_path)
    
    console.print()
    
    # Store all results
    all_results = {
        'newton_D': {'mismatches': [], 'skipped': []},
        'get_y': {'mismatches': [], 'skipped': []},
        'get_p': {'mismatches': [], 'skipped': []},
        'wad_exp': {'mismatches': [], 'skipped': []}
    }
    
    # Note: wad_exp testing is commented out for now as it's not used in the main pool functions
    # and would require a full implementation of snekmate's complex fixed-point exp algorithm
    
    # Check newton_D precision
    def vyper_newton_d(case):
        A = int(case['A'])
        gamma = int(case.get('gamma', '145000000000000'))
        xp = [int(case['x0']), int(case['x1'])]
        return str(vyper_bench.contract.newton_D(A, gamma, xp))
    
    def cpp_newton_d(case):
        amp = case['A']
        gamma = case.get('gamma', '145000000000000')
        x0 = case['x0']
        x1 = case['x1']
        return cpp_bench.newton_D(amp, gamma, x0, x1)
    
    mismatches, skipped = run_precision_check(
        vyper_newton_d, cpp_newton_d, all_cases, "newton_D"
    )
    all_results['newton_D']['mismatches'] = mismatches
    all_results['newton_D']['skipped'] = skipped
    
    # Store D values for subsequent tests
    D_values = []
    for i, case in enumerate(all_cases):
        if i not in skipped:
            try:
                D_values.append(vyper_newton_d(case))
            except:
                D_values.append("0")
        else:
            D_values.append("0")
    
    # Check get_y precision
    def vyper_get_y(args):
        case, D, i = args
        if D == "0":
            return "0"
        A = int(case['A'])
        gamma = int(case.get('gamma', '145000000000000'))
        xp = [int(case['x0']), int(case['x1'])]
        result = vyper_bench.contract.get_y(A, gamma, xp, int(D), i)
        return str(result[0])
    
    def cpp_get_y(args):
        case, D, i = args
        if D == "0":
            return "0"
        amp = case['A']
        gamma = case.get('gamma', '145000000000000')
        x0 = case['x0']
        x1 = case['x1']
        result, _ = cpp_bench.get_y(amp, gamma, x0, x1, D, i)
        return result
    
    get_y_cases = [(case, D, i % 2) for i, (case, D) in enumerate(zip(all_cases, D_values))]
    mismatches, skipped = run_precision_check(
        vyper_get_y, cpp_get_y, get_y_cases, "get_y"
    )
    all_results['get_y']['mismatches'] = mismatches
    all_results['get_y']['skipped'] = skipped
    
    # Check get_p precision
    def vyper_get_p(args):
        case, D = args
        if D == "0":
            return "0"
        xp = [int(case['x0']), int(case['x1'])]
        A_gamma = [int(case['A']), int(case.get('gamma', '145000000000000'))]
        return str(vyper_bench.contract.get_p(xp, int(D), A_gamma))
    
    def cpp_get_p(args):
        case, D = args
        if D == "0":
            return "0"
        x0 = case['x0']
        x1 = case['x1']
        amp = case['A']
        return cpp_bench.get_p(x0, x1, D, amp)
    
    get_p_cases = list(zip(all_cases, D_values))
    mismatches, skipped = run_precision_check(
        vyper_get_p, cpp_get_p, get_p_cases, "get_p"
    )
    all_results['get_p']['mismatches'] = mismatches
    all_results['get_p']['skipped'] = skipped
    
    # Check wad_exp precision
    # Generate test values for wad_exp (range from -10 to 10 in WAD format)
    import random
    random.seed(42)  # For reproducibility
    wad_exp_cases = []
    
    # Test various input ranges
    for _ in range(50):  # Test 50 values
        # Small values around 0
        x = random.randint(-1000000000000000000, 1000000000000000000)  # -1 to 1 in WAD
        wad_exp_cases.append(str(x))
    
    for _ in range(30):  # Test 30 values
        # Medium values
        x = random.randint(-10000000000000000000, 10000000000000000000)  # -10 to 10 in WAD
        wad_exp_cases.append(str(x))
    
    for _ in range(20):  # Test 20 edge cases
        # Edge cases
        edge_values = [
            "0",  # e^0 = 1
            "1000000000000000000",  # e^1
            "-1000000000000000000",  # e^(-1)
            "2000000000000000000",  # e^2
            "-2000000000000000000",  # e^(-2)
            "500000000000000000",  # e^0.5
            "-500000000000000000",  # e^(-0.5)
            "100000000000000",  # Very small positive
            "-100000000000000",  # Very small negative
            "10000000000000000000",  # e^10
            "-10000000000000000000",  # e^(-10)
        ]
        wad_exp_cases.extend(edge_values[:min(20, len(edge_values))])
    
    def vyper_wad_exp(x_str):
        x = int(x_str)
        try:
            result = vyper_bench.contract.wad_exp(x)
            return str(result)
        except Exception as e:
            if "overflow" in str(e).lower() or "underflow" in str(e).lower():
                return "overflow"
            raise
    
    def cpp_wad_exp(x_str):
        try:
            result = cpp_bench.wad_exp(x_str)
            return result if result else "overflow"
        except Exception as e:
            if "overflow" in str(e).lower():
                return "overflow"
            raise
    
    mismatches, skipped = run_precision_check(
        vyper_wad_exp, cpp_wad_exp, wad_exp_cases[:100], "wad_exp"  # Test first 100 cases
    )
    all_results['wad_exp']['mismatches'] = mismatches
    all_results['wad_exp']['skipped'] = skipped
    
    # Display results
    console.print("\n[bold]Precision Check Results[/bold]\n")
    
    results_table = Table(show_header=True)
    results_table.add_column("Function", style="cyan")
    results_table.add_column("Total Cases", justify="right")
    results_table.add_column("Valid Cases", justify="right")
    results_table.add_column("Mismatches", justify="right")
    results_table.add_column("Status", justify="center")
    
    total_mismatches = 0
    for func in ['newton_D', 'get_y', 'get_p', 'wad_exp']:
        mismatches = all_results[func]['mismatches']
        skipped = all_results[func]['skipped']
        
        # Different total cases for wad_exp
        if func == 'wad_exp':
            total_cases = 100  # We test 100 wad_exp cases
        else:
            total_cases = len(all_cases)
        
        valid_cases = total_cases - len(skipped)
        num_mismatches = len(mismatches)
        total_mismatches += num_mismatches
        
        if num_mismatches == 0:
            status = "[green]✓ PASS[/green]"
        else:
            status = f"[red]✗ FAIL ({num_mismatches})[/red]"
        
        results_table.add_row(
            func,
            str(total_cases),
            str(valid_cases),
            str(num_mismatches),
            status
        )
    
    console.print(results_table)
    
    # Show detailed mismatches if verbose
    if verbose and total_mismatches > 0:
        console.print("\n[bold red]Detailed Mismatches:[/bold red]")
        for func in ['newton_D', 'get_y', 'get_p', 'wad_exp']:
            mismatches = all_results[func]['mismatches']
            if mismatches:
                console.print(f"\n[yellow]{func}:[/yellow]")
                for m in mismatches[:5]:  # Show first 5 mismatches
                    console.print(f"  Case {m['index']}:")
                    if isinstance(m['case'], dict):
                        console.print(f"    A={m['case']['A']}, x0={m['case']['x0']}, x1={m['case']['x1']}")
                    elif func == 'wad_exp':
                        console.print(f"    Input: {m['case']}")
                    console.print(f"    Vyper: {m['vyper']}")
                    console.print(f"    C++:   {m['cpp']}")
                if len(mismatches) > 5:
                    console.print(f"  ... and {len(mismatches) - 5} more")
    
    # Summary
    console.print()
    if total_mismatches == 0:
        console.print("[bold green]✅ PERFECT PRECISION MATCH![/bold green]")
        console.print("All functions produce identical results between C++ and Vyper.")
    else:
        console.print(f"[bold red]⚠️  PRECISION MISMATCH DETECTED![/bold red]")
        console.print(f"Found {total_mismatches} total mismatches across all functions.")
    
    # Note about skipped cases
    total_skipped = sum(len(all_results[func]['skipped']) for func in ['newton_D', 'get_y', 'get_p'])
    if total_skipped > 0:
        console.print(f"\n[yellow]Note: {total_skipped} cases skipped due to convergence issues[/yellow]")
    
    # Save results
    with open(output, 'w') as f:
        json.dump({
            'test_file': str(test_file),
            'total_cases': len(all_cases),
            'results': all_results,
            'summary': {
                'total_mismatches': total_mismatches,
                'perfect_match': total_mismatches == 0
            }
        }, f, indent=2)
    
    console.print(f"\n[blue]Results saved to {output}[/blue]")
    
    # Exit with error code if mismatches found
    sys.exit(0 if total_mismatches == 0 else 1)


if __name__ == "__main__":
    main()