"""
Vyper implementation benchmark using titanoboa.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import boa
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn

console = Console()


class VyperBenchmark:
    """Benchmark Vyper StableswapMath implementation."""
    
    def __init__(self):
        self.contract = None
        self.deploy_contract()
    
    def deploy_contract(self):
        """Deploy the Vyper StableswapMath contract."""
        console.print("[yellow]Deploying Vyper contract...[/yellow]")
        
        contract_path = Path(__file__).parent.parent / "twocrypto-ng" / "contracts" / "main" / "StableswapMath.vy"
        
        if not contract_path.exists():
            raise FileNotFoundError(f"Vyper contract not found at {contract_path}")
        
        with boa.env.prank("0x0000000000000000000000000000000000000001"):
            self.contract = boa.load_partial(str(contract_path)).deploy()
        
        console.print("[green]✓ Vyper contract deployed[/green]")
    
    def benchmark_newton_D(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Benchmark newton_D function."""
        results = []
        times = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking Vyper newton_D...", total=len(test_cases))
            
            for case in test_cases:
                try:
                    A = int(case['A'])
                    gamma = int(case['gamma'])
                    xp = [int(case['x0']), int(case['x1'])]
                    
                    # Warm-up call (not timed)
                    _ = self.contract.newton_D(A, gamma, xp)
                    
                    # Timed call
                    start = time.perf_counter_ns()
                    result = self.contract.newton_D(A, gamma, xp)
                    elapsed = time.perf_counter_ns() - start
                    
                    results.append(str(result))
                    times.append(elapsed)
                except Exception as e:
                    # Skip non-convergent cases
                    if "Did not converge" in str(e):
                        results.append("0")  # Mark as failed
                        times.append(0)
                    else:
                        raise
                
                progress.update(task, advance=1)
        
        return {
            'function': 'newton_D',
            'results': results,
            'times_ns': times,
            'avg_time_ms': sum(times) / len(times) / 1_000_000,
            'total_time_ms': sum(times) / 1_000_000
        }
    
    def benchmark_get_y(self, test_cases: List[Dict], D_values: List[str]) -> Dict[str, Any]:
        """Benchmark get_y function."""
        results = []
        times = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking Vyper get_y...", total=len(test_cases))
            
            for i, (case, D) in enumerate(zip(test_cases, D_values)):
                A = int(case['A'])
                gamma = int(case['gamma'])
                xp = [int(case['x0']), int(case['x1'])]
                D = int(D)
                j = i % 2  # Alternate between coins
                
                # Warm-up call
                _ = self.contract.get_y(A, gamma, xp, D, j)
                
                # Timed call
                start = time.perf_counter_ns()
                result = self.contract.get_y(A, gamma, xp, D, j)
                elapsed = time.perf_counter_ns() - start
                
                # get_y returns [y, k] - we only need y
                results.append(str(result[0]))
                times.append(elapsed)
                progress.update(task, advance=1)
        
        return {
            'function': 'get_y',
            'results': results,
            'times_ns': times,
            'avg_time_ms': sum(times) / len(times) / 1_000_000,
            'total_time_ms': sum(times) / 1_000_000
        }
    
    def benchmark_get_p(self, test_cases: List[Dict], D_values: List[str]) -> Dict[str, Any]:
        """Benchmark get_p function."""
        results = []
        times = []
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking Vyper get_p...", total=len(test_cases))
            
            for case, D in zip(test_cases, D_values):
                xp = [int(case['x0']), int(case['x1'])]
                D = int(D)
                A_gamma = [int(case['A']), int(case['gamma'])]
                
                # Warm-up call
                _ = self.contract.get_p(xp, D, A_gamma)
                
                # Timed call
                start = time.perf_counter_ns()
                result = self.contract.get_p(xp, D, A_gamma)
                elapsed = time.perf_counter_ns() - start
                
                results.append(str(result))
                times.append(elapsed)
                progress.update(task, advance=1)
        
        return {
            'function': 'get_p',
            'results': results,
            'times_ns': times,
            'avg_time_ms': sum(times) / len(times) / 1_000_000,
            'total_time_ms': sum(times) / 1_000_000
        }
    
    def run_benchmark(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        console.print("\n[bold blue]Running Vyper benchmarks[/bold blue]")
        
        # Benchmark newton_D and collect D values
        newton_d_results = self.benchmark_newton_D(test_cases)
        D_values = newton_d_results['results']
        
        # Benchmark get_y using D values
        get_y_results = self.benchmark_get_y(test_cases, D_values)
        
        # Benchmark get_p
        get_p_results = self.benchmark_get_p(test_cases, D_values)
        
        total_time = (newton_d_results['total_time_ms'] + 
                     get_y_results['total_time_ms'] + 
                     get_p_results['total_time_ms'])
        
        console.print(f"\n[green]✓ Vyper benchmarks complete[/green]")
        console.print(f"  Total time: {total_time:.2f}ms")
        
        return {
            'implementation': 'vyper',
            'newton_D': newton_d_results,
            'get_y': get_y_results,
            'get_p': get_p_results,
            'total_time_ms': total_time,
            'D_values': D_values  # Store for comparison
        }