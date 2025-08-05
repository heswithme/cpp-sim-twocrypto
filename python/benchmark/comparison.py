"""
Compare results and generate reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

console = Console()


class ResultComparator:
    """Compare benchmark results between implementations."""
    
    def __init__(self, vyper_results: Dict, cpp_results: Dict):
        self.vyper = vyper_results
        self.cpp = cpp_results
        self.comparison = {}
    
    def compare_accuracy(self) -> Dict[str, Any]:
        """Compare numerical accuracy of results."""
        accuracy = {}
        
        for func in ['newton_D', 'get_y', 'get_p']:
            vyper_vals = self.vyper[func]['results']
            cpp_vals = self.cpp[func]['results']
            
            matches = 0
            max_diff = 0
            total_diff = 0
            
            for v_val, c_val in zip(vyper_vals, cpp_vals):
                v = int(v_val)
                c = int(c_val)
                
                if v == c:
                    matches += 1
                else:
                    diff = abs(v - c)
                    rel_diff = diff / max(v, 1)  # Relative difference
                    max_diff = max(max_diff, rel_diff)
                    total_diff += rel_diff
            
            accuracy[func] = {
                'exact_matches': matches,
                'total_cases': len(vyper_vals),
                'match_rate': matches / len(vyper_vals) * 100,
                'max_relative_diff': max_diff,
                'avg_relative_diff': total_diff / len(vyper_vals)
            }
        
        return accuracy
    
    def compare_performance(self) -> Dict[str, Any]:
        """Compare performance metrics."""
        performance = {}
        
        for func in ['newton_D', 'get_y', 'get_p']:
            vyper_times = self.vyper[func]['times_ns']
            cpp_times = self.cpp[func]['times_ns']
            
            speedups = [v/c for v, c in zip(vyper_times, cpp_times) if c > 0]
            
            performance[func] = {
                'vyper_avg_ms': self.vyper[func]['avg_time_ms'],
                'cpp_avg_ms': self.cpp[func]['avg_time_ms'],
                'vyper_total_ms': self.vyper[func]['total_time_ms'],
                'cpp_total_ms': self.cpp[func]['total_time_ms'],
                'avg_speedup': sum(speedups) / len(speedups),
                'min_speedup': min(speedups),
                'max_speedup': max(speedups),
                'total_speedup': self.vyper[func]['total_time_ms'] / self.cpp[func]['total_time_ms']
            }
        
        # Overall performance
        performance['overall'] = {
            'vyper_total_ms': self.vyper['total_time_ms'],
            'cpp_total_ms': self.cpp['total_time_ms'],
            'speedup': self.vyper['total_time_ms'] / self.cpp['total_time_ms']
        }
        
        return performance
    
    def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        accuracy = self.compare_accuracy()
        performance = self.compare_performance()
        
        # Create accuracy table
        accuracy_table = Table(title="Accuracy Comparison", show_header=True)
        accuracy_table.add_column("Function", style="cyan")
        accuracy_table.add_column("Match Rate", justify="right", style="green")
        accuracy_table.add_column("Max Rel. Diff", justify="right")
        accuracy_table.add_column("Avg Rel. Diff", justify="right")
        
        for func in ['newton_D', 'get_y', 'get_p']:
            acc = accuracy[func]
            accuracy_table.add_row(
                func,
                f"{acc['match_rate']:.1f}%",
                f"{acc['max_relative_diff']:.2e}",
                f"{acc['avg_relative_diff']:.2e}"
            )
        
        # Create performance table
        perf_table = Table(title="Performance Comparison", show_header=True)
        perf_table.add_column("Function", style="cyan")
        perf_table.add_column("Vyper (ms)", justify="right")
        perf_table.add_column("C++ (ms)", justify="right")
        perf_table.add_column("Speedup", justify="right", style="bold green")
        
        for func in ['newton_D', 'get_y', 'get_p']:
            perf = performance[func]
            perf_table.add_row(
                func,
                f"{perf['vyper_total_ms']:.2f}",
                f"{perf['cpp_total_ms']:.2f}",
                f"{perf['total_speedup']:.1f}x"
            )
        
        # Add overall row
        perf_table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{performance['overall']['vyper_total_ms']:.2f}[/bold]",
            f"[bold]{performance['overall']['cpp_total_ms']:.2f}[/bold]",
            f"[bold yellow]{performance['overall']['speedup']:.1f}x[/bold yellow]"
        )
        
        # Display tables
        console.print("\n")
        console.print(accuracy_table)
        console.print("\n")
        console.print(perf_table)
        
        self.comparison = {
            'accuracy': accuracy,
            'performance': performance
        }
        
        return self.comparison
    
    def generate_plots(self, output_dir: Path):
        """Generate performance visualization plots."""
        output_dir.mkdir(exist_ok=True)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Comparison', 'Speedup Distribution',
                          'Time per Function', 'Accuracy'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        perf = self.comparison['performance']
        acc = self.comparison['accuracy']
        
        # Performance comparison bar chart
        functions = ['newton_D', 'get_y', 'get_p']
        vyper_times = [perf[f]['vyper_total_ms'] for f in functions]
        cpp_times = [perf[f]['cpp_total_ms'] for f in functions]
        
        fig.add_trace(
            go.Bar(name='Vyper', x=functions, y=vyper_times, marker_color='blue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='C++', x=functions, y=cpp_times, marker_color='green'),
            row=1, col=1
        )
        
        # Speedup distribution
        speedups = [perf[f]['avg_speedup'] for f in functions]
        fig.add_trace(
            go.Bar(x=functions, y=speedups, marker_color='orange', showlegend=False),
            row=1, col=2
        )
        
        # Time breakdown
        fig.add_trace(
            go.Bar(name='Avg Time (ms)', 
                  x=['Vyper newton_D', 'C++ newton_D', 'Vyper get_y', 'C++ get_y', 'Vyper get_p', 'C++ get_p'],
                  y=[perf['newton_D']['vyper_avg_ms'], perf['newton_D']['cpp_avg_ms'],
                     perf['get_y']['vyper_avg_ms'], perf['get_y']['cpp_avg_ms'],
                     perf['get_p']['vyper_avg_ms'], perf['get_p']['cpp_avg_ms']],
                  marker_color=['blue', 'green', 'blue', 'green', 'blue', 'green'],
                  showlegend=False),
            row=2, col=1
        )
        
        # Accuracy comparison
        match_rates = [acc[f]['match_rate'] for f in functions]
        fig.add_trace(
            go.Bar(x=functions, y=match_rates, marker_color='purple', showlegend=False),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="StableswapMath Benchmark Results",
            height=800,
            showlegend=True
        )
        fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Speedup Factor", row=1, col=2)
        fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Match Rate (%)", row=2, col=2)
        
        # Save plot
        plot_path = output_dir / "benchmark_results.html"
        fig.write_html(str(plot_path))
        console.print(f"[green]✓ Plots saved to {plot_path}[/green]")
        
        return plot_path
    
    def save_results(self, output_path: Path):
        """Save comparison results to JSON."""
        with open(output_path, 'w') as f:
            json.dump({
                'vyper': self.vyper,
                'cpp': self.cpp,
                'comparison': self.comparison
            }, f, indent=2)
        
        console.print(f"[green]✓ Results saved to {output_path}[/green]")