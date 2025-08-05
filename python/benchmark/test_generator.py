"""
Generate test cases for benchmarking math functions.
Can be run directly: uv run benchmark/test_generator.py
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import click
from rich.console import Console

console = Console()


class TestCaseGenerator:
    """Generate diverse test cases for StableswapMath functions."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.test_cases = []
    
    def generate_realistic_cases(self, n: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic test cases based on typical pool parameters."""
        cases = []
        
        for i in range(n):
            # A values typically range from 100 to 10000
            A = int(np.random.uniform(10, 10000)) * 10000  # Already multiplied by A_MULTIPLIER
            
            # Generate realistic balance ranges
            # Small pools: 1e18 to 1e21
            # Medium pools: 1e21 to 1e24
            # Large pools: 1e24 to 1e27
            pool_size = np.random.choice(['small', 'medium', 'large'], p=[0.3, 0.5, 0.2])
            
            if pool_size == 'small':
                x0 = str(int(10 ** np.random.uniform(18, 21)))
                x1 = str(int(10 ** np.random.uniform(18, 21)))
            elif pool_size == 'medium':
                x0 = str(int(10 ** np.random.uniform(21, 24)))
                x1 = str(int(10 ** np.random.uniform(21, 24)))
            else:  # large
                x0 = str(int(10 ** np.random.uniform(24, 27)))
                x1 = str(int(10 ** np.random.uniform(24, 27)))
            
            # Add some imbalance
            imbalance = np.random.uniform(0.5, 2.0)
            x1 = str(int(float(x1) * imbalance))
            
            cases.append({
                'id': i,
                'A': str(A),
                'gamma': '145000000000000',  # Fixed for stableswap
                'x0': x0,
                'x1': x1,
                'pool_size': pool_size,
                'description': f'{pool_size} pool with A={A//10000}'
            })
        
        return cases
    
    def generate_edge_cases(self) -> List[Dict[str, Any]]:
        """Generate edge cases to test boundary conditions."""
        edge_cases = []
        
        # Very small values
        edge_cases.append({
            'id': 'edge_small',
            'A': str(100 * 10000),
            'x0': str(10**18),  # 1 token
            'x1': str(10**18),  # 1 token
            'description': 'Minimum viable pool'
        })
        
        # Very large values
        edge_cases.append({
            'id': 'edge_large',
            'A': str(10000 * 10000),
            'x0': str(10**27),  # 1 billion tokens
            'x1': str(10**27),
            'description': 'Maximum pool size'
        })
        
        # Highly imbalanced
        edge_cases.append({
            'id': 'edge_imbalanced',
            'A': str(1000 * 10000),
            'x0': str(10**24),
            'x1': str(10**18),  # 1 million times smaller
            'description': 'Highly imbalanced pool'
        })
        
        # Perfect balance
        edge_cases.append({
            'id': 'edge_balanced',
            'A': str(1000 * 10000),
            'x0': str(10**22),
            'x1': str(10**22),
            'description': 'Perfectly balanced pool'
        })
        
        for case in edge_cases:
            case['gamma'] = '145000000000000'
        
        return edge_cases
    
    def generate_stress_cases(self, n: int = 100) -> List[Dict[str, Any]]:
        """Generate many cases for stress testing."""
        cases = []
        
        for i in range(n):
            A = str(int(np.random.uniform(10, 10000)) * 10000)
            
            # Random powers between 18 and 27
            power0 = np.random.uniform(18, 27)
            power1 = np.random.uniform(18, 27)
            
            x0 = str(int(10 ** power0))
            x1 = str(int(10 ** power1))
            
            cases.append({
                'id': f'stress_{i}',
                'A': A,
                'gamma': '145000000000000',
                'x0': x0,
                'x1': x1,
                'description': f'Stress test case {i}'
            })
        
        return cases
    
    def save_test_suite(self, output_path: Path):
        """Save complete test suite to JSON."""
        test_suite = {
            'metadata': {
                'version': '1.0',
                'description': 'StableswapMath test cases for benchmarking',
                'total_cases': 0
            },
            'realistic': self.generate_realistic_cases(100),
            'edge_cases': self.generate_edge_cases(),
            'stress': self.generate_stress_cases(100)
        }
        
        # Calculate total
        total = len(test_suite['realistic']) + len(test_suite['edge_cases']) + len(test_suite['stress'])
        test_suite['metadata']['total_cases'] = total
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(test_suite, f, indent=2)
        
        console.print(f"[green]✓ Generated {total} test cases[/green]")
        console.print(f"  - Realistic: {len(test_suite['realistic'])}")
        console.print(f"  - Edge cases: {len(test_suite['edge_cases'])}")
        console.print(f"  - Stress test: {len(test_suite['stress'])}")
        console.print(f"[blue]Saved to: {output_path}[/blue]")
        
        return test_suite


def generate_test_cases(output_file: str = "test_cases.json"):
    """Generate and save test cases."""
    generator = TestCaseGenerator()
    output_path = Path(output_file)
    generator.save_test_suite(output_path)
    return output_path


@click.command()
@click.option('--output', '-o', default='benchmark/test_cases.json', help='Output file for test cases')
@click.option('--realistic', '-r', default=100, help='Number of realistic test cases')
@click.option('--stress', '-s', default=100, help='Number of stress test cases')
def main(output, realistic, stress):
    """Generate test cases for StableswapMath benchmarking."""
    console.print("[bold blue]Test Case Generator[/bold blue]")
    console.print(f"Generating {realistic} realistic cases and {stress} stress cases...")
    
    generator = TestCaseGenerator()
    test_suite = {
        'metadata': {
            'version': '1.0',
            'description': 'StableswapMath test cases'
        },
        'realistic': generator.generate_realistic_cases(realistic),
        'edge_cases': generator.generate_edge_cases(),
        'stress': generator.generate_stress_cases(stress)
    }
    
    # Save test suite
    output_path = Path(output)
    with open(output_path, 'w') as f:
        json.dump(test_suite, f, indent=2)
    
    total = len(test_suite['realistic']) + len(test_suite['edge_cases']) + len(test_suite['stress'])
    console.print(f"\n[green]✅ Generated {total} test cases[/green]")
    console.print(f"[blue]Saved to: {output_path}[/blue]")


if __name__ == "__main__":
    main()