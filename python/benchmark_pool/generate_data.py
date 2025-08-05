#!/usr/bin/env python3
"""
Generate benchmark test data for pool configurations and trading sequences
"""
import json
import os
import random
from typing import List, Dict, Any


def generate_pool_configs(num_pools: int = 10) -> List[Dict[str, Any]]:
    """Generate diverse pool configurations."""
    pools = []
    
    for i in range(num_pools):
        # Generate pool parameters with realistic ranges
        pool = {
            "name": f"pool_{i:02d}",
            "A": str(random.randint(100000, 100000000)),
            "gamma": str(random.randint(10**11, 10**16)),
            "mid_fee": str(random.randint(1000000, 50000000)),
            "out_fee": str(random.randint(30000000, 60000000)),
            "fee_gamma": str(random.randint(10**10, 10**18)),
            "allowed_extra_profit": str(random.randint(10**10, 10**13)),
            "adjustment_step": str(random.randint(10**10, 10**14)),
            "ma_time": str(random.randint(60, 3600)),
            "initial_price": "1000000000000000000",  # 1.0
            "initial_liquidity": [
                str(random.randint(1000, 10000) * 10**18),
                str(random.randint(1000, 10000) * 10**18)
            ]
        }
        pools.append(pool)
    
    return pools


def generate_action_sequences(num_sequences: int = 5, trades_per_sequence: int = 20) -> List[Dict[str, Any]]:
    """Generate diverse trading sequences."""
    sequences = []
    patterns = ["balanced", "trending_up", "trending_down", "volatile", "accumulation"]
    
    for i in range(num_sequences):
        pattern = patterns[i % len(patterns)]
        actions = []
        
        for j in range(trades_per_sequence):
            if pattern == "balanced":
                # Alternate between directions
                direction = j % 2
            elif pattern == "trending_up":
                # More buys of token 1
                direction = 0 if random.random() < 0.7 else 1
            elif pattern == "trending_down":
                # More sells of token 1
                direction = 1 if random.random() < 0.7 else 0
            elif pattern == "volatile":
                # Large random trades
                direction = random.randint(0, 1)
                multiplier = random.uniform(5, 100)
            else:  # accumulation
                # Small steady trades
                direction = 0 if random.random() < 0.6 else 1
                multiplier = random.uniform(0.1, 5)
            
            # Set trade size based on pattern
            if pattern == "volatile":
                trade_size = int(random.uniform(1, 100) * 10**18)
            elif pattern == "accumulation":
                trade_size = int(random.uniform(0.1, 5) * 10**18)
            else:
                trade_size = int(random.uniform(1, 20) * 10**18)
            
            action = {
                "type": "exchange",
                "i": direction,
                "j": 1 - direction,
                "dx": str(trade_size),
                "time_delta": random.randint(10, 1800)
            }
            actions.append(action)
        
        sequence = {
            "name": f"{pattern}_sequence_{i:02d}",
            "actions": actions
        }
        sequences.append(sequence)
    
    return sequences


def format_json_file(filepath: str):
    """Format a JSON file with proper indentation."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Generate benchmark data files."""
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate pool configurations
    print("Generating pool configurations...")
    pools = generate_pool_configs(num_pools=10)
    pools_file = os.path.join(data_dir, "benchmark_pools.json")
    
    with open(pools_file, 'w') as f:
        json.dump({"pools": pools}, f, indent=2)
    
    print(f"✓ Generated {len(pools)} pool configurations")
    
    # Generate action sequences
    print("Generating action sequences...")
    sequences = generate_action_sequences(num_sequences=5, trades_per_sequence=20)
    sequences_file = os.path.join(data_dir, "benchmark_sequences.json")
    
    with open(sequences_file, 'w') as f:
        json.dump({"sequences": sequences}, f, indent=2)
    
    print(f"✓ Generated {len(sequences)} sequences with {len(sequences[0]['actions'])} trades each")
    
    # Summary
    total_tests = len(pools) * len(sequences)
    print(f"\n✓ Total test combinations: {total_tests}")
    print(f"✓ Data saved to {data_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())