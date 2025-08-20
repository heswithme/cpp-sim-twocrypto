#!/usr/bin/env python3
"""
Generate benchmark test data for pool configurations and trading sequences
"""
import json
import os
import random
from typing import List, Dict, Any


def generate_pool_configs(num_pools: int = 3) -> List[Dict[str, Any]]:
    """Generate diverse pool configurations."""
    pools = []
    
    for i in range(num_pools):
        # Generate pool parameters with realistic ranges
        # Generate fees ensuring out_fee >= mid_fee to satisfy factory checks
        mid_fee_val = random.randint(1_000_000, 50_000_000)
        out_fee_rand = random.randint(30_000_000, 60_000_000)
        out_fee_val = max(out_fee_rand, mid_fee_val)

        pool = {
            "name": f"pool_{i:02d}",
            "A": str(random.randint(100000, 100000000)),
            "gamma": str(random.randint(10**11, 10**16)),
            "mid_fee": str(mid_fee_val),
            "out_fee": str(out_fee_val),
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


def generate_action_sequences(num_sequences: int = 4, trades_per_sequence: int = 20) -> List[Dict[str, Any]]:
    """Generate diverse trading sequences with majority donations and time travel actions."""
    sequences: List[Dict[str, Any]] = []
    START_TS = 1_700_000_000

    # Build a list of patterns with majority donations
    base_patterns = ["donations"] * max(1, num_sequences * 2 // 3)  # ~2/3 donations
    other_patterns_pool = ["balanced", "trending_up", "trending_down", "volatile", "accumulation"]
    while len(base_patterns) < num_sequences:
        base_patterns.append(other_patterns_pool[len(base_patterns) % len(other_patterns_pool)])

    for i, pattern in enumerate(base_patterns[:num_sequences]):
        actions: List[Dict[str, Any]] = []
        current_ts = START_TS

        for j in range(trades_per_sequence):
            # Periodically insert absolute time travel actions to exercise
            # donation unlocking and EMA dynamics in a deterministic manner.
            if j % 3 == 0:
                # jump forward 5 to 60 minutes
                current_ts += random.randint(300, 3600)
                actions.append({
                    "type": "time_travel",
                    "timestamp": current_ts
                })

            if pattern == "donations":
                # Alternate donation adds and small exchanges
                if j % 2 == 0:
                    amt0 = int(random.uniform(0.5, 5) * 10**18)
                    amt1 = int(random.uniform(0.5, 5) * 10**18)
                    actions.append({
                        "type": "add_liquidity",
                        "amounts": [str(amt0), str(amt1)],
                        "donation": True
                    })
                    continue
                direction = random.randint(0, 1)
                trade_size = int(random.uniform(0.1, 2) * 10**18)
                actions.append({
                    "type": "exchange",
                    "i": direction,
                    "j": 1 - direction,
                    "dx": str(trade_size)
                })
                continue

            # Non-donation patterns
            if pattern == "balanced":
                direction = j % 2
            elif pattern == "trending_up":
                direction = 0 if random.random() < 0.7 else 1
            elif pattern == "trending_down":
                direction = 1 if random.random() < 0.7 else 0
            elif pattern == "volatile":
                direction = random.randint(0, 1)
            else:  # accumulation
                direction = 0 if random.random() < 0.6 else 1

            if pattern == "volatile":
                trade_size = int(random.uniform(1, 100) * 10**18)
            elif pattern == "accumulation":
                trade_size = int(random.uniform(0.1, 5) * 10**18)
            else:
                trade_size = int(random.uniform(1, 20) * 10**18)

            actions.append({
                "type": "exchange",
                "i": direction,
                "j": 1 - direction,
                "dx": str(trade_size)
            })

        sequence = {
            "name": f"{pattern}_sequence_{i:02d}",
            "start_timestamp": START_TS,
            "actions": actions,
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
    import argparse
    parser = argparse.ArgumentParser(description="Generate TwoCrypto benchmark data")
    parser.add_argument("--pools", type=int, default=3, help="Number of pools to generate")
    parser.add_argument("--sequences", type=int, default=4, help="Number of sequences to generate")
    parser.add_argument("--trades", type=int, default=20, help="Trades per sequence")
    args = parser.parse_args()
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate pool configurations
    print("Generating pool configurations...")
    pools = generate_pool_configs(num_pools=args.pools)
    pools_file = os.path.join(data_dir, "benchmark_pools.json")
    
    with open(pools_file, 'w') as f:
        json.dump({"pools": pools}, f, indent=2)
    
    print(f"✓ Generated {len(pools)} pool configurations")
    
    # Generate action sequences
    print("Generating action sequences...")
    sequences = generate_action_sequences(num_sequences=args.sequences, trades_per_sequence=args.trades)
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
