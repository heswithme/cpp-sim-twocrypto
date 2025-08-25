#!/usr/bin/env python3
"""
Generate benchmark test data for pool configurations and trading sequences.

Time travel actions use absolute timestamps ("timestamp").
"""
import json
import os
import random
from typing import List, Dict, Any
import math

def generate_pool_configs(num_pools: int = 3) -> List[Dict[str, Any]]:
    """Generate diverse pool configurations."""
    pools = []
    
    for i in range(num_pools):
        # Generate pool parameters with realistic ranges
        # Generate fees ensuring out_fee >= mid_fee to satisfy factory checks
        mid_fee_val = random.randint(1, (10**10)//2)
        out_fee_rand = random.randint(1, (10**10)//2)
        out_fee_val = max(out_fee_rand, mid_fee_val)

        pool = {
            "name": f"pool_{i:02d}",
            "A": str(random.randint(2, 10_000) * 10_000),
            "gamma": str(random.randint(10**11, 10**16)),
            "mid_fee": str(mid_fee_val),
            "out_fee": str(out_fee_val),
            "fee_gamma": str(random.randint(10**10, 10**18)),
            "allowed_extra_profit": str(random.randint(10**10, 10**13)),
            "adjustment_step": str(random.randint(10**10, 10**14)),
            "ma_time": str(1+int(random.randint(60, 3600)/math.log(2))),
            "initial_price": str(10**18),  # 1.0
            "initial_liquidity": [
                str(random.randint(1000, 10000) * 10**18),
                str(random.randint(1000, 10000) * 10**18)
            ]
        }
        pools.append(pool)
    
    return pools


def generate_action_sequences(trades_per_sequence: int = 20) -> List[Dict[str, Any]]:
    """Generate a single random trading sequence within safe boundaries.
    Time travel actions are absolute ("timestamp")."""
    START_TS = 1_700_000_000
    actions: List[Dict[str, Any]] = []
    ts = START_TS

    for j in range(trades_per_sequence):
        # Periodic absolute time travel to exercise EMA + donations unlocking
        if j % 3 == 0:
            # add 5 min to 1 hour to timestamp
            ts += random.randint(300, 3600)
            actions.append({"type": "time_travel", "timestamp": ts})

        # Randomly choose between donation add and exchange
        if random.random() < 0.5:
            # donation add
            amt0 = int(random.uniform(0.5, 5) * 10**18)
            amt1 = int(random.uniform(0.5, 5) * 10**18)
            actions.append({
                "type": "add_liquidity",
                "amounts": [str(amt0), str(amt1)],
                "donation": True
            })
        else:
            # exchange
            direction = random.randint(0, 1)
            trade_size = int(random.uniform(0.1, 5) * 10**18)
            actions.append({
                "type": "exchange",
                "i": direction,
                "j": 1 - direction,
                "dx": str(trade_size)
            })

    return [{"name": "default", "start_timestamp": START_TS, "actions": actions}]


def format_json_file(filepath: str):
    """Format a JSON file with proper indentation."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Generate benchmark data files."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate TwoCrypto benchmark data (single sequence)")
    parser.add_argument("--pools", type=int, default=3, help="Number of pools to generate")
    parser.add_argument("--trades", type=int, default=20, help="Trades per sequence")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    # Seed RNG for reproducibility if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate pool configurations (aggregate into pools.json)
    print("Generating pool configurations...")
    pools = generate_pool_configs(num_pools=args.pools)
    pools_file = os.path.join(data_dir, "pools.json")
    
    with open(pools_file, 'w') as f:
        json.dump({"pools": pools}, f, indent=2)
    
    print(f"✓ Generated {len(pools)} pool configurations")
    
    # Generate action sequences (aggregate into sequences.json)
    print("Generating action sequences...")
    sequences = generate_action_sequences(trades_per_sequence=args.trades)
    sequences_file = os.path.join(data_dir, "sequences.json")
    
    with open(sequences_file, 'w') as f:
        json.dump({"sequences": sequences}, f, indent=2)
    
    print(f"✓ Generated 1 sequence with {len(sequences[0]['actions'])} trades")
    
    # Summary
    total_tests = len(pools)
    print(f"\n✓ Total tests: {total_tests}")
    print(f"✓ Data saved to {data_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
