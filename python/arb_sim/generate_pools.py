#!/usr/bin/env python3
"""
Generate a grid of pool configs over two parameters.

Smart detect which params are ranged (len>1) vs fixed (len==1);
if both have len==1, output a single pool.

Usage examples:
  python3 python/arb/generate_pools.py --out python/arb/run_data/pools.json \
    --param-x A --x-values 50000,100000,150000 \
    --param-y mid_fee_bps --y-values 2.5,3.0

  python3 python/arb/generate_pools.py --out python/arb/run_data/pools.json \
    --param-x A --x-values 100000 \
    --param-y out_fee_bps --y-values 5.0
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List


def to_wad(x: float) -> str:
    return str(int(x * 1e18))


def to_fee_bps_scaled(x_bps: float) -> str:
    return str(int((x_bps / 1e4) * 1e10))


BASE_POOL = {
    "initial_liquidity": [to_wad(1e6), to_wad(1e6)],
    "A": "100000",
    "gamma": "0",
    "mid_fee": to_fee_bps_scaled(3.0),
    "out_fee": to_fee_bps_scaled(5.0),
    "fee_gamma": to_wad(0.23),
    "allowed_extra_profit": to_wad(1e-3),
    "adjustment_step": to_wad(1e-3),
    "ma_time": "600",
    "initial_price": to_wad(1.0),
}

BASE_COSTS = {
    "arb_fee_bps": 10.0,
    "gas_coin0": to_wad(0.0),
    "max_trade_frac": 0.25,
    "use_volume_cap": False,
    "volume_cap_mult": 1.0,
}


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def apply_param(pool: Dict, costs: Dict, name: str, value: float):
    if name == "A":
        pool["A"] = str(value)
    elif name == "mid_fee_bps":
        pool["mid_fee"] = to_fee_bps_scaled(value)
    elif name == "out_fee_bps":
        pool["out_fee"] = to_fee_bps_scaled(value)
    elif name == "fee_gamma":
        pool["fee_gamma"] = to_wad(value)
    elif name == "allowed_extra_profit":
        pool["allowed_extra_profit"] = to_wad(value)
    elif name == "adjustment_step":
        pool["adjustment_step"] = to_wad(value)
    elif name == "ma_time":
        pool["ma_time"] = str(int(value))
    elif name == "initial_price":
        pool["initial_price"] = to_wad(value)
    elif name == "arb_fee_bps":
        costs["arb_fee_bps"] = value
    elif name == "gas_coin0":
        costs["gas_coin0"] = to_wad(value)
    elif name == "max_trade_frac":
        costs["max_trade_frac"] = value
    elif name == "use_volume_cap":
        costs["use_volume_cap"] = bool(value)
    elif name == "volume_cap_mult":
        costs["volume_cap_mult"] = value
    else:
        raise ValueError(f"Unsupported parameter name: {name}")


def main():
    ap = argparse.ArgumentParser(description="Generate pools grid over two parameters")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    ap.add_argument("--param-x", type=str, required=True, help="First parameter name (e.g., A, mid_fee_bps)")
    ap.add_argument("--x-values", type=str, required=True, help="Comma-separated values for param X")
    ap.add_argument("--param-y", type=str, required=True, help="Second parameter name")
    ap.add_argument("--y-values", type=str, required=True, help="Comma-separated values for param Y")
    args = ap.parse_args()

    x_vals = parse_csv_floats(args.x_values)
    y_vals = parse_csv_floats(args.y_values)

    pools = []
    for xv in x_vals:
        for yv in y_vals:
            p = dict(BASE_POOL)
            c = dict(BASE_COSTS)
            apply_param(p, c, args.param_x, xv)
            apply_param(p, c, args.param_y, yv)
            tag = f"{args.param_x}_{xv}_{args.param_y}_{yv}".replace('.', 'p')
            pools.append({"tag": tag, "pool": p, "costs": c})

    out = {"pools": pools, "params": {"x": args.param_x, "y": args.param_y}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(pools)} pool configs to {out_path}")


if __name__ == "__main__":
    main()

