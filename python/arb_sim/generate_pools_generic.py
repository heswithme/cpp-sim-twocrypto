#!/usr/bin/env python3
"""
Generate a fixed grid of pool configurations (no CLI args) with simple rules:

- Pool parameters are specified in their native units:
  - Integers for fees (1e10), WAD-like fields (1e18), balances (1e18).
  - Floats are allowed for harness-only fields like donation_apy (plain fraction, e.g., 0.05).
- Values are stringified in the output JSON under the "pool" object; floats are
  preserved as decimal strings.
- Uses numpy.logspace for stable grids.

Writes a pretty JSON to python/arb_sim/run_data/pools.json with entries of the
form {tag, pool, costs}.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from time import time
from pool_helpers import _first_candle_ts, _initial_price_from_file, strify_pool
import math

FEE_EQUALIZE = False
# -------------------- Grid Definition --------------------
GRID_SIZE = 1
N_GRID_X = GRID_SIZE
N_GRID_Y = GRID_SIZE

# 1. A-mid_fee log 

X_name = "mid_fee"  
xmin = int(1/10_000*10**10)
xmax = int(500/10_000*10**10)
xlogspace = True
FEE_EQUALIZE = True

Y_name = "A"  
ymin = 1*10_000
ymax =  200*10_000
ylogspace = True


# # 2. A-mid_fee zoom_lin

# X_name = "mid_fee"  
# xmin = int(10/10_000*10**10)
# xmax = int(50/10_000*10**10)
# xlogspace = False
# FEE_EQUALIZE = True

# Y_name = "A"  
# ymin = 1*10_000
# ymax =  50*10_000
# ylogspace = False

# # 3. mid_fee-out_fee 
# X_name = "mid_fee"  
# xmin = int(1/10_000*10**10)
# xmax = int(100/10_000*10**10)
# xlogspace = False

# Y_name = "out_fee"  
# ymin = int(1/10_000*10**10)
# ymax = int(100/10_000*10**10)
# ylogspace = False

# 4. out_fee-fee_gamma

# X_name = "fee_gamma"  
# xmin = int(1e-6 * 10**18)
# xmax = int(0.1 * 10**18)
# xlogspace = True

# Y_name = "out_fee"  
# ymin = int(10/10_000*10**10)
# ymax = int(100/10_000*10**10)
# ylogspace = False


# 5. A-ma_time

# X_name = "ma_time"  
# xmin = 10*60/math.log(2)
# xmax = 24*60*60/math.log(2)
# xlogspace = False

# Y_name = "A"  
# ymin = 1*10_000
# ymax =  50*10_000
# ylogspace = False


# X_name = "mid_fee"  
# xmin = int(1/10_000*10**10)
# xmax = int(100/10_000*10**10)
# xlogspace = True

# X_name = "donation_apy"  
# xmin = 0.01
# xmax = 0.1
# xlogspace = False


# X_name = "fee_gamma"  
# xmin = int(1e-5 * 10**18)
# xmax = int(0.1 * 10**18)
# xlogspace = True


# Y_name = "A"  
# ymin = 1*10_000
# ymax =  100*10_000
# ylogspace = False


# X_name = "mid_fee"  
# xmin = int(4850 / 10_000 * 10**10)
# xmax = xmin
# xlogspace = False

# Y_name = "A"  # 
# ymin = 120*10_000
# ymax =  ymin
# ylogspace = False
# N_GRID_X = 1
# N_GRID_Y = 1


if xlogspace:
    X_vals = np.logspace(np.log10(xmin), np.log10(xmax), N_GRID_X).round().tolist()
else:
    X_vals = np.linspace(xmin, xmax, N_GRID_X).tolist()

if ylogspace:
    Y_vals = np.logspace(np.log10(ymin), np.log10(ymax), N_GRID_Y).round().tolist()
else:
    Y_vals = np.linspace(ymin, ymax, N_GRID_Y).tolist()

# #optionally int-ify
# X_vals = [int(x) for x in X_vals]
# Y_vals = [int(x) for x in Y_vals]

DEFAULT_DATAFILE = "python/arb_sim/trade_data/btcusd/binance-2025.json"


START_TS = _first_candle_ts(DEFAULT_DATAFILE)
init_price = _initial_price_from_file(DEFAULT_DATAFILE)
init_liq = 1_000_000 # in coin0


INVERT_LIQ = False
if INVERT_LIQ:
    #inverse if necessary (make it coin1)
    init_liq *= init_price

# -------------------- Base Templates --------------------
BASE_POOL = {
    # All values are integers in their native units
    "initial_liquidity": [int(init_liq * 10**18//2), int(init_liq * 10**18//2 / init_price)],
    "A": 20 * 10_000,
    "gamma": 10**14, #unused in twocrypto
    "mid_fee": int(5 / 10_000 * 10**10),
    "out_fee": int(50 / 10_000 * 10**10),
    "fee_gamma": int(0.001 * 10**18),
    "allowed_extra_profit": int(1e-12 * 10**18),
    "adjustment_step": int(1e-7 * 10**18),
    "ma_time": 866, #int(86400 / math.log(2)), #5200,
    "initial_price": int(init_price * 10**18),
    "start_timestamp": START_TS,

    # Donations (harness-only):
    # - donation_apy: plain fraction per year (0.05 => 5%).
    # - donation_frequency: seconds between donations.
    # - donation_coins_ratio: fraction of donation in coin1 (0=all coin0, 1=all coin1)
    "donation_apy": 0.10,
    "donation_frequency": int(7*86400),
    "donation_coins_ratio": 0.5,
}

BASE_COSTS = {
    "arb_fee_bps": 1.0,
    "gas_coin0": 0.0,
    "use_volume_cap": False,
    "volume_cap_mult": 1,
}


def build_grid():
    pools = []
    for xv in X_vals:
        for yv in Y_vals:
            pool = dict(BASE_POOL)  # start from base with all params (ints)
            # Apply X then Y onto pool
            pool[X_name] = xv
            pool[Y_name] = yv
            # Enforce out_fee >= mid_fee
            mid_fee_val = int(pool.get("mid_fee", 0))
            cur_out_val = int(pool.get("out_fee", 0))
            # pool["out_fee"] = max(mid_fee_val, cur_out_val)
            pool["mid_fee"] = int(mid_fee_val)
            pool["out_fee"] = mid_fee_val if FEE_EQUALIZE else max(mid_fee_val, cur_out_val) + 1
            costs = dict(BASE_COSTS)
            tag_x = f"{X_name}_{xv}"
            tag_y = f"{Y_name}_{yv}"
            tag = f"{tag_x}__{tag_y}"
            pools.append({"tag": tag, "pool": strify_pool(pool), "costs": costs})
    return pools


def main():
    pools = build_grid()
    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid": {
                "X": {"name": X_name, "min": X_vals[0], "max": X_vals[-1], "n": len(X_vals)},
                "Y": {"name": Y_name, "min": Y_vals[0], "max": Y_vals[-1], "n": len(Y_vals)},
            },
            "datafile": DEFAULT_DATAFILE,
            "base_pool": strify_pool(BASE_POOL),
        },
        "pools": pools,
    }

    out_path = Path(__file__).resolve().parent / "run_data" / "pool_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(pools)} pool configs to {out_path}")


if __name__ == "__main__":
    main()