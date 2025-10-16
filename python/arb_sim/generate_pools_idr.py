#!/usr/bin/env python3
import json
import math
import numpy as np
from pool_helpers import _first_candle_ts, _initial_price_from_file, strify_pool

N_GRID_X = 32
N_GRID_Y = 32

# Parameter ranges based on sweep type
if "mid_fee_log" == "mid_fee_log":
    X_name = "mid_fee"
    xmin = int(1 / 10_000 * 10**10)  # 1 bps
    xmax = int(100 / 10_000 * 10**10)  # 100 bps
    xlogspace = False  # Linear for fees
elif "mid_fee_log" == "donation_apy_log":
    X_name = "donation_apy"
    xmin = 0.01  # 1%
    xmax = 0.10  # 10%
    xlogspace = False  # Linear for percentages
elif "mid_fee_log" == "ma_time_log":
    X_name = "ma_time"
    xmin = int(600 / math.log(2))  # 10 minutes
    xmax = int(7*86400 / math.log(2))  # 7 days
    xlogspace = True  # Log for time
elif "mid_fee_log" == "fee_gamma_log":
    X_name = "fee_gamma"
    xmin = int(0.001 * 10**18)  # 0.001
    xmax = int(0.1 * 10**18)  # 0.1
    xlogspace = True  # Log for gamma

Y_name = "A"
ymin = 10*10_000  # 10k
ymax = 200*10_000  # 200k
ylogspace = True  # Log for A

# Generate parameter values
if xlogspace:
    X_vals = np.logspace(np.log10(xmin), np.log10(xmax), N_GRID_X).round().tolist()
else:
    X_vals = np.linspace(xmin, xmax, N_GRID_X).tolist()

Y_vals = np.logspace(np.log10(ymin), np.log10(ymax), N_GRID_Y).round().tolist()

init_liq = 1_000_000
DEFAULT_DATAFILE = "python/arb_sim/trade_data/idrusd/idr_minute_prices.json"
START_TS = _first_candle_ts(DEFAULT_DATAFILE)
init_price = _initial_price_from_file(DEFAULT_DATAFILE)

BASE_POOL = {
    "initial_liquidity": [int(init_liq * 10**18//2), int(init_liq * 10**18//2 / init_price)],
    "A": 32 * 10_000,  # CHF-proven value
    "gamma": 10**14,
    "mid_fee": int(10 / 10_000 * 10**10),  # 10 bps
    "out_fee": int(20 / 10_000 * 10**10),  # 20 bps
    "fee_gamma": int(0.001 * 10**18),  # CHF-proven value
    "allowed_extra_profit": int(1e-12 * 10**18),  # CHF-proven value
    "adjustment_step": int(1e-7 * 10**18),  # CHF-proven value
    "ma_time": int(3600 / math.log(2)),  # 1 hour
    "initial_price": int(init_price * 10**18),
    "start_timestamp": START_TS,
    "donation_apy": 0.05,  # 5% yearly
    "donation_frequency": int(7*86400),
    "donation_coins_ratio": 0.5,
}

BASE_COSTS = {"arb_fee_bps": 10.0, "gas_coin0": 0.0, "use_volume_cap": False, "volume_cap_mult": 1}

def build_grid():
    pools = []
    for xv in X_vals:
        for yv in Y_vals:
            pool = dict(BASE_POOL)
            pool[X_name] = xv
            pool[Y_name] = yv
            costs = dict(BASE_COSTS)
            tag = f"{X_name}_{xv}__{Y_name}_{yv}"
            pools.append({"tag": tag, "pool": strify_pool(pool), "costs": costs})
    return pools

def main():
    pools = build_grid()
    out = {
        "meta": {
            "created_utc": "2025-01-16T13:00:00Z",
            "grid": {"X": {"name": X_name, "min": X_vals[0], "max": X_vals[-1], "n": len(X_vals)},
                     "Y": {"name": Y_name, "min": Y_vals[0], "max": Y_vals[-1], "n": len(Y_vals)}},
            "datafile": DEFAULT_DATAFILE,
            "base_pool": strify_pool(BASE_POOL),
        },
        "pools": pools,
    }
    out_path = "python/arb_sim/run_data/pool_config.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(pools)} pool configs to {out_path}")

if __name__ == "__main__":
    main()
