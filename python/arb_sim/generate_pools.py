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


def _first_candle_ts(path: str) -> int:
    """Return the first candle timestamp (seconds). If millis, convert to seconds.

    On failure, return current UTC seconds.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], list) and data[0]:
            ts = int(data[0][0])
            if ts > 10_000_000_000:  # ms
                ts //= 1000
            return int(ts)
    except Exception:
        pass
    return int(time())


# -------------------- Helpers --------------------
def strify_pool(pool: dict) -> dict:
    """Convert pool values to strings for JSON.

    - Lists are treated as integer arrays and stringified element-wise.
    - Integers are stringified as ints.
    - Floats are preserved as decimal strings (no int cast), e.g., donation_apy=0.05 -> "0.05".
    """
    out = {}
    for k, v in pool.items():
        if isinstance(v, list):
            out[k] = [str(int(x)) for x in v]
        elif isinstance(v, float):
            out[k] = str(v)
        else:
            out[k] = str(int(v))
    return out


# -------------------- Grid Definition --------------------
N_GRID = 32

X_name = "A"  # can be changed to any pool key
X_vals = np.logspace(np.log10(5 * 10_000), np.log10(500 * 10_000), N_GRID).round().astype(int).tolist()

Y_name = "mid_fee"  # default second param; also applied to out_fee
Y_vals = np.logspace(np.log10(1e-4 * 10**10), np.log10(.05 * 10**10), N_GRID).round().astype(int).tolist()

init_liq = 1_000_000 # in coin0
init_price = 0.190865 #brlusd
DEFAULT_DATAFILE = "python/arb_sim/trade_data/brlusd/brlusd-1m.json"
START_TS = _first_candle_ts(DEFAULT_DATAFILE)
# -------------------- Base Templates --------------------
BASE_POOL = {
    # All values are integers in their native units
    "initial_liquidity": [int(init_liq//2) * 10**18, int(init_liq//2 / init_price) * 10**18],
    "A": 100 * 10_000,
    "gamma": 10**14, #unused in twocrypto
    "mid_fee": int(0.001 * 10**10),
    "out_fee": int(0.002 * 10**10),
    "fee_gamma": int(0.003 * 10**18),
    "allowed_extra_profit": int(1e-12 * 10**18),
    "adjustment_step": int(5.5e-3 * 10**18),
    "ma_time": 866,
    "initial_price": int(init_price * 10**18),
    "start_timestamp": START_TS,
    # Donations (harness-only):
    # - donation_apy: plain fraction per year (0.05 => 5%).
    # - donation_frequency: seconds between donations.
    "donation_apy": 0.00,
    "donation_frequency": int(3600),
}

BASE_COSTS = {
    "arb_fee_bps": 50.0,
    "gas_coin0": 0.0,
    "use_volume_cap": True,
    "volume_cap_mult": 1,
}


def build_grid():
    pools = []
    for xv in X_vals:
        for yv in Y_vals:
            pool = dict(BASE_POOL)  # start from base with all params (ints)
            # Apply X then Y onto pool
            pool[X_name] = int(xv)
            pool[Y_name] = int(yv)
            # Enforce out_fee >= mid_fee
            mid_fee_val = int(pool.get("mid_fee", 0))
            cur_out_val = int(pool.get("out_fee", 0))
            pool["out_fee"] = max(mid_fee_val, cur_out_val)

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
