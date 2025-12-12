#!/usr/bin/env python3
"""
Compare output from old arb_harness vs new modular arb_harness.
Reports field-by-field differences with tolerance.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Tolerance for floating-point comparison
REL_TOL = 1e-6
ABS_TOL = 1e-10


def compare_values(old_val: Any, new_val: Any, path: str) -> list[str]:
    """Compare two values, return list of differences."""
    diffs = []

    if type(old_val) != type(new_val):
        # Allow int/float comparison
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            pass
        else:
            diffs.append(
                f"{path}: type mismatch: {type(old_val).__name__} vs {type(new_val).__name__}"
            )
            return diffs

    if isinstance(old_val, dict):
        all_keys = set(old_val.keys()) | set(new_val.keys())
        for k in all_keys:
            if k not in old_val:
                diffs.append(f"{path}.{k}: missing in old (new={new_val[k]})")
            elif k not in new_val:
                diffs.append(f"{path}.{k}: missing in new (old={old_val[k]})")
            else:
                diffs.extend(compare_values(old_val[k], new_val[k], f"{path}.{k}"))

    elif isinstance(old_val, list):
        if len(old_val) != len(new_val):
            diffs.append(f"{path}: length mismatch: {len(old_val)} vs {len(new_val)}")
        else:
            for i, (ov, nv) in enumerate(zip(old_val, new_val)):
                diffs.extend(compare_values(ov, nv, f"{path}[{i}]"))

    elif isinstance(old_val, (int, float)):
        if old_val == 0 and new_val == 0:
            pass
        elif abs(old_val) < ABS_TOL and abs(new_val) < ABS_TOL:
            pass
        elif old_val != 0:
            rel_diff = abs(old_val - new_val) / abs(old_val)
            if rel_diff > REL_TOL:
                diffs.append(
                    f"{path}: {old_val} vs {new_val} (rel_diff={rel_diff:.6e})"
                )
        else:
            if abs(new_val) > ABS_TOL:
                diffs.append(f"{path}: {old_val} vs {new_val}")

    elif isinstance(old_val, str):
        if old_val != new_val:
            # For numeric strings, try parsing
            try:
                ov_num = float(old_val)
                nv_num = float(new_val)
                if ov_num != 0:
                    rel_diff = abs(ov_num - nv_num) / abs(ov_num)
                    if rel_diff > REL_TOL:
                        diffs.append(
                            f"{path}: {old_val} vs {new_val} (rel_diff={rel_diff:.6e})"
                        )
                elif abs(nv_num) > ABS_TOL:
                    diffs.append(f"{path}: {old_val} vs {new_val}")
            except ValueError:
                diffs.append(f"{path}: '{old_val}' vs '{new_val}'")

    elif old_val != new_val:
        diffs.append(f"{path}: {old_val} vs {new_val}")

    return diffs


def run_harness(
    harness_path: str,
    pools_json: str,
    candles_json: str,
    n_candles: int,
    output_json: str,
) -> dict:
    """Run harness and return JSON output."""
    cmd = [
        harness_path,
        pools_json,
        candles_json,
        output_json,
        "--n-candles",
        str(n_candles),
        "--threads",
        "1",  # Single thread for reproducibility
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    with open(output_json) as f:
        return json.load(f)


def compare_core_metrics(old_run: dict, new_run: dict) -> list[str]:
    """Compare core metrics that both harnesses should produce."""
    diffs = []

    # Key fields that must match
    core_fields = [
        "events",
        "trades",
        "total_notional_coin0",
        "lp_fee_coin0",
        "arb_pnl_coin0",
        "n_rebalances",
        "donations",
        "donation_coin0_total",
    ]

    old_result = old_run.get("result", {})
    new_result = new_run.get("result", {})

    for field in core_fields:
        if field in old_result and field in new_result:
            diffs.extend(
                compare_values(old_result[field], new_result[field], f"result.{field}")
            )
        elif field in old_result:
            diffs.append(f"result.{field}: missing in new (old={old_result[field]})")

    # Compare final_state balances and D
    old_state = old_run.get("final_state", {})
    new_state = new_run.get("final_state", {})

    state_fields = [
        "balances",
        "D",
        "totalSupply",
        "price_scale",
        "price_oracle",
        "virtual_price",
    ]
    for field in state_fields:
        if field in old_state and field in new_state:
            diffs.extend(
                compare_values(
                    old_state[field], new_state[field], f"final_state.{field}"
                )
            )

    return diffs


def main():
    parser = argparse.ArgumentParser(description="Parity test for arb_harness")
    parser.add_argument("pools_json", help="Pool configuration JSON file")
    parser.add_argument("candles_json", help="Candles/events JSON file")
    parser.add_argument(
        "--n-candles", type=int, default=1000, help="Number of candles to process"
    )
    parser.add_argument(
        "--old-harness",
        default="cpp/build/arb_harness",
        help="Path to old harness binary",
    )
    parser.add_argument(
        "--new-harness",
        default="cpp_modular/build/arb_harness",
        help="Path to new modular harness binary",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all field comparisons"
    )
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent

    old_harness = project_root / args.old_harness
    new_harness = project_root / args.new_harness

    if not old_harness.exists():
        print(f"ERROR: Old harness not found: {old_harness}")
        return 1
    if not new_harness.exists():
        print(f"ERROR: New harness not found: {new_harness}")
        return 1

    # Run both harnesses
    with tempfile.TemporaryDirectory() as tmpdir:
        old_out = Path(tmpdir) / "old_out.json"
        new_out = Path(tmpdir) / "new_out.json"

        print(f"Running old harness: {old_harness}")
        old_data = run_harness(
            str(old_harness),
            args.pools_json,
            args.candles_json,
            args.n_candles,
            str(old_out),
        )

        print(f"Running new harness: {new_harness}")
        new_data = run_harness(
            str(new_harness),
            args.pools_json,
            args.candles_json,
            args.n_candles,
            str(new_out),
        )

    # Compare runs
    old_runs = old_data.get("runs", [])
    new_runs = new_data.get("runs", [])

    if len(old_runs) != len(new_runs):
        print(f"FAIL: Different number of runs: {len(old_runs)} vs {len(new_runs)}")
        return 1

    all_diffs = []
    for i, (old_run, new_run) in enumerate(zip(old_runs, new_runs)):
        print(f"\n--- Run {i}: comparing pools ---")
        diffs = compare_core_metrics(old_run, new_run)
        all_diffs.extend(diffs)

        if diffs:
            print(f"  Differences found:")
            for d in diffs:
                print(f"    {d}")
        else:
            print(f"  All core metrics match!")

    print("\n" + "=" * 60)
    if all_diffs:
        print(f"FAIL: {len(all_diffs)} difference(s) found")
        return 1
    else:
        print("PASS: All core metrics match within tolerance")
        return 0


if __name__ == "__main__":
    exit(main())
