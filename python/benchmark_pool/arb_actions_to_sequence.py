#!/usr/bin/env python3
"""
Convert arb_sim extended actions into a benchmark-compatible sequence and pool.

Input: an aggregated arb_sim run JSON (written by python/arb_sim/arb_sim.py
with --save-actions), which contains per-pool entries under runs[] and, for
each run, an optional actions[] list with fields:
  { ts, i, j, dx, dy_after_fee, fee_tokens, profit_coin0, p_cex, p_pool_before }

Outputs:
  - sequences.json in the benchmark format, containing a single sequence:
    - name: string
    - start_timestamp: first action timestamp
    - actions: list of {time_travel, exchange} entries
  - pools.json containing a single pool matching the selected run, flattened
    to the benchmark schema with a "name" field.

Transformations:
  - Inserts absolute time_travel actions using each action's timestamp (seconds).
  - Exchanges: converts `dx` from float tokens to 1e18-scaled integer string.
  - Donations: converts to `add_liquidity` with `donation=True` and `amounts`
    scaled to 1e18 integer strings based on the recorded donation amounts.

Usage:
  uv run python/benchmark_pool/arb_actions_to_sequence.py \
    python/arb_sim/run_data/arb_run_20240101T000000Z.json \
    --run-index 0 \
    --output-seq python/benchmark_pool/data/sequences.json \
    --output-pools python/benchmark_pool/data/pools.json

Optional selection:
  - --x-val / --y-val: pick a run matching x_val/y_val from arb_sim metadata.
  - --pool-config: optional path to python/arb_sim/run_data/pool_config.json
    (default used if omitted). Required to generate pools.json.
  - --name: set sequence name (default derives from x/y if present).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import glob
from decimal import Decimal, ROUND_HALF_UP


def _load_json(path: Path, *, as_decimal: bool = False) -> Any:
    with path.open("r") as f:
        if as_decimal:
            return json.load(f, parse_float=Decimal)
        return json.load(f)


def _pick_run(runs: List[Dict[str, Any]], run_index: Optional[int], x_val: Optional[str], y_val: Optional[str]) -> Dict[str, Any]:
    # If user specified x/y, honor that first
    if x_val is not None or y_val is not None:
        for rr in runs:
            ok_x = (x_val is None) or (str(rr.get("x_val")) == str(x_val))
            ok_y = (y_val is None) or (str(rr.get("y_val")) == str(y_val))
            if ok_x and ok_y:
                return rr
        raise ValueError(f"No run matched x_val={x_val}, y_val={y_val}")

    # If a specific index is given, use it
    if run_index is not None:
        idx = int(run_index)
        if idx < 0 or idx >= len(runs):
            raise IndexError(f"run-index {idx} out of range (0..{len(runs)-1})")
        return runs[idx]

    # Otherwise expect a single actions-carrying run; if multiple, pick the last one
    with_actions = [rr for rr in runs if rr.get("actions")]
    if not with_actions:
        raise ValueError("No runs contain actions[]; make sure arb_sim was run with --save-actions")
    if len(with_actions) > 1:
        # Prefer the last one for convenience
        return with_actions[-1]
    return with_actions[0]


def _find_latest_arb_run(repo_root: Path) -> Path:
    rd = repo_root / "python" / "arb_sim" / "run_data"
    files = sorted(glob.glob(str(rd / "arb_run_*.json")))
    if not files:
        raise FileNotFoundError(f"No arb_run_*.json files found under {rd}")
    # Pick the newest by mtime
    latest = max(files, key=lambda p: os.path.getmtime(p))
    return Path(latest)


def _scale_dx_to_str(dx_tokens: Decimal | float) -> str:
    # Use Decimal for stable scaling + rounding; accept float fallback
    if not isinstance(dx_tokens, Decimal):
        dx_tokens = Decimal(str(dx_tokens))
    scaled = (dx_tokens * Decimal(10) ** 18).to_integral_value(rounding=ROUND_HALF_UP)
    if scaled < 0:
        scaled = Decimal(0)
    return str(int(scaled))


def _scale_amounts_to_strs(amts: List[Decimal | float]) -> List[str]:
    return [_scale_dx_to_str(x) for x in amts]


def convert_actions(arb_run_path: Path, *, run_index: Optional[int] = None,
                    x_val: Optional[str] = None, y_val: Optional[str] = None,
                    sequence_name: Optional[str] = None,
                    start_ts_hint: Optional[int] = None) -> Dict[str, Any]:
    data = _load_json(arb_run_path, as_decimal=True)
    runs: List[Dict[str, Any]] = data.get("runs", [])
    if not runs:
        raise ValueError("No runs[] found in arb_run JSON (did you pass the correct file?)")
    rr = _pick_run(runs, run_index, x_val, y_val)
    actions = rr.get("actions")
    if not actions:
        raise ValueError("Selected run has no actions[] (did you run arb_sim with --save-actions?)")

    # Derive name
    if not sequence_name:
        xk = rr.get("x_key")
        yk = rr.get("y_key")
        xv = rr.get("x_val")
        yv = rr.get("y_val")
        if xk and yk:
            sequence_name = f"arb_sim__{xk}_{xv}__{yk}_{yv}"
        elif xk:
            sequence_name = f"arb_sim__{xk}_{xv}"
        else:
            sequence_name = "arb_sim_default"

    # Determine start timestamp to match arb_harness dynamics:
    # - If pool.start_timestamp is present in run params, use it (explicit alignment)
    # - Otherwise DO NOT set start_timestamp in the sequence to mirror arb_harness behavior
    #   (which initializes last_timestamp to "now" in the constructor, and never rewinds it).
    pool_params = rr.get("params", {}).get("pool") if isinstance(rr.get("params"), dict) else None
    has_start_ts = bool(pool_params and pool_params.get("start_timestamp") is not None)
    start_ts = int(pool_params.get("start_timestamp")) if has_start_ts else 0
    # If caller provided a start_ts hint (e.g., from pool_config.json), prefer it
    if not has_start_ts and start_ts_hint is not None:
        start_ts = int(start_ts_hint)
        has_start_ts = True

    # Ensure actions sorted by timestamp, though input should already be ordered
    actions_sorted = sorted(actions, key=lambda a: int(a.get("ts", 0)))
    if not actions_sorted:
        raise ValueError("No actions to convert")
    # If start_timestamp wasn't provided by pool params, seed it from first action
    if not has_start_ts:
        try:
            start_ts = int(actions_sorted[0].get("ts", 0))
            has_start_ts = True
        except Exception:
            pass

    out_actions: List[Dict[str, Any]] = []
    for a in actions_sorted:
        atype = str(a.get("type") or "exchange")
        # Donation actions saved by arb_harness record ts (pool time) and ts_due (schedule).
        # Prefer pool-execution time to match actual state transitions.
        ts_key = "ts"
        if atype == "donation" and "ts_pool" in a:
            ts_key = "ts_pool"
        ts = int(a.get(ts_key, start_ts))
        # Absolute timestamp to avoid drift
        out_actions.append({"type": "time_travel", "timestamp": ts})
        if atype == "exchange":
            # Transform exchange
            i = int(a.get("i", 0))
            j = int(a.get("j", 1))
            # Prefer pre-scaled wei field if provided by harness; else scale tokens
            if "dx_wei" in a:
                dx_out = str(a.get("dx_wei"))
            else:
                dx_raw = a.get("dx", Decimal(0))
                dx_out = _scale_dx_to_str(dx_raw)
            out_actions.append({
                "type": "exchange",
                "i": i,
                "j": j,
                "dx": dx_out,
            })
        elif atype == "donation":
            # Balanced donation amounts come from arb_harness action
            amts = a.get("amounts") or [0, 0]
            # amounts are in tokens (float); scale to 1e18 integer strings
            try:
                if "amounts_wei" in a:
                    amt0s = str(a.get("amounts_wei")[0])
                    amt1s = str(a.get("amounts_wei")[1])
                    out_actions.append({
                        "type": "add_liquidity",
                        "amounts": [amt0s, amt1s],
                        "donation": True,
                    })
                    continue
                amt0 = amts[0]
                amt1 = amts[1]
            except Exception:
                amt0 = 0
                amt1 = 0
            out_actions.append({
                "type": "add_liquidity",
                "amounts": _scale_amounts_to_strs([amt0, amt1]),
                "donation": True,
            })
        else:
            # Unknown action type; skip
            continue
    # If available, align final timestamp to arb_sim's last processed event
    # (does not affect EMA since no exchange follows).
    try:
        final_ts = int(rr.get("final_state", {}).get("timestamp"))
        if out_actions and final_ts and final_ts > int(actions_sorted[-1].get("ts", 0)):
            out_actions.append({"type": "time_travel", "timestamp": final_ts})
    except Exception:
        pass

    seq_obj: Dict[str, Any] = {
        "name": sequence_name,
        "actions": out_actions,
    }
    if has_start_ts:
        seq_obj["start_timestamp"] = start_ts

    return {
        "sequences": [seq_obj]
    }


def build_pool_for_benchmark(pool_config_path: Path, *, x_key: Optional[str], y_key: Optional[str],
                             x_val: Optional[str], y_val: Optional[str],
                             fallback_name: str) -> Dict[str, Any]:
    """Pick the matching pool entry from arb_sim pool_config.json and flatten to benchmark format.

    Returns {"pools": [pool_dict]} ready to write as pools.json.
    """
    cfg = _load_json(pool_config_path)
    pools = cfg.get("pools", [])
    if not pools:
        raise ValueError(f"No pools[] in {pool_config_path}")

    def matches(p: Dict[str, Any]) -> bool:
        pool = p.get("pool", {})
        okx = True if not x_key or x_val is None else str(pool.get(x_key)) == str(x_val)
        oky = True if not y_key or y_val is None else str(pool.get(y_key)) == str(y_val)
        return okx and oky

    picked = None
    for p in pools:
        if matches(p):
            picked = p
            break
    if picked is None:
        raise ValueError(f"No pool in {pool_config_path} matched x_key={x_key}, x_val={x_val}, y_key={y_key}, y_val={y_val}")

    pool_obj = picked.get("pool", {})
    name = picked.get("tag") or fallback_name or "arb_sim_pool"
    # Flatten to benchmark schema
    flat = {
        "name": name,
        "A": pool_obj.get("A"),
        "gamma": pool_obj.get("gamma"),
        "mid_fee": pool_obj.get("mid_fee"),
        "out_fee": pool_obj.get("out_fee"),
        "fee_gamma": pool_obj.get("fee_gamma"),
        "allowed_extra_profit": pool_obj.get("allowed_extra_profit"),
        "adjustment_step": pool_obj.get("adjustment_step"),
        "ma_time": pool_obj.get("ma_time"),
        "initial_price": pool_obj.get("initial_price"),
        "initial_liquidity": pool_obj.get("initial_liquidity"),
    }
    return {"pools": [flat]}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Convert arb_sim extended actions to benchmark sequences.json")
    ap.add_argument("arb_run", nargs="?", default=None, help="Path to arb_run_*.json; if omitted, uses latest in python/arb_sim/run_data/")
    ap.add_argument("--output-seq", type=str, default=None, help="Output sequences.json path (default: python/benchmark_pool/data/sequences.json)")
    ap.add_argument("--output-pools", type=str, default=None, help="Output pools.json path (default: python/benchmark_pool/data/pools.json)")
    ap.add_argument("--run-index", type=int, default=None, help="Index into runs[] if multiple pools present")
    ap.add_argument("--x-val", type=str, default=None, help="Select run by matching x_val")
    ap.add_argument("--y-val", type=str, default=None, help="Select run by matching y_val")
    ap.add_argument("--name", type=str, default=None, help="Sequence name in output JSON")
    ap.add_argument("--pool-config", type=str, default=None, help="Path to arb_sim pool_config.json (default: python/arb_sim/run_data/pool_config.json)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    default_seq = repo_root / "python" / "benchmark_pool" / "data" / "sequences.json"
    default_pools = repo_root / "python" / "benchmark_pool" / "data" / "pools.json"
    seq_out = Path(args.output_seq) if args.output_seq else default_seq
    pools_out = Path(args.output_pools) if args.output_pools else default_pools
    seq_out.parent.mkdir(parents=True, exist_ok=True)
    pools_out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve arb_run path: use latest if not provided
    arb_run_path = Path(args.arb_run) if args.arb_run else _find_latest_arb_run(repo_root)
    print(f"Using arb_run file: {arb_run_path}")

    # Try to load pool-config start_timestamp as a hint if available
    run_sel = _pick_run(_load_json(arb_run_path).get("runs", []), args.run_index, args.x_val, args.y_val)
    x_key = run_sel.get("x_key"); y_key = run_sel.get("y_key")
    x_val = run_sel.get("x_val"); y_val = run_sel.get("y_val")
    pc_path = Path(args.pool_config) if args.pool_config else (repo_root / "python" / "arb_sim" / "run_data" / "pool_config.json")
    start_ts_hint = None
    try:
        pc = _load_json(pc_path)
        for p in pc.get("pools", []):
            pool = p.get("pool", {})
            okx = True if not x_key or x_val is None else str(pool.get(x_key)) == str(x_val)
            oky = True if not y_key or y_val is None else str(pool.get(y_key)) == str(y_val)
            if okx and oky:
                if pool.get("start_timestamp") is not None:
                    start_ts_hint = int(pool.get("start_timestamp"))
                break
    except Exception:
        start_ts_hint = None

    obj = convert_actions(arb_run_path, run_index=args.run_index,
                          x_val=args.x_val, y_val=args.y_val,
                          sequence_name=args.name,
                          start_ts_hint=start_ts_hint)

    with seq_out.open("w") as f:
        json.dump(obj, f, indent=2)
    print(f"✓ Wrote sequence with {len(obj['sequences'][0]['actions'])} actions to {seq_out}")

    # Also write a single-pool pools.json matching the selected run
    run_sel = _pick_run(_load_json(arb_run_path).get("runs", []), args.run_index, args.x_val, args.y_val)
    x_key = run_sel.get("x_key"); y_key = run_sel.get("y_key")
    x_val = run_sel.get("x_val"); y_val = run_sel.get("y_val")
    pc_path = Path(args.pool_config) if args.pool_config else (repo_root / "python" / "arb_sim" / "run_data" / "pool_config.json")
    # If the run includes the exact pool params, use them directly for the benchmark pool
    pool_from_run = run_sel.get("params", {}).get("pool") if isinstance(run_sel.get("params"), dict) else None
    if pool_from_run:
        name = run_sel.get("params", {}).get("tag") or run_sel.get("result", {}).get("tag") or obj["sequences"][0]["name"]
        pools_obj = {"pools": [{"name": name, **{k: pool_from_run[k] for k in [
            "A","gamma","mid_fee","out_fee","fee_gamma","allowed_extra_profit","adjustment_step","ma_time","initial_price","initial_liquidity"
        ] if k in pool_from_run}}]}
    else:
        pools_obj = build_pool_for_benchmark(pc_path, x_key=x_key, y_key=y_key, x_val=x_val, y_val=y_val, fallback_name=obj["sequences"][0]["name"])
    with pools_out.open("w") as f:
        json.dump(pools_obj, f, indent=2)
    print(f"✓ Wrote matching pool to {pools_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
