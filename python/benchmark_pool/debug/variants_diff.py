#!/usr/bin/env python3
"""
Compare C++ variants (_i vs _d) for the latest (or given) run_cpp_variants_* folder.

Usage:
  uv run python/benchmark_pool/debug/variants_diff.py                 # auto-detect latest run_cpp_variants_*
  uv run python/benchmark_pool/debug/variants_diff.py <run_dir>       # specific run folder

Reports per-pool:
- Earliest divergence step and per-metric diffs (abs + %).
- Threshold-crossing for virtual_price error (config via --vp-threshold).
- Max-error step for virtual_price and price_scale.
- Final-state relative errors for key metrics.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

KEYS = [
    "balances", "xp", "D", "virtual_price", "xcp_profit",
    "price_scale", "price_oracle", "last_prices", "totalSupply",
]


def _find_latest_variants_run(results_dir: Path) -> Path:
    runs = sorted([p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("run_cpp_variants_")])
    if not runs:
        raise SystemExit(f"No run_cpp_variants_* folders found under {results_dir}")
    return runs[-1]


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to load {path}: {e}")


def _extract_states(results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for test in results.get("results", []):
        key = test.get("pool_config") or test.get("pool_name")
        res = test.get("result", {})
        st = res.get("states")
        if not st:
            final = res.get("final_state")
            st = [final] if final is not None else []
        out[key] = st
    return out


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def _norm(v: Any) -> Any:
    if isinstance(v, list):
        return [_to_int(x) for x in v]
    return _to_int(v)


def _rel_err(a: int, b: int) -> float:
    if b == 0:
        return 0.0 if a == 0 else float('inf')
    return abs(a - b) * 100.0 / abs(b)


def _first_divergence(i_states: List[Dict[str, Any]], d_states: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    n = min(len(i_states), len(d_states))
    for idx in range(n):
        diffs: Dict[str, Dict[str, Any]] = {}
        for k in KEYS:
            if k in i_states[idx] and k in d_states[idx]:
                iv = i_states[idx][k]
                dv = d_states[idx][k]
                if isinstance(iv, list) and isinstance(dv, list):
                    ival = [_to_int(x) for x in iv]
                    dval = [_to_int(x) for x in dv]
                    if ival != dval:
                        diffs[k] = {"i": ival, "d": dval}
                else:
                    ival = _to_int(iv)
                    dval = _to_int(dv)
                    if ival != dval:
                        diffs[k] = {"i": ival, "d": dval}
        if diffs:
            return idx, diffs
    return -1, {}


def _final_rel_errors(i_final: Dict[str, Any], d_final: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in KEYS:
        if k in i_final and k in d_final:
            iv = i_final[k]
            dv = d_final[k]
            if isinstance(iv, list) and isinstance(dv, list):
                ivn = [_to_int(x) for x in iv]
                dvn = [_to_int(x) for x in dv]
                out[k] = [ _rel_err(dvn[i], ivn[i]) for i in range(min(len(ivn), len(dvn))) ]
            else:
                out[k] = _rel_err(_to_int(dv), _to_int(iv))
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Compare C++ _i vs _d variants (step-wise)")
    ap.add_argument("run_dir", nargs="?", help="Path to run_cpp_variants_* folder (default: latest)")
    ap.add_argument("--vp-threshold", type=float, default=0.5, help="Report first step where virtual_price rel error exceeds this percent")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    results_root = repo_root / "python" / "benchmark_pool" / "data" / "results"
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _find_latest_variants_run(results_root)

    # Support both naming schemes (combined vs short)
    candidates_i = [run_dir / "cpp_i_combined.json", run_dir / "cpp_i.json"]
    candidates_d = [run_dir / "cpp_d_combined.json", run_dir / "cpp_d.json"]
    i_path = next((p for p in candidates_i if p.exists()), None)
    d_path = next((p for p in candidates_d if p.exists()), None)
    if not i_path or not d_path:
        print(f"Missing variant outputs in {run_dir}. Tried i: {[p.name for p in candidates_i]} d: {[p.name for p in candidates_d]}")
        return 1

    I = _load(i_path)
    D = _load(d_path)
    i_states = _extract_states(I)
    d_states = _extract_states(D)

    pools = sorted(set(i_states.keys()) & set(d_states.keys()))
    if not pools:
        print("No overlapping pools found between _i and _d results")
        return 1

    print(f"Inspecting variants run: {run_dir}")
    # Load actions to annotate steps (optional)
    actions = None
    seq_path = results_root.parent / "sequences.json"
    try:
        if seq_path.exists():
            actions = json.loads(seq_path.read_text())["sequences"][0]["actions"]
    except Exception:
        actions = None

    for pool in pools:
        cs = i_states[pool]
        ds = d_states[pool]
        step, diffs = _first_divergence(cs, ds)
        print(f"\nPool: {pool}")
        if step < 0:
            print("  No divergence detected across tracked keys.")
            continue
        print(f"  First divergence at step {step} (action index {step-1})")
        # Show a few important metrics' diffs
        for k in ("virtual_price", "price_scale", "last_prices", "D", "balances"):
            if k in diffs:
                iv = diffs[k]["i"]; dv = diffs[k]["d"]
                def fmt(v):
                    return v if isinstance(v, list) else str(v)
                print(f"    {k}: i={fmt(iv)} d={fmt(dv)}")
        # Find first threshold crossing for virtual_price error
        vp_cross = None
        max_vp = (0.0, -1)
        max_ps = (0.0, -1)
        for idx in range(min(len(cs), len(ds))):
            ivp = _to_int(cs[idx].get("virtual_price", 0))
            dvp = _to_int(ds[idx].get("virtual_price", 0))
            vp_err = _rel_err(dvp, ivp)
            if vp_err > max_vp[0]:
                max_vp = (vp_err, idx)
            ips = _to_int(cs[idx].get("price_scale", 0))
            dps = _to_int(ds[idx].get("price_scale", 0))
            ps_err = _rel_err(dps, ips)
            if ps_err > max_ps[0]:
                max_ps = (ps_err, idx)
            if vp_cross is None and vp_err > args.vp_threshold:
                vp_cross = (idx, vp_err)
        if vp_cross:
            idx, err = vp_cross
            print(f"  virtual_price error crossed {args.vp_threshold:.3f}% at step {idx} (err={err:.6f}%)")
            # print context metrics at that step
            ctx = {
                "D": (_to_int(ds[idx].get("D", 0)), _to_int(cs[idx].get("D", 0))),
                "price_scale": (_to_int(ds[idx].get("price_scale", 0)), _to_int(cs[idx].get("price_scale", 0))),
                "last_prices": (_to_int(ds[idx].get("last_prices", 0)), _to_int(cs[idx].get("last_prices", 0))),
                "totalSupply": (_to_int(ds[idx].get("totalSupply", 0)), _to_int(cs[idx].get("totalSupply", 0))),
            }
            for k, (dv, iv) in ctx.items():
                print(f"    {k}: d={dv} i={iv} (rel_err={_rel_err(dv, iv):.6f}%)")
            # Action context (step index maps to action index step-1)
            if actions and 0 <= idx-1 < len(actions):
                print(f"    action[{idx-1}]: {actions[idx-1]}")
        if max_vp[1] >= 0:
            print(f"  max virtual_price error: {max_vp[0]:.6f}% at step {max_vp[1]}")
            if actions and 0 <= max_vp[1]-1 < len(actions):
                print(f"    action[{max_vp[1]-1}]: {actions[max_vp[1]-1]}")
        if max_ps[1] >= 0:
            print(f"  max price_scale error: {max_ps[0]:.6f}% at step {max_ps[1]}")
            if actions and 0 <= max_ps[1]-1 < len(actions):
                print(f"    action[{max_ps[1]-1}]: {actions[max_ps[1]-1]}")
        # Also show final-state relative errors
        i_final = cs[-1] if cs else {}
        d_final = ds[-1] if ds else {}
        rel = _final_rel_errors(i_final, d_final)
        print("  Final-state relative errors (%):")
        for mk in ("virtual_price", "D", "price_scale", "balances", "totalSupply"):
            if mk in rel:
                val = rel[mk]
                if isinstance(val, list):
                    print(f"    {mk}: {[f'{x:.6f}%' if x != float('inf') else 'inf' for x in val]}")
                else:
                    print(f"    {mk}: {val:.6f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
