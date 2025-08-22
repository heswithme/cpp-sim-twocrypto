#!/usr/bin/env python3
"""
Unified arbitrage runner and grid search for the C++ arb_harness.

- Single run: provide candles and (optionally) overrides; generates a pool JSON
  and runs the C++ harness once.
- Grid run: pass comma-separated lists for A and/or fees; script detects if
  lengths > 1 and runs a grid across combinations using threads.

Outputs are written under python/arb/run_data/ by default.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def to_wad(x: float) -> str:
    return str(int(x * 1e18))


def to_fee_bps_scaled(x_bps: float) -> str:
    # Pool expects 1e10 scale for fees; convert bps to fraction first
    # bps -> fraction: x_bps / 1e4; then fraction * 1e10
    scaled = (x_bps / 1e4) * 1e10
    return str(int(scaled))


def default_pool_config(initial_liquidity0: float, initial_liquidity1: float, 
                        A: float = 100_000.0, gamma: float = 0.0,
                        mid_fee_bps: float = 3.0, out_fee_bps: float = 5.0,
                        fee_gamma: float = 0.23, allowed_extra_profit: float = 1e-3,
                        adjustment_step: float = 1e-3, ma_time: int = 600,
                        initial_price: float = 1.0) -> dict:
    return {
        "pool": {
            "initial_liquidity": [to_wad(initial_liquidity0), to_wad(initial_liquidity1)],
            "A": str(A),
            "gamma": str(gamma),
            "mid_fee": to_fee_bps_scaled(mid_fee_bps),
            "out_fee": to_fee_bps_scaled(out_fee_bps),
            "fee_gamma": to_wad(fee_gamma),
            "allowed_extra_profit": to_wad(allowed_extra_profit),
            "adjustment_step": to_wad(adjustment_step),
            "ma_time": str(ma_time),
            "initial_price": to_wad(initial_price)
        }
    }


def default_costs(arb_fee_bps: float = 10.0, gas_coin0: float = 0.0,
                  max_trade_frac: float = 0.25, use_volume_cap: bool = False,
                  volume_cap_mult: float = 1.0) -> dict:
    return {
        "costs": {
            "arb_fee_bps": arb_fee_bps,
            "gas_coin0": to_wad(gas_coin0),
            "max_trade_frac": max_trade_frac,
            "use_volume_cap": use_volume_cap,
            "volume_cap_mult": volume_cap_mult,
        }
    }


class ArbHarnessRunner:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.cpp_dir = self.repo_root / "cpp"
        self.build_dir = self.cpp_dir / "build"
        self.exe_path = self.build_dir / "arb_harness"

    def configure_build(self):
        self.build_dir.mkdir(parents=True, exist_ok=True)
        print("Configuring C++ build (Release)...")
        r = subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=self.build_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError("CMake configure failed")

    def build(self):
        self.configure_build()
        print("Building arb_harness...")
        r = subprocess.run(["cmake", "--build", ".", "--target", "arb_harness"], cwd=self.build_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError("Build failed")
        if not self.exe_path.exists():
            raise FileNotFoundError(f"Missing executable: {self.exe_path}")
        print(f"✓ Built: {self.exe_path}")

    def run(self, pool_json_path: Path, candles_path: Path, out_json_path: Path, n_candles: int = 0, save_actions: bool = False):
        print("Running arb_harness...")
        cmd = [str(self.exe_path), str(pool_json_path), str(candles_path), str(out_json_path)]
        if n_candles and n_candles > 0:
            cmd += ["--n-candles", str(n_candles)]
        if save_actions:
            cmd += ["--save-actions"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError("arb_harness failed")
        if r.stdout.strip():
            print(r.stdout)
        print(f"✓ Results: {out_json_path}")
        with open(out_json_path, 'r') as f:
            return json.load(f)


def pretty_format_json(path: Path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run C++ arbitrage harness over candle data (single or grid)")
    parser.add_argument("candles", type=str, help="Path to candles JSON (array of [ts,o,h,l,c,vol])")
    parser.add_argument("--out", type=str, default=None, help="Output result JSON path (single) or directory (grid)")
    parser.add_argument("--n-candles", type=int, default=0, help="Limit to first N candles (default: all)")
    parser.add_argument("--pool-json", type=str, default=None, help="Existing pool+cost JSON to use (single run only)")
    parser.add_argument("--initial-liq", type=float, nargs=2, metavar=("X0","X1"), default=(1e6, 1e6), help="Initial liquidity for coin0, coin1 (token units)")
    parser.add_argument("--arb-fee-bps", type=float, default=10.0)
    parser.add_argument("--gas-coin0", type=float, default=0.0)
    parser.add_argument("--max-trade-frac", type=float, default=0.25)
    parser.add_argument("--use-volume-cap", action="store_true")
    parser.add_argument("--volume-cap-mult", type=float, default=1.0)
    parser.add_argument("--save-actions", action="store_true", help="Ask C++ harness to save executed trades/actions")
    parser.add_argument("--arb-step", type=float, default=None, help="Multiplicative arb step (>1.0). Example: 1.1 scales 10% per step")
    # Grid controls
    parser.add_argument("--A", type=str, default=None, help="Comma-separated A values for grid (e.g., '50000,100000')")
    parser.add_argument("--mid-fee-bps", type=str, default=None, help="Comma-separated mid fee bps values (e.g., '2.5,3.0')")
    parser.add_argument("--out-fee-bps", type=str, default=None, help="Comma-separated out fee bps values (e.g., '5,6')")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers for grid mode")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runner = ArbHarnessRunner(repo_root)
    runner.build()

    def parse_csv(s: str) -> List[float]:
        return [float(x.strip()) for x in s.split(',') if x.strip()]

    As = parse_csv(args.A) if args.A else []
    mids = parse_csv(args.mid_fee_bps) if args.mid_fee_bps else []
    outs = parse_csv(args.out_fee_bps) if args.out_fee_bps else []
    grid_mode = (len(As) > 1) or (len(mids) > 1) or (len(outs) > 1)

    if not As: As = [100_000.0]
    if not mids: mids = [3.0]
    if not outs: outs = [5.0]

    if not grid_mode:
        # Single run
        pool_json_path = Path(args.pool_json) if args.pool_json else (repo_root / "python" / "arb_sim" / "run_data" / "pool_config.json")
        pool_json_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.pool_json:
            pool_cfg = default_pool_config(args.initial_liq[0], args.initial_liq[1], A=As[0], mid_fee_bps=mids[0], out_fee_bps=outs[0])
            costs_cfg = default_costs(
                arb_fee_bps=args.arb_fee_bps,
                gas_coin0=args.gas_coin0,
                max_trade_frac=args.max_trade_frac,
                use_volume_cap=args.use_volume_cap,
                volume_cap_mult=args.volume_cap_mult,
            )
            cfg = {**pool_cfg, **costs_cfg}
            with open(pool_json_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"✓ Wrote pool config: {pool_json_path}")

        out_json_path = Path(args.out) if args.out else (repo_root / "python" / "arb_sim" / "run_data" / f"arb_run_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        # Prepare command extras
        result = runner.run(pool_json_path, Path(args.candles), out_json_path, n_candles=args.n_candles, save_actions=args.save_actions)
        pretty_format_json(out_json_path)
        res = result.get("result", {})
        print("Summary:")
        for k in ("events", "trades", "total_notional_coin0", "lp_fee_coin0", "arb_pnl_coin0"):
            if k in res:
                print(f"  {k}: {res[k]}")
        # Print timing if present
        for k in ("candles_read_ms", "exec_ms"):
            if k in res:
                print(f"  {k}: {res[k]}")
        return 0

    # Grid mode
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out_dir = Path(args.out) if args.out else (repo_root / "python" / "arb" / "run_data" / f"arb_grid_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_pool_config(initial_liq0: float, initial_liq1: float, A: float, mid_fee_bps: float, out_fee_bps: float) -> Dict:
        return {
            "pool": {
                "initial_liquidity": [to_wad(initial_liq0), to_wad(initial_liq1)],
                "A": str(A),
                "gamma": str(0.0),
                "mid_fee": to_fee_bps_scaled(mid_fee_bps),
                "out_fee": to_fee_bps_scaled(out_fee_bps),
                "fee_gamma": to_wad(0.23),
                "allowed_extra_profit": to_wad(1e-3),
                "adjustment_step": to_wad(1e-3),
                "ma_time": str(600),
                "initial_price": to_wad(1.0),
            },
            "costs": {
                "arb_fee_bps": args.arb_fee_bps,
                "gas_coin0": to_wad(args.gas_coin0),
                "max_trade_frac": args.max_trade_frac,
                "use_volume_cap": args.use_volume_cap,
                "volume_cap_mult": args.volume_cap_mult,
            },
        }

    def run_one(tag: str, cfg: Dict) -> Tuple[str, Dict]:
        cfg_path = out_dir / f"pool_{tag}.json"
        out_path = out_dir / f"result_{tag}.json"
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        cmd = [str(runner.exe_path), str(cfg_path), args.candles, str(out_path)]
        if args.n_candles and args.n_candles > 0:
            cmd += ["--n-candles", str(args.n_candles)]
        if args.save_actions:
            cmd += ["--save-actions"]
        if args.arb_step and args.arb_step > 1.0:
            cmd += ["--arb-step", str(args.arb_step)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return tag, {"error": r.stderr.strip() or "arb_harness failed"}
        try:
            with open(out_path, 'r') as f:
                data = json.load(f)
            # Pretty format result JSON
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
            return tag, data.get("result", {})
        except Exception as e:
            return tag, {"error": f"Failed to read result: {e}"}

    jobs: List[Tuple[str, Dict]] = []
    for A in As:
        for mid in mids:
            for out in outs:
                tag = f"A{A}_mid{mid}_out{out}".replace('.', 'p')
                cfg = make_pool_config(args.initial_liq[0], args.initial_liq[1], A, mid, out)
                jobs.append((tag, cfg))

    print(f"Planned {len(jobs)} simulations...")
    results: List[Tuple[str, Dict]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(run_one, tag, cfg): tag for tag, cfg in jobs}
        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                t, res = fut.result()
            except Exception as e:
                t, res = tag, {"error": str(e)}
            results.append((t, res))
            if "error" in res:
                print(f"[{t}] ERROR: {res['error']}")
            else:
                print(f"[{t}] trades={res.get('trades')} arb_pnl={res.get('arb_pnl_coin0')}")

    rows = [{"tag": tag, **res} for tag, res in results]
    rows.sort(key=lambda r: float(r.get("arb_pnl_coin0", 0) or 0), reverse=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({"results": rows, "sort_by": "arb_pnl_coin0"}, f, indent=2)
    print(f"\n✓ Wrote summary: {summary_path}")
    print("Top 5:")
    for r in rows[:5]:
        print(f"  {r['tag']}: arb_pnl={r.get('arb_pnl_coin0')} trades={r.get('trades')} lp_fee={r.get('lp_fee_coin0')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
