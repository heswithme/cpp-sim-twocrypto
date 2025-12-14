#!/usr/bin/env python3
"""
Arbitrage runner for the C++ arb_harness (multi-pool, threaded in C++).

- Always reads pools from python/arb_sim/run_data/pool_config.json
- Calls the C++ harness once with all pools; C++ handles internal threading.
- Emits an aggregated arb_run JSON with x/y keys, per-pool final_state, and result.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone


class ArbHarnessRunner:
    def __init__(self, repo_root: Path, real: str = "double"):
        self.repo_root = Path(repo_root)
        self.cpp_dir = self.repo_root / "cpp_old"
        self.cpp_dir = self.repo_root / "cpp_modular"  # modular harness
        self.build_dir = self.cpp_dir / "build"
        # Resolve binary name based on real type
        real = (real or "double").lower()
        if real in ("float", "f"):
            self.target = "arb_harness_f"
        elif real in ("longdouble", "long_double", "ld", "long"):
            self.target = "arb_harness_ld"
        else:
            self.target = "arb_harness"
        self.exe_path = self.build_dir / self.target

    def configure_build(self):
        self.build_dir.mkdir(parents=True, exist_ok=True)
        print("Configuring C++ build (Release)...")
        r = subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=self.build_dir,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError("CMake configure failed")

    def build(self):
        self.configure_build()
        print(f"Building {self.target}...")
        r = subprocess.run(
            ["cmake", "--build", ".", "--target", self.target],
            cwd=self.build_dir,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError("Build failed")
        if not self.exe_path.exists():
            raise FileNotFoundError(f"Missing executable: {self.exe_path}")
        print(f"✓ Built: {self.exe_path}")

    def run(
        self,
        pools_json: Path,
        candles_path: Path,
        out_json_path: Path,
        n_candles: int = 0,
        save_actions: bool = False,
        events: bool = False,
        min_swap: float = 1e-10,
        max_swap: float = 1.0,
        threads: int = 1,
        dustswapfreq: int | None = None,
        userswapfreq: int | None = None,
        userswapsize: float | None = None,
        userswapthresh: float | None = None,
        apy_period_days: float | None = None,
        apy_period_cap: int | None = None,
        detailed_log: bool = False,
    ) -> Dict[str, Any]:
        print("Running arb_harness...")
        cmd = [
            str(self.exe_path),
            str(pools_json),
            str(candles_path),
            str(out_json_path),
        ]
        if n_candles and n_candles > 0:
            cmd += ["--n-candles", str(n_candles)]
        if save_actions:
            cmd += ["--save-actions"]
        if events:
            cmd += ["--events"]
        if min_swap is not None:
            cmd += ["--min-swap", str(min_swap)]
        if max_swap is not None:
            cmd += ["--max-swap", str(max_swap)]
        # Always pass threads to ensure explicit control (even when 1)
        cmd += ["--threads", str(max(1, threads))]
        if dustswapfreq is not None:
            cmd += ["--dustswapfreq", str(int(dustswapfreq))]
        if userswapfreq is not None:
            cmd += ["--userswapfreq", str(int(userswapfreq))]
        if userswapsize is not None:
            cmd += ["--userswapsize", str(userswapsize)]
        if userswapthresh is not None:
            cmd += ["--userswapthresh", str(userswapthresh)]
        if apy_period_days is not None:
            cmd += ["--apy-period-days", str(apy_period_days)]
        if apy_period_cap is not None:
            cmd += ["--apy-period-cap", str(int(apy_period_cap))]
        if detailed_log:
            cmd += ["--detailed-log"]
        # Stream harness stdout/stderr directly to the console for live progress
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise RuntimeError("arb_harness failed")
        print(f"✓ Results: {out_json_path}")
        with open(out_json_path, "r") as f:
            return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run C++ multi-pool arbitrage harness over candle data or events"
    )
    parser.add_argument(
        "candles",
        nargs="?",
        default=None,
        type=str,
        help="Path to candles JSON; if omitted, use pool_config meta.datafile",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Aggregated output JSON path"
    )
    parser.add_argument(
        "--n-candles",
        type=int,
        default=0,
        help="Limit to first N candles (default: all)",
    )
    parser.add_argument(
        "--save-actions",
        action="store_true",
        help="Ask C++ harness to save executed trades/actions",
    )
    parser.add_argument(
        "--min-swap",
        type=float,
        default=1e-6,
        help="Minimum swap fraction of from-side balance (default: 1e-6)",
    )
    parser.add_argument(
        "--max-swap",
        type=float,
        default=1.0,
        help="Maximum swap fraction of from-side balance (default: 1.0)",
    )
    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=1,
        help="Threads in C++ harness (default: 1)",
    )
    parser.add_argument(
        "--real",
        type=str,
        default="double",
        choices=["float", "double", "longdouble"],
        help="Numeric precision for C++ harness",
    )
    parser.add_argument(
        "--dustswapfreq",
        type=int,
        default=None,
        help="Seconds between dust swaps when no arb trade (cooldown)",
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Treat input file as events [[ts,price,volume]] and skip candle conversion",
    )
    parser.add_argument(
        "--candle-filter",
        type=float,
        default=10.0,
        help="Filter candles +/- PCT (default: 10.0)",
    )
    parser.add_argument(
        "--userswapfreq",
        type=int,
        default=0,
        help="Seconds between synthetic user swaps (default: 0)",
    )
    parser.add_argument(
        "--userswapsize",
        type=float,
        default=0,
        help="Fraction of from-side balance per user swap (default: 0)",
    )
    parser.add_argument(
        "--userswapthresh",
        type=float,
        default=0,
        help="Max relative deviation vs CEX to allow user swap",
    )
    parser.add_argument(
        "--apy-period-days",
        type=float,
        default=None,
        help="Rolling APY window length in days (forwarded to arb_harness)",
    )
    parser.add_argument(
        "--apy-period-cap",
        type=int,
        default=None,
        help="Cap per-window annualized APY percent (forwarded to arb_harness)",
    )
    parser.add_argument(
        "--detailed-log",
        action="store_true",
        help="Write per-candle detailed_log.json next to output file",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runner = ArbHarnessRunner(repo_root, real=args.real)
    runner.build()

    # Load pool_config.json
    pool_config_path = (
        repo_root / "python" / "arb_sim" / "run_data" / "pool_config.json"
    )
    if not pool_config_path.exists():
        raise FileNotFoundError(f"Missing pool config: {pool_config_path}")
    with open(pool_config_path, "r") as f:
        cfg = json.load(f)

    def resolve_candles_path() -> Path:
        if args.candles:
            return Path(args.candles)
        meta = cfg.get("meta") if isinstance(cfg, dict) else None
        candidate = None
        if isinstance(meta, dict):
            candidate = meta.get("datafile")
        if not candidate:
            raise ValueError(
                "Candles path not provided and meta.datafile missing in pool_config.json"
            )
        cand_path = Path(candidate)
        if not cand_path.is_absolute():
            cand_path = repo_root / cand_path
        return cand_path

    candles_path = resolve_candles_path()
    if not candles_path.exists():
        raise FileNotFoundError(f"Candles file not found: {candles_path}")

    # Invoke the harness once over the entire config
    out_json_path = (
        Path(args.out)
        if args.out
        # repo_root / "python" / "arb_sim" / "run_data" / f"arb_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        else (repo_root / "python" / "arb_sim" / "run_data" / "arb_run_1.json")
    )
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    raw = runner.run(
        pool_config_path,
        candles_path,
        out_json_path,
        n_candles=args.n_candles,
        save_actions=args.save_actions,
        events=args.events,
        min_swap=args.min_swap,
        max_swap=args.max_swap,
        threads=max(1, args.threads),
        dustswapfreq=args.dustswapfreq,
        userswapfreq=args.userswapfreq,
        userswapsize=args.userswapsize,
        userswapthresh=args.userswapthresh,
        apy_period_days=args.apy_period_days,
        apy_period_cap=args.apy_period_cap,
        detailed_log=args.detailed_log,
    )

    runs_raw: List[Dict[str, Any]] = raw.get("runs", [])
    print(f"Time taken: {(datetime.now() - ts).total_seconds()} seconds")

    # Derive x/y and base_pool from pool_config meta
    def get_meta(conf: Dict[str, Any]):
        meta = conf.get("meta", {}) if isinstance(conf, dict) else {}
        grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
        x_name = (grid.get("X") or {}).get("name") if isinstance(grid, dict) else None
        y_name = (grid.get("Y") or {}).get("name") if isinstance(grid, dict) else None
        base_pool = meta.get("base_pool") if isinstance(meta, dict) else None
        return x_name, y_name, base_pool

    x_name, y_name, base_pool_meta = get_meta(cfg)

    # Derive base_pool from actual pools if meta missing
    def pools_list():
        if isinstance(cfg, dict) and "pools" in cfg:
            return cfg["pools"]
        elif isinstance(cfg, dict) and "pool" in cfg:
            return [{"pool": cfg["pool"], "costs": cfg.get("costs", {})}]
        return []

    plist = list(pools_list())
    base_pool: Dict[str, Any] = {}
    if not base_pool_meta:

        def to_strish(v):
            if isinstance(v, list):
                return [str(x) for x in v]
            return str(v)

        if plist:
            keys = set(plist[0].get("pool", {}).keys())
            for e in plist[1:]:
                keys &= set(e.get("pool", {}).keys())
            for k in sorted(keys):
                if k == x_name or k == y_name:
                    continue
                vals = [to_strish(e.get("pool", {}).get(k)) for e in plist]
                if all(v == vals[0] for v in vals):
                    base_pool[k] = vals[0]
    else:
        base_pool = base_pool_meta

    # Enrich runs with x/y keys/values
    enriched_runs: List[Dict[str, Any]] = []
    total_trades = 0
    for rr in runs_raw:
        pool_obj = rr.get("params", {}).get("pool", {})
        xv = str(pool_obj.get(x_name)) if x_name and x_name in pool_obj else None
        yv = str(pool_obj.get(y_name)) if y_name and y_name in pool_obj else None
        # Accumulate total trades for metadata (no duplicate field in result)
        result_obj = rr.get("result", {}) or {}
        try:
            total_trades += int(result_obj.get("trades", 0))
        except Exception:
            pass

        enriched = {
            "x_key": x_name,
            "x_val": xv,
            "y_key": y_name,
            "y_val": yv,
            "result": result_obj,
            "final_state": rr.get("final_state", {}),
        }
        if "actions" in rr:
            enriched["actions"] = rr.get("actions")
        if "states" in rr:
            enriched["states"] = rr.get("states")
        enriched_runs.append(enriched)

    agg = {
        "metadata": {
            "candles_file": str(candles_path),
            "input_is_events": bool(args.events),
            "threads": max(1, args.threads),
            "base_pool": base_pool,
            "grid": cfg.get("meta", {}).get("grid") if isinstance(cfg, dict) else None,
            "candles_read_ms": raw.get("metadata", {}).get("candles_read_ms"),
            "exec_ms": raw.get("metadata", {}).get("exec_ms"),
            "total_trades": total_trades,
        },
        "runs": enriched_runs,
    }
    with open(out_json_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n✓ Wrote aggregated run: {out_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
