#!/usr/bin/env python3
"""
compare_plots: Plot price_scale trajectories from multiple arb_run_*.json and
multiple trades-*.json/JSONL files against a single, shared CEX price series.

- Input is dynamic: any number of arb_run_* files and trades-* files.
- Legend labels use basenames of the input files.
- CEX price must be consistent across all inputs; we align on the intersection
  of timestamps and verify that the price matches within tolerance.

Defaults:
- --arb-dir omitted: python/arb_sim/run_data/
- --trades-dir omitted: ./cryptopool-simulator/ then ./comparison/

Usage examples:
  uv run python python/arb_sim/compare_plots.py \
    --arb-dir python/arb_sim/run_data \
    --trades-dir cryptopool-simulator \
    --out python/arb_sim/plots/compare.png

  # Only arb runs (no trades files)
  uv run python python/arb_sim/compare_plots.py --arb-dir python/arb_sim/run_data --out out.png

  # Only trades files (no arb runs)
  uv run python python/arb_sim/compare_plots.py --trades-dir comparison --out out.png

  # Also plot time-series metrics per run (from actions), one figure per metric
  uv run python python/arb_sim/compare_plots.py --arb-dir python/arb_sim/run_data \
      --metrics inst_slippage,inst_liq_density --out python/arb_sim/plots/compare.png
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

"""
Note on plotting backend:
- We select Agg only when --out is provided (file save mode).
- If --out is omitted, we try interactive show(); if the current backend is
  non-interactive, we fall back to saving under python/arb_sim/plots/compare.png.
"""


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        if (d / ".git").exists():
            return d
        if (d / "README.md").exists() and (d / "python").exists():
            return d
    return here.parents[2]


def _f(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _load_arb_series(path: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    """Extract (ts, p_cex, price_scale) series from all runs in an arb_run_*.json file.

    Returns a dict: label -> list[(ts, p_cex, ps_post)], one series per run that contains actions.
    The label attempts to use pool name if present; otherwise falls back to an index.
    """
    data = json.loads(path.read_text())
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"arb_run JSON has no runs[]: {path}")
    out: Dict[str, List[Tuple[int, float, float]]] = {}
    for idx, r in enumerate(runs):
        actions = r.get("actions")
        if not actions:
            continue
        # Build a short label: prefer pool name; else use x/y or index
        pool_name = None
        pool_obj = (r.get("params") or {}).get("pool") if isinstance(r.get("params"), dict) else None
        if isinstance(pool_obj, dict):
            pool_name = pool_obj.get("name")
        if not pool_name:
            xv = r.get("x_val"); yv = r.get("y_val")
            if xv is not None and yv is not None:
                pool_name = f"x={xv},y={yv}"
        if not pool_name:
            pool_name = f"run_{idx:02d}"
        label = f"{path.name}:{pool_name}"

        series: List[Tuple[int, float, float]] = []
        for a in actions:
            ts = a.get("ts")
            if ts is None:
                continue
            ts = int(ts)
            p_cex = _f(a.get("p_cex"))
            ps_post = _f(a.get("ps_after", a.get("psafter", a.get("ps_post"))))
            # Fallback: if post missing but pre present, record pre to show continuity
            if ps_post is None:
                ps_post = _f(a.get("ps_before"))
            if p_cex is None or ps_post is None:
                continue
            series.append((ts, p_cex, ps_post))
        series.sort(key=lambda t: t[0])
        if series:
            out[label] = series
    return out


def _parse_json_line(line: str) -> Any:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        # Handle ", }" or trailing comma
        sanitized = re.sub(r",\s*}\s*$", "}", line)
        try:
            return json.loads(sanitized)
        except Exception:
            try:
                return json.loads(line.rstrip(","))
            except Exception:
                return None


def _load_trades_series_jsonl(path: Path) -> List[Tuple[int, float, float]]:
    series: List[Tuple[int, float, float]] = []
    for raw in path.read_text().splitlines():
        ev = _parse_json_line(raw)
        if not isinstance(ev, dict):
            continue
        ts = ev.get("t") or ev.get("ts")
        if ts is None:
            continue
        ts = int(ts)
        ty = ev.get("type")
        if ty == "tweak_price":
            p_cex = _f(ev.get("p_cex"))
            ps_post = _f(ev.get("ps_post", ev.get("ps_after")))
            if p_cex is None or ps_post is None:
                # fallback to pre if post missing
                ps_post = _f(ev.get("ps_pre"))
            if p_cex is not None and ps_post is not None:
                series.append((ts, p_cex, ps_post))
        else:
            # Trade lines: often no ps_post, but may include merged tweaks
            p_cex = _f(ev.get("cex_price", ev.get("p_cex")))
            ps_post = _f(ev.get("ps_post", ev.get("ps_after")))
            if p_cex is None or ps_post is None:
                # fallback to pre values
                ps_post = _f(ev.get("ps_pre"))
            if p_cex is not None and ps_post is not None:
                series.append((ts, p_cex, ps_post))
    series.sort(key=lambda t: t[0])
    return series


def _load_trades_series_json(path: Path) -> List[Tuple[int, float, float]]:
    """Support common JSON structures: list of events, or {events:[...]}, or {actions:[...]}.
    Extract (ts, p_cex/cex_price, ps_post/ps_after or fallback ps_pre).
    """
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for key in ("events", "actions", "data"):
            if isinstance(obj.get(key), list):
                items = obj[key]
                break
        else:
            # Best effort: if dict has trade-like keys directly
            items = [obj]
    else:
        items = []
    series: List[Tuple[int, float, float]] = []
    for ev in items:
        if not isinstance(ev, dict):
            continue
        ts = ev.get("t") or ev.get("ts")
        if ts is None:
            continue
        ts = int(ts)
        ty = ev.get("type")
        if ty == "tweak_price":
            p_cex = _f(ev.get("p_cex", ev.get("cex_price")))
            ps_post = _f(ev.get("ps_post", ev.get("ps_after")))
            if p_cex is None or ps_post is None:
                ps_post = _f(ev.get("ps_pre"))
            if p_cex is not None and ps_post is not None:
                series.append((ts, p_cex, ps_post))
        else:
            p_cex = _f(ev.get("cex_price", ev.get("p_cex")))
            ps_post = _f(ev.get("ps_post", ev.get("ps_after")))
            if p_cex is None or ps_post is None:
                ps_post = _f(ev.get("ps_pre"))
            if p_cex is not None and ps_post is not None:
                series.append((ts, p_cex, ps_post))
    series.sort(key=lambda t: t[0])
    return series


def _align_and_validate_cex(all_series: Dict[str, List[Tuple[int, float, float]]], rtol: float, atol: float) -> Tuple[List[int], List[float]]:
    """Return canonical (timestamps, cex_prices) intersected across all series.
    Assert that per-timestamp CEX prices match within tolerance across series.
    """
    if not all_series:
        return [], []
    # Build intersection of timestamps across all series
    ts_sets = []
    for _name, series in all_series.items():
        ts_sets.append({t for (t, _pc, _ps) in series})
    common_ts = set.intersection(*ts_sets) if ts_sets else set()
    if not common_ts:
        # If there is only one series, allow it
        if len(all_series) == 1:
            only = next(iter(all_series.values()))
            return [t for (t, _pc, _ps) in only], [pc for (_t, pc, _ps) in only]
        raise SystemExit("No common timestamps across inputs — cannot align CEX price.")

    # Use the first series as canonical reference for cex
    first_name = next(iter(all_series.keys()))
    ref = {t: pc for (t, pc, _ps) in all_series[first_name] if t in common_ts}
    # Verify all match
    for name, series in all_series.items():
        if name == first_name:
            continue
        cur = {t: pc for (t, pc, _ps) in series if t in common_ts}
        for t in common_ts:
            a, b = ref.get(t), cur.get(t)
            if a is None or b is None:
                continue
            if abs(a - b) > max(atol, rtol * max(abs(a), abs(b), 1.0)):
                raise SystemExit(f"CEX price mismatch at ts={t} for {name}: {a} vs {b}")
    # Sorted canonical series
    ts_sorted = sorted(common_ts)
    cex_series = [ref[t] for t in ts_sorted]
    return ts_sorted, cex_series


def _extract_metric_series_from_arb(path: Path, metric: str) -> Dict[str, List[Tuple[int, float]]]:
    """Return per-run time series for an action-level metric.

    Expects metric to be a numeric field on actions, e.g., inst_slippage, inst_liq_density.
    Labeling mirrors _load_arb_series: uses pool name if present, else x/y, else index.
    """
    data = json.loads(path.read_text())
    runs = data.get("runs", [])
    if not runs:
        return {}
    out: Dict[str, List[Tuple[int, float]]] = {}
    for idx, r in enumerate(runs):
        actions = r.get("actions")
        if not actions:
            continue
        pool_name = None
        pool_obj = (r.get("params") or {}).get("pool") if isinstance(r.get("params"), dict) else None
        if isinstance(pool_obj, dict):
            pool_name = pool_obj.get("name")
        if not pool_name:
            xv = r.get("x_val"); yv = r.get("y_val")
            if xv is not None and yv is not None:
                pool_name = f"x={xv},y={yv}"
        if not pool_name:
            pool_name = f"run_{idx:02d}"
        label = f"{path.name}:{pool_name}"
        series: List[Tuple[int, float]] = []
        # Build preferred key order: exact metric, alias variants
        keys = [metric]
        if metric.startswith("inst_"):
            keys.append(metric[len("inst_"):])
        else:
            keys.append("inst_" + metric)
        # Also support historical names
        if metric == "slippage" and "inst_slippage" not in keys:
            keys.append("inst_slippage")
        if metric == "liq_density" and "inst_liq_density" not in keys:
            keys.append("inst_liq_density")

        for a in actions:
            ts = a.get("ts")
            if ts is None:
                continue
            v = None
            for k in keys:
                if k in a and a[k] is not None:
                    v = a.get(k)
                    break
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            series.append((int(ts), fv))
        series.sort(key=lambda t: t[0])
        if series:
            out[label] = series
    return out


def _align_union_timestamps(series_by_name: Dict[str, List[Tuple[int, float]]]) -> Tuple[List[int], Dict[str, List[float]]]:
    """Build a union time grid across runs and carry-forward values per run to align.

    Returns (ts_union, aligned_values_by_name)
    """
    ts_union = sorted({t for s in series_by_name.values() for (t, _v) in s})
    aligned: Dict[str, List[float]] = {}
    for name, s in series_by_name.items():
        # map ts->value
        m = {t: v for (t, v) in s}
        last = None
        vals: List[float] = []
        for t in ts_union:
            v = m.get(t)
            if v is None:
                vals.append(last)
            else:
                vals.append(v)
                last = v
        aligned[name] = vals
    return ts_union, aligned


def main():
    ap = argparse.ArgumentParser(description="Plot price_scale vs shared CEX price for multiple arb_run_* and trades-* files")
    ap.add_argument("--arb-dir", type=str, default=None, help="Directory with arb_run_*.json files (default: python/arb_sim/run_data)")
    ap.add_argument("--trades-dir", type=str, default=None, help="Directory with trades-*.json or trades-*.jsonl files (default: cryptopool-simulator or comparison)")
    ap.add_argument("--out", type=str, default=None, help="Output image path (png). If not set, shows the plot interactively.")
    ap.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance for CEX price equality check")
    ap.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance for CEX price equality check")
    ap.add_argument("--title", type=str, default="price_scale vs CEX", help="Plot title")
    ap.add_argument("--xlim", type=int, nargs=2, default=None, help="Optional x-axis (timestamp) limits: start end")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="Optional y-axis limits: min max")
    ap.add_argument("--metrics", type=str, default=None, help="Comma-separated list of action metrics to plot by time (e.g., inst_slippage,inst_liq_density)")
    args = ap.parse_args()

    root = _repo_root()

    # Import matplotlib after parsing so we can decide backend
    import matplotlib as mpl
    if args.out:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve arb_dir
    arb_dir = Path(args.arb_dir) if args.arb_dir else (root / "python" / "arb_sim" / "run_data")
    if not arb_dir.exists():
        print(f"Note: arb-dir not found: {arb_dir}")

    # Resolve trades_dir or default single-file selection like compare_sims.py
    trades_dir = None
    default_trade_candidates: List[Path] = []
    if args.trades_dir:
        trades_dir = Path(args.trades_dir)
    else:
        default_trade_candidates = [
            (root.parent / "cryptopool-simulator" / "trades-0.jsonl").resolve(),
            (root / "cryptopool-simulator" / "trades-0.jsonl").resolve(),
            (root / "comparison" / "trades-0.jsonl").resolve(),
            (root.parent / "cryptopool-simulator" / "trades-0.json").resolve(),
            (root / "cryptopool-simulator" / "trades-0.json").resolve(),
            (root / "comparison" / "trades-0.json").resolve(),
        ]
    
    arb_files: List[Path] = []
    if arb_dir and arb_dir.exists():
        arb_files = sorted(p for p in arb_dir.glob("arb_run_*.json"))

    trade_files: List[Path] = []
    if trades_dir and trades_dir.exists():
        trade_files = sorted(list(trades_dir.glob("trades-*.json")) + list(trades_dir.glob("trades-*.jsonl")))
    else:
        # Use first matching default candidate if present (compare_sims default behavior)
        default_trade = next((p for p in default_trade_candidates if p.exists()), None)
        if default_trade is not None:
            trade_files = [default_trade]
        else:
            print("Note: No trades-dir provided and no default trades-0.(json|jsonl) found; proceeding without trades.")

    if not arb_files and not trade_files:
        raise SystemExit("No input files found. Provide --arb-dir and/or --trades-dir with matching files.")

    # Load series
    series_by_name: Dict[str, List[Tuple[int, float, float]]] = {}
    for p in arb_files:
        try:
            multi = _load_arb_series(p)
            for name, s in multi.items():
                series_by_name[name] = s
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
    for p in trade_files:
        try:
            if p.suffix.lower() == ".jsonl":
                s = _load_trades_series_jsonl(p)
            else:
                s = _load_trades_series_json(p)
            if s:
                series_by_name[p.name] = s
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")

    if not series_by_name and not args.metrics:
        raise SystemExit("No usable series extracted from inputs.")

    # If metrics provided, create a single figure with subplots: price on top, metrics below.
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
        n_sub = (1 if series_by_name else 0) + len(metrics)
        if n_sub == 0:
            raise SystemExit("No data to plot: no price series and no metrics found.")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n_sub, 1, figsize=(12, max(4, 3*n_sub)), sharex=True, constrained_layout=True)
        if n_sub == 1:
            axes = [axes]

        global_min_ts = None
        global_max_ts = None

        ax_idx = 0
        if series_by_name:
            ts, cex = _align_and_validate_cex(series_by_name, args.rtol, args.atol)
            axes[ax_idx].plot(ts, cex, label="cex price", color="orange", linewidth=1.0, alpha=0.8)
            for name, s in series_by_name.items():
                m = {t: ps for (t, _pc, ps) in s}
                y = [m.get(t) for t in ts]
                yy: List[float] = []
                last = None
                for v in y:
                    if v is None:
                        yy.append(last)
                    else:
                        yy.append(v)
                        last = v
                axes[ax_idx].plot(ts, yy, label=name)
            axes[ax_idx].set_title(args.title)
            axes[ax_idx].set_ylabel("price")
            axes[ax_idx].legend(loc="best")
            axes[ax_idx].grid(True, alpha=0.25)
            if ts:
                global_min_ts = ts[0] if global_min_ts is None else min(global_min_ts, ts[0])
                global_max_ts = ts[-1] if global_max_ts is None else max(global_max_ts, ts[-1])
            ax_idx += 1

        # Metric subplots
        for metric in metrics:
            metric_series: Dict[str, List[Tuple[int, float]]] = {}
            for p in arb_files:
                metric_series.update(_extract_metric_series_from_arb(p, metric))
            ax = axes[ax_idx]
            if not metric_series:
                ax.set_title(f"{metric}: no data")
                ax_idx += 1
                continue
            ts_union, aligned = _align_union_timestamps(metric_series)
            for name, vals in aligned.items():
                ax.plot(ts_union, vals, label=name)
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            if ts_union:
                global_min_ts = ts_union[0] if global_min_ts is None else min(global_min_ts, ts_union[0])
                global_max_ts = ts_union[-1] if global_max_ts is None else max(global_max_ts, ts_union[-1])
            ax_idx += 1

        # Set shared x limits
        if global_min_ts is not None and global_max_ts is not None:
            axes[0].set_xlim(global_min_ts, global_max_ts)
        axes[-1].set_xlabel("timestamp")

        # Optional user xlim/ylim applied to all
        if args.xlim:
            for ax in axes:
                ax.set_xlim(args.xlim[0], args.xlim[1])
        if args.ylim:
            for ax in axes:
                ax.set_ylim(args.ylim[0], args.ylim[1])

        # Save or show
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            print(f"✓ Saved combined plot to {out_path}")
        else:
            backend = plt.get_backend().lower()
            interactive = {"macosx", "qt5agg", "qtagg", "tkagg", "gtk3agg", "nbagg", "webagg"}
            if backend in interactive:
                plt.show()
            else:
                default_out = root / "python" / "arb_sim" / "plots" / "compare_combined.png"
                default_out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(default_out, dpi=150)
                print(f"(non-interactive backend: {backend}) Saved plot to {default_out}")
        return 0

    # No metrics requested: keep price-only behavior
    if series_by_name:
        ts, cex = _align_and_validate_cex(series_by_name, args.rtol, args.atol)
        if not ts:
            raise SystemExit("Empty canonical timeline after alignment.")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(ts, cex, label="cex price", color="orange", linewidth=1.0, alpha=0.8)
        for name, s in series_by_name.items():
            m = {t: ps for (t, _pc, ps) in s}
            y = [m.get(t) for t in ts]
            yy: List[float] = []
            last = None
            for v in y:
                if v is None:
                    yy.append(last)
                else:
                    yy.append(v)
                    last = v
            plt.plot(ts, yy, label=name)
        plt.title(args.title)
        plt.xlabel("timestamp")
        plt.ylabel("price")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.25)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print(f"✓ Saved plot to {out_path}")
        else:
            backend = plt.get_backend().lower()
            interactive = {"macosx", "qt5agg", "qtagg", "tkagg", "gtk3agg", "nbagg", "webagg"}
            if backend in interactive:
                plt.tight_layout()
                plt.show()
            else:
                default_out = root / "python" / "arb_sim" / "plots" / "compare.png"
                default_out.parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(default_out, dpi=150)
                print(f"(non-interactive backend: {backend}) Saved plot to {default_out}")
    return 0


if __name__ == "__main__":
    main()
