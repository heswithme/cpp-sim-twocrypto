#!/usr/bin/env python3
"""
Plot a heatmap from the latest (or given) arb_sim aggregated run JSON.

Assumes a two-parameter grid (X, Y) across runs and uses the final_state
metric as Z. By default, Z = virtual_price / 1e18.

Usage:
  uv run python/arb_sim/plot_heatmap.py                  # latest arb_run_*
  uv run python/arb_sim/plot_heatmap.py --arb <path>     # explicit file
  uv run python/arb_sim/plot_heatmap.py --metric D       # other metric
  uv run python/arb_sim/plot_heatmap.py --out heat.png   # save to file
  uv run python/arb_sim/plot_heatmap.py --show           # display window
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "run_data"


def _latest_arb_run() -> Path:
    files = sorted([p for p in RUN_DIR.glob("arb_run_*.json")])
    if not files:
        raise SystemExit(f"No arb_run_*.json found under {RUN_DIR}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        try:
            return float(int(x))
        except Exception:
            return float("nan")


def _extract_grid(data: Dict[str, Any], metric: str, scale_1e18: bool, scale_percent: bool) -> Tuple[str, str, List[float], List[float], np.ndarray]:
    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("No runs[] found in arb_run JSON")

    # Determine axis names
    x_name = runs[0].get("x_key") or data.get("metadata", {}).get("grid", {}).get("X", {}).get("name") or "X"
    y_name = runs[0].get("y_key") or data.get("metadata", {}).get("grid", {}).get("Y", {}).get("name") or "Y"

    # Collect unique axis values and z values
    points: Dict[Tuple[float, float], float] = {}
    xs: List[float] = []
    ys: List[float] = []
    for r in runs:
        xv = _to_float(r.get("x_val")) if r.get("x_val") is not None else float("nan")
        yv = _to_float(r.get("y_val")) if r.get("y_val") is not None else float("nan")
        fs = r.get("final_state", {})
        res = r.get("result", {})
        # Prefer final_state; fall back to result summary (for metrics like apy)
        if metric in fs:
            z = _to_float(fs.get(metric))
        else:
            z = _to_float(res.get(metric))
        if scale_1e18 and np.isfinite(z):
            z = z / 1e18
        if scale_percent and np.isfinite(z):
            z = z * 100.0
        if np.isfinite(xv) and np.isfinite(yv) and np.isfinite(z):
            points[(xv, yv)] = z
            xs.append(xv)
            ys.append(yv)

    if not points:
        raise SystemExit("No valid (x,y,z) points found in runs; ensure x_val/y_val and final_state exist.")

    xs_sorted = sorted(sorted(set(xs)))
    ys_sorted = sorted(sorted(set(ys)))

    Z = np.full((len(ys_sorted), len(xs_sorted)), np.nan)
    for (xv, yv), z in points.items():
        i = ys_sorted.index(yv)
        j = xs_sorted.index(xv)
        Z[i, j] = z

    return x_name, y_name, xs_sorted, ys_sorted, Z


def _axis_normalization(name: str) -> Tuple[float, str]:
    """Return (scale_factor, unit_suffix) for axis values based on key name.

    - A: stored with 1e4 multiplier → divide by 1e4
    - *fee*: stored with 1e10 scale → convert to bps: value/1e10 * 1e4
      (equivalently, scale = 1e10/1e4 = 1e6; but we compute directly for clarity)
    - *liquidity* or *balance*: stored with 1e18 → divide by 1e18
    - default: scale 1.0, no suffix
    """
    key = (name or "").lower()
    if name == "A" or key == "a":
        return 1e4, " (÷1e4)"
    if "fee" in key:
        # We will compute bps directly in labels, return sentinel scale 0
        return 0.0, " (bps)"
    if "liquidity" in key or "balance" in key:
        return 1e18, " (/1e18)"
    return 1.0, ""


def _format_axis_labels(name: str, values: List[float]) -> Tuple[List[str], str]:
    scale, suffix = _axis_normalization(name)
    labels: List[str] = []
    if scale == 0.0 and "fee" in (name or "").lower():
        # Convert 1e10-scaled fee to bps: val/1e10 * 1e4
        labels = [f"{(v / 1e10 * 1e4):.1f}" for v in values]
        return labels, f"{name} (bps)"
    if scale != 1.0:
        labels = [f"{(v / scale):.1f}" for v in values]
        return labels, f"{name}{suffix}"
    # default
    labels = [f"{v:.1f}" for v in values]
    return labels, name


def _select_ticks(values: List[float], max_ticks: int) -> List[int]:
    n = len(values)
    if n == 0:
        return []
    if max_ticks <= 0 or n <= max_ticks:
        return list(range(n))
    idxs = np.linspace(0, n - 1, num=max_ticks, dtype=int)
    uniq = sorted(set(int(i) for i in idxs))
    if uniq[-1] != n - 1:
        uniq[-1] = n - 1
    return uniq


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Plot heatmap from arb_run grid")
    ap.add_argument("--arb", type=str, default=None, help="Path to arb_run_*.json (default: latest)")
    ap.add_argument("--metric", type=str, default="virtual_price", help="Final_state metric for Z (default: virtual_price)")
    ap.add_argument("--no-scale", action="store_true", help="Disable 1e18 scaling for Z")
    ap.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap (default: viridis)")
    ap.add_argument("--out", type=str, default=None, help="Output image path (default: run_data/heatmap_<metric>.png)")
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    ap.add_argument("--annot", action="store_true", help="Annotate cells with values")
    ap.add_argument("--max-xticks", type=int, default=12, help="Max X tick labels (default: 12)")
    ap.add_argument("--max-yticks", type=int, default=12, help="Max Y tick labels (default: 12)")
    ap.add_argument("--font-size", type=int, default=16, help="Tick label font size (default: 12)")
    args = ap.parse_args()

    arb_path = Path(args.arb) if args.arb else _latest_arb_run()
    data = _load(arb_path)

    # Heuristic: scale by 1e18 for common 1e18-scaled metrics unless --no-scale
    metric = args.metric
    scale_1e18 = (not args.no_scale) and metric in {"virtual_price", "xcp_profit", "price_scale", "D", "totalSupply"}
    scale_percent = ('apy' in metric.lower())

    x_name, y_name, xs, ys, Z = _extract_grid(data, metric, scale_1e18, scale_percent)

    # Scale figure size to grid density to reduce overlap
    fig_w = max(6.0, min(14.0, 0.4 * max(1, len(xs))))
    fig_h = max(4.0, min(10.0, 0.3 * max(1, len(ys))))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=args.cmap)
    # Downsample ticks to avoid clutter
    xticks = _select_ticks(xs, args.max_xticks)
    yticks = _select_ticks(ys, args.max_yticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # Normalized labels for selected ticks
    xlab_full, xlabel = _format_axis_labels(x_name, xs)
    ylab_full, ylabel = _format_axis_labels(y_name, ys)
    xlabels = [xlab_full[i] for i in xticks]
    ylabels = [ylab_full[i] for i in yticks]
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=args.font_size)
    ax.set_yticklabels(ylabels, fontsize=args.font_size)
    ax.set_xlabel(xlabel, fontsize=args.font_size + 2)
    ax.set_ylabel(ylabel, fontsize=args.font_size + 2)
    if scale_percent:
        title_scale = " (%)"
    else:
        title_scale = " (scaled 1e18)" if scale_1e18 else ""
    ax.set_title(f"Heatmap: {metric}{title_scale}", fontsize=args.font_size + 4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(metric + (" (%)" if scale_percent else ""), fontsize=args.font_size)
    cb.ax.tick_params(labelsize=args.font_size)

    if args.annot:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                val = Z[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3g}", va="center", ha="center", color="white", fontsize=max(6, args.font_size - 2))
    # Output
    out_path = Path(args.out) if args.out else (RUN_DIR / f"heatmap_{metric}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
