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
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Use non-interactive backend for headless systems (must be before pyplot import)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "run_data"

LN2 = math.log(2)
SECONDS_PER_HOUR = 3600.0
MA_TIME_TO_HOURS = LN2 / SECONDS_PER_HOUR


def _edges_from_centers(centers: List[float], log_scale: bool = False) -> np.ndarray:
    """Compute bin edges from bin centers for pcolormesh.

    - Linear: arithmetic midpoints, with first/last extrapolated by mirroring.
    - Log: operate in log-space using geometric midpoints; requires centers > 0.
    """
    c = np.asarray(centers, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("centers must be a 1D non-empty sequence")
    if c.size == 1:
        # Degenerate case: create a tiny band around the single center
        x0 = float(c[0])
        if log_scale:
            if x0 <= 0:
                raise ValueError("log-scale edges require positive centers")
            f = np.sqrt(10.0)
            return np.array([x0 / f, x0 * f])
        else:
            d = abs(x0) * 0.5 if x0 != 0 else 0.5
            return np.array([x0 - d, x0 + d])

    if log_scale:
        if np.any(c <= 0):
            raise ValueError("log-scale edges require all centers > 0")
        logs = np.log(c)
        mids = (logs[:-1] + logs[1:]) / 2.0
        first = logs[0] - (mids[0] - logs[0])  # mirror distance
        last = logs[-1] + (logs[-1] - mids[-1])
        edges_log = np.concatenate([[first], mids, [last]])
        return np.exp(edges_log)
    else:
        mids = (c[:-1] + c[1:]) / 2.0
        first = c[0] - (mids[0] - c[0])
        last = c[-1] + (c[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]])


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


def _axis_transformer(name: str):
    """Return a transformer for axis values based on the parameter name."""
    key = (name or "").lower()
    if "ma_time" in key:

        def _transform(value: float) -> float:
            return value * MA_TIME_TO_HOURS

        return _transform
    return lambda value: value


def _extract_grid(
    data: Dict[str, Any], metric: str, scale_1e18: bool, scale_percent: bool
) -> Tuple[str, str, List[float], List[float], np.ndarray]:
    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("No runs[] found in arb_run JSON")

    # Determine axis names
    x_name = (
        runs[0].get("x_key")
        or data.get("metadata", {}).get("grid", {}).get("X", {}).get("name")
        or "X"
    )
    y_name = (
        runs[0].get("y_key")
        or data.get("metadata", {}).get("grid", {}).get("Y", {}).get("name")
        or "Y"
    )

    x_transform = _axis_transformer(x_name)
    y_transform = _axis_transformer(y_name)

    # Collect unique axis values and z values
    points: Dict[Tuple[float, float], float] = {}
    xs: List[float] = []
    ys: List[float] = []
    for r in runs:
        raw_x = r.get("x_val")
        raw_y = r.get("y_val")
        xv = _to_float(raw_x) if raw_x is not None else float("nan")
        yv = _to_float(raw_y) if raw_y is not None else float("nan")
        xv = x_transform(xv)
        yv = y_transform(yv)
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
        if math.isfinite(xv) and math.isfinite(yv) and np.isfinite(z):
            points[(xv, yv)] = z
            xs.append(xv)
            ys.append(yv)

    if not points:
        raise SystemExit(
            "No valid (x,y,z) points found in runs; ensure x_val/y_val and final_state exist."
        )

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
    if "fee" in key and "gamma" not in key:
        # We will compute bps directly in labels, return sentinel scale 0
        return 0.0, " (bps)"
    if "ma_time" in key:
        suffix = "" if "hrs" in key else " (hrs)"
        return 1.0, suffix
    if "gamma" in key:
        return 1e18, " (/1e18)"
    if "liquidity" in key or "balance" in key:
        return 1e18, " (/1e18)"
    return 1.0, ""


def _format_axis_labels(name: str, values: List[float]) -> Tuple[List[str], str]:
    scale, suffix = _axis_normalization(name)
    labels: List[str] = []
    key = (name or "").lower()
    display_name = name or ""
    if suffix and suffix not in (display_name or ""):
        display_name = f"{display_name}{suffix}"
    if scale == 0.0 and "fee" in key and "gamma" not in key:
        # Convert 1e10-scaled fee to bps: val/1e10 * 1e4
        labels = [f"{(v / 1e10 * 1e4):.2f}" for v in values]
        return labels, f"{name} (bps)"
    if "ma_time" in key:
        labels = [f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}" for v in values]
        return labels, display_name
    if "gamma" in key:
        labels = [f"{(v / 1e18):.5f}" for v in values]
        return labels, f"{name}"
    if scale != 1.0:
        labels = [f"{(v / scale):.2f}" for v in values]
        return labels, display_name
    # default
    labels = [f"{v:.2f}" for v in values]
    return labels, display_name


def _auto_font_size(nx: int, ny: int) -> int:
    """Choose a tick font size based on grid resolution.

    - <=4 cells → 10 pt
    - <=8 cells → 12 pt
    - <=16 cells → 14 pt
    - >=32 cells → 24 pt (cap)
    - between 16 and 32 uses linear interpolation toward 24 pt
    """
    grid = max(1, nx, ny)
    if grid <= 4:
        return 8
    if grid <= 8:
        return 12
    if grid <= 16:
        return 16
    if grid >= 32:
        return 24
    span = 32 - 16
    t = (grid - 16) / span
    size = 12 + t * (24 - 12)
    return int(round(size))


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

    ap = argparse.ArgumentParser(description="Plot heatmap(s) from arb_run grid")
    ap.add_argument(
        "--arb", type=str, default=None, help="Path to arb_run_*.json (default: latest)"
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="virtual_price",
        help="Single metric for Z (default: virtual_price)",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated list of metrics to plot side-by-side (overrides --metric)",
    )
    ap.add_argument(
        "--no-scale", action="store_true", help="Disable 1e18 scaling for Z"
    )
    ap.add_argument(
        "--cmap", type=str, default="turbo", help="Matplotlib colormap (default: turbo)"
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: run_data/heatmap_<metric>.png)",
    )
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    ap.add_argument("--annot", action="store_true", help="Annotate cells with values")
    ap.add_argument(
        "--max-xticks", type=int, default=12, help="Max X tick labels (default: 12)"
    )
    ap.add_argument(
        "--max-yticks", type=int, default=12, help="Max Y tick labels (default: 12)"
    )
    ap.add_argument(
        "--font-size",
        type=int,
        default=0,
        help="Tick label font size (default: auto)",
    )
    ap.add_argument(
        "--log-x",
        dest="log_x",
        action="store_true",
        help="Use log scale on X (default: disabled)",
        default=False,
    )
    ap.add_argument(
        "--no-log-x", dest="log_x", action="store_false", help="Disable log scale on X"
    )
    ap.add_argument(
        "--log-y",
        dest="log_y",
        action="store_true",
        help="Use log scale on Y (default: disabled)",
        default=False,
    )
    ap.add_argument(
        "--no-log-y", dest="log_y", action="store_false", help="Disable log scale on Y"
    )
    ap.add_argument(
        "--square",
        dest="square",
        action="store_true",
        help="Force a square plot with square cells (default)",
    )
    ap.add_argument(
        "--no-square",
        dest="square",
        action="store_false",
        help="Disable square plot; size adapts to grid",
    )
    ap.add_argument(
        "--ncol",
        type=int,
        default=3,
        help="Number of columns for multi-metric layout (default: 3)",
    )
    ap.add_argument(
        "--clamp", action="store_true", default=False, help="Clamp negative values to 0"
    )
    ap.set_defaults(square=True)
    args = ap.parse_args()

    arb_path = Path(args.arb) if args.arb else _latest_arb_run()
    data = _load(arb_path)

    # Determine metrics list
    metrics: List[str]
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = [args.metric]

    # Build first grid to define axes
    def metric_scale_flags(m: str) -> Tuple[bool, bool]:
        mlow = (m or "").lower()
        scale_1e18 = (not args.no_scale) and m in {
            "virtual_price",
            "xcp_profit",
            "price_scale",
            "D",
            "totalSupply",
        }
        scale_percent = (
            mlow in {"vpminusone", "apy"}
            or "apy" in mlow
            or "tw_real_slippage" in mlow
            or "geom_mean" in mlow
        )
        return scale_1e18, scale_percent

    first_m = metrics[0]
    s18, sperc = metric_scale_flags(first_m)
    x_name, y_name, xs, ys, Z0 = _extract_grid(data, first_m, s18, sperc)

    base_font = (
        args.font_size if args.font_size > 0 else _auto_font_size(len(xs), len(ys))
    )
    tick_font = base_font
    label_font = max(8, base_font + 2)
    title_font = max(label_font, base_font + 4)
    annot_font = max(6, base_font - 2)
    colorbar_font = base_font

    # Prepare figure with configurable columns
    n = len(metrics)
    cols = max(1, int(args.ncol))
    cols = min(cols, n) if n > 0 else cols
    rows = int(np.ceil(n / cols)) if n > 0 else 1

    if args.square:
        base = max(len(xs), len(ys))
        side = max(4.5, min(12.0, 0.35 * max(1, base)))
        fig_w, fig_h = side * cols, side * rows
        fig, axes = plt.subplots(
            rows, cols, figsize=(fig_w, fig_h), constrained_layout=True
        )
    else:
        unit_w = max(5.5, min(12.0, 0.35 * max(1, len(xs))))
        unit_h = max(4.0, min(10.0, 0.30 * max(1, len(ys))))
        fig_w, fig_h = unit_w * cols, unit_h * rows
        fig, axes = plt.subplots(
            rows, cols, figsize=(fig_w, fig_h), constrained_layout=True
        )
    axes = np.atleast_1d(axes).reshape(rows, cols)

    # Precompute tick indices and labels
    xticks = _select_ticks(xs, args.max_xticks)
    yticks = _select_ticks(ys, args.max_yticks)
    xlab_full, xlabel = _format_axis_labels(x_name, xs)
    ylab_full, ylabel = _format_axis_labels(y_name, ys)
    xlabels = [xlab_full[i] for i in xticks]
    ylabels = [ylab_full[i] for i in yticks]

    # Plot each metric in row-major order, hide any unused axes
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx >= n:
                ax.axis("off")
                continue
            m = metrics[idx]
            s18, sperc = metric_scale_flags(m)
            _, _, _, _, Z = _extract_grid(data, m, s18, sperc)
            # Optional log axes controlled via CLI
            log_x_flag = bool(args.log_x)
            log_y_flag = bool(args.log_y)
            if log_x_flag and any(x <= 0 for x in xs):
                log_x_flag = False
            if log_y_flag and any(y <= 0 for y in ys):
                log_y_flag = False

            if log_x_flag or log_y_flag:
                Xedges = _edges_from_centers(xs, log_x_flag)
                Yedges = _edges_from_centers(ys, log_y_flag)
                # Use edges with pcolormesh and auto shading so
                # (len(xs)+1, len(ys)+1) edges match C=(len(ys),len(xs))
                im = ax.pcolormesh(Xedges, Yedges, Z, cmap=args.cmap, shading="auto")
                if log_x_flag:
                    try:
                        ax.set_xscale("log")
                    except Exception:
                        pass
                if log_y_flag:
                    try:
                        ax.set_yscale("log")
                    except Exception:
                        pass
                if args.square:
                    ny, nx = Z.shape
                    try:
                        ax.set_box_aspect(ny / nx)
                    except Exception:
                        ax.set_aspect("equal", adjustable="box")
                # Tick placement at data values
                ax.set_xticks([xs[i] for i in xticks])
                ax.set_yticks([ys[i] for i in yticks])
            else:
                aspect = "auto"
                im = ax.imshow(Z, origin="lower", aspect=aspect, cmap=args.cmap)
                if args.square:
                    ny, nx = Z.shape
                    try:
                        ax.set_box_aspect(ny / nx)
                    except Exception:
                        ax.set_aspect("equal", adjustable="box")
                # Tick placement at index positions
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=tick_font)
            # Only label y on first column
            if c == 0:
                ax.set_yticklabels(ylabels, fontsize=tick_font)
                ax.set_ylabel(ylabel, fontsize=label_font)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(xlabel, fontsize=label_font)
            title_scale = " (%)" if sperc else (" (scaled 1e18)" if s18 else "")
            ax.set_title(f"{m}{title_scale}", fontsize=title_font)

            # Compute absolute min/max for color mapping and ticks
            finite_vals = Z[np.isfinite(Z)]
            # clamp negatives in --clamp flag is used
            if args.clamp:
                finite_vals = np.where(finite_vals < 0, 0, finite_vals)
            if finite_vals.size:
                zmin = float(np.min(finite_vals))
                zmax = float(np.max(finite_vals))
                if zmax == zmin:
                    # Avoid degenerate clim; expand slightly
                    eps = 1e-12 if zmax == 0 else abs(zmax) * 1e-12
                    im.set_clim(zmin - eps, zmax + eps)
                    tick_vals = [zmin]
                else:
                    im.set_clim(zmin, zmax)
                    # Ensure min and max appear as ticks
                    tick_vals = list(np.linspace(zmin, zmax, num=5))
            else:
                tick_vals = None

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(m + (" (%)" if sperc else ""), fontsize=colorbar_font)
            cb.ax.tick_params(labelsize=tick_font)
            if tick_vals is not None:
                cb.set_ticks(tick_vals)
            # Fixed decimal formatter for colorbar ticks
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
            if args.annot:
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        val = Z[i, j]
                        if not np.isfinite(val):
                            continue
                        if log_x_flag or log_y_flag:
                            x_pos = xs[j]
                            y_pos = ys[i]
                        else:
                            x_pos = j
                            y_pos = i
                        ax.text(
                            x_pos,
                            y_pos,
                            f"{val:.3g}",
                            va="center",
                            ha="center",
                            color="white",
                            fontsize=annot_font,
                        )
            idx += 1

    # Output
    if args.out:
        out_path = Path(args.out)
    else:
        if len(metrics) == 1:
            out_path = RUN_DIR / f"heatmap_{metrics[0]}.png"
        else:
            tag = "_".join(m.replace(" ", "") for m in metrics[:3])
            if len(metrics) > 3:
                tag += f"_plus{len(metrics) - 3}"
            out_path = RUN_DIR / f"heatmaps_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap(s) to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
