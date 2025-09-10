#!/usr/bin/env python3
"""
Plot diffs between a current arb_run_*.json and a reference results.json.

- Uses parsing logic consistent with plot_heatmap.py to build the (X,Y)->Z grid.
- Reference loader accepts ref_plot_all-style results.json with:
  { "configuration": [ {"A":..., "mid_fee":..., "Result": { ...metrics... } }, ... ] }

Usage examples:
  uv run python/arb_sim/plot_diffs.py \
      --metrics apy,apy_coin0,apy_coin0_boost,total_notional_coin0,trades,donation_coin0_total

  uv run python/arb_sim/plot_diffs.py --arb path/to/arb_run.json --ref path/to/results.json --relative
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import matplotlib.cm as cm


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


def _extract_grid(data: Dict[str, Any], metric: str, scale_percent: bool, scale_1e18: bool) -> Tuple[str, str, List[float], List[float], np.ndarray]:
    runs = data.get("runs", [])
    if not runs:
        raise SystemExit("No runs[] found in arb_run JSON")

    x_name = runs[0].get("x_key") or data.get("metadata", {}).get("grid", {}).get("X", {}).get("name") or "X"
    y_name = runs[0].get("y_key") or data.get("metadata", {}).get("grid", {}).get("Y", {}).get("name") or "Y"

    points: Dict[Tuple[float, float], float] = {}
    xs: List[float] = []
    ys: List[float] = []
    for r in runs:
        # Prefer explicit x_val/y_val; fallback to params.pool using axis names
        xv_raw = r.get("x_val")
        yv_raw = r.get("y_val")
        if xv_raw is None or yv_raw is None:
            pool_obj = (r.get("params") or {}).get("pool", {})
            xv_raw = pool_obj.get(x_name) if xv_raw is None else xv_raw
            yv_raw = pool_obj.get(y_name) if yv_raw is None else yv_raw
        xv = _to_float(xv_raw) if xv_raw is not None else float("nan")
        yv = _to_float(yv_raw) if yv_raw is not None else float("nan")
        fs = r.get("final_state", {})
        res = r.get("result", {})
        if metric in fs:
            z = _to_float(fs.get(metric))
        else:
            z = _to_float(res.get(metric))
        if scale_percent and np.isfinite(z):
            z = z * 100.0
        if scale_1e18 and np.isfinite(z):
            z = z / 1e18
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


def _load_ref_results(path: Path) -> Dict[str, Any]:
    return _load(path)


def _build_ref_grid(ref: Dict[str, Any], x_name: str, y_name: str, xs: List[float], ys: List[float],
                    metric: str, scale_percent: bool) -> np.ndarray:
    # Map our metric name to ref key variants
    name_map = {
        "apy": "APY",
        "apy_coin0": "APY_coin0",
        "apy_coin0_boost": "APY_coin0_boost",
        "total_notional_coin0": "trade_volume",
        "trades": "n_trades",
        "donation_coin0_total": "donation_coin0_total",
        "arb_pnl_coin0": "arb_profit_coin0",
        "n_rebalances": "n_rebalances",
        "vpminusone": "vpminusone",
        # New relative tracking metrics (bps)
        "avg_rel_bps": "avg_rel_bps",
        "max_rel_bps": "max_rel_bps",
        "cex_follow_time_frac": "cex_follow_time_frac",
        "xcp_profit": "xcp_profit",
    }
    ref_key = name_map.get(metric, metric)

    rows = ref.get("configuration", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Reference results.json missing 'configuration' list")

    # Build list of reference points in reference axis units
    pts: List[Tuple[float, float, float]] = []  # (x_ref, y_ref, z)
    for row in rows:
        try:
            xv = _to_float(row[x_name])
            yv = _to_float(row[y_name])
            res = row.get("Result", {})
            z = _to_float(res.get(ref_key))
        except Exception:
            continue
        if np.isfinite(xv) and np.isfinite(yv) and np.isfinite(z):
            pts.append((xv, yv, z * (100.0 if scale_percent else 1.0)))

    # Convert our new-run axes to reference axis units for alignment
    def _new_to_ref_scale(name: str) -> float:
        key = (name or "").lower()
        if name == "A" or key == "a":
            return 1e4  # new stores A with 1e4 multiplier
        if "fee" in key:
            return 1e10  # new stores fees with 1e10 multiplier
        if "xcp_profit" in key:
            return 1e18  # new stores xcp_profit with 1e18 multiplier
        return 1.0

    sx = _new_to_ref_scale(x_name)
    sy = _new_to_ref_scale(y_name)

    def _close(a: float, b: float, rel: float = 1e-8, abs_tol: float = 1e-12) -> bool:
        return abs(a - b) <= max(abs_tol, rel * max(1.0, abs(a), abs(b)))

    Z = np.full((len(ys), len(xs)), np.nan)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            xr = xv / sx
            yr = yv / sy
            match_val = np.nan
            # Try exact-ish match first
            for (xa, ya, zv) in pts:
                if _close(xa, xr) and _close(ya, yr):
                    match_val = zv
                    break
            if np.isnan(match_val):
                # Fallback: nearest neighbor by sum of squared distances
                best = None
                best_d = float('inf')
                for (xa, ya, zv) in pts:
                    d = (xa - xr)**2 + (ya - yr)**2
                    if d < best_d:
                        best_d = d
                        best = zv
                match_val = best if best is not None else np.nan
            Z[i, j] = match_val
    return Z


def _axis_normalization(name: str) -> Tuple[float, str]:
    key = (name or "").lower()
    if name == "A" or key == "a":
        return 1e4, " (÷1e4)"
    if "fee" in key:
        return 0.0, " (bps)"  # sentinel for bps
    if "liquidity" in key or "balance" in key:
        return 1e18, " (/1e18)"
    if "xcp_profit" in key:
        return 1e18, " (/1e18)"
    return 1.0, ""


def _format_axis_labels(name: str, values: List[float]) -> Tuple[List[str], str]:
    scale, suffix = _axis_normalization(name)
    if scale == 0.0 and "fee" in (name or "").lower():
        labels = [f"{(v / 1e10 * 1e4):.2f}" for v in values]
        return labels, f"{name} (bps)"
    if scale != 1.0:
        labels = [f"{(v / scale):.2f}" for v in values]
        return labels, f"{name}{suffix}"
    return [f"{v:.2f}" for v in values], name


def _edges_from_centers(centers: List[float]) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        x0 = float(c[0]); d = abs(x0) * 0.5 if x0 != 0 else 0.5
        return np.array([x0 - d, x0 + d])
    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (mids[0] - c[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


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
    ap = argparse.ArgumentParser(description="Plot diffs between arb_run grid and reference results.json")
    ap.add_argument("--arb", type=str, default=None, help="Path to arb_run_*.json (default: latest)")
    ap.add_argument("--ref", type=str, default="results.json", help="Path to reference results.json (ref_plot_all schema)")
    ap.add_argument("--relative", action="store_true", help="Plot relative diffs: (new-ref)/abs(ref)")
    ap.add_argument("--cmap", type=str, default="bluepink", help="Matplotlib colormap for diffs (default: bluepink)")
    ap.add_argument("--out", type=str, default=None, help="Output image path")
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    ap.add_argument("--annot", action="store_true", help="Annotate cells with values")
    ap.add_argument("--max-xticks", type=int, default=12, help="Max X tick labels")
    ap.add_argument("--max-yticks", type=int, default=12, help="Max Y tick labels")
    ap.add_argument("--font-size", type=int, default=18, help="Tick label font size")
    args = ap.parse_args()

    arb_path = Path(args.arb) if args.arb else _latest_arb_run()
    ref_path = Path(args.ref)
    data = _load(arb_path)
    ref = _load_ref_results(ref_path)

    # Hardcoded metrics in the same order as heatmap/ref_plot_all
    metrics: List[str] = [
        "xcp_profit",
        "donation_coin0_total",
        "n_rebalances",
        "total_notional_coin0",
        "trades",
        "cex_follow_time_frac",
    ]

    # Build first grid to define axes
    def metric_scale_percent(m: str) -> bool:
        return m.lower() in {"vpminusone", "apy"}

    first_m = metrics[0]
    s_perc = metric_scale_percent(first_m)
    s_1e18 = first_m.lower() in {"xcp_profit"}
    x_name, y_name, xs, ys, Z_new0 = _extract_grid(data, first_m, s_perc, s_1e18)

    # Figure layout
    n = len(metrics)
    # Always use 3 columns; rows = ceil(n/3)
    cols = 3
    rows = int(np.ceil(n / cols))

    base = max(len(xs), len(ys))
    side = max(4.5, min(12.0, 0.35 * max(1, base)))
    fig_w, fig_h = side * cols, side * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    # Build zero-centered diverging colormap
    def _build_bluepink():
        neg = cm.get_cmap('Blues_r', 128)
        pos = cm.get_cmap('Reds', 128)
        colors = np.vstack((neg(np.linspace(0, 1, 128)), pos(np.linspace(0, 1, 128))))
        return ListedColormap(colors, name='bluepink')

    cmap = _build_bluepink() if args.cmap == 'bluepink' else plt.get_cmap(args.cmap)

    # Tick labels
    xticks = _select_ticks(xs, args.max_xticks)
    yticks = _select_ticks(ys, args.max_yticks)
    xlab_full, xlabel = _format_axis_labels(x_name, xs)
    ylab_full, ylabel = _format_axis_labels(y_name, ys)
    xlabels = xlab_full
    ylabels = ylab_full
    xlabels_sel = [xlabels[i] for i in xticks]
    ylabels_sel = [ylabels[i] for i in yticks]

    # Plot per metric
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx >= n:
                ax.axis('off')
                continue
            m = metrics[idx]
            s_perc = metric_scale_percent(m)
            s_1e18 = m.lower() in {"xcp_profit"}
            # New grid uses metric names exactly as in arb_run
            _, _, _, _, Z_new = _extract_grid(data, m, s_perc, s_1e18)
            # Ref grid aligned to xs, ys
            Z_ref = _build_ref_grid(ref, x_name, y_name, xs, ys, m, s_perc)

            # Debug: print values and diffs for each grid point
            print(f"\n=== Metric: {m} ({'percent' if s_perc else 'raw'}) ===")
            for i, yv in enumerate(ys):
                for j, xv in enumerate(xs):
                    new = Z_new[i, j]
                    refv = Z_ref[i, j]
                    if np.isfinite(new) and np.isfinite(refv):
                        absdiff = new - refv
                        denom = abs(refv) if abs(refv) > 0 else np.nan
                        reldiff = absdiff / denom if np.isfinite(denom) else np.nan
                        print(f"x={xv:.6g} y={yv:.6g} new={new:.6g} ref={refv:.6g} abs={absdiff:.6g} rel={reldiff:.6g}")
                    elif np.isfinite(new) and not np.isfinite(refv):
                        print(f"x={xv:.6g} y={yv:.6g} new={new:.6g} ref=NaN (no ref match)")
                    elif not np.isfinite(new) and np.isfinite(refv):
                        print(f"x={xv:.6g} y={yv:.6g} new=NaN (no new match) ref={refv:.6g}")

            # Compute diff
            if args.relative:
                denom = np.where(np.isfinite(Z_ref) & (np.abs(Z_ref) > 0), np.abs(Z_ref), np.nan)
                Z = (Z_new - Z_ref) / denom
            else:
                Z = (Z_new - Z_ref)

            # Plot using edges and pcolormesh
            Xedges = _edges_from_centers(xs)
            Yedges = _edges_from_centers(ys)
            # Zero-centered normalization for precise separation at 0
            finite = Z[np.isfinite(Z)]
            vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            im = ax.pcolormesh(Xedges, Yedges, Z, cmap=cmap, norm=norm, shading='auto')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ny, nx = Z.shape
            try:
                ax.set_box_aspect(ny / nx)
            except Exception:
                ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([xs[i] for i in xticks])
            ax.set_yticks([ys[i] for i in yticks])
            ax.set_xticklabels(xlabels_sel, rotation=45, ha='right', fontsize=args.font_size)
            if c == 0:
                ax.set_yticklabels(ylabels_sel, fontsize=args.font_size)
                ax.set_ylabel(y_name, fontsize=args.font_size + 2)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(xlabel, fontsize=args.font_size + 2)
            title = f"{m} diff" + (" (rel)" if args.relative else "")
            ax.set_title(title, fontsize=args.font_size + 4)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(title, fontsize=args.font_size)
            cb.ax.tick_params(labelsize=args.font_size)
            if args.annot:
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        val = Z[i, j]
                        if not np.isfinite(val):
                            continue
                        ax.text(xs[j], ys[i], f"{val:.3g}", va='center', ha='center', color='white', fontsize=max(6, args.font_size - 2))
            idx += 1

    # Output
    if args.out:
        out_path = Path(args.out)
    else:
        tag = "_".join(m.replace(' ', '') for m in metrics)
        mode = "rel" if args.relative else "abs"
        out_path = RUN_DIR / f"diffs_{mode}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved diffs to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
