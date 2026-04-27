"""
forecasting_plot.py
====================
Standalone script: seed-aggregated DD-aligned forecast plots + pre/post tables.
Produces one figure per metric x horizon, with 1 x n_bins subplots (one per DD bin).
Run directly: python forecasting_plot.py
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Resolve project root
# -----------------------------------------------------------------------------
cwd = Path.cwd().resolve()
project_root = cwd
while not (project_root / "experiments" / "metric_loader.py").exists() and project_root != project_root.parent:
    project_root = project_root.parent

if not (project_root / "experiments" / "metric_loader.py").exists():
    raise RuntimeError(f"Could not locate EviTrack project root from {cwd}")

for path in (project_root, project_root / "experiments"):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from experiments.metric_loader import MetricLoader


# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = project_root / "results" / "MAIN_RUN_04_16_2026" / "doublewell_analytical"
# DATA_PATH = project_root / "results" / "G_ABLATION_04_18_2026" / "doublewell_analytical"
FOLDER_NAME = "MAIN"  # subfolder in paper_figures to save into
SAVE_DIR  = project_root / "paper_figures" / FOLDER_NAME

# =============================================================================
# WHAT TO RUN
# =============================================================================
FIGURE_NAME = "main"  # e.g. "main", "scoring_ablation", "random_beam"

# Any subset of "pll", "mse", "ba"
METRICS = ["pll", "mse", "ba"]

# Horizon index: int, list of ints, or None -> all
HORIZON_IDX = 2

# DD bins to show as subplots — None -> all bins from metadata
# e.g. ["30-80", "80-140", "140-170"]
DD_BIN_LABELS = None

# DD-aligned window half-width (plots span -WINDOW to +WINDOW relative to DD)
WINDOW = 20

# =============================================================================
# ENGINE CONFIG
# =============================================================================
# Which engines to plot and in what legend order.
# None -> all available, ordered by ENGINE_ORDER_PREFERRED.
if FOLDER_NAME == "MAIN" and FIGURE_NAME == "main":
    ENGINE_NAMES = ["EviTrack-J-Ginf", "Bootstrap-PF", "SIS-PF"]
elif FOLDER_NAME == "SCORING_ABLATION" and FIGURE_NAME == "scoring_ablation":
    ENGINE_NAMES = ["EviTrack-J-Ginf", "EviTrack-E-Ginf", "EviTrack-TBD-Ginf"]
elif FOLDER_NAME == "G_ABLATION" and FIGURE_NAME == "g_ablation":
    ENGINE_NAMES = ["EviTrack-J-G1", "EviTrack-J-G5", "EviTrack-J-G10", "EviTrack-J-G20", "EviTrack-J-Ginf"]

# Single source of truth for color / alias / line style across ALL plots.
ENGINE_DISPLAY_CONFIG = {
    # ------------------------
    # Main / Scoring variants
    # ------------------------
    "EviTrack-J-Ginf": {
        "color": "#ff7f0e",
        "alias": "EviTrack-J",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "EviTrack-E-Ginf": {
        "color": "#1f77b4",
        "alias": "EviTrack-E",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "EviTrack-TBD-Ginf": {
        "color": "#22B449",
        "alias": "EviTrack-TBD",
        "linestyle": "-",
        "linewidth": 2.0,
    },

    # ------------------------
    # Adaptive pruning
    # ------------------------
    "EviTrack-E-MaxW": {
        "color": "#A1C7A1",
        "alias": "EviTrack-E-MaxW",
        "linestyle": "-",
        "linewidth": 1.6,
    },
    "EviTrack-J-MaxW": {
        "color": "#d62728",
        "alias": "EviTrack-J-MaxW",
        "linestyle": "-",
        "linewidth": 1.6,
    },

    # ------------------------
    # Baselines
    # ------------------------
    "SIS-PF": {
        "color": "#9467bd",
        "alias": "SIS",
        "linestyle": "-",
        "linewidth": 1.6,
    },
    "Bootstrap-PF": {
        "color": "#8c564b",
        "alias": "BPF",
        "linestyle": "-",
        "linewidth": 1.6,
    },
    "Random-Beam": {
        "color": "#7f7f7f",
        "alias": "Random Beam",
        "linestyle": "--",
        "linewidth": 1.6,
    },

    # ------------------------
    # G ablation (non-clashing)
    # ------------------------
    "EviTrack-J-G1": {
        "color": "#e41a1c",
        "alias": r"$G=1$",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "EviTrack-J-G5": {
        "color": "#377eb8",
        "alias": r"$G=5$",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "EviTrack-J-G10": {
        "color": "#4daf4a",
        "alias": r"$G=10$",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    "EviTrack-J-G20": {
        "color": "#984ea3",
        "alias": r"$G=20$",
        "linestyle": ":",
        "linewidth": 2.0,
    },
}

ENGINE_ORDER_PREFERRED = [
    "EviTrack-J-G1", "EviTrack-J-G5", "EviTrack-J-G10", "EviTrack-J-G20","EviTrack-J-Ginf", "EviTrack-E-Ginf", "EviTrack-TBD-Ginf",
    "EviTrack-E-MaxW", "EviTrack-J-MaxW",
    "SIS-PF", "Bootstrap-PF", "Random-Beam",
]

# =============================================================================
# CONTEXT-DEPENDENT ALIAS OVERRIDES
# =============================================================================

# EviTrack-J-Ginf appears in multiple figures → override alias per context
if FOLDER_NAME == "MAIN" and FIGURE_NAME == "main":
    ENGINE_DISPLAY_CONFIG["EviTrack-J-Ginf"]["alias"] = "EviTrack-J"

elif FOLDER_NAME == "SCORING_ABLATION" and FIGURE_NAME == "scoring_ablation":
    ENGINE_DISPLAY_CONFIG["EviTrack-J-Ginf"]["alias"] = "EviTrack-J"

elif FOLDER_NAME == "G_ABLATION" and FIGURE_NAME == "g_ablation":
    ENGINE_DISPLAY_CONFIG["EviTrack-J-Ginf"]["alias"] = r"$G=\infty$"

# =============================================================================
# METRIC CONFIGS
# =============================================================================
METRIC_CONFIGS = {
    "pll": {"key": "pll",     "name": "Predictive Log-Likelihood", "ylabel": "PLL",            "clip": None, "ylim": None},
    "mse": {"key": "obs_mse", "name": "Observation MSE",           "ylabel": "MSE",            "clip": None, "ylim": None},
    "ba":  {"key": "ba",      "name": "Branch Accuracy",           "ylabel": "Branch Accuracy","clip": None, "ylim": (0.0, 1.0)},
}

# =============================================================================
# PLOT STYLE CONFIG — every visual parameter in one place
# =============================================================================
PLOT_CONFIG = {
    # --- Figure dimensions ---
    "subplot_width":       4.2,    # width per subplot (inches)
    "subplot_height":      3.0,    # height per subplot (inches)
    "subplot_hspace":      0.08,   # vertical space between subplots (unused for 1xN but kept)
    "subplot_wspace":      0.08,   # horizontal space between subplots

    # --- Fonts (NeurIPS uses Times / Computer Modern; "serif" maps to that) ---
    "font_family":         "serif",
    "fontsize":            9,
    "labelsize":           10,
    "ticksize":            8,
    "legendsize":          8,
    "titlesize":           9,
    "suptitlesize":        10,

    # --- Axes / ticks ---
    "axes_linewidth":      0.6,
    "tick_width":          0.6,
    "tick_length":         3.0,
    "tick_direction":      "out",   # "in" or "out"

    # --- DD vertical line ---
    "dd_line_color":       "#333333",
    "dd_line_style":       "--",
    "dd_line_width":       0.9,
    "dd_line_alpha":       0.7,

    # --- Uncertainty band ---
    "band_alpha":          0.12,

    # --- Grid ---
    "show_grid":           True,
    "grid_alpha":          0.25,
    "grid_linewidth":      0.5,
    "grid_linestyle":      ":",

    # --- Legend ---
    "show_legend":         True,
    "legend_ncol":         4,            # columns in the shared legend
    "legend_loc":          "lower center",
    "legend_bbox":         (0.5, -0.02), # anchor relative to figure
    "legend_framealpha":   0.9,
    "legend_edgecolor":    "#cccccc",
    "legend_handlelength": 1.8,
    "legend_handletextpad":0.5,
    "legend_columnspacing":1.0,

    # --- Titles / labels ---
    "show_suptitle":       False,   # figure-level title (metric + horizon)
    "show_bin_title":      True,    # subplot title (DD bin label)
    "show_xlabel":         True,    # x-label on all subplots
    "show_ylabel":         True,    # y-label on leftmost subplot only
    "show_grid":           True,

    # --- Spine visibility ---
    "show_top_spine":      False,
    "show_right_spine":    False,

    # --- Output ---
    "tight_layout_pad":    0.5,
    "legend_figure_y":     -0.09,    # figure-level legend y position (fraction)
}

# =============================================================================
# CONTEXT-DEPENDENT PLOT OVERRIDES
# =============================================================================

if FOLDER_NAME == "G_ABLATION" and FIGURE_NAME == "g_ablation":
    PLOT_CONFIG["legend_ncol"] = len(ENGINE_NAMES)  # one row: G=1,5,10,20,inf
    PLOT_CONFIG["legend_figure_y"] = -0.08
    PLOT_CONFIG["bottom_adjust"] = 0.20


# =============================================================================
# HELPERS
# =============================================================================

def ensure_list_or_none(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, set, np.ndarray)):
        x = list(x)
        return None if len(x) == 0 else x
    return [x]


def sort_bin_labels(labels: List[str]) -> List[str]:
    return sorted(labels, key=lambda x: int(str(x).split("-")[0]))


def preferred_engine_order(available, preferred):
    available = list(available)
    ordered = [e for e in preferred if e in available]
    ordered += [e for e in available if e not in ordered]
    return ordered


def select_available_indices(requested, available, name):
    requested = ensure_list_or_none(requested)
    available = list(available)
    if requested is None:
        return available
    missing = [x for x in requested if x not in available]
    if missing:
        raise ValueError(f"Requested {name} {missing} not in available: {available}")
    return requested


def get_engine_style(engine: str) -> dict:
    cfg = ENGINE_DISPLAY_CONFIG.get(engine, {})
    return {
        "color":     cfg.get("color", "#333333"),
        "linestyle": cfg.get("linestyle", "-"),
        "linewidth": cfg.get("linewidth", 1.6),
    }


def get_engine_alias(engine: str) -> str:
    return ENGINE_DISPLAY_CONFIG.get(engine, {}).get("alias", engine)


def apply_clip(y, clip):
    if clip is None:
        return y
    lo, hi = clip
    return np.clip(y, -np.inf if lo is None else lo, np.inf if hi is None else hi)


def load_dd_bin_ranges(dataset_dir: Path) -> Dict[str, Tuple[int, int]]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    bin_ranges = {}
    for label in metadata["bin_start_indices"].keys():
        lo, hi = str(label).split("-")
        bin_ranges[str(label)] = (int(lo), int(hi))
    return bin_ranges


def stratify_by_dd_time(
    data_by_traj_seed, engines, traj_indices, seed_indices, bin_ranges
) -> Dict[str, List[int]]:
    ref_engine = engines[0]
    ref_seed   = seed_indices[0]
    bin_to_trajs = {label: [] for label in bin_ranges}
    for traj_idx in traj_indices:
        dd_time = data_by_traj_seed[ref_engine][traj_idx][ref_seed]["dd_time_truth"]
        for label, (lo, hi) in bin_ranges.items():
            if lo <= dd_time < hi:
                bin_to_trajs[label].append(traj_idx)
                break
    return bin_to_trajs


def apply_neurips_style(ax, pc: dict, is_leftmost: bool = False):
    """Apply NeurIPS-quality spine / tick styling to an axis."""
    ax.spines["top"].set_visible(pc["show_top_spine"])
    ax.spines["right"].set_visible(pc["show_right_spine"])
    for spine in ax.spines.values():
        spine.set_linewidth(pc["axes_linewidth"])

    ax.tick_params(
        which="both",
        width=pc["tick_width"],
        length=pc["tick_length"],
        direction=pc["tick_direction"],
        labelsize=pc["ticksize"],
    )

    if pc["show_grid"]:
        ax.set_axisbelow(True)
        ax.grid(
            True,
            alpha=pc["grid_alpha"],
            linewidth=pc["grid_linewidth"],
            linestyle=pc["grid_linestyle"],
            color="#aaaaaa",
        )


# =============================================================================
# PRE/POST TABLE
# =============================================================================

def compute_pre_post_table(
    data_by_traj_seed, engines, seed_indices, traj_indices, metric_key
) -> pd.DataFrame:
    rows = []
    for engine in engines:
        pre_seed_means, post_seed_means = [], []
        for seed in seed_indices:
            pre_vals, post_vals = [], []
            for traj_idx in traj_indices:
                metrics = data_by_traj_seed[engine][traj_idx][seed]
                dd_time = int(metrics["dd_time_truth"])
                arr     = np.asarray(metrics[metric_key], dtype=float)
                if dd_time < 0 or dd_time > len(arr):
                    continue
                pre_arr  = arr[:dd_time]
                post_arr = arr[dd_time:]
                if pre_arr.size > 0:
                    pre_vals.append(float(np.nanmean(pre_arr)))
                if post_arr.size > 0:
                    post_vals.append(float(np.nanmean(post_arr)))
            if pre_vals:
                pre_seed_means.append(np.mean(pre_vals))
            if post_vals:
                post_seed_means.append(np.mean(post_vals))

        def fmt(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return "nan ± nan"
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            return f"{m:.3f} ± {s:.3f}"

        rows.append({
            "engine":  engine,
            "pre_DD":  fmt(pre_seed_means),
            "post_DD": fmt(post_seed_means),
            "n_seeds": len(pre_seed_means),
        })

    df = pd.DataFrame(rows)
    order_map = {e: i for i, e in enumerate(engines)}
    df["_order"] = df["engine"].map(order_map)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# =============================================================================
# CURVES
# =============================================================================

def compute_agg_curves(
    data_by_traj_seed, engines, seed_indices, traj_indices, metric_key, window
) -> Dict[str, dict]:
    agg = {}
    for engine in engines:
        seed_centers = []
        for seed in seed_indices:
            aligned = []
            for traj_idx in traj_indices:
                metrics = data_by_traj_seed[engine][traj_idx][seed]
                dd_time = int(metrics["dd_time_truth"])
                arr     = np.asarray(metrics[metric_key], dtype=float)
                start, stop = dd_time - window, dd_time + window + 1
                if start < 0 or stop > len(arr):
                    continue
                aligned.append(arr[start:stop])
            if not aligned:
                continue
            aligned = np.stack(aligned, axis=0)
            seed_centers.append(np.nanmean(aligned, axis=0))

        if not seed_centers:
            continue
        centers = np.stack(seed_centers, axis=0)
        center  = centers.mean(axis=0)
        std     = centers.std(axis=0, ddof=1) if centers.shape[0] > 1 else np.zeros_like(center)
        agg[engine] = {
            "center":  center,
            "lower":   center - std,
            "upper":   center + std,
            "t_rel":   np.arange(-window, window + 1),
            "n_seeds": centers.shape[0],
        }
    return agg


# =============================================================================
# PLOT: 1 x n_bins subplots
# =============================================================================

def plot_bins(
    bin_agg_curves: Dict[str, Dict[str, dict]],
    bin_order: List[str],
    engines: List[str],
    metric_name: str,
    ylabel: str,
    clip,
    ylim,
    H_value: int,
    save_stem: Optional[Path] = None,
):
    pc = PLOT_CONFIG

    # Apply global rcParams
    plt.rcParams.update({
        "font.family":          pc["font_family"],
        "font.size":            pc["fontsize"],
        "axes.labelsize":       pc["labelsize"],
        "axes.titlesize":       pc["titlesize"],
        "xtick.labelsize":      pc["ticksize"],
        "ytick.labelsize":      pc["ticksize"],
        "axes.linewidth":       pc["axes_linewidth"],
        "xtick.major.width":    pc["tick_width"],
        "ytick.major.width":    pc["tick_width"],
        "xtick.major.size":     pc["tick_length"],
        "ytick.major.size":     pc["tick_length"],
        "xtick.direction":      pc["tick_direction"],
        "ytick.direction":      pc["tick_direction"],
        "legend.framealpha":    pc["legend_framealpha"],
        "legend.edgecolor":     pc["legend_edgecolor"],
        "figure.dpi":           150,
        "savefig.dpi":          300,
    })

    n_bins = len(bin_order)

    # Reserve extra bottom space for shared legend
    legend_space = 0.55  # inches at the bottom
    fig_h = pc["subplot_height"] + legend_space / pc["subplot_height"]  # approximate
    fig_w = pc["subplot_width"] * n_bins + pc["subplot_wspace"] * (n_bins - 1)

    fig, axes = plt.subplots(
        1, n_bins,
        figsize=(pc["subplot_width"] * n_bins, pc["subplot_height"]),
        squeeze=False,
    )
    axes = axes[0]  # shape: (n_bins,)

    # Shared y-axis limits: compute global min/max across all bins for auto-ylim
    if ylim is None:
        all_vals = []
        for bin_label in bin_order:
            for engine, data in bin_agg_curves.get(bin_label, {}).items():
                all_vals.extend(apply_clip(data["lower"], clip).tolist())
                all_vals.extend(apply_clip(data["upper"], clip).tolist())
        if all_vals:
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            margin = (vmax - vmin) * 0.08
            computed_ylim = (vmin - margin, vmax + margin)
        else:
            computed_ylim = None
    else:
        computed_ylim = ylim

    for i, bin_label in enumerate(bin_order):
        ax = axes[i]
        agg_curves = bin_agg_curves.get(bin_label, {})
        is_leftmost = (i == 0)

        for engine in engines:
            if engine not in agg_curves:
                continue
            data   = agg_curves[engine]
            t_rel  = data["t_rel"]
            center = apply_clip(data["center"], clip)
            lower  = apply_clip(data["lower"],  clip)
            upper  = apply_clip(data["upper"],  clip)
            style  = get_engine_style(engine)
            alias  = get_engine_alias(engine)

            ax.plot(t_rel, center, label=alias, zorder=3, **style)
            ax.fill_between(
                t_rel, lower, upper,
                alpha=pc["band_alpha"],
                color=style["color"],
                linewidth=0,
                zorder=2,
            )

        # DD line
        ax.axvline(
            0,
            color=pc["dd_line_color"],
            linestyle=pc["dd_line_style"],
            linewidth=pc["dd_line_width"],
            alpha=pc["dd_line_alpha"],
            zorder=4,
        )

        apply_neurips_style(ax, pc, is_leftmost=is_leftmost)

        if computed_ylim is not None:
            ax.set_ylim(computed_ylim)

        ax.set_xlim(-WINDOW, WINDOW)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

        if pc["show_bin_title"]:
            n_trajs = len(next(iter(agg_curves.values()), {}).get("center", [])) if agg_curves else 0
            ax.set_title(f"DD bin: {bin_label}", pad=4)

        if pc["show_xlabel"]:
            ax.set_xlabel("Time relative to $t_{\\mathrm{DD}}$", labelpad=4)

        if pc["show_ylabel"] and is_leftmost:
            ax.set_ylabel(ylabel, labelpad=4)
        elif not is_leftmost:
            # Hide y tick labels on non-leftmost for clean shared look
            ax.set_yticklabels([])

    # Shared figure-level legend below all subplots
    if pc["show_legend"]:
        legend_handles = [
            Line2D(
                [0], [0],
                color=get_engine_style(e)["color"],
                linestyle=get_engine_style(e)["linestyle"],
                linewidth=get_engine_style(e)["linewidth"] * 0.85,
                label=get_engine_alias(e),
            )
            for e in engines
            if any(e in bin_agg_curves.get(b, {}) for b in bin_order)
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, pc["legend_figure_y"]),
            ncol=pc["legend_ncol"],
            fontsize=pc["legendsize"],
            framealpha=pc["legend_framealpha"],
            edgecolor=pc["legend_edgecolor"],
            handlelength=pc["legend_handlelength"],
            handletextpad=pc["legend_handletextpad"],
            columnspacing=pc["legend_columnspacing"],
            borderpad=0.5,
        )

    if pc["show_suptitle"]:
        fig.suptitle(
            f"{metric_name}  (H={H_value})",
            fontsize=pc["suptitlesize"],
            y=1.02,
        )

    # Tight layout but leave room at bottom for legend
    plt.tight_layout(pad=pc["tight_layout_pad"], w_pad=0.4)
    fig.subplots_adjust(bottom=0.22)  # space for legend

    if save_stem is not None:
        save_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_stem) + ".pdf", bbox_inches="tight")
        fig.savefig(str(save_stem) + ".png", bbox_inches="tight", dpi=300)
        print(f"  Saved: {save_stem}.pdf / .png")

    plt.show()
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    results_dir = Path(DATA_PATH)
    replay_dir  = results_dir / "replay"
    dataset_dir = results_dir / "dataset"

    if not replay_dir.exists():
        raise FileNotFoundError(f"Replay dir not found: {replay_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    bin_ranges_all = load_dd_bin_ranges(dataset_dir)

    base_loader = MetricLoader(replay_dir=replay_dir, verbose=True)
    base_loader.load().organize(horizon_idx=HORIZON_IDX)

    engines  = preferred_engine_order(
        select_available_indices(ENGINE_NAMES, base_loader.engines, "engines"),
        ENGINE_ORDER_PREFERRED,
    )
    seeds    = list(base_loader.all_seed_indices)
    horizons = select_available_indices(
        ensure_list_or_none(HORIZON_IDX),
        list(range(len(base_loader.horizons))),
        "horizons",
    )

    for metric_key_name in METRICS:
        if metric_key_name not in METRIC_CONFIGS:
            print(f"Unknown metric '{metric_key_name}', skipping.")
            continue

        mcfg        = METRIC_CONFIGS[metric_key_name]
        metric_key  = mcfg["key"]
        metric_name = mcfg["name"]
        ylabel      = mcfg["ylabel"]
        clip        = mcfg["clip"]
        ylim        = mcfg["ylim"]

        for h_idx in horizons:
            H_value = int(base_loader.horizons[h_idx])

            print(f"\n{'='*80}")
            print(f"METRIC: {metric_name}  |  H={H_value}")
            print(f"{'='*80}")

            loader = MetricLoader(replay_dir=replay_dir, verbose=False)
            loader.load().organize(horizon_idx=h_idx)
            data_by_traj_seed = loader.data_by_traj_seed

            engines_h = [e for e in engines if e in loader.engines]
            seeds_h   = select_available_indices(seeds, loader.all_seed_indices, "seeds")
            trajs_h   = list(loader.all_traj_indices)

            # Stratify into DD bins
            bin_to_trajs_all = stratify_by_dd_time(
                data_by_traj_seed, engines_h, trajs_h, seeds_h, bin_ranges_all
            )

            if DD_BIN_LABELS is not None:
                bin_to_trajs = {k: bin_to_trajs_all[k] for k in DD_BIN_LABELS if k in bin_to_trajs_all}
            else:
                bin_to_trajs = bin_to_trajs_all

            bin_order = sort_bin_labels([b for b in bin_to_trajs if bin_to_trajs[b]])

            print("DD bin counts:")
            for label in bin_order:
                print(f"  {label:>10}: {len(bin_to_trajs[label])} trajectories")

            # --- TABLE: ALL trajs ---
            print(f"\nPre/Post DD Summary — ALL (mean ± std across seeds):")
            print("-" * 60)
            df = compute_pre_post_table(
                data_by_traj_seed, engines_h, seeds_h, trajs_h, metric_key
            )
            print(df.to_string(index=False))

            # --- TABLE: per bin ---
            for label in bin_order:
                trajs_bin = bin_to_trajs[label]
                if not trajs_bin:
                    continue
                print(f"\nPre/Post DD Summary — bin {label} (mean ± std across seeds):")
                print("-" * 60)
                df_bin = compute_pre_post_table(
                    data_by_traj_seed, engines_h, seeds_h, trajs_bin, metric_key
                )
                print(df_bin.to_string(index=False))

            # --- CURVES per bin ---
            bin_agg_curves = {}
            for label in bin_order:
                trajs_bin = bin_to_trajs[label]
                if not trajs_bin:
                    continue
                bin_agg_curves[label] = compute_agg_curves(
                    data_by_traj_seed, engines_h, seeds_h,
                    trajs_bin, metric_key, WINDOW
                )

            # --- PLOT ---
            save_stem = Path(SAVE_DIR) / "forecast_plots" / f"{FIGURE_NAME}_{metric_key_name}_H{H_value}_bins"
            plot_bins(
                bin_agg_curves=bin_agg_curves,
                bin_order=bin_order,
                engines=engines_h,
                metric_name=metric_name,
                ylabel=ylabel,
                clip=clip,
                ylim=ylim,
                H_value=H_value,
                save_stem=save_stem,
            )


if __name__ == "__main__":
    main()