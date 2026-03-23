# utils/ plots.py
# TODO: V&V needed
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# Global plot style
# -----------------------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.95,
    rc={
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "lines.linewidth": 2.5,
    },
)

_ENGINE_ORDER = [
    "evitrack_evidence",
    "evitrack_joint",
    "evitrack_tbd",
    "bootstrap_pf",
    "random_beam",
]

_ENGINE_LABELS = {
    "evitrack_evidence": "EviTrack (Evidence)",
    "evitrack_joint": "EviTrack (Joint)",
    "evitrack_tbd": "EviTrack (TBD)",
    "bootstrap_pf": "Bootstrap PF",
    "random_beam": "Random Beam",
}

_ENGINE_COLORS = {
    "evitrack_evidence": "#1f77b4",
    "evitrack_joint": "#ff7f0e",
    "evitrack_tbd": "#2ca02c",
    "bootstrap_pf": "#d62728",
    "random_beam": "#9467bd",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ordered_engine_names(results_by_engine: Mapping[str, Any]) -> List[str]:
    present = list(results_by_engine.keys())
    ordered = [k for k in _ENGINE_ORDER if k in present]
    extras = [k for k in present if k not in ordered]
    return ordered + sorted(extras)


def _engine_label(name: str) -> str:
    return _ENGINE_LABELS.get(name, name.replace("_", " ").title())


def _engine_color(name: str, idx: int) -> Any:
    if name in _ENGINE_COLORS:
        return _ENGINE_COLORS[name]
    palette = sns.color_palette("tab10", n_colors=max(idx + 1, 10))
    return palette[idx % len(palette)]


def _ensure_list_of_runs(engine_results: Any) -> List[Mapping[str, Any]]:
    if isinstance(engine_results, Mapping):
        return [engine_results]
    if isinstance(engine_results, (list, tuple)):
        if not engine_results:
            raise ValueError("Engine result list is empty.")
        if not all(isinstance(x, Mapping) for x in engine_results):
            raise TypeError("Each engine run must be a mapping/dict.")
        return list(engine_results)
    raise TypeError(
        "Each entry in results_by_engine must be either a result dict or a list of result dicts."
    )


def _check_run_compatibility(runs: Sequence[Mapping[str, Any]]) -> None:
    ref_horizons = np.asarray(runs[0]["horizons"])
    ref_T = int(runs[0]["T"])
    for run in runs[1:]:
        horizons = np.asarray(run["horizons"])
        T = int(run["T"])
        if ref_T != T:
            raise ValueError(f"All runs for one engine must share T. Got {ref_T} and {T}.")
        if ref_horizons.shape != horizons.shape or not np.all(ref_horizons == horizons):
            raise ValueError("All runs for one engine must share the same horizons array.")


def _merge_engine_runs(engine_name: str, engine_results: Any) -> Dict[str, Any]:
    """
    Accept either:
      - a single replay-result dict, or
      - a list of replay-result dicts corresponding to different inference seeds.

    Returns a merged dict in which trajectories are concatenated across seeds so that
    confidence bands are computed over the combined (seed, trajectory) sample axis.
    """
    runs = _ensure_list_of_runs(engine_results)
    _check_run_compatibility(runs)

    merged: Dict[str, Any] = {
        "engine_name": engine_name,
        "horizons": list(np.asarray(runs[0]["horizons"]).tolist()),
        "T": int(runs[0]["T"]),
    }

    concat_keys = [
        "traj_index",
        "delayed_flag",
        "disamb_time",
        "pll",
        "mse",
        "branch_acc",
    ]

    for key in concat_keys:
        arrays = [np.asarray(run[key]) for run in runs]
        merged[key] = np.concatenate(arrays, axis=0)

    return merged


def _prepare_results(results_by_engine: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    prepared: Dict[str, Dict[str, Any]] = {}
    for engine_name in _ordered_engine_names(results_by_engine):
        prepared[engine_name] = _merge_engine_runs(engine_name, results_by_engine[engine_name])
    return prepared


def _savefig(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def _mask_from_split(result: Mapping[str, Any], delayed_value: Optional[bool]) -> np.ndarray:
    delayed = np.asarray(result["delayed_flag"], dtype=bool)
    if delayed_value is None:
        return np.ones_like(delayed, dtype=bool)
    return delayed == delayed_value


def _trajectory_mean_curve(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    arr:
      - [N, T, H] -> returns [N_selected, H] after averaging over time
      - [N, T]    -> returns [N_selected, T]
    """
    arr = np.asarray(arr)
    sel = arr[mask]
    if sel.size == 0:
        if arr.ndim == 3:
            return np.empty((0, arr.shape[-1]), dtype=float)
        if arr.ndim == 2:
            return np.empty((0, arr.shape[-1]), dtype=float)
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    if sel.ndim == 3:
        return np.nanmean(sel, axis=1)
    if sel.ndim == 2:
        return sel.astype(float)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _mean_and_std(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if curves.shape[0] == 0:
        return np.full((curves.shape[-1],), np.nan), np.full((curves.shape[-1],), np.nan)
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    return mean, std


def _setup_panels(split_by_delayed: bool, figsize_one=(10, 6), figsize_two=(16, 6)):
    if split_by_delayed:
        fig, axes = plt.subplots(1, 2, figsize=figsize_two, sharey=True)
        panels = [(False, axes[0], "Non-delayed trajectories"), (True, axes[1], "Delayed trajectories")]
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize_one)
        panels = [(None, ax, "All trajectories")]
    return fig, panels


def _finalize_axis(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------------
# Public plotting API
# -----------------------------------------------------------------------------

def plot_pll_vs_horizon(
    results_by_engine: Mapping[str, Any],
    horizon_idx: Optional[int],
    *,
    split_by_delayed: bool = True,
    save_path: Optional[str | Path] = None,
):
    """
    Plot mean predictive log-likelihood (PLL) versus forecast horizon.

    Notes
    -----
    `horizon_idx` is retained for API compatibility with earlier replay code, but
    this plot is intrinsically indexed by the full horizon vector and therefore the
    argument is not used.
    """
    _ = horizon_idx  # intentionally unused
    prepared = _prepare_results(results_by_engine)

    fig, panels = _setup_panels(split_by_delayed)

    for panel_value, ax, panel_title in panels:
        for idx, engine_name in enumerate(_ordered_engine_names(prepared)):
            result = prepared[engine_name]
            mask = _mask_from_split(result, panel_value)
            curves = _trajectory_mean_curve(np.asarray(result["pll"], dtype=float), mask)
            mean, std = _mean_and_std(curves)
            horizons = np.asarray(result["horizons"], dtype=float)
            color = _engine_color(engine_name, idx)

            ax.plot(horizons, mean, marker="o", color=color, label=_engine_label(engine_name))
            ax.fill_between(horizons, mean - std, mean + std, color=color, alpha=0.18)

        _finalize_axis(
            ax,
            xlabel="Forecast horizon $H$",
            ylabel="Predictive log-likelihood (PLL)",
            title=panel_title,
        )
        ax.legend(loc="best", fontsize=10)

    fig.suptitle("Predictive log-likelihood vs forecast horizon", y=1.02)
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig



def plot_branch_accuracy_vs_time(
    results_by_engine: Mapping[str, Any],
    horizon_idx: int = 0,
    *,
    split_by_delayed: bool = True,
    save_path: Optional[str | Path] = None,
):
    """
    Plot branch accuracy versus conditioning time with shaded ±1 std bands.

    Parameters
    ----------
    horizon_idx : int
        Which horizon to plot (index into the horizons array). Default is 0 (first horizon).
    """
    prepared = _prepare_results(results_by_engine)
    fig, panels = _setup_panels(split_by_delayed)

    for panel_value, ax, panel_title in panels:
        for idx, engine_name in enumerate(_ordered_engine_names(prepared)):
            result = prepared[engine_name]
            mask = _mask_from_split(result, panel_value)

            # branch_acc has shape [N, T, H]
            branch_acc_full = np.asarray(result["branch_acc"], dtype=float)

            # Select the specified horizon: [N, T, H] -> [N, T]
            branch_acc_at_h = branch_acc_full[:, :, horizon_idx]

            # Select trajectories by mask: [N, T] -> [N_selected, T]
            branch_acc_sel = branch_acc_at_h[mask]

            # Compute mean and std over trajectories: [N_selected, T] -> [T]
            if branch_acc_sel.shape[0] == 0:
                mean = np.full(branch_acc_sel.shape[1], np.nan)
                std = np.full(branch_acc_sel.shape[1], np.nan)
            else:
                mean = np.nanmean(branch_acc_sel, axis=0)
                std = np.nanstd(branch_acc_sel, axis=0)

            T = int(result["T"])
            t_axis = np.arange(T)
            color = _engine_color(engine_name, idx)

            ax.plot(t_axis, mean, color=color, label=_engine_label(engine_name))
            ax.fill_between(t_axis, mean - std, mean + std, color=color, alpha=0.18)

            disamb = np.asarray(result["disamb_time"], dtype=float)[mask]
            disamb = disamb[np.isfinite(disamb)]
            disamb = disamb[disamb >= 0]
            if disamb.size > 0:
                ax.axvline(
                    float(np.mean(disamb)),
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.75,
                )

        horizons = result["horizons"]
        horizon_val = horizons[horizon_idx]
        _finalize_axis(
            ax,
            xlabel="Conditioning time $t$",
            ylabel=f"Branch accuracy (H={horizon_val})",
            title=panel_title,
        )
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=10)

    fig.suptitle(f"Branch accuracy over time (horizon {horizon_val})", y=1.02)
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig



def plot_mse_vs_horizon(
    results_by_engine: Mapping[str, Any],
    *,
    split_by_delayed: bool = True,
    save_path: Optional[str | Path] = None,
):
    """
    Plot mean forecast MSE versus horizon with shaded ±1 std bands.
    """
    prepared = _prepare_results(results_by_engine)
    fig, panels = _setup_panels(split_by_delayed)

    for panel_value, ax, panel_title in panels:
        for idx, engine_name in enumerate(_ordered_engine_names(prepared)):
            result = prepared[engine_name]
            mask = _mask_from_split(result, panel_value)
            curves = _trajectory_mean_curve(np.asarray(result["mse"], dtype=float), mask)
            mean, std = _mean_and_std(curves)
            horizons = np.asarray(result["horizons"], dtype=float)
            color = _engine_color(engine_name, idx)

            ax.plot(horizons, mean, marker="o", color=color, label=_engine_label(engine_name))
            ax.fill_between(horizons, mean - std, mean + std, color=color, alpha=0.18)

        _finalize_axis(
            ax,
            xlabel="Forecast horizon $H$",
            ylabel="Mean squared error (MSE)",
            title=panel_title,
        )
        ax.legend(loc="best", fontsize=10)

    fig.suptitle("Forecast MSE vs horizon", y=1.02)
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig



def plot_disambiguation_time_histogram(
    results_by_engine: Mapping[str, Any],
    *,
    save_path: Optional[str | Path] = None,
):
    """
    Overlay histogram of disambiguation times for all engines.
    """
    prepared = _prepare_results(results_by_engine)

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    all_disamb = []
    Ts = []
    for engine_name in _ordered_engine_names(prepared):
        result = prepared[engine_name]
        disamb = np.asarray(result["disamb_time"], dtype=float)
        disamb = disamb[np.isfinite(disamb)]
        disamb = disamb[disamb >= 0]
        if disamb.size:
            all_disamb.append(disamb)
        Ts.append(int(result["T"]))

    if not all_disamb:
        raise ValueError("No valid disambiguation times found.")

    stacked = np.concatenate(all_disamb)
    T_ref = int(np.median(Ts))
    max_time = max(int(np.nanmax(stacked)), T_ref)
    bins = np.arange(-0.5, max_time + 1.5, 1.0)

    for idx, engine_name in enumerate(_ordered_engine_names(prepared)):
        result = prepared[engine_name]
        disamb = np.asarray(result["disamb_time"], dtype=float)
        disamb = disamb[np.isfinite(disamb)]
        disamb = disamb[disamb >= 0]
        if disamb.size == 0:
            continue

        ax.hist(
            disamb,
            bins=bins,
            density=False,
            alpha=0.35,
            color=_engine_color(engine_name, idx),
            label=_engine_label(engine_name),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.axvline(0.5 * T_ref, color="black", linestyle="--", linewidth=2.0, label=r"$0.5T$")
    _finalize_axis(
        ax,
        xlabel="Disambiguation time",
        ylabel="Trajectory count",
        title="Disambiguation-time distribution",
    )
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig



def plot_oracle_vs_engine_heatmap(
    oracle_pzH: np.ndarray,
    engine_pzH: np.ndarray,
    *,
    z_grid: np.ndarray,
    title: str = "",
    save_path: Optional[str | Path] = None,
):
    """
    Side-by-side heatmaps for oracle and engine predictive densities.

    Expected shapes
    ---------------
    oracle_pzH : [Nz, T_cond] or [T_cond, Nz]
    engine_pzH : [Nz, T_cond] or [T_cond, Nz]
    z_grid     : [Nz]
    """
    oracle = np.asarray(oracle_pzH, dtype=float)
    engine = np.asarray(engine_pzH, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)

    if oracle.shape != engine.shape:
        raise ValueError(f"oracle_pzH and engine_pzH must have the same shape, got {oracle.shape} and {engine.shape}.")

    if oracle.ndim != 2:
        raise ValueError(f"Heatmap inputs must be 2D, got shape {oracle.shape}.")

    # Accept either [Nz, T] or [T, Nz]. Convert to [Nz, T].
    if oracle.shape[0] == z_grid.shape[0]:
        oracle_plot = oracle
        engine_plot = engine
    elif oracle.shape[1] == z_grid.shape[0]:
        oracle_plot = oracle.T
        engine_plot = engine.T
    else:
        raise ValueError(
            "One axis of the heatmap arrays must match len(z_grid). "
            f"Got heatmap shape {oracle.shape} and len(z_grid)={len(z_grid)}."
        )

    T_cond = oracle_plot.shape[1]
    vmin = float(np.nanmin([oracle_plot.min(), engine_plot.min()]))
    vmax = float(np.nanmax([oracle_plot.max(), engine_plot.max()]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    extent = [0, T_cond - 1, float(z_grid.min()), float(z_grid.max())]

    im0 = axes[0].imshow(
        oracle_plot,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="mako",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Oracle quadrature")
    axes[0].set_xlabel("Conditioning time $t$")
    axes[0].set_ylabel(r"Latent value $z$")

    im1 = axes[1].imshow(
        engine_plot,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="mako",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Engine predictive")
    axes[1].set_xlabel("Conditioning time $t$")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(r"$p(z_{t+H} \mid x_{1:t})$")

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


# -----------------------------------------------------------------------------
# Optional convenience loader
# -----------------------------------------------------------------------------

def load_replay_result(path: str | Path) -> Dict[str, Any]:
    """
    Load a replay-result dictionary saved with np.savez(...).

    This helper is optional and intentionally conservative. It expects the keys
    described in the experiment protocol.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        out: Dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if key == "engine_name":
                out[key] = str(value.tolist())
            elif key == "horizons":
                out[key] = list(np.asarray(value).tolist())
            elif key == "T":
                out[key] = int(np.asarray(value).item())
            else:
                out[key] = value
    return out
