# experiments/doublewell_analytical.py
"""
Double-well 1D analytical experiment.

Uses the ground-truth world model (no training).
Compares EviTrack (evidence / joint / tbd) vs Bootstrap PF vs Random Beam.

Three stages:
    run_inference(cfg, dataset) — run all engines, save .npz states
    run_replay(cfg, dataset)    — compute PLL, MSE, branch accuracy
    run_plots(cfg)              — generate and save figures

Dataset is a plain dict:
    dataset["x"]             : Tensor [N, T, dx]
    dataset["z"]             : Tensor [N, T, dz]
    dataset["data_seed_ids"] : Tensor [N]
    dataset["delayed_flag"]  : Tensor [N] bool
    dataset["disamb_time"]   : Tensor [N] int64
    dataset["meta"]          : dict
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np

from world_model import WorldModelConfig
from world_model.analytical import AnalyticalWorldModel
from data.dataset_io import save_dataset
from data.synthetic_tasks.doublewell_1d import make_prior, make_transition, make_emission
from experiments.inference_eval import run_and_save_inference_states
from experiments.metric_replay import ReplayConfig, run_metric_replay, load_replay_results
from utils.plots import (
    plot_pll_vs_horizon,
    plot_branch_accuracy_vs_time,
    plot_mse_vs_horizon,
    plot_disambiguation_time_histogram,
)


# ============================================================
# Config — everything in one place
# ============================================================

@dataclass
class DoubleWellAnalyticalConfig:

    # --- Output ---
    results_dir: str = "results/doublewell_analytical"
    plots_dir:   str = "results/plots/doublewell_analytical"
    device:      str = "cpu"
    overwrite:   bool = False
    verbose:     bool = True

    # --- Task parameters ---
    T:       int   = 120
    a:       float = 3.0
    V:       float = 0.06
    dt:      float = 1.0
    sigma_z: float = 0.05
    d:       float = 2.0
    n:       int   = 1
    sigma_x: float = 0.12
    z0_mean: float = 0.0
    z0_std:  float = 1.0

    # --- Inference ---
    # K:               int       = 5
    # C:               int       = 3
    # G:               int       = 1
    # N_pf:            int       = 5     # Bootstrap PF particles
    inference_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # --- Replay ---
    horizons:          List[int] = field(default_factory=lambda: [5, 10, 20, 30, 50])
    n_rollout_samples: int       = 50


# ============================================================
# World model
# ============================================================

def build_wm(cfg: DoubleWellAnalyticalConfig) -> AnalyticalWorldModel:
    """Build the ground-truth analytical world model."""
    prior      = make_prior(z0_mean=cfg.z0_mean, z0_std=cfg.z0_std)
    transition = make_transition(a=cfg.a, V=cfg.V, dt=cfg.dt, sigma_z=cfg.sigma_z)
    emission   = make_emission(d=cfg.d, n=cfg.n, sigma_x=cfg.sigma_x)
    wm = AnalyticalWorldModel(
        cfg=WorldModelConfig(dz=1, dx=1),
        prior_mu0=prior.mu0,
        prior_cov0=prior.cov0,
        trans_mean=transition.mean_fn,
        trans_cov=transition.cov_fn,
        emit_mean=emission.mean_fn,
        emit_cov=emission.cov_fn,
    )
    return wm.to(device=cfg.device, dtype=torch.float32).eval()


# ============================================================
# Stage 1: Inference
# ============================================================

def _inference_sweeps(cfg: DoubleWellAnalyticalConfig) -> Dict[str, Dict[str, Any]]:
    # return {
    #     "EviTrack-E-ESS": dict(
    #         engine="evitrack", K=32, C=2,
    #         expand="transition", prune_score="evidence", weight_mode="evidence",
    #         use_ess_trigger=True, ess_threshold_frac=0.5),
    #     "EviTrack-J-ESS": dict(
    #         engine="evitrack", K=32, C=2,
    #         expand="transition", prune_score="joint", weight_mode="joint",
    #         use_ess_trigger=True, ess_threshold_frac=0.5),
    #     "EviTrack-TBD-ESS": dict(
    #         engine="evitrack", K=32, C=2,
    #         expand="transition", prune_score="tbd_joint", weight_mode="tbd_joint",
    #         use_ess_trigger=True, ess_threshold_frac=0.5),
    #     "Bootstrap-PF": dict(engine="bootstrap_pf",  N=64, resample_every_step=False, ess_threshold_frac=0.5),
    #     "SIS-PF": dict(engine="bootstrap_pf",  N=64, resample_every_step=False, ess_threshold_frac=0.0),
    #     "Random-Beam":  dict(engine="random_beam",   K=32, C=2, G=1,
    #                          expand="transition", weight_mode="joint"),
    # }
    return {
    "EviTrack-J-Ginf": dict(
        engine="evitrack",
        K=32, C=2, G=10**9,
        expand="transition",
        prune_score="joint",
        weight_mode="joint",
        global_trigger_mode="constant",
        global_trigger_source="parents",
    ),

    "EviTrack-J-MaxW": dict(
        engine="evitrack",
        K=32, C=2,
        expand="transition",
        prune_score="joint",
        weight_mode="joint",
        global_trigger_mode="max",
        global_trigger_source="parents",
        max_weight_threshold=0.90,
    ),

    "EviTrack-J-Entropy": dict(
        engine="evitrack",
        K=32, C=2,
        expand="transition",
        prune_score="joint",
        weight_mode="joint",
        global_trigger_mode="entropy",
        global_trigger_source="parents",
        entropy_threshold=0.20,
        normalize_entropy=True,
    ),

    # "Bootstrap-PF": dict(
    #     engine="bootstrap_pf",
    #     N=64,
    #     resample_every_step=False,
    #     ess_threshold_frac=0.5
    # ),

    # "Random-Beam": dict(
    #     engine="random_beam",
    #     K=32, C=2, G=1,
    #     expand="transition",
    #     weight_mode="joint"
    # ),
}


def run_inference(
    cfg: DoubleWellAnalyticalConfig,
    dataset: Dict[str, Any],
    wm: AnalyticalWorldModel,
) -> None:
    """
    Run all engines on every trajectory for all inference seeds.
    Saves .npz files to:
        cfg.results_dir/<engine_name>/<k_tag>/inference_seed_<s>/traj_XXXX.npz
    """
    # Wrap dataset in a lightweight object so inference_eval can index it
    artifact = _DatasetView(dataset)
    sweeps = _inference_sweeps(cfg)

    for engine_name, engine_cfg in sweeps.items():
        # k_tag = _cfg_to_tag(engine_cfg)
        for inf_seed in cfg.inference_seeds:
            out_dir = (
                Path(cfg.results_dir)
                / engine_name
                / f"inference_seed_{inf_seed:03d}"
            )
            if cfg.verbose:
                print(f"[inference] {engine_name} | seed={inf_seed}")

            run_and_save_inference_states(
                wm=wm,
                proposal=None,
                artifact=artifact,
                engine_name=engine_name,
                engine_cfg=engine_cfg,
                out_dir=out_dir,
                inference_seed=inf_seed,
                device=torch.device(cfg.device),
                dtype=torch.float32,
                overwrite=cfg.overwrite,
                verbose=cfg.verbose,
            )

    # Save experiment metadata once
    meta_path = Path(cfg.results_dir) / "experiment_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump({
            "experiment": "doublewell_analytical",
            "task": {k: getattr(cfg, k) for k in
                     ("T","a","V","dt","sigma_z","d","n","sigma_x","z0_mean","z0_std")},
            "inference": {k: getattr(cfg, k) for k in
                          ("inference_seeds",)},
            "engines": list(sweeps.keys()),
            "N": dataset["x"].shape[0],
        }, f, indent=2)

    print(f"[inference] Done. Results in {cfg.results_dir}")


# ============================================================
# Stage 2: Replay
# ============================================================

def run_replay(
    cfg: DoubleWellAnalyticalConfig,
    dataset: Dict[str, Any],
    wm: AnalyticalWorldModel,
    *,
    ablations: bool = False,
) -> None:
    """
    Run metric replay on saved inference states.
    Set ablations=True to also replay the ablation results.
    """
    results_dir = Path(cfg.results_dir)
    dataset_path = str(results_dir / "dataset")

    # Save dataset to canonical path so replay can find it
    if not (Path(dataset_path) / "data.pt").exists():
        save_dataset(dataset, dataset_path)

    engines = list(_inference_sweeps(cfg).keys())
    if ablations:
        # Also discover ablation engine dirs
        ablation_dir = results_dir / "ablations"
        if ablation_dir.exists():
            engines += [str(d.relative_to(results_dir))
                        for d in ablation_dir.iterdir() if d.is_dir()]

    replay_cfg = ReplayConfig(
        results_dir=str(results_dir),
        dataset_path=dataset_path,
        engines=engines,
        horizons=cfg.horizons,
        n_rollout_samples=cfg.n_rollout_samples,
        device=cfg.device,
        dtype=torch.float32,
        save_dir=str(results_dir / "replay"),
        verbose=cfg.verbose,
    )
    run_metric_replay(replay_cfg, wm)
    print(f"[replay] Done. Results in {results_dir}/replay/")


# ============================================================
# Stage 3: Plots
# ============================================================

def run_plots(cfg: DoubleWellAnalyticalConfig) -> None:
    """Generate all standard plots from replay results."""
    replay_dir = Path(cfg.results_dir) / "replay"
    out_dir = Path(cfg.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_by_engine: Dict[str, Any] = {}
    for npz_path in sorted(replay_dir.glob("*.npz")):
        result = load_replay_results(npz_path)
        engine_name = str(result.get("engine_name", npz_path.stem))
        results_by_engine[engine_name] = result

    if not results_by_engine:
        print(f"[plots] No replay results found in {replay_dir}, skipping.")
        return

    print(f"[plots] Generating plots → {out_dir}")

    plot_pll_vs_horizon(
        results_by_engine, horizon_idx=None,
        split_by_delayed=True,
        save_path=out_dir / "pll_vs_horizon.pdf",
    )
    plot_mse_vs_horizon(
        results_by_engine,
        split_by_delayed=True,
        save_path=out_dir / "mse_vs_horizon.pdf",
    )
    plot_branch_accuracy_vs_time(
        results_by_engine,
        split_by_delayed=True,
        save_path=out_dir / "branch_accuracy_vs_time.pdf",
    )
    plot_disambiguation_time_histogram(
        results_by_engine,
        save_path=out_dir / "disamb_time_histogram.pdf",
    )

    print(f"[plots] Done.")


# ============================================================
# Helpers
# ============================================================

class _DatasetView:
    """
    Minimal wrapper so a plain dict works with run_and_save_inference_states,
    which expects artifact.x, artifact.delayed_flag, artifact.disamb_time.
    """
    def __init__(self, dataset: Dict[str, Any]):
        self.x             = dataset["x"]
        self.z             = dataset["z"]
        self.delayed_flag  = dataset["delayed_flag"]
        self.disamb_time   = dataset["disamb_time"]
        self.data_seed_ids = dataset["data_seed_ids"]
        self.meta          = dataset["meta"]


def _cfg_to_tag(cfg: Dict[str, Any]) -> str:
    parts = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            parts.append(f"{k}{int(v)}")
        elif isinstance(v, float) and v == int(v):
            parts.append(f"{k}{int(v)}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}{v}")
    return "_".join(parts) if parts else "default"
