# experiments/global_pruning_ablation.py
"""
Global pruning frequency ablation for EviTrack.

Sweeps G (global prune interval) across EviTrack variants (Evidence, Joint, TBD).
Everything else held fixed: K=5, C=3, same double-well params as main experiment.

Two stages:
    run_inference(cfg, dataset, wm)  — run all engines, save .npz states
    run_replay(cfg, dataset, wm)     — compute metrics via metric replay

No run_plots() — all plotting done in Jupyter.

Engine naming: EviTrack-E-G1, EviTrack-E-G5, EviTrack-J-G1, etc.
Results saved to: cfg.results_dir/<engine_name>/inference_seed_<s>/traj_XXXX.npz
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch

from world_model import WorldModelConfig
from world_model.analytical import AnalyticalWorldModel
from data.dataset_io import save_dataset
from data.synthetic_tasks.doublewell_1d import make_prior, make_transition, make_emission
from experiments.inference_eval import run_and_save_inference_states
from experiments.metric_replay import ReplayConfig, run_metric_replay


# ============================================================
# Config
# ============================================================

@dataclass
class GlobalPruningAblationConfig:

    # --- Output ---
    results_dir: str  = "results/global_pruning_ablation"
    device:      str  = "cpu"
    overwrite:   bool = False
    verbose:     bool = True

    # --- Task parameters ---
    T:       int   = 200
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
    inference_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # --- Sweep parameters (edit these to change the ablation) ---
    G_values:  List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    K:         int       = 5
    C:         int       = 3

    # --- Replay ---
    horizons:          List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    n_rollout_samples: int       = 20


# ============================================================
# World model
# ============================================================

def build_wm(cfg: GlobalPruningAblationConfig) -> AnalyticalWorldModel:
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
# Sweep definition
# ============================================================

def _inference_sweeps(cfg: GlobalPruningAblationConfig) -> Dict[str, Dict[str, Any]]:
    """
    Build engine sweep: 3 variants x len(G_values) engines.
    Edit cfg.G_values or cfg.K/C to change the sweep.
    """
    variants = [
        ("E",   "evidence",  "evidence"),
        ("J",   "joint",     "joint"),
        ("TBD", "tbd_joint", "tbd_joint"),
    ]
    sweeps = {}
    for G in cfg.G_values:
        for short, prune_score, weight_mode in variants:
            name = f"EviTrack-{short}-G{G}"
            sweeps[name] = dict(
                engine="evitrack",
                K=cfg.K,
                C=cfg.C,
                G=G,
                expand="transition",
                prune_score=prune_score,
                weight_mode=weight_mode,
            )
    return sweeps


# ============================================================
# Stage 1: Inference
# ============================================================

def run_inference(
    cfg: GlobalPruningAblationConfig,
    dataset: Dict[str, Any],
    wm: AnalyticalWorldModel,
) -> None:
    artifact = _DatasetView(dataset)
    sweeps   = _inference_sweeps(cfg)

    for engine_name, engine_cfg in sweeps.items():
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

    # Save experiment metadata
    meta_path = Path(cfg.results_dir) / "experiment_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump({
            "experiment": "global_pruning_ablation",
            "task": {k: getattr(cfg, k) for k in
                     ("T","a","V","dt","sigma_z","d","n","sigma_x","z0_mean","z0_std")},
            "inference": {k: getattr(cfg, k) for k in
                          ("inference_seeds", "G_values", "K", "C")},
            "engines": list(sweeps.keys()),
            "N": dataset["x"].shape[0],
        }, f, indent=2)

    print(f"[inference] Done. Results in {cfg.results_dir}")


# ============================================================
# Stage 2: Replay
# ============================================================

def run_replay(
    cfg: GlobalPruningAblationConfig,
    dataset: Dict[str, Any],
    wm: AnalyticalWorldModel,
) -> None:
    results_dir  = Path(cfg.results_dir)
    dataset_path = str(results_dir / "dataset")

    if not (Path(dataset_path) / "data.pt").exists():
        save_dataset(dataset, dataset_path)

    engines = list(_inference_sweeps(cfg).keys())

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
# Helpers
# ============================================================

class _DatasetView:
    def __init__(self, dataset: Dict[str, Any]):
        self.x             = dataset["x"]
        self.z             = dataset["z"]
        self.delayed_flag  = dataset["delayed_flag"]
        self.disamb_time   = dataset["disamb_time"]
        self.data_seed_ids = dataset["data_seed_ids"]
        self.meta          = dataset["meta"]