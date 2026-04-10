"""
main.py — EviTrack experiment entry point
==========================================
Reproducibility entry point for all experiments in the paper.

For reviewers: just run this file (F5 in VS Code, or `python main.py`).
All parameters are hardcoded below — no CLI arguments needed.

To run everything:       RUN_ALL = True
To run one stage:        RUN_ALL = False, EXP_NAME = "doublewell_analytical_inference"

Dataset is generated automatically on first run and cached to disk.
Set DATASET_PATH to reuse an existing dataset across runs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass
import torch

from data.dataset_io import save_dataset, load_dataset
from data.synthetic_tasks.doublewell_1d_dataset import (
    build_doublewell_1d_dataset, build_doublewell_1d_dataset_with_bins
    )
from experiments.doublewell_analytical import (
    DoubleWellAnalyticalConfig,
    build_wm,
    run_inference,
    run_replay,
    run_plots,
)

from experiments.global_pruning_ablation import (
    GlobalPruningAblationConfig,
    build_wm as build_wm_pruning,
    run_inference as run_inference_pruning,
    run_replay as run_replay_pruning,
)

from experiments.inference_budget_ablation import (
    InferenceBudgetAblationConfig,
    build_wm as build_wm_budget,
    run_inference as run_inference_budget,
    run_replay as run_replay_budget,
)


# ============================================================
# TOP-LEVEL CONTROLS
# ============================================================

RUN_ALL  = True
EXP_NAME = "doublewell_analytical_inference"   # used when RUN_ALL = False

# Folder name for organizing results
FOLDER_NAME = "delete_3"  # Change this for different runs (e.g., "main_run", "ablation_1")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = f"results/{FOLDER_NAME}"
PLOTS_DIR   = f"results/{FOLDER_NAME}/plots"

# Path to an existing dataset directory (data.pt + metadata.json).
# None = generate and cache to RESULTS_DIR/dataset/
DATASET_PATH = None  # Set to f"results/{EXPERIMENT_NAME}" to reuse dataset

@dataclass
class DoubleWellParams:
    """Single source of truth for double-well task parameters."""
    T: int = 200
    a: float = 3.0
    V: float = 0.06
    dt: float = 1.0
    sigma_z: float = 0.05
    d: float = 2.0
    n: int = 1
    sigma_x: float = 0.12
    z0_mean: float = 0.0
    z0_std: float = 1.0
    threshold: float = 0.9

# Global instance
DOUBLEWELL_PARAMS = DoubleWellParams()

# Disambiguation time bin targets for main experiment
DD_TIME_BIN_TARGETS = {
    (0, 40): 20,      # Early disambiguation
    (40, 80): 20,     # Mid disambiguation
    (80, 120): 20,    # Late disambiguation
    (120, 200): 20,    # Very late disambiguation
}
# Disambiguation time bin targets for pruning ablation
DD_TIME_BIN_TARGETS_PRUNING = {
    (0, 40):    100,
    (40, 80):   100,
    (80, 120):  100,
    (120, 200): 100,
}
# DD time bins for budget ablation
DD_TIME_BIN_TARGETS_BUDGET = {
    (0, 40):    100,
    (40, 80):   100,
    (80, 120):  100,
    (120, 200): 100,
}

# ============================================================
# Dataset
# ============================================================


def _get_dataset_path() -> str:
    if DATASET_PATH is not None:
        return DATASET_PATH
    return str(Path(f"{RESULTS_DIR}/doublewell_analytical/dataset"))


def _load_or_generate_dataset() -> Dict[str, Any]:
    path = Path(_get_dataset_path())
    if (path / "data.pt").exists():
        return load_dataset(path)

    print("[dataset] Generating double-well benchmark dataset with controlled bins...")
    p = DOUBLEWELL_PARAMS

    artifact = build_doublewell_1d_dataset_with_bins(  # CHANGED
        T=p.T,
        dd_time_targets=DD_TIME_BIN_TARGETS,  # NEW
        search_seed_start=0,
        max_seed_search=100_000_000,  # Increased for binned search
        device="cpu",
        dtype=torch.float32,
        a=p.a,
        V=p.V,
        dt=p.dt,
        sigma_z=p.sigma_z,
        d=p.d,
        n=p.n,
        sigma_x=p.sigma_x,
        z0_mean=p.z0_mean,
        z0_std=p.z0_std,
        threshold=p.threshold,
        verbose=True,
    )

    # Convert artifact to plain dict
    dataset = {
        "x":             artifact.x,
        "z":             artifact.z,
        "data_seed_ids": artifact.data_seed_ids,
        "delayed_flag":  artifact.delayed_flag,
        "disamb_time":   artifact.disamb_time,
        "meta":          artifact.meta,
    }
    save_dataset(dataset, path)
    return dataset


def _get_pruning_dataset_path() -> str:
    return str(Path(f"{RESULTS_DIR}/global_pruning_ablation/dataset"))


def _load_or_generate_pruning_dataset() -> Dict[str, Any]:
    path = Path(_get_pruning_dataset_path())
    if (path / "data.pt").exists():
        return load_dataset(path)

    print("[dataset] Generating pruning ablation dataset...")
    p = DOUBLEWELL_PARAMS
    artifact = build_doublewell_1d_dataset_with_bins(
        T=p.T,
        dd_time_targets=DD_TIME_BIN_TARGETS_PRUNING,
        search_seed_start=0,
        max_seed_search=100_000_000,
        device="cpu",
        dtype=torch.float32,
        a=p.a, V=p.V, dt=p.dt, sigma_z=p.sigma_z,
        d=p.d, n=p.n, sigma_x=p.sigma_x,
        z0_mean=p.z0_mean, z0_std=p.z0_std,
        threshold=p.threshold,
        verbose=True,
    )
    dataset = {
        "x":             artifact.x,
        "z":             artifact.z,
        "data_seed_ids": artifact.data_seed_ids,
        "delayed_flag":  artifact.delayed_flag,
        "disamb_time":   artifact.disamb_time,
        "meta":          artifact.meta,
    }
    save_dataset(dataset, path)
    return dataset

def _get_budget_dataset_path() -> str:
    return str(Path(f"{RESULTS_DIR}/inference_budget_ablation/dataset"))

def _load_or_generate_budget_dataset() -> Dict[str, Any]:
    path = Path(_get_budget_dataset_path())
    if (path / "data.pt").exists():
        return load_dataset(path)
    print("[dataset] Generating budget ablation dataset...")
    p = DOUBLEWELL_PARAMS
    artifact = build_doublewell_1d_dataset_with_bins(
        T=p.T,
        dd_time_targets=DD_TIME_BIN_TARGETS_BUDGET,
        search_seed_start=0,
        max_seed_search=100_000_000,
        device="cpu",
        dtype=torch.float32,
        a=p.a, V=p.V, dt=p.dt, sigma_z=p.sigma_z,
        d=p.d, n=p.n, sigma_x=p.sigma_x,
        z0_mean=p.z0_mean, z0_std=p.z0_std,
        threshold=p.threshold,
        verbose=True,
    )
    dataset = {
        "x":             artifact.x,
        "z":             artifact.z,
        "data_seed_ids": artifact.data_seed_ids,
        "delayed_flag":  artifact.delayed_flag,
        "disamb_time":   artifact.disamb_time,
        "meta":          artifact.meta,
    }
    save_dataset(dataset, path)
    return dataset


# ============================================================
# Experiment configs
# ============================================================

def _analytical_cfg() -> DoubleWellAnalyticalConfig:
    p = DOUBLEWELL_PARAMS  # Use shared params

    return DoubleWellAnalyticalConfig(
        results_dir=f"{RESULTS_DIR}/doublewell_analytical",
        plots_dir=PLOTS_DIR,
        device=DEVICE,
        overwrite=False,
        verbose=True,

        # Task — uses shared params
        T=p.T,
        a=p.a,
        V=p.V,
        dt=p.dt,
        sigma_z=p.sigma_z,
        d=p.d,
        n=p.n,
        sigma_x=p.sigma_x,
        z0_mean=p.z0_mean,
        z0_std=p.z0_std,

        # Inference
        inference_seeds=[0,1],

        # Replay
        horizons=[1],
        n_rollout_samples=1,
    )

def _pruning_ablation_cfg() -> GlobalPruningAblationConfig:
    p = DOUBLEWELL_PARAMS
    return GlobalPruningAblationConfig(
        results_dir=f"{RESULTS_DIR}/global_pruning_ablation",
        device=DEVICE,
        overwrite=False,
        verbose=True,
        T=p.T, a=p.a, V=p.V, dt=p.dt, sigma_z=p.sigma_z,
        d=p.d, n=p.n, sigma_x=p.sigma_x,
        z0_mean=p.z0_mean, z0_std=p.z0_std,
        G_values=[1, 5, 10, 20],
        K=5,
        C=3,
        inference_seeds=[0, 1, 2],
        horizons=[1, 5, 10, 20, 50],
        n_rollout_samples=20,
    )

def _budget_ablation_cfg() -> InferenceBudgetAblationConfig:
    p = DOUBLEWELL_PARAMS
    return InferenceBudgetAblationConfig(
        results_dir=f"{RESULTS_DIR}/inference_budget_ablation",
        device=DEVICE,
        overwrite=False,
        verbose=True,
        T=p.T, a=p.a, V=p.V, dt=p.dt, sigma_z=p.sigma_z,
        d=p.d, n=p.n, sigma_x=p.sigma_x,
        z0_mean=p.z0_mean, z0_std=p.z0_std,
        budget_sweeps=[(3,2), (5,3), (10,3), (15,3)],
        G=1,
        inference_seeds=[0, 1, 2],                  # [0, 1, 2]
        horizons=[1, 5, 10, 20, 50],                      # [1, 5, 10, 20, 50]
        n_rollout_samples=20,                  # 20
    )

# ============================================================
# Registry
# ============================================================

def exp_doublewell_analytical_inference() -> None:
    """Run inference with ground-truth world model. Paper Section 5.1."""
    dataset = _load_or_generate_dataset()
    cfg = _analytical_cfg()
    wm  = build_wm(cfg)
    run_inference(cfg, dataset, wm)


def exp_doublewell_analytical_replay() -> None:
    """Metric replay on analytical inference states. Requires inference first."""
    dataset = _load_or_generate_dataset()
    cfg = _analytical_cfg()
    wm  = build_wm(cfg)
    run_replay(cfg, dataset, wm)

def exp_pruning_ablation_inference() -> None:
    dataset = _load_or_generate_pruning_dataset()
    cfg = _pruning_ablation_cfg()
    wm = build_wm_pruning(cfg)
    run_inference_pruning(cfg, dataset, wm)

def exp_pruning_ablation_replay() -> None:
    dataset = _load_or_generate_pruning_dataset()
    cfg = _pruning_ablation_cfg()
    wm = build_wm_pruning(cfg)
    run_replay_pruning(cfg, dataset, wm)

def exp_budget_ablation_inference() -> None:
    dataset = _load_or_generate_budget_dataset()
    cfg = _budget_ablation_cfg()
    wm = build_wm_budget(cfg)
    run_inference_budget(cfg, dataset, wm)

def exp_budget_ablation_replay() -> None:
    dataset = _load_or_generate_budget_dataset()
    cfg = _budget_ablation_cfg()
    wm = build_wm_budget(cfg)
    run_replay_budget(cfg, dataset, wm)


def exp_doublewell_analytical_plots() -> None:
    """Generate plots from analytical replay results. Requires replay first."""
    cfg = _analytical_cfg()
    run_plots(cfg)


# def exp_doublewell_trained_inference() -> None:
#     """Train world model + proposal, then run inference. Paper Section 5.2."""
#     dataset = _load_or_generate_dataset()
#     cfg = _trained_cfg()
#     run_trained_inference(cfg, dataset)


# def exp_doublewell_trained_replay() -> None:
#     """Metric replay on trained inference states. Requires inference first."""
#     dataset = _load_or_generate_dataset()
#     cfg = _trained_cfg()
#     run_trained_replay(cfg, dataset)


# def exp_doublewell_trained_plots() -> None:
#     """Generate plots from trained replay results. Requires replay first."""
#     cfg = _trained_cfg()
#     run_trained_plots(cfg)


REGISTRY: Dict[str, Any] = {
    "doublewell_analytical_inference": exp_doublewell_analytical_inference,
    "doublewell_analytical_replay":    exp_doublewell_analytical_replay,
    # "pruning_ablation_inference": exp_pruning_ablation_inference,
    # "pruning_ablation_replay":    exp_pruning_ablation_replay,
    # "budget_ablation_inference": exp_budget_ablation_inference,
    # "budget_ablation_replay":    exp_budget_ablation_replay,
    # "doublewell_analytical_plots":     exp_doublewell_analytical_plots,
    # "doublewell_trained_inference":  exp_doublewell_trained_inference,
    # "doublewell_trained_replay":     exp_doublewell_trained_replay,
    # "doublewell_trained_plots":      exp_doublewell_trained_plots,
}


# ============================================================
# Runner
# ============================================================

def run_experiments() -> None:
    if RUN_ALL:
        exps_to_run = list(REGISTRY.keys())
    else:
        if EXP_NAME not in REGISTRY:
            print(f"Unknown experiment: '{EXP_NAME}'")
            print("Available:\n" + "\n".join(f"  {k}" for k in REGISTRY))
            sys.exit(1)
        exps_to_run = [EXP_NAME]

    print(f"\nEviTrack | {len(exps_to_run)} stage(s) | device={DEVICE}\n")

    for name in exps_to_run:
        print(f"{'='*60}\n  {name}\n{'='*60}")
        REGISTRY[name]()
        print(f"  Done: {name}\n")

    print("All stages complete.")


if __name__ == "__main__":
    run_experiments()
