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


# ============================================================
# TOP-LEVEL CONTROLS
# ============================================================

RUN_ALL  = True
EXP_NAME = "doublewell_analytical_inference"   # used when RUN_ALL = False

# Experiment name for organizing results
EXPERIMENT_NAME = "test_run_3"  # Change this for different runs (e.g., "main_run", "ablation_1")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = f"results/{EXPERIMENT_NAME}"
PLOTS_DIR   = f"results/{EXPERIMENT_NAME}/plots"

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
    threshold: float = 0.8

# Global instance
DOUBLEWELL_PARAMS = DoubleWellParams()

# Disambiguation time bin targets
DD_TIME_BIN_TARGETS = {
    (0, 40): 250,      # Early disambiguation
    (40, 80): 250,     # Mid disambiguation
    (80, 120): 250,    # Late disambiguation
    (120, 200): 250,    # Very late disambiguation
}


# ============================================================
# Dataset
# ============================================================


def _get_dataset_path() -> str:
    if DATASET_PATH is not None:
        return DATASET_PATH
    return str(Path(RESULTS_DIR) / "dataset")


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


# ============================================================
# Experiment configs
# ============================================================

def _analytical_cfg() -> DoubleWellAnalyticalConfig:
    p = DOUBLEWELL_PARAMS  # Use shared params

    return DoubleWellAnalyticalConfig(
        results_dir=RESULTS_DIR,
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
        inference_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

        # Replay
        horizons=[1, 5, 10, 20, 50],
        n_rollout_samples=20,
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
    "doublewell_analytical_plots":     exp_doublewell_analytical_plots,
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
