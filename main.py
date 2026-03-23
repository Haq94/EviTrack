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

import torch

from data.dataset_io import save_dataset, load_dataset
from data.synthetic_tasks.doublewell_1d_dataset import build_doublewell_1d_dataset
from experiments.doublewell_analytical import (
    DoubleWellAnalyticalConfig,
    build_wm,
    run_inference,
    run_replay,
    run_plots,
)
# from experiments.doublewell_trained import (
#     DoubleWellTrainedConfig,
#     run_inference  as run_trained_inference,
#     run_replay     as run_trained_replay,
#     run_plots      as run_trained_plots,
# )


# ============================================================
# TOP-LEVEL CONTROLS  <-- reviewers edit only this section
# ============================================================

RUN_ALL  = True
EXP_NAME = "doublewell_analytical_inference"   # used when RUN_ALL = False

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"

# Path to an existing dataset directory (data.pt + metadata.json).
# None = generate and cache to RESULTS_DIR/doublewell_analytical/dataset/
DATASET_PATH = None


# ============================================================
# Dataset
# ============================================================

def _get_dataset_path() -> str:
    if DATASET_PATH is not None:
        return DATASET_PATH
    return str(Path(RESULTS_DIR) / "doublewell_analytical" / "dataset")


def _load_or_generate_dataset() -> Dict[str, Any]:
    path = Path(_get_dataset_path())
    if (path / "data.pt").exists():
        return load_dataset(path)

    print("[dataset] Generating double-well benchmark dataset ...")
    artifact = build_doublewell_1d_dataset(
        T=120,
        n_delayed=500,
        n_non_delayed=500,
        search_seed_start=0,
        max_seed_search=100_000,
        device="cpu",
        dtype=torch.float32,
        a=3.0, V=0.06, dt=1.0, sigma_z=0.05,
        d=2.0, n=1, sigma_x=0.12,
        z0_mean=0.0, z0_std=1.0,
        threshold=0.8,
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
    return DoubleWellAnalyticalConfig(
        results_dir=str(Path(RESULTS_DIR) / "doublewell_analytical"),
        plots_dir=str(Path(PLOTS_DIR) / "doublewell_analytical"),
        device=DEVICE,
        overwrite=False,
        verbose=True,

        # Task — must match dataset generation above
        T=120,
        a=3.0, V=0.06, dt=1.0, sigma_z=0.05,
        d=2.0, n=1, sigma_x=0.12,
        z0_mean=0.0, z0_std=1.0,

        # Inference
        inference_seeds=[0, 1, 2, 3 , 4],

        # Replay
        horizons=[1, 5, 10, 20, 30],
        n_rollout_samples=50,
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
