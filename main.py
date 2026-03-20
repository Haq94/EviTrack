"""
main.py — EviTrack experiment entry point
==========================================
Reproducibility entry point for all experiments in the paper.

For reviewers: just run this file (F5 in VS Code, or `python main.py`).
All parameters are hardcoded below — no CLI arguments needed.

To isolate one experiment:  set RUN_ALL = False, set EXP_NAME
To reuse existing dataset:  set DATASET_PATH to artifact directory
To regenerate dataset:      set DATASET_PATH = None
"""
from __future__ import annotations

import sys
from typing import Any, Dict

import torch

from world_model import WorldModelConfig
from experiments.analytical_synthetic_experiment import (
    AnalyticalSyntheticExperiment,
    AnalyticalSyntheticConfig,
    DatasetConfig,
)
# from experiments.trained_synthetic_experiment import (
#     TrainedSyntheticExperiment,
#     TrainedSyntheticConfig,
# )


# ============================================================
# TOP-LEVEL CONTROLS  <-- reviewers edit only this section
# ============================================================

RUN_ALL     = True
EXP_NAME    = "doublewell_analytical"   # used when RUN_ALL = False

SEED        = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"

# Path to a pre-generated DoubleWell1DDatasetArtifact directory.
# None = auto-generate and cache inside results/doublewell_analytical/dataset/
# str  = load from that path (e.g. "data/datasets/doublewell_1d/benchmark_v1")
DATASET_PATH = None


# ============================================================
# Experiment definitions
# ============================================================

def exp_doublewell_analytical(seed: int, device: str, results_dir: str) -> Dict[str, Any]:
    """
    Double-well 1D benchmark with the ground-truth analytical world model.
    No training required.

    Compares three EviTrack flavors vs Bootstrap PF vs Random Beam.
    Saves full inference states per trajectory per timestep for offline metric replay.

    Folder structure:
        results/doublewell_analytical/
            experiment_meta.json          <- dataset path + experiment config
            summary.json                  <- index of all runs
            dataset/                      <- cached artifact if DATASET_PATH=None
            evitrack_evidence/K5_C3/
                inference_seed_000/
                    traj_0000.pt          <- List[EviTrackState] length T + traj_index
                    ...
                    run_meta.json
                inference_seed_001/
                inference_seed_002/
            evitrack_joint/K5_C3/
                ...
            evitrack_tbd/K5_C3/
                ...
            bootstrap_pf/N5/
                ...
            random_beam/K5_C3/
                ...

    Paper reference: Section 5.1
    """
    cfg = AnalyticalSyntheticConfig(
        experiment_name="doublewell_analytical_experiment",
        run_root=results_dir,
        seed=seed,
        device=device,
        dtype=torch.float32,

        dataset=DatasetConfig(
            path=DATASET_PATH,
            generate=True,
            T=120,
            n_delayed=3,
            n_non_delayed=4,
            a=3.0, V=0.06, dt=1.0, sigma_z=0.05,
            d=2.0, n=1, sigma_x=0.12,
            z0_mean=0.0, z0_std=1.0,
            threshold=0.8,
        ),

        wm_cfg=WorldModelConfig(dz=1, dx=1),

        # engine_name -> list of cfg dicts
        # prune_score / weight_mode are baked into engine_name in _engine_from_spec
        # so cfg dicts here only need K, C (structural params)
        inference_sweeps={
            "evitrack_evidence": [dict(K=5, C=3, G=2, expand="transition")],
            "evitrack_joint":    [dict(K=5, C=3, G=2, expand="transition")],
            "evitrack_tbd":      [dict(K=5, C=3, G=2, expand="transition")],
            "bootstrap_pf":      [dict(N=5)],
            "random_beam":       [dict(K=5, C=3, G=2, expand="transition")],
        },

        inference_seeds=[0, 1, 2],

        overwrite=False,
        verbose=True,
    )

    exp = AnalyticalSyntheticExperiment(cfg)
    return exp.run()


# def exp_doublewell_trained(seed: int, device: str, results_dir: str) -> Dict[str, Any]:
#     """
#     Double-well 1D with learned neural world models (EviTrack-WM, beta-ELBO).
#     Runs multiple model seeds over the same fixed dataset.
#     Paper reference: Section 5.2
#     """
#     ...

# def exp_bimodal_analytical(seed: int, device: str, results_dir: str) -> Dict[str, Any]:
#     """Bimodal initial condition benchmark. Paper reference: Section 5.1"""
#     ...

# def exp_symmetry_breaking(seed: int, device: str, results_dir: str) -> Dict[str, Any]:
#     """Symmetry-breaking identifiability benchmark. Paper reference: Section 5.1"""
#     ...

# def exp_jena_climate(seed: int, device: str, results_dir: str) -> Dict[str, Any]:
#     """Jena Climate real-data forecasting. Paper reference: Section 5.3"""
#     ...


# ============================================================
# Registry
# ============================================================

REGISTRY: Dict[str, Any] = {
    "doublewell_analytical": exp_doublewell_analytical,
    # "doublewell_trained":    exp_doublewell_trained,
    # "bimodal_analytical":    exp_bimodal_analytical,
    # "symmetry_breaking":     exp_symmetry_breaking,
    # "jena_climate":          exp_jena_climate,
}


# ============================================================
# Runner
# ============================================================

def run_experiments() -> None:
    kwargs = dict(seed=SEED, device=DEVICE, results_dir=RESULTS_DIR)

    if RUN_ALL:
        exps_to_run = list(REGISTRY.keys())
    else:
        if EXP_NAME not in REGISTRY:
            print(f"Unknown experiment: '{EXP_NAME}'")
            print(f"Available: {list(REGISTRY.keys())}")
            sys.exit(1)
        exps_to_run = [EXP_NAME]

    print(f"\nEviTrack | Running {len(exps_to_run)} experiment(s)")
    print(f"  seed={SEED}  device={DEVICE}  results_dir={RESULTS_DIR}\n")

    for name in exps_to_run:
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        REGISTRY[name](**kwargs)
        print(f"  Done: {name}\n")

    print("All experiments complete.")


if __name__ == "__main__":
    run_experiments()
