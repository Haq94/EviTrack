# main.py
from __future__ import annotations

import torch

from experiments.analytical_synthetic_experiment import (
    AnalyticalSyntheticConfig,
    AnalyticalSyntheticExperiment,
)
from experiments.trained_synthetic_experiment import (
    TrainedSyntheticConfig,
    TrainedSyntheticExperiment,
)

from training.runner import RunConfig, DataConfig
from world_model import WorldModelConfig
from proposal import ProposalConfig


def build_experiments():
    experiments = []

    # -------------------------------------------------
    # 1) Analytical synthetic experiment
    # -------------------------------------------------
    analytical_cfg = AnalyticalSyntheticConfig(
        experiment_name="analytic_doublewell",
        run_root="results",
        seed=0,
        device="cpu",
        dtype=torch.float32,
        T=120,
        n_val=256,
        batch_size=64,
        wm_cfg=WorldModelConfig(dz=1, dx=1),
        proposal_cfg=None,  # or ProposalConfig(dz=1, dx=1, ...)
        data_builder_name="doublewell_1d",
        data_builder_kwargs={
            "well_separation": 2.5,
            "barrier_height": 1.0,
            "transition_noise": 0.15,
            "emission_noise": 0.10,
        },
        inference_sweeps={
            "evitrack": [
                {"K": 16, "C": 4, "tau": 1, "G": 1, "expand": "transition",
                 "prune_score": "evidence", "weight_mode": "evidence", "sigma_bg": 1.0},
            ],
            "particle_filter": [
                {"N": 16, "proposal_mode": "transition", "resample": True,
                 "resample_every_step": True, "ess_threshold_frac": 0.5},
            ],
            "random_beam": [
                {"K": 16, "C": 4, "tau": 1, "G": 1, "expand": "transition", "replace": False},
            ],
        },
    )
    experiments.append(AnalyticalSyntheticExperiment(analytical_cfg))

    # -------------------------------------------------
    # 2) Trained synthetic experiment
    # -------------------------------------------------
    run_cfg = RunConfig(
        experiment_name="trained_doublewell",
        run_root="results",
        seed=0,
        device="cpu",
        dtype="float32",
        epochs=10,
        wm_kind="markov",
        wm_cfg=WorldModelConfig(dz=1, dx=1),
        proposal_cfg=ProposalConfig(dz=1, dx=1),
        data_cfg=DataConfig(
            kind="synthetic",
            T=120,
            n_train=2048,
            n_val=256,
            batch_size=64,
            builder="data.synthetic_specs:build_doublewell_bundle",  # replace with your actual path
            builder_kwargs={
                "well_separation": 2.5,
                "barrier_height": 1.0,
                "transition_noise": 0.15,
                "emission_noise": 0.10,
            },
            include_latents_in_train=False,
            include_latents_in_val=True,
        ),
        note="trained WM on doublewell synthetic",
    )

    trained_cfg = TrainedSyntheticConfig(
        run_cfg=run_cfg,
        inference_seed=0,
        inference_sweeps={
            "evitrack": [
                {"K": 16, "C": 4, "tau": 1, "G": 1, "expand": "proposal",
                 "prune_score": "evidence", "weight_mode": "evidence", "sigma_bg": 1.0},
            ],
            "particle_filter": [
                {"N": 16, "proposal_mode": "proposal", "resample": True,
                 "resample_every_step": True, "ess_threshold_frac": 0.5},
            ],
            "random_beam": [
                {"K": 16, "C": 4, "tau": 1, "G": 1, "expand": "proposal", "replace": False},
            ],
        },
    )
    experiments.append(TrainedSyntheticExperiment(trained_cfg))

    return experiments


def main():
    experiments = build_experiments()

    for exp in experiments:
        print(f"\n=== Running {exp.name} (seed={exp.seed}) ===")
        exp.run()


if __name__ == "__main__":
    main()