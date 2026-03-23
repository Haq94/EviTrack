# tests/trained_synthetic_experiment_script.py
# TODO: V&V needed
"""
Smoke test for TrainedSyntheticExperiment.

Verifies that after a tiny run (T=10, n=5 trajectories, 1 epoch) the expected
folder structure and key files are produced.
"""
from __future__ import annotations

import torch
import pytest

from data.synthetic_tasks.doublewell_1d_dataset import DoubleWell1DDatasetArtifact

from experiments.trained_synthetic_experiment import (
    TrainedSyntheticConfig,
    TrainedSyntheticExperiment,
)
from training.trainer import TrainerConfig
from world_model import WorldModelConfig
from world_model.base import HeadConfig
from proposal.proposal import ProposalConfig


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_tiny_artifact(T: int = 10, N: int = 5, dz: int = 1, dx: int = 1) -> DoubleWell1DDatasetArtifact:
    """Build a minimal fake artifact without running the heavy dataset generator."""
    torch.manual_seed(0)
    x = torch.randn(N, T, dx)
    z = torch.randn(N, T, dz)
    # Mix delayed and non-delayed flags
    delayed_flag = torch.tensor([True, True, False, False, False], dtype=torch.bool)
    disamb_time = torch.tensor([5, 7, -1, -1, -1], dtype=torch.long)
    data_seed_ids = torch.arange(N, dtype=torch.long)
    meta = {
        "task": "doublewell_1d",
        "T": T,
        "dz": dz,
        "dx": dx,
        "num_sequences": N,
        "num_delayed": int(delayed_flag.sum().item()),
        "num_non_delayed": int((~delayed_flag).sum().item()),
    }
    return DoubleWell1DDatasetArtifact(
        x=x,
        z=z,
        data_seed_ids=data_seed_ids,
        delayed_flag=delayed_flag,
        disamb_time=disamb_time,
        meta=meta,
    )


def _tiny_head() -> HeadConfig:
    """Smallest valid HeadConfig to keep the test fast."""
    return HeadConfig(hidden_dim=8, num_layers=1)


def _tiny_wm_cfg(dz: int = 1, dx: int = 1) -> WorldModelConfig:
    return WorldModelConfig(
        dz=dz,
        dx=dx,
        z_mem_dim=8,
        x_mem_dim=8,
        transition=_tiny_head(),
        emission=_tiny_head(),
    )


def _tiny_proposal_cfg(dz: int = 1, dx: int = 1) -> ProposalConfig:
    return ProposalConfig(
        dz=dz,
        dx=dx,
        z_mem_dim=8,
        x_mem_dim=8,
        head=_tiny_head(),
    )


# ---------------------------------------------------------------
# Test
# ---------------------------------------------------------------

def test_folder_structure(tmp_path):
    """
    End-to-end smoke test: tiny artifact, 1 model seed, 1 inference seed,
    1 epoch.  Checks that all expected directories and files are created.
    """
    T, N = 10, 5
    dz, dx = 1, 1

    # --- save fake dataset ---
    artifact = _make_tiny_artifact(T=T, N=N, dz=dz, dx=dx)
    dataset_dir = tmp_path / "dataset"
    artifact.save(dataset_dir)

    # --- config ---
    cfg = TrainedSyntheticConfig(
        experiment_name="doublewell_trained",
        run_root=str(tmp_path / "results"),
        device="cpu",
        dtype=torch.float32,
        dataset_path=str(dataset_dir),
        model_seeds=[0],
        inference_seeds=[0],
        wm_cfg=_tiny_wm_cfg(dz=dz, dx=dx),
        proposal_cfg=_tiny_proposal_cfg(dz=dz, dx=dx),
        trainer_cfg=TrainerConfig(objective="beta_elbo", beta=1.0, K=2, lr=1e-3),
        epochs=1,
        T=T,
        batch_size=2,
        inference_sweeps={
            # EviTrack with proposal (expand injected automatically)
            "evitrack_evidence": [{"K": 2, "C": 2}],
            # Bootstrap PF (no proposal needed)
            "bootstrap_pf": [{"N": 2}],
        },
        overwrite=True,
        verbose=False,
    )

    # --- run ---
    exp = TrainedSyntheticExperiment(cfg)
    summary = exp.run()

    # --- verify top-level files ---
    run_dir = tmp_path / "results" / "doublewell_trained"
    assert (run_dir / "experiment_meta.json").exists(), "experiment_meta.json missing"
    assert (run_dir / "summary.json").exists(), "summary.json missing"

    # --- verify model_seed directory ---
    ms_dir = run_dir / "model_seed_000"
    assert ms_dir.is_dir(), "model_seed_000/ directory missing"

    # --- verify training output (ExperimentRunner saves to training/seed_000/) ---
    training_dir = ms_dir / "training" / "seed_000"
    assert training_dir.is_dir(), "training/seed_000/ directory missing"
    assert (training_dir / "run_config.json").exists(), "run_config.json missing"
    assert (training_dir / "final").is_dir(), "training/seed_000/final/ checkpoint missing"

    # --- verify EviTrack inference output ---
    # "expand" is a string key, so _cfg_to_tag produces "K2_C2"
    evitrack_dir = ms_dir / "evitrack_evidence" / "K2_C2" / "inference_seed_000"
    assert evitrack_dir.is_dir(), f"EviTrack output dir missing: {evitrack_dir}"
    assert (evitrack_dir / "run_meta.json").exists(), "EviTrack run_meta.json missing"

    # Check that traj files exist for every trajectory in the artifact
    for i in range(N):
        traj_file = evitrack_dir / f"traj_{i:04d}.npz"
        assert traj_file.exists(), f"Missing {traj_file.name} in EviTrack output"

    # --- verify bootstrap_pf inference output ---
    bpf_dir = ms_dir / "bootstrap_pf" / "N2" / "inference_seed_000"
    assert bpf_dir.is_dir(), f"BPF output dir missing: {bpf_dir}"
    assert (bpf_dir / "run_meta.json").exists(), "BPF run_meta.json missing"

    for i in range(N):
        traj_file = bpf_dir / f"traj_{i:04d}.npz"
        assert traj_file.exists(), f"Missing {traj_file.name} in BPF output"

    # --- verify summary content ---
    assert summary["N"] == N
    assert len(summary["model_seeds"]) == 1
    assert summary["model_seeds"][0]["model_seed"] == 0
    assert len(summary["model_seeds"][0]["runs"]) == 2  # 2 engines x 1 cfg x 1 seed


def test_expand_is_patched_for_evitrack():
    """_patch_expand injects expand='proposal' for EviTrack engines but not BPF."""
    from claude_code_dev.experiments.trained_synthetic_experiment import (
        TrainedSyntheticExperiment,
        TrainedSyntheticConfig,
    )

    cfg = TrainedSyntheticConfig()
    exp = TrainedSyntheticExperiment.__new__(TrainedSyntheticExperiment)
    exp.cfg = cfg

    patched = exp._patch_expand("evitrack_evidence", {"K": 5, "C": 3})
    assert patched["expand"] == "proposal"

    # Already set — should not be overwritten
    patched2 = exp._patch_expand("evitrack_joint", {"K": 5, "C": 3, "expand": "transition"})
    assert patched2["expand"] == "transition"

    # BPF — should be unchanged
    bpf = exp._patch_expand("bootstrap_pf", {"N": 20})
    assert "expand" not in bpf


def test_cfg_to_tag():
    """_cfg_to_tag produces compact tags, skipping string-valued keys."""
    from claude_code_dev.experiments.trained_synthetic_experiment import _cfg_to_tag

    assert _cfg_to_tag({"K": 5, "C": 3}) == "K5_C3"
    assert _cfg_to_tag({"K": 5, "C": 3, "expand": "proposal"}) == "K5_C3"
    assert _cfg_to_tag({"N": 20}) == "N20"
    assert _cfg_to_tag({}) == "default"
    assert _cfg_to_tag({"tau": 2.0}) == "tau2"
