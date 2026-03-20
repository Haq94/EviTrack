# experiments/analytical_synthetic_experiment.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from experiments.base_experiment import BaseExperiment
from experiments.inference_eval import run_and_save_inference_states

from world_model import WorldModelConfig
from world_model.analytical import AnalyticalWorldModel

from data.synthetic_tasks.doublewell_1d import make_prior, make_transition, make_emission
from data.synthetic_tasks.doublewell_1d_dataset import (
    DoubleWell1DDatasetArtifact,
    build_doublewell_1d_dataset,
)


# ---------------------------------------------------------------
# Configs
# ---------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Controls how the fixed benchmark dataset is obtained."""

    # Path to existing artifact (data.pt + metadata.json).
    # If None and generate=True, dataset is built and saved to a default location.
    path: Optional[str] = None

    # Generate if not found
    generate: bool = True

    # Generation parameters
    T: int = 120
    n_delayed: int = 25
    n_non_delayed: int = 25
    search_seed_start: int = 0
    max_seed_search: int = 100_000

    # Double-well task parameters
    a: float = 3.0
    V: float = 0.06
    dt: float = 1.0
    sigma_z: float = 0.05
    d: float = 2.0
    n: int = 1
    sigma_x: float = 0.12
    z0_mean: float = 0.0
    z0_std: float = 1.0

    # Quadrature disambiguation detection
    threshold: float = 0.8
    zmin: float = -4.0
    zmax: float = 4.0
    G: int = 1000


@dataclass
class AnalyticalSyntheticConfig:
    experiment_name: str = "doublewell_analytical"
    run_root: str = "results"
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wm_cfg: WorldModelConfig = field(default_factory=lambda: WorldModelConfig(dz=1, dx=1))

    # engine_name -> list of engine_cfg dicts (one run per cfg)
    # engine_name must be one of:
    #   evitrack_evidence, evitrack_joint, evitrack_tbd, bootstrap_pf, random_beam
    inference_sweeps: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Multiple inference seeds — controls stochasticity in sampling/resampling
    inference_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    overwrite: bool = False
    verbose: bool = True


# ---------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------

class AnalyticalSyntheticExperiment(BaseExperiment):

    def __init__(self, cfg: AnalyticalSyntheticConfig):
        super().__init__(
            name=cfg.experiment_name,
            run_root=cfg.run_root,
            seed=cfg.seed,
            use_seed_dir=False,
        )
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

    # ----------------------------------------------------------
    # Dataset: load or generate
    # ----------------------------------------------------------

    def _resolve_dataset(self) -> DoubleWell1DDatasetArtifact:
        dc = self.cfg.dataset

        # 1) Try explicit path first
        if dc.path is not None:
            p = Path(dc.path)
            if (p / "data.pt").exists():
                if self.cfg.verbose:
                    print(f"[dataset] Loading from {p}")
                return DoubleWell1DDatasetArtifact.load(p)
            elif not dc.generate:
                raise FileNotFoundError(
                    f"Dataset not found at {p} and generate=False."
                )

        # 2) Default cache location
        default_path = self.run_dir / "dataset"
        if (default_path / "data.pt").exists() and not self.cfg.overwrite:
            if self.cfg.verbose:
                print(f"[dataset] Loading cached dataset from {default_path}")
            return DoubleWell1DDatasetArtifact.load(default_path)

        # 3) Generate
        if not dc.generate:
            raise FileNotFoundError("No dataset path provided and generate=False.")

        if self.cfg.verbose:
            print(f"[dataset] Generating {dc.n_delayed} delayed + "
                  f"{dc.n_non_delayed} non-delayed trajectories (T={dc.T}) ...")

        artifact = build_doublewell_1d_dataset(
            T=dc.T,
            n_delayed=dc.n_delayed,
            n_non_delayed=dc.n_non_delayed,
            search_seed_start=dc.search_seed_start,
            max_seed_search=dc.max_seed_search,
            device=str(self.device),
            dtype=self.dtype,
            a=dc.a, V=dc.V, dt=dc.dt, sigma_z=dc.sigma_z,
            d=dc.d, n=dc.n, sigma_x=dc.sigma_x,
            z0_mean=dc.z0_mean, z0_std=dc.z0_std,
            threshold=dc.threshold,
            zmin=dc.zmin, zmax=dc.zmax, G=dc.G,
            verbose=self.cfg.verbose,
        )

        save_path = Path(dc.path) if dc.path is not None else default_path
        artifact.save(save_path)
        if self.cfg.verbose:
            print(f"[dataset] Saved to {save_path}")

        return artifact

    # ----------------------------------------------------------
    # World model (analytical — no training)
    # ----------------------------------------------------------

    def _build_wm(self) -> AnalyticalWorldModel:
        dc = self.cfg.dataset
        prior      = make_prior(z0_mean=dc.z0_mean, z0_std=dc.z0_std)
        transition = make_transition(a=dc.a, V=dc.V, dt=dc.dt, sigma_z=dc.sigma_z)
        emission   = make_emission(d=dc.d, n=dc.n, sigma_x=dc.sigma_x)

        wm = AnalyticalWorldModel(
            cfg=self.cfg.wm_cfg,
            prior_mu0=prior.mu0.to(device=self.device, dtype=self.dtype),
            prior_cov0=prior.cov0.to(device=self.device, dtype=self.dtype),
            trans_mean=transition.mean_fn,
            trans_cov=transition.cov_fn,
            emit_mean=emission.mean_fn,
            emit_cov=emission.cov_fn,
        )
        return wm.to(device=self.device, dtype=self.dtype)

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        artifact = self._resolve_dataset()
        wm = self._build_wm()

        N = artifact.x.shape[0]
        n_delayed     = int(artifact.delayed_flag.sum().item())
        n_non_delayed = int((~artifact.delayed_flag).sum().item())

        if self.cfg.verbose:
            print(f"\n[experiment] {self.cfg.experiment_name}")
            print(f"  trajectories  : {N}  ({n_delayed} delayed, {n_non_delayed} non-delayed)")
            print(f"  device        : {self.device}")
            print(f"  engines       : {list(self.cfg.inference_sweeps.keys())}")
            print(f"  inference seeds: {self.cfg.inference_seeds}\n")

        # Save top-level experiment metadata once
        # Points back to dataset — no duplication of x/z/flags
        dataset_path = (
            self.cfg.dataset.path
            if self.cfg.dataset.path is not None
            else str(self.run_dir / "dataset")
        )
        self.save_metadata({
            "experiment_name":  self.cfg.experiment_name,
            "seed":             self.cfg.seed,
            "device":           str(self.device),
            "N":                N,
            "n_delayed":        n_delayed,
            "n_non_delayed":    n_non_delayed,
            "dataset_path":     dataset_path,
            "dataset_meta":     artifact.meta,
            "inference_seeds":  self.cfg.inference_seeds,
            "inference_sweeps": self.cfg.inference_sweeps,
        })

        run_summary: Dict[str, Any] = {
            "experiment_name": self.cfg.experiment_name,
            "N": N,
            "runs": [],
        }

        # Loop: engine -> K config -> inference seed
        # Folder: run_dir / engine_name / K{k} / inference_seed_{s} / traj_XXXX.pt
        for engine_name, cfg_list in self.cfg.inference_sweeps.items():
            for engine_cfg in cfg_list:
                k_tag = _cfg_to_tag(engine_cfg)

                for inf_seed in self.cfg.inference_seeds:
                    out_dir = (
                        self.run_dir
                        / engine_name
                        / k_tag
                        / f"inference_seed_{inf_seed:03d}"
                    )

                    if self.cfg.verbose:
                        print(f"[inference] {engine_name} | {k_tag} | "
                              f"inference_seed={inf_seed}")

                    saved = run_and_save_inference_states(
                        wm=wm,
                        proposal=None,          # analytical: no proposal
                        artifact=artifact,
                        engine_name=engine_name,
                        engine_cfg=engine_cfg,
                        out_dir=out_dir,
                        inference_seed=inf_seed,
                        device=self.device,
                        dtype=self.dtype,
                        overwrite=self.cfg.overwrite,
                        verbose=self.cfg.verbose,
                    )

                    run_summary["runs"].append({
                        "engine_name":    engine_name,
                        "engine_cfg":     engine_cfg,
                        "inference_seed": inf_seed,
                        "out_dir":        str(out_dir),
                        "n_saved":        len(saved),
                    })

        self.save_json(self.run_dir / "summary.json", run_summary)

        if self.cfg.verbose:
            print(f"\n[done] Results saved to: {self.run_dir}")

        return run_summary


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _cfg_to_tag(cfg: Dict[str, Any]) -> str:
    """
    Convert engine cfg dict to a short directory-safe tag.
    e.g. {"K": 5, "C": 3} -> "K5_C3"
    Skips string-valued keys (expand, prune_score etc) for brevity.
    """
    parts = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            parts.append(f"{k}{int(v)}")
        elif isinstance(v, float) and v == int(v):
            parts.append(f"{k}{int(v)}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}{v}")
    return "_".join(parts) if parts else "default"
