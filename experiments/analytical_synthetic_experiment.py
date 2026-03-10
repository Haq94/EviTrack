# experiments/analytical_synthetic_experiment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from experiments.base_experiment import BaseExperiment
from experiments.inference_eval import run_online_inference, save_inference_result

from world_model import WorldModelConfig
from world_model.analytical import AnalyticalWorldModel
from proposal import Proposal, ProposalConfig

from data.synthetic_generator import build_synthetic_bundle
from data.synthetic_tasks.doublewell_1d import (
    make_prior,
    make_transition,
    make_emission,
)


@dataclass
class AnalyticalSyntheticConfig:
    experiment_name: str
    run_root: str = "results"
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    T: int = 120
    n_train: int = 0
    n_val: int = 512
    n_test: int = 0
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    wm_cfg: Optional[WorldModelConfig] = None
    proposal_cfg: Optional[ProposalConfig] = None

    data_builder_name: str = "doublewell_1d"
    data_builder_kwargs: Optional[Dict[str, Any]] = None

    inference_sweeps: Optional[Dict[str, list]] = None


class AnalyticalSyntheticExperiment(BaseExperiment):
    def __init__(self, cfg: AnalyticalSyntheticConfig):
        super().__init__(
            name=cfg.experiment_name,
            run_root=cfg.run_root,
            seed=cfg.seed,
        )
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

    def _task_kwargs(self) -> Dict[str, Any]:
        return dict(self.cfg.data_builder_kwargs or {})

    def _build_spec(self) -> Dict[str, Any]:
        if self.cfg.data_builder_name != "doublewell_1d":
            raise ValueError(
                f"Unsupported analytical synthetic task '{self.cfg.data_builder_name}'. "
                "Currently only 'doublewell_1d' is wired."
            )

        kwargs = self._task_kwargs()
        prior = make_prior(
            z0_mean=kwargs.get("z0_mean", 0.0),
            z0_std=kwargs.get("z0_std", 1.0),
        )
        transition = make_transition(
            a=kwargs.get("a", 3.0),
            V=kwargs.get("V", 0.06),
            dt=kwargs.get("dt", 1.0),
            sigma_z=kwargs.get("sigma_z", 0.05),
        )
        emission = make_emission(
            d=kwargs.get("d", 2.0),
            n=kwargs.get("n", 1),
            sigma_x=kwargs.get("sigma_x", 0.12),
        )

        meta = {
            "task": "doublewell_1d",
            "dz": 1,
            "dx": 1,
            "a": kwargs.get("a", 3.0),
            "V": kwargs.get("V", 0.06),
            "dt": kwargs.get("dt", 1.0),
            "sigma_z": kwargs.get("sigma_z", 0.05),
            "d": kwargs.get("d", 2.0),
            "n": kwargs.get("n", 1),
            "sigma_x": kwargs.get("sigma_x", 0.12),
            "z0_mean": kwargs.get("z0_mean", 0.0),
            "z0_std": kwargs.get("z0_std", 1.0),
        }

        return {
            "prior": prior,
            "transition": transition,
            "emission": emission,
            "meta": meta,
        }

    def _build_data(self, spec):
        return build_synthetic_bundle(
            T=self.cfg.T,
            n_train=self.cfg.n_train,
            n_val=self.cfg.n_val,
            n_test=self.cfg.n_test,
            prior=spec["prior"],
            transition=spec["transition"],
            emission=spec["emission"],
            seed=self.cfg.seed,
            batch_size=self.cfg.batch_size,
            device=str(self.device),
            dtype=self.dtype,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last_train=False,
            include_latents_in_train=False,
            include_latents_in_val=True,
            include_latents_in_test=True,
            extras=None,
            meta=spec["meta"],
        )

    def _build_wm(self, spec) -> AnalyticalWorldModel:
        wm_cfg = self.cfg.wm_cfg
        if wm_cfg is None:
            raise ValueError("cfg.wm_cfg must be provided.")

        wm = AnalyticalWorldModel(
            cfg=wm_cfg,
            prior_mu0=spec["prior"].mu0.to(device=self.device, dtype=self.dtype),
            prior_cov0=spec["prior"].cov0.to(device=self.device, dtype=self.dtype),
            trans_mean=spec["transition"].mean_fn,
            trans_cov=spec["transition"].cov_fn,
            emit_mean=spec["emission"].mean_fn,
            emit_cov=spec["emission"].cov_fn,
        )
        return wm.to(device=self.device, dtype=self.dtype)

    def _build_proposal(self, wm):
        if self.cfg.proposal_cfg is None:
            return None
        q = Proposal(self.cfg.proposal_cfg, wm=wm)
        return q.to(device=self.device, dtype=self.dtype)

    def run(self) -> Dict[str, Any]:
        spec = self._build_spec()
        data_bundle = self._build_data(spec)

        wm = self._build_wm(spec)
        proposal = self._build_proposal(wm)

        self.save_metadata({
            "kind": "analytical_synthetic",
            "seed": self.cfg.seed,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "data_builder_name": self.cfg.data_builder_name,
            "data_builder_kwargs": self.cfg.data_builder_kwargs or {},
            "wm_cfg": self.cfg.wm_cfg,
            "proposal_cfg": self.cfg.proposal_cfg,
            "data_meta": data_bundle.meta,
        })

        results = {
            "kind": "analytical_synthetic",
            "seed": self.cfg.seed,
            "data_meta": data_bundle.meta,
            "engines": [],
        }

        val_loader = data_bundle.val

        for engine_name, cfg_list in (self.cfg.inference_sweeps or {}).items():
            for i, engine_cfg in enumerate(cfg_list):
                out = run_online_inference(
                    wm=wm,
                    proposal=proposal,
                    data_loader=val_loader,
                    engine_name=engine_name,
                    engine_cfg=engine_cfg,
                    seed=self.cfg.seed,
                    device=self.device,
                    dtype=self.dtype,
                    max_batches=None,
                )
                save_inference_result(
                    self.run_dir / "inference" / engine_name / f"run_{i:03d}.json",
                    out,
                )
                results["engines"].append(out)

        self.save_json(self.run_dir / "summary.json", results)
        return results