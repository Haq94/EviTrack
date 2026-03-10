# experiments/trained_synthetic_experiment.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from experiments.base_experiment import BaseExperiment
from experiments.inference_eval import run_online_inference, save_inference_result

from training.runner import RunConfig, ExperimentRunner
from utils.bundle import ModelBundle


@dataclass
class TrainedSyntheticConfig:
    run_cfg: RunConfig
    inference_sweeps: Dict[str, list]
    inference_seed: int = 0


class TrainedSyntheticExperiment(BaseExperiment):
    def __init__(self, cfg: TrainedSyntheticConfig):
        super().__init__(
            name=cfg.run_cfg.experiment_name,
            run_root=cfg.run_cfg.run_root,
            seed=cfg.run_cfg.seed,
        )
        self.cfg = cfg
        self.device = torch.device(cfg.run_cfg.device)
        self.dtype = ExperimentRunner._parse_dtype(cfg.run_cfg.dtype)

    def run(self) -> Dict[str, Any]:
        runner = ExperimentRunner(self.cfg.run_cfg)
        runner.setup()
        runner.fit()

        bundle_dir = runner.run_dir / "final"
        bundle = ModelBundle.load(
            bundle_dir,
            device=self.device,
            dtype=self.dtype,
            strict=True,
            load_proposal=True,
        ).eval()

        results = {
            "kind": "trained_synthetic",
            "seed": self.cfg.run_cfg.seed,
            "bundle_dir": str(bundle_dir),
            "engines": [],
        }

        val_loader = runner.val_loader
        if val_loader is None:
            raise ValueError("Expected val_loader for inference evaluation, got None.")

        for engine_name, cfg_list in self.cfg.inference_sweeps.items():
            for i, engine_cfg in enumerate(cfg_list):
                out = run_online_inference(
                    wm=bundle.wm,
                    proposal=bundle.proposal,
                    data_loader=val_loader,
                    engine_name=engine_name,
                    engine_cfg=engine_cfg,
                    seed=self.cfg.inference_seed,
                    device=self.device,
                    dtype=self.dtype,
                )
                save_inference_result(
                    self.run_dir / "inference" / engine_name / f"run_{i:03d}.json",
                    out,
                )
                results["engines"].append(out)

        self.save_json(self.run_dir / "summary.json", results)
        return results