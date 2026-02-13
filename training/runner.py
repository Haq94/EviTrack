# training/runner.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.trainer import Trainer, TrainerConfig
from utils.bundle import ModelBundle
from utils.builders import (
    make_wm_config_dict,
    make_proposal_config_dict,
)
from world_model import WorldModelConfig, MarkovWorldModel, NonMarkovWorldModel
from proposal import Proposal, ProposalConfig


# -------------------------
# Configs
# -------------------------
ObjectiveKind = Literal["beta_elbo", "iwae"]
WMKind = Literal["markov", "nonmarkov"]
DataKind = Literal["synthetic", "real"]


@dataclass
class DataConfig:
    kind: DataKind = "synthetic"

    # Dataloader
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True

    # Synthetic only (runner can generate basic linear Gaussian if you don't override)
    n_train: int = 2048
    n_val: int = 512
    T: int = 25
    dz: int = 4
    dx: int = 3
    process_noise: float = 0.10
    emit_noise: float = 0.15

    # Real data only: your dataset pointer (path, name, etc.)
    dataset_path: Optional[str] = None


@dataclass
class RunConfig:
    experiment_name: str = "default_experiment"
    run_root: str = "results"

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"  # "float32" | "float16" | "bfloat16"

    # Training length
    epochs: int = 5
    log_every_steps: int = 25
    save_every_steps: int = 200
    save_final: bool = True

    # Model kinds/configs
    wm_kind: WMKind = "nonmarkov"
    wm_cfg: WorldModelConfig = field(default_factory=lambda: WorldModelConfig(dz=4, dx=3))
    proposal_cfg: ProposalConfig = field(default_factory=lambda: ProposalConfig(dz=4, dx=3))

    # Trainer objective/optimizer
    trainer_cfg: TrainerConfig = field(default_factory=lambda: TrainerConfig(objective="beta_elbo", beta=1.0, K=16))

    # Data
    data_cfg: DataConfig = field(default_factory=DataConfig)

    # Notes / meta
    note: str = ""


# -------------------------
# Runner
# -------------------------
class ExperimentRunner:
    """
    Orchestrates:
      - seeding
      - model building
      - data building/loading
      - training loops (epochs/steps)
      - saving bundles + metrics

    Keeps Trainer focused on "single step".
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = self._parse_dtype(cfg.dtype)

        self.run_dir = self._make_run_dir(cfg.run_root, cfg.experiment_name, cfg.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "run_config.json"

        # Will be set in setup()
        self.wm: Optional[torch.nn.Module] = None
        self.proposal: Optional[torch.nn.Module] = None
        self.trainer: Optional[Trainer] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        # Save config immediately
        self._write_json(self.config_path, self._run_config_to_json(cfg))

    # ---------
    # Public API
    # ---------
    def setup(self) -> None:
        self._seed_everything(self.cfg.seed)
        self.wm = self.build_world_model()
        self.proposal = self.build_proposal(self.wm)
        self.trainer = Trainer(wm=self.wm, proposal=self.proposal, cfg=self.cfg.trainer_cfg)

        self.train_loader, self.val_loader = self.build_or_load_data()

    def fit(self) -> None:
        assert self.trainer is not None, "Call setup() first."
        assert self.train_loader is not None, "No train_loader."

        global_step = 0
        t0 = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            for (xb,) in self.train_loader:
                global_step += 1
                stats = self.trainer.train_step({"x": xb})

                if global_step % self.cfg.log_every_steps == 0 or global_step == 1:
                    msg = {
                        "time": time.time(),
                        "elapsed_s": time.time() - t0,
                        "epoch": epoch,
                        "step": global_step,
                        **stats,
                    }
                    self._append_jsonl(self.metrics_path, msg)

                if global_step % self.cfg.save_every_steps == 0:
                    self.save_bundle(tag=f"ckpt_step_{global_step:06d}", step=global_step, epoch=epoch)

            # Optional end-of-epoch eval
            if self.val_loader is not None:
                val = self.evaluate()
                msg = {
                    "time": time.time(),
                    "elapsed_s": time.time() - t0,
                    "epoch": epoch,
                    "step": global_step,
                    "val_loss": val["loss"],
                }
                self._append_jsonl(self.metrics_path, msg)

        if self.cfg.save_final:
            self.save_bundle(tag="final", step=global_step, epoch=self.cfg.epochs)

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        assert self.trainer is not None
        if self.val_loader is None:
            return {"loss": float("nan")}

        losses: List[float] = []
        for i, (xb,) in enumerate(self.val_loader):
            out = self.trainer.eval_step({"x": xb})
            losses.append(out["loss"])
            if max_batches is not None and (i + 1) >= max_batches:
                break
        return {"loss": float(np.mean(losses))}

    # ---------
    # Model building
    # ---------
    def build_world_model(self):
        kind = self.cfg.wm_kind.lower()
        wm_cfg = self.cfg.wm_cfg

        if kind in ("markov", "m"):
            wm = MarkovWorldModel(wm_cfg)
        elif kind in ("nonmarkov", "non-markov", "nm"):
            wm = NonMarkovWorldModel(wm_cfg)
        else:
            raise ValueError(f"Unknown wm_kind={self.cfg.wm_kind}")

        return wm.to(device=self.device, dtype=self.dtype)

    def build_proposal(self, wm):
        q = Proposal(self.cfg.proposal_cfg, wm=wm)
        return q.to(device=self.device, dtype=self.dtype)

    # ---------
    # Data building/loading
    # ---------
    def build_or_load_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Dispatch between synthetic generation and real-data loading.
        If you implement real data, override load_data().
        """
        if self.cfg.data_cfg.kind == "synthetic":
            return self._build_synthetic_data()
        if self.cfg.data_cfg.kind == "real":
            return self.load_data()
        raise ValueError(f"Unknown data kind: {self.cfg.data_cfg.kind}")

    def load_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Override this for real datasets.

        Expected return:
          train_loader: yields batches shaped [B, T, dx] under key 'x'
          val_loader: optional

        Default implementation raises to force you to implement it.
        """
        raise NotImplementedError(
            "RunConfig.data_cfg.kind='real' but ExperimentRunner.load_data() is not implemented. "
            "Override load_data() in a subclass (or edit runner.py) to return DataLoaders."
        )

    # ---------
    # Saving
    # ---------
    def save_bundle(self, *, tag: str, step: int, epoch: int) -> None:
        assert self.wm is not None
        assert self.proposal is not None

        out_dir = self.run_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        wm_cfg_dict = make_wm_config_dict(self.cfg.wm_cfg, kind=self.cfg.wm_kind)
        q_cfg_dict = make_proposal_config_dict(self.cfg.proposal_cfg)

        bundle = ModelBundle(
            wm=self.wm,
            proposal=self.proposal,
            wm_config=wm_cfg_dict,
            proposal_config=q_cfg_dict,
            meta={
                "experiment_name": self.cfg.experiment_name,
                "seed": self.cfg.seed,
                "wm_kind": self.cfg.wm_kind,
                "objective": self.cfg.trainer_cfg.objective,
                "beta": self.cfg.trainer_cfg.beta,
                "K": self.cfg.trainer_cfg.K,
                "epoch": int(epoch),
                "step": int(step),
                "note": self.cfg.note,
            },
        )
        bundle.save(out_dir)

    # -------------------------
    # Synthetic fallback generator (simple linear Gaussian)
    # -------------------------
    def _build_synthetic_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Minimal synthetic generator so runner works out-of-the-box.
        If you prefer your own generator, replace this with your file.
        """
        dc = self.cfg.data_cfg
        rng = np.random.default_rng(self.cfg.seed)

        dz, dx, T = dc.dz, dc.dx, dc.T
        A = rng.standard_normal((dz, dz)) * 0.2
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        s = np.clip(s, 0.0, 0.9)
        A = (u * s) @ vt
        C = rng.standard_normal((dx, dz)) * 0.8

        def gen(N: int, seed0: int) -> torch.Tensor:
            xs = np.zeros((N, T, dx), dtype=np.float32)
            for i in range(N):
                r = np.random.default_rng(seed0 + i)
                z = r.standard_normal((T, dz)).astype(np.float32)
                # build z trajectory
                for t in range(1, T):
                    z[t] = (A @ z[t - 1]) + (dc.process_noise * r.standard_normal(dz).astype(np.float32))
                # emit x
                x = (z @ C.T) + (dc.emit_noise * r.standard_normal((T, dx)).astype(np.float32))
                xs[i] = x
            return torch.tensor(xs, dtype=self.dtype)

        x_train = gen(dc.n_train, seed0=1000)
        x_val = gen(dc.n_val, seed0=9000)

        train_loader = DataLoader(
            TensorDataset(x_train),
            batch_size=dc.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory and (self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            TensorDataset(x_val),
            batch_size=dc.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory and (self.device.type == "cuda"),
        )
        return train_loader, val_loader

    # -------------------------
    # Utils
    # -------------------------
    @staticmethod
    def _parse_dtype(s: str) -> torch.dtype:
        s = s.lower()
        if s in ("float32", "fp32"):
            return torch.float32
        if s in ("float16", "fp16"):
            return torch.float16
        if s in ("bfloat16", "bf16"):
            return torch.bfloat16
        raise ValueError(f"Unknown dtype: {s}")

    @staticmethod
    def _make_run_dir(run_root: str, experiment_name: str, seed: int) -> Path:
        return Path(run_root) / experiment_name / f"seed_{seed:03d}"

    @staticmethod
    def _seed_everything(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _write_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @staticmethod
    def _append_jsonl(path: Union[str, Path], row: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def _run_config_to_json(cfg: RunConfig) -> Dict[str, Any]:
        # dataclasses -> JSON, including nested dataclasses
        d = asdict(cfg)
        return d
