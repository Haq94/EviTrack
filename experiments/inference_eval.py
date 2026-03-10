# experiments/inference_eval.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch

from inference.evitrack import EviTrackEngine, EviTrackConfig
from inference.baselines.pf import ParticleFilterEngine, ParticleFilterConfig
from inference.baselines.random_beam import RandomBeamEngine, RandomBeamConfig


def _get_x_from_batch(batch) -> torch.Tensor:
    if isinstance(batch, Mapping):
        return batch["x"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if torch.is_tensor(batch):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _engine_from_spec(name: str, wm, proposal, cfg_dict: Dict[str, Any]):
    name = name.lower()
    if name == "evitrack":
        return EviTrackEngine(wm=wm, proposal=proposal, cfg=EviTrackConfig(**cfg_dict))
    if name == "particle_filter":
        return ParticleFilterEngine(wm=wm, proposal=proposal, cfg=ParticleFilterConfig(**cfg_dict))
    if name == "random_beam":
        return RandomBeamEngine(wm=wm, proposal=proposal, cfg=RandomBeamConfig(**cfg_dict))
    raise ValueError(f"Unknown engine: {name}")


@torch.no_grad()
def run_online_inference(
    *,
    wm,
    proposal,
    data_loader,
    engine_name: str,
    engine_cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    _seed_everything(seed)

    wm.eval()
    if proposal is not None:
        proposal.eval()

    engine = _engine_from_spec(engine_name, wm=wm, proposal=proposal, cfg_dict=engine_cfg)

    batch_summaries: List[Dict[str, Any]] = []
    total_steps = 0

    for bidx, batch in enumerate(data_loader):
        x = _get_x_from_batch(batch).to(device=device, dtype=dtype)  # [B,T,dx]
        B, T, _ = x.shape

        state = engine.init_state(B=B, device=str(device), dtype=dtype)
        stats_per_t = []

        for t in range(T):
            x_t = x[:, t, :]
            state, stats = engine.step(state, x_t)
            stats_per_t.append({
                "t": int(stats.t),
                "kept": int(stats.kept),
                "candidates": int(stats.candidates),
            })
            total_steps += 1

        w, support = engine.get_mixture(state)

        batch_summaries.append({
            "batch_index": bidx,
            "B": int(B),
            "T": int(T),
            "final_num_hypotheses": int(w.shape[1]),
            "mean_entropy": float((-w.clamp_min(1e-12).log() * w).sum(dim=1).mean().item()),
            "stats_per_t": stats_per_t,
        })

        if max_batches is not None and (bidx + 1) >= max_batches:
            break

    return {
        "engine": engine_name,
        "engine_cfg": engine_cfg,
        "seed": int(seed),
        "num_batches": len(batch_summaries),
        "total_steps": int(total_steps),
        "batches": batch_summaries,
    }


def save_inference_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)