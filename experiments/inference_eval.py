# experiments/inference_eval.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from inference.evitrack import EviTrackEngine, EviTrackConfig
from inference.baselines.pf import ParticleFilterEngine, ParticleFilterConfig
from inference.baselines.bpf import BPFEngine, BPFConfig
from inference.baselines.random_beam import RandomBeamEngine, RandomBeamConfig


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _engine_from_spec(name: str, wm, proposal, cfg_dict: Dict[str, Any]):
    """
    Build an inference engine from name + cfg dict.
    prune_score / weight_mode are baked in by engine name for EviTrack variants.
    expand and K/C/N must always be supplied explicitly in cfg_dict.
    """
    name = name.lower()
    if name == "evitrack_evidence":
        return EviTrackEngine(
            wm=wm, proposal=proposal,
            cfg=EviTrackConfig(prune_score="evidence", weight_mode="evidence", **cfg_dict),
        )
    if name == "evitrack_joint":
        return EviTrackEngine(
            wm=wm, proposal=proposal,
            cfg=EviTrackConfig(prune_score="joint", weight_mode="joint", **cfg_dict),
        )
    if name == "evitrack_tbd":
        return EviTrackEngine(
            wm=wm, proposal=proposal,
            cfg=EviTrackConfig(prune_score="tbd_joint", weight_mode="tbd_joint", **cfg_dict),
        )
    if name == "bootstrap_pf":
        return BPFEngine(wm=wm, cfg=BPFConfig(**cfg_dict))
    if name == "random_beam":
        return RandomBeamEngine(wm=wm, proposal=proposal, cfg=RandomBeamConfig(**cfg_dict))
    raise ValueError(
        f"Unknown engine: '{name}'. "
        f"Valid: evitrack_evidence, evitrack_joint, evitrack_tbd, bootstrap_pf, random_beam"
    )


def _extract_state_snapshot(state) -> Dict[str, np.ndarray]:
    """
    Extract a compact numpy snapshot from EviTrackState or ParticleState at one timestep.

    Markov (hidden states None): saves only z + scores.
    NonMarkov: also saves whichever hidden states are not None.

    Returns a flat dict of numpy arrays, all shape [K, ...].
    """
    # EviTrackState (EviTrack flavors + RandomBeam — both use .hyps)
    if hasattr(state, "hyps"):
        hyps = state.hyps[0]  # B=1
        snap: Dict[str, np.ndarray] = {
            "type_id": np.array(0, dtype=np.int8),   # 0 = evitrack
            "z":       np.stack([h.z_t.squeeze(0).cpu().numpy() for h in hyps]),  # [K, dz]
            "J":       np.array([h.J.item()     for h in hyps], dtype=np.float32),
            "E":       np.array([h.E.item()     for h in hyps], dtype=np.float32),
            "J_tbd":   np.array([h.J_tbd.item() for h in hyps], dtype=np.float32),
        }
        if hyps[0].wm_z_state is not None:
            snap["wm_z_state"] = np.stack(
                [h.wm_z_state.squeeze(0).cpu().numpy() for h in hyps])
        if hyps[0].wm_x_state is not None:
            snap["wm_x_state"] = np.stack(
                [h.wm_x_state.squeeze(0).cpu().numpy() for h in hyps])
        if hyps[0].q_z_state is not None:
            snap["q_z_state"]  = np.stack(
                [h.q_z_state.squeeze(0).cpu().numpy()  for h in hyps])
        if hyps[0].q_x_state is not None:
            snap["q_x_state"]  = np.stack(
                [h.q_x_state.squeeze(0).cpu().numpy()  for h in hyps])
        return snap

    # ParticleState (bootstrap PF)
    if hasattr(state, "particles"):
        parts = state.particles[0]  # B=1
        snap = {
            "type_id": np.array(1, dtype=np.int8),   # 1 = particle
            "z":       np.stack([p.z_t.squeeze(0).cpu().numpy() for p in parts]),
            "logw":    np.array([p.logw.item() for p in parts], dtype=np.float32),
        }
        if parts[0].wm_z_state is not None:
            snap["wm_z_state"] = np.stack(
                [p.wm_z_state.squeeze(0).cpu().numpy() for p in parts])
        if parts[0].wm_x_state is not None:
            snap["wm_x_state"] = np.stack(
                [p.wm_x_state.squeeze(0).cpu().numpy() for p in parts])
        if parts[0].q_z_state is not None:
            snap["q_z_state"]  = np.stack(
                [p.q_z_state.squeeze(0).cpu().numpy()  for p in parts])
        if parts[0].q_x_state is not None:
            snap["q_x_state"]  = np.stack(
                [p.q_x_state.squeeze(0).cpu().numpy()  for p in parts])
        return snap

    raise TypeError(f"Unknown state type: {type(state)}")


# ---------------------------------------------------------------
# Per-trajectory inference + numpy storage
# ---------------------------------------------------------------

@torch.no_grad()
def run_and_save_inference_states(
    *,
    wm,
    proposal,
    artifact,
    engine_name: str,
    engine_cfg: Dict[str, Any],
    out_dir: Path,
    inference_seed: int,
    device: torch.device,
    dtype: torch.dtype,
    overwrite: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """
    Runs inference on every trajectory in `artifact` one at a time (B=1).
    Saves compact numpy arrays at every timestep to a .npz file per trajectory.

    Directory layout:
        out_dir/
            traj_0000.npz
            traj_0001.npz
            ...
            run_meta.json

    Each traj_XXXX.npz contains:
        z        : float32 [T, K, dz]   hypothesis positions
        J        : float32 [T, K]       joint score        (evitrack only)
        E        : float32 [T, K]       evidence score     (evitrack only)
        J_tbd    : float32 [T, K]       tbd score          (evitrack only)
        logw     : float32 [T, K]       log weights        (particle only)
        type_id  : int8    scalar       0=evitrack 1=particle
        traj_index: int64  scalar       join key into dataset artifact
        # optional NonMarkov fields:
        wm_z_state, wm_x_state, q_z_state, q_x_state : float32 [T, K, h_dim]

    Loading at replay time:
        data = np.load("traj_0000.npz")
        z    = torch.from_numpy(data["z"])     # [T, K, dz]
        E    = torch.from_numpy(data["E"])     # [T, K]

    Ground truth (x, z, delayed_flag, disamb_time) lives in the dataset artifact
    and is accessed via traj_index — never duplicated here.
    """
    _seed_everything(inference_seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wm.eval()
    if proposal is not None:
        proposal.eval()

    N = artifact.x.shape[0]
    saved_paths: List[Path] = []

    for i in range(N):
        out_path = out_dir / f"traj_{i:04d}.npz"

        if out_path.exists() and not overwrite:
            if verbose:
                print(f"  [skip] {out_path.name} already exists")
            saved_paths.append(out_path)
            continue

        x_i = artifact.x[i].to(device=device, dtype=dtype)   # [T, dx]
        T   = x_i.shape[0]

        engine = _engine_from_spec(engine_name, wm=wm, proposal=proposal, cfg_dict=engine_cfg)
        state  = engine.init_state(B=1, device=str(device), dtype=dtype)

        snapshots: List[Dict[str, np.ndarray]] = []
        for t in range(T):
            state, _ = engine.step(state, x_i[t].unsqueeze(0))
            snapshots.append(_extract_state_snapshot(state))

        # Stack list-of-dicts -> dict-of-arrays, each key: [T, K, ...]
        array_keys = [k for k in snapshots[0] if k != "type_id"]
        stacked: Dict[str, np.ndarray] = {
            "type_id":   snapshots[0]["type_id"],
            "traj_index": np.array(i, dtype=np.int64),
        }
        for k in array_keys:
            stacked[k] = np.stack([s[k] for s in snapshots])  # [T, K, ...]

        np.savez_compressed(out_path, **stacked)
        saved_paths.append(out_path)

        if verbose:
            flag_str = "delayed" if artifact.delayed_flag[i].item() else "non-delayed"
            dt = int(artifact.disamb_time[i].item())
            print(f"  [{i+1:03d}/{N}] {out_path.name}  ({flag_str}, disamb_t={dt})")

    # Run-level metadata
    run_meta = {
        "engine_name":     engine_name,
        "engine_cfg":      engine_cfg,
        "inference_seed":  inference_seed,
        "N":               N,
        "num_delayed":     int(artifact.delayed_flag.sum().item()),
        "num_non_delayed": int((~artifact.delayed_flag).sum().item()),
    }
    with (out_dir / "run_meta.json").open("w") as f:
        json.dump(run_meta, f, indent=2)

    return saved_paths
