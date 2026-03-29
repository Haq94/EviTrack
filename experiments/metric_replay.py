# experiments/metric_replay.py
# TODO: V&V needed
"""
Offline metric replay for EviTrack inference results.

Loads saved .npz inference snapshots, joins with ground-truth from a
DoubleWell1DDatasetArtifact, performs H-step Monte Carlo rollouts through the
world model, and computes predictive metrics (PLL, MSE, branch accuracy)
without re-running inference.

Weight modes for EviTrack hypotheses ("evidence", "joint", "tbd") can be
changed at replay time, enabling ablations without new inference runs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.dataset_io import load_dataset


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

# @dataclass
# class ReplayConfig:
#     results_dir:       str
#     dataset_path:      str
#     engines:           List[str]
#     horizons:          List[int]
#     n_rollout_samples: int
#     device:            str        = "cpu"
#     dtype:             torch.dtype = torch.float32
#     save_dir:          str        = ""
#     verbose:           bool       = True


@dataclass
class ReplayConfig:
    results_dir:       str
    dataset_path:      str
    engines:           List[str]
    horizons:          List[int]
    n_rollout_samples: int
    device:            str         = "cpu"
    dtype:             torch.dtype = torch.float32
    save_dir:          str         = ""
    verbose:           bool        = True
    rollout_mode:      str         = "autoregressive"  # "autoregressive" | "frozen"


# ---------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------

def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over last axis."""
    c = logits.max(axis=-1, keepdims=True)
    shifted = logits - c
    lse = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - lse


def compute_weights_evitrack(
    npz: Dict[str, np.ndarray],
    weight_mode: str,
) -> np.ndarray:
    """
    Derive normalized log-weights from saved EviTrack scores.

    Args:
        npz: loaded .npz dict with keys "E", "J", "J_tbd" each [T, K].
        weight_mode: one of "evidence", "joint", "tbd".

    Returns:
        log_weights: float64 array [T, K], log-sum-exp normalized per row.
    """
    if weight_mode == "evidence":
        scores = npz["E"].astype(np.float64)
    elif weight_mode == "joint":
        scores = npz["J"].astype(np.float64)
    elif weight_mode == "tbd_joint":
        scores = npz["J_tbd"].astype(np.float64)
    else:
        raise ValueError(
            f"Unknown weight_mode '{weight_mode}' for evitrack. "
            "Expected 'evidence', 'joint', or 'tbd'."
        )
    return _log_softmax(scores)  # [T, K]


def compute_weights_particle(npz: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Derive normalized log-weights from saved particle log-weights.

    Args:
        npz: loaded .npz dict with key "logw" [T, K].

    Returns:
        log_weights: float64 array [T, K], log-sum-exp normalized per row.
    """
    logw = npz["logw"].astype(np.float64)
    return _log_softmax(logw)  # [T, K]


def compute_log_weights(
    npz: Dict[str, np.ndarray],
    engine_cfg: Dict[str, Any],
) -> np.ndarray:
    type_id = int(npz["type_id"])
    if type_id == 1:
        return compute_weights_particle(npz)
    # EviTrack / RandomBeam — read weight_mode from saved engine_cfg
    weight_mode = engine_cfg["weight_mode"]
    return compute_weights_evitrack(npz, weight_mode)


# ---------------------------------------------------------------
# H-step rollout via world model
# ---------------------------------------------------------------

@torch.no_grad()
def rollout_h_steps(
    wm: torch.nn.Module,
    z_start: torch.Tensor,              # [K, dz]
    H_max: int,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    z_state_start: Optional[torch.Tensor] = None,   # [K, h_dim] or None
    x_state_start: Optional[torch.Tensor] = None,   # [K, h_dim] or None
    rollout_mode: str = "autoregressive",            # "autoregressive" | "frozen"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Roll forward H_max steps from K hypothesis positions,
    drawing n_samples independent continuations per hypothesis.

    autoregressive: x_state updates at each step using sampled x
    frozen:         x_state held fixed (old behavior, only correct when x_mode="none")

    Returns:
        z_rollout:   [K, n_samples, H_max, dz]
        emit_mu:     [K, n_samples, H_max, dx]
        emit_logstd: [K, n_samples, H_max, dx]
    """
    assert rollout_mode in ("autoregressive", "frozen"), \
        f"rollout_mode must be 'autoregressive' or 'frozen', got '{rollout_mode}'"

    K, dz = z_start.shape
    KM = K * n_samples

    # Tile each hypothesis n_samples times: [K, dz] -> [K*M, dz]
    def _tile(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        # t: [K, ...] -> [K*M, ...]
        trailing = t.shape[1:]
        return t.unsqueeze(1).expand(K, n_samples, *trailing).reshape(KM, *trailing).contiguous()

    z_prev   = _tile(z_start)
    z_state  = _tile(z_state_start)
    x_state  = _tile(x_state_start)

    z_list        = []
    emit_mu_list  = []
    emit_logstd_list = []

    for h in range(H_max):
        # 1. Transition: sample z_{t+h+1}
        trans_params  = wm.transition_params(z_prev=z_prev, z_state_prev=z_state)
        z_new         = wm.sample_transition(trans_params)          # [KM, dz]

        # 2. Current z-state (Markov: z_new, NonMarkov: GRU update)
        z_state_curr  = wm.z_state_curr(z_state, z_new)

        # 3. Emission params at z_new
        emit_params   = wm.emission_params(
            z_state_curr=z_state_curr,
            x_state_prev=x_state,
        )

        # 4. Sample observation (needed for autoregressive x_state update)
        x_new = wm.sample_emission(emit_params)                     # [KM, dx]

        # 5. Update states
        z_state = wm.update_z_state(z_state, z_new)

        if rollout_mode == "autoregressive":
            x_state = wm.update_x_state(x_state, x_new)
        # frozen: x_state unchanged

        # 6. Store
        z_list.append(z_new)
        emit_mu_list.append(emit_params.mu)
        emit_logstd_list.append(emit_params.logstd)

        # Advance
        z_prev = z_new

    # Stack: [KM, H_max, dim] -> [K, M, H_max, dim]
    dx = emit_mu_list[0].shape[-1]
    z_roll    = torch.stack(z_list,         dim=1).reshape(K, n_samples, H_max, dz)
    e_mu      = torch.stack(emit_mu_list,   dim=1).reshape(K, n_samples, H_max, dx)
    e_logstd  = torch.stack(emit_logstd_list, dim=1).reshape(K, n_samples, H_max, dx)

    # Sanity check in autoregressive mode: sampled x's should differ across samples
    # (will be identical only if emission variance is 0, which shouldn't happen)
    if rollout_mode == "autoregressive" and x_state_start is not None:
        # x_state should have changed if x_mode != "none"
        if x_state is not None:
            assert not torch.allclose(
                x_state.reshape(K, n_samples, -1)[:, 0],
                x_state.reshape(K, n_samples, -1)[:, 1],
            ), "x_state identical across samples — check update_x_state is working"

    return z_roll, e_mu, e_logstd


# ---------------------------------------------------------------
# Per-timestep metric computation
# ---------------------------------------------------------------

def _gaussian_diag_logprob_np(
    x: np.ndarray,
    mu: np.ndarray,
    logstd: np.ndarray,
) -> np.ndarray:
    """
    Log N(x | mu, diag(exp(2*logstd))).

    x:      [dx]
    mu:     [..., dx]
    logstd: [..., dx]

    Returns: [...] scalar log-probs.
    """
    D = x.shape[-1]
    inv_std = np.exp(-logstd)
    z = (x - mu) * inv_std
    return -0.5 * (
        np.sum(z * z, axis=-1)
        + 2.0 * np.sum(logstd, axis=-1)
        + D * math.log(2.0 * math.pi)
    )


def compute_metrics_at_t(
    *,
    log_weights_t: np.ndarray,       # [K] log-normalized
    emit_mu: np.ndarray,             # [K, M, H_max, dx]
    emit_logstd: np.ndarray,         # [K, M, H_max, dx]
    z_rollout: np.ndarray,           # [K, M, H_max, dz]
    x_true_future: np.ndarray,       # [H_max, dx]
    z_true_future: np.ndarray,       # [H_max, dz]
    horizons: List[int],
    H_max: int,
    coverage_levels: List[float] = (0.5, 0.9),
) -> Dict[str, np.ndarray]:
    """
    Compute PLL, MSE, branch accuracy, and coverage for a single timestep t.

    Returns dict with:
        "pll":              [num_horizons]
        "mse":              [num_horizons]
        "branch_acc_per_h": [num_horizons]
        "coverage_50":      [num_horizons]   (if 0.5 in coverage_levels)
        "coverage_90":      [num_horizons]   (if 0.9 in coverage_levels)
    """
    K, M = emit_mu.shape[0], emit_mu.shape[1]
    weights_t = np.exp(log_weights_t)   # [K] normalized

    num_horizons = len(horizons)
    pll        = np.full(num_horizons, np.nan)
    mse        = np.full(num_horizons, np.nan)
    branch_acc = np.full(num_horizons, np.nan)

    # coverage dict: level -> array [num_horizons]
    coverages = {lvl: np.full(num_horizons, np.nan) for lvl in coverage_levels}

    for hi, H in enumerate(horizons):
        if H > H_max or H < 1:
            continue
        h_idx = H - 1

        x_target = x_true_future[h_idx]    # [dx]
        z_target = z_true_future[h_idx]    # [dz]

        mu_km     = emit_mu[:, :, h_idx, :]      # [K, M, dx]
        logstd_km = emit_logstd[:, :, h_idx, :]  # [K, M, dx]

        # ---- PLL ----
        logp_km = _gaussian_diag_logprob_np(x_target, mu_km, logstd_km)   # [K, M]
        max_m   = logp_km.max(axis=1, keepdims=True)
        per_hyp_logp = (
            np.log(np.sum(np.exp(logp_km - max_m), axis=1))
            + max_m.squeeze(1)
            - math.log(M)
        )   # [K]
        log_mix = log_weights_t + per_hyp_logp   # [K]
        max_k   = log_mix.max()
        pll[hi] = max_k + math.log(np.sum(np.exp(log_mix - max_k)))

        # ---- MSE ----
        mean_per_hyp = mu_km.mean(axis=1)                              # [K, dx]
        pred_mean    = np.sum(weights_t[:, None] * mean_per_hyp, axis=0)  # [dx]
        mse[hi]      = float(np.sum((pred_mean - x_target) ** 2))

        # ---- Branch accuracy ----
        z_target_sign = np.sign(z_target[0])
        if z_target_sign == 0.0:
            branch_acc[hi] = 1.0
        else:
            z_roll_h = z_rollout[:, :, h_idx, 0]                          # [K, M]
            correct_per_hyp = np.mean(
                np.sign(z_roll_h) == z_target_sign, axis=1)               # [K]
            branch_acc[hi] = float(np.sum(weights_t * correct_per_hyp))

        # ---- Coverage ----
        # Build weighted sample set: K*M samples, each with weight w_k / M
        # x samples: sample from emission N(mu_km, sigma_km)
        # For coverage we use actual emission samples, not just the mean
        std_km   = np.exp(logstd_km)                        # [K, M, dx]
        eps      = np.random.randn(*std_km.shape)           # [K, M, dx]
        x_samples = mu_km + std_km * eps                    # [K, M, dx]

        # Flatten to [K*M, dx] with weights [K*M]
        sample_weights = np.repeat(weights_t, M) / M        # [K*M] sums to 1
        x_flat = x_samples.reshape(K * M, -1)               # [K*M, dx]

        # For dx=1: compute weighted quantiles per dimension
        for lvl in coverage_levels:
            alpha  = (1.0 - lvl) / 2.0     # e.g. 0.05 for 90%
            # Weighted quantile via sorted cumulative weight
            covered = True
            for d in range(x_flat.shape[1]):
                vals = x_flat[:, d]
                sort_idx = np.argsort(vals)
                sorted_vals    = vals[sort_idx]
                sorted_weights = sample_weights[sort_idx]
                cumw = np.cumsum(sorted_weights)

                lower_idx = np.searchsorted(cumw, alpha)
                upper_idx = np.searchsorted(cumw, 1.0 - alpha)

                lower = sorted_vals[lower_idx] if lower_idx < len(sorted_vals) else sorted_vals[-1]
                upper = sorted_vals[upper_idx] if upper_idx < len(sorted_vals) else sorted_vals[-1]

                if not (lower <= x_target[d] <= upper):
                    covered = False
                    break

            coverages[lvl][hi] = float(covered)

    result = {
        "pll":              pll,
        "mse":              mse,
        "branch_acc_per_h": branch_acc,
    }
    for lvl in coverage_levels:
        key = f"coverage_{int(lvl * 100)}"
        result[key] = coverages[lvl]

    return result


# ---------------------------------------------------------------
# Single trajectory replay
# ---------------------------------------------------------------

@torch.no_grad()
def replay_single_trajectory(
    *,
    npz_path: Path,
    dataset: Dict[str, Any],
    wm: torch.nn.Module,
    engine_cfg: Dict[str, Any],
    horizons: List[int],
    n_rollout_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    rollout_mode: str = "autoregressive",   # NEW
) -> Dict[str, np.ndarray]:
    data     = dict(np.load(npz_path, allow_pickle=False))
    traj_idx = int(data["traj_index"])

    z_hyps   = data["z"]           # [T, K, dz]
    T, K, dz = z_hyps.shape

    x_true = dataset["x"][traj_idx].cpu().numpy()   # [T, dx]
    z_true = dataset["z"][traj_idx].cpu().numpy()   # [T, dz]

    log_w = compute_log_weights(data, engine_cfg)

    H_max        = max(horizons)
    num_horizons = len(horizons)

    pll_all         = np.full((T, num_horizons), np.nan, dtype=np.float64)
    mse_all         = np.full((T, num_horizons), np.nan, dtype=np.float64)
    branch_acc_all  = np.full((T, num_horizons), np.nan, dtype=np.float64)
    coverage_50_all = np.full((T, num_horizons), np.nan, dtype=np.float64)
    coverage_90_all = np.full((T, num_horizons), np.nan, dtype=np.float64)

    p_positive_all  = np.full((T, num_horizons), np.nan, dtype=np.float64)  # P(Z_{t+H} > 0)
    p_negative_all  = np.full((T, num_horizons), np.nan, dtype=np.float64)  # P(Z_{t+H} < 0)
    ess_all         = np.full(T, np.nan, dtype=np.float64)                  # ESS at each t

    has_wm_z_state = "wm_z_state" in data
    has_wm_x_state = "wm_x_state" in data

    for t in range(T):
        # Compute ESS from weights at this timestep
        weights_t = np.exp(log_w[t])  # [K]
        weights_t = weights_t / (np.sum(weights_t) + 1e-12)  # Normalize
        ess_all[t] = 1.0 / (np.sum(weights_t ** 2) + 1e-12)

        remaining = T - t - 1
        if remaining < 1:
            break

        effective_H_max = min(H_max, remaining)
        z_t = torch.from_numpy(z_hyps[t]).to(device=device, dtype=dtype)

        z_state_t = None
        if has_wm_z_state:
            z_state_t = torch.from_numpy(
                data["wm_z_state"][t]).to(device=device, dtype=dtype)

        x_state_t = None
        if has_wm_x_state:
            x_state_t = torch.from_numpy(
                data["wm_x_state"][t]).to(device=device, dtype=dtype)

        z_roll, e_mu, e_logstd = rollout_h_steps(
            wm=wm,
            z_start=z_t,
            H_max=effective_H_max,
            n_samples=n_rollout_samples,
            device=device,
            dtype=dtype,
            z_state_start=z_state_t,
            x_state_start=x_state_t,
            rollout_mode=rollout_mode,
        )

        z_roll_np   = z_roll.cpu().numpy()
        e_mu_np     = e_mu.cpu().numpy()
        e_logstd_np = e_logstd.cpu().numpy()

        x_future = x_true[t + 1 : t + 1 + effective_H_max]
        z_future = z_true[t + 1 : t + 1 + effective_H_max]

        valid_horizons = [h for h in horizons if h <= effective_H_max]
        if not valid_horizons:
            continue

        metrics = compute_metrics_at_t(
            log_weights_t=log_w[t],
            emit_mu=e_mu_np,
            emit_logstd=e_logstd_np,
            z_rollout=z_roll_np,
            x_true_future=x_future,
            z_true_future=z_future,
            horizons=valid_horizons,
            H_max=effective_H_max,
        )

        for vi, h in enumerate(valid_horizons):
            full_idx = horizons.index(h)
            pll_all[t, full_idx]         = metrics["pll"][vi]
            mse_all[t, full_idx]         = metrics["mse"][vi]
            branch_acc_all[t, full_idx]  = metrics["branch_acc_per_h"][vi]
            coverage_50_all[t, full_idx] = metrics["coverage_50"][vi]   # NEW
            coverage_90_all[t, full_idx] = metrics["coverage_90"][vi]   # NEW

            # NEW: Compute branch probabilities
            # z_roll_np is [K, M, H_max, dz]
            h_idx = h - 1  # 0-indexed
            if h_idx < z_roll_np.shape[2]:
                z_roll_h = z_roll_np[:, :, h_idx, 0]  # [K, M] - position at horizon h

                # For each hypothesis k, what fraction of samples are positive?
                frac_pos_per_hyp = np.mean(z_roll_h > 0, axis=1)  # [K]
                frac_neg_per_hyp = np.mean(z_roll_h < 0, axis=1)  # [K]

                # Weight by hypothesis weights
                weights_t_norm = np.exp(log_w[t])
                weights_t_norm = weights_t_norm / (np.sum(weights_t_norm) + 1e-12)

                p_pos = np.sum(weights_t_norm * frac_pos_per_hyp)
                p_neg = np.sum(weights_t_norm * frac_neg_per_hyp)

                p_positive_all[t, full_idx] = p_pos
                p_negative_all[t, full_idx] = p_neg

    return {
            "pll":         pll_all,
            "mse":         mse_all,
            "branch_acc":  branch_acc_all,
            "coverage_50": coverage_50_all,
            "coverage_90": coverage_90_all,
            "p_positive":  p_positive_all,    # NEW
            "p_negative":  p_negative_all,    # NEW
            "ess":         ess_all,           # NEW
            "traj_index":  traj_idx,
        }


# ---------------------------------------------------------------
# Directory traversal helpers
# ---------------------------------------------------------------

def _discover_engine_dirs(results_dir: Path, engine_name: str) -> List[Path]:
    """Find all inference_seed directories under results_dir/engine_name."""
    engine_dir = results_dir / engine_name
    if not engine_dir.is_dir():
        return []

    return sorted([
        d for d in engine_dir.iterdir()
        if d.is_dir() and d.name.startswith("inference_seed_")
    ])


def _list_traj_npz(seed_dir: Path) -> List[Path]:
    """List all traj_XXXX.npz files in a seed directory, sorted."""
    paths = sorted(seed_dir.glob("traj_*.npz"))
    return paths


def _load_run_meta(seed_dir: Path) -> Dict[str, Any]:
    """Load run_meta.json from a seed directory."""
    meta_path = seed_dir / "run_meta.json"
    if meta_path.exists():
        with meta_path.open("r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------
# Aggregation for one engine + k_tag
# ---------------------------------------------------------------

def _replay_engine(
    *,
    seed_dirs: List[Path],
    dataset: Dict[str, Any],
    wm: torch.nn.Module,
    engine_cfg: Dict[str, Any],
    horizons: List[int],
    n_rollout_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    rollout_mode: str = "autoregressive",   # NEW
    verbose: bool = True,
) -> Dict[str, np.ndarray]:

    x = dataset["x"]
    z = dataset["z"]
    N, T, dx = x.shape
    num_h = len(horizons)

    pll_accum         = []
    mse_accum         = []
    branch_acc_accum  = []
    coverage_50_accum = []
    coverage_90_accum = []
    p_positive_accum  = []  # NEW
    p_negative_accum  = []  # NEW
    ess_accum         = []  # NEW
    traj_indices      = []
    seed_indices      = []

    for seed_idx, seed_dir in enumerate(seed_dirs):
        traj_files = _list_traj_npz(seed_dir)

        for traj_file in traj_files:
            data = dict(np.load(traj_file, allow_pickle=False))
            traj_idx = int(data["traj_index"])

            # x_i = x[traj_idx].numpy()
            # z_i = z[traj_idx].numpy()

            result = replay_single_trajectory(
                npz_path=traj_file,
                dataset=dataset,
                wm=wm,
                engine_cfg=engine_cfg,
                horizons=horizons,
                n_rollout_samples=n_rollout_samples,
                device=device,
                dtype=dtype,
                rollout_mode=rollout_mode,
            )

            pll_accum.append(result["pll"])
            mse_accum.append(result["mse"])
            branch_acc_accum.append(result["branch_acc"])
            coverage_50_accum.append(result["coverage_50"])
            coverage_90_accum.append(result["coverage_90"])
            p_positive_accum.append(result["p_positive"])  # NEW
            p_negative_accum.append(result["p_negative"])  # NEW
            ess_accum.append(result["ess"])                # NEW
            traj_indices.append(traj_idx)
            seed_indices.append(seed_idx)

    if not pll_accum:
        return None

    return {
        "pll":         np.stack(pll_accum, axis=0),
        "mse":         np.stack(mse_accum, axis=0),
        "branch_acc":  np.stack(branch_acc_accum, axis=0),
        "coverage_50": np.stack(coverage_50_accum, axis=0),
        "coverage_90": np.stack(coverage_90_accum, axis=0),
        "p_positive":  np.stack(p_positive_accum, axis=0),  # NEW
        "p_negative":  np.stack(p_negative_accum, axis=0),  # NEW
        "ess":         np.stack(ess_accum, axis=0),         # NEW
        "traj_index":  np.array(traj_indices, dtype=np.int64),
        "seed_index":  np.array(seed_indices, dtype=np.int64),
    }


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def run_metric_replay(cfg: ReplayConfig, wm: torch.nn.Module) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    dtype = cfg.dtype
    results_dir = Path(cfg.results_dir)
    save_dir = Path(cfg.save_dir) if cfg.save_dir else results_dir / "replay"
    save_dir.mkdir(parents=True, exist_ok=True)

    horizons = sorted(cfg.horizons)

    if cfg.verbose:
        print(f"[replay] Loading dataset from {cfg.dataset_path}")

    # CHANGED: load plain dict instead of DoubleWell1DDatasetArtifact
    dataset = load_dataset(cfg.dataset_path)
    N = dataset["x"].shape[0]
    T = dataset["x"].shape[1]
    delayed_flag = dataset["delayed_flag"].numpy()
    disamb_time  = dataset["disamb_time"].numpy()

    wm.eval()

    summary: Dict[str, Any] = {
        "results_dir":      str(results_dir),
        "dataset_path":     cfg.dataset_path,
        "horizons":         horizons,
        "n_rollout_samples": cfg.n_rollout_samples,
        "N": N,
        "T": T,
        "saved_files": [],
    }

    for engine_name in cfg.engines:
        if cfg.verbose:
            print(f"\n[replay] Engine: {engine_name}")

        seed_dirs = _discover_engine_dirs(results_dir, engine_name)
        if not seed_dirs:
            if cfg.verbose:
                print(f"  No directories found for {engine_name}, skipping.")
            continue

        if cfg.verbose:
            print(f"  [{engine_name}] {len(seed_dirs)} seed dir(s)")

        run_meta   = _load_run_meta(seed_dirs[0])
        engine_cfg = run_meta.get("engine_cfg", {})

        agg = _replay_engine(
            seed_dirs=seed_dirs,
            dataset=dataset,
            wm=wm,
            engine_cfg=engine_cfg,
            horizons=horizons,
            n_rollout_samples=cfg.n_rollout_samples,
            device=device,
            dtype=dtype,
            rollout_mode=cfg.rollout_mode,
            verbose=cfg.verbose,
        )

        if not agg:
            if cfg.verbose:
                print(f"  No trajectories found, skipping.")
            continue

        traj_indices          = agg["traj_index"]
        agg["delayed_flag"]   = delayed_flag[traj_indices]
        agg["disamb_time"]    = disamb_time[traj_indices]

        # CHANGED: output filename uses engine_name only, no k_tag or weight_mode
        out_name  = f"{engine_name}.npz"
        out_path  = save_dir / out_name

        # CHANGED: weight_mode read from engine_cfg, not cfg
        weight_mode_saved = engine_cfg.get("weight_mode", "unknown")

        np.savez_compressed(
            out_path,
            pll=agg["pll"],
            mse=agg["mse"],
            branch_acc=agg["branch_acc"],
            coverage_50=agg["coverage_50"],
            coverage_90=agg["coverage_90"],
            p_positive=agg["p_positive"],    # NEW
            p_negative=agg["p_negative"],    # NEW
            ess=agg["ess"],                  # NEW
            traj_index=agg["traj_index"],
            seed_index=agg["seed_index"],
            delayed_flag=agg["delayed_flag"],
            disamb_time=agg["disamb_time"],
            horizons=np.array(horizons, dtype=np.int64),
            T=np.array(T, dtype=np.int64),
            engine_name=np.array(engine_name),
            engine_cfg=np.array(json.dumps(engine_cfg)),
            weight_mode=np.array(weight_mode_saved),
            n_rollout_samples=np.array(cfg.n_rollout_samples, dtype=np.int64),
        )

        n_total       = agg["pll"].shape[0]
        n_delayed_cnt = int(agg["delayed_flag"].sum())
        n_non_delayed = n_total - n_delayed_cnt

        if cfg.verbose:
            valid_mask = ~np.isnan(agg["pll"])
            mean_pll = np.nanmean(agg["pll"]) if valid_mask.any() else float("nan")
            mean_mse = np.nanmean(agg["mse"]) if valid_mask.any() else float("nan")
            mean_ba  = np.nanmean(agg["branch_acc"]) if valid_mask.any() else float("nan")
            print(f"  Saved: {out_path}")
            print(f"  Trajectories: {n_total} ({n_delayed_cnt} delayed, {n_non_delayed} non-delayed)")
            print(f"  Mean PLL: {mean_pll:.4f}  Mean MSE: {mean_mse:.6f}  Mean BA: {mean_ba:.4f}")

        summary["saved_files"].append({
            "engine_name": engine_name,
            "path":        str(out_path),
            "n_total":     n_total,
            "n_delayed":   n_delayed_cnt,
            "n_non_delayed": n_non_delayed,
        })

    summary_path = save_dir / "replay_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    if cfg.verbose:
        print(f"\n[replay] Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------
# Convenience: loading and stratifying results
# ---------------------------------------------------------------

def load_replay_results(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a saved .npz replay result file."""
    data = dict(np.load(path, allow_pickle=True))
    # Decode string arrays
    for key in ("engine_name", "engine_cfg", "weight_mode"):
        if key in data:
            data[key] = str(data[key])
    return data


def stratify_by_delayed(
    results: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split replay results into delayed and non-delayed subsets.

    Returns:
        (delayed_results, non_delayed_results) — each a dict with
        pll, mse, branch_acc arrays sliced to the corresponding subset.
    """
    mask_d = results["delayed_flag"].astype(bool)
    mask_nd = ~mask_d

    def _slice(m):
        return {
            "pll": results["pll"][m],
            "mse": results["mse"][m],
            "branch_acc": results["branch_acc"][m],
            "traj_index": results["traj_index"][m],
            "disamb_time": results["disamb_time"][m],
            "horizons": results["horizons"],
        }

    return _slice(mask_d), _slice(mask_nd)


def stratify_by_disamb_bins(
    results: Dict[str, np.ndarray],
    bin_edges: Optional[List[int]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Bin replay results by disamb_time.

    Args:
        results: loaded replay results.
        bin_edges: list of bin boundaries. Default: [0, 20, 40, 60, 80, 100, 120].
            Each bin is [lo, hi).  A final bin captures disamb_time == -1 (never).

    Returns:
        Dict mapping bin label -> sliced results.
    """
    if bin_edges is None:
        bin_edges = [0, 20, 40, 60, 80, 100, 120]

    dt = results["disamb_time"]
    out = {}

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (dt >= lo) & (dt < hi)
        label = f"disamb_{lo}_{hi}"
        if mask.any():
            out[label] = {
                "pll": results["pll"][mask],
                "mse": results["mse"][mask],
                "branch_acc": results["branch_acc"][mask],
                "traj_index": results["traj_index"][mask],
                "disamb_time": dt[mask],
                "horizons": results["horizons"],
            }

    # Never-disambiguated bin
    mask_never = dt == -1
    if mask_never.any():
        out["disamb_never"] = {
            "pll": results["pll"][mask_never],
            "mse": results["mse"][mask_never],
            "branch_acc": results["branch_acc"][mask_never],
            "traj_index": results["traj_index"][mask_never],
            "disamb_time": dt[mask_never],
            "horizons": results["horizons"],
        }

    return out
