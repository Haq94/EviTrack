# data/synthetic_tasks/doublewell_1d_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np
import torch

from data.synthetic_generator import generate_sequences
from data.synthetic_tasks.doublewell_1d import (
    make_prior,
    make_transition,
    make_emission,
)


@dataclass
class DoubleWell1DDatasetArtifact:
    x: torch.Tensor                 # [N, T, dx]
    z: torch.Tensor                 # [N, T, dz]
    data_seed_ids: torch.Tensor     # [N]
    delayed_flag: torch.Tensor      # [N] bool
    disamb_time: torch.Tensor       # [N] int64, -1 if none
    meta: Dict[str, Any]

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "x": self.x.cpu(),
                "z": self.z.cpu(),
                "data_seed_ids": self.data_seed_ids.cpu(),
                "delayed_flag": self.delayed_flag.cpu(),
                "disamb_time": self.disamb_time.cpu(),
            },
            root / "data.pt",
        )

        with (root / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, sort_keys=True)

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "DoubleWell1DDatasetArtifact":
        root = Path(root)
        payload = torch.load(root / "data.pt", map_location=map_location)

        with (root / "metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        return cls(
            x=payload["x"],
            z=payload["z"],
            data_seed_ids=payload["data_seed_ids"],
            delayed_flag=payload["delayed_flag"],
            disamb_time=payload["disamb_time"],
            meta=meta,
        )


def _transition_mean_np(z: np.ndarray, *, a: float, V: float, dt: float) -> np.ndarray:
    dU = (4.0 * V / (a ** 4)) * z * (z ** 2 - a ** 2)
    return z - dU * dt


def _emission_mean_np(z: np.ndarray, *, d: float, n: int) -> np.ndarray:
    z = np.asarray(z)
    inside = np.abs(z) <= d
    out = np.empty_like(z, dtype=float)
    out[inside] = z[inside] ** (2 * n)
    out[~inside] = z[~inside]
    return out


def estimate_disambiguation_time_quadrature(
    x_obs: torch.Tensor,   # [T, 1] or [T]
    *,
    T: int,
    a: float,
    V: float,
    dt: float,
    sigma_z: float,
    d: float,
    n: int,
    sigma_x: float,
    z0_std: float,
    threshold: float = 0.8,
    zmin: float = -4.0,
    zmax: float = 4.0,
    G: int = 1000,
) -> int:
    """
    Returns the first time index t in {1,...,T} such that

        | P(z_t > 0 | x_<t) - P(z_t < 0 | x_<t) | > threshold

    using predictive quadrature. Returns -1 if no crossing occurs.
    """
    x_obs_np = x_obs.detach().cpu().numpy().reshape(T)

    z_grid = np.linspace(zmin, zmax, G)
    dz = z_grid[1] - z_grid[0]

    # Prior p(z1)
    p_pred = np.zeros((T, G), dtype=float)
    p_filt = np.zeros((T, G), dtype=float)

    p0 = (1.0 / (np.sqrt(2.0 * np.pi) * z0_std)) * np.exp(-0.5 * (z_grid / z0_std) ** 2)
    p0 = p0 / (np.sum(p0) * dz + 1e-12)
    p_pred[0] = p0

    # Transition kernel K[i,j] = p(z_next=z_i | z=z_j)
    mu_zj = _transition_mean_np(z_grid, a=a, V=V, dt=dt)
    var_z = sigma_z ** 2
    diff = z_grid[:, None] - mu_zj[None, :]
    K = (1.0 / np.sqrt(2.0 * np.pi * var_z)) * np.exp(-0.5 * (diff ** 2) / var_z)

    # Emission mean on z-grid
    mu_x_grid = _emission_mean_np(z_grid, d=d, n=n)
    var_x = sigma_x ** 2

    # Filtering / predictive recursion
    for t in range(T):
        xt = x_obs_np[t]

        logL = -0.5 * ((xt - mu_x_grid) ** 2) / var_x - 0.5 * np.log(2.0 * np.pi * var_x)
        logL -= np.max(logL)
        L = np.exp(logL)

        pf = L * p_pred[t]
        pf = pf / (np.sum(pf) * dz + 1e-12)
        p_filt[t] = pf

        if t < T - 1:
            pn = K @ (pf * dz)
            pn = pn / (np.sum(pn) * dz + 1e-12)
            p_pred[t + 1] = pn

    pos_mask = z_grid >= 0
    neg_mask = z_grid < 0

    for t in range(T):
        pt = p_pred[t]
        m_pos = np.sum(pt[pos_mask]) * dz
        m_neg = np.sum(pt[neg_mask]) * dz
        mass_diff = m_pos - m_neg
        if abs(mass_diff) > threshold:
            return t + 1  # 1-based time index

    return -1


def build_doublewell_1d_dataset(
    *,
    T: int,
    n_delayed: int,
    n_non_delayed: int,
    search_seed_start: int = 0,
    max_seed_search: int = 100000,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    # double-well params
    a: float = 3.0,
    V: float = 0.06,
    dt: float = 1.0,
    sigma_z: float = 0.05,
    d: float = 2.0,
    n: int = 1,
    sigma_x: float = 0.12,
    z0_mean: float = 0.0,
    z0_std: float = 1.0,
    # delayed-disambiguation annotation
    threshold: float = 0.8,
    zmin: float = -4.0,
    zmax: float = 4.0,
    G: int = 1000,
    verbose: bool = True,
) -> DoubleWell1DDatasetArtifact:
    """
    Searches over data seeds until it collects:
      - n_delayed delayed-disambiguation examples
      - n_non_delayed non-delayed examples

    Returns one combined artifact with delayed examples first, then non-delayed.
    """
    if n_delayed < 0 or n_non_delayed < 0:
        raise ValueError("n_delayed and n_non_delayed must be >= 0")
    if n_delayed == 0 and n_non_delayed == 0:
        raise ValueError("At least one of n_delayed or n_non_delayed must be > 0")

    prior = make_prior(z0_mean=z0_mean, z0_std=z0_std)
    transition = make_transition(a=a, V=V, dt=dt, sigma_z=sigma_z)
    emission = make_emission(d=d, n=n, sigma_x=sigma_x)

    delayed_examples = []
    non_delayed_examples = []

    delayed_seeds = []
    non_delayed_seeds = []

    delayed_times = []
    non_delayed_times = []

    seeds_checked = 0
    seed = int(search_seed_start)

    while seeds_checked < max_seed_search:
        if len(delayed_examples) >= n_delayed and len(non_delayed_examples) >= n_non_delayed:
            break

        batch = generate_sequences(
            T=T,
            B=1,
            prior=prior,
            transition=transition,
            emission=emission,
            seed=seed,
            return_logp=False,
            extras=None,
            device=device,
            dtype=dtype,
        )

        x_i = batch.x[0].detach().cpu()   # [T, dx]
        z_i = batch.z[0].detach().cpu()   # [T, dz]

        disamb_t = estimate_disambiguation_time_quadrature(
            x_i,
            T=T,
            a=a,
            V=V,
            dt=dt,
            sigma_z=sigma_z,
            d=d,
            n=n,
            sigma_x=sigma_x,
            z0_std=z0_std,
            threshold=threshold,
            zmin=zmin,
            zmax=zmax,
            G=G,
        )

        is_delayed = (disamb_t >= 0)

        if is_delayed and len(delayed_examples) < n_delayed:
            delayed_examples.append((x_i, z_i))
            delayed_seeds.append(seed)
            delayed_times.append(disamb_t)

        elif (not is_delayed) and len(non_delayed_examples) < n_non_delayed:
            non_delayed_examples.append((x_i, z_i))
            non_delayed_seeds.append(seed)
            non_delayed_times.append(disamb_t)  # will be -1

        seeds_checked += 1
        seed += 1

        if verbose and seeds_checked % 100 == 0:
            print(
                f"[search] checked={seeds_checked} | "
                f"delayed={len(delayed_examples)}/{n_delayed} | "
                f"non_delayed={len(non_delayed_examples)}/{n_non_delayed}"
            )

    if len(delayed_examples) < n_delayed or len(non_delayed_examples) < n_non_delayed:
        raise RuntimeError(
            "Failed to collect requested dataset size within max_seed_search. "
            f"Got delayed={len(delayed_examples)}/{n_delayed}, "
            f"non_delayed={len(non_delayed_examples)}/{n_non_delayed}, "
            f"checked={seeds_checked}."
        )

    # Delayed examples first, then non-delayed
    all_examples = delayed_examples + non_delayed_examples
    all_seed_ids = delayed_seeds + non_delayed_seeds
    all_disamb_time = delayed_times + non_delayed_times
    all_delayed_flag = [True] * len(delayed_examples) + [False] * len(non_delayed_examples)

    x = torch.stack([ex[0] for ex in all_examples], dim=0)   # [N, T, dx]
    z = torch.stack([ex[1] for ex in all_examples], dim=0)   # [N, T, dz]

    delayed_indices = list(range(len(delayed_examples)))
    non_delayed_indices = list(range(len(delayed_examples), len(all_examples)))

    meta = {
        "task": "doublewell_1d",
        "T": int(T),
        "dz": int(z.shape[-1]),
        "dx": int(x.shape[-1]),
        "a": float(a),
        "V": float(V),
        "dt": float(dt),
        "sigma_z": float(sigma_z),
        "d": float(d),
        "n": int(n),
        "sigma_x": float(sigma_x),
        "z0_mean": float(z0_mean),
        "z0_std": float(z0_std),
        "delay_threshold": float(threshold),
        "quadrature_zmin": float(zmin),
        "quadrature_zmax": float(zmax),
        "quadrature_G": int(G),
        "search_seed_start": int(search_seed_start),
        "max_seed_search": int(max_seed_search),
        "num_sequences": int(len(all_examples)),
        "num_delayed": int(len(delayed_examples)),
        "num_non_delayed": int(len(non_delayed_examples)),
        "delayed_indices": delayed_indices,
        "non_delayed_indices": non_delayed_indices,
        "data_seed_ids_delayed": delayed_seeds,
        "data_seed_ids_non_delayed": non_delayed_seeds,
    }

    return DoubleWell1DDatasetArtifact(
        x=x,
        z=z,
        data_seed_ids=torch.tensor(all_seed_ids, dtype=torch.long),
        delayed_flag=torch.tensor(all_delayed_flag, dtype=torch.bool),
        disamb_time=torch.tensor(all_disamb_time, dtype=torch.long),
        meta=meta,
    )


if __name__ == "__main__":
    # Edit these as needed for direct dataset generation from this script.
    artifact = build_doublewell_1d_dataset(
        T=120,
        n_delayed=200,
        n_non_delayed=50,
        search_seed_start=0,
        max_seed_search=50000,
        device="cpu",
        dtype=torch.float32,
        a=3.0,
        V=0.06,
        dt=1.0,
        sigma_z=0.05,
        d=2.0,
        n=1,
        sigma_x=0.12,
        z0_mean=0.0,
        z0_std=1.0,
        threshold=0.8,
        zmin=-4.0,
        zmax=4.0,
        G=1000,
        verbose=True,
    )

    out_root = Path("data/datasets/doublewell_1d/benchmark_v1")
    artifact.save(out_root)

    print("\nSaved dataset artifact to:", out_root)
    print("Metadata summary:")
    for k, v in artifact.meta.items():
        if k in {"delayed_indices", "non_delayed_indices", "data_seed_ids_delayed", "data_seed_ids_non_delayed"}:
            print(f"  {k}: <len={len(v)}>")
        else:
            print(f"  {k}: {v}")