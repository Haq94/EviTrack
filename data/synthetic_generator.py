# data/synthetic_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore


Array = np.ndarray

# ---------- batched Gaussian utils ----------

def _batchify_vec(v: Array, B: int, d: int) -> Array:
    v = np.asarray(v, dtype=float)
    if v.shape == (d,):
        return np.broadcast_to(v, (B, d)).copy()
    if v.shape == (B, d):
        return v
    raise ValueError(f"Expected {(d,)} or {(B, d)}, got {v.shape}")


def _batchify_mat(M: Array, B: int, d: int) -> Array:
    M = np.asarray(M, dtype=float)
    if M.shape == (d, d):
        return np.broadcast_to(M, (B, d, d)).copy()
    if M.shape == (B, d, d):
        return M
    raise ValueError(f"Expected {(d, d)} or {(B, d, d)}, got {M.shape}")


def sample_gaussian(mu: Array, cov: Array, rng: np.random.Generator) -> Array:
    """mu (B,d), cov (B,d,d) -> samples (B,d)"""
    B, d = mu.shape
    eps = rng.standard_normal((B, d))
    out = np.empty((B, d), dtype=float)
    for b in range(B):
        L = np.linalg.cholesky(cov[b])
        out[b] = mu[b] + L @ eps[b]
    return out


def logpdf_gaussian(x: Array, mu: Array, cov: Array) -> Array:
    """x,mu (B,d), cov (B,d,d) -> logpdf (B,)"""
    B, d = x.shape
    const = -0.5 * d * np.log(2.0 * np.pi)
    lp = np.empty((B,), dtype=float)
    for b in range(B):
        L = np.linalg.cholesky(cov[b])
        y = np.linalg.solve(L, x[b] - mu[b])
        maha = float(y @ y)
        logdet = float(2.0 * np.sum(np.log(np.diag(L))))
        lp[b] = const - 0.5 * (logdet + maha)
    return lp


# ---------- stationary model specs ----------

@dataclass
class GaussianPrior:
    """z1 ~ N(mu0, cov0)"""
    mu0: Array        # (dz,)
    cov0: Array       # (dz,dz)


@dataclass
class GaussianTransition:
    """
    z_t | z_{t-1} ~ N( mu(z_{t-1}), cov(z_{t-1}) )

    mean_fn: (z_prev (B,dz), extras) -> (B,dz) or (dz,)
    cov_fn:  (z_prev (B,dz), extras) -> (B,dz,dz) or (dz,dz)
    """
    mean_fn: Callable[[Array, Dict], Array]
    cov_fn: Callable[[Array, Dict], Array]


@dataclass
class GaussianEmission:
    """
    x_t | z_t ~ N( mu(z_t), cov(z_t) )

    mean_fn: (z (B,dz), extras) -> (B,dx) or (dx,)
    cov_fn:  (z (B,dz), extras) -> (B,dx,dx) or (dx,dx)
    """
    mean_fn: Callable[[Array, Dict], Array]
    cov_fn: Callable[[Array, Dict], Array]


@dataclass
class TrajectoryBatch:
    z: Array                 # (B,T,dz)
    x: Array                 # (B,T,dx)
    logp: Optional[Array]    # (B,) if requested


def generate_sequences(
    T: int,
    B: int,
    prior: GaussianPrior,
    transition: GaussianTransition,
    emission: GaussianEmission,
    *,
    seed: int = 0,
    return_logp: bool = False,
    extras: Optional[Dict] = None,
) -> TrajectoryBatch:
    """
    Stationary latent-variable sequence generator (NO explicit time dependence).
    """
    if T < 1 or B < 1:
        raise ValueError("T and B must be >= 1")

    extras = {} if extras is None else dict(extras)
    rng = np.random.default_rng(seed)

    mu0_vec = np.asarray(prior.mu0, dtype=float)
    dz = int(mu0_vec.shape[0])

    # z1
    mu0 = _batchify_vec(mu0_vec, B, dz)
    cov0 = _batchify_mat(prior.cov0, B, dz)
    z1 = sample_gaussian(mu0, cov0, rng)  # (B,dz)

    # infer dx from emission mean
    mu_x1_raw = np.asarray(emission.mean_fn(z1, extras), dtype=float)
    if mu_x1_raw.ndim == 1:
        dx = int(mu_x1_raw.shape[0])
        mu_x1 = _batchify_vec(mu_x1_raw, B, dx)
    elif mu_x1_raw.ndim == 2:
        dx = int(mu_x1_raw.shape[1])
        mu_x1 = mu_x1_raw
    else:
        raise ValueError(f"Emission mean must be (dx,) or (B,dx), got {mu_x1_raw.shape}")

    z = np.empty((B, T, dz), dtype=float)
    x = np.empty((B, T, dx), dtype=float)
    z[:, 0, :] = z1

    logp = np.zeros((B,), dtype=float) if return_logp else None
    if return_logp:
        logp += logpdf_gaussian(z1, mu0, cov0)

    # x1
    cov_x1 = _batchify_mat(emission.cov_fn(z1, extras), B, dx)
    x1 = sample_gaussian(mu_x1, cov_x1, rng)
    x[:, 0, :] = x1
    if return_logp:
        logp += logpdf_gaussian(x1, mu_x1, cov_x1)

    # rollout
    for _t in range(1, T):
        z_prev = z[:, _t - 1, :]

        mu_t = _batchify_vec(np.asarray(transition.mean_fn(z_prev, extras), dtype=float), B, dz)
        cov_t = _batchify_mat(np.asarray(transition.cov_fn(z_prev, extras), dtype=float), B, dz)

        z_t = sample_gaussian(mu_t, cov_t, rng)
        z[:, _t, :] = z_t
        if return_logp:
            logp += logpdf_gaussian(z_t, mu_t, cov_t)

        mu_xt = _batchify_vec(np.asarray(emission.mean_fn(z_t, extras), dtype=float), B, dx)
        cov_xt = _batchify_mat(np.asarray(emission.cov_fn(z_t, extras), dtype=float), B, dx)

        x_t = sample_gaussian(mu_xt, cov_xt, rng)
        x[:, _t, :] = x_t
        if return_logp:
            logp += logpdf_gaussian(x_t, mu_xt, cov_xt)

    return TrajectoryBatch(z=z, x=x, logp=logp)


# ---------- PyTorch Dataset + DataLoader wrapper ----------

class SequenceDataset(Dataset):
    """
    Simple dataset that yields dicts: {"x": (T,dx), "z": (T,dz)} (and optional "logp": scalar)
    """
    def __init__(self, batch: TrajectoryBatch, *, device: str = "cpu", dtype=torch.float32):
        if torch is None:
            raise RuntimeError("PyTorch is not available, cannot create SequenceDataset.")

        self.x = torch.tensor(batch.x, dtype=dtype, device=device)  # (B,T,dx)
        self.z = torch.tensor(batch.z, dtype=dtype, device=device)  # (B,T,dz)
        self.logp = None
        if batch.logp is not None:
            self.logp = torch.tensor(batch.logp, dtype=dtype, device=device)  # (B,)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        out = {"x": self.x[idx], "z": self.z[idx]}
        if self.logp is not None:
            out["logp"] = self.logp[idx]
        return out


def make_dataloader(
    T: int,
    B: int,
    prior: GaussianPrior,
    transition: GaussianTransition,
    emission: GaussianEmission,
    *,
    seed: int = 0,
    return_logp: bool = False,
    extras: Optional[Dict] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
    dtype=None,
) -> "DataLoader":
    """
    Convenience wrapper:
      generates a dataset of B sequences of length T and returns a torch DataLoader.
    """
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is not available, cannot create DataLoader.")

    if dtype is None:
        dtype = torch.float32

    batch = generate_sequences(
        T=T,
        B=B,
        prior=prior,
        transition=transition,
        emission=emission,
        seed=seed,
        return_logp=return_logp,
        extras=extras,
    )
    ds = SequenceDataset(batch, device=device, dtype=dtype)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ---------- regression test ----------

if __name__ == "__main__":
    # Minimal stationary linear-Gaussian example (no explicit time dependence)

    # dims
    dz = 2
    dx = 2

    # prior
    prior = GaussianPrior(
        mu0=np.zeros((dz,)),
        cov0=0.5 * np.eye(dz),
    )

    # transition: z_t = A z_{t-1} + noise
    A = np.array([[0.95, 0.05],
                  [0.00, 0.98]], dtype=float)
    Q = 0.05 * np.eye(dz)

    def trans_mean(z_prev: Array, extras: Dict) -> Array:
        return (z_prev @ A.T)  # (B,dz)

    def trans_cov(z_prev: Array, extras: Dict) -> Array:
        # stationary covariance (could be state-dependent; keep simple)
        return Q  # (dz,dz) broadcast OK

    transition = GaussianTransition(mean_fn=trans_mean, cov_fn=trans_cov)

    # emission: x_t = H z_t + noise
    H = np.eye(dx, dz)
    R = 0.10 * np.eye(dx)

    def emit_mean(z: Array, extras: Dict) -> Array:
        return (z @ H.T)  # (B,dx)

    def emit_cov(z: Array, extras: Dict) -> Array:
        return R  # (dx,dx) broadcast OK

    emission = GaussianEmission(mean_fn=emit_mean, cov_fn=emit_cov)

    # generate one batch (numpy)
    T = 25
    B = 128
    batch = generate_sequences(
        T=T,
        B=B,
        prior=prior,
        transition=transition,
        emission=emission,
        seed=123,
        return_logp=True,
    )
    print("Generated:")
    print("  z:", batch.z.shape, "x:", batch.x.shape, "logp:", None if batch.logp is None else batch.logp.shape)

    # wrap in DataLoader (torch)
    if torch is None:
        print("PyTorch not available; skipping DataLoader regression.")
    else:
        loader = make_dataloader(
            T=T,
            B=B,
            prior=prior,
            transition=transition,
            emission=emission,
            seed=123,
            return_logp=True,
            batch_size=32,
            shuffle=True,
            device="cpu",
        )
        one = next(iter(loader))
        print("\nDataLoader batch:")
        print("  x:", tuple(one["x"].shape), "(batch,T,dx)")
        print("  z:", tuple(one["z"].shape), "(batch,T,dz)")
        if "logp" in one:
            print("  logp:", tuple(one["logp"].shape), "(batch,)")
