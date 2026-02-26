# data/synthetic_generator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data.data_bundle import DataBundle

Tensor = torch.Tensor

# ---------- batched Gaussian utils ----------

def _batchify_vec(v: Tensor, B: int, d: int) -> Tensor:
    if v.shape == (d,):
        return v.unsqueeze(0).expand(B, d).contiguous()
    if v.shape == (B, d):
        return v
    raise ValueError(f"Expected {(d,)} or {(B, d)}, got {tuple(v.shape)}")


def _batchify_mat(M: Tensor, B: int, d: int) -> Tensor:
    if M.shape == (d, d):
        return M.unsqueeze(0).expand(B, d, d).contiguous()
    if M.shape == (B, d, d):
        return M
    raise ValueError(f"Expected {(d, d)} or {(B, d, d)}, got {tuple(M.shape)}")


def sample_gaussian(mu: Tensor, cov: Tensor, gen: torch.Generator) -> Tensor:
    """
    mu: (B,d), cov: (B,d,d) -> samples: (B,d)
    """
    B, d = mu.shape
    eps = torch.randn((B, d), device=mu.device, dtype=mu.dtype, generator=gen)
    L = torch.linalg.cholesky(cov)            # (B,d,d)
    return mu + torch.einsum("bij,bj->bi", L, eps)


def logpdf_gaussian(x: Tensor, mu: Tensor, cov: Tensor) -> Tensor:
    """
    x,mu: (B,d), cov: (B,d,d) -> logpdf: (B,)
    """
    B, d = x.shape
    L = torch.linalg.cholesky(cov)            # (B,d,d)
    diff = (x - mu).unsqueeze(-1)             # (B,d,1)

    # Solve L y = diff
    y = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)  # (B,d)
    maha = (y * y).sum(dim=-1)                # (B,)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (B,)

    const = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, device=x.device, dtype=x.dtype))
    return const - 0.5 * (logdet + maha)


# ---------- stationary model specs ----------

@dataclass
class GaussianPrior:
    mu0: Tensor   # (dz,)
    cov0: Tensor  # (dz,dz)


@dataclass
class GaussianTransition:
    mean_fn: Callable[[Tensor, Dict], Tensor]   # (B,dz)->(B,dz) or (dz,)
    cov_fn: Callable[[Tensor, Dict], Tensor]    # (B,dz)->(B,dz,dz) or (dz,dz)


@dataclass
class GaussianEmission:
    mean_fn: Callable[[Tensor, Dict], Tensor]   # (B,dz)->(B,dx) or (dx,)
    cov_fn: Callable[[Tensor, Dict], Tensor]    # (B,dz)->(B,dx,dx) or (dx,dx)


@dataclass
class TrajectoryBatch:
    z: Tensor                 # (B,T,dz)
    x: Tensor                 # (B,T,dx)
    logp: Optional[Tensor]    # (B,) if requested


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
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TrajectoryBatch:
    if T < 1 or B < 1:
        raise ValueError("T and B must be >= 1")

    extras = {} if extras is None else dict(extras)
    device = torch.device(device)

    gen = torch.Generator(device="cpu")  # torch.Generator is CPU-backed; that’s fine
    gen.manual_seed(seed)

    mu0_vec = prior.mu0.to(device=device, dtype=dtype)
    cov0_mat = prior.cov0.to(device=device, dtype=dtype)
    dz = int(mu0_vec.shape[0])

    mu0 = _batchify_vec(mu0_vec, B, dz)
    cov0 = _batchify_mat(cov0_mat, B, dz)

    # z1
    z1 = sample_gaussian(mu0, cov0, gen)  # (B,dz)

    # infer dx from emission mean
    mu_x1_raw = emission.mean_fn(z1, extras)
    if mu_x1_raw.ndim == 1:
        dx = int(mu_x1_raw.shape[0])
        mu_x1 = _batchify_vec(mu_x1_raw.to(device=device, dtype=dtype), B, dx)
    elif mu_x1_raw.ndim == 2:
        dx = int(mu_x1_raw.shape[1])
        mu_x1 = mu_x1_raw.to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Emission mean must be (dx,) or (B,dx), got {tuple(mu_x1_raw.shape)}")

    z = torch.empty((B, T, dz), device=device, dtype=dtype)
    x = torch.empty((B, T, dx), device=device, dtype=dtype)
    z[:, 0, :] = z1

    logp = torch.zeros((B,), device=device, dtype=dtype) if return_logp else None
    if return_logp:
        logp = logp + logpdf_gaussian(z1, mu0, cov0)

    # x1
    cov_x1_raw = emission.cov_fn(z1, extras).to(device=device, dtype=dtype)
    cov_x1 = _batchify_mat(cov_x1_raw, B, dx)
    x1 = sample_gaussian(mu_x1, cov_x1, gen)
    x[:, 0, :] = x1
    if return_logp:
        logp = logp + logpdf_gaussian(x1, mu_x1, cov_x1)

    # rollout
    for t in range(1, T):
        z_prev = z[:, t - 1, :]

        mu_t_raw = transition.mean_fn(z_prev, extras)
        mu_t = _batchify_vec(mu_t_raw.to(device=device, dtype=dtype), B, dz)

        cov_t_raw = transition.cov_fn(z_prev, extras).to(device=device, dtype=dtype)
        cov_t = _batchify_mat(cov_t_raw, B, dz)

        z_t = sample_gaussian(mu_t, cov_t, gen)
        z[:, t, :] = z_t
        if return_logp:
            logp = logp + logpdf_gaussian(z_t, mu_t, cov_t)

        mu_xt_raw = emission.mean_fn(z_t, extras)
        mu_xt = _batchify_vec(mu_xt_raw.to(device=device, dtype=dtype), B, dx)

        cov_xt_raw = emission.cov_fn(z_t, extras).to(device=device, dtype=dtype)
        cov_xt = _batchify_mat(cov_xt_raw, B, dx)

        x_t = sample_gaussian(mu_xt, cov_xt, gen)
        x[:, t, :] = x_t
        if return_logp:
            logp = logp + logpdf_gaussian(x_t, mu_xt, cov_xt)

    return TrajectoryBatch(z=z, x=x, logp=logp)


# ---------- PyTorch Dataset + DataLoader wrapper ----------

class SequenceDataset(Dataset):
    def __init__(self, batch: TrajectoryBatch):
        self.x = batch.x
        self.z = batch.z
        self.logp = batch.logp

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        device=device,
        dtype=dtype,
    )
    ds = SequenceDataset(batch)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_synthetic_bundle(
    *,
    T: int,
    n_train: int,
    n_val: int,
    n_test: int,
    prior: GaussianPrior,
    transition: GaussianTransition,
    emission: GaussianEmission,
    seed: int,
    batch_size: int,
    device: str = "cpu",
    dtype=None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = True,
    include_latents_in_train: bool = False,
    include_latents_in_val: bool = True,
    include_latents_in_test: bool = True,
    extras: Optional[Dict] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> DataBundle:
    """
    Generic builder: from (prior, transition, emission) -> DataBundle(train/val/test).
    Train batches default to (x,) so Trainer stays unchanged.
    Val/test can optionally return (x, z) for diagnostics.
    """
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is not available, cannot create DataLoaders.")

    if dtype is None:
        dtype = torch.float32

    def _make_loader(N: int, seed_offset: int, *, shuffle: bool, drop_last: bool, include_latents: bool):
        batch = generate_sequences(
            T=T,
            B=N,
            prior=prior,
            transition=transition,
            emission=emission,
            seed=seed + seed_offset,
            return_logp=False,
            extras=extras,
        )
        x = batch.x
        z = batch.z
        ds = TensorDataset(x, z) if include_latents else TensorDataset(x)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = _make_loader(
        n_train, 1000, shuffle=True, drop_last=drop_last_train, include_latents=include_latents_in_train
    )
    val_loader = _make_loader(
        n_val, 9000, shuffle=False, drop_last=False, include_latents=include_latents_in_val
    )

    test_loader = None
    if n_test and n_test > 0:
        test_loader = _make_loader(
            n_test, 15000, shuffle=False, drop_last=False, include_latents=include_latents_in_test
        )

    train_keys = ("x", "z") if include_latents_in_train else ("x",)

    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        meta=meta or {},
        batch_keys=train_keys,
    )


# ---------- regression test ----------

if __name__ == "__main__":
    # Minimal stationary linear-Gaussian example (no explicit time dependence)

    dz, dx = 2, 2
    prior = GaussianPrior(mu0=torch.zeros(dz), cov0=0.5 * torch.eye(dz))

    A = torch.tensor([[0.95, 0.05], [0.00, 0.98]])
    Q = 0.05 * torch.eye(dz)

    def trans_mean(z_prev, extras): return z_prev @ A.T
    def trans_cov(z_prev, extras):  return Q

    transition = GaussianTransition(mean_fn=trans_mean, cov_fn=trans_cov)

    H = torch.eye(dx, dz)
    R = 0.10 * torch.eye(dx)

    def emit_mean(z, extras): return z @ H.T
    def emit_cov(z, extras):  return R

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
