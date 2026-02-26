# data/synthetic_tasks/doublewell_1d.py

import numpy as np
import torch

from data.synthetic_generator import (
    GaussianPrior,
    GaussianTransition,
    GaussianEmission,
    build_synthetic_bundle,
)

def make_prior(*, z0_mean: float = 0.0, z0_std: float = 1.0) -> GaussianPrior:
    return GaussianPrior(
        mu0=torch.tensor([z0_mean], dtype=torch.float32),
        cov0=torch.tensor([[z0_std**2]], dtype=torch.float32),
    )

def make_transition(*, a: float = 3.0, V: float = 0.06, dt: float = 1.0, sigma_z: float = 0.05) -> GaussianTransition:
    def grad_U(z):
        return (4.0 * V / (a ** 4)) * z * (z**2 - a**2)

    def mean_fn(z_prev, extras):
        return z_prev - grad_U(z_prev) * dt

    def cov_fn(z_prev, extras):
        return torch.full(
            (z_prev.shape[0], 1, 1),
            sigma_z**2,
            device=z_prev.device,
            dtype=z_prev.dtype,
            )

    return GaussianTransition(mean_fn=mean_fn, cov_fn=cov_fn)

def make_emission(*, d: float = 2.0, n: int = 1, sigma_x: float = 0.12) -> GaussianEmission:

    def mean_fn(z_t, extras):
        mask = (z_t.abs() <= d)
        out = torch.empty_like(z_t)
        out[mask] = z_t[mask] ** (2 * n)
        out[~mask] = z_t[~mask]
        return out

    def cov_fn(z_t, extras):
        return torch.full(
            (z_t.shape[0], 1, 1),
            sigma_x**2,
            device=z_t.device,
            dtype=z_t.dtype,
            )

    return GaussianEmission(mean_fn=mean_fn, cov_fn=cov_fn)

def build_loaders(
    *,
    T: int,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = True,
    include_latents_in_train: bool = False,
    include_latents_in_val: bool = True,
    include_latents_in_test: bool = True,
    # task params
    a: float = 3.0,
    V: float = 0.06,
    dt: float = 1.0,
    sigma_z: float = 0.05,
    d: float = 2.0,
    n: int = 1,
    sigma_x: float = 0.12,
    z0_mean: float = 0.0,
    z0_std: float = 1.0,
):
    prior = make_prior(z0_mean=z0_mean, z0_std=z0_std)
    transition = make_transition(a=a, V=V, dt=dt, sigma_z=sigma_z)
    emission = make_emission(d=d, n=n, sigma_x=sigma_x)

    meta = dict(task="doublewell_1d", dz=1, dx=1, a=a, V=V, d=d, n=n)

    return build_synthetic_bundle(
        T=T,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        prior=prior,
        transition=transition,
        emission=emission,
        seed=seed,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last_train=drop_last_train,
        include_latents_in_train=include_latents_in_train,
        include_latents_in_val=include_latents_in_val,
        include_latents_in_test=include_latents_in_test,
        extras=None,
        meta=meta,
    )