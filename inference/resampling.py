# inference/resampling.py
from __future__ import annotations
import torch
from .utils import tree_clone

Tensor = torch.Tensor


def effective_sample_size(w: Tensor, eps: float = 1e-12) -> float:
    """
    w: normalized weights [N]
    returns: ESS in [1, N]
    """
    return float(1.0 / (w.clamp_min(eps).pow(2).sum()).item())


def multinomial_resample_indices(w: Tensor, N: int) -> Tensor:
    """
    Standard multinomial resampling.
    """
    return torch.multinomial(w, num_samples=N, replacement=True)


def systematic_resample_indices(w: Tensor, N: int) -> Tensor:
    """
    Systematic resampling for normalized weights w [N].

    Algorithm:
      u ~ Uniform(0, 1/N)
      u_k = u + k/N,  k=0,...,N-1
      pick smallest i with cdf[i] >= u_k
    """
    assert w.ndim == 1, f"Expected w shape [N], got {tuple(w.shape)}"
    assert w.numel() == N, f"Expected {N} weights, got {w.numel()}"

    # normalize defensively
    w = w / w.sum()
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0  # guard against roundoff

    # one random offset in [0, 1/N)
    u0 = torch.rand(1, device=w.device, dtype=w.dtype) / N
    positions = u0 + torch.arange(N, device=w.device, dtype=w.dtype) / N  # [N]

    # searchsorted gives the ancestor index for each position
    idx = torch.searchsorted(cdf, positions, right=False)
    idx = torch.clamp(idx, max=N - 1)
    return idx


def resample_particles(particles, idx):
    """
    Clone resampled particles and reset log weights to zero.
    """
    new_particles = []
    for i in idx.tolist():
        p_new = tree_clone(particles[i])
        p_new.logw = torch.zeros_like(p_new.logw)
        new_particles.append(p_new)
    return new_particles