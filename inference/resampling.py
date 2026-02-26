# inference/resampling.py
from __future__ import annotations
import torch

Tensor = torch.Tensor

def effective_sample_size(w: Tensor, eps: float = 1e-12) -> float:
    # w: [N] normalized
    return float(1.0 / (w.clamp_min(eps).pow(2).sum()).item())

def multinomial_resample_indices(w: Tensor, N: int) -> Tensor:
    # w: [N] normalized
    return torch.multinomial(w, num_samples=N, replacement=True)

def resample_particles(particles, w: Tensor, idx: Tensor):
    # particles: list[Particle], idx: [N]
    new_particles = [particles[i] for i in idx.tolist()]
    # After resampling, weights become uniform -> logw reset to 0
    for p in new_particles:
        p.logw = torch.zeros_like(p.logw)
    return new_particles