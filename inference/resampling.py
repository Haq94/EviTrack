# inference/resampling.py
from __future__ import annotations
import torch
from .utils import tree_clone

Tensor = torch.Tensor

def effective_sample_size(w: Tensor, eps: float = 1e-12) -> float:
    return float(1.0 / (w.clamp_min(eps).pow(2).sum()).item())

def multinomial_resample_indices(w: Tensor, N: int) -> Tensor:
    return torch.multinomial(w, num_samples=N, replacement=True)

def resample_particles(particles, w, idx):
    # w is unused here but kept for a consistent call signature.
    new_particles = []
    for i in idx.tolist():
        p = particles[i]
        p_new = tree_clone(p)
        p_new.logw = torch.zeros_like(p_new.logw)
        new_particles.append(p_new)
    return new_particles