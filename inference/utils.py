# inference/utils.py
from __future__ import annotations
import torch
from typing import List, Tuple

Tensor = torch.Tensor

def logsumexp_stable(x: Tensor, dim: int) -> Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return (m + (x - m).exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)

def normalize_logweights(logw: Tensor, dim: int = 0) -> Tensor:
    # returns normalized weights (sum=1)
    lse = logsumexp_stable(logw, dim=dim)
    return (logw - lse.unsqueeze(dim)).exp()

def topk_indices(scores: Tensor, k: int) -> Tensor:
    # scores: [Kcand] (scalar per candidate) OR [Kcand, B] (per-batch)
    # For now we use batch-averaged score for pruning; you can upgrade later.
    if scores.ndim == 2:
        s = scores.mean(dim=1)  # average over batch
    else:
        s = scores
    return torch.topk(s, k=k, largest=True).indices

def stack_scores(scores: List[Tensor]) -> Tensor:
    # list of [B] -> [K, B]
    return torch.stack(scores, dim=0)