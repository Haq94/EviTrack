# inference/utils.py
from __future__ import annotations
import torch
from typing import List
import dataclasses

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

def tree_clone(obj):
    """
    Recursively clone nested tensors inside lists/tuples/dicts/dataclasses.
    Prevents aliasing after resampling.
    """
    if torch.is_tensor(obj):
        return obj.clone()

    # ---- dataclasses support (Particle, Hypothesis, etc.) ----
    if dataclasses.is_dataclass(obj):
        kwargs = {}
        for f in dataclasses.fields(obj):
            kwargs[f.name] = tree_clone(getattr(obj, f.name))
        return obj.__class__(**kwargs)

    if isinstance(obj, list):
        return [tree_clone(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(tree_clone(x) for x in obj)
    if isinstance(obj, dict):
        return {k: tree_clone(v) for k, v in obj.items()}
    return obj