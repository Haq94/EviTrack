# inference/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple
import torch
from .types import StepStats

Tensor = torch.Tensor

class InferenceEngine(ABC):
    def __init__(self, *, wm: torch.nn.Module, proposal: torch.nn.Module | None, cfg: Any):
        self.wm = wm
        self.proposal = proposal
        self.cfg = cfg

    @abstractmethod
    def init_state(self, B: int, device: str, dtype: torch.dtype):
        ...

    @abstractmethod
    def step(self, state, x_t: Tensor) -> Tuple[Any, StepStats]:
        ...

    @abstractmethod
    def get_mixture(self, state):
        """Return (weights, support) for forecasting. support is hyps/particles list."""
        ...