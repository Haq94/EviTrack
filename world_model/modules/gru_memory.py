# world_model/modules/gru_memory.py

from __future__ import annotations
import torch
import torch.nn as nn


class GRUMemory(nn.Module):
    """
    A tiny wrapper around GRUCell to update a memory state one step at a time.
    State shape: (B, mem_dim)
    Input shape: (B, in_dim)
    """
    def __init__(self, in_dim: int, mem_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.cell = nn.GRUCell(input_size=in_dim, hidden_size=mem_dim)

    def init_state(self, B: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(B, self.mem_dim, device=device, dtype=dtype)

    def step(self, state: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        return self.cell(inp, state)
