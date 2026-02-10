# world_model/markov.py

from __future__ import annotations
import torch
from .base import WorldModelBase, WorldModelConfig


class MarkovWorldModel(WorldModelBase):
    """
    Markov: p(z_t | z_{t-1}), so transition context is z_prev.
    No z-memory.
    """
    def __init__(self, cfg: WorldModelConfig):
        super().__init__(cfg)
        # transition head
        self._transition_head = self._build_head(
            in_dim=self.transition_in_dim(),
            out_dim=self.dz,
            head_cfg=self.cfg.transition,
        )

    def transition_in_dim(self) -> int:
        return self.dz

    def transition_context(self, z_prev, z_state_prev):
        if z_prev is None:
            raise ValueError("MarkovWorldModel requires z_prev.")
        return z_prev

