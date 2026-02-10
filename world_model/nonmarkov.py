from __future__ import annotations
from typing import Optional
import torch

from .base import WorldModelBase, WorldModelConfig
from .modules.gru_memory import GRUMemory


class NonMarkovWorldModel(WorldModelBase):
    """
    Non-Markov WM:

      - Transition: p(z_t | z_{1:t-1}) implemented as p(z_t | z_state_prev)
      - z_state is a GRU summary over latents
      - Emission uses (z_state_curr, x_state_prev) where
            z_state_curr = GRU(z_state_prev, z_t)
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__(cfg)
        # transition head
        self._transition_head = self._build_head(
            in_dim=self.transition_in_dim(),
            out_dim=self.dz,
            head_cfg=self.cfg.transition,
        )
        # z-memory over latents
        self.z_memory = GRUMemory(in_dim=self.dz, mem_dim=cfg.z_mem_dim)

    # ---- z-state semantics ----

    def z_state_dim(self) -> int:
        return self.cfg.z_mem_dim

    def init_z_state(self, B: int, device=None, dtype=None) -> Optional[torch.Tensor]:
        return self.z_memory.init_state(B, device=device, dtype=dtype)

    def update_z_state(self, z_state: Optional[torch.Tensor], z_t: torch.Tensor) -> Optional[torch.Tensor]:
        assert z_state is not None
        return self.z_memory.step(z_state, z_t)

    def z_state_curr(self, z_state_prev: Optional[torch.Tensor], z_t: torch.Tensor) -> torch.Tensor:
        """
        Current z-summary includes z_t (what you want).
        """
        assert z_state_prev is not None
        z_state_curr = self.z_memory.step(z_state_prev, z_t)
        return z_state_curr

    # ---- Transition ----

    def transition_in_dim(self) -> int:
        # transition network consumes z_state_prev
        return self.cfg.z_mem_dim

    def transition_context(self, z_prev: Optional[torch.Tensor], z_state_prev: Optional[torch.Tensor]) -> torch.Tensor:
        # z_prev is ignored; we use the full-history summary
        assert z_state_prev is not None
        return z_state_prev

