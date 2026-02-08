# world_model/base.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import torch.nn as nn

from .modules.gru_memory import GRUMemory
from .modules.gaussian import (
    GaussianDiagHead,
    GaussianDiagParams,
    GaussianLowRankDiagHead,
    GaussianLowRankDiagParams,
)


# -----------------------------
# Config
# -----------------------------

@dataclass
class HeadConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "relu"
    cov_type: str = "diag"  # "fixed_diag" | "diag" | "lowrank"
    rank: int = 4           # only for lowrank
    U_scale: float = 0.05   # only for lowrank
    min_logstd: float = -8.0
    max_logstd: float = 3.0
    init_logstd: float = -1.0


@dataclass
class WorldModelConfig:
    dz: int
    dx: int

    prior_mu0: float = 0.0
    prior_logstd0: float = 0.0

    # x-memory (optional) to summarize x_{<t}
    use_x_memory_in_emission: bool = False
    x_mem_dim: int = 32

    # z-memory dim (NonMarkov); Markov will effectively use dz as "z_state_dim"
    z_mem_dim: int = 32

    transition: HeadConfig = field(default_factory=HeadConfig)
    emission: HeadConfig = field(default_factory=HeadConfig)


# -----------------------------
# Base WM
# -----------------------------

class WorldModelBase(nn.Module):
    """
    WM-only base class. No inference. No pruning. Pure generative pieces.

    Your chosen semantics:
      - Transition produces z_t
      - Define "current z-state" z_state_curr:
          * Markov:    z_state_curr = z_t               (dim dz)
          * NonMarkov: z_state_curr = GRU(z_state_prev, z_t)  (dim z_mem_dim)
      - Emission uses (z_state_curr, x_state_prev) where x_state_prev summarizes x_{<t} if enabled.
    """
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.dz = cfg.dz
        self.dx = cfg.dx

        # Prior params (diagonal Gaussian, global)
        self.prior_mu = nn.Parameter(torch.full((self.dz,), float(cfg.prior_mu0)))
        self.prior_logstd = nn.Parameter(torch.full((self.dz,), float(cfg.prior_logstd0)))

        # Optional x-memory (for emission context)
        self.x_memory: Optional[GRUMemory] = None
        if cfg.use_x_memory_in_emission:
            self.x_memory = GRUMemory(in_dim=self.dx, mem_dim=cfg.x_mem_dim)

        # Transition head built lazily (input dim defined by subclass)
        self._transition_head = None  # may be diag or lowrank

        # Emission head: input is z_state_curr (+ optional x_state_prev)
        emit_z_dim = self.z_state_dim()  # Markov returns dz; NonMarkov returns z_mem_dim
        emit_in_dim = emit_z_dim + (cfg.x_mem_dim if cfg.use_x_memory_in_emission else 0)

        self._emission_head = self._build_head(
            in_dim=emit_in_dim,
            out_dim=self.dx,
            head_cfg=cfg.emission,
        )

    # -----------------------------
    # helpers: head builder
    # -----------------------------

    def _build_head(self, in_dim: int, out_dim: int, head_cfg: HeadConfig):
        if head_cfg.cov_type in ("fixed_diag", "diag"):
            return GaussianDiagHead(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=head_cfg.hidden_dim,
                num_layers=head_cfg.num_layers,
                activation=head_cfg.activation,
                cov_type=head_cfg.cov_type,
                min_logstd=head_cfg.min_logstd,
                max_logstd=head_cfg.max_logstd,
                init_logstd=head_cfg.init_logstd,
            )
        elif head_cfg.cov_type == "lowrank":
            return GaussianLowRankDiagHead(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=head_cfg.rank,
                hidden_dim=head_cfg.hidden_dim,
                num_layers=head_cfg.num_layers,
                activation=head_cfg.activation,
                min_logstd=head_cfg.min_logstd,
                max_logstd=head_cfg.max_logstd,
                init_logstd=head_cfg.init_logstd,
                U_scale=head_cfg.U_scale,
            )
        else:
            raise ValueError(f"Unknown cov_type={head_cfg.cov_type}")

    # -----------------------------
    # Prior
    # -----------------------------

    def prior_params(self, B: int, device=None, dtype=None) -> GaussianDiagParams:
        mu = self.prior_mu.unsqueeze(0).expand(B, self.dz).to(device=device, dtype=dtype)
        logstd = self.prior_logstd.unsqueeze(0).expand(B, self.dz).to(device=device, dtype=dtype)
        return GaussianDiagParams(mu=mu, logstd=logstd)

    def sample_z1(self, B: int, device=None, dtype=None, eps: torch.Tensor | None = None) -> torch.Tensor:
        params = self.prior_params(B, device=device, dtype=dtype)
        return GaussianDiagHead.sample(params, eps=eps)

    def log_prob_z1(self, z1: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        params = self.prior_params(B, device=z1.device, dtype=z1.dtype)
        return GaussianDiagHead.log_prob(z1, params)

    # -----------------------------
    # x-memory
    # -----------------------------

    def init_x_state(self, B: int, device=None, dtype=None) -> Optional[torch.Tensor]:
        if self.x_memory is None:
            return None
        return self.x_memory.init_state(B, device=device, dtype=dtype)

    def update_x_state(self, x_state: Optional[torch.Tensor], x_t: torch.Tensor) -> Optional[torch.Tensor]:
        if self.x_memory is None:
            return None
        assert x_state is not None
        return self.x_memory.step(x_state, x_t)

    # -----------------------------
    # z-state semantics (Mode 2)
    # -----------------------------

    def z_state_dim(self) -> int:
        """
        Markov: dz
        NonMarkov: z_mem_dim
        """
        return self.dz

    def z_state_curr(self, z_state_prev: Optional[torch.Tensor], z_t: torch.Tensor) -> torch.Tensor:
        """
        Markov default: current z-state is just z_t.
        NonMarkov overrides to return GRU-updated z_state.
        """
        return z_t

    # -----------------------------
    # Transition (subclass defines context)
    # -----------------------------

    def transition_in_dim(self) -> int:
        raise NotImplementedError

    def transition_context(self, z_prev: Optional[torch.Tensor], z_state_prev: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def build_transition_head_if_needed(self):
        if self._transition_head is None:
            in_dim = self.transition_in_dim()
            self._transition_head = self._build_head(
                in_dim=in_dim,
                out_dim=self.dz,
                head_cfg=self.cfg.transition,
            )

    def transition_params(self, z_prev: Optional[torch.Tensor], z_state_prev: Optional[torch.Tensor]):
        self.build_transition_head_if_needed()
        ctx = self.transition_context(z_prev, z_state_prev)
        return self._transition_head(ctx)

    def sample_transition(self, params, eps: torch.Tensor | None = None) -> torch.Tensor:
        # params type depends on cov_type
        if isinstance(params, GaussianDiagParams):
            return GaussianDiagHead.sample(params, eps=eps)
        elif isinstance(params, GaussianLowRankDiagParams):
            # eps here corresponds to diag eps; rank eps is drawn internally unless you pass it separately
            return GaussianLowRankDiagHead.sample(params, eps_diag=eps, eps_rank=None)
        else:
            raise TypeError(f"Unknown transition params type: {type(params)}")

    def log_prob_transition(self, z_t: torch.Tensor, params) -> torch.Tensor:
        if isinstance(params, GaussianDiagParams):
            return GaussianDiagHead.log_prob(z_t, params)
        elif isinstance(params, GaussianLowRankDiagParams):
            return GaussianLowRankDiagHead.log_prob(z_t, params)
        else:
            raise TypeError(f"Unknown transition params type: {type(params)}")

    # -----------------------------
    # Emission (your semantics: use z_state_curr + x_state_prev)
    # -----------------------------

    def emission_params(self, z_state_curr: torch.Tensor, x_state_prev: Optional[torch.Tensor]):
        if self.cfg.use_x_memory_in_emission:
            assert x_state_prev is not None
            inp = torch.cat([z_state_curr, x_state_prev], dim=-1)
        else:
            inp = z_state_curr
        return self._emission_head(inp)

    def sample_emission(self, params, eps: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(params, GaussianDiagParams):
            return GaussianDiagHead.sample(params, eps=eps)
        elif isinstance(params, GaussianLowRankDiagParams):
            return GaussianLowRankDiagHead.sample(params, eps_diag=eps, eps_rank=None)
        else:
            raise TypeError(f"Unknown emission params type: {type(params)}")

    def log_prob_emission(self, x_t: torch.Tensor, params) -> torch.Tensor:
        if isinstance(params, GaussianDiagParams):
            return GaussianDiagHead.log_prob(x_t, params)
        elif isinstance(params, GaussianLowRankDiagParams):
            return GaussianLowRankDiagHead.log_prob(x_t, params)
        else:
            raise TypeError(f"Unknown emission params type: {type(params)}")

    # -----------------------------
    # NonMarkov hooks (optional)
    # -----------------------------

    def init_z_state(self, B: int, device=None, dtype=None) -> Optional[torch.Tensor]:
        return None

    def update_z_state(self, z_state: Optional[torch.Tensor], z_t: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    # -----------------------------
    # Step (Convention A)
    # -----------------------------

    @torch.no_grad()
    def step(
        self,
        *,
        B: int,
        z_prev: Optional[torch.Tensor],
        z_state_prev: Optional[torch.Tensor],
        x_state_prev: Optional[torch.Tensor],
        device=None,
        dtype=None,
        eps_z: Optional[torch.Tensor] = None,
        eps_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Convention A + your emission semantics:

          1) z_t ~ p(z_t | transition_context(z_prev, z_state_prev))
          2) z_state_curr = current z-summary:
               Markov: z_state_curr = z_t
               NonMarkov: z_state_curr = GRU(z_state_prev, z_t)
          3) x_t ~ p(x_t | z_state_curr, x_state_prev)
          4) update stored states:
               z_state_t = update_z_state(z_state_prev, z_t)   (NonMarkov)
               x_state_t = update_x_state(x_state_prev, x_t)

        Note:
          - Markov can set z_state_t = None; z_state_curr is still z_t.
        """
        # Infer device/dtype
        if device is None:
            if z_prev is not None:
                device = z_prev.device
            elif z_state_prev is not None:
                device = z_state_prev.device
            elif x_state_prev is not None:
                device = x_state_prev.device
            else:
                device = "cpu"

        if dtype is None:
            if z_prev is not None:
                dtype = z_prev.dtype
            elif z_state_prev is not None:
                dtype = z_state_prev.dtype
            elif x_state_prev is not None:
                dtype = x_state_prev.dtype
            else:
                dtype = torch.float32

        # --- Transition ---
        trans_params = self.transition_params(z_prev=z_prev, z_state_prev=z_state_prev)
        z_t = self.sample_transition(trans_params, eps=eps_z)
        logp_z = self.log_prob_transition(z_t, trans_params)

        # --- Current z-state for emission ---
        z_state_t = self.update_z_state(z_state_prev, z_t) 

        # --- Emission ---
        emit_params = self.emission_params(z_state_curr=z_state_t, x_state_prev=x_state_prev)
        x_t = self.sample_emission(emit_params, eps=eps_x)
        logp_x = self.log_prob_emission(x_t, emit_params)

        # --- Update stored states AFTER sampling ---
        # z_state_t = self.update_z_state(z_state_prev, z_t)  # NonMarkov overrides
        x_state_t = self.update_x_state(x_state_prev, x_t)

        out = {
            "z_t": z_t,
            "x_t": x_t,
            "z_state_t": z_state_t,
            "x_state_t": x_state_t,
            "logp_z": logp_z,
            "logp_x": logp_x,
        }

        # optionally expose params for debugging
        if isinstance(trans_params, GaussianDiagParams):
            out["trans_params_mu"] = trans_params.mu
            out["trans_params_logstd"] = trans_params.logstd
        if isinstance(emit_params, GaussianDiagParams):
            out["emit_params_mu"] = emit_params.mu
            out["emit_params_logstd"] = emit_params.logstd

        return out
