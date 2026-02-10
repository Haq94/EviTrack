# proposal/proposal.py
#
# Forecasting-causal amortized proposal for latent worldlines.
# We implement a sequential factorization where the proposal at time t uses ONLY x_{<t}:
#
#   q_phi(z_t | z_{<t}, x_{<t})
#
# This is the right object if you want to (1) generate children before seeing x_t, then
# (2) do evidence scoring / pruning with the world model likelihood p(x_t | z_{\le t}, x_{<t}).
#
# Supports:
#   - z_mode: "markov" (use z_{t-1}) or "memory" (GRU summary over sampled z history)
#   - x_mode: "none" | "markov" (use x_{t-1}) | "memory" (GRU summary over observed x history)
#   - optional sharing of GRU modules with a WorldModel (wm.z_memory / wm.x_memory)
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn


from world_model.modules.gru_memory import GRUMemory
from world_model.modules.gaussian import (
    GaussianDiagHead,
    GaussianDiagParams,
    GaussianLowRankDiagHead,
    GaussianLowRankDiagParams,
)
from world_model.base import HeadConfig


# -----------------------------
# Config
# -----------------------------

@dataclass
class ProposalConfig:
    dz: int
    dx: int

    # proposal conditioning modes
    # z_mode: "markov" uses z_{t-1}; "memory" uses a GRU summary of sampled latents
    z_mode: str = "memory"     # "markov" | "memory"
    z_mem_dim: int = 32        # only used if z_mode == "memory"

    # x_mode: proposal sees ONLY x_{<t} (forecasting-causal)
    # "none": ignore x entirely
    # "markov": use x_{t-1}
    # "memory": GRU summary of observed x history up to t-1
    x_mode: str = "memory"     # "none" | "markov" | "memory"
    x_mem_dim: int = 32        # only used if x_mode == "memory"

    # Share GRUs from a provided world model (wm)
    share_z_gru_from_wm: bool = False
    share_x_gru_from_wm: bool = False

    # Gaussian head for q(z_t | context)
    head: HeadConfig = field(default_factory=HeadConfig)

    # Behavior when requested sharing is impossible/mismatched
    strict_share: bool = True

    # For markov z-mode, what to use at t=1 when z_prev is None
    # "zeros" is safest; you can switch to a learned parameter later if desired.
    z1_markov_init: str = "zeros"  # "zeros"


# -----------------------------
# Proposal
# -----------------------------

class Proposal(nn.Module):
    """
    Forecasting-causal proposal:
        q_phi(z_t | z_{<t}, x_{<t})

    State carried forward:
      - z_state: None (z_mode="markov") or GRU hidden state (z_mode="memory")
      - x_state: None (x_mode="none"), x_{t-1} (x_mode="markov"), or GRU hidden (x_mode="memory")

    Recommended usage pattern in an online loop:
      1) Maintain (z_prev, z_state, x_state) corresponding to information available BEFORE x_t arrives.
      2) Propose z_t via propose_z(...).
      3) Score with world model likelihood of x_t.
      4) After observing x_t, call observe_x(...) to update x_state for the next step.
      5) Set z_prev <- z_t and z_state <- returned z_state_t (if memory).

    Note: This module never consumes x_t at time t by design.
    """

    def __init__(self, cfg: ProposalConfig, wm: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.dz = cfg.dz
        self.dx = cfg.dx

        # ---- z-memory (optional) ----
        self.z_memory: Optional[GRUMemory] = None
        if cfg.z_mode.lower() == "memory":
            if cfg.share_z_gru_from_wm:
                shared = getattr(wm, "z_memory", None) if wm is not None else None
                if isinstance(shared, GRUMemory):
                    # Best-effort dim check
                    if hasattr(shared, "mem_dim") and int(shared.mem_dim) != int(cfg.z_mem_dim):
                        msg = f"WM z_memory.mem_dim={getattr(shared,'mem_dim',None)} != cfg.z_mem_dim={cfg.z_mem_dim}"
                        if cfg.strict_share:
                            raise ValueError(msg)
                    self.z_memory = shared
                else:
                    msg = "Requested share_z_gru_from_wm=True, but wm.z_memory not found / not GRUMemory."
                    if cfg.strict_share:
                        raise ValueError(msg)
            if self.z_memory is None:
                self.z_memory = GRUMemory(in_dim=self.dz, mem_dim=cfg.z_mem_dim)
        elif cfg.z_mode.lower() in ("markov",):
            self.z_memory = None
        else:
            raise ValueError(f"Unknown z_mode={cfg.z_mode}. Use 'markov' or 'memory'.")

        # ---- x-memory (optional) ----
        self.x_memory: Optional[GRUMemory] = None
        if cfg.x_mode.lower() == "memory":
            if cfg.share_x_gru_from_wm:
                shared = getattr(wm, "x_memory", None) if wm is not None else None
                if isinstance(shared, GRUMemory):
                    if hasattr(shared, "mem_dim") and int(shared.mem_dim) != int(cfg.x_mem_dim):
                        msg = f"WM x_memory.mem_dim={getattr(shared,'mem_dim',None)} != cfg.x_mem_dim={cfg.x_mem_dim}"
                        if cfg.strict_share:
                            raise ValueError(msg)
                    self.x_memory = shared
                else:
                    msg = "Requested share_x_gru_from_wm=True, but wm.x_memory not found / not GRUMemory."
                    if cfg.strict_share:
                        raise ValueError(msg)
            if self.x_memory is None:
                self.x_memory = GRUMemory(in_dim=self.dx, mem_dim=cfg.x_mem_dim)
        elif cfg.x_mode.lower() in ("none", "markov"):
            self.x_memory = None
        else:
            raise ValueError(f"Unknown x_mode={cfg.x_mode}. Use 'none', 'markov', or 'memory'.")

        # ---- head ----
        in_dim = self._context_dim()
        self._head = self._build_head(in_dim=in_dim, out_dim=self.dz, head_cfg=cfg.head)

        # Optional learned init for Markov z at t=1 could go here later.
        # For now, we use zeros (cfg.z1_markov_init == "zeros").

    # -----------------------------
    # Helpers
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
        if head_cfg.cov_type == "lowrank":
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
        raise ValueError(f"Unknown cov_type={head_cfg.cov_type}")

    def _z_ctx_dim(self) -> int:
        if self.cfg.z_mode.lower() == "markov":
            return self.dz
        if self.cfg.z_mode.lower() == "memory":
            return int(self.cfg.z_mem_dim)
        raise ValueError(f"Unknown z_mode={self.cfg.z_mode}")

    def _x_ctx_dim(self) -> int:
        xm = self.cfg.x_mode.lower()
        if xm == "none":
            return 0
        if xm == "markov":
            return self.dx
        if xm == "memory":
            return int(self.cfg.x_mem_dim)
        raise ValueError(f"Unknown x_mode={self.cfg.x_mode}")

    def _context_dim(self) -> int:
        return self._z_ctx_dim() + self._x_ctx_dim()

    def _infer_device_dtype(
        self,
        device=None,
        dtype=None,
        z_prev: Optional[torch.Tensor] = None,
        z_state_prev: Optional[torch.Tensor] = None,
        x_state_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.device, torch.dtype]:
        if device is None:
            for t in (z_prev, z_state_prev, x_state_prev):
                if t is not None:
                    device = t.device
                    break
            if device is None:
                device = torch.device("cpu")
        if dtype is None:
            for t in (z_prev, z_state_prev, x_state_prev):
                if t is not None:
                    dtype = t.dtype
                    break
            if dtype is None:
                dtype = torch.float32
        return torch.device(device), dtype

    # -----------------------------
    # State init/update
    # -----------------------------

    def init_z_state(self, B: int, device=None, dtype=None) -> Optional[torch.Tensor]:
        device, dtype = self._infer_device_dtype(device=device, dtype=dtype)
        if self.cfg.z_mode.lower() == "memory":
            assert self.z_memory is not None
            return self.z_memory.init_state(B, device=device, dtype=dtype)
        return None

    def init_x_state(self, B: int, device=None, dtype=None) -> Optional[torch.Tensor]:
        device, dtype = self._infer_device_dtype(device=device, dtype=dtype)
        xm = self.cfg.x_mode.lower()
        if xm == "none":
            return None
        if xm == "markov":
            # x_{t-1} unavailable at t=1; use zeros so we can still concatenate
            return torch.zeros(B, self.dx, device=device, dtype=dtype)
        if xm == "memory":
            assert self.x_memory is not None
            return self.x_memory.init_state(B, device=device, dtype=dtype)
        raise ValueError(f"Unknown x_mode={self.cfg.x_mode}")

    @torch.no_grad()
    def observe_x(self, x_t: torch.Tensor, x_state_prev: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Update the proposal's x_state AFTER observing x_t.
        This makes x_t available as part of x_{<t+1} for the next proposal step.

        - x_mode="none": returns None
        - x_mode="markov": returns x_t
        - x_mode="memory": returns GRU_x(x_state_prev, x_t)
        """
        xm = self.cfg.x_mode.lower()
        if xm == "none":
            return None
        if xm == "markov":
            return x_t
        if xm == "memory":
            assert self.x_memory is not None
            assert x_state_prev is not None
            return self.x_memory.step(x_state_prev, x_t)
        raise ValueError(f"Unknown x_mode={self.cfg.x_mode}")

    @torch.no_grad()
    def update_z_state(self, z_state_prev: Optional[torch.Tensor], z_t: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Update proposal z_state AFTER sampling z_t.
        - z_mode="markov": returns None
        - z_mode="memory": returns GRU_z(z_state_prev, z_t)
        """
        zm = self.cfg.z_mode.lower()
        if zm == "markov":
            return None
        if zm == "memory":
            assert self.z_memory is not None
            assert z_state_prev is not None
            return self.z_memory.step(z_state_prev, z_t)
        raise ValueError(f"Unknown z_mode={self.cfg.z_mode}")

    # -----------------------------
    # Context building
    # -----------------------------

    def z_context(self, z_prev: Optional[torch.Tensor], z_state_prev: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        zm = self.cfg.z_mode.lower()
        if zm == "markov":
            if z_prev is None:
                if self.cfg.z1_markov_init == "zeros":
                    return torch.zeros(B, self.dz, device=device, dtype=dtype)
                raise ValueError(f"Unknown z1_markov_init={self.cfg.z1_markov_init}")
            return z_prev.to(device=device, dtype=dtype)
        if zm == "memory":
            assert z_state_prev is not None
            return z_state_prev.to(device=device, dtype=dtype)
        raise ValueError(f"Unknown z_mode={self.cfg.z_mode}")

    def x_context(self, x_state_prev: Optional[torch.Tensor], B: int, device, dtype) -> Optional[torch.Tensor]:
        xm = self.cfg.x_mode.lower()
        if xm == "none":
            return None
        assert x_state_prev is not None, "x_state_prev must be initialized when x_mode != 'none'"
        return x_state_prev.to(device=device, dtype=dtype)

    def context(self, *, B: int, z_prev: Optional[torch.Tensor], z_state_prev: Optional[torch.Tensor], x_state_prev: Optional[torch.Tensor], device, dtype) -> torch.Tensor:
        z_ctx = self.z_context(z_prev=z_prev, z_state_prev=z_state_prev, B=B, device=device, dtype=dtype)
        x_ctx = self.x_context(x_state_prev=x_state_prev, B=B, device=device, dtype=dtype)
        if x_ctx is None:
            return z_ctx
        return torch.cat([z_ctx, x_ctx], dim=-1)

    # -----------------------------
    # Proposal step
    # -----------------------------

    def proposal_params(
        self,
        *,
        B: int,
        z_prev: Optional[torch.Tensor],
        z_state_prev: Optional[torch.Tensor],
        x_state_prev: Optional[torch.Tensor],
        device=None,
        dtype=None,
    ):
        """
        Return Gaussian params for q(z_t | z_{<t}, x_{<t}).
        """
        device, dtype = self._infer_device_dtype(device=device, dtype=dtype, z_prev=z_prev, z_state_prev=z_state_prev, x_state_prev=x_state_prev)
        ctx = self.context(
            B=B,
            z_prev=z_prev,
            z_state_prev=z_state_prev,
            x_state_prev=x_state_prev,
            device=device,
            dtype=dtype,
        )
        return self._head(ctx)

    def sample(self, params, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(params, GaussianDiagParams):
            return GaussianDiagHead.sample(params, eps=eps)
        if isinstance(params, GaussianLowRankDiagParams):
            return GaussianLowRankDiagHead.sample(params, eps_diag=eps, eps_rank=None)
        raise TypeError(f"Unknown params type: {type(params)}")

    def log_prob(self, z_t: torch.Tensor, params) -> torch.Tensor:
        if isinstance(params, GaussianDiagParams):
            return GaussianDiagHead.log_prob(z_t, params)
        if isinstance(params, GaussianLowRankDiagParams):
            return GaussianLowRankDiagHead.log_prob(z_t, params)
        raise TypeError(f"Unknown params type: {type(params)}")

    @torch.no_grad()
    def propose(
        self,
        *,
        B: int,
        z_prev: Optional[torch.Tensor],
        z_state_prev: Optional[torch.Tensor],
        x_state_prev: Optional[torch.Tensor],
        device=None,
        dtype=None,
        eps_z: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        One forecasting-causal proposal step:
          - params_t = q_phi(. | z_{<t}, x_{<t})
          - z_t ~ q_phi
          - z_state_t = updated z summary (if memory)
          - returns logq_t

        NOTE: This does NOT update x_state (because x_t is not known yet).
              Call observe_x(x_t, x_state_prev) AFTER you observe/score x_t.
        """
        device, dtype = self._infer_device_dtype(device=device, dtype=dtype, z_prev=z_prev, z_state_prev=z_state_prev, x_state_prev=x_state_prev)

        params = self.proposal_params(
            B=B,
            z_prev=z_prev,
            z_state_prev=z_state_prev,
            x_state_prev=x_state_prev,
            device=device,
            dtype=dtype,
        )
        z_t = self.sample(params, eps=eps_z)
        logq = self.log_prob(z_t, params)

        z_state_t = self.update_z_state(z_state_prev, z_t)

        out: Dict[str, Any] = {
            "z_t": z_t,
            "z_state_t": z_state_t,
            "logq": logq,
        }
        if isinstance(params, GaussianDiagParams):
            out["params_mu"] = params.mu
            out["params_logstd"] = params.logstd
        return out
