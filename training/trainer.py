# training/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn


ObjectiveKind = Literal["beta_elbo", "iwae"]


@dataclass
class TrainerConfig:
    objective: ObjectiveKind = "beta_elbo"
    beta: float = 1.0          # used for beta_elbo
    K: int = 16                # used for iwae
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    amp: bool = False          # optional mixed precision
    reduce_time: Literal["mean", "sum"] = "mean"  # how to reduce over T


class Trainer:
    """
    Trains a WorldModel (wm) + Proposal (q) for forecasting-causal inference:

        q_phi(z_t | z_{<t}, x_{<t})

    Supports:
      - beta-ELBO:   E_q[ sum_t log p(x_t | z_<=t, x_<t) + beta*(log p(z_t|...) - log q(z_t|...)) ]
      - IWAE(K):     E[ log(1/K sum_k exp(sum_t log p - log q)) ]

    Assumes batch is a dict with:
      batch["x"]: Tensor[B, T, dx]
    """

    def __init__(self, *, wm: nn.Module, proposal: nn.Module, cfg: TrainerConfig):
        self.wm = wm
        self.proposal = proposal
        self.cfg = cfg

        # one optimizer over both modules
        params = list(self.wm.parameters()) + list(self.proposal.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        # self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        device_type = next(self.wm.parameters()).device.type
        self.scaler = torch.amp.GradScaler(device_type, enabled=cfg.amp)


    # -------------------------
    # public API
    # -------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.wm.train()
        self.proposal.train()

        x = batch["x"]
        device = next(self.wm.parameters()).device
        device_type = next(self.wm.parameters()).device.type
        x = x.to(device=device)

        self.opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, enabled=self.cfg.amp):
            if self.cfg.objective == "beta_elbo":
                loss, stats = self._beta_elbo(x, beta=self.cfg.beta)
            elif self.cfg.objective == "iwae":
                loss, stats = self._iwae(x, K=self.cfg.K)
            else:
                raise ValueError(f"Unknown objective={self.cfg.objective}")

        # DEBUG--------------------------------------------  
        if not torch.isfinite(loss):
            print("Non-finite loss detected:", loss.item())
            raise RuntimeError("Loss became NaN or Inf")

        self.scaler.scale(loss).backward()

        # DEBUG-----------------------------------
        for name, p in self.wm.named_parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    print(f"NaN in gradient: {name}")
                    raise RuntimeError("Gradient NaN detected")

        if self.cfg.grad_clip_norm is not None:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                list(self.wm.parameters()) + list(self.proposal.parameters()),
                max_norm=float(self.cfg.grad_clip_norm),
            )

        self.scaler.step(self.opt)
        self.scaler.update()

        out = {"loss": float(loss.detach().cpu())}
        out.update({k: float(v) for k, v in stats.items()})
        return out

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.wm.eval()
        self.proposal.eval()

        x = batch["x"]
        device = next(self.wm.parameters()).device
        x = x.to(device=device)

        if self.cfg.objective == "beta_elbo":
            loss, stats = self._beta_elbo(x, beta=self.cfg.beta)
        elif self.cfg.objective == "iwae":
            loss, stats = self._iwae(x, K=self.cfg.K)
        else:
            raise ValueError(f"Unknown objective={self.cfg.objective}")

        out = {"loss": float(loss.detach().cpu())}
        out.update({k: float(v) for k, v in stats.items()})
        return out

    # -------------------------
    # objectives
    # -------------------------

    def _reduce_time(self, s_bt: torch.Tensor) -> torch.Tensor:
        # s_bt: [B, T] -> [B]
        if self.cfg.reduce_time == "mean":
            return s_bt.mean(dim=1)
        if self.cfg.reduce_time == "sum":
            return s_bt.sum(dim=1)
        raise ValueError(f"Unknown reduce_time={self.cfg.reduce_time}")

    def _beta_elbo(self, x: torch.Tensor, beta: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single-sample beta-ELBO.
        """
        B, T, dx = x.shape
        assert dx == getattr(self.wm, "dx", dx), "batch dx mismatch with wm.dx?"

        # ---- init WM states (for likelihood terms) ----
        wm_z_state = self.wm.init_z_state(B, device=x.device, dtype=x.dtype)
        wm_x_state = self.wm.init_x_state(B, device=x.device, dtype=x.dtype)

        # ---- init Proposal states (for q terms) ----
        q_z_state = self.proposal.init_z_state(B, device=x.device, dtype=x.dtype)
        q_x_state = self.proposal.init_x_state(B, device=x.device, dtype=x.dtype)

        z_prev: Optional[torch.Tensor] = None

        logp_x = []
        logp_z = []
        logq_z = []

        for t in range(T):
            # forecasting-causal: proposal sees x_{<t}, not x_t
            q_out = self.proposal.propose(
                B=B,
                z_prev=z_prev,
                z_state_prev=q_z_state,
                x_state_prev=q_x_state,
                device=x.device,
                dtype=x.dtype,
            )
            z_t = q_out["z_t"]
            logq_t = q_out["logq"]
            q_z_state = q_out["z_state_t"]  # None if z_mode="markov"

            # ---- p(z_t | ...) ----
            if t == 0:
                logpzt = self.wm.log_prob_z1(z_t)
            else:
                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state)
                logpzt = self.wm.log_prob_transition(z_t, trans_params)

            # ---- p(x_t | z_<=t, x_<t) ----
            z_state_curr = self.wm.z_state_curr(wm_z_state, z_t)
            emit_params = self.wm.emission_params(z_state_curr=z_state_curr, x_state_prev=wm_x_state)
            x_t = x[:, t, :]
            logpxt = self.wm.log_prob_emission(x_t, emit_params)

            # update WM stored states AFTER using x_t
            wm_z_state = self.wm.update_z_state(wm_z_state, z_t)
            wm_x_state = self.wm.update_x_state(wm_x_state, x_t)

            # update proposal x_state AFTER observing x_t (still forecasting-causal)
            q_x_state = self.proposal.update_x_state(x_t=x_t, x_state_prev=q_x_state)

            z_prev = z_t

            logp_x.append(logpxt)   # [B]
            logp_z.append(logpzt)   # [B]
            logq_z.append(logq_t)   # [B]

        logp_x = torch.stack(logp_x, dim=1)   # [B, T]
        logp_z = torch.stack(logp_z, dim=1)   # [B, T]
        logq_z = torch.stack(logq_z, dim=1)   # [B, T]

        # beta-ELBO per time step: log p(x_t|...) + beta*(log p(z_t|...) - log q(z_t|...))
        elbo_bt = logp_x + beta * (logp_z - logq_z)  # [B, T]
        elbo_b = self._reduce_time(elbo_bt)          # [B]
        loss = -elbo_b.mean()

        stats = {
            "elbo": elbo_b.mean().detach(),
            "logp_x": self._reduce_time(logp_x).mean().detach(),
            "logp_z": self._reduce_time(logp_z).mean().detach(),
            "logq_z": self._reduce_time(logq_z).mean().detach(),
            "beta": torch.tensor(beta, device=x.device),
        }
        return loss, stats

    def _iwae(self, x: torch.Tensor, K: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized IWAE(K) by flattening K into the batch dimension.
        """
        B, T, dx = x.shape
        KB = K * B

        # replicate x across K: [K, B, T, dx] -> [KB, T, dx]
        x_rep = x.unsqueeze(0).expand(K, B, T, dx).contiguous().view(KB, T, dx)

        # init states for KB streams
        wm_z_state = self.wm.init_z_state(KB, device=x.device, dtype=x.dtype)
        wm_x_state = self.wm.init_x_state(KB, device=x.device, dtype=x.dtype)

        q_z_state = self.proposal.init_z_state(KB, device=x.device, dtype=x.dtype)
        q_x_state = self.proposal.init_x_state(KB, device=x.device, dtype=x.dtype)

        z_prev: Optional[torch.Tensor] = None

        # accumulate log weights: logw = sum_t (logp_x + logp_z - logq)
        logw = torch.zeros(KB, device=x.device, dtype=x.dtype)

        for t in range(T):
            q_out = self.proposal.propose(
                B=KB,
                z_prev=z_prev,
                z_state_prev=q_z_state,
                x_state_prev=q_x_state,
                device=x.device,
                dtype=x.dtype,
            )
            z_t = q_out["z_t"]
            logq_t = q_out["logq"]
            q_z_state = q_out["z_state_t"]

            if t == 0:
                logpzt = self.wm.log_prob_z1(z_t)
            else:
                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state)
                logpzt = self.wm.log_prob_transition(z_t, trans_params)

            z_state_curr = self.wm.z_state_curr(wm_z_state, z_t)
            emit_params = self.wm.emission_params(z_state_curr=z_state_curr, x_state_prev=wm_x_state)
            x_t = x_rep[:, t, :]
            logpxt = self.wm.log_prob_emission(x_t, emit_params)

            wm_z_state = self.wm.update_z_state(wm_z_state, z_t)
            wm_x_state = self.wm.update_x_state(wm_x_state, x_t)

            q_x_state = self.proposal.update_x_state(x_t=x_t, x_state_prev=q_x_state)

            z_prev = z_t

            logw = logw + (logpxt + logpzt - logq_t)  # [KB]

        # reshape to [K, B]
        logw_kb = logw.view(K, B)

        # IWAE objective: E[ logmeanexp_k logw_k ]
        # numerically stable logmeanexp:
        m = logw_kb.max(dim=0).values              # [B]
        lme = m + torch.log(torch.exp(logw_kb - m.unsqueeze(0)).mean(dim=0))  # [B]
        loss = -lme.mean()

        stats = {
            "iwae": lme.mean().detach(),
            "K": torch.tensor(float(K), device=x.device),
            "logw_mean": logw_kb.mean().detach(),
            "logw_max": logw_kb.max().detach(),
        }
        return loss, stats
