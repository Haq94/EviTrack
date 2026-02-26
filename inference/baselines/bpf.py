# inference/baselines/bpf.py
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..base import InferenceEngine
from ..types import Particle, ParticleState, StepStats, CostCounter
from ..utils import normalize_logweights
from ..resampling import effective_sample_size, multinomial_resample_indices, resample_particles

Tensor = torch.Tensor

@dataclass
class BPFConfig:
    N: int
    resample_every_step: bool = True
    ess_threshold_frac: float = 0.5  # used if resample_every_step=False

class BFPEngine(InferenceEngine):
    def __init__(self, *, wm, cfg: BPFConfig):
        super().__init__(wm=wm, proposal=None, cfg=cfg)

    def init_state(self, B, device, dtype):
        particles = []
        for _ in range(self.cfg.N):
            wm_z_state = self.wm.init_z_state(B, device=device, dtype=dtype)
            wm_x_state = self.wm.init_x_state(B, device=device, dtype=dtype)

            z0 = torch.zeros(B, self.wm.dz, device=device, dtype=dtype)
            logw = torch.zeros(B, device=device, dtype=dtype)

            particles.append(Particle(
                z_t=z0,
                wm_z_state=wm_z_state,
                wm_x_state=wm_x_state,
                q_z_state=None,
                q_x_state=None,
                logw=logw,
            ))
        return ParticleState(particles=particles, t=0, cost=CostCounter())

    def step(self, state: ParticleState, x_t: Tensor):
        t_new = state.t + 1
        new_particles = []

        for p in state.particles:
            z_prev = p.z_t if t_new > 1 else None

            # --- sample from prior/transition ---
            if t_new == 1:
                z_t = self.wm.sample_z1(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
            else:
                trans_params = self.wm.transition_params(
                    z_prev=z_prev,
                    z_state_prev=p.wm_z_state,
                )
                z_t = self.wm.sample_transition(trans_params)

            # --- emission log prob ---
            z_state_curr = self.wm.z_state_curr(p.wm_z_state, z_t)
            emit_params = self.wm.emission_params(
                z_state_curr=z_state_curr,
                x_state_prev=p.wm_x_state,
            )
            logpxt = self.wm.log_prob_emission(x_t, emit_params)

            # --- bootstrap weight update (likelihood only) ---
            logw_new = p.logw + logpxt

            # --- state updates AFTER using x_t ---
            wm_z_state_new = self.wm.update_z_state(p.wm_z_state, z_t)
            wm_x_state_new = self.wm.update_x_state(p.wm_x_state, x_t)

            new_particles.append(Particle(
                z_t=z_t,
                wm_z_state=wm_z_state_new,
                wm_x_state=wm_x_state_new,
                q_z_state=None,
                q_x_state=None,
                logw=logw_new,
            ))

        # --- normalized weights for resampling decision ---
        logw_vec = torch.stack([p.logw.mean() for p in new_particles], dim=0)  # [N] (B=1 recommended)
        w = normalize_logweights(logw_vec, dim=0)

        if self.cfg.resample_every_step:
            do_resample = True
            ess = float("nan")
        else:
            ess = effective_sample_size(w)
            do_resample = ess < (self.cfg.ess_threshold_frac * self.cfg.N)

        extra = {"ess": ess, "resampled": False}
        if do_resample:
            idx = multinomial_resample_indices(w, self.cfg.N)
            new_particles = resample_particles(new_particles, w, idx)
            extra["resampled"] = True

        new_state = ParticleState(particles=new_particles, t=t_new, cost=state.cost)
        stats = StepStats(t=t_new, kept=len(new_particles), candidates=len(new_particles), cost=new_state.cost, extra=extra)
        return new_state, stats

    def get_mixture(self, state: ParticleState):
        logw = torch.stack([p.logw.mean() for p in state.particles], dim=0)
        w = normalize_logweights(logw, dim=0)
        return w, state.particles