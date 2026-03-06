# inference/baselines/pf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch

from ..base import InferenceEngine
from ..types import Particle, ParticleState, StepStats, CostCounter
from ..utils import normalize_logweights, tree_clone
from ..resampling import effective_sample_size, systematic_resample_indices, resample_particles

Tensor = torch.Tensor


@dataclass
class ParticleFilterConfig:
    N: int
    proposal_mode: str = "proposal"   # "proposal" or "transition"
    resample: bool = True
    resample_every_step: bool = True
    ess_threshold_frac: float = 0.5

    def __post_init__(self):
        assert self.proposal_mode in ("proposal", "transition")


class ParticleFilterEngine(InferenceEngine):
    def __init__(self, *, wm, proposal=None, cfg: ParticleFilterConfig):
        super().__init__(wm=wm, proposal=proposal, cfg=cfg)

        if cfg.proposal_mode == "proposal":
            assert proposal is not None, "proposal_mode='proposal' requires a proposal."
        if cfg.proposal_mode == "transition":
            # proposal is optional and ignored
            pass

    def init_state(self, B: int, device: str, dtype: torch.dtype) -> ParticleState:
        wm_z_state_B = self.wm.init_z_state(B, device=device, dtype=dtype)
        wm_x_state_B = self.wm.init_x_state(B, device=device, dtype=dtype)

        if self.proposal is not None:
            q_z_state_B = self.proposal.init_z_state(B, device=device, dtype=dtype)
            q_x_state_B = self.proposal.init_x_state(B, device=device, dtype=dtype)
        else:
            q_z_state_B, q_x_state_B = None, None

        dz = self.wm.cfg.dz
        particles: List[List[Particle]] = []

        for b in range(B):
            wm_z_state = wm_z_state_B[b:b+1] if wm_z_state_B is not None else None
            wm_x_state = wm_x_state_B[b:b+1] if wm_x_state_B is not None else None
            q_z_state = q_z_state_B[b:b+1] if q_z_state_B is not None else None
            q_x_state = q_x_state_B[b:b+1] if q_x_state_B is not None else None

            z0 = torch.zeros((1, dz), device=device, dtype=dtype)
            logw0 = torch.zeros((), device=device, dtype=dtype)

            parts_b = []
            for _ in range(self.cfg.N):
                parts_b.append(
                    Particle(
                        z_t=z0.clone(),
                        wm_z_state=None if wm_z_state is None else wm_z_state.clone(),
                        wm_x_state=None if wm_x_state is None else wm_x_state.clone(),
                        q_z_state=None if q_z_state is None else q_z_state.clone(),
                        q_x_state=None if q_x_state is None else q_x_state.clone(),
                        logw=logw0.clone(),
                    )
                )
            particles.append(parts_b)

        return ParticleState(particles=particles, t=0, cost=CostCounter())

    def step(self, state: ParticleState, x_t: Tensor):
        """
        x_t: [B, dx]
        """
        B = x_t.shape[0]
        t_new = state.t + 1

        new_particles: List[List[Particle]] = []
        total_candidates = 0
        total_kept = 0
        resampled_flags = []

        for b in range(B):
            parts_b, did_resample = self._step_one(
                particles=state.particles[b],
                x_tb=x_t[b:b+1],
                t_new=t_new,
                cost=state.cost,
            )
            new_particles.append(parts_b)
            total_candidates += len(parts_b)
            total_kept += len(parts_b)
            resampled_flags.append(bool(did_resample))

        state.particles = new_particles
        state.t = t_new

        stats = StepStats(
            t=t_new,
            kept=total_kept,
            candidates=total_candidates,
            cost=state.cost,
            extra={
                "B": B,
                "resampled_per_batch": resampled_flags,
                "num_resampled": int(sum(resampled_flags)),
                "proposal_mode": self.cfg.proposal_mode,
                "resample": self.cfg.resample,
            },
        )
        return state, stats

    def _step_one(
        self,
        *,
        particles: List[Particle],
        x_tb: Tensor,      # [1, dx]
        t_new: int,
        cost: CostCounter,
    ) -> Tuple[List[Particle], bool]:
        assert x_tb.shape[0] == 1, "ParticleFilterEngine._step_one expects [1, dx]"

        new_particles: List[Particle] = []

        for p in particles:
            z_prev = None if t_new == 1 else p.z_t

            # -----------------------------
            # 1) sample z_t and compute log proposal density
            # -----------------------------
            if self.cfg.proposal_mode == "proposal":
                cost.add_proposal(1)
                q_out = self.proposal.propose(
                    B=1,
                    z_prev=z_prev,
                    z_state_prev=p.q_z_state,
                    x_state_prev=p.q_x_state,
                    device=x_tb.device,
                    dtype=x_tb.dtype,
                )
                z_t = q_out["z_t"]
                logq_t = q_out["logq"].squeeze()
                q_z_state_t = q_out["z_state_t"]

            else:  # transition proposal (bootstrap)
                if t_new == 1:
                    z_t = self.wm.sample_z1(1, device=x_tb.device, dtype=x_tb.dtype)
                    logq_t = self.wm.log_prob_z1(z_t).squeeze()
                else:
                    cost.add_transition(1)
                    trans_params = self.wm.transition_params(
                        z_prev=z_prev,
                        z_state_prev=p.wm_z_state,
                    )
                    z_t = self.wm.sample_transition(trans_params)
                    logq_t = self.wm.log_prob_transition(z_t, trans_params).squeeze()

                q_z_state_t = p.q_z_state  # unchanged / unused

            # -----------------------------
            # 2) latent model term log p(z_t | ...)
            # -----------------------------
            if self.cfg.proposal_mode == "proposal":
                if t_new == 1:
                    logpzt = self.wm.log_prob_z1(z_t).squeeze()
                else:
                    cost.add_transition(1)
                    trans_params = self.wm.transition_params(
                        z_prev=z_prev,
                        z_state_prev=p.wm_z_state,
                    )
                    logpzt = self.wm.log_prob_transition(z_t, trans_params).squeeze()
            else:
                # bootstrap / transition proposal: q == p, so ratio cancels
                logpzt = logq_t

            # -----------------------------
            # 3) emission likelihood
            # -----------------------------
            cost.add_emission(1)
            z_state_curr = self.wm.z_state_curr(p.wm_z_state, z_t)
            emit_params = self.wm.emission_params(
                z_state_curr=z_state_curr,
                x_state_prev=p.wm_x_state,
            )
            logpxt = self.wm.log_prob_emission(x_tb, emit_params).squeeze()

            # -----------------------------
            # 4) importance weight update
            # -----------------------------
            logw_new = p.logw + logpxt + logpzt - logq_t

            # -----------------------------
            # 5) state updates
            # -----------------------------
            wm_z_state_new = self.wm.update_z_state(p.wm_z_state, z_t)
            wm_x_state_new = self.wm.update_x_state(p.wm_x_state, x_tb)

            if self.proposal is not None:
                q_x_state_new = self.proposal.update_x_state(
                    x_t=x_tb,
                    x_state_prev=p.q_x_state,
                )
            else:
                q_x_state_new = p.q_x_state

            new_particles.append(
                Particle(
                    z_t=z_t,
                    wm_z_state=wm_z_state_new,
                    wm_x_state=wm_x_state_new,
                    q_z_state=q_z_state_t,
                    q_x_state=q_x_state_new,
                    logw=logw_new,
                )
            )

        # -----------------------------
        # 6) normalize weights
        # -----------------------------
        logw = torch.stack([p.logw for p in new_particles], dim=0)   # [N]
        w = normalize_logweights(logw, dim=0)

        # -----------------------------
        # 7) optional resampling
        # -----------------------------
        do_resample = False
        if self.cfg.resample:
            if self.cfg.resample_every_step:
                do_resample = True
            else:
                ess = effective_sample_size(w)
                do_resample = ess < (self.cfg.ess_threshold_frac * self.cfg.N)

        if do_resample:
            idx = systematic_resample_indices(w, self.cfg.N)
            new_particles = resample_particles(new_particles, idx)

        return new_particles, do_resample

    def get_mixture(self, state: ParticleState):
        """
        Returns:
            w: [B, N]
            support: List[List[Particle]]
        """
        B = len(state.particles)
        assert B > 0

        N = len(state.particles[0])
        for b in range(B):
            assert len(state.particles[b]) == N, "All particle sets must have same N."

        device = state.particles[0][0].logw.device
        dtype = state.particles[0][0].logw.dtype

        logw = torch.empty((B, N), device=device, dtype=dtype)
        for b in range(B):
            for n in range(N):
                logw[b, n] = state.particles[b][n].logw

        w = normalize_logweights(logw, dim=1)
        return w, state.particles