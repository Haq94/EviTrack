# inference/evitrack.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Any
import torch

from .base import InferenceEngine
from .types import Hypothesis, EviTrackState, StepStats, CostCounter
from .utils import topk_indices, stack_scores, normalize_logweights

Tensor = torch.Tensor

@dataclass
class EviTrackConfig:
    K: int                  # keep
    C: int                  # children per parent
    tau: int = 1            # prune every tau steps
    expand: str = "proposal"  # {"proposal","transition"}
    prune_score: str = "evidence"  # {"evidence","joint"}
    weight_mode: str = "evidence"  # {"evidence","joint"}

class EviTrackEngine(InferenceEngine):
    def __init__(self, *, wm, proposal, cfg: EviTrackConfig):
        super().__init__(wm=wm, proposal=proposal, cfg=cfg)
        assert cfg.expand in ("proposal", "transition")
        assert cfg.prune_score in ("evidence", "joint")
        assert cfg.weight_mode in ("evidence", "joint")
        if cfg.expand == "proposal":
            assert proposal is not None, "proposal required for expand='proposal'"

    def init_state(self, B: int, device: str, dtype: torch.dtype) -> EviTrackState:
        # initialize WM and proposal states once; replicate per-hypothesis
        wm_z_state = self.wm.init_z_state(B, device=device)     # your API
        wm_x_state = self.wm.init_x_state(B, device=device)

        if self.proposal is not None:
            q_z_state = self.proposal.init_z_state(B, device=device)
            q_x_state = self.proposal.init_x_state(B, device=device)
        else:
            q_z_state, q_x_state = None, None

        # initial hypothesis: no z yet. We'll sample z1 at first step.
        z0 = torch.zeros((B, getattr(self.wm.cfg, "dz", 1)), device=device, dtype=dtype)

        J0 = torch.zeros((B,), device=device, dtype=dtype)
        E0 = torch.zeros((B,), device=device, dtype=dtype)

        h0 = Hypothesis(
            z_t=z0,
            wm_z_state=wm_z_state,
            wm_x_state=wm_x_state,
            q_z_state=q_z_state,
            q_x_state=q_x_state,
            J=J0,
            E=E0,
        )
        return EviTrackState(hyps=[h0], t=0, cost=CostCounter())

    def step(self, state: EviTrackState, x_t: Tensor) -> Tuple[EviTrackState, StepStats]:
        cfg = self.cfg
        B = x_t.shape[0]
        t_new = state.t + 1

        # Expand parents -> candidates
        candidates: List[Hypothesis] = []
        for parent in state.hyps:
            for _ in range(cfg.C):
                child = self._expand_one(parent, x_t, t_new)
                candidates.append(child)

        # Possibly delay pruning
        do_prune = (t_new % cfg.tau == 0)

        if do_prune:
            # choose score tensor [Kcand,B]
            if cfg.prune_score == "joint":
                S = stack_scores([c.J for c in candidates])
            else:
                S = stack_scores([c.E for c in candidates])

            keep_idx = topk_indices(S, k=cfg.K).tolist()
            kept = [candidates[i] for i in keep_idx]
        else:
            # keep all candidates (can grow); still cap if you want safety
            kept = candidates

        new_state = EviTrackState(hyps=kept, t=t_new, cost=state.cost)

        stats = StepStats(
            t=t_new,
            kept=len(kept),
            candidates=len(candidates),
            cost=new_state.cost,
            extra={"do_prune": do_prune},
        )
        return new_state, stats

    def get_mixture(self, state: EviTrackState):
        # returns normalized weights over hypotheses (scalar per hypothesis) + hyps
        cfg = self.cfg
        assert len(state.hyps) > 0

        # use batch-averaged log-weight for each hypothesis
        if cfg.weight_mode == "joint":
            logw = torch.stack([h.J.mean() for h in state.hyps], dim=0)  # [K]
        else:
            logw = torch.stack([h.E.mean() for h in state.hyps], dim=0)  # [K]

        w = normalize_logweights(logw, dim=0)  # [K], sums to 1
        return w, state.hyps

    # ------------------------
    # internal: expand a single child
    # ------------------------
    def _expand_one(self, parent: Hypothesis, x_t: Tensor, t: int) -> Hypothesis:
        cfg = self.cfg
        B = x_t.shape[0]

        z_prev = parent.z_t if t > 1 else None
        wm_z_state = parent.wm_z_state
        wm_x_state = parent.wm_x_state
        q_z_state = parent.q_z_state
        q_x_state = parent.q_x_state

        # ---------------------------------------------------
        # 1) Sample z_t
        # ---------------------------------------------------

        if t == 1:
            # Prior
            z_t = self.wm.sample_z1(B, device=x_t.device, dtype=x_t.dtype)
            logpzt = self.wm.log_prob_z1(z_t)
            logq_t = torch.zeros_like(logpzt)

        else:
            if cfg.expand == "transition":
                trans_params = self.wm.transition_params(
                    z_prev=z_prev,
                    z_state_prev=wm_z_state,
                )
                z_t = self.wm.sample_transition(trans_params)
                logpzt = self.wm.log_prob_transition(z_t, trans_params)
                logq_t = torch.zeros_like(logpzt)

            else:  # proposal expansion
                q_out = self.proposal.propose(
                    B=B,
                    z_prev=z_prev,
                    z_state_prev=q_z_state,
                    x_state_prev=q_x_state,
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                z_t = q_out["z_t"]
                logq_t = q_out["logq"]
                q_z_state = q_out["z_state_t"]

                if t == 1:
                    logpzt = self.wm.log_prob_z1(z_t)
                else:
                    trans_params = self.wm.transition_params(
                        z_prev=z_prev,
                        z_state_prev=wm_z_state,
                    )
                    logpzt = self.wm.log_prob_transition(z_t, trans_params)

        # ---------------------------------------------------
        # 2) Emission likelihood
        # ---------------------------------------------------

        z_state_curr = self.wm.z_state_curr(wm_z_state, z_t)
        emit_params = self.wm.emission_params(
            z_state_curr=z_state_curr,
            x_state_prev=wm_x_state,
        )
        logpxt = self.wm.log_prob_emission(x_t, emit_params)

        # ---------------------------------------------------
        # 3) Update accumulated scores
        # ---------------------------------------------------

        E_new = parent.E + logpxt
        J_new = parent.J + logpxt + logpzt

        # ---------------------------------------------------
        # 4) Update stored states AFTER using x_t
        # ---------------------------------------------------

        wm_z_state_new = self.wm.update_z_state(wm_z_state, z_t)
        wm_x_state_new = self.wm.update_x_state(wm_x_state, x_t)

        if self.proposal is not None:
            q_x_state_new = self.proposal.update_x_state(
                x_t=x_t,
                x_state_prev=q_x_state,
            )
        else:
            q_x_state_new = None

        return Hypothesis(
            z_t=z_t,
            wm_z_state=wm_z_state_new,
            wm_x_state=wm_x_state_new,
            q_z_state=q_z_state,
            q_x_state=q_x_state_new,
            J=J_new,
            E=E_new,
        )