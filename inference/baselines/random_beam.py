# inference/baselines/random_beam.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from ..base import InferenceEngine
from ..types import Hypothesis, EviTrackState, StepStats, CostCounter
from ..utils import normalize_logweights

Tensor = torch.Tensor


@dataclass
class RandomBeamConfig:
    K: int
    C: int
    tau: int = 1
    G: int = 1
    expand: str = "proposal"   # "proposal" or "transition"
    replace: bool = False      # global prune: sample with replacement?


class RandomBeamEngine(InferenceEngine):
    """
    Random beam-search baseline.

    Same hypothesis expansion / WM / proposal plumbing as EviTrack, but pruning is random:
      - LOCAL prune: choose 1 random child per parent
      - GLOBAL prune: choose K random children from the full K*C pool

    Mixture weights are uniform over the retained beam.
    """

    def __init__(self, *, wm, proposal, cfg: RandomBeamConfig):
        super().__init__(wm=wm, proposal=proposal, cfg=cfg)
        assert cfg.expand in ("proposal", "transition")
        assert cfg.K >= 1
        assert cfg.C >= 1
        assert cfg.tau >= 1
        assert cfg.G >= 1
        if cfg.expand == "proposal":
            assert proposal is not None, "proposal required for expand='proposal'"

    def init_state(self, B: int, device: str, dtype: torch.dtype) -> EviTrackState:
        # initialize WM and proposal states at batch=B, then slice into batch=1 per example
        wm_z_state_B = self.wm.init_z_state(B, device=device, dtype=dtype)
        wm_x_state_B = self.wm.init_x_state(B, device=device, dtype=dtype)

        if self.proposal is not None:
            q_z_state_B = self.proposal.init_z_state(B, device=device, dtype=dtype)
            q_x_state_B = self.proposal.init_x_state(B, device=device, dtype=dtype)
        else:
            q_z_state_B, q_x_state_B = None, None

        dz = self.wm.cfg.dz

        hyps: List[List[Hypothesis]] = []
        for b in range(B):
            wm_z_state = wm_z_state_B[b:b + 1] if wm_z_state_B is not None else None
            wm_x_state = wm_x_state_B[b:b + 1] if wm_x_state_B is not None else None
            q_z_state = q_z_state_B[b:b + 1] if q_z_state_B is not None else None
            q_x_state = q_x_state_B[b:b + 1] if q_x_state_B is not None else None

            # root has no z yet; keep a dummy z0
            z0 = torch.zeros((1, dz), device=device, dtype=dtype)
            zero = torch.zeros((), device=device, dtype=dtype)

            h0 = Hypothesis(
                z_t=z0,
                wm_z_state=wm_z_state,
                wm_x_state=wm_x_state,
                q_z_state=q_z_state,
                q_x_state=q_x_state,
                J=zero,
                E=zero,
                J_tbd=zero,
            )
            hyps.append([h0])

        return EviTrackState(hyps=hyps, t=0, cost=CostCounter())

    def step(self, state: EviTrackState, x_t: Tensor):
        """
        x_t: [B, dx]
        """
        B = x_t.shape[0]
        t_new = state.t + 1

        new_hyps: List[List[Hypothesis]] = []
        total_candidates = 0
        total_kept = 0

        for b in range(B):
            kept_b, cand_b = self._step_one(
                beam=state.hyps[b],
                x_tb=x_t[b:b + 1],   # [1, dx]
                t_new=t_new,
                cost=state.cost,
            )
            new_hyps.append(kept_b)
            total_candidates += cand_b
            total_kept += len(kept_b)

        state.hyps = new_hyps
        state.t = t_new

        stats = StepStats(
            t=t_new,
            kept=total_kept,
            candidates=total_candidates,
            cost=state.cost,
            extra={"B": B},
        )
        return state, stats

    def _step_one(
        self,
        *,
        beam: List[Hypothesis],
        x_tb: Tensor,   # [1, dx]
        t_new: int,
        cost: CostCounter,
    ) -> Tuple[List[Hypothesis], int]:
        """
        Advances a single batch element by one time step.

        Returns:
            kept: List[Hypothesis]
            num_candidates: int
        """
        cfg = self.cfg

        candidate_groups: List[List[Hypothesis]] = []

        if t_new == 1:
            root = beam[0]
            group = [self._expand_one(root, x_tb, t_new, cost=cost) for _ in range(cfg.K)]
            candidate_groups = [group]
        else:
            for parent in beam:
                group = [self._expand_one(parent, x_tb, t_new, cost=cost) for _ in range(cfg.C)]
                candidate_groups.append(group)

        num_candidates = sum(len(g) for g in candidate_groups)

        do_prune = (t_new % cfg.tau == 0)
        if do_prune:
            kept = self.prune(t_new=t_new, candidate_groups=candidate_groups)
        else:
            kept = [c for g in candidate_groups for c in g]

        return kept, num_candidates

    def get_mixture(self, state: EviTrackState):
        """
        Uniform mixture over retained hypotheses.

        Returns:
            w: [B, K]
            support: List[List[Hypothesis]]
        """
        B = len(state.hyps)
        assert B > 0

        K = len(state.hyps[0])
        for b in range(B):
            assert len(state.hyps[b]) == K, "All beams must have same K to return tensor weights."

        device = state.hyps[0][0].E.device
        dtype = state.hyps[0][0].E.dtype

        logw = torch.zeros((B, K), device=device, dtype=dtype)
        w = normalize_logweights(logw, dim=1)
        return w, state.hyps

    def _expand_one(
        self,
        parent: Hypothesis,
        x_t: Tensor,
        t: int,
        *,
        cost: CostCounter,
    ) -> Hypothesis:
        """
        Expand one hypothesis (B=1) at time t by sampling z_t, scoring with x_t,
        updating WM/proposal states, and returning a new Hypothesis.

        Scores are still accumulated for bookkeeping / comparability, but are NOT used
        for pruning or weighting in this baseline.
        """
        cfg = self.cfg
        assert x_t.shape[0] == 1, "RandomBeam _expand_one expects batch=1 tensor [1, dx]"

        # --- unpack parent ---
        z_prev = parent.z_t
        wm_z_state_prev = parent.wm_z_state
        wm_x_state_prev = parent.wm_x_state
        q_z_state_prev = parent.q_z_state
        q_x_state_prev = parent.q_x_state

        J_prev = parent.J
        E_prev = parent.E
        J_tbd_prev = parent.J_tbd

        # ---------------------------------------------------
        # 1) Sample z_t
        # ---------------------------------------------------
        if cfg.expand == "proposal":
            cost.add_proposal(1)
            q_out = self.proposal.propose(
                B=1,
                z_prev=None if t == 1 else z_prev,
                z_state_prev=q_z_state_prev,
                x_state_prev=q_x_state_prev,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            z_t = q_out["z_t"]
            q_z_state_t = q_out["z_state_t"]
            trans_params = None
        else:
            if t == 1:
                z_t = self.wm.sample_z1(1, device=x_t.device, dtype=x_t.dtype)
            else:
                cost.add_transition(1)
                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state_prev)
                z_t = self.wm.sample_transition(trans_params)

            q_z_state_t = q_z_state_prev

        # ---------------------------------------------------
        # 2) Joint latent term log p(z_t | ...)
        # ---------------------------------------------------
        if t == 1:
            logpzt = self.wm.log_prob_z1(z_t)
        else:
            if trans_params is None:
                cost.add_transition(1)
                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state_prev)
            logpzt = self.wm.log_prob_transition(z_t, trans_params)

        # ---------------------------------------------------
        # 3) Emission likelihood log p(x_t | z_state_curr, x_state_prev)
        # ---------------------------------------------------
        cost.add_emission(1)
        z_state_curr = self.wm.z_state_curr(wm_z_state_prev, z_t)
        emit_params = self.wm.emission_params(z_state_curr=z_state_curr, x_state_prev=wm_x_state_prev)
        logpxt = self.wm.log_prob_emission(x_t, emit_params)

        # ---------------------------------------------------
        # 4) Accumulate scores (unused by prune, but kept for consistency)
        # ---------------------------------------------------
        E_t = E_prev + logpxt.squeeze()
        J_t = J_prev + (logpxt + logpzt).squeeze()
        J_tbd = J_tbd_prev + (logpxt + logpzt).squeeze()

        # ---------------------------------------------------
        # 5) Update stored WM states AFTER observing (z_t, x_t)
        # ---------------------------------------------------
        wm_z_state_t = self.wm.update_z_state(wm_z_state_prev, z_t)
        wm_x_state_t = self.wm.update_x_state(wm_x_state_prev, x_t)

        # ---------------------------------------------------
        # 6) Update proposal x-state AFTER observing x_t
        # ---------------------------------------------------
        if self.proposal is not None:
            q_x_state_t = self.proposal.update_x_state(x_t, q_x_state_prev)
        else:
            q_x_state_t = None

        # ---------------------------------------------------
        # 7) Return child hypothesis
        # ---------------------------------------------------
        return Hypothesis(
            z_t=z_t,
            wm_z_state=wm_z_state_t,
            wm_x_state=wm_x_state_t,
            q_z_state=q_z_state_t,
            q_x_state=q_x_state_t,
            J=J_t,
            E=E_t,
            J_tbd=J_tbd,
        )

    def prune(self, *, t_new: int, candidate_groups: List[List[Hypothesis]]) -> List[Hypothesis]:
        """
        candidate_groups: list of groups, one per parent.

        LOCAL:
            each parent keeps one random child

        GLOBAL:
            flatten all children and keep K random children
        """
        cfg = self.cfg
        do_global = (t_new == 1) or (t_new % cfg.G == 0)

        if do_global:
            all_cands = [c for g in candidate_groups for c in g]
            N = len(all_cands)
            if N == 0:
                return []

            K_eff = min(cfg.K, N)

            if cfg.replace:
                idx = torch.randint(low=0, high=N, size=(K_eff,)).tolist()
            else:
                idx = torch.randperm(N)[:K_eff].tolist()

            return [all_cands[i] for i in idx]

        # local random prune: one random child per parent
        kept = []
        for g in candidate_groups:
            j = int(torch.randint(low=0, high=len(g), size=(1,)).item())
            kept.append(g[j])

        return kept