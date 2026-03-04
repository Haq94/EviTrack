# inference/evitrack.py
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import torch

from .base import InferenceEngine
from .types import Hypothesis, EviTrackState, StepStats, CostCounter
from .utils import topk_per_batch, stack_scores, normalize_logweights

Tensor = torch.Tensor


@dataclass
class EviTrackConfig:
    K: int
    C: int
    tau: int = 1
    G: int = 1                 # every G steps do GLOBAL prune; otherwise LOCAL prune
    local_keep: Optional[int] = None  # how many children to keep per parent during LOCAL prune (<= C). default: min(C,K)
    expand: str = "proposal"   # "proposal" or "transition"
    prune_score: str = "evidence"  # "evidence" or "joint"
    weight_mode: str = "evidence"   # "evidence" or "joint"

class EviTrackEngine(InferenceEngine):
    def __init__(self, *, wm, proposal, cfg: EviTrackConfig):
        super().__init__(wm=wm, proposal=proposal, cfg=cfg)
        assert cfg.expand in ("proposal", "transition")
        assert cfg.prune_score in ("evidence", "joint")
        assert cfg.weight_mode in ("evidence", "joint")
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
            wm_z_state = wm_z_state_B[b:b+1] if wm_z_state_B is not None else None
            wm_x_state = wm_x_state_B[b:b+1] if wm_x_state_B is not None else None
            q_z_state = q_z_state_B[b:b+1] if q_z_state_B is not None else None
            q_x_state = q_x_state_B[b:b+1] if q_x_state_B is not None else None

            # root has no z yet; keep a dummy z0
            z0 = torch.zeros((dz,), device=device, dtype=dtype)
            J0 = torch.zeros((), device=device, dtype=dtype)  # scalar
            E0 = torch.zeros((), device=device, dtype=dtype)

            h0 = Hypothesis(
                z_t=z0,
                wm_z_state=wm_z_state,
                wm_x_state=wm_x_state,
                q_z_state=q_z_state,
                q_x_state=q_x_state,
                J=J0,
                E=E0,
            )
            hyps.append([h0])  # beam for example b

        return EviTrackState(hyps=hyps, t=0, cost=CostCounter())

    def step(self, state: EviTrackState, x_t: Tensor):
        """
        x_t: [B, dx]
        """
        cfg = self.cfg
        B = x_t.shape[0]
        t_new = state.t + 1

        new_hyps: List[List[Hypothesis]] = []
        total_candidates = 0
        total_kept = 0

        for b in range(B):
            beam = state.hyps[b]
            x_tb = x_t[b:b+1]  # [1, dx]

            # build candidate groups (one list per parent)
            candidate_groups: List[List[Hypothesis]] = []

            if t_new == 1:
                # initialize beam with K children from the root
                root = beam[0]
                group = [self._expand_one(root, x_tb, t_new) for _ in range(cfg.K)]
                candidate_groups = [group]
            else:
                for parent in beam:
                    group = [self._expand_one(parent, x_tb, t_new) for _ in range(cfg.C)]
                    candidate_groups.append(group)

            total_candidates += sum(len(g) for g in candidate_groups)

            do_prune = (t_new % cfg.tau == 0)
            if do_prune:
                kept = self.prune(t_new=t_new, candidate_groups=candidate_groups)
            else:
                kept = [c for g in candidate_groups for c in g]

            total_kept += len(kept)
            new_hyps.append(kept)

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

    def get_mixture(self, state: EviTrackState):
        """
        Returns:
        w: [B, K]  (assumes each beam has length K after pruning)
        support: List[List[Hypothesis]] (beams per example)
        """
        cfg = self.cfg
        B = len(state.hyps)
        assert B > 0

        K = len(state.hyps[0])
        for b in range(B):
            assert len(state.hyps[b]) == K, "All beams must have same K to return tensor weights."

        logw = torch.empty((B, K), dtype=state.hyps[0][0].E.dtype, device=state.hyps[0][0].E.device)

        for b in range(B):
            for k in range(K):
                h = state.hyps[b][k]
                logw[b, k] = h.J if cfg.weight_mode == "joint" else h.E

        w = normalize_logweights(logw, dim=1)  # normalize over k
        return w, state.hyps

    # ------------------------
    # internal: expand a single child
    # ------------------------
    def _expand_one(self, parent: Hypothesis, x_t: Tensor, t: int) -> Hypothesis:
        """
        Single-example expansion.
        x_t: [1, dx]
        parent.z_t: [dz]
        Internally we call WM/proposal with batch=1 tensors.
        """
        cfg = self.cfg

        x_t1 = x_t.unsqueeze(0) if x_t.ndim == 1 else x_t           # [1, dx]
        z_prev1 = parent.z_t if t > 1  else None  # [1, dz] or None
        wm_z_state = parent.wm_z_state
        wm_x_state = parent.wm_x_state
        q_z_state = parent.q_z_state
        q_x_state = parent.q_x_state

        # 1) Sample z_t (batch=1)
        if t == 1:
            if cfg.expand == "proposal":
                q_out = self.proposal.propose(
                    B=1,
                    z_prev=None,
                    z_state_prev=q_z_state,
                    x_state_prev=q_x_state,
                    device=x_t1.device,
                    dtype=x_t1.dtype,
                )
                z_t1 = q_out["z_t"]              # [1, dz]
                q_z_state = q_out["z_state_t"]
            else:
                z_t1 = self.wm.sample_z1(1, device=x_t1.device, dtype=x_t1.dtype)

            logpzt1 = self.wm.log_prob_z1(z_t1)  # [1]
        else:
            if cfg.expand == "transition":
                trans_params = self.wm.transition_params(z_prev=z_prev1, z_state_prev=wm_z_state)
                z_t1 = self.wm.sample_transition(trans_params)
                logpzt1 = self.wm.log_prob_transition(z_t1, trans_params)
            else:
                q_out = self.proposal.propose(
                    B=1,
                    z_prev=z_prev1,
                    z_state_prev=q_z_state,
                    x_state_prev=q_x_state,
                    device=x_t1.device,
                    dtype=x_t1.dtype,
                )
                z_t1 = q_out["z_t"]
                q_z_state = q_out["z_state_t"]

                trans_params = self.wm.transition_params(z_prev=z_prev1, z_state_prev=wm_z_state)
                logpzt1 = self.wm.log_prob_transition(z_t1, trans_params)

        # 2) Emission likelihood
        z_state_curr = self.wm.z_state_curr(wm_z_state, z_t1)
        emit_params = self.wm.emission_params(z_state_curr=z_state_curr, x_state_prev=wm_x_state)
        logpxt1 = self.wm.log_prob_emission(x_t1, emit_params)  # [1]

        # 3) Update accumulated scores (scalars)
        E_new = parent.E + logpxt1.squeeze(0)
        J_new = parent.J + (logpxt1 + logpzt1).squeeze(0)

        # 4) Update stored states AFTER using x_t
        wm_z_state_new = self.wm.update_z_state(wm_z_state, z_t1)
        wm_x_state_new = self.wm.update_x_state(wm_x_state, x_t1)

        q_x_state_new = None
        if self.proposal is not None:
            q_x_state_new = self.proposal.update_x_state(x_t=x_t1, x_state_prev=q_x_state)

        return Hypothesis(
            z_t=z_t1,          # [1, dz]
            wm_z_state=wm_z_state_new,    # batch=1 state
            wm_x_state=wm_x_state_new,
            q_z_state=q_z_state,
            q_x_state=q_x_state_new,
            J=J_new,
            E=E_new,
        )
    
    # ------------------------
    # internal: pruning
    # ------------------------
    def prune(self, *, t_new: int, candidate_groups: List[List[Hypothesis]]) -> List[Hypothesis]:
        """
        candidate_groups: list of groups, one per parent.
        group_i = children of parent i  (length C)

        LOCAL (default between globals):
        each parent keeps its top-1 child (by score)

        GLOBAL (every G steps):
        flatten all children and keep top-K
        """
        cfg = self.cfg
        do_global = (cfg.G == 1) or (t_new % cfg.G == 0)

        def get_score(h: Hypothesis) -> Tensor:
            return h.J if cfg.prune_score == "joint" else h.E

        if do_global:
            all_cands = [c for g in candidate_groups for c in g]
            scores = torch.stack([get_score(c) for c in all_cands], dim=0)  # [Ncand]
            K_eff = min(cfg.K, scores.shape[0])
            idx = torch.topk(scores, k=K_eff, largest=True).indices.tolist()
            return [all_cands[i] for i in idx]

        # local: keep best child per parent
        kept = []
        for g in candidate_groups:
            scores = torch.stack([get_score(c) for c in g], dim=0)  # [C]
            best = int(torch.argmax(scores).item())
            kept.append(g[best])
        return kept