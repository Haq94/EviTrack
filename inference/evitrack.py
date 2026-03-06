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
            z0 = torch.zeros((1,dz), device=device, dtype=dtype)
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
        B = x_t.shape[0]
        t_new = state.t + 1

        new_hyps: List[List[Hypothesis]] = []
        total_candidates = 0
        total_kept = 0

        for b in range(B):
            kept_b, cand_b = self._step_one(
                beam=state.hyps[b],
                x_tb=x_t[b:b+1],      # [1, dx]
                t_new=t_new,
                cost=state.cost,      # global counter (counts all batch elems)
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
        x_tb: Tensor,        # [1, dx]
        t_new: int,
        cost: CostCounter,
        ) -> Tuple[List[Hypothesis], int]:
        """
        Advances a single batch element (one sequence) by one time step.

        Returns:
        kept: List[Hypothesis] (new beam for this example)
        num_candidates: int
        """
        cfg = self.cfg

        # build candidate groups (one list per parent)
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

        # NOTE: remove tau logic later if you want no prune delay.
        do_prune = (t_new % cfg.tau == 0)
        if do_prune:
            kept = self.prune(t_new=t_new, candidate_groups=candidate_groups)
        else:
            kept = [c for g in candidate_groups for c in g]

        return kept, num_candidates

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

        Sampling rule (per your intent):
        - expand="proposal": use proposal for all t (including t==1).
        - expand="transition": use prior for t==1, transition for t>1.

        Scoring:
        - evidence: accumulates log p(x_t | z_t, x_<t)
        - joint:    evidence + log p(z_t | z_{t-1}) (or log p(z1))
        """

        cfg = self.cfg
        assert x_t.shape[0] == 1, "EviTrack _expand_one expects batch=1 tensor [1, dx]"

        # --- unpack parent ---
        z_prev = parent.z_t                  # [1, dz] (dummy z0 for t==1 is OK)
        wm_z_state_prev = parent.wm_z_state  # None (Markov) or GRU state (NonMarkov)
        wm_x_state_prev = parent.wm_x_state  # None/markov/memory state
        q_z_state_prev = parent.q_z_state    # None (z_mode=markov) or GRU state
        q_x_state_prev = parent.q_x_state    # None/markov/memory state

        J_prev = parent.J
        E_prev = parent.E

        # ---------------------------------------------------
        # 1) Sample z_t
        # ---------------------------------------------------
        if cfg.expand == "proposal":
            # Proposal: forecasting-causal, depends only on x_{<t} via q_x_state_prev
            cost.add_proposal(1)
            q_out = self.proposal.propose(
                B=1,
                z_prev=None if t == 1 else z_prev,      # proposal supports z_prev=None at t==1 :contentReference[oaicite:7]{index=7}
                z_state_prev=q_z_state_prev,
                x_state_prev=q_x_state_prev,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            z_t = q_out["z_t"]                 # [1, dz]
            q_z_state_t = q_out["z_state_t"]   # updated AFTER sampling z_t (if memory) :contentReference[oaicite:8]{index=8}
            trans_params = None                # Needed for accurate counter book keeping
        else:
            # Transition expansion: prior at t==1, transition at t>1
            if t == 1:
                z_t = self.wm.sample_z1(1, device=x_t.device, dtype=x_t.dtype)  
            else:
                cost.add_transition(1)
                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state_prev)  
                z_t = self.wm.sample_transition(trans_params)  

            # proposal z-state doesn't change if we didn't use proposal to sample
            q_z_state_t = q_z_state_prev

        # ---------------------------------------------------
        # 2) Joint latent term log p(z_t | ...)
        # ---------------------------------------------------
        if t == 1:
            # Always model prior for joint term
            logpzt = self.wm.log_prob_z1(z_t)  
        else:
            if trans_params is None:
                # Always model transition for joint term (even if sampled from proposal)
                cost.add_transition(1)
                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state_prev) 
            logpzt = self.wm.log_prob_transition(z_t, trans_params)  

        # ---------------------------------------------------
        # 3) Emission likelihood log p(x_t | z_state_curr, x_state_prev)
        # ---------------------------------------------------
        cost.add_emission(1)
        z_state_curr = self.wm.z_state_curr(wm_z_state_prev, z_t)  # Markov: z_t; NonMarkov: GRU(z_state_prev,z_t) 
        emit_params = self.wm.emission_params(z_state_curr=z_state_curr, x_state_prev=wm_x_state_prev)  
        logpxt = self.wm.log_prob_emission(x_t, emit_params)  

        # ---------------------------------------------------
        # 4) Accumulate scores
        # ---------------------------------------------------
        E_t = E_prev + logpxt.squeeze()
        J_t = J_prev + (logpxt + logpzt).squeeze()

        # ---------------------------------------------------
        # 5) Update stored WM states AFTER observing (z_t, x_t)
        # ---------------------------------------------------
        # Note: Markov WM returns None for z_state_t by default; NonMarkov updates GRU state 
        wm_z_state_t = self.wm.update_z_state(wm_z_state_prev, z_t)  
        wm_x_state_t = self.wm.update_x_state(wm_x_state_prev, x_t)  

        # ---------------------------------------------------
        # 6) Update proposal x-state AFTER observing x_t (forecasting-causal)
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
        do_global = (t_new == 1) or (t_new % cfg.G == 0)

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