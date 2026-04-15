# inference/evitrack.py
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import torch

from .base import InferenceEngine
from .types import Hypothesis, EviTrackState, StepStats, CostCounter
from .utils import topk_per_batch, stack_scores, normalize_logweights
# from .resampling import effective_sample_size

Tensor = torch.Tensor


# @dataclass
# class EviTrackConfig:
#     K: int
#     C: int
#     tau: int = 1
#     G: int = 1
#     expand: str = "transition"      # "proposal" or "transition"

#     # scoring used for pruning
#     prune_score: str = "evidence"  # "evidence" | "joint" | "tbd_joint"

#     # scoring used for predictive mixture weights
#     weight_mode: str = "evidence"  # "evidence" | "joint" | "tbd_joint"

#     # background latent random-walk std for TBD score
#     sigma_bg: float = 1.0

#     # ESS-based adaptive global pruning (alternative to fixed G)
#     use_ess_trigger: bool = False
#     ess_threshold_frac: float = 0.5  # trigger global pruning when ESS < ess_threshold_frac * K

@dataclass
class EviTrackConfig:
    K: int
    C: int
    tau: int = 1
    G: int = 1
    expand: str = "transition"      # "proposal" or "transition"

    # scoring used for pruning
    prune_score: str = "evidence"   # "evidence" | "joint" | "tbd_joint"

    # scoring used for predictive mixture weights
    weight_mode: str = "evidence"   # "evidence" | "joint" | "tbd_joint"

    # background latent random-walk std for TBD score
    sigma_bg: float = 1.0

    # Global pruning trigger
    global_trigger_mode: str = "constant"   # "constant" | "max" | "entropy"
    global_trigger_source: str = "parents"  # "parents" | "children"

    # Used when global_trigger_mode == "max"
    max_weight_threshold: float = 0.9

    # Used when global_trigger_mode == "entropy"
    entropy_threshold: float = 0.2          # normalized entropy threshold
    normalize_entropy: bool = True


class EviTrackEngine(InferenceEngine):
    def __init__(self, *, wm, proposal, cfg: EviTrackConfig):
        super().__init__(wm=wm, proposal=proposal, cfg=cfg)
        assert cfg.expand in ("proposal", "transition")
        assert cfg.prune_score in ("evidence", "joint", "tbd_joint")
        assert cfg.weight_mode in ("evidence", "joint", "tbd_joint")
        assert cfg.sigma_bg > 0.0, "sigma_bg must be positive."

        assert cfg.global_trigger_mode in ("constant", "max", "entropy")
        assert cfg.global_trigger_source in ("parents", "children")
        assert 0.0 < cfg.max_weight_threshold <= 1.0
        assert cfg.entropy_threshold >= 0.0
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
            J_tbd0 = torch.zeros((), device=device, dtype=dtype)

            h0 = Hypothesis(
                z_t=z0,
                wm_z_state=wm_z_state,
                wm_x_state=wm_x_state,
                q_z_state=q_z_state,
                q_x_state=q_x_state,
                J=J0,
                E=E0,
                J_tbd=J_tbd0,
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
            # kept = self.prune(t_new=t_new, candidate_groups=candidate_groups)
            kept = self.prune(t_new=t_new, candidate_groups=candidate_groups, parent_beam=beam)
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
                if cfg.weight_mode == "joint":
                    logw[b, k] = h.J
                elif cfg.weight_mode == "evidence":
                    logw[b, k] = h.E
                elif cfg.weight_mode == "tbd_joint":
                    logw[b, k] = h.J_tbd
                else:
                    raise ValueError(f"Unknown weight_mode={cfg.weight_mode}")

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
        J_tbd_prev = parent.J_tbd

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

        if t == 1:
            logp_bg_z1 = self._log_prob_bg_initial(z_t)
            J_tbd = J_tbd_prev + (logpxt + logpzt - logp_bg_z1).squeeze()
        else:
            logp_bg_zt = self._log_prob_bg_transition(z_t, z_prev)  # [1]
            J_tbd = J_tbd_prev + (logpxt + logpzt - logp_bg_zt).squeeze()

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
            J_tbd=J_tbd,
        )

    # ------------------------
    # internal: pruning
    # ------------------------
    def _get_score(self, h: Hypothesis, mode: str) -> Tensor:
        if mode == "joint":
            return h.J
        if mode == "evidence":
            return h.E
        if mode == "tbd_joint":
            return h.J_tbd
        raise ValueError(f"Unknown score mode={mode}")

    def _normalized_weights_from_hyps(self, hyps: List[Hypothesis], mode: str) -> Tensor:
        logw = torch.stack([self._get_score(h, mode) for h in hyps], dim=0)  # [N]
        return normalize_logweights(logw, dim=0)

    def _entropy_from_weights(self, w: Tensor) -> Tensor:
        eps = 1e-12
        w_safe = w.clamp_min(eps)
        H = -(w_safe * torch.log(w_safe)).sum()

        if self.cfg.normalize_entropy:
            N = w.numel()
            if N <= 1:
                return torch.zeros((), device=w.device, dtype=w.dtype)
            H_max = torch.log(torch.tensor(float(N), device=w.device, dtype=w.dtype))
            return H / H_max

        return H

    def prune(
        self,
        *,
        t_new: int,
        candidate_groups: List[List[Hypothesis]],
        parent_beam: List[Hypothesis],
    ) -> List[Hypothesis]:
        """
        candidate_groups: list of groups, one per parent.
        group_i = children of parent i  (length C)

        parent_beam: current retained beam before expansion

        LOCAL:
            keep best child per parent

        GLOBAL:
            flatten all children and keep top-K

        Trigger modes:
            - constant: use fixed G
            - max: trigger global if max normalized weight exceeds threshold
            - entropy: trigger global if normalized entropy drops below threshold

        Trigger sources:
            - parents: compute trigger statistic on current beam
            - children: compute trigger statistic on candidate pool
        """
        cfg = self.cfg

        all_cands = [c for g in candidate_groups for c in g]

        # ---------------------------------------------------
        # Decide trigger set: parents or children
        # ---------------------------------------------------
        if cfg.global_trigger_source == "parents":
            trigger_hyps = parent_beam
        elif cfg.global_trigger_source == "children":
            trigger_hyps = all_cands
        else:
            raise ValueError(f"Unknown global_trigger_source={cfg.global_trigger_source}")

        # ---------------------------------------------------
        # Determine if we should do global pruning
        # ---------------------------------------------------
        if cfg.global_trigger_mode == "constant":
            do_global = (t_new == 1) or (t_new % cfg.G == 0)

        else:
            w_trigger = self._normalized_weights_from_hyps(trigger_hyps, cfg.weight_mode)

            if cfg.global_trigger_mode == "max":
                max_w = torch.max(w_trigger)
                do_global = (t_new == 1) or (max_w > cfg.max_weight_threshold)

            elif cfg.global_trigger_mode == "entropy":
                H = self._entropy_from_weights(w_trigger)
                do_global = (t_new == 1) or (H < cfg.entropy_threshold)

            else:
                raise ValueError(f"Unknown global_trigger_mode={cfg.global_trigger_mode}")

        # ---------------------------------------------------
        # Global prune: top-K over all children
        # ---------------------------------------------------
        if do_global:
            scores = torch.stack([self._get_score(c, cfg.prune_score) for c in all_cands], dim=0)
            K_eff = min(cfg.K, scores.shape[0])
            idx = torch.topk(scores, k=K_eff, largest=True).indices.tolist()
            return [all_cands[i] for i in idx]

        # ---------------------------------------------------
        # Local prune: best child per parent
        # ---------------------------------------------------
        kept = []
        for g in candidate_groups:
            scores = torch.stack([self._get_score(c, cfg.prune_score) for c in g], dim=0)
            best = int(torch.argmax(scores).item())
            kept.append(g[best])

        return kept

    # def prune(self, *, t_new: int, candidate_groups: List[List[Hypothesis]]) -> List[Hypothesis]:
    #     """
    #     candidate_groups: list of groups, one per parent.
    #     group_i = children of parent i  (length C)

    #     LOCAL (default between globals):
    #     each parent keeps its top-1 child (by score)

    #     GLOBAL (triggered by fixed G or adaptive ESS):
    #     flatten all children and keep top-K

    #     ESS trigger: compute ESS from current beam weights and trigger global
    #     pruning when ESS < ess_threshold_frac * K
    #     """
    #     cfg = self.cfg

    #     # def get_score(h: Hypothesis, mode: str) -> Tensor:
    #     #     """Get score from hypothesis based on specified mode."""
    #     #     if mode == "joint":
    #     #         return h.J
    #     #     if mode == "evidence":
    #     #         return h.E
    #     #     if mode == "tbd_joint":
    #     #         return h.J_tbd
    #     #     raise ValueError(f"Unknown score mode={mode}")

    #     # Determine if we should do global pruning
    #     if cfg.use_ess_trigger:
    #         # Adaptive ESS-based trigger
    #         # Compute ESS from candidate weights
    #         all_cands = [c for g in candidate_groups for c in g]

    #         # Compute weights from scores
    #         logw = torch.stack([self._get_score(c, cfg.weight_mode) for c in all_cands], dim=0)  # [N_cand]
    #         w = normalize_logweights(logw, dim=0)
    #         ess = effective_sample_size(w)

    #         # Trigger global pruning if ESS is low
    #         do_global = (t_new == 1) or (ess < cfg.ess_threshold_frac * cfg.K)
    #     else:
    #         # Fixed-frequency global pruning (original behavior)
    #         do_global = (t_new == 1) or (t_new % cfg.G == 0)

    #     if do_global:
    #         all_cands = [c for g in candidate_groups for c in g]
    #         scores = torch.stack([self._get_score(c, cfg.prune_score) for c in all_cands], dim=0)  # [Ncand]
    #         K_eff = min(cfg.K, scores.shape[0])
    #         idx = torch.topk(scores, k=K_eff, largest=True).indices.tolist()
    #         return [all_cands[i] for i in idx]

    #     # local: keep best child per parent
    #     kept = []
    #     for g in candidate_groups:
    #         scores = torch.stack([self._get_score(c, cfg.prune_score) for c in g], dim=0)  # [C]
    #         best = int(torch.argmax(scores).item())
    #         kept.append(g[best])
    #     return kept

    def _log_prob_bg_initial(self, z_1: Tensor) -> Tensor:
        sigma = float(self.cfg.sigma_bg)
        dz = z_1.shape[-1]
        sq = torch.sum(z_1 * z_1, dim=-1)
        log_norm = dz * torch.log(
            torch.tensor(2.0 * torch.pi * sigma * sigma, device=z_1.device, dtype=z_1.dtype)
        )
        return -0.5 * (sq / (sigma * sigma) + log_norm)

    def _log_prob_bg_transition(
        self,
        z_t: Tensor,
        z_prev: Tensor,
        ) -> Tensor:
        """
        Background latent dynamics:
            p_bg(z_t | z_{t-1}) = N(z_{t-1}, sigma_bg^2 I)

        Args:
            z_t:    [1, dz]
            z_prev: [1, dz]

        Returns:
            logp_bg: [1]
        """
        sigma = float(self.cfg.sigma_bg)
        dz = z_t.shape[-1]

        diff = z_t - z_prev
        sq = torch.sum(diff * diff, dim=-1)  # [1]

        log_norm = dz * torch.log(
            torch.tensor(2.0 * torch.pi * sigma * sigma, device=z_t.device, dtype=z_t.dtype)
        )
        return -0.5 * (sq / (sigma * sigma) + log_norm)