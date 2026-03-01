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

# @dataclass
# class EviTrackConfig:
#     K: int                  # keep
#     C: int                  # children per parent
#     tau: int = 1            # prune every tau steps
#     expand: str = "proposal"  # {"proposal","transition"}
#     prune_score: str = "evidence"  # {"evidence","joint"}
#     weight_mode: str = "evidence"  # {"evidence","joint"}

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
        # initialize WM and proposal states once; replicate per-hypothesis
        wm_z_state = self.wm.init_z_state(B, device=device, dtype=dtype)  
        wm_x_state = self.wm.init_x_state(B, device=device, dtype=dtype)  

        if self.proposal is not None:
            q_z_state = self.proposal.init_z_state(B, device=device, dtype=dtype)  
            q_x_state = self.proposal.init_x_state(B, device=device, dtype=dtype)  
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

    def step(self, state: EviTrackState, x_t: Tensor):
        cfg = self.cfg
        t_new = state.t + 1

        # Always create groups per parent (even if we later do global)
        candidate_groups = []

        if t_new == 1:
            # Start with exactly K children from root (your desired semantics)
            root = state.hyps[0]
            group = []
            for _ in range(cfg.K):
                group.append(self._expand_one(root, x_t, t_new))
            candidate_groups = [group]
        else:
            for parent in state.hyps:
                group = []
                for _ in range(cfg.C):
                    group.append(self._expand_one(parent, x_t, t_new))
                candidate_groups.append(group)

        do_prune = (t_new % cfg.tau == 0)
        if do_prune:
            kept = self.prune(t_new=t_new, candidate_groups=candidate_groups)
        else:
            # no pruning => flatten all groups
            kept = [c for g in candidate_groups for c in g]

        state.hyps = kept
        state.t = t_new

        stats = StepStats(t=t_new, kept=len(kept), candidates=sum(len(g) for g in candidate_groups), cost=state.cost)
        return state, stats

    def get_mixture(self, state: EviTrackState):
        if self.cfg.weight_mode == "joint":
            logw = torch.stack([h.J for h in state.hyps], dim=0)  # [K,B]
        else:
            logw = torch.stack([h.E for h in state.hyps], dim=0)  # [K,B]
        w = normalize_logweights(logw, dim=0).T  # [B,K]
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

        # 1) Sample z_t
        if t == 1:
            if cfg.expand == "proposal":
                # proposal for z1 (z_prev=None)
                q_out = self.proposal.propose(
                    B=B,
                    z_prev=None,
                    z_state_prev=q_z_state,
                    x_state_prev=q_x_state,
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                z_t = q_out["z_t"]
                logq_t = q_out["logq"]
                q_z_state = q_out["z_state_t"]
            else:
                z_t = self.wm.sample_z1(B, device=x_t.device, dtype=x_t.dtype)
                logq_t = torch.zeros((B,), device=x_t.device, dtype=x_t.dtype)

            # TODO: This might be an issue when we use the proposal instead of the prior since we are sampling from the proposal but using the prior to calculate the initial logp
            logpzt = self.wm.log_prob_z1(z_t)

        else:
            if cfg.expand == "transition":
                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state)
                z_t = self.wm.sample_transition(trans_params)
                logpzt = self.wm.log_prob_transition(z_t, trans_params)
                logq_t = torch.zeros_like(logpzt)
            else:
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

                trans_params = self.wm.transition_params(z_prev=z_prev, z_state_prev=wm_z_state)
                logpzt = self.wm.log_prob_transition(z_t, trans_params)

        # 2) Emission likelihood
        z_state_curr = self.wm.z_state_curr(wm_z_state, z_t)
        emit_params = self.wm.emission_params(z_state_curr=z_state_curr, x_state_prev=wm_x_state)
        logpxt = self.wm.log_prob_emission(x_t, emit_params)

        # 3) Update accumulated scores
        E_new = parent.E + logpxt
        J_new = parent.J + logpxt + logpzt

        # 4) Update stored states AFTER using x_t
        wm_z_state_new = self.wm.update_z_state(wm_z_state, z_t)
        wm_x_state_new = self.wm.update_x_state(wm_x_state, x_t)

        q_x_state_new = None
        if self.proposal is not None:
            q_x_state_new = self.proposal.update_x_state(x_t=x_t, x_state_prev=q_x_state)

        return Hypothesis(
            z_t=z_t,
            wm_z_state=wm_z_state_new,
            wm_x_state=wm_x_state_new,
            q_z_state=q_z_state,
            q_x_state=q_x_state_new,
            J=J_new,
            E=E_new,
        )
    
    # ------------------------
    # internal: pruning
    # ------------------------
    def prune(self, *, t_new: int, candidate_groups):
        """
        candidate_groups: list of lists.
            Each inner list = children from ONE parent.

        LOCAL prune:
            each parent keeps exactly 1 child (per batch element).

        GLOBAL prune:
            pool all children and keep top-K (per batch element).
        """
        cfg = self.cfg

        # --------------------------------------------------
        # Helper: stack list of hypotheses into [N,B,...]
        # --------------------------------------------------
        def stack_list(xs):
            x0 = xs[0]
            if torch.is_tensor(x0):
                return torch.stack(xs, dim=0)
            if dataclasses.is_dataclass(x0):
                kwargs = {
                    f.name: stack_list([getattr(x, f.name) for x in xs])
                    for f in dataclasses.fields(x0)
                }
                return x0.__class__(**kwargs)
            if isinstance(x0, dict):
                return {k: stack_list([x[k] for x in xs]) for k in x0.keys()}
            if isinstance(x0, (list, tuple)):
                return type(x0)(
                    [stack_list([x[i] for x in xs]) for i in range(len(x0))]
                )
            return x0

        # --------------------------------------------------
        # Helper: gather top-k per batch
        # --------------------------------------------------
        def gather_bk(stacked, idx_bk):
            """
            stacked tensors: [N,B,...]
            idx_bk: [B,K]
            returns list length K of hypotheses with tensors [B,...]
            """
            B, K = idx_bk.shape

            def gather_node(node):
                if torch.is_tensor(node):
                    nodeB = node.transpose(0, 1)  # [B,N,...]
                    idx = idx_bk
                    for _ in range(nodeB.ndim - 2):
                        idx = idx.unsqueeze(-1)
                    idx = idx.expand((B, K) + nodeB.shape[2:])
                    out = torch.gather(nodeB, 1, idx)  # [B,K,...]
                    return out
                if dataclasses.is_dataclass(node):
                    kwargs = {
                        f.name: gather_node(getattr(node, f.name))
                        for f in dataclasses.fields(node)
                    }
                    return node.__class__(**kwargs)
                if isinstance(node, dict):
                    return {k: gather_node(v) for k, v in node.items()}
                if isinstance(node, (list, tuple)):
                    return type(node)([gather_node(v) for v in node])
                return node

            gathered = gather_node(stacked)

            def split_k(node, k):
                if torch.is_tensor(node):
                    return node[:, k]
                if dataclasses.is_dataclass(node):
                    kwargs = {
                        f.name: split_k(getattr(node, f.name), k)
                        for f in dataclasses.fields(node)
                    }
                    return node.__class__(**kwargs)
                if isinstance(node, dict):
                    return {kk: split_k(v, k) for kk, v in node.items()}
                if isinstance(node, (list, tuple)):
                    return type(node)([split_k(v, k) for v in node])
                return node

            return [split_k(gathered, k) for k in range(K)]

        # --------------------------------------------------
        # Score helper
        # --------------------------------------------------
        def score(hyps):
            if cfg.prune_score == "joint":
                return torch.stack([h.J for h in hyps], dim=0)
            else:
                return torch.stack([h.E for h in hyps], dim=0)

        # --------------------------------------------------
        # Decide mode
        # --------------------------------------------------
        do_global = (cfg.G == 1) or (t_new % cfg.G == 0)

        # ==========================
        # LOCAL PRUNE
        # ==========================
        if not do_global:
            kept = []

            for group in candidate_groups:
                S = score(group)             # [C,B]
                idx = torch.topk(S.T, k=1, dim=1).indices  # [B,1]
                stacked = stack_list(group)  # [C,B,...]
                best = gather_bk(stacked, idx)  # length 1 list
                kept.extend(best)

            return kept

        # ==========================
        # GLOBAL PRUNE
        # ==========================
        all_cands = [c for g in candidate_groups for c in g]
        S = score(all_cands)                 # [K*C,B]
        K_eff = min(cfg.K, S.shape[0])
        idx = torch.topk(S.T, k=K_eff, dim=1).indices  # [B,K]
        stacked = stack_list(all_cands)
        return gather_bk(stacked, idx)