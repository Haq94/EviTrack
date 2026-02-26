# inference/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import torch

Tensor = torch.Tensor

@dataclass
class CostCounter:
    transition_evals: int = 0
    proposal_evals: int = 0
    emission_evals: int = 0

    def add_transition(self, n: int = 1): self.transition_evals += n
    def add_proposal(self, n: int = 1): self.proposal_evals += n
    def add_emission(self, n: int = 1): self.emission_evals += n

@dataclass
class StepStats:
    # optional scalar logs for debugging/plots
    t: int
    kept: int
    candidates: int
    cost: CostCounter
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Hypothesis:
    # minimal hypothesis state for EviTrack
    z_t: Tensor              # [B, dz]
    wm_z_state: Any          # whatever WM uses
    wm_x_state: Any
    q_z_state: Any
    q_x_state: Any
    # scores accumulated up to current t
    J: Tensor                # [B] (joint score)
    E: Tensor                # [B] (evidence score)

@dataclass
class EviTrackState:
    hyps: List[Hypothesis]
    t: int
    cost: CostCounter

@dataclass
class Particle:
    z_t: Tensor
    wm_z_state: Any
    wm_x_state: Any
    q_z_state: Any
    q_x_state: Any
    logw: Tensor             # [B] or [B,] per particle

@dataclass
class ParticleState:
    particles: List[Particle]
    t: int
    cost: CostCounter