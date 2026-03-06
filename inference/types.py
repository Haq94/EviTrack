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
    # single-example hypothesis
    z_t: Tensor              # [dz]
    wm_z_state: Any          # WM state for batch=1
    wm_x_state: Any
    q_z_state: Any           # proposal state for batch=1 (or None)
    q_x_state: Any
    J: Tensor                # scalar tensor
    E: Tensor                # scalar tensor

@dataclass
class EviTrackState:
    # hyps[b] is the beam (list of hypotheses) for example b
    hyps: List[List[Hypothesis]]
    t: int
    cost: CostCounter

@dataclass
class Particle:
    # single-example particle
    z_t: Tensor              # [1, dz]
    wm_z_state: Any          # WM state for batch=1
    wm_x_state: Any
    q_z_state: Any
    q_x_state: Any
    logw: Tensor             # scalar tensor

@dataclass
class ParticleState:
    # particles[b] is the particle set (list of particles) for example b
    particles: List[List[Particle]]
    t: int
    cost: CostCounter