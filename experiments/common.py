# experiments/common.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InferenceSweepConfig:
    evitrack: List[Dict[str, Any]] = field(default_factory=list)
    particle_filter: List[Dict[str, Any]] = field(default_factory=list)
    random_beam: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentSeeds:
    model_seeds: List[int] = field(default_factory=lambda: [0])
    data_seeds: List[int] = field(default_factory=lambda: [0])
    inference_seeds: List[int] = field(default_factory=lambda: [0])


@dataclass
class ExperimentSpec:
    name: str
    kind: str
    enabled: bool = True
    run_root: str = "results"

    # optional knobs
    device: str = "cpu"
    dtype: str = "float32"

    seeds: ExperimentSeeds = field(default_factory=ExperimentSeeds)
    params: Dict[str, Any] = field(default_factory=dict)
    inference: InferenceSweepConfig = field(default_factory=InferenceSweepConfig)