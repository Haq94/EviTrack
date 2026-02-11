# utils/builders.py
from __future__ import annotations

from typing import Any, Dict
import copy

from world_model.base import WorldModelConfig, HeadConfig
from world_model.markov import MarkovWorldModel
from world_model.nonmarkov import NonMarkovWorldModel

from proposal import Proposal, ProposalConfig


def _coerce_head_config(x: Any) -> HeadConfig:
    """Convert dict -> HeadConfig (pass through if already HeadConfig)."""
    if isinstance(x, HeadConfig):
        return x
    if isinstance(x, dict):
        return HeadConfig(**x)
    raise TypeError(f"Expected HeadConfig or dict, got {type(x)}")


def _rehydrate_wm_cfg(cfg_dict: Dict[str, Any]) -> WorldModelConfig:
    d = copy.deepcopy(cfg_dict)
    d.pop("kind", None)

    # nested dataclasses
    if "transition" in d:
        d["transition"] = _coerce_head_config(d["transition"])
    if "emission" in d:
        d["emission"] = _coerce_head_config(d["emission"])

    return WorldModelConfig(**d)


def _rehydrate_proposal_cfg(cfg_dict: Dict[str, Any]) -> ProposalConfig:
    d = copy.deepcopy(cfg_dict)
    if "head" in d:
        d["head"] = _coerce_head_config(d["head"])
    return ProposalConfig(**d)


def build_world_model(cfg_dict: Dict[str, Any]):
    if "kind" not in cfg_dict:
        raise ValueError("wm_config.json missing key 'kind' (e.g. 'markov' or 'nonmarkov').")

    kind = str(cfg_dict["kind"]).lower()
    cfg = _rehydrate_wm_cfg(cfg_dict)

    if kind in ("markov", "m"):
        return MarkovWorldModel(cfg)
    if kind in ("nonmarkov", "non-markov", "nm"):
        return NonMarkovWorldModel(cfg)

    raise ValueError(f"Unknown WM kind='{cfg_dict['kind']}'")


def build_proposal(cfg_dict: Dict[str, Any], wm):
    cfg = _rehydrate_proposal_cfg(cfg_dict)
    return Proposal(cfg, wm=wm)


def make_wm_config_dict(cfg: WorldModelConfig, *, kind: str) -> Dict[str, Any]:
    # WorldModelConfig is a dataclass; __dict__ keeps nested dataclasses as objects.
    # We want JSONable dicts, so convert nested HeadConfig to dicts.
    d = cfg.__dict__.copy()
    d["kind"] = kind
    d["transition"] = cfg.transition.__dict__.copy()
    d["emission"] = cfg.emission.__dict__.copy()
    return d


def make_proposal_config_dict(cfg: ProposalConfig) -> Dict[str, Any]:
    d = cfg.__dict__.copy()
    d["head"] = cfg.head.__dict__.copy()
    return d
