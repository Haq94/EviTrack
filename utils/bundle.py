# utils/bundle.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, sort_keys=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_build_wm_fn(cfg_dict: Dict[str, Any]) -> nn.Module:
    from world_model.base import WorldModelConfig, HeadConfig
    from world_model.markov import MarkovWorldModel
    from world_model.nonmarkov import NonMarkovWorldModel

    if "kind" not in cfg_dict:
        raise ValueError(
            "wm_config.json missing key 'kind'. Add kind='markov' or kind='nonmarkov' when saving."
        )

    kind = str(cfg_dict["kind"]).lower()
    d = dict(cfg_dict)
    d.pop("kind", None)

    # Rehydrate nested dataclasses
    if isinstance(d.get("transition"), dict):
        d["transition"] = HeadConfig(**d["transition"])
    if isinstance(d.get("emission"), dict):
        d["emission"] = HeadConfig(**d["emission"])

    cfg = WorldModelConfig(**d)

    if kind in ("markov", "m"):
        return MarkovWorldModel(cfg)
    if kind in ("nonmarkov", "non-markov", "nm"):
        return NonMarkovWorldModel(cfg)

    raise ValueError(f"Unknown WM kind='{cfg_dict['kind']}'")


def _default_build_proposal_fn(cfg_dict: Dict[str, Any], wm: nn.Module) -> nn.Module:
    from proposal import Proposal, ProposalConfig
    from world_model.base import HeadConfig

    d = dict(cfg_dict)
    if isinstance(d.get("head"), dict):
        d["head"] = HeadConfig(**d["head"])

    cfg = ProposalConfig(**d)
    return Proposal(cfg, wm=wm)


class ModelBundle:
    """
    Composite artifact: (wm, proposal) + configs + metadata.
    """

    def __init__(
        self,
        *,
        wm: nn.Module,
        proposal: Optional[nn.Module],
        wm_config: Any,
        proposal_config: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.wm = wm
        self.proposal = proposal
        self.wm_config = wm_config
        self.proposal_config = proposal_config
        self.meta = meta or {}

    def to(self, *, device=None, dtype=None) -> "ModelBundle":
        self.wm.to(device=device, dtype=dtype)
        if self.proposal is not None:
            self.proposal.to(device=device, dtype=dtype)
        return self

    def eval(self) -> "ModelBundle":
        self.wm.eval()
        if self.proposal is not None:
            self.proposal.eval()
        return self

    def save(self, run_dir: Union[str, Path]) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.wm.state_dict(), run_dir / "wm_state.pt")
        if self.proposal is not None:
            torch.save(self.proposal.state_dict(), run_dir / "proposal_state.pt")

        _save_json(run_dir / "wm_config.json", self.wm_config)
        if self.proposal_config is not None:
            _save_json(run_dir / "proposal_config.json", self.proposal_config)
        _save_json(run_dir / "meta.json", self.meta)

    @staticmethod
    def load(
        run_dir: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        strict: bool = True,
        # Optional override builders:
        build_wm_fn: Optional[Callable[[Dict[str, Any]], nn.Module]] = None,
        build_proposal_fn: Optional[Callable[[Dict[str, Any], nn.Module], nn.Module]] = None,
        load_proposal: bool = True,
    ) -> "ModelBundle":
        """
        Load and rebuild modules.

        - If build_wm_fn is None: infer WM class from wm_config["kind"].
        - If build_proposal_fn is None: use default Proposal(cfg, wm).
        - If load_proposal=False: do not load proposal even if present.
        """
        run_dir = Path(run_dir)

        wm_cfg = _load_json(run_dir / "wm_config.json")
        meta = _load_json(run_dir / "meta.json") if (run_dir / "meta.json").exists() else {}

        prop_cfg = None
        prop_cfg_path = run_dir / "proposal_config.json"
        if prop_cfg_path.exists():
            prop_cfg = _load_json(prop_cfg_path)

        # Build WM
        if build_wm_fn is None:
            build_wm_fn = _default_build_wm_fn
        wm = build_wm_fn(wm_cfg)
        wm_sd = torch.load(run_dir / "wm_state.pt", map_location="cpu")
        wm.load_state_dict(wm_sd, strict=strict)

        # Build Proposal (optional)
        proposal = None
        prop_state_path = run_dir / "proposal_state.pt"
        if load_proposal and prop_state_path.exists():
            if prop_cfg is None:
                raise ValueError(
                    "proposal_state.pt exists but proposal_config.json is missing. "
                    "Either save proposal_config.json or load_proposal=False."
                )
            if build_proposal_fn is None:
                build_proposal_fn = _default_build_proposal_fn
            proposal = build_proposal_fn(prop_cfg, wm)
            prop_sd = torch.load(prop_state_path, map_location="cpu")
            proposal.load_state_dict(prop_sd, strict=strict)

        bundle = ModelBundle(
            wm=wm,
            proposal=proposal,
            wm_config=wm_cfg,
            proposal_config=prop_cfg,
            meta=meta,
        )

        if device is not None or dtype is not None:
            bundle.to(device=device, dtype=dtype)

        return bundle
