# experiments/base_experiment.py
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class BaseExperiment:
    """
    High-level experiment abstraction.

    Each experiment owns:
      - its output directory
      - its config/metadata saving
      - its model/data preparation
      - its training or direct analytic model construction
      - its inference evaluation
    """

    def __init__(self, *, name: str, run_root: str = "results", seed: int = 0, use_seed_dir: bool = True):
        self.name = name
        self.run_root = Path(run_root)
        self.seed = int(seed)

        if use_seed_dir:
            self.run_dir = self.run_root / self.name / f"seed_{self.seed:03d}"
        else:
            self.run_dir = self.run_root / self.name

        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(data), f, indent=2, sort_keys=True)

    def save_metadata(self, meta: Dict[str, Any]) -> None:
        self.save_json(self.run_dir / "experiment_meta.json", meta)

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError