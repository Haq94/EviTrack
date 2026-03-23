# data/dataset_io.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import json

import torch


def save_dataset(dataset: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v for k, v in dataset.items() if k != "meta"},
        path / "data.pt",
    )
    with (path / "metadata.json").open("w") as f:
        json.dump(dataset["meta"], f, indent=2, sort_keys=True)
    print(f"[dataset] Saved to {path}")


def load_dataset(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    d = torch.load(path / "data.pt", map_location=map_location)
    with (path / "metadata.json").open() as f:
        d["meta"] = json.load(f)
    print(f"[dataset] Loaded {d['x'].shape[0]} trajectories from {path}")
    return d