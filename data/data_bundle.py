# data/data_bundle.py

from dataclasses import dataclass
from typing import Optional, Any
from torch.utils.data import DataLoader


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: Optional[DataLoader] = None

    # metadata (task info, latent dim, etc.)
    meta: dict[str, Any] | None = None

    # describes what batch contains
    # e.g. ("x",) or ("x","z")
    batch_keys: tuple[str, ...] = ("x",)