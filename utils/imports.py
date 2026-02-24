# utils/imports.py
from __future__ import annotations

import importlib
from typing import Callable


def load_attr(module_path: str, attr: str):
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ImportError(f"Module '{module_path}' has no attribute '{attr}'") from e


def load_callable(spec: str) -> Callable:
    """
    spec format: "package.module:callable_name"
    Example: "data.tasks.doublewell_1d:build_loaders"
    """
    if ":" not in spec:
        raise ValueError(f"Bad spec '{spec}'. Expected 'module:callable'")
    module_path, fn_name = spec.split(":", 1)
    fn = load_attr(module_path, fn_name)
    if not callable(fn):
        raise TypeError(f"'{spec}' is not callable")
    return fn