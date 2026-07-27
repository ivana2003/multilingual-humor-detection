"""Seed control and run metadata."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions(names: tuple[str, ...] = (
    "numpy", "pandas", "scikit-learn", "PyYAML", "torch", "transformers", "openai"
)) -> dict[str, str]:
    """Return installed versions without importing optional packages."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_metadata(**extra: Any) -> dict[str, Any]:
    """Create machine-readable execution metadata."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        **extra,
    }
