"""Installed command wrappers."""

from __future__ import annotations

import runpy
from pathlib import Path

from .io import project_root


def prepare_data_main() -> None:
    runpy.run_path(str(project_root() / "scripts" / "prepare_data.py"), run_name="__main__")
