"""Normalized dataset loading and paper-statistic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_yaml, project_root

REQUIRED_COLUMNS = ("id", "text", "label")


def dataset_config(dataset: str, config_path: Path | None = None) -> dict[str, Any]:
    """Return one dataset's canonical configuration."""
    path = config_path or project_root() / "configs" / "datasets.yaml"
    config = load_yaml(path)
    try:
        value = config["datasets"][dataset]
    except KeyError as exc:
        choices = ", ".join(sorted(config.get("datasets", {})))
        raise KeyError(f"Unknown dataset '{dataset}'. Available: {choices}") from exc
    return value


def map_spanish_rating(value: object) -> int:
    """Map paper ratings 1/2/3 to binary labels 0/1/1."""
    try:
        rating = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Spanish rating: {value!r}") from exc
    mapping = {1: 0, 2: 1, 3: 1}
    if rating not in mapping:
        raise ValueError(f"Spanish rating must be one of 1, 2, 3; got {rating}")
    return mapping[rating]


def normalize_binary_label(value: object) -> int:
    """Normalize common binary representations without inventing labels."""
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    mapping = {
        "0": 0, "1": 1,
        "false": 0, "true": 1,
        "not humorous": 0, "non-humorous": 0, "non humorous": 0,
        "humorous": 1, "humor": 1, "non-humor": 0,
    }
    if text not in mapping:
        raise ValueError(f"Cannot safely map label {value!r} to binary 0/1")
    return mapping[text]


def validate_normalized_frame(
    frame: pd.DataFrame,
    *,
    expected_size: int | None = None,
    expected_humorous_fraction: float | None = None,
) -> dict[str, Any]:
    """Validate schema, identifiers, labels, and optional paper statistics."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing normalized columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame["id"].isna().any() or frame["id"].astype(str).str.len().eq(0).any():
        raise ValueError("Identifiers must be non-empty")
    if frame["id"].astype(str).duplicated().any():
        duplicates = frame.loc[frame["id"].astype(str).duplicated(), "id"].head().tolist()
        raise ValueError(f"Dataset identifiers are not unique; examples: {duplicates}")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Texts must be non-empty")
    labels = set(frame["label"].tolist())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError(f"Labels must be binary integers; observed {sorted(labels, key=str)}")
    total = len(frame)
    humorous = int(frame["label"].sum())
    fraction = humorous / total
    if expected_size is not None and total != expected_size:
        raise ValueError(f"Expected {expected_size} samples, found {total}")
    if expected_humorous_fraction is not None:
        observed_percent = round(fraction * 100, 1)
        expected_percent = round(expected_humorous_fraction * 100, 1)
        if observed_percent != expected_percent:
            raise ValueError(
                f"Expected {expected_percent:.1f}% humorous after rounding, found {observed_percent:.1f}%"
            )
    return {
        "total": total,
        "class_counts": {"0": int(total - humorous), "1": humorous},
        "humorous_fraction": fraction,
    }


def load_normalized_dataset(dataset: str, root: Path | None = None) -> pd.DataFrame:
    """Load and validate data/processed/<dataset>.csv against paper statistics."""
    base = root or project_root()
    path = base / "data" / "processed" / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {path}. Run scripts/prepare_data.py first."
        )
    frame = pd.read_csv(path, dtype={"id": str})
    frame["label"] = frame["label"].astype(int)
    config = dataset_config(dataset, base / "configs" / "datasets.yaml")
    validate_normalized_frame(
        frame,
        expected_size=int(config["expected_size"]),
        expected_humorous_fraction=float(config["expected_humorous_fraction"]),
    )
    return frame.loc[:, list(REQUIRED_COLUMNS)].copy()
