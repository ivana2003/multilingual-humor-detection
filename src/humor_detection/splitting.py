"""Deterministic, stratified 70/10/20 splitting with leakage checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .io import atomic_write_json, sha256_file, sha256_strings

SCHEMA_VERSION = 1


def make_split_ids(frame: pd.DataFrame, seed: int = 42) -> dict[str, list[str]]:
    """Create deterministic stratified train/validation/test identifier lists."""
    if not {"id", "label"}.issubset(frame.columns):
        raise ValueError("Splitting requires id and label columns")
    ids = frame["id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("Identifiers must be unique before splitting")
    labels = frame["label"].astype(int)
    if not set(labels).issubset({0, 1}):
        raise ValueError("Labels must be binary before splitting")

    train_ids, remainder_ids, train_y, remainder_y = train_test_split(
        ids,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    validation_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=2 / 3,
        random_state=seed,
        stratify=remainder_y,
    )
    splits = {
        "train": train_ids.astype(str).tolist(),
        "validation": validation_ids.astype(str).tolist(),
        "test": test_ids.astype(str).tolist(),
    }
    validate_split_ids(splits, expected_ids=set(ids))
    return splits


def validate_split_ids(splits: dict[str, list[str]], expected_ids: set[str] | None = None) -> None:
    """Verify uniqueness, disjointness, and optional full coverage."""
    expected_names = {"train", "validation", "test"}
    if set(splits) != expected_names:
        raise ValueError(f"Split keys must be {sorted(expected_names)}")
    sets = {name: set(map(str, values)) for name, values in splits.items()}
    for name, values in splits.items():
        if len(values) != len(sets[name]):
            raise ValueError(f"Duplicate identifiers inside {name} split")
    overlaps = {
        "train_validation": sets["train"] & sets["validation"],
        "train_test": sets["train"] & sets["test"],
        "validation_test": sets["validation"] & sets["test"],
    }
    nonempty = {key: sorted(value)[:5] for key, value in overlaps.items() if value}
    if nonempty:
        raise ValueError(f"Split overlap detected: {nonempty}")
    union = sets["train"] | sets["validation"] | sets["test"]
    if expected_ids is not None and union != set(map(str, expected_ids)):
        missing = sorted(set(map(str, expected_ids)) - union)[:5]
        extra = sorted(union - set(map(str, expected_ids)))[:5]
        raise ValueError(f"Split coverage mismatch; missing={missing}, extra={extra}")


def split_frames(frame: pd.DataFrame, splits: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Materialize split dataframes in manifest order."""
    validate_split_ids(splits, expected_ids=set(frame["id"].astype(str)))
    indexed = frame.assign(id=frame["id"].astype(str)).set_index("id", drop=False)
    return {name: indexed.loc[ids].reset_index(drop=True) for name, ids in splits.items()}


def build_manifest(
    dataset: str,
    frame: pd.DataFrame,
    source_path: Path,
    *,
    source_reference: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a complete split manifest with counts and content hashes."""
    splits = make_split_ids(frame, seed=seed)
    materialized = split_frames(frame, splits)
    summaries: dict[str, Any] = {}
    for name, part in materialized.items():
        counts = part["label"].value_counts().reindex([0, 1], fill_value=0)
        summaries[name] = {
            "size": len(part),
            "class_counts": {"0": int(counts.loc[0]), "1": int(counts.loc[1])},
            "humorous_fraction": float(part["label"].mean()),
            "ids_sha256": sha256_strings(splits[name]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "seed": seed,
        "ratios": {"train": 0.70, "validation": 0.10, "test": 0.20},
        "source": {
            "path": source_reference or source_path.name,
            "sha256": sha256_file(source_path),
            "rows": len(frame),
        },
        "summaries": summaries,
        "splits": splits,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Validate and atomically write a split manifest."""
    validate_split_ids(manifest["splits"])
    atomic_write_json(path, manifest)
