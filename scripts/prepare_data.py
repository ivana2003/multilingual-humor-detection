#!/usr/bin/env python3
"""Normalize a downloaded dataset and generate its deterministic split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from humor_detection.data import (
    dataset_config,
    map_spanish_rating,
    normalize_binary_label,
    validate_normalized_frame,
)
from humor_detection.io import project_root
from humor_detection.splitting import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["en", "es", "pt", "zh", "hi_en"])
    parser.add_argument("--input", required=True, type=Path, help="Downloaded CSV, TSV, JSON, or JSONL file")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--id-column", help="Existing stable identifier column; otherwise row-based IDs are created")
    parser.add_argument("--separator", help="Explicit CSV/TSV separator")
    parser.add_argument("--output-root", type=Path, default=project_root())
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_table(path: Path, separator: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = separator or ("\t" if suffix == ".tsv" else ",")
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    source = read_table(args.input, args.separator)
    missing = [c for c in [args.text_column, args.label_column] if c not in source.columns]
    if args.id_column and args.id_column not in source.columns:
        missing.append(args.id_column)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(set(missing))}")

    identifiers = (
        source[args.id_column]
        if args.id_column
        else pd.Series([f"{args.dataset}-{i:08d}" for i in range(len(source))])
    )
    if args.dataset == "es":
        labels = source[args.label_column].map(map_spanish_rating)
    else:
        labels = source[args.label_column].map(normalize_binary_label)
    normalized = pd.DataFrame({
        "id": identifiers,
        "text": source[args.text_column],
        "label": labels.astype(int),
    })

    config_path = args.output_root / "configs" / "datasets.yaml"
    config = dataset_config(args.dataset, config_path)
    summary = validate_normalized_frame(
        normalized,
        expected_size=int(config["expected_size"]),
        expected_humorous_fraction=float(config["expected_humorous_fraction"]),
    )
    normalized["id"] = normalized["id"].astype(str)
    normalized["text"] = normalized["text"].astype(str)
    output_path = args.output_root / "data" / "processed" / f"{args.dataset}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    manifest = build_manifest(
        args.dataset,
        normalized,
        output_path,
        source_reference=output_path.relative_to(args.output_root).as_posix(),
        seed=args.seed,
    )
    manifest_path = args.output_root / "data" / "splits" / f"{args.dataset}.json"
    write_manifest(manifest_path, manifest)
    print(json.dumps({"dataset": args.dataset, "output": str(output_path), "manifest": str(manifest_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
