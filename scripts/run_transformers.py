#!/usr/bin/env python3
"""Fine-tune a paper-reported encoder and evaluate the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from humor_detection.data import load_normalized_dataset
from humor_detection.io import atomic_write_json, append_jsonl, load_yaml, project_root, sha256_file
from humor_detection.metrics import binary_classification_metrics
from humor_detection.reproducibility import run_metadata, set_seed
from humor_detection.splitting import split_frames, validate_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["en", "es", "pt", "zh", "hi_en"])
    parser.add_argument("--model", required=True, choices=["xlm-r", "language-specific"])
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_parts(root: Path, dataset: str):
    frame = load_normalized_dataset(dataset, root)
    manifest_path = root / "data" / "splits" / f"{dataset}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    splits = json.loads(manifest_path.read_text(encoding="utf-8"))["splits"]
    validate_split_ids(splits)
    return split_frames(frame, splits), manifest_path


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Transformer dependencies are missing. Install requirements-transformers.txt."
        ) from exc

    if torch.cuda.device_count() > 1:
        raise RuntimeError(
            "The published configuration uses one GPU. Set CUDA_VISIBLE_DEVICES to expose exactly one GPU."
        )
    root = args.root.resolve()
    config = load_yaml(root / "configs" / "transformers.yaml")
    seed = int(config["seed"])
    set_seed(seed)
    if args.model == "xlm-r":
        model_config = config["models"]["xlm-r"]
        paper_name = "XLM-R-base"
    else:
        model_config = config["models"]["language_specific"][args.dataset]
        paper_name = model_config["paper_name"]
    model_id = model_config["model_id"]
    parts, manifest_path = load_parts(root, args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_length = int(config["training"]["max_length"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    datasets = {}
    for name, frame in parts.items():
        ds = Dataset.from_pandas(frame.rename(columns={"label": "labels"}), preserve_index=False)
        datasets[name] = ds.map(tokenize, batched=True, remove_columns=["id", "text"])

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    destination = args.output_dir or root / "results" / "reproduced" / "transformers" / args.dataset / args.model
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}; pass --overwrite to replace it")
    destination.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {"macro_f1": binary_classification_metrics(labels.tolist(), predictions.tolist())["macro_f1"]}

    use_fp16 = bool(config["training"]["mixed_precision_when_available"] and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=str(destination / "checkpoints"),
        learning_rate=float(model_config["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["batch_size"]),
        per_device_eval_batch_size=int(config["training"]["batch_size"]),
        num_train_epochs=float(config["training"]["epochs"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        weight_decay=0.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        seed=seed,
        data_seed=seed,
        report_to=[],
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    output = trainer.predict(datasets["test"])
    predictions = np.argmax(output.predictions, axis=-1).astype(int).tolist()
    gold = parts["test"]["label"].astype(int).tolist()
    metrics = binary_classification_metrics(gold, predictions)
    prediction_path = destination / "predictions.jsonl"
    prediction_path.unlink(missing_ok=True)
    for row, prediction in zip(parts["test"].itertuples(index=False), predictions, strict=True):
        append_jsonl(prediction_path, {"id": str(row.id), "gold": int(row.label), "prediction": prediction})
    result = run_metadata(
        family="transformer",
        dataset=args.dataset,
        model=paper_name,
        model_id=model_id,
        split="held_out_test",
        split_manifest={"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)},
        hyperparameters={
            "learning_rate": float(model_config["learning_rate"]),
            "batch_size": int(config["training"]["batch_size"]),
            "epochs": int(config["training"]["epochs"]),
            "max_length": max_length,
            "warmup_ratio": float(config["training"]["warmup_ratio"]),
            "paper_reported_warmup_range": config["training"]["paper_reported_warmup_range"],
            "fp16": use_fp16,
        },
        metrics={"test": metrics},
        predictions_file=prediction_path.name,
    )
    atomic_write_json(destination / "result.json", result)
    print(destination / "result.json")


if __name__ == "__main__":
    main()
