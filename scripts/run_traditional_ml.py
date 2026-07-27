#!/usr/bin/env python3
"""Run the paper-aligned TF-IDF Logistic Regression or linear SVM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from humor_detection.data import load_normalized_dataset
from humor_detection.io import atomic_write_json, load_yaml, project_root, append_jsonl, sha256_file
from humor_detection.metrics import binary_classification_metrics
from humor_detection.reproducibility import run_metadata, set_seed
from humor_detection.splitting import split_frames, validate_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["en", "es", "pt", "zh", "hi_en"])
    parser.add_argument("--model", required=True, choices=["lr", "svm"])
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_splits(root: Path, dataset: str) -> tuple[dict[str, list[str]], Path]:
    path = root / "data" / "splits" / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}. Run prepare_data.py first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    validate_split_ids(splits)
    return splits, path


def build_estimator(model: str, seed: int) -> object:
    if model == "lr":
        return LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    return LinearSVC(random_state=seed)


def run(dataset: str, model: str, root: Path, output_dir: Path | None = None) -> Path:
    if dataset == "es" and model == "svm":
        raise ValueError("Spanish SVM is not reported in Table 2 and is not part of the primary paper pipeline")
    config = load_yaml(root / "configs" / "traditional_ml.yaml")
    seed = int(config["seed"])
    set_seed(seed)
    frame = load_normalized_dataset(dataset, root)
    splits, manifest_path = load_splits(root, dataset)
    parts = split_frames(frame, splits)
    fit_frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=tuple(config["features"]["ngram_range"]),
            min_df=int(config["features"]["min_df"]),
            lowercase=bool(config["features"]["lowercase"]),
        )),
        ("classifier", build_estimator(model, seed)),
    ])
    folds = StratifiedKFold(
        n_splits=int(config["cross_validation"]["folds"]),
        shuffle=bool(config["cross_validation"]["shuffle"]),
        random_state=seed,
    )
    search = GridSearchCV(
        pipeline,
        param_grid={"classifier__C": list(config["cross_validation"]["c_values"])},
        scoring="f1_macro",
        cv=folds,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(fit_frame["text"], fit_frame["label"])
    predictions = search.predict(parts["test"]["text"]).astype(int).tolist()
    metrics = binary_classification_metrics(parts["test"]["label"].astype(int).tolist(), predictions)

    destination = output_dir or root / "results" / "reproduced" / "traditional_ml" / dataset / model
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "predictions.jsonl"
    prediction_path.unlink(missing_ok=True)
    for row, prediction in zip(parts["test"].itertuples(index=False), predictions, strict=True):
        append_jsonl(prediction_path, {"id": str(row.id), "gold": int(row.label), "prediction": prediction})
    result = run_metadata(
        family="traditional_ml",
        dataset=dataset,
        model=model,
        split="held_out_test",
        split_manifest={"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)},
        hyperparameters={
            "tfidf_ngram_range": config["features"]["ngram_range"],
            "min_df": config["features"]["min_df"],
            "cv_folds": config["cross_validation"]["folds"],
            "c_grid": config["cross_validation"]["c_values"],
            "selected_c": float(search.best_params_["classifier__C"]),
            "final_fit": "train_plus_validation",
        },
        metrics={"test": metrics, "cv_best_macro_f1": float(search.best_score_)},
        predictions_file=prediction_path.name,
    )
    atomic_write_json(destination / "result.json", result)
    return destination / "result.json"


def main() -> None:
    args = parse_args()
    result_path = run(args.dataset, args.model, args.root.resolve(), args.output_dir)
    print(result_path)


if __name__ == "__main__":
    main()
