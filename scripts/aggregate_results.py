
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from humor_detection.io import project_root

COLUMNS = ["Model", "EN", "ES", "HI-EN", "PT", "ZH"]
DATASET_COLUMNS = {"en": "EN", "es": "ES", "hi_en": "HI-EN", "pt": "PT", "zh": "ZH"}
MODEL_ROWS = [
    "SVM", "LR", "XLM-R-base", "DeBERTa-v3-base", "BETO", "MuRIL", "BERTimbau", "BERT-Chinese",
    "GPT-4o-mini zero-shot", "GPT-4o-mini few-shot", "Qwen zero-shot", "Qwen few-shot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    return parser.parse_args()


def canonical_row(result: dict) -> str:
    family = result.get("family")
    model = result.get("model")
    if family == "traditional_ml":
        return "LR" if model == "lr" else "SVM"
    if family == "transformer":
        return str(model)
    if family == "llm":
        prefix = "GPT-4o-mini" if model == "GPT-4o-mini" else "Qwen"
        suffix = "zero-shot" if result.get("shot") == "zero" else "few-shot"
        return f"{prefix} {suffix}"
    raise ValueError(f"Unknown result family: {family!r}")


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["Model"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    table = {name: {column: "NA" for column in COLUMNS[1:]} for name in MODEL_ROWS}
    seen = {}
    for path in sorted((root / "results" / "reproduced").rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("split") != "held_out_test":
            continue
        try:
            score = float(result["metrics"]["test"]["macro_f1"])
            column = DATASET_COLUMNS[result["dataset"]]
            row = canonical_row(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid result file {path}: {exc}") from exc
        key = (row, column)
        if key == ("SVM", "ES"):
            raise ValueError(f"{path} contains Spanish SVM, which is absent from Table 2")
        if key in seen:
            raise ValueError(f"Multiple reproduced held-out test results for {row}/{column}: {seen[key]} and {path}")
        seen[key] = path
        table[row][column] = f"{score:.6f}"
    reproduced_rows = [{"Model": model, **table[model]} for model in MODEL_ROWS]
    reproduced_path = root / "results" / "reproduced" / "table2_reproduced.csv"
    write_csv(reproduced_path, reproduced_rows, COLUMNS)

    reported = read_csv(root / "results" / "paper" / "table2_reported.csv")
    differences = []
    for row in reproduced_rows:
        for column in COLUMNS[1:]:
            reproduced = row[column]
            reference = reported[row["Model"]][column]
            difference = "NA"
            if reproduced != "NA" and reference != "NA":
                difference = f"{float(reproduced) - float(reference):+.6f}"
            differences.append({
                "Model": row["Model"], "Language": column,
                "Reported": reference, "Reproduced": reproduced, "Difference": difference,
            })
    comparison_path = root / "results" / "reproduced" / "table2_comparison.csv"
    write_csv(comparison_path, differences, ["Model", "Language", "Reported", "Reproduced", "Difference"])
    print(reproduced_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
