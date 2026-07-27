import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def test_aggregator_uses_only_held_out_test_macro_f1(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "results/paper").mkdir(parents=True)
    (tmp_path / "results/reproduced/lr").mkdir(parents=True)
    (tmp_path / "results/reproduced/debug").mkdir(parents=True)
    (tmp_path / "results/paper/table2_reported.csv").write_text(
        (repository / "results/paper/table2_reported.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    held_out = {
        "family": "traditional_ml",
        "dataset": "en",
        "model": "lr",
        "split": "held_out_test",
        "metrics": {"test": {"macro_f1": 0.812345}},
    }
    debug = {
        "family": "traditional_ml",
        "dataset": "en",
        "model": "lr",
        "split": "debug_subset_not_paper_result",
        "metrics": {"test": {"macro_f1": 0.999999}},
    }
    (tmp_path / "results/reproduced/lr/result.json").write_text(json.dumps(held_out))
    (tmp_path / "results/reproduced/debug/result.json").write_text(json.dumps(debug))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    subprocess.run(
        [sys.executable, str(repository / "scripts/aggregate_results.py"), "--root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    with (tmp_path / "results/reproduced/table2_reproduced.csv").open(newline="") as handle:
        rows = {row["Model"]: row for row in csv.DictReader(handle)}
    assert rows["LR"]["EN"] == "0.812345"
