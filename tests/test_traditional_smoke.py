import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

from humor_detection.splitting import build_manifest, write_manifest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_traditional_ml.py"
    spec = importlib.util.spec_from_file_location("run_traditional_ml", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_actual_traditional_runner_on_synthetic_data(tmp_path):
    for directory in ["configs", "data/processed", "data/splits", "results/reproduced"]:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "id": [f"r{i}" for i in range(100)],
        "text": [f"ordinary factual statement neutral {i}" for i in range(50)]
              + [f"funny joke punchline laughter {i}" for i in range(50)],
        "label": [0] * 50 + [1] * 50,
    })
    data_path = tmp_path / "data/processed/en.csv"
    frame.to_csv(data_path, index=False)
    datasets = {
        "datasets": {
            "en": {"expected_size": 100, "expected_humorous_fraction": 0.5}
        }
    }
    (tmp_path / "configs/datasets.yaml").write_text(yaml.safe_dump(datasets))
    traditional = {
        "seed": 42,
        "features": {"ngram_range": [1, 2], "min_df": 2, "lowercase": True},
        "cross_validation": {"folds": 5, "shuffle": True, "c_values": [0.1, 1.0, 10.0]},
    }
    (tmp_path / "configs/traditional_ml.yaml").write_text(yaml.safe_dump(traditional))
    manifest = build_manifest(
        "en", frame, data_path, source_reference="data/processed/en.csv", seed=42
    )
    assert manifest["source"]["path"] == "data/processed/en.csv"
    assert not Path(manifest["source"]["path"]).is_absolute()
    write_manifest(tmp_path / "data/splits/en.json", manifest)

    result_path = load_runner().run("en", "lr", tmp_path)
    result = json.loads(result_path.read_text())
    assert result["split"] == "held_out_test"
    assert result["metrics"]["test"]["macro_f1"] > 0.9
    assert result["hyperparameters"]["min_df"] == 2
