import importlib.util
from pathlib import Path

import pandas as pd


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_llms.py"
    spec = importlib.util.spec_from_file_location("run_llms", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_few_shot_uses_exactly_five_training_examples_per_class():
    train = pd.DataFrame({
        "id": [f"train-{i}" for i in range(20)],
        "text": [f"train text {i}" for i in range(20)],
        "label": [0] * 10 + [1] * 10,
    })
    examples = load_runner().choose_few_shot_examples(train, seed=42, per_class=5)
    assert len(examples) == 10
    assert [example.label for example in examples].count(0) == 5
    assert [example.label for example in examples].count(1) == 5
    assert all(example.text.startswith("train text") for example in examples)
