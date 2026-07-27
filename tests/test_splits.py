import pandas as pd

from humor_detection.splitting import make_split_ids, split_frames, validate_split_ids


def sample_frame():
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(100)],
        "text": [f"text {i}" for i in range(100)],
        "label": [0] * 40 + [1] * 60,
    })


def test_splits_are_deterministic_disjoint_and_complete():
    frame = sample_frame()
    first = make_split_ids(frame, seed=42)
    second = make_split_ids(frame, seed=42)
    assert first == second
    validate_split_ids(first, expected_ids=set(frame["id"]))
    assert [len(first[name]) for name in ["train", "validation", "test"]] == [70, 10, 20]


def test_stratification_preserves_distribution():
    parts = split_frames(sample_frame(), make_split_ids(sample_frame(), seed=42))
    for part in parts.values():
        assert part["label"].mean() == 0.6
