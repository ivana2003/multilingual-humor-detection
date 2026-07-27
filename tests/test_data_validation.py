import pandas as pd
import pytest

from humor_detection.data import validate_normalized_frame


def frame(size: int, humorous: int) -> pd.DataFrame:
    labels = [1] * humorous + [0] * (size - humorous)
    return pd.DataFrame({
        "id": [f"id-{i}" for i in range(size)],
        "text": [f"text {i}" for i in range(size)],
        "label": labels,
    })


def test_expected_dataset_size_and_distribution_are_checked():
    summary = validate_normalized_frame(
        frame(100, 60), expected_size=100, expected_humorous_fraction=0.60
    )
    assert summary["total"] == 100
    with pytest.raises(ValueError, match="Expected 101 samples"):
        validate_normalized_frame(frame(100, 60), expected_size=101)
    with pytest.raises(ValueError, match="Expected 50.0% humorous"):
        validate_normalized_frame(
            frame(100, 60), expected_size=100, expected_humorous_fraction=0.50
        )


def test_missing_text_and_duplicate_ids_are_rejected():
    invalid = frame(10, 5)
    invalid.loc[0, "text"] = None
    with pytest.raises(ValueError, match="Texts must be non-empty"):
        validate_normalized_frame(invalid)
    invalid = frame(10, 5)
    invalid.loc[1, "id"] = invalid.loc[0, "id"]
    with pytest.raises(ValueError, match="not unique"):
        validate_normalized_frame(invalid)
