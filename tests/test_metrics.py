import pytest

from humor_detection.metrics import binary_classification_metrics


def test_macro_f1_perfect():
    result = binary_classification_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        binary_classification_metrics([0, 2], [0, 1])
