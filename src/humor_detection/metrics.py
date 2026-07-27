from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def binary_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Compute macro-F1 and supporting binary classification diagnostics."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    if not y_true:
        raise ValueError("Cannot evaluate an empty prediction set")
    valid = {0, 1}
    if not set(y_true).issubset(valid) or not set(y_pred).issubset(valid):
        raise ValueError("Labels and predictions must be binary integers 0 or 1")
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
