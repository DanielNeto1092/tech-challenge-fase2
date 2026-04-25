from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class MetricsResult:
    recall: float
    specificity: float
    f1_score: float
    accuracy: float
    precision: float

    def to_dict(self) -> dict[str, float]:
        return {
            "recall": self.recall,
            "specificity": self.specificity,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "precision": self.precision,
        }


def compute_specificity(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return 0.0 if (tn + fp) == 0 else tn / (tn + fp)


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    return MetricsResult(
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=compute_specificity(y_true, y_pred),
        f1_score=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
    )


def compute_fitness(metrics: MetricsResult) -> float:
    return float((0.6 * metrics.recall) + (0.3 * metrics.f1_score) + (0.1 * metrics.specificity))
