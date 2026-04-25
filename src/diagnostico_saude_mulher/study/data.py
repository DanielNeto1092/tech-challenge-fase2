from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StudyDataset:
    """Dataset preparado para os experimentos do estudo."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    dataset_name: str


def load_breast_cancer_study_dataset(random_state: int = 42, test_size: float = 0.2) -> StudyDataset:
    """Carrega a base de câncer de mama com split 80/20 e normalização."""

    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    # O alvo original do sklearn é 0=maligno, 1=benigno.
    y = (1 - raw.target).rename("diagnostico_maligno")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return StudyDataset(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=list(X.columns),
        dataset_name="Breast Cancer Wisconsin Diagnostic",
    )
