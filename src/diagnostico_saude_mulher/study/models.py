from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    search_space: dict[str, list[Any]]


MODEL_SPECS: dict[str, ModelSpec] = {
    "KNeighborsClassifier": ModelSpec(
        model_name="KNeighborsClassifier",
        search_space={
            "n_neighbors": list(range(1, 31)),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    ),
    "DecisionTreeClassifier": ModelSpec(
        model_name="DecisionTreeClassifier",
        search_space={
            "max_depth": list(range(1, 31)),
            "min_samples_split": list(range(2, 51)),
            "min_samples_leaf": list(range(1, 21)),
            "criterion": ["gini", "entropy"],
        },
    ),
    "LogisticRegression": ModelSpec(
        model_name="LogisticRegression",
        search_space={
            "C": [round(value, 2) for value in [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]],
            "max_iter": list(range(100, 1001, 100)),
            "solver": ["liblinear", "saga"],
        },
    ),
    "RandomForestClassifier": ModelSpec(
        model_name="RandomForestClassifier",
        search_space={
            "n_estimators": list(range(50, 501, 50)),
            "max_depth": list(range(5, 51, 5)),
            "min_samples_split": list(range(2, 51, 4)),
            "min_samples_leaf": list(range(1, 21, 2)),
        },
    ),
}


def build_baseline_model(model_name: str, random_state: int = 42) -> ClassifierMixin:
    if model_name == "KNeighborsClassifier":
        return KNeighborsClassifier()
    if model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=random_state)
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    raise ValueError(f"Modelo nao suportado: {model_name}")


def build_optimized_model(model_name: str, params: dict[str, Any], random_state: int = 42) -> ClassifierMixin:
    if model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(
            n_neighbors=int(params["n_neighbors"]),
            weights=str(params["weights"]),
            p=int(params["p"]),
        )
    if model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            criterion=str(params["criterion"]),
            random_state=random_state,
        )
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=float(params["C"]),
            max_iter=int(params["max_iter"]),
            solver=str(params["solver"]),
            random_state=random_state,
        )
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Modelo nao suportado: {model_name}")
