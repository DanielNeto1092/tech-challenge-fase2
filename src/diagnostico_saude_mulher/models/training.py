from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from diagnostico_saude_mulher.config.settings import AppSettings
from diagnostico_saude_mulher.evaluation.metrics import gerar_metricas_classificacao
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTrainingBundle:
    """Resultado de treinamento com métricas de validação cruzada e teste."""

    modelo: ClassifierMixin
    metrics_cv: ClassificationMetrics
    metrics_teste: ClassificationMetrics


def criar_modelo_random_forest_baseline(seed: int) -> RandomForestClassifier:
    """Cria o baseline de RandomForest do projeto."""

    return RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def criar_modelo_base(seed: int) -> RandomForestClassifier:
    """Mantém compatibilidade com o baseline principal em RandomForest."""

    return criar_modelo_random_forest_baseline(seed)


def criar_modelo_logistic_regression(seed: int) -> Pipeline:
    """Cria baseline de regressão logística com padronização."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        ]
    )


def criar_modelo_decision_tree(seed: int) -> DecisionTreeClassifier:
    """Cria baseline de árvore de decisão."""

    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
    )


def criar_modelo_knn() -> Pipeline:
    """Cria baseline KNN com padronização."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )


def criar_catalogo_modelos_baseline(seed: int) -> dict[str, ClassifierMixin]:
    """Retorna o catálogo de modelos baseline comparados no projeto."""

    return {
        "RandomForestClassifier": criar_modelo_random_forest_baseline(seed),
        "LogisticRegression": criar_modelo_logistic_regression(seed),
        "DecisionTreeClassifier": criar_modelo_decision_tree(seed),
        "KNeighborsClassifier": criar_modelo_knn(),
    }


def criar_modelo_logistic_regression_otimizado(parametros: dict[str, Any], seed: int) -> Pipeline:
    """Cria pipeline de regressão logística com hiperparâmetros definidos."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(
                    with_mean=bool(parametros.get("scaler__with_mean", True)),
                    with_std=bool(parametros.get("scaler__with_std", True)),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=float(parametros["classifier__C"]),
                    solver=str(parametros["classifier__solver"]),
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def criar_modelo_decision_tree_otimizado(parametros: dict[str, Any], seed: int) -> DecisionTreeClassifier:
    """Cria árvore de decisão com hiperparâmetros definidos."""

    return DecisionTreeClassifier(
        max_depth=int(parametros["max_depth"]),
        min_samples_split=int(parametros["min_samples_split"]),
        min_samples_leaf=int(parametros["min_samples_leaf"]),
        criterion=str(parametros["criterion"]),
        splitter=str(parametros["splitter"]),
        class_weight="balanced",
        random_state=seed,
    )


def criar_modelo_knn_otimizado(parametros: dict[str, Any]) -> Pipeline:
    """Cria pipeline KNN com hiperparâmetros definidos."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(
                    with_mean=bool(parametros.get("scaler__with_mean", True)),
                    with_std=bool(parametros.get("scaler__with_std", True)),
                ),
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=int(parametros["classifier__n_neighbors"]),
                    weights=str(parametros["classifier__weights"]),
                    p=int(parametros["classifier__p"]),
                    leaf_size=int(parametros["classifier__leaf_size"]),
                ),
            ),
        ]
    )


def criar_modelo_random_forest(parametros: dict[str, Any], seed: int) -> RandomForestClassifier:
    """Cria um RandomForest com hiperparâmetros definidos pelo AG."""

    return RandomForestClassifier(
        n_estimators=int(parametros["n_estimators"]),
        max_depth=int(parametros["max_depth"]),
        min_samples_split=int(parametros["min_samples_split"]),
        min_samples_leaf=int(parametros["min_samples_leaf"]),
        max_features=parametros["max_features"],
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def criar_modelo_otimizado_por_nome(nome_modelo: str, parametros: dict[str, Any], seed: int) -> ClassifierMixin:
    """Cria modelo otimizado a partir do nome da família e dos hiperparâmetros."""

    if nome_modelo == "RandomForestClassifier":
        return criar_modelo_random_forest(parametros, seed)
    if nome_modelo == "LogisticRegression":
        return criar_modelo_logistic_regression_otimizado(parametros, seed)
    if nome_modelo == "DecisionTreeClassifier":
        return criar_modelo_decision_tree_otimizado(parametros, seed)
    if nome_modelo == "KNeighborsClassifier":
        return criar_modelo_knn_otimizado(parametros)
    raise ValueError(f"Modelo nao suportado para otimizacao: {nome_modelo}")


def avaliar_por_validacao_cruzada(
    modelo: ClassifierMixin,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    settings: AppSettings,
) -> ClassificationMetrics:
    """Executa validação cruzada estratificada para obter métricas robustas."""

    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed)
    y_pred = cross_val_predict(modelo, X_treino, y_treino, cv=cv, method="predict")
    y_score = cross_val_predict(modelo, X_treino, y_treino, cv=cv, method="predict_proba")[:, 1]
    return gerar_metricas_classificacao(y_treino, y_pred, y_score)


def treinar_e_avaliar_modelo(
    modelo: ClassifierMixin,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    settings: AppSettings,
) -> ModelTrainingBundle:
    """Treina o modelo e retorna métricas de CV e teste."""

    LOGGER.info("Iniciando treinamento do modelo %s", modelo.__class__.__name__)
    metrics_cv = avaliar_por_validacao_cruzada(modelo, X_treino, y_treino, settings)
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)
    y_score = modelo.predict_proba(X_teste)[:, 1]
    metrics_teste = gerar_metricas_classificacao(y_teste, y_pred, y_score)
    LOGGER.info(
        "Treinamento concluido | recall_teste=%.4f | especificidade_teste=%.4f | f1_teste=%.4f",
        metrics_teste.recall,
        metrics_teste.specificity,
        metrics_teste.f1_score,
    )
    return ModelTrainingBundle(modelo=modelo, metrics_cv=metrics_cv, metrics_teste=metrics_teste)


def prever_amostra(modelo: ClassifierMixin, amostra: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Retorna classes previstas e probabilidades para uma amostra."""

    predicoes = modelo.predict(amostra)
    probabilidades = modelo.predict_proba(amostra)[:, 1]
    return predicoes, probabilidades
