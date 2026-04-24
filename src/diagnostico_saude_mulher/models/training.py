from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from diagnostico_saude_mulher.config.settings import AppSettings
from diagnostico_saude_mulher.evaluation.metrics import gerar_metricas_classificacao
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics


@dataclass(frozen=True)
class ModelTrainingBundle:
    """Resultado de treinamento com métricas de validação cruzada e teste."""

    modelo: RandomForestClassifier
    metrics_cv: ClassificationMetrics
    metrics_teste: ClassificationMetrics


def criar_modelo_base(seed: int) -> RandomForestClassifier:
    """Cria o modelo base do projeto."""

    return RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
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


def avaliar_por_validacao_cruzada(
    modelo: RandomForestClassifier,
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
    modelo: RandomForestClassifier,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    settings: AppSettings,
) -> ModelTrainingBundle:
    """Treina o modelo e retorna métricas de CV e teste."""

    metrics_cv = avaliar_por_validacao_cruzada(modelo, X_treino, y_treino, settings)
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)
    y_score = modelo.predict_proba(X_teste)[:, 1]
    metrics_teste = gerar_metricas_classificacao(y_teste, y_pred, y_score)
    return ModelTrainingBundle(modelo=modelo, metrics_cv=metrics_cv, metrics_teste=metrics_teste)


def prever_amostra(modelo: RandomForestClassifier, amostra: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Retorna classes previstas e probabilidades para uma amostra."""

    predicoes = modelo.predict(amostra)
    probabilidades = modelo.predict_proba(amostra)[:, 1]
    return predicoes, probabilidades

