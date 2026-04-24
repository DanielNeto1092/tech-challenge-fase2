from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from diagnostico_saude_mulher.models.schemas import ClassificationMetrics


def calcular_especificidade(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula especificidade a partir da matriz de confusão binária."""

    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return 0.0 if (tn + fp) == 0 else tn / (tn + fp)


def calcular_gap_equidade(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    grupos: pd.Series | np.ndarray | None,
) -> float | None:
    """Calcula a maior diferença de recall entre grupos, quando disponível."""

    if grupos is None:
        return None

    serie_grupos = pd.Series(grupos)
    serie_true = pd.Series(y_true)
    serie_pred = pd.Series(y_pred)
    recalls: list[float] = []
    for grupo in sorted(serie_grupos.unique()):
        mascara = serie_grupos == grupo
        if mascara.sum() == 0 or serie_true[mascara].sum() == 0:
            continue
        recalls.append(recall_score(serie_true[mascara], serie_pred[mascara], zero_division=0))
    if len(recalls) < 2:
        return None
    return float(max(recalls) - min(recalls))


def gerar_metricas_classificacao(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    grupos: pd.Series | np.ndarray | None = None,
) -> ClassificationMetrics:
    """Gera o conjunto de métricas do projeto."""

    return ClassificationMetrics(
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=calcular_especificidade(y_true, y_pred),
        f1_score=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_score),
        precision=precision_score(y_true, y_pred, zero_division=0),
        fairness_gap=calcular_gap_equidade(y_true, y_pred, grupos),
    )


@dataclass(frozen=True)
class FitnessWeights:
    """Pesos da função fitness."""

    recall: float = 0.55
    specificity: float = 0.25
    f1_score: float = 0.20
    fairness_penalty: float = 0.10


def calcular_fitness(metrics: ClassificationMetrics, weights: FitnessWeights | None = None) -> float:
    """Combina métricas em uma função objetivo escalar.

    Quanto maior o valor retornado, melhor o indivíduo.
    """

    pesos = weights or FitnessWeights()
    fitness = (
        metrics.recall * pesos.recall
        + metrics.specificity * pesos.specificity
        + metrics.f1_score * pesos.f1_score
    )
    if metrics.fairness_gap is not None:
        fitness -= metrics.fairness_gap * pesos.fairness_penalty
    return float(fitness)

