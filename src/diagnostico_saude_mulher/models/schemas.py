from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClassificationMetrics:
    """Métricas de classificação voltadas ao projeto."""

    recall: float
    specificity: float
    f1_score: float
    accuracy: float
    roc_auc: float
    precision: float
    fairness_gap: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    """Resultado consolidado de um experimento."""

    nome_experimento: str
    algoritmo: str
    parametros: dict[str, Any]
    metrics_cv: ClassificationMetrics
    metrics_teste: ClassificationMetrics
    historico_geracoes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics_cv"] = self.metrics_cv.to_dict()
        payload["metrics_teste"] = self.metrics_teste.to_dict()
        return payload

