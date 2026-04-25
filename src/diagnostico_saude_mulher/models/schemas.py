from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return str(value)


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
    modelo: str
    algoritmo: str
    parametros: dict[str, Any]
    metrics_cv: ClassificationMetrics
    metrics_teste: ClassificationMetrics
    fitness_cv: float | None = None
    historico_geracoes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rotulo_exibicao(self) -> str:
        return f"{self.nome_experimento} ({self.modelo})"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics_cv"] = self.metrics_cv.to_dict()
        payload["metrics_teste"] = self.metrics_teste.to_dict()
        payload["parametros"] = _to_json_safe(payload["parametros"])
        payload["rotulo_exibicao"] = self.rotulo_exibicao
        return payload
