from __future__ import annotations

import json

import pandas as pd

from diagnostico_saude_mulher.llm.service import ExplanationRecord
from diagnostico_saude_mulher.models.schemas import ExperimentResult


def experiment_results_to_dataframe(experimentos: list[ExperimentResult]) -> pd.DataFrame:
    """Converte experimentos em tabela para UI."""

    return pd.DataFrame(
        [
            {
                "experimento": item.rotulo_exibicao,
                "fitness_cv": item.fitness_cv,
                "recall_teste": item.metrics_teste.recall,
                "especificidade_teste": item.metrics_teste.specificity,
                "f1_teste": item.metrics_teste.f1_score,
                "parametros": json.dumps(item.parametros, ensure_ascii=False),
            }
            for item in experimentos
        ]
    )


def llm_history_to_dataframe(history: list[ExplanationRecord]) -> pd.DataFrame:
    """Converte histórico da LLM em tabela para UI."""

    return pd.DataFrame(
        [
            {
                "classificacao": item.classificacao,
                "probabilidade": item.probabilidade,
                "provider": item.provider,
                "model": item.model,
                "resposta": item.resposta[:180],
            }
            for item in reversed(history)
        ]
    )
