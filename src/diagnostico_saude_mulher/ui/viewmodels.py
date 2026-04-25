from __future__ import annotations

import json

import pandas as pd

from diagnostico_saude_mulher.llm.service import ExplanationRecord
from diagnostico_saude_mulher.models.schemas import ExperimentResult


def _format_model_name(model_name: str) -> str:
    mapping = {
        "RandomForestClassifier": "Random Forest",
        "LogisticRegression": "Regressão Logística",
        "DecisionTreeClassifier": "Árvore de Decisão",
        "KNeighborsClassifier": "KNN",
    }
    return mapping.get(model_name, model_name)


def _format_experiment_label(item: ExperimentResult) -> str:
    model_label = _format_model_name(item.modelo)

    if item.algoritmo == "baseline":
        return f"Baseline {model_label}"

    experiment_mapping = {
        "ag_experimento_1": "AG Experimento 1",
        "ag_experimento_2": "AG Experimento 2",
        "ag_experimento_3": "AG Experimento 3",
    }
    experiment_label = experiment_mapping.get(item.nome_experimento, item.nome_experimento.replace("_", " ").title())
    return f"{experiment_label} ({model_label})"


def _format_parameter_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _summarize_parameters(item: ExperimentResult) -> str:
    params = item.to_dict()["parametros"]

    key_groups = {
        "RandomForestClassifier": [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        ],
        "LogisticRegression": [
            "classifier__C",
            "classifier__solver",
            "classifier__max_iter",
            "scaler__with_mean",
            "scaler__with_std",
        ],
        "DecisionTreeClassifier": [
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "criterion",
            "splitter",
        ],
        "KNeighborsClassifier": [
            "classifier__n_neighbors",
            "classifier__weights",
            "classifier__p",
            "classifier__leaf_size",
            "scaler__with_mean",
            "scaler__with_std",
        ],
    }

    selected_keys = [key for key in key_groups.get(item.modelo, []) if key in params]
    if selected_keys:
        return ", ".join(f"{key}={_format_parameter_value(params[key])}" for key in selected_keys)

    compact = json.dumps(params, ensure_ascii=False)
    return compact if len(compact) <= 180 else compact[:177] + "..."


def experiment_results_to_dataframe(experimentos: list[ExperimentResult]) -> pd.DataFrame:
    """Converte experimentos em tabela para UI."""

    return pd.DataFrame(
        [
            {
                "Experimento": _format_experiment_label(item),
                "Fitness CV": f"{item.fitness_cv:.4f}" if item.fitness_cv is not None else "-",
                "Recall Teste": _format_percentage(item.metrics_teste.recall),
                "Especificidade Teste": _format_percentage(item.metrics_teste.specificity),
                "F1 Teste": _format_percentage(item.metrics_teste.f1_score),
                "Parâmetros": _summarize_parameters(item),
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
