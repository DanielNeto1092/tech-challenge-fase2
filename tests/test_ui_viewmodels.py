from __future__ import annotations

import unittest

from diagnostico_saude_mulher.llm.service import ExplanationRecord
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics, ExperimentResult
from diagnostico_saude_mulher.ui.viewmodels import experiment_results_to_dataframe, llm_history_to_dataframe


def _build_experiment() -> ExperimentResult:
    metrics = ClassificationMetrics(
        recall=0.91,
        specificity=0.89,
        f1_score=0.90,
        accuracy=0.90,
        roc_auc=0.94,
        precision=0.89,
    )
    return ExperimentResult(
        nome_experimento="ag_experimento_1",
        modelo="RandomForestClassifier",
        algoritmo="genetico",
        parametros={"n_estimators": 120},
        metrics_cv=metrics,
        metrics_teste=metrics,
        fitness_cv=0.91,
    )


class UIViewModelTests(unittest.TestCase):
    def test_experiment_results_to_dataframe(self) -> None:
        dataframe = experiment_results_to_dataframe([_build_experiment()])
        self.assertEqual(len(dataframe), 1)
        self.assertIn("experimento", dataframe.columns)
        self.assertIn("parametros", dataframe.columns)
        self.assertIn("fitness_cv", dataframe.columns)
        self.assertIn("RandomForestClassifier", dataframe.iloc[0]["experimento"])

    def test_llm_history_to_dataframe(self) -> None:
        history = [
            ExplanationRecord(
                classificacao="maligno",
                probabilidade=0.87,
                contexto_clinico="teste",
                system_prompt="sys",
                user_prompt="user",
                resposta="resposta longa",
                provider="mock",
                model="mock",
            )
        ]
        dataframe = llm_history_to_dataframe(history)
        self.assertEqual(len(dataframe), 1)
        self.assertIn("classificacao", dataframe.columns)
        self.assertIn("resposta", dataframe.columns)
