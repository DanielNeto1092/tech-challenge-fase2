from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from diagnostico_saude_mulher.models.schemas import ClassificationMetrics, ExperimentResult
from diagnostico_saude_mulher.visualization.plots import (
    plot_convergence,
    plot_experiment_comparison,
    plot_experiment_summary,
    plot_generation_metrics,
    plot_hyperparameter_behavior,
)


def _build_result(nome: str, fitness: float) -> ExperimentResult:
    metrics = ClassificationMetrics(
        recall=0.91,
        specificity=0.87,
        f1_score=0.89,
        accuracy=0.88,
        roc_auc=0.94,
        precision=0.88,
    )
    return ExperimentResult(
        nome_experimento=nome,
        modelo="RandomForestClassifier",
        algoritmo="genetico",
        parametros={"n_estimators": 120},
        metrics_cv=metrics,
        metrics_teste=metrics,
        historico_geracoes=[
            {
                "generation": 0,
                "best_fitness": fitness - 0.02,
                "avg_fitness": fitness - 0.05,
                "best_recall": 0.89,
                "best_specificity": 0.84,
                "best_f1_score": 0.86,
            },
            {
                "generation": 1,
                "best_fitness": fitness,
                "avg_fitness": fitness - 0.01,
                "best_recall": 0.91,
                "best_specificity": 0.87,
                "best_f1_score": 0.89,
            },
        ],
    )


class VisualizationTests(unittest.TestCase):
    def test_gera_arquivos_de_graficos(self) -> None:
        baseline = _build_result("baseline", 0.90)
        experimentos = [_build_result("ag_experimento_1", 0.91), _build_result("ag_experimento_2", 0.93)]
        experimentos[0].parametros.update({"max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "sqrt"})
        experimentos[1].parametros.update({"max_depth": 10, "min_samples_split": 6, "min_samples_leaf": 2, "max_features": "log2"})
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            convergencia = plot_convergence(experimentos[0], output_dir)
            comparacao = plot_experiment_comparison(baseline, experimentos, output_dir)
            resumo = plot_experiment_summary(experimentos, output_dir)
            metricas = plot_generation_metrics(experimentos[0], output_dir)
            comportamento = plot_hyperparameter_behavior(experimentos, output_dir)
            self.assertTrue(convergencia.exists())
            self.assertTrue(comparacao.exists())
            self.assertTrue(resumo.exists())
            self.assertTrue(metricas.exists())
            self.assertTrue(comportamento.exists())


if __name__ == "__main__":
    unittest.main()
