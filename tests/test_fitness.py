from __future__ import annotations

import unittest

from diagnostico_saude_mulher.evaluation.metrics import FitnessWeights, calcular_fitness
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics


class FitnessTests(unittest.TestCase):
    def test_fitness_prioriza_recall(self) -> None:
        forte_recall = ClassificationMetrics(
            recall=0.95,
            specificity=0.70,
            f1_score=0.78,
            accuracy=0.80,
            roc_auc=0.88,
            precision=0.66,
        )
        forte_specificity = ClassificationMetrics(
            recall=0.75,
            specificity=0.92,
            f1_score=0.79,
            accuracy=0.83,
            roc_auc=0.87,
            precision=0.84,
        )
        self.assertGreater(calcular_fitness(forte_recall), calcular_fitness(forte_specificity))

    def test_fitness_penaliza_gap_equidade(self) -> None:
        sem_gap = ClassificationMetrics(0.90, 0.88, 0.89, 0.89, 0.93, 0.88, fairness_gap=0.0)
        com_gap = ClassificationMetrics(0.90, 0.88, 0.89, 0.89, 0.93, 0.88, fairness_gap=0.3)
        pesos = FitnessWeights(fairness_penalty=0.2)
        self.assertGreater(calcular_fitness(sem_gap, pesos), calcular_fitness(com_gap, pesos))


if __name__ == "__main__":
    unittest.main()
