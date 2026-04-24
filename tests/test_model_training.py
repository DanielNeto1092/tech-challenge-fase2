from __future__ import annotations

import unittest

from diagnostico_saude_mulher.config.settings import AppSettings
from diagnostico_saude_mulher.data.datasets import carregar_dataset_cancer_mama
from diagnostico_saude_mulher.models.training import criar_modelo_base, treinar_e_avaliar_modelo


class ModelTrainingTests(unittest.TestCase):
    def test_treinamento_basico_retorna_metricas_validas(self) -> None:
        settings = AppSettings()
        dataset = carregar_dataset_cancer_mama(settings)
        resultado = treinar_e_avaliar_modelo(
            criar_modelo_base(settings.random_seed),
            dataset.X_treino,
            dataset.y_treino,
            dataset.X_teste,
            dataset.y_teste,
            settings,
        )
        self.assertGreaterEqual(resultado.metrics_teste.recall, 0.80)
        self.assertGreaterEqual(resultado.metrics_teste.roc_auc, 0.90)


if __name__ == "__main__":
    unittest.main()
