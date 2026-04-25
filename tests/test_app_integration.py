from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diagnostico_saude_mulher.app import executar_experimentos
from diagnostico_saude_mulher.models.train_model import treinar_e_exportar_modelo_final


class AppIntegrationTests(unittest.TestCase):
    def test_pipeline_principal_retorna_payload_coerente(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "llm.jsonl"
            with mock.patch.dict("os.environ", {"LLM_OUTPUT_PATH": str(output_path), "LLM_PROVIDER": "mock"}, clear=False):
                payload = executar_experimentos()
            self.assertIn("modelos_baseline", payload)
            self.assertIn("baseline", payload)
            self.assertIn("experimentos_geneticos", payload)
            self.assertIn("comparacao", payload)
            self.assertIn("explicacao_llm", payload)
            self.assertEqual(len(payload["modelos_baseline"]), 4)
            self.assertEqual(len(payload["experimentos_geneticos"]), 3)
            self.assertEqual(payload["baseline"]["modelo"], "LogisticRegression")
            self.assertTrue(all(item["modelo"] == "LogisticRegression" for item in payload["experimentos_geneticos"]))
            self.assertTrue(output_path.exists())

    def test_treinamento_exporta_artefatos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destino = Path(directory) / "modelos"
            payload = treinar_e_exportar_modelo_final(destino)
            self.assertTrue(Path(payload["modelo_path"]).exists())
            self.assertTrue(Path(payload["metricas_path"]).exists())
            self.assertTrue(Path(payload["colunas_path"]).exists())
            metricas = json.loads(Path(payload["metricas_path"]).read_text(encoding="utf-8"))
            self.assertIn("metrics_teste", metricas)


if __name__ == "__main__":
    unittest.main()
