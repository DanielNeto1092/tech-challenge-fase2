from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diagnostico_saude_mulher.llm.clients import MockLLMClient
from diagnostico_saude_mulher.llm.prompting import construir_system_prompt, construir_user_prompt
from diagnostico_saude_mulher.llm.service import DiagnosticExplanationService
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics


class LLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = ClassificationMetrics(
            recall=0.92,
            specificity=0.84,
            f1_score=0.88,
            accuracy=0.87,
            roc_auc=0.93,
            precision=0.85,
        )

    def test_prompt_contem_avisos_necessarios(self) -> None:
        prompt = construir_user_prompt("maligno", 0.87, self.metrics, "Paciente em rastreio complementar.")
        self.assertIn("nao substitui avaliacao medica", prompt)
        self.assertIn("linguagem sensivel a genero", prompt)
        self.assertIn("Probabilidade estimada", prompt)

    def test_cliente_mock_retorna_resposta_padronizada(self) -> None:
        response = MockLLMClient().generate(construir_system_prompt(), "teste")
        self.assertEqual(response.provider, "mock")
        self.assertIn("avaliacao medica", response.content)

    def test_servico_persiste_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "respostas.jsonl"
            service = DiagnosticExplanationService(MockLLMClient(), output_path)
            registro = service.generate_explanation("maligno", 0.91, self.metrics, "Suspeita radiologica.")
            self.assertEqual(registro.provider, "mock")
            linhas = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(linhas), 1)
            payload = json.loads(linhas[0])
            self.assertEqual(payload["classificacao"], "maligno")


if __name__ == "__main__":
    unittest.main()
