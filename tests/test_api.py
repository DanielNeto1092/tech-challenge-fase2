from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diagnostico_saude_mulher.api.app import PredictionRequest, create_app
from diagnostico_saude_mulher.llm.service import load_explanation_history


class ApiTests(unittest.TestCase):
    def test_create_app_expoe_rotas_esperadas(self) -> None:
        app = create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/dataset", paths)
        self.assertIn("/predict", paths)
        self.assertIn("/llm/history", paths)
        self.assertIn("/logs", paths)

    def test_prediction_request_defaults(self) -> None:
        payload = PredictionRequest(features={"mean radius": 14.0})
        self.assertTrue(payload.usar_modelo_otimizado)
        self.assertFalse(payload.gerar_explicacao)

    def test_llm_history_loader_for_api_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "llm.jsonl"
            with mock.patch.dict("os.environ", {"LLM_OUTPUT_PATH": str(output_path), "LLM_PROVIDER": "mock"}, clear=False):
                history = load_explanation_history(output_path)
                self.assertEqual(history, [])
