from __future__ import annotations

import json
import urllib.error
import urllib.request

from diagnostico_saude_mulher.config.settings import LLMSettings
from diagnostico_saude_mulher.llm.base import LLMClient, LLMResponse


class MockLLMClient:
    """Cliente de mock determinístico para testes e uso local sem API."""

    def __init__(self, model: str = "mock-clinico") -> None:
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        classificacao = "indeterminada"
        probability_text = "nao informada"
        for line in user_prompt.splitlines():
            if line.startswith("Classificacao prevista:"):
                classificacao = line.split(":", 1)[1].strip().strip(".")
            if line.startswith("Probabilidade estimada de malignidade:"):
                probability_text = line.split(":", 1)[1].strip().strip(".")
        risco_texto = "baixo" if classificacao == "benigno" else "elevado"
        return LLMResponse(
            content=(
                f"Resumo clinico assistido: a classificacao prevista foi {classificacao}, "
                f"com probabilidade estimada de malignidade em {probability_text}. "
                f"O resultado sugere risco {risco_texto}, mas deve ser usado apenas como apoio "
                "a priorizacao diagnostica. Confirme com avaliacao medica, exame fisico, "
                "imagem e conduta especializada."
            ),
            provider="mock",
            model=self.model,
        )


class HTTPLLMClient:
    """Cliente HTTP compatível com endpoints estilo chat completions."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.settings.endpoint:
            raise ValueError("LLM_ENDPOINT nao configurado para provedor HTTP.")

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        request = urllib.request.Request(
            self.settings.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Falha ao consultar a LLM HTTP.") from exc

        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, provider=self.settings.provider, model=self.settings.model)


def build_llm_client(settings: LLMSettings) -> LLMClient:
    """Cria um cliente LLM a partir da configuração."""

    if settings.provider == "mock":
        return MockLLMClient(model=settings.model)
    if settings.provider == "http":
        return HTTPLLMClient(settings=settings)
    raise ValueError(f"Provedor de LLM nao suportado: {settings.provider}")
