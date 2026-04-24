from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from diagnostico_saude_mulher.llm.base import LLMClient, LLMResponse
from diagnostico_saude_mulher.llm.prompting import construir_system_prompt, construir_user_prompt
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics


@dataclass(frozen=True)
class ExplanationRecord:
    """Registro persistido de explicações geradas pela LLM."""

    classificacao: str
    probabilidade: float
    contexto_clinico: str
    system_prompt: str
    user_prompt: str
    resposta: str
    provider: str
    model: str


class DiagnosticExplanationService:
    """Orquestra prompts, chamada à LLM e persistência das respostas."""

    def __init__(self, client: LLMClient, output_path: Path) -> None:
        self.client = client
        self.output_path = output_path

    def generate_explanation(
        self,
        classificacao: str,
        probabilidade: float,
        metrics: ClassificationMetrics,
        contexto_clinico: str,
    ) -> ExplanationRecord:
        """Gera e persiste uma explicação em linguagem natural."""

        system_prompt = construir_system_prompt()
        user_prompt = construir_user_prompt(classificacao, probabilidade, metrics, contexto_clinico)
        response: LLMResponse = self.client.generate(system_prompt, user_prompt)
        record = ExplanationRecord(
            classificacao=classificacao,
            probabilidade=probabilidade,
            contexto_clinico=contexto_clinico,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            resposta=response.content,
            provider=response.provider,
            model=response.model,
        )
        self._append_jsonl(record)
        return record

    def _append_jsonl(self, record: ExplanationRecord) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
