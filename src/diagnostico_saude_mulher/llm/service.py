from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from diagnostico_saude_mulher.llm.base import LLMClient, LLMResponse
from diagnostico_saude_mulher.llm.prompting import construir_system_prompt, construir_user_prompt
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics

LOGGER = logging.getLogger(__name__)


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

        LOGGER.info("Gerando explicacao LLM | provider=%s | model_hint=%s", type(self.client).__name__, getattr(self.client, "model", "n/a"))
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
        LOGGER.info("Explicacao LLM persistida em %s", self.output_path)
        return record

    def _append_jsonl(self, record: ExplanationRecord) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def load_explanation_history(path: Path) -> list[ExplanationRecord]:
    """Carrega o histórico de respostas da LLM a partir do JSONL."""

    if not path.exists():
        return []
    history: list[ExplanationRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        history.append(ExplanationRecord(**payload))
    return history
