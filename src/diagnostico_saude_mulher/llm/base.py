from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    """Resposta padronizada da LLM."""

    content: str
    provider: str
    model: str


class LLMClient(Protocol):
    """Contrato para clientes de LLM."""

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Gera uma resposta textual a partir dos prompts."""

