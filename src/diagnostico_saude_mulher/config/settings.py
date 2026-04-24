from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMSettings:
    """Configurações de integração com LLM."""

    provider: str = os.getenv("LLM_PROVIDER", "mock")
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key: str | None = os.getenv("LLM_API_KEY")
    endpoint: str | None = os.getenv("LLM_ENDPOINT")
    output_path: Path = Path(os.getenv("LLM_OUTPUT_PATH", "artifacts/llm_responses.jsonl"))


@dataclass(frozen=True)
class AppSettings:
    """Configurações globais da aplicação."""

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 3
    target_name: str = "diagnostico_maligno"
    positive_label: int = 1
