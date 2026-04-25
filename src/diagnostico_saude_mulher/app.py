from __future__ import annotations

import json

from diagnostico_saude_mulher.experiments.runner import executar_pipeline_completo


def executar_experimentos() -> dict[str, object]:
    """Executa baseline, busca genética e geração de explicação com LLM."""

    return executar_pipeline_completo().to_dict()


def main() -> None:
    """Ponto de entrada da aplicação."""

    payload = executar_experimentos()
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
