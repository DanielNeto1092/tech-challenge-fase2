from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib

from diagnostico_saude_mulher.config.settings import AppSettings
from diagnostico_saude_mulher.data.datasets import carregar_dataset_cancer_mama
from diagnostico_saude_mulher.models.training import criar_modelo_base, treinar_e_avaliar_modelo


def treinar_e_exportar_modelo_final(output_dir: Path | None = None) -> dict[str, object]:
    """Treina o modelo final baseline e exporta artefatos para uso posterior."""

    settings = AppSettings()
    dataset = carregar_dataset_cancer_mama(settings)
    bundle = treinar_e_avaliar_modelo(
        criar_modelo_base(settings.random_seed),
        dataset.X_treino,
        dataset.y_treino,
        dataset.X_teste,
        dataset.y_teste,
        settings,
    )

    destino = output_dir or settings.model_output_dir
    destino.mkdir(parents=True, exist_ok=True)

    modelo_path = destino / "modelo_cancer_mama.joblib"
    metricas_path = destino / "metricas_modelo_final.json"
    colunas_path = destino / "colunas_entrada.json"

    joblib.dump(bundle.modelo, modelo_path)
    metricas_payload = {
        "dataset": dataset.nome,
        "metrics_cv": bundle.metrics_cv.to_dict(),
        "metrics_teste": bundle.metrics_teste.to_dict(),
    }
    metricas_path.write_text(json.dumps(metricas_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    colunas_path.write_text(json.dumps(list(dataset.X_treino.columns), indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "modelo_path": str(modelo_path),
        "metricas_path": str(metricas_path),
        "colunas_path": str(colunas_path),
        "metrics_cv": asdict(bundle.metrics_cv),
        "metrics_teste": asdict(bundle.metrics_teste),
    }


def main() -> None:
    """Executa o treinamento e imprime o resumo dos artefatos gerados."""

    payload = treinar_e_exportar_modelo_final()
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
