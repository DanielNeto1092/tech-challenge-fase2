from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diagnostico_saude_mulher.config.settings import AppSettings, LLMSettings
from diagnostico_saude_mulher.data.datasets import carregar_dataset_cancer_mama
from diagnostico_saude_mulher.experiments.runner import (
    build_comparison_payload,
    build_genetic_configs,
    choose_best_experiment,
    run_experiments,
)
from diagnostico_saude_mulher.llm.clients import build_llm_client
from diagnostico_saude_mulher.llm.service import DiagnosticExplanationService, load_explanation_history
from diagnostico_saude_mulher.models.training import criar_modelo_otimizado_por_nome
from diagnostico_saude_mulher.utils.logging_utils import configure_logging


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Mapa de atributos de entrada.")
    usar_modelo_otimizado: bool = True
    gerar_explicacao: bool = False
    contexto_clinico: str = "Paciente em avaliacao complementar de lesao mamaria."


def create_app() -> FastAPI:
    settings = AppSettings()
    configure_logging(settings.log_level, settings.log_output_dir)
    app = FastAPI(title="Diagnostico Saude da Mulher API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/dataset")
    def dataset_info() -> dict[str, Any]:
        dataset = carregar_dataset_cancer_mama(settings)
        return {
            "nome": dataset.nome,
            "descricao": dataset.descricao,
            "amostras": len(dataset.atributos),
            "atributos": list(dataset.atributos.columns),
        }

    @app.post("/baseline")
    def executar_baseline() -> dict[str, Any]:
        dataset = carregar_dataset_cancer_mama(settings)
        modelos_baseline, baseline, _ = run_experiments(settings, dataset, [])
        return {
            "algoritmo": "baseline",
            "modelo_referencia_ag": baseline.modelo,
            "baseline_referencia": baseline.to_dict(),
            "modelos_baseline": [item.to_dict() for item in modelos_baseline],
        }

    @app.post("/optimize")
    def executar_otimizacao() -> dict[str, Any]:
        dataset = carregar_dataset_cancer_mama(settings)
        modelos_baseline, baseline, experimentos = run_experiments(settings, dataset, build_genetic_configs())
        melhor = choose_best_experiment(experimentos)
        return {
            "modelos_baseline": [item.to_dict() for item in modelos_baseline],
            "baseline": baseline.to_dict(),
            "experimentos": [item.to_dict() for item in experimentos],
            "melhor": melhor.to_dict(),
            "comparacao": build_comparison_payload(baseline, melhor),
        }

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        dataset = carregar_dataset_cancer_mama(settings)
        sample = dataset.X_teste.head(1).copy()
        for column, value in request.features.items():
            if column not in sample.columns:
                raise HTTPException(status_code=400, detail=f"Atributo invalido: {column}")
            sample.iloc[0, sample.columns.get_loc(column)] = value

        if request.usar_modelo_otimizado:
            _, _, experimentos = run_experiments(settings, dataset, build_genetic_configs())
            melhor = choose_best_experiment(experimentos)
            modelo = criar_modelo_otimizado_por_nome(melhor.modelo, melhor.parametros, settings.random_seed)
            metrics = melhor.metrics_teste
            algoritmo = "genetico"
            nome_modelo = melhor.modelo
        else:
            _, baseline, _ = run_experiments(settings, dataset, [])
            modelo = criar_modelo_otimizado_por_nome(baseline.modelo, baseline.parametros, settings.random_seed)
            metrics = baseline.metrics_teste
            algoritmo = "baseline"
            nome_modelo = baseline.modelo

        modelo.fit(dataset.X_treino, dataset.y_treino)
        predicao = int(modelo.predict(sample)[0])
        probabilidade = float(modelo.predict_proba(sample)[0][1])
        classificacao = "maligno" if predicao == settings.positive_label else "benigno"
        payload: dict[str, Any] = {
            "algoritmo": algoritmo,
            "modelo": nome_modelo,
            "classificacao": classificacao,
            "probabilidade": probabilidade,
            "metricas_modelo": metrics.to_dict(),
        }
        if request.gerar_explicacao:
            llm_settings = LLMSettings()
            service = DiagnosticExplanationService(build_llm_client(llm_settings), llm_settings.output_path)
            explicacao = service.generate_explanation(
                classificacao=classificacao,
                probabilidade=probabilidade,
                metrics=metrics,
                contexto_clinico=request.contexto_clinico,
            )
            payload["explicacao"] = explicacao.resposta
        return payload

    @app.get("/llm/history")
    def llm_history() -> list[dict[str, Any]]:
        history = load_explanation_history(LLMSettings().output_path)
        return [
            {
                "classificacao": item.classificacao,
                "probabilidade": item.probabilidade,
                "provider": item.provider,
                "model": item.model,
                "resposta": item.resposta,
            }
            for item in history
        ]

    @app.get("/logs")
    def logs() -> dict[str, Any]:
        path = settings.log_output_dir / "aplicacao.log"
        if not path.exists():
            return {"path": str(path), "lines": []}
        return {"path": str(path), "lines": path.read_text(encoding="utf-8").splitlines()[-100:]}

    return app


app = create_app()
