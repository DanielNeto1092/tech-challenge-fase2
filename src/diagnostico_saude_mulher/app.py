from __future__ import annotations

import json
import logging
from dataclasses import asdict

from dotenv import load_dotenv

from diagnostico_saude_mulher.config.settings import AppSettings, LLMSettings
from diagnostico_saude_mulher.data.datasets import carregar_dataset_cancer_mama
from diagnostico_saude_mulher.genetic_algorithm.engine import GeneticConfig, HyperparameterGeneticOptimizer
from diagnostico_saude_mulher.llm.clients import build_llm_client
from diagnostico_saude_mulher.llm.service import DiagnosticExplanationService
from diagnostico_saude_mulher.models.schemas import ExperimentResult
from diagnostico_saude_mulher.models.training import (
    criar_modelo_base,
    criar_modelo_random_forest,
    prever_amostra,
    treinar_e_avaliar_modelo,
)
from diagnostico_saude_mulher.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def executar_experimentos() -> dict[str, object]:
    """Executa baseline, busca genética e geração de explicação com LLM."""

    load_dotenv()
    settings = AppSettings()
    llm_settings = LLMSettings()
    configure_logging(settings.log_level)
    dataset = carregar_dataset_cancer_mama(settings)

    baseline_bundle = treinar_e_avaliar_modelo(
        criar_modelo_base(settings.random_seed),
        dataset.X_treino,
        dataset.y_treino,
        dataset.X_teste,
        dataset.y_teste,
        settings,
    )
    baseline = ExperimentResult(
        nome_experimento="baseline_random_forest",
        algoritmo="baseline",
        parametros=criar_modelo_base(settings.random_seed).get_params(),
        metrics_cv=baseline_bundle.metrics_cv,
        metrics_teste=baseline_bundle.metrics_teste,
    )

    configs = [
        GeneticConfig(population_size=6, generations=3, mutation_rate=0.10, selection_strategy="tournament"),
        GeneticConfig(population_size=8, generations=4, mutation_rate=0.15, selection_strategy="tournament"),
        GeneticConfig(population_size=10, generations=5, mutation_rate=0.20, selection_strategy="roulette"),
    ]

    experimentos_otimizados: list[ExperimentResult] = []
    melhor_resultado: ExperimentResult | None = None
    melhor_modelo = None

    for indice, config in enumerate(configs, start=1):
        LOGGER.info("Executando experimento genetico %s com config=%s", indice, config)
        optimizer = HyperparameterGeneticOptimizer(config=config, settings=settings)
        result = optimizer.optimize(dataset.X_treino, dataset.y_treino)
        modelo_otimizado = criar_modelo_random_forest(result.best_params, settings.random_seed)
        bundle = treinar_e_avaliar_modelo(
            modelo_otimizado,
            dataset.X_treino,
            dataset.y_treino,
            dataset.X_teste,
            dataset.y_teste,
            settings,
        )
        experimento = ExperimentResult(
            nome_experimento=f"ag_experimento_{indice}",
            algoritmo="genetico",
            parametros=result.best_params,
            metrics_cv=bundle.metrics_cv,
            metrics_teste=bundle.metrics_teste,
            historico_geracoes=[asdict(item) for item in result.history],
        )
        experimentos_otimizados.append(experimento)
        if melhor_resultado is None or experimento.metrics_teste.recall > melhor_resultado.metrics_teste.recall:
            melhor_resultado = experimento
            melhor_modelo = bundle.modelo

    assert melhor_resultado is not None
    assert melhor_modelo is not None

    amostra = dataset.X_teste.head(1)
    predicoes, probabilidades = prever_amostra(melhor_modelo, amostra)
    classificacao = "maligno" if int(predicoes[0]) == settings.positive_label else "benigno"
    contexto_clinico = "Paciente com exame de imagem e achados citológicos em investigação de lesão mamária."
    llm_service = DiagnosticExplanationService(build_llm_client(llm_settings), llm_settings.output_path)
    explicacao = llm_service.generate_explanation(
        classificacao=classificacao,
        probabilidade=float(probabilidades[0]),
        metrics=melhor_resultado.metrics_teste,
        contexto_clinico=contexto_clinico,
    )
    comparacao = {
        "baseline_vs_melhor": {
            "recall_delta": melhor_resultado.metrics_teste.recall - baseline.metrics_teste.recall,
            "specificity_delta": melhor_resultado.metrics_teste.specificity - baseline.metrics_teste.specificity,
            "f1_delta": melhor_resultado.metrics_teste.f1_score - baseline.metrics_teste.f1_score,
        }
    }

    return {
        "dataset": {"nome": dataset.nome, "descricao": dataset.descricao, "amostras": len(dataset.atributos)},
        "baseline": baseline.to_dict(),
        "experimentos_geneticos": [item.to_dict() for item in experimentos_otimizados],
        "melhor_experimento": melhor_resultado.to_dict(),
        "comparacao": comparacao,
        "explicacao_llm": asdict(explicacao),
    }


def main() -> None:
    """Ponto de entrada da aplicação."""

    payload = executar_experimentos()
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
