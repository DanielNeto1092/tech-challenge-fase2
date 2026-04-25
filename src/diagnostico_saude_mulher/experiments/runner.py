from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from diagnostico_saude_mulher.config.settings import AppSettings, LLMSettings
from diagnostico_saude_mulher.data.datasets import DatasetBundle, carregar_dataset_cancer_mama
from diagnostico_saude_mulher.evaluation.metrics import FitnessWeights, calcular_fitness
from diagnostico_saude_mulher.genetic_algorithm.engine import GeneticConfig, HyperparameterGeneticOptimizer
from diagnostico_saude_mulher.llm.clients import build_llm_client
from diagnostico_saude_mulher.llm.service import DiagnosticExplanationService, ExplanationRecord
from diagnostico_saude_mulher.models.schemas import ExperimentResult
from diagnostico_saude_mulher.models.training import (
    criar_catalogo_modelos_baseline,
    criar_modelo_otimizado_por_nome,
    prever_amostra,
    treinar_e_avaliar_modelo,
)
from diagnostico_saude_mulher.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Resultado consolidado do pipeline completo."""

    dataset: DatasetBundle
    modelos_baseline: list[ExperimentResult]
    baseline: ExperimentResult
    experimentos_geneticos: list[ExperimentResult]
    melhor_experimento: ExperimentResult
    comparacao: dict[str, dict[str, float]]
    explicacao_llm: ExplanationRecord

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": {
                "nome": self.dataset.nome,
                "descricao": self.dataset.descricao,
                "amostras": len(self.dataset.atributos),
            },
            "modelos_baseline": [item.to_dict() for item in self.modelos_baseline],
            "baseline": self.baseline.to_dict(),
            "experimentos_geneticos": [item.to_dict() for item in self.experimentos_geneticos],
            "melhor_experimento": self.melhor_experimento.to_dict(),
            "comparacao": self.comparacao,
            "explicacao_llm": asdict(self.explicacao_llm),
        }


def build_genetic_configs() -> list[GeneticConfig]:
    """Retorna os experimentos padrão do algoritmo genético."""

    return [
        GeneticConfig(population_size=6, generations=3, mutation_rate=0.10, selection_strategy="tournament", random_seed=42),
        GeneticConfig(population_size=8, generations=4, mutation_rate=0.15, selection_strategy="tournament", random_seed=84),
        GeneticConfig(population_size=10, generations=5, mutation_rate=0.20, selection_strategy="roulette", random_seed=126),
    ]


def run_experiments(
    settings: AppSettings,
    dataset: DatasetBundle,
    configs: list[GeneticConfig] | None = None,
) -> tuple[list[ExperimentResult], ExperimentResult, list[ExperimentResult]]:
    """Executa baseline e experimentos genéticos."""

    catalogo_baseline = criar_catalogo_modelos_baseline(settings.random_seed)
    modelos_baseline: list[ExperimentResult] = []
    for nome_modelo, modelo in catalogo_baseline.items():
        LOGGER.info("Executando baseline do modelo %s", nome_modelo)
        bundle = treinar_e_avaliar_modelo(
            modelo,
            dataset.X_treino,
            dataset.y_treino,
            dataset.X_teste,
            dataset.y_teste,
            settings,
        )
        parametros = modelo.get_params()
        modelos_baseline.append(
            ExperimentResult(
                nome_experimento=f"baseline_{nome_modelo.lower().replace('classifier', '').replace('regression', '_regression')}",
                modelo=nome_modelo,
                algoritmo="baseline",
                parametros=parametros,
                metrics_cv=bundle.metrics_cv,
                metrics_teste=bundle.metrics_teste,
                fitness_cv=calcular_fitness(bundle.metrics_cv),
            )
        )

    baseline = choose_reference_baseline(modelos_baseline)
    LOGGER.info("Modelo de referencia para o AG selecionado por fitness: %s", baseline.modelo)

    experimentos_otimizados: list[ExperimentResult] = []
    experiment_configs = build_genetic_configs() if configs is None else configs
    for indice, config in enumerate(experiment_configs, start=1):
        LOGGER.info("Executando experimento genetico %s com config=%s", indice, config)
        optimizer = HyperparameterGeneticOptimizer(model_name=baseline.modelo, config=config, settings=settings)
        result = optimizer.optimize(dataset.X_treino, dataset.y_treino)
        modelo_otimizado = criar_modelo_otimizado_por_nome(baseline.modelo, result.best_params, settings.random_seed)
        bundle = treinar_e_avaliar_modelo(
            modelo_otimizado,
            dataset.X_treino,
            dataset.y_treino,
            dataset.X_teste,
            dataset.y_teste,
            settings,
        )
        experimentos_otimizados.append(
            ExperimentResult(
                nome_experimento=f"ag_experimento_{indice}",
                modelo=baseline.modelo,
                algoritmo="genetico",
                parametros=result.best_params,
                metrics_cv=bundle.metrics_cv,
                metrics_teste=bundle.metrics_teste,
                fitness_cv=result.best_fitness,
                historico_geracoes=[asdict(item) for item in result.history],
            )
        )
    return modelos_baseline, baseline, experimentos_otimizados


def choose_reference_baseline(
    modelos_baseline: list[ExperimentResult],
    fitness_weights: FitnessWeights | None = None,
) -> ExperimentResult:
    """Seleciona o baseline mais performático segundo a função de fitness em CV."""

    pesos = fitness_weights or FitnessWeights()
    return max(
        modelos_baseline,
        key=lambda item: (
            item.fitness_cv if item.fitness_cv is not None else calcular_fitness(item.metrics_cv, pesos),
            item.metrics_cv.recall,
            item.metrics_cv.f1_score,
        ),
    )


def choose_best_experiment(experimentos: list[ExperimentResult]) -> ExperimentResult:
    """Seleciona o melhor experimento com prioridade em recall no teste."""

    return max(
        experimentos,
        key=lambda item: (
            item.metrics_teste.recall,
            item.metrics_teste.f1_score,
            item.metrics_teste.specificity,
        ),
    )


def build_comparison_payload(baseline: ExperimentResult, melhor: ExperimentResult) -> dict[str, dict[str, float]]:
    """Gera payload comparativo entre baseline e melhor experimento."""

    return {
        "baseline_vs_melhor": {
            "modelo_referencia": baseline.modelo,
            "modelo_otimizado": melhor.modelo,
            "fitness_baseline_cv": float(calcular_fitness(baseline.metrics_cv)),
            "fitness_melhor_cv": float(calcular_fitness(melhor.metrics_cv)),
            "recall_delta": float(melhor.metrics_teste.recall - baseline.metrics_teste.recall),
            "specificity_delta": float(melhor.metrics_teste.specificity - baseline.metrics_teste.specificity),
            "f1_delta": float(melhor.metrics_teste.f1_score - baseline.metrics_teste.f1_score),
        }
    }


def generate_explanation_for_best_model(
    dataset: DatasetBundle,
    melhor_experimento: ExperimentResult,
    settings: AppSettings,
    llm_settings: LLMSettings,
) -> ExplanationRecord:
    """Gera explicação para uma amostra usando o melhor modelo encontrado."""

    modelo = criar_modelo_otimizado_por_nome(melhor_experimento.modelo, melhor_experimento.parametros, settings.random_seed)
    modelo.fit(dataset.X_treino, dataset.y_treino)
    amostra = dataset.X_teste.head(1)
    predicoes, probabilidades = prever_amostra(modelo, amostra)
    classificacao = "maligno" if int(predicoes[0]) == settings.positive_label else "benigno"
    llm_service = DiagnosticExplanationService(build_llm_client(llm_settings), llm_settings.output_path)
    return llm_service.generate_explanation(
        classificacao=classificacao,
        probabilidade=float(probabilidades[0]),
        metrics=melhor_experimento.metrics_teste,
        contexto_clinico="Paciente com exame de imagem e achados citologicos em investigacao de lesao mamaria.",
    )


def executar_pipeline_completo() -> PipelineExecutionResult:
    """Executa o pipeline completo do projeto."""

    load_dotenv()
    settings = AppSettings()
    llm_settings = LLMSettings()
    configure_logging(settings.log_level, settings.log_output_dir)
    dataset = carregar_dataset_cancer_mama(settings)
    modelos_baseline, baseline, experimentos = run_experiments(settings, dataset)
    melhor_experimento = choose_best_experiment(experimentos)
    comparacao = build_comparison_payload(baseline, melhor_experimento)
    explicacao = generate_explanation_for_best_model(dataset, melhor_experimento, settings, llm_settings)
    return PipelineExecutionResult(
        dataset=dataset,
        modelos_baseline=modelos_baseline,
        baseline=baseline,
        experimentos_geneticos=experimentos,
        melhor_experimento=melhor_experimento,
        comparacao=comparacao,
        explicacao_llm=explicacao,
    )


def save_experiment_payload(payload: dict[str, object], output_path: Path) -> Path:
    """Salva o payload de experimentos em JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path
