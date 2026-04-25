from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

from diagnostico_saude_mulher.study.data import StudyDataset, load_breast_cancer_study_dataset
from diagnostico_saude_mulher.study.evaluation import MetricsResult, compute_fitness, compute_metrics
from diagnostico_saude_mulher.study.genetic_algorithm import (
    GAExperimentConfig,
    GAOptimizationResult,
    GeneticOptimizer,
    generation_history_to_dict,
)
from diagnostico_saude_mulher.study.models import MODEL_SPECS, build_baseline_model, build_optimized_model
from diagnostico_saude_mulher.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


EXPERIMENT_CONFIGS: list[GAExperimentConfig] = [
    GAExperimentConfig(
        name="experimento_1_exploracao",
        population_size=20,
        generations=15,
        crossover_rate=0.80,
        mutation_rate=0.30,
        tournament_k=3,
        elitism=2,
    ),
    GAExperimentConfig(
        name="experimento_2_equilibrio",
        population_size=30,
        generations=20,
        crossover_rate=0.85,
        mutation_rate=0.15,
        tournament_k=4,
        elitism=3,
    ),
    GAExperimentConfig(
        name="experimento_3_refinamento",
        population_size=40,
        generations=25,
        crossover_rate=0.90,
        mutation_rate=0.05,
        tournament_k=5,
        elitism=4,
    ),
]


@dataclass(frozen=True)
class StudyModelResult:
    model_name: str
    baseline_metrics: MetricsResult
    best_experiment_name: str
    optimized_metrics: MetricsResult
    baseline_fitness: float
    optimized_fitness: float
    optimized_params: dict[str, Any]
    experiments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "baseline_fitness": self.baseline_fitness,
            "best_experiment_name": self.best_experiment_name,
            "optimized_metrics": self.optimized_metrics.to_dict(),
            "optimized_fitness": self.optimized_fitness,
            "optimized_params": self.optimized_params,
            "experiments": self.experiments,
        }


def _evaluate_baseline(dataset: StudyDataset, model_name: str, random_state: int) -> MetricsResult:
    model = build_baseline_model(model_name, random_state=random_state)
    model.fit(dataset.X_train, dataset.y_train)
    predictions = model.predict(dataset.X_test)
    return compute_metrics(dataset.y_test, predictions)


def _prepare_inner_validation(
    dataset: StudyDataset,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        dataset.X_train,
        dataset.y_train,
        test_size=0.25,
        stratify=dataset.y_train,
        random_state=random_state,
    )


def _choose_best_experiment(results: list[tuple[GAOptimizationResult, MetricsResult]]) -> tuple[GAOptimizationResult, MetricsResult]:
    return max(
        results,
        key=lambda item: (
            item[0].best_fitness,
            item[1].recall,
            item[1].f1_score,
            item[1].specificity,
        ),
    )


def _plot_convergence(model_name: str, result: GAOptimizationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    generations = [item.generation for item in result.history]
    best = [item.best_fitness for item in result.history]
    avg = [item.avg_fitness for item in result.history]
    unique_individuals = [item.unique_individuals for item in result.history]
    diversity_ratio = [item.diversity_ratio for item in result.history]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axis_fitness = axes[0, 0]
    axis_diversity = axes[0, 1]
    axis_unique = axes[1, 0]
    axis_params = axes[1, 1]

    axis_fitness.plot(generations, best, label="Melhor fitness", linewidth=2)
    axis_fitness.plot(generations, avg, label="Fitness medio", linewidth=2, linestyle="--")
    axis_fitness.set_title("Convergencia de fitness")
    axis_fitness.set_xlabel("Geracao")
    axis_fitness.set_ylabel("Fitness")
    axis_fitness.grid(True, linestyle="--", alpha=0.4)
    axis_fitness.legend()

    axis_diversity.plot(generations, diversity_ratio, color="tab:green", linewidth=2)
    axis_diversity.set_title("Diversidade da populacao")
    axis_diversity.set_xlabel("Geracao")
    axis_diversity.set_ylabel("Razao de diversidade")
    axis_diversity.set_ylim(0.0, 1.05)
    axis_diversity.grid(True, linestyle="--", alpha=0.4)

    axis_unique.bar(generations, unique_individuals, color="tab:orange", alpha=0.8)
    axis_unique.set_title("Individuos unicos por geracao")
    axis_unique.set_xlabel("Geracao")
    axis_unique.set_ylabel("Contagem")
    axis_unique.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis_unique.grid(True, axis="y", linestyle="--", alpha=0.4)

    gene_frame = pd.DataFrame([item.best_genes for item in result.history])
    numeric_columns = [
        column for column in gene_frame.columns if pd.api.types.is_numeric_dtype(gene_frame[column])
    ]
    categorical_columns = [column for column in gene_frame.columns if column not in numeric_columns]

    if numeric_columns:
        for column in numeric_columns:
            axis_params.plot(generations, gene_frame[column], linewidth=2, label=column)
        axis_params.set_ylabel("Valor")
    if categorical_columns:
        for column in categorical_columns:
            categories = list(dict.fromkeys(gene_frame[column].tolist()))
            mapping = {value: index for index, value in enumerate(categories)}
            encoded = gene_frame[column].map(mapping)
            axis_params.plot(generations, encoded, linewidth=2, linestyle="--", label=column)
            axis_params.set_yticks(list(mapping.values()))
            axis_params.set_yticklabels(list(mapping.keys()))

    axis_params.set_title("Evolucao dos parametros do melhor individuo")
    axis_params.set_xlabel("Geracao")
    axis_params.grid(True, linestyle="--", alpha=0.4)
    axis_params.legend(fontsize=8)

    figure.suptitle(f"{model_name} - {result.experiment_name}", fontsize=14)

    path = output_dir / f"convergencia_{model_name}_{result.experiment_name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_baseline_vs_optimized(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("Recall", "recall_baseline", "recall_otimizado"),
        ("Especificidade", "specificity_baseline", "specificity_otimizado"),
        ("F1-score", "f1_baseline", "f1_otimizado"),
    ]

    for axis, (title, baseline_col, optimized_col) in zip(axes, metrics):
        x = range(len(summary))
        axis.bar([value - 0.2 for value in x], summary[baseline_col], width=0.4, label="Baseline")
        axis.bar([value + 0.2 for value in x], summary[optimized_col], width=0.4, label="Otimizado")
        axis.set_title(title)
        axis.set_xticks(list(x))
        axis.set_xticklabels(summary["modelo"], rotation=20)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0].legend()
    path = output_dir / "comparacao_baseline_vs_otimizado.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _build_summary_dataframe(results: list[StudyModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "modelo": item.model_name,
                "recall_baseline": item.baseline_metrics.recall,
                "recall_otimizado": item.optimized_metrics.recall,
                "delta_recall": item.optimized_metrics.recall - item.baseline_metrics.recall,
                "specificity_baseline": item.baseline_metrics.specificity,
                "specificity_otimizado": item.optimized_metrics.specificity,
                "f1_baseline": item.baseline_metrics.f1_score,
                "f1_otimizado": item.optimized_metrics.f1_score,
                "accuracy_baseline": item.baseline_metrics.accuracy,
                "accuracy_otimizado": item.optimized_metrics.accuracy,
                "melhor_experimento": item.best_experiment_name,
            }
            for item in results
        ]
    )


def _build_automatic_analysis(results: list[StudyModelResult], summary: pd.DataFrame) -> str:
    maior_ganho = summary.sort_values("delta_recall", ascending=False).iloc[0]
    melhor_equilibrio = max(results, key=lambda item: (item.optimized_fitness, item.optimized_metrics.recall, item.optimized_metrics.f1_score))
    tradeoffs = []
    for row in summary.itertuples(index=False):
        delta_spec = row.specificity_otimizado - row.specificity_baseline
        tradeoffs.append(
            f"{row.modelo}: delta recall {row.delta_recall:.4f}, delta especificidade {delta_spec:.4f}, "
            f"F1 final {row.f1_otimizado:.4f}."
        )

    return (
        f"O maior ganho de recall foi obtido por {maior_ganho.modelo}, com delta de {maior_ganho.delta_recall:.4f}. "
        f"Os principais trade-offs observados foram: {' '.join(tradeoffs)} "
        f"Considerando a funcao de fitness e o desempenho final, o melhor equilibrio geral foi obtido por "
        f"{melhor_equilibrio.model_name}. Para uso clinico, a recomendacao automatica e priorizar o modelo "
        f"{melhor_equilibrio.model_name}, pois ele combinou o maior desempenho otimizado com recall elevado, "
        f"boa especificidade e F1-score consistente."
    )


def run_full_study(random_state: int = 42) -> dict[str, Any]:
    configure_logging("INFO", Path("artifacts/logs"))
    dataset = load_breast_cancer_study_dataset(random_state=random_state)
    X_train_fit, X_val, y_train_fit, y_val = _prepare_inner_validation(dataset, random_state)

    results: list[StudyModelResult] = []
    output_dir = Path("artifacts/ga_study")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name, spec in MODEL_SPECS.items():
        LOGGER.info("Executando estudo para o modelo %s", model_name)
        baseline_metrics = _evaluate_baseline(dataset, model_name, random_state)
        baseline_fitness = compute_fitness(baseline_metrics)

        experiment_runs: list[tuple[GAOptimizationResult, MetricsResult]] = []
        experiment_payloads: list[dict[str, Any]] = []
        for index, config in enumerate(EXPERIMENT_CONFIGS, start=1):
            optimizer = GeneticOptimizer(
                model_name=model_name,
                search_space=spec.search_space,
                config=config,
                random_state=random_state + index,
            )
            optimization_result = optimizer.optimize(X_train_fit, y_train_fit, X_val, y_val)
            final_model = build_optimized_model(model_name, optimization_result.best_genes, random_state=random_state)
            final_model.fit(dataset.X_train, dataset.y_train)
            test_predictions = final_model.predict(dataset.X_test)
            optimized_metrics = compute_metrics(dataset.y_test, test_predictions)
            experiment_runs.append((optimization_result, optimized_metrics))
            convergence_path = _plot_convergence(model_name, optimization_result, plots_dir)
            experiment_payloads.append(
                {
                    "experiment_name": config.name,
                    "config": asdict(config),
                    "best_genes": optimization_result.best_genes,
                    "best_fitness": optimization_result.best_fitness,
                    "validation_metrics": optimization_result.validation_metrics.to_dict(),
                    "test_metrics": optimized_metrics.to_dict(),
                    "history": generation_history_to_dict(optimization_result.history),
                    "convergence_plot": str(convergence_path),
                }
            )

        best_run, best_test_metrics = _choose_best_experiment(experiment_runs)
        results.append(
            StudyModelResult(
                model_name=model_name,
                baseline_metrics=baseline_metrics,
                best_experiment_name=best_run.experiment_name,
                optimized_metrics=best_test_metrics,
                baseline_fitness=baseline_fitness,
                optimized_fitness=best_run.best_fitness,
                optimized_params=best_run.best_genes,
                experiments=experiment_payloads,
            )
        )

    summary = _build_summary_dataframe(results)
    comparison_plot = _plot_baseline_vs_optimized(summary, plots_dir)
    analysis_text = _build_automatic_analysis(results, summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_results.csv"
    summary.to_csv(summary_path, index=False)
    analysis_path = output_dir / "automatic_analysis.txt"
    analysis_path.write_text(analysis_text, encoding="utf-8")

    payload = {
        "dataset": {
            "name": dataset.dataset_name,
            "train_size": len(dataset.X_train),
            "test_size": len(dataset.X_test),
        },
        "summary_table": summary.to_dict(orient="records"),
        "comparison_plot": str(comparison_plot),
        "analysis_text": analysis_text,
        "models": [item.to_dict() for item in results],
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload
