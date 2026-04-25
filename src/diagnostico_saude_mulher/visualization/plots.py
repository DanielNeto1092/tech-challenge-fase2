from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from diagnostico_saude_mulher.models.schemas import ExperimentResult


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_convergence(experimento: ExperimentResult, output_dir: Path) -> Path:
    """Gera gráfico de convergência do AG para um experimento."""

    _ensure_output_dir(output_dir)
    history = experimento.historico_geracoes
    generations = [item["generation"] for item in history]
    best = [item["best_fitness"] for item in history]
    avg = [item["avg_fitness"] for item in history]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(generations, best, label="Melhor fitness", linewidth=2)
    axis.plot(generations, avg, label="Fitness medio", linewidth=2, linestyle="--")
    axis.set_title(f"Convergencia - {experimento.rotulo_exibicao}")
    axis.set_xlabel("Geracao")
    axis.set_ylabel("Fitness")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()

    path = output_dir / f"convergencia_{experimento.nome_experimento}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_experiment_comparison(baseline: ExperimentResult, experimentos: list[ExperimentResult], output_dir: Path) -> Path:
    """Gera gráfico de barras comparando baseline de referência e experimentos."""

    _ensure_output_dir(output_dir)
    labels = ["Recall", "Especificidade", "F1-score"]
    baseline_values = [
        baseline.metrics_teste.recall,
        baseline.metrics_teste.specificity,
        baseline.metrics_teste.f1_score,
    ]
    x = np.arange(len(labels))
    width = 0.18

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(x - width * 1.5, baseline_values, width, label=baseline.rotulo_exibicao)

    offsets = [-0.5, 0.5, 1.5]
    for offset, experimento in zip(offsets, experimentos):
        values = [
            experimento.metrics_teste.recall,
            experimento.metrics_teste.specificity,
            experimento.metrics_teste.f1_score,
        ]
        axis.bar(x + width * offset, values, width, label=experimento.rotulo_exibicao)

    axis.set_title(f"Comparacao de metricas no conjunto de teste - referencia: {baseline.modelo}")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    axis.legend()

    path = output_dir / "comparacao_experimentos.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_experiment_summary(experimentos: list[ExperimentResult], output_dir: Path) -> Path:
    """Gera gráfico resumo do fitness final dos experimentos."""

    _ensure_output_dir(output_dir)
    labels = [item.rotulo_exibicao for item in experimentos]
    values = [item.historico_geracoes[-1]["best_fitness"] for item in experimentos]

    figure, axis = plt.subplots(figsize=(10, 6))
    color_cycle = ["#7bafd4", "#f4a261", "#3a7d44", "#d1495b", "#4d908e"]
    bars = axis.bar(labels, values, color=[color_cycle[index % len(color_cycle)] for index in range(len(labels))])
    axis.set_title("Resumo de fitness final por experimento")
    axis.set_ylabel("Fitness")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center")

    path = output_dir / "resumo_experimentos.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_generation_metrics(experimento: ExperimentResult, output_dir: Path) -> Path:
    """Gera gráfico da evolução de recall, especificidade e F1 por geração."""

    _ensure_output_dir(output_dir)
    history = experimento.historico_geracoes
    generations = [item["generation"] for item in history]
    recall = [item["best_recall"] for item in history]
    specificity = [item["best_specificity"] for item in history]
    f1_score = [item["best_f1_score"] for item in history]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(generations, recall, label="Recall", linewidth=2)
    axis.plot(generations, specificity, label="Especificidade", linewidth=2)
    axis.plot(generations, f1_score, label="F1-score", linewidth=2)
    axis.set_title(f"Evolucao das metricas - {experimento.rotulo_exibicao}")
    axis.set_xlabel("Geracao")
    axis.set_ylabel("Valor")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()

    path = output_dir / f"metricas_por_geracao_{experimento.nome_experimento}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_hyperparameter_behavior(experimentos: list[ExperimentResult], output_dir: Path) -> Path:
    """Gera gráfico mostrando o comportamento do fitness ao longo das gerações."""

    _ensure_output_dir(output_dir)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    history_points = []
    for experimento in experimentos:
        for registro in experimento.historico_geracoes:
            history_points.append(
                {
                    "label": f"{experimento.rotulo_exibicao}-g{registro['generation']}",
                    "fitness": registro["best_fitness"],
                    "genes": registro.get("best_genes", experimento.parametros),
                }
            )

    if not history_points:
        path = output_dir / "comportamento_hiperparametros.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path

    first_genes = history_points[0]["genes"]
    numeric_params = [
        param
        for param, value in first_genes.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    ][:4]
    categorical_params = [
        param
        for param, value in first_genes.items()
        if not (isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool))
    ]

    for index, param in enumerate(numeric_params):
        axis = axes[index]
        values = [item["genes"][param] for item in history_points]
        fitness = [item["fitness"] for item in history_points]
        axis.scatter(values, fitness, s=90, color="#3a7d44", alpha=0.75)
        axis.set_title(param)
        axis.set_xlabel("Valor observado no melhor individuo da geracao")
        axis.set_ylabel("Fitness")
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, linestyle="--", alpha=0.4)

    categorical_axis = axes[4]
    if categorical_params:
        param = categorical_params[0]
        categories = [str(item["genes"][param]) for item in history_points]
        fitness = [item["fitness"] for item in history_points]
        category_order = list(dict.fromkeys(categories))
        category_map = {name: idx for idx, name in enumerate(category_order)}
        x_values = [category_map[name] for name in categories]
        categorical_axis.scatter(x_values, fitness, s=90, color="#f4a261", alpha=0.75)
        categorical_axis.set_xticks(list(category_map.values()))
        categorical_axis.set_xticklabels(category_order)
        categorical_axis.set_title(param)
        categorical_axis.set_ylabel("Fitness")
        categorical_axis.set_ylim(0.0, 1.05)
        categorical_axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        categorical_axis.axis("off")
        categorical_axis.text(0.0, 0.8, "Sem hiperparametro categorico\nno modelo otimizado.", fontsize=11, va="top")

    axes[5].axis("off")
    axes[5].text(
        0.0,
        0.9,
        f"Leitura:\n- Cada ponto representa o melhor individuo de uma geracao.\n- O eixo Y mostra o fitness observado naquela geracao.\n- O modelo otimizado nesta execucao foi: {experimentos[0].modelo}.",
        fontsize=11,
        va="top",
    )

    for hidden_index in range(len(numeric_params), 4):
        axes[hidden_index].axis("off")

    path = output_dir / "comportamento_hiperparametros.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
