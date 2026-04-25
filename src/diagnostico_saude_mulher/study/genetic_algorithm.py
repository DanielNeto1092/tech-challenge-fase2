from __future__ import annotations

import copy
import logging
import random
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin

from diagnostico_saude_mulher.study.evaluation import MetricsResult, compute_fitness, compute_metrics
from diagnostico_saude_mulher.study.models import build_optimized_model

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GAExperimentConfig:
    name: str
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_k: int
    elitism: int


@dataclass
class GAIndividual:
    genes: dict[str, Any]
    fitness: float = 0.0
    metrics: MetricsResult | None = None

    def clone(self) -> "GAIndividual":
        return GAIndividual(genes=copy.deepcopy(self.genes), fitness=self.fitness, metrics=self.metrics)


@dataclass(frozen=True)
class GenerationHistory:
    generation: int
    best_fitness: float
    avg_fitness: float
    unique_individuals: int
    diversity_ratio: float
    best_genes: dict[str, Any]
    best_recall: float
    best_specificity: float
    best_f1_score: float


@dataclass(frozen=True)
class GAOptimizationResult:
    model_name: str
    experiment_name: str
    best_genes: dict[str, Any]
    best_fitness: float
    validation_metrics: MetricsResult
    history: list[GenerationHistory]


class GeneticOptimizer:
    """AG genérico para otimização de hiperparâmetros por família de modelo."""

    def __init__(
        self,
        model_name: str,
        search_space: dict[str, list[Any]],
        config: GAExperimentConfig,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.search_space = search_space
        self.config = config
        self.random = random.Random(random_state)
        self.cache: dict[tuple[tuple[str, Any], ...], tuple[MetricsResult, float]] = {}

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> GAOptimizationResult:
        population = [GAIndividual(genes=self._sample_genes()) for _ in range(self.config.population_size)]
        history: list[GenerationHistory] = []

        for generation in range(self.config.generations):
            self._evaluate_population(population, X_train, y_train, X_val, y_val)
            population.sort(key=lambda item: item.fitness, reverse=True)
            history.append(self._build_generation_history(generation, population))
            LOGGER.info(
                "Modelo=%s | Experimento=%s | Geracao=%s | melhor_fitness=%.4f | genes=%s",
                self.model_name,
                self.config.name,
                generation,
                population[0].fitness,
                population[0].genes,
            )
            population = self._next_population(population)

        self._evaluate_population(population, X_train, y_train, X_val, y_val)
        population.sort(key=lambda item: item.fitness, reverse=True)
        best = population[0]
        assert best.metrics is not None
        return GAOptimizationResult(
            model_name=self.model_name,
            experiment_name=self.config.name,
            best_genes=best.genes,
            best_fitness=best.fitness,
            validation_metrics=best.metrics,
            history=history,
        )

    def _sample_genes(self) -> dict[str, Any]:
        return {key: self.random.choice(values) for key, values in self.search_space.items()}

    def _evaluate_population(
        self,
        population: list[GAIndividual],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        for individual in population:
            if individual.metrics is not None:
                continue
            cache_key = tuple(sorted(individual.genes.items()))
            cached = self.cache.get(cache_key)
            if cached is not None:
                individual.metrics, individual.fitness = cached
                continue
            model = build_optimized_model(self.model_name, individual.genes)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            metrics = compute_metrics(y_val, predictions)
            fitness = compute_fitness(metrics)
            individual.metrics = metrics
            individual.fitness = fitness
            self.cache[cache_key] = (metrics, fitness)

    def _next_population(self, ranked_population: list[GAIndividual]) -> list[GAIndividual]:
        next_population = [individual.clone() for individual in ranked_population[: self.config.elitism]]
        while len(next_population) < self.config.population_size:
            parent1 = self._select_tournament(ranked_population)
            parent2 = self._select_tournament(ranked_population)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            next_population.append(child)
        return next_population[: self.config.population_size]

    def _select_tournament(self, population: list[GAIndividual]) -> GAIndividual:
        contenders = self.random.sample(population, k=min(self.config.tournament_k, len(population)))
        return max(contenders, key=lambda item: item.fitness)

    def _crossover(self, parent1: GAIndividual, parent2: GAIndividual) -> GAIndividual:
        if self.random.random() > self.config.crossover_rate:
            return parent1.clone()
        genes: dict[str, Any] = {}
        for key in self.search_space:
            genes[key] = parent1.genes[key] if self.random.random() < 0.5 else parent2.genes[key]
        return GAIndividual(genes=genes)

    def _mutate(self, individual: GAIndividual) -> GAIndividual:
        mutated = individual.clone()
        mutated_flag = False
        for key, values in self.search_space.items():
            if self.random.random() < self.config.mutation_rate:
                mutated.genes[key] = self.random.choice(values)
                mutated_flag = True
        if mutated_flag:
            mutated.metrics = None
            mutated.fitness = 0.0
        return mutated

    def _build_generation_history(self, generation: int, population: list[GAIndividual]) -> GenerationHistory:
        best = population[0]
        assert best.metrics is not None
        avg_fitness = sum(item.fitness for item in population) / len(population)
        unique_individuals = len({tuple(sorted(item.genes.items())) for item in population})
        return GenerationHistory(
            generation=generation,
            best_fitness=best.fitness,
            avg_fitness=avg_fitness,
            unique_individuals=unique_individuals,
            diversity_ratio=unique_individuals / len(population),
            best_genes=copy.deepcopy(best.genes),
            best_recall=best.metrics.recall,
            best_specificity=best.metrics.specificity,
            best_f1_score=best.metrics.f1_score,
        )


def generation_history_to_dict(history: list[GenerationHistory]) -> list[dict[str, Any]]:
    return [asdict(item) for item in history]
