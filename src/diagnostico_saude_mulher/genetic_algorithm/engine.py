from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from diagnostico_saude_mulher.config.settings import AppSettings
from diagnostico_saude_mulher.evaluation.metrics import FitnessWeights, calcular_fitness
from diagnostico_saude_mulher.models.schemas import ClassificationMetrics
from diagnostico_saude_mulher.models.training import avaliar_por_validacao_cruzada, criar_modelo_random_forest

LOGGER = logging.getLogger(__name__)

ESPACO_BUSCA: dict[str, list[Any]] = {
    "n_estimators": [80, 120, 160, 220, 300],
    "max_depth": [3, 4, 5, 6, 8, 10, 12],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": ["sqrt", "log2", None],
}


@dataclass(frozen=True)
class GeneticConfig:
    """Configuração do algoritmo genético."""

    population_size: int = 12
    generations: int = 8
    mutation_rate: float = 0.15
    crossover_rate: float = 0.9
    elite_size: int = 2
    tournament_size: int = 3
    selection_strategy: str = "tournament"
    random_seed: int = 42


@dataclass
class Individual:
    """Representa um indivíduo da população."""

    genes: dict[str, Any]
    fitness: float = 0.0
    metrics: ClassificationMetrics | None = None

    def clone(self) -> "Individual":
        return Individual(genes=copy.deepcopy(self.genes), fitness=self.fitness, metrics=self.metrics)


@dataclass(frozen=True)
class GenerationRecord:
    """Registro resumido de uma geração."""

    generation: int
    best_fitness: float
    avg_fitness: float
    best_genes: dict[str, Any]
    best_recall: float
    best_specificity: float
    best_f1_score: float


@dataclass(frozen=True)
class GeneticSearchResult:
    """Resultado final da busca genética."""

    best_params: dict[str, Any]
    best_fitness: float
    best_metrics_cv: ClassificationMetrics
    history: list[GenerationRecord] = field(default_factory=list)


class HyperparameterGeneticOptimizer:
    """Otimizador genético de hiperparâmetros para RandomForest."""

    def __init__(
        self,
        config: GeneticConfig,
        settings: AppSettings,
        fitness_weights: FitnessWeights | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.fitness_weights = fitness_weights or FitnessWeights()
        self._random = random.Random(config.random_seed)
        self._fitness_cache: dict[tuple[tuple[str, Any], ...], tuple[ClassificationMetrics, float]] = {}

    def optimize(self, X_treino: pd.DataFrame, y_treino: pd.Series) -> GeneticSearchResult:
        """Executa o AG e retorna o melhor conjunto de hiperparâmetros."""

        population = self._initialize_population()
        history: list[GenerationRecord] = []

        for generation in range(self.config.generations):
            self._evaluate_population(population, X_treino, y_treino)
            population.sort(key=lambda item: item.fitness, reverse=True)
            history.append(self._build_generation_record(generation, population))
            LOGGER.info(
                "Geracao %s | melhor fitness=%.4f | recall=%.4f | params=%s",
                generation,
                population[0].fitness,
                population[0].metrics.recall if population[0].metrics else 0.0,
                population[0].genes,
            )
            population = self._evolve_population(population)

        self._evaluate_population(population, X_treino, y_treino)
        population.sort(key=lambda item: item.fitness, reverse=True)
        best = population[0]
        assert best.metrics is not None
        return GeneticSearchResult(
            best_params=best.genes,
            best_fitness=best.fitness,
            best_metrics_cv=best.metrics,
            history=history,
        )

    def _initialize_population(self) -> list[Individual]:
        return [Individual(genes=self._sample_genes()) for _ in range(self.config.population_size)]

    def _sample_genes(self) -> dict[str, Any]:
        return {nome: self._random.choice(valores) for nome, valores in ESPACO_BUSCA.items()}

    def _evaluate_population(
        self,
        population: list[Individual],
        X_treino: pd.DataFrame,
        y_treino: pd.Series,
    ) -> None:
        for individual in population:
            if individual.metrics is not None:
                continue
            cache_key = tuple(sorted(individual.genes.items()))
            cached = self._fitness_cache.get(cache_key)
            if cached is not None:
                individual.metrics, individual.fitness = cached
                continue
            modelo = criar_modelo_random_forest(individual.genes, self.settings.random_seed)
            metrics = avaliar_por_validacao_cruzada(modelo, X_treino, y_treino, self.settings)
            individual.metrics = metrics
            individual.fitness = calcular_fitness(metrics, self.fitness_weights)
            self._fitness_cache[cache_key] = (metrics, individual.fitness)

    def _evolve_population(self, population: list[Individual]) -> list[Individual]:
        elites = [individual.clone() for individual in population[: self.config.elite_size]]
        next_population = elites
        while len(next_population) < self.config.population_size:
            parent1 = self._select_parent(population)
            parent2 = self._select_parent(population)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            next_population.append(child)
        return next_population[: self.config.population_size]

    def _select_parent(self, population: list[Individual]) -> Individual:
        if self.config.selection_strategy == "roulette":
            total_fitness = sum(max(ind.fitness, 0.0001) for ind in population)
            threshold = self._random.uniform(0.0, total_fitness)
            cumulative = 0.0
            for individual in population:
                cumulative += max(individual.fitness, 0.0001)
                if cumulative >= threshold:
                    return individual
            return population[-1]
        contestants = self._random.sample(population, k=min(self.config.tournament_size, len(population)))
        return max(contestants, key=lambda item: item.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        if self._random.random() > self.config.crossover_rate:
            return parent1.clone()
        child_genes: dict[str, Any] = {}
        for nome in ESPACO_BUSCA:
            child_genes[nome] = parent1.genes[nome] if self._random.random() < 0.5 else parent2.genes[nome]
        return Individual(genes=child_genes)

    def _mutate(self, individual: Individual) -> Individual:
        mutated = individual.clone()
        changed = False
        for nome, valores in ESPACO_BUSCA.items():
            if self._random.random() < self.config.mutation_rate:
                mutated.genes[nome] = self._random.choice(valores)
                changed = True
        if changed:
            mutated.metrics = None
            mutated.fitness = 0.0
        return mutated

    def _build_generation_record(self, generation: int, population: list[Individual]) -> GenerationRecord:
        best = population[0]
        assert best.metrics is not None
        avg_fitness = sum(item.fitness for item in population) / len(population)
        return GenerationRecord(
            generation=generation,
            best_fitness=best.fitness,
            avg_fitness=avg_fitness,
            best_genes=copy.deepcopy(best.genes),
            best_recall=best.metrics.recall,
            best_specificity=best.metrics.specificity,
            best_f1_score=best.metrics.f1_score,
        )
