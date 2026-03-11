from __future__ import annotations

from dataclasses import asdict, dataclass

from .baselines import nearest_feasible_baseline
from .domain import ProblemInstance
from .ga_solver import GeneticVRPSolver, SolverConfig


@dataclass(frozen=True)
class BenchmarkResult:
    baseline_fitness: float
    ga_fitness: float
    baseline_distance_km: float
    ga_distance_km: float
    baseline_unassigned: int
    ga_unassigned: int
    fitness_improvement_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_benchmark(problem: ProblemInstance, config: SolverConfig | None = None) -> BenchmarkResult:
    baseline = nearest_feasible_baseline(problem)
    ga_solution = GeneticVRPSolver(problem, config or SolverConfig()).solve()
    baseline_fitness = baseline.evaluation.fitness
    ga_fitness = ga_solution.evaluation.fitness
    improvement = 0.0 if baseline_fitness == 0 else ((baseline_fitness - ga_fitness) / baseline_fitness) * 100.0
    return BenchmarkResult(
        baseline_fitness=baseline_fitness,
        ga_fitness=ga_fitness,
        baseline_distance_km=baseline.evaluation.total_distance_km,
        ga_distance_km=ga_solution.evaluation.total_distance_km,
        baseline_unassigned=baseline.evaluation.unassigned_visits,
        ga_unassigned=ga_solution.evaluation.unassigned_visits,
        fitness_improvement_pct=round(improvement, 2),
    )
