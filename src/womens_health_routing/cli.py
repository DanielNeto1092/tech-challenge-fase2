from __future__ import annotations

import json

from .ga_solver import GeneticVRPSolver, SolverConfig
from .reporting import NarrativeGenerator
from .sample_data import build_sample_problem


def main() -> None:
    problem = build_sample_problem()
    solver = GeneticVRPSolver(problem, SolverConfig())
    solution = solver.solve()
    report = NarrativeGenerator().generate(problem, solution)
    payload = {
        "fitness": solution.evaluation.fitness,
        "distance_km": solution.evaluation.total_distance_km,
        "cost": solution.evaluation.total_cost,
        "late_visits": solution.evaluation.late_visits,
        "unassigned_visits": solution.evaluation.unassigned_visits,
        "routes": [
            {
                "vehicle_id": route.vehicle.vehicle_id,
                "stops": [stop.visit.visit_id for stop in route.stops],
                "distance_km": route.total_distance_km,
            }
            for route in solution.routes
        ],
        "qa_examples": report.qa_examples,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
