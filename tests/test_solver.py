from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from womens_health_routing.benchmarking import run_benchmark
from womens_health_routing.domain import Depot, LocationType, ProblemInstance, ServiceType, Vehicle, VehicleType, Visit
from womens_health_routing.ga_solver import GeneticVRPSolver, SolverConfig
from womens_health_routing.reporting import NarrativeGenerator
from womens_health_routing.sample_data import build_sample_problem, build_synthetic_problem


class SolverTests(unittest.TestCase):
    def test_solution_covers_all_visits(self) -> None:
        problem = build_sample_problem()
        solver = GeneticVRPSolver(problem, SolverConfig(generations=40, population_size=40, random_seed=7))
        solution = solver.solve()
        served = {stop.visit.visit_id for route in solution.routes for stop in route.stops}
        self.assertEqual(served, {visit.visit_id for visit in problem.visits})
        self.assertEqual(solution.evaluation.unassigned_visits, 0)

    def test_refrigerated_visits_use_supported_vehicle(self) -> None:
        problem = build_sample_problem()
        solver = GeneticVRPSolver(problem, SolverConfig(generations=30, population_size=30, random_seed=11))
        solution = solver.solve()
        for route in solution.routes:
            for stop in route.stops:
                if stop.visit.requires_refrigeration:
                    self.assertTrue(route.vehicle.supports_refrigeration)

    def test_report_mentions_priority_guidance(self) -> None:
        problem = build_sample_problem()
        solver = GeneticVRPSolver(problem, SolverConfig(generations=20, population_size=25, random_seed=3))
        solution = solver.solve()
        bundle = NarrativeGenerator().generate(problem, solution)
        self.assertIn("Priorizar emergencias obstetricas", bundle.operations_manual)
        self.assertIn("Roteiro detalhado de visitas do dia.", bundle.visit_script)

    def test_vehicle_service_type_restriction_is_enforced(self) -> None:
        problem = ProblemInstance(
            name="servico restrito",
            depot=Depot(x=0, y=0, start_minute=8 * 60),
            vehicles=[
                Vehicle(
                    vehicle_id="CAR",
                    label="Carro",
                    vehicle_type=VehicleType.CAR,
                    max_distance_km=100,
                    max_stops=2,
                    max_supply_units=10,
                    speed_kmh=40,
                    cost_per_km=1.0,
                    allowed_service_types=(ServiceType.POSTPARTUM,),
                )
            ],
            visits=[
                Visit(
                    visit_id="EM1",
                    patient_name="Paciente",
                    x=3,
                    y=4,
                    service_type=ServiceType.OBSTETRIC_EMERGENCY,
                    demand_units=1,
                    service_minutes=10,
                    earliest_start=8 * 60,
                    latest_start=9 * 60,
                    location_type=LocationType.CLINIC,
                )
            ],
        )
        solution = GeneticVRPSolver(problem, SolverConfig(generations=5, population_size=6)).solve()
        self.assertEqual(solution.evaluation.unassigned_visits, 1)

    def test_safe_home_window_is_enforced(self) -> None:
        problem = ProblemInstance(
            name="janela segura",
            depot=Depot(x=0, y=0, start_minute=8 * 60, safe_home_start=8 * 60, safe_home_end=18 * 60),
            vehicles=[
                Vehicle(
                    vehicle_id="CAR",
                    label="Carro",
                    vehicle_type=VehicleType.CAR,
                    max_distance_km=100,
                    max_stops=2,
                    max_supply_units=10,
                    speed_kmh=30,
                    cost_per_km=1.0,
                    allowed_service_types=(ServiceType.POSTPARTUM,),
                )
            ],
            visits=[
                Visit(
                    visit_id="HOME1",
                    patient_name="Paciente",
                    x=2,
                    y=2,
                    service_type=ServiceType.POSTPARTUM,
                    demand_units=1,
                    service_minutes=10,
                    earliest_start=19 * 60,
                    latest_start=20 * 60,
                    location_type=LocationType.HOME,
                )
            ],
        )
        solution = GeneticVRPSolver(problem, SolverConfig(generations=5, population_size=6)).solve()
        self.assertEqual(solution.evaluation.unassigned_visits, 1)

    def test_hospital_window_is_enforced(self) -> None:
        problem = ProblemInstance(
            name="janela hospitalar",
            depot=Depot(x=0, y=0, start_minute=7 * 60),
            vehicles=[
                Vehicle(
                    vehicle_id="VAN",
                    label="Van",
                    vehicle_type=VehicleType.VAN,
                    max_distance_km=100,
                    max_stops=2,
                    max_supply_units=10,
                    speed_kmh=20,
                    cost_per_km=1.0,
                    allowed_service_types=(ServiceType.HORMONAL_MEDICATION,),
                    supports_refrigeration=True,
                )
            ],
            visits=[
                Visit(
                    visit_id="H1",
                    patient_name="Paciente",
                    x=35,
                    y=0,
                    service_type=ServiceType.HORMONAL_MEDICATION,
                    demand_units=1,
                    service_minutes=10,
                    earliest_start=8 * 60,
                    latest_start=8 * 60 + 30,
                    location_type=LocationType.HOSPITAL,
                    requires_refrigeration=True,
                )
            ],
        )
        solution = GeneticVRPSolver(problem, SolverConfig(generations=5, population_size=6)).solve()
        self.assertEqual(solution.evaluation.unassigned_visits, 1)

    def test_max_transport_time_is_enforced(self) -> None:
        problem = ProblemInstance(
            name="tempo de transporte",
            depot=Depot(x=0, y=0, start_minute=8 * 60),
            vehicles=[
                Vehicle(
                    vehicle_id="VAN",
                    label="Van",
                    vehicle_type=VehicleType.VAN,
                    max_distance_km=100,
                    max_stops=2,
                    max_supply_units=10,
                    speed_kmh=20,
                    cost_per_km=1.0,
                    allowed_service_types=(ServiceType.HORMONAL_MEDICATION,),
                    supports_refrigeration=True,
                )
            ],
            visits=[
                Visit(
                    visit_id="MED1",
                    patient_name="Paciente",
                    x=20,
                    y=0,
                    service_type=ServiceType.HORMONAL_MEDICATION,
                    demand_units=1,
                    service_minutes=10,
                    earliest_start=8 * 60,
                    latest_start=12 * 60,
                    location_type=LocationType.HOSPITAL,
                    requires_refrigeration=True,
                    max_transport_minutes=30,
                )
            ],
        )
        solution = GeneticVRPSolver(problem, SolverConfig(generations=5, population_size=6)).solve()
        self.assertEqual(solution.evaluation.unassigned_visits, 1)

    def test_benchmark_returns_comparison_payload(self) -> None:
        result = run_benchmark(build_sample_problem(), SolverConfig(generations=20, population_size=25, random_seed=5))
        self.assertIn("fitness_improvement_pct", result.to_dict())
        self.assertGreaterEqual(result.baseline_unassigned, result.ga_unassigned)

    def test_synthetic_problem_respects_requested_patient_count(self) -> None:
        problem = build_synthetic_problem(patient_count=35, seed=17)
        self.assertEqual(len(problem.visits), 35)

    @mock.patch.dict("os.environ", {}, clear=False)
    def test_llm_provider_falls_back_to_rule_based_without_endpoint(self) -> None:
        problem = build_sample_problem()
        solution = GeneticVRPSolver(problem, SolverConfig(generations=10, population_size=10, random_seed=2)).solve()
        bundle = NarrativeGenerator(llm_provider="http").generate(problem, solution)
        self.assertIn("Manual operacional", bundle.operations_manual)


if __name__ == "__main__":
    unittest.main()
