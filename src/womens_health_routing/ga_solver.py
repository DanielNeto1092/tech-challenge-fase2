from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .domain import Evaluation, LocationType, ProblemInstance, Route, RouteStop, Solution, Vehicle, Visit


@dataclass(frozen=True)
class SolverConfig:
    population_size: int = 80
    generations: int = 120
    mutation_rate: float = 0.22
    tournament_size: int = 4
    elite_size: int = 6
    random_seed: int = 42
    specialized_vehicle_bias: float = 0.0


class GeneticVRPSolver:
    def __init__(self, problem: ProblemInstance, config: SolverConfig | None = None) -> None:
        self.problem = problem
        self.config = config or SolverConfig()
        self._random = random.Random(self.config.random_seed)
        self._visits_by_id = problem.visit_by_id()

    def solve(self) -> Solution:
        population = self._initialize_population()
        best = min((self._evaluate(chromosome) for chromosome in population), key=lambda item: item.evaluation.fitness)

        for _ in range(self.config.generations):
            scored_population = [self._evaluate(chromosome) for chromosome in population]
            scored_population.sort(key=lambda item: item.evaluation.fitness)
            if scored_population[0].evaluation.fitness < best.evaluation.fitness:
                best = scored_population[0]

            next_population = [solution.chromosome[:] for solution in scored_population[: self.config.elite_size]]
            while len(next_population) < self.config.population_size:
                parent1 = self._tournament_select(scored_population)
                parent2 = self._tournament_select(scored_population)
                child = self._order_crossover(parent1.chromosome, parent2.chromosome)
                child = self._mutate(child)
                next_population.append(child)
            population = next_population
        return best

    def _initialize_population(self) -> list[list[str]]:
        visit_ids = [visit.visit_id for visit in self.problem.visits]
        population = [self._priority_seed()]
        while len(population) < self.config.population_size:
            chromosome = visit_ids[:]
            self._random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _priority_seed(self) -> list[str]:
        return [
            visit.visit_id
            for visit in sorted(
                self.problem.visits,
                key=lambda item: (-item.priority, item.latest_start, self._distance_from_depot(item)),
            )
        ]

    def _distance_from_depot(self, visit: Visit) -> float:
        return math.dist((self.problem.depot.x, self.problem.depot.y), (visit.x, visit.y))

    def _evaluate(self, chromosome: list[str]) -> Solution:
        remaining = [self._visits_by_id[visit_id] for visit_id in chromosome]
        routes: list[Route] = []
        notes: list[str] = []
        total_penalty = 0.0
        delayed_priority_penalty = 0.0
        late_visits = 0

        for vehicle in self._vehicle_order():
            route, served, route_penalty, route_priority_penalty, route_late, route_notes = self._build_route(vehicle, remaining)
            routes.append(route)
            total_penalty += route_penalty
            delayed_priority_penalty += route_priority_penalty
            late_visits += route_late
            notes.extend(route_notes)
            remaining = [visit for visit in remaining if visit.visit_id not in served]

        if remaining:
            total_penalty += 5000.0 * len(remaining)
            notes.append(f"{len(remaining)} visitas ficaram sem atendimento na frota disponivel.")

        total_distance = sum(route.total_distance_km for route in routes)
        total_cost = sum(route.total_cost for route in routes)
        fitness = total_distance + total_cost + total_penalty
        evaluation = Evaluation(
            total_distance_km=round(total_distance, 2),
            total_cost=round(total_cost, 2),
            total_penalty=round(total_penalty, 2),
            fitness=round(fitness, 2),
            late_visits=late_visits,
            unassigned_visits=len(remaining),
            delayed_priority_penalty=round(delayed_priority_penalty, 2),
            notes=notes,
        )
        return Solution(chromosome=chromosome[:], routes=routes, evaluation=evaluation)

    def _vehicle_order(self) -> list[Vehicle]:
        if self.config.specialized_vehicle_bias <= 0:
            return list(self.problem.vehicles)
        return sorted(
            self.problem.vehicles,
            key=lambda vehicle: (
                len(vehicle.allowed_service_types) if vehicle.allowed_service_types is not None else 999,
                vehicle.max_supply_units,
                vehicle.max_stops,
            ),
        )

    def _build_route(
        self,
        vehicle: Vehicle,
        remaining: list[Visit],
    ) -> tuple[Route, set[str], float, float, int, list[str]]:
        current_x = self.problem.depot.x
        current_y = self.problem.depot.y
        current_time = self.problem.depot.start_minute
        total_distance = 0.0
        total_cost = 0.0
        used_capacity = 0
        served: set[str] = set()
        stops: list[RouteStop] = []
        penalty = 0.0
        priority_penalty = 0.0
        late_visits = 0
        notes: list[str] = []

        for visit in remaining:
            if len(stops) >= vehicle.max_stops:
                continue
            if not self._vehicle_can_serve_visit(vehicle, visit):
                continue

            trip_distance = math.dist((current_x, current_y), (visit.x, visit.y))
            trip_minutes = self._travel_minutes(trip_distance, vehicle.speed_kmh)
            return_distance = math.dist((visit.x, visit.y), (self.problem.depot.x, self.problem.depot.y))
            projected_distance = total_distance + trip_distance + return_distance
            if projected_distance > vehicle.max_distance_km:
                continue

            arrival = current_time + trip_minutes
            service_start = max(arrival, visit.earliest_start)
            if not self._respects_temporal_constraints(visit, service_start, trip_minutes):
                continue
            departure = service_start + visit.service_minutes
            if service_start > visit.latest_start:
                late_minutes = service_start - visit.latest_start
                penalty += late_minutes * 14.0
                priority_penalty += late_minutes * visit.priority * 6.0
                late_visits += 1
            response_minutes = service_start - self.problem.depot.start_minute
            priority_penalty += response_minutes * visit.priority * 0.12

            stops.append(
                RouteStop(
                    visit=visit,
                    arrival_minute=arrival,
                    service_start_minute=service_start,
                    departure_minute=departure,
                    distance_from_previous=round(trip_distance, 2),
                    vehicle_id=vehicle.vehicle_id,
                )
            )
            served.add(visit.visit_id)
            used_capacity += visit.demand_units
            total_distance += trip_distance
            total_cost += trip_distance * vehicle.cost_per_km
            current_x = visit.x
            current_y = visit.y
            current_time = departure

        if stops:
            back_distance = math.dist((current_x, current_y), (self.problem.depot.x, self.problem.depot.y))
            total_distance += back_distance
            total_cost += back_distance * vehicle.cost_per_km
        else:
            notes.append(f"{vehicle.vehicle_id} nao recebeu visitas viaveis nesta iteracao.")

        if total_distance > vehicle.max_distance_km:
            overflow = total_distance - vehicle.max_distance_km
            penalty += overflow * 60.0
            notes.append(f"{vehicle.vehicle_id} excedeu distancia maxima em {overflow:.2f} km.")

        route = Route(
            vehicle=vehicle,
            stops=stops,
            total_distance_km=round(total_distance, 2),
            total_cost=round(total_cost, 2),
        )
        return route, served, penalty, priority_penalty, late_visits, notes

    def _vehicle_can_serve_visit(self, vehicle: Vehicle, visit: Visit) -> bool:
        if vehicle.allowed_service_types and visit.service_type not in vehicle.allowed_service_types:
            return False
        if visit.requires_refrigeration and not vehicle.supports_refrigeration:
            return False
        if visit.requires_secure_protocol and not vehicle.supports_secure_protocol:
            return False
        return True

    def _respects_temporal_constraints(self, visit: Visit, service_start: int, trip_minutes: int) -> bool:
        if visit.location_type == LocationType.HOME:
            if service_start < self.problem.depot.safe_home_start or service_start > self.problem.depot.safe_home_end:
                return False
        if visit.location_type == LocationType.HOSPITAL:
            if service_start < visit.earliest_start or service_start > visit.latest_start:
                return False
        if visit.max_transport_minutes is not None and trip_minutes > visit.max_transport_minutes:
            return False
        return True

    def _travel_minutes(self, distance_km: float, speed_kmh: float) -> int:
        return math.ceil((distance_km / speed_kmh) * 60.0)

    def _tournament_select(self, population: list[Solution]) -> Solution:
        contestants = self._random.sample(population, k=min(self.config.tournament_size, len(population)))
        return min(contestants, key=lambda item: item.evaluation.fitness)

    def _order_crossover(self, parent1: list[str], parent2: list[str]) -> list[str]:
        if len(parent1) < 2:
            return parent1[:]
        start = self._random.randint(0, len(parent1) - 2)
        end = self._random.randint(start + 1, len(parent1) - 1)
        child: list[str | None] = [None] * len(parent1)
        child[start : end + 1] = parent1[start : end + 1]
        missing = [gene for gene in parent2 if gene not in child]
        cursor = 0
        for index, gene in enumerate(child):
            if gene is None:
                child[index] = missing[cursor]
                cursor += 1
        return [gene for gene in child if gene is not None]

    def _mutate(self, chromosome: list[str]) -> list[str]:
        mutated = chromosome[:]
        if self._random.random() >= self.config.mutation_rate or len(mutated) < 2:
            return mutated
        left = self._random.randint(0, len(mutated) - 2)
        right = self._random.randint(left + 1, len(mutated) - 1)
        mutated[left : right + 1] = reversed(mutated[left : right + 1])
        return mutated
