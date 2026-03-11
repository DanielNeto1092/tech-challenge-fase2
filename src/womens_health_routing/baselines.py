from __future__ import annotations

import math

from .domain import Evaluation, ProblemInstance, Route, RouteStop, Solution, Vehicle, Visit


def nearest_feasible_baseline(problem: ProblemInstance) -> Solution:
    remaining = problem.visits[:]
    routes: list[Route] = []
    notes: list[str] = []
    total_penalty = 0.0
    delayed_priority_penalty = 0.0
    late_visits = 0

    for vehicle in problem.vehicles:
        route, served, route_penalty, route_priority_penalty, route_late, route_notes = _build_greedy_route(vehicle, remaining, problem)
        routes.append(route)
        total_penalty += route_penalty
        delayed_priority_penalty += route_priority_penalty
        late_visits += route_late
        notes.extend(route_notes)
        remaining = [visit for visit in remaining if visit.visit_id not in served]

    if remaining:
        total_penalty += 5000.0 * len(remaining)
        notes.append(f"{len(remaining)} visitas ficaram sem atendimento no baseline guloso.")

    total_distance = sum(route.total_distance_km for route in routes)
    total_cost = sum(route.total_cost for route in routes)
    evaluation = Evaluation(
        total_distance_km=round(total_distance, 2),
        total_cost=round(total_cost, 2),
        total_penalty=round(total_penalty, 2),
        fitness=round(total_distance + total_cost + total_penalty, 2),
        late_visits=late_visits,
        unassigned_visits=len(remaining),
        delayed_priority_penalty=round(delayed_priority_penalty, 2),
        notes=notes,
    )
    chromosome = [visit.visit_id for visit in problem.visits]
    return Solution(chromosome=chromosome, routes=routes, evaluation=evaluation)


def _build_greedy_route(
    vehicle: Vehicle,
    remaining: list[Visit],
    problem: ProblemInstance,
) -> tuple[Route, set[str], float, float, int, list[str]]:
    current_x = problem.depot.x
    current_y = problem.depot.y
    current_time = problem.depot.start_minute
    total_distance = 0.0
    total_cost = 0.0
    used_capacity = 0
    served: set[str] = set()
    stops: list[RouteStop] = []
    penalty = 0.0
    priority_penalty = 0.0
    late_visits = 0
    notes: list[str] = []
    candidates = remaining[:]

    while candidates and len(stops) < vehicle.max_stops:
        feasible = []
        for visit in candidates:
            if vehicle.allowed_service_types and visit.service_type not in vehicle.allowed_service_types:
                continue
            if visit.requires_refrigeration and not vehicle.supports_refrigeration:
                continue
            if visit.requires_secure_protocol and not vehicle.supports_secure_protocol:
                continue
            if used_capacity + visit.demand_units > vehicle.max_supply_units:
                continue
            trip_distance = math.dist((current_x, current_y), (visit.x, visit.y))
            trip_minutes = math.ceil((trip_distance / vehicle.speed_kmh) * 60.0)
            return_distance = math.dist((visit.x, visit.y), (problem.depot.x, problem.depot.y))
            if total_distance + trip_distance + return_distance > vehicle.max_distance_km:
                continue
            arrival = current_time + trip_minutes
            service_start = max(arrival, visit.earliest_start)
            if visit.location_type.value == "home":
                if not (problem.depot.safe_home_start <= service_start <= problem.depot.safe_home_end):
                    continue
            if visit.location_type.value == "hospital":
                if not (visit.earliest_start <= service_start <= visit.latest_start):
                    continue
            if visit.max_transport_minutes is not None and trip_minutes > visit.max_transport_minutes:
                continue
            feasible.append((visit, trip_distance, trip_minutes, service_start))

        if not feasible:
            break

        feasible.sort(key=lambda item: (-item[0].priority, item[1], item[0].latest_start))
        visit, trip_distance, trip_minutes, service_start = feasible[0]
        departure = service_start + visit.service_minutes
        if service_start > visit.latest_start:
            late_minutes = service_start - visit.latest_start
            penalty += late_minutes * 14.0
            priority_penalty += late_minutes * visit.priority * 6.0
            late_visits += 1
        response_minutes = service_start - problem.depot.start_minute
        priority_penalty += response_minutes * visit.priority * 0.12
        stops.append(
            RouteStop(
                visit=visit,
                arrival_minute=current_time + trip_minutes,
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
        current_x, current_y, current_time = visit.x, visit.y, departure
        candidates = [candidate for candidate in candidates if candidate.visit_id != visit.visit_id]

    if stops:
        back_distance = math.dist((current_x, current_y), (problem.depot.x, problem.depot.y))
        total_distance += back_distance
        total_cost += back_distance * vehicle.cost_per_km
    else:
        notes.append(f"{vehicle.vehicle_id} nao recebeu visitas viaveis no baseline.")

    route = Route(
        vehicle=vehicle,
        stops=stops,
        total_distance_km=round(total_distance, 2),
        total_cost=round(total_cost, 2),
    )
    return route, served, penalty, priority_penalty, late_visits, notes
