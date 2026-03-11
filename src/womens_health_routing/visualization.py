from __future__ import annotations

from matplotlib.figure import Figure

from .domain import SERVICE_LABELS, ProblemInstance, Route, ServiceType, Solution


SERVICE_COLORS = {
    ServiceType.OBSTETRIC_EMERGENCY: "#c1121f",
    ServiceType.DOMESTIC_VIOLENCE: "#6a040f",
    ServiceType.HORMONAL_MEDICATION: "#3a86ff",
    ServiceType.POSTPARTUM: "#ff7f11",
    ServiceType.ONCOLOGY_SUPPORT: "#588157",
}


def build_route_figure(problem: ProblemInstance, solution: Solution, allowed_visit_ids: set[str] | None = None) -> Figure:
    figure = Figure(figsize=(8, 6))
    axis = figure.add_subplot(1, 1, 1)
    axis.scatter(problem.depot.x, problem.depot.y, c="#111111", s=120, marker="s", label="Base")
    axis.annotate("Base", (problem.depot.x + 0.2, problem.depot.y + 0.2))

    for visit in problem.visits:
        if allowed_visit_ids is not None and visit.visit_id not in allowed_visit_ids:
            axis.scatter(visit.x, visit.y, c="#d9d9d9", s=50, alpha=0.35)
            continue
        axis.scatter(visit.x, visit.y, c=SERVICE_COLORS[visit.service_type], s=90)
        axis.annotate(visit.visit_id, (visit.x + 0.15, visit.y + 0.15), fontsize=8)

    for route in solution.routes:
        _draw_route(axis, problem, route, allowed_visit_ids)

    legend_handles = []
    for service_type, color in SERVICE_COLORS.items():
        handle = axis.scatter([], [], c=color, s=90, label=SERVICE_LABELS[service_type])
        legend_handles.append(handle)
    axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axis.set_title("Mapa operacional das rotas por tipo de atendimento")
    axis.set_xlabel("Coordenada X")
    axis.set_ylabel("Coordenada Y")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _draw_route(axis, problem: ProblemInstance, route: Route, allowed_visit_ids: set[str] | None = None) -> None:
    filtered_stops = [stop for stop in route.stops if allowed_visit_ids is None or stop.visit.visit_id in allowed_visit_ids]
    if not filtered_stops:
        return
    path_x = [problem.depot.x] + [stop.visit.x for stop in filtered_stops] + [problem.depot.x]
    path_y = [problem.depot.y] + [stop.visit.y for stop in filtered_stops] + [problem.depot.y]
    axis.plot(path_x, path_y, linewidth=1.8, alpha=0.65, label=route.vehicle.vehicle_id)
