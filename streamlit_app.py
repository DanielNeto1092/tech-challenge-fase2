from __future__ import annotations

import os
import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from womens_health_routing.domain import LocationType, SERVICE_LABELS, ServiceType
from womens_health_routing.ga_solver import GeneticVRPSolver, SolverConfig
from womens_health_routing.reporting import NarrativeGenerator, minute_to_hhmm
from womens_health_routing.sample_data import build_sample_problem
from womens_health_routing.visualization import SERVICE_COLORS, build_route_figure


st.set_page_config(page_title="Rotas para Saude da Mulher", layout="wide")
st.title("Otimizacao de rotas para atendimento especializado a mulher")
st.caption("Algoritmo genetico com prioridades clinicas, janelas de tempo, frota multipla e protocolos de seguranca.")

OK_COLOR = "#d8f3dc"
ALERT_COLOR = "#ffccd5"
NEUTRAL_COLOR = "#f1f3f5"


def build_compliance_summary(problem, solution):
    assigned_stops = [stop for route in solution.routes for stop in route.stops]
    assigned_ids = {stop.visit.visit_id for stop in assigned_stops}
    unassigned = [visit for visit in problem.visits if visit.visit_id not in assigned_ids]
    refrigerated_required = sum(1 for stop in assigned_stops if stop.visit.requires_refrigeration)
    refrigerated_ok = sum(
        1 for route in solution.routes for stop in route.stops if stop.visit.requires_refrigeration and route.vehicle.supports_refrigeration
    )
    secure_required = sum(1 for stop in assigned_stops if stop.visit.requires_secure_protocol)
    secure_ok = sum(
        1 for route in solution.routes for stop in route.stops if stop.visit.requires_secure_protocol and route.vehicle.supports_secure_protocol
    )
    hospital_required = sum(1 for stop in assigned_stops if stop.visit.location_type == LocationType.HOSPITAL)
    hospital_in_window = sum(
        1
        for stop in assigned_stops
        if stop.visit.location_type == LocationType.HOSPITAL
        and stop.visit.earliest_start <= stop.service_start_minute <= stop.visit.latest_start
    )
    home_required = sum(1 for stop in assigned_stops if stop.visit.location_type == LocationType.HOME)
    home_in_safe_window = sum(
        1
        for stop in assigned_stops
        if stop.visit.location_type == LocationType.HOME
        and problem.depot.safe_home_start <= stop.service_start_minute <= problem.depot.safe_home_end
    )
    urgent_required = sum(1 for stop in assigned_stops if stop.visit.max_transport_minutes is not None)
    urgent_transport_ok = sum(
        1
        for stop in assigned_stops
        if stop.visit.max_transport_minutes is not None and stop.arrival_minute - problem.depot.start_minute <= stop.visit.max_transport_minutes
    )
    return {
        "assigned": assigned_stops,
        "unassigned": unassigned,
        "refrigerated_required": refrigerated_required,
        "refrigerated_ok": refrigerated_ok,
        "secure_required": secure_required,
        "secure_ok": secure_ok,
        "hospital_required": hospital_required,
        "hospital_in_window": hospital_in_window,
        "home_required": home_required,
        "home_in_safe_window": home_in_safe_window,
        "urgent_required": urgent_required,
        "urgent_transport_ok": urgent_transport_ok,
    }


def build_vehicle_rows(problem, solution):
    rows = []
    for route in solution.routes:
        used_units = sum(stop.visit.demand_units for stop in route.stops)
        distance_ok = route.total_distance_km <= route.vehicle.max_distance_km
        stops_ok = len(route.stops) <= route.vehicle.max_stops
        capacity_ok = used_units <= route.vehicle.max_supply_units
        rows.append(
            {
                "Veiculo": route.vehicle.vehicle_id,
                "Tipo": route.vehicle.vehicle_type.value,
                "Paradas": len(route.stops),
                "Max paradas": route.vehicle.max_stops,
                "Carga usada": used_units,
                "Capacidade": route.vehicle.max_supply_units,
                "Distancia km": route.total_distance_km,
                "Max km": route.vehicle.max_distance_km,
                "Custo/km": route.vehicle.cost_per_km,
                "Velocidade": route.vehicle.speed_kmh,
                "Frio": "Sim" if route.vehicle.supports_refrigeration else "Nao",
                "Sigilo": "Sim" if route.vehicle.supports_secure_protocol else "Nao",
                "Status distancia": "OK" if distance_ok else "Violacao",
                "Status paradas": "OK" if stops_ok else "Violacao",
                "Status carga": "OK" if capacity_ok else "Violacao",
            }
        )
    return rows


def build_stop_rows(problem, solution):
    rows = []
    for route in solution.routes:
        for stop in route.stops:
            home_ok = stop.visit.location_type != LocationType.HOME or (
                problem.depot.safe_home_start <= stop.service_start_minute <= problem.depot.safe_home_end
            )
            hospital_ok = stop.visit.location_type != LocationType.HOSPITAL or (
                stop.visit.earliest_start <= stop.service_start_minute <= stop.visit.latest_start
            )
            urgent_ok = (
                stop.visit.max_transport_minutes is None
                or (stop.arrival_minute - problem.depot.start_minute) <= stop.visit.max_transport_minutes
            )
            rows.append(
                {
                    "Veiculo": route.vehicle.vehicle_id,
                    "Visita": stop.visit.visit_id,
                    "Paciente": stop.visit.patient_name,
                    "Servico": SERVICE_LABELS[stop.visit.service_type],
                    "Local": stop.visit.location_type.value,
                    "Inicio": minute_to_hhmm(stop.service_start_minute),
                    "Janela": f"{minute_to_hhmm(stop.visit.earliest_start)}-{minute_to_hhmm(stop.visit.latest_start)}",
                    "Carga": stop.visit.demand_units,
                    "Frio": "Sim" if stop.visit.requires_refrigeration else "Nao",
                    "Sigilo": "Sim" if stop.visit.requires_secure_protocol else "Nao",
                    "Prazo transporte": str(stop.visit.max_transport_minutes) if stop.visit.max_transport_minutes is not None else "-",
                    "Status hospital": "OK" if hospital_ok else "Violacao",
                    "Status urgente": "OK" if urgent_ok else "Violacao",
                    "Status janela": "OK" if home_ok and hospital_ok else "Violacao",
                }
            )
    return rows


def style_status_table(dataframe):
    def color_value(value):
        if value == "OK":
            return f"background-color: {OK_COLOR}"
        if value == "Violacao":
            return f"background-color: {ALERT_COLOR}"
        return ""

    def color_service(value):
        reverse_labels = {label: service_type for service_type, label in SERVICE_LABELS.items()}
        service_type = reverse_labels.get(value)
        if service_type is None:
            return ""
        return f"background-color: {SERVICE_COLORS[service_type]}; color: white;"

    status_columns = [column for column in dataframe.columns if column.startswith("Status")]
    styled = dataframe.style.map(color_value, subset=status_columns)
    if "Servico" in dataframe.columns:
        styled = styled.map(color_service, subset=["Servico"])
    return styled


def status_card(label, ok_value, required_value):
    is_ok = ok_value == required_value
    color = OK_COLOR if is_ok else ALERT_COLOR
    st.markdown(
        f"""
        <div style="background:{color};padding:0.9rem 1rem;border-radius:0.75rem;border:1px solid rgba(0,0,0,0.08);">
            <div style="font-size:0.85rem;opacity:0.8;">{label}</div>
            <div style="font-size:1.5rem;font-weight:700;">{ok_value}/{required_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_service_badges(selected_services):
    if not selected_services:
        st.caption("Nenhum tipo de atendimento selecionado.")
        return
    html = []
    for label in selected_services:
        service_type = next((key for key, value in SERVICE_LABELS.items() if value == label), None)
        color = SERVICE_COLORS.get(service_type, "#999999")
        html.append(
            f"<span style='display:inline-block;background:{color};color:white;padding:0.25rem 0.55rem;"
            f"border-radius:999px;margin:0 0.35rem 0.35rem 0;font-size:0.82rem;'>{label}</span>"
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def filter_stop_rows(rows, selected_vehicles, selected_services, selected_locations):
    filtered = []
    for row in rows:
        if row["Veiculo"] not in selected_vehicles:
            continue
        if row["Servico"] not in selected_services:
            continue
        if row["Local"] not in selected_locations:
            continue
        filtered.append(row)
    return filtered


def hhmm_label(total_minutes):
    return minute_to_hhmm(int(total_minutes))


def minutes_to_time(total_minutes):
    total_minutes = int(total_minutes) % (24 * 60)
    return time(hour=total_minutes // 60, minute=total_minutes % 60)


def time_to_minutes(value):
    return (value.hour * 60) + value.minute

with st.sidebar:
    with st.expander("Cenario", expanded=True):
        seed = st.number_input("Seed", min_value=1, max_value=9999, value=42)

    with st.expander("Algoritmo", expanded=True):
        population_size = st.slider("Populacao", min_value=20, max_value=160, value=80, step=10)
        generations = st.slider("Geracoes", min_value=20, max_value=250, value=120, step=10)
        mutation_rate = st.slider("Mutacao", min_value=0.05, max_value=0.5, value=0.22, step=0.01)
        specialized_vehicle_bias = st.slider(
            "Preferencia por uso da frota especializada",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="0.0 preserva a ordem original dos veiculos. Valores maiores priorizam veiculos mais especializados.",
        )

    with st.expander("Janelas", expanded=False):
        base_start = st.time_input("Inicio da operacao", value=minutes_to_time(7 * 60), step=900)
        safe_home_start = st.time_input("Inicio seguro domiciliar", value=minutes_to_time(7 * 60), step=900)
        safe_home_end = st.time_input("Fim seguro domiciliar", value=minutes_to_time(19 * 60), step=900)

base_problem = build_sample_problem()
visit_overrides = {}
with st.sidebar:
    with st.expander("Visitas", expanded=False):
        for visit in base_problem.visits:
            with st.expander(f"{visit.visit_id} - {visit.patient_name}", expanded=False):
                service_type = st.selectbox(
                    f"{visit.visit_id} tipo de atendimento",
                    options=list(ServiceType),
                    index=list(ServiceType).index(visit.service_type),
                    format_func=lambda item: SERVICE_LABELS[item],
                    key=f"{visit.visit_id}_service_type",
                )
                location_type = st.selectbox(
                    f"{visit.visit_id} local",
                    options=list(LocationType),
                    index=list(LocationType).index(visit.location_type),
                    format_func=lambda item: item.value,
                    key=f"{visit.visit_id}_location_type",
                )
                earliest_start = st.time_input(
                    f"{visit.visit_id} inicio janela",
                    value=minutes_to_time(visit.earliest_start),
                    step=900,
                    key=f"{visit.visit_id}_earliest",
                )
                latest_start = st.time_input(
                    f"{visit.visit_id} fim janela",
                    value=minutes_to_time(visit.latest_start),
                    step=900,
                    key=f"{visit.visit_id}_latest",
                )
                service_minutes = st.slider(
                    f"{visit.visit_id} tempo de servico",
                    min_value=5,
                    max_value=120,
                    value=visit.service_minutes,
                    step=5,
                )
                demand_units = st.slider(
                    f"{visit.visit_id} quantidade de suprimentos",
                    min_value=1,
                    max_value=10,
                    value=visit.demand_units,
                    step=1,
                )
                max_transport_default = visit.max_transport_minutes if visit.max_transport_minutes is not None else 0
                max_transport_minutes = st.slider(
                    f"{visit.visit_id} prazo transporte",
                    min_value=0,
                    max_value=180,
                    value=max_transport_default,
                    step=5,
                )
                requires_refrigeration = st.checkbox(
                    f"{visit.visit_id} exige cadeia fria",
                    value=visit.requires_refrigeration,
                    key=f"{visit.visit_id}_cold",
                )
                requires_secure_protocol = st.checkbox(
                    f"{visit.visit_id} exige protocolo sigiloso",
                    value=visit.requires_secure_protocol,
                    key=f"{visit.visit_id}_secure",
                )
                visit_overrides[visit.visit_id] = {
                    "service_type": service_type,
                    "location_type": location_type,
                    "earliest_start": time_to_minutes(earliest_start),
                    "latest_start": time_to_minutes(latest_start),
                    "service_minutes": service_minutes,
                    "demand_units": demand_units,
                    "max_transport_minutes": max_transport_minutes or None,
                    "requires_refrigeration": requires_refrigeration,
                    "requires_secure_protocol": requires_secure_protocol,
                }

depot_overrides = {
    "start_minute": time_to_minutes(base_start),
    "safe_home_start": time_to_minutes(safe_home_start),
    "safe_home_end": time_to_minutes(safe_home_end),
}
problem = build_sample_problem(depot_overrides=depot_overrides, visit_overrides=visit_overrides)
solver = GeneticVRPSolver(
    problem,
    SolverConfig(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        random_seed=seed,
        specialized_vehicle_bias=specialized_vehicle_bias,
    ),
)
solution = solver.solve()
bundle = NarrativeGenerator(os.getenv("ROUTE_LLM_PROVIDER")).generate(problem, solution)
compliance = build_compliance_summary(problem, solution)
all_stop_rows = build_stop_rows(problem, solution)
vehicle_options = [vehicle.vehicle_id for vehicle in problem.vehicles]
service_options = list(SERVICE_LABELS.values())
location_options = sorted({row["Local"] for row in all_stop_rows}) or [location.value for location in LocationType]

with st.sidebar:
    with st.expander("Filtros", expanded=False):
        selected_vehicles = st.multiselect("Veiculos", options=vehicle_options, default=vehicle_options)
        selected_services = st.multiselect("Tipos de atendimento", options=service_options, default=service_options)
        selected_locations = st.multiselect("Locais", options=location_options, default=location_options)

filtered_stop_rows = filter_stop_rows(all_stop_rows, selected_vehicles, selected_services, selected_locations)
filtered_visit_ids = {row["Visita"] for row in filtered_stop_rows}
filtered_vehicle_rows = [row for row in build_vehicle_rows(problem, solution) if row["Veiculo"] in selected_vehicles]

left, right = st.columns([1.2, 1.0])

with left:
    st.subheader("Mapa operacional")
    render_service_badges(selected_services)
    st.pyplot(build_route_figure(problem, solution, filtered_visit_ids), clear_figure=True)

with right:
    st.subheader("Indicadores")
    metric_a, metric_b = st.columns(2)
    with metric_a:
        st.metric("Fitness", f"{solution.evaluation.fitness:.2f}")
        st.metric("Distancia total", f"{solution.evaluation.total_distance_km:.1f} km")
        st.metric("Visitas fora da janela", solution.evaluation.late_visits)
        st.metric("Sigilo atendido", compliance["secure_ok"])
    with metric_b:
        st.metric("Custo estimado", f"R$ {solution.evaluation.total_cost:.2f}")
        st.metric("Visitas nao alocadas", solution.evaluation.unassigned_visits)
        st.metric("Cadeia fria atendida", compliance["refrigerated_ok"])
        st.metric("Janela hospitalar OK", compliance["hospital_in_window"])

    st.subheader("Rotas")
    for route in solution.routes:
        if route.vehicle.vehicle_id not in selected_vehicles:
            continue
        st.markdown(f"**{route.vehicle.label} ({route.vehicle.vehicle_id})**")
        route_stops = [
            stop
            for stop in route.stops
            if stop.visit.visit_id in filtered_visit_ids
            and SERVICE_LABELS[stop.visit.service_type] in selected_services
            and stop.visit.location_type.value in selected_locations
        ]
        if not route_stops:
            st.write("Sem visitas alocadas.")
            continue
        for stop in route_stops:
            color = SERVICE_COLORS[stop.visit.service_type]
            st.markdown(
                f"<span style='display:inline-block;background:{color};color:white;padding:0.18rem 0.45rem;"
                f"border-radius:999px;margin-right:0.45rem;'>{SERVICE_LABELS[stop.visit.service_type]}</span>"
                f"{stop.visit.visit_id} | inicio {stop.service_start_minute // 60:02d}:{stop.service_start_minute % 60:02d}",
                unsafe_allow_html=True,
            )

st.subheader("Conformidade das restricoes")
st.caption(
    f"Base {hhmm_label(problem.depot.start_minute)} | janela segura domiciliar "
    f"{hhmm_label(problem.depot.safe_home_start)}-{hhmm_label(problem.depot.safe_home_end)}"
)
rule_a, rule_b, rule_c = st.columns(3)
with rule_a:
    status_card("Janela domiciliar segura", compliance["home_in_safe_window"], compliance["home_required"])
    status_card("Prazo transporte urgente", compliance["urgent_transport_ok"], compliance["urgent_required"])
with rule_b:
    status_card("Cadeia fria", compliance["refrigerated_ok"], compliance["refrigerated_required"])
    status_card("Protocolos de sigilo", compliance["secure_ok"], compliance["secure_required"])
with rule_c:
    status_card("Janela hospitalar", compliance["hospital_in_window"], compliance["hospital_required"])
    status_card("Alocacao de visitas", len(compliance["assigned"]), len(problem.visits))

st.subheader("Uso da frota e capacidade")
st.dataframe(style_status_table(pd.DataFrame(filtered_vehicle_rows)), width="stretch", hide_index=True)

st.subheader("Tabela operacional das paradas")
st.dataframe(style_status_table(pd.DataFrame(filtered_stop_rows)), width="stretch", hide_index=True)

if compliance["unassigned"]:
    st.subheader("Visitas nao alocadas")
    st.dataframe(
        [
            {
                "Visita": visit.visit_id,
                "Servico": SERVICE_LABELS[visit.service_type],
                "Local": visit.location_type.value,
                "Janela": f"{minute_to_hhmm(visit.earliest_start)}-{minute_to_hhmm(visit.latest_start)}",
            }
            for visit in compliance["unassigned"]
            if SERVICE_LABELS[visit.service_type] in selected_services and visit.location_type.value in selected_locations
        ],
        width="stretch",
        hide_index=True,
    )

if solution.evaluation.notes:
    st.subheader("Observacoes do solver")
    for note in solution.evaluation.notes:
        st.write(f"- {note}")

st.subheader("Manual operacional")
st.text(bundle.operations_manual)

st.subheader("Roteiro detalhado")
st.text(bundle.visit_script)

st.subheader("Perguntas em linguagem natural")
for example in bundle.qa_examples:
    st.write(f"- {example}")
