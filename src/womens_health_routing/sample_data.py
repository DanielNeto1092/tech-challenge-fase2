from __future__ import annotations

from dataclasses import replace
import random

from .domain import Depot, LocationType, ProblemInstance, ServiceType, Vehicle, VehicleType, Visit


def build_sample_problem(
    depot_overrides: dict | None = None,
    visit_overrides: dict[str, dict] | None = None,
) -> ProblemInstance:
    depot = Depot(x=0.0, y=0.0, start_minute=7 * 60)
    if depot_overrides:
        depot = replace(depot, **depot_overrides)
    vehicles = [
        Vehicle(
            vehicle_id="MOTO-1",
            label="Moto de resposta rapida",
            vehicle_type=VehicleType.MOTORBIKE,
            max_distance_km=80.0,
            max_stops=4,
            max_supply_units=8,
            speed_kmh=48.0,
            cost_per_km=1.1,
            allowed_service_types=(
                ServiceType.OBSTETRIC_EMERGENCY,
                ServiceType.DOMESTIC_VIOLENCE,
                ServiceType.ONCOLOGY_SUPPORT,
            ),
            supports_secure_protocol=True,
        ),
        Vehicle(
            vehicle_id="VAN-1",
            label="Van refrigerada",
            vehicle_type=VehicleType.VAN,
            max_distance_km=140.0,
            max_stops=6,
            max_supply_units=22,
            speed_kmh=38.0,
            cost_per_km=2.0,
            allowed_service_types=tuple(ServiceType),
            supports_refrigeration=True,
            supports_secure_protocol=True,
        ),
        Vehicle(
            vehicle_id="CAR-1",
            label="Carro de apoio clinico",
            vehicle_type=VehicleType.CAR,
            max_distance_km=110.0,
            max_stops=5,
            max_supply_units=12,
            speed_kmh=42.0,
            cost_per_km=1.5,
            allowed_service_types=(
                ServiceType.POSTPARTUM,
                ServiceType.ONCOLOGY_SUPPORT,
                ServiceType.DOMESTIC_VIOLENCE,
            ),
        ),
    ]
    visits = [
        Visit(
            visit_id="V01",
            patient_name="Paciente A",
            x=4.0,
            y=8.0,
            service_type=ServiceType.OBSTETRIC_EMERGENCY,
            demand_units=2,
            service_minutes=25,
            earliest_start=7 * 60,
            latest_start=8 * 60,
            location_type=LocationType.HOME,
            notes="Dor intensa e sangramento. Acionar protocolo maternidade de referencia.",
        ),
        Visit(
            visit_id="V02",
            patient_name="Paciente B",
            x=7.0,
            y=2.0,
            service_type=ServiceType.DOMESTIC_VIOLENCE,
            demand_units=1,
            service_minutes=20,
            earliest_start=8 * 60,
            latest_start=10 * 60,
            location_type=LocationType.HOME,
            requires_secure_protocol=True,
            notes="Evitar contato telefonico antes da chegada.",
        ),
        Visit(
            visit_id="V03",
            patient_name="Paciente C",
            x=12.0,
            y=4.0,
            service_type=ServiceType.HORMONAL_MEDICATION,
            demand_units=4,
            service_minutes=15,
            earliest_start=8 * 60,
            latest_start=11 * 60,
            location_type=LocationType.HOSPITAL,
            requires_refrigeration=True,
            max_transport_minutes=45,
            notes="Medicacao deve permanecer refrigerada ate a entrega.",
        ),
        Visit(
            visit_id="V04",
            patient_name="Paciente D",
            x=2.0,
            y=14.0,
            service_type=ServiceType.POSTPARTUM,
            demand_units=2,
            service_minutes=30,
            earliest_start=9 * 60,
            latest_start=12 * 60,
            location_type=LocationType.HOME,
            notes="Atendimento domiciliar somente com acompanhante informado.",
        ),
        Visit(
            visit_id="V05",
            patient_name="Paciente E",
            x=14.0,
            y=11.0,
            service_type=ServiceType.OBSTETRIC_EMERGENCY,
            demand_units=1,
            service_minutes=20,
            earliest_start=7 * 60 + 30,
            latest_start=9 * 60,
            location_type=LocationType.CLINIC,
            notes="Encaminhar para centro cirurgico se houver agravamento.",
        ),
        Visit(
            visit_id="V06",
            patient_name="Paciente F",
            x=16.0,
            y=3.0,
            service_type=ServiceType.ONCOLOGY_SUPPORT,
            demand_units=3,
            service_minutes=20,
            earliest_start=10 * 60,
            latest_start=13 * 60,
            location_type=LocationType.CLINIC,
            notes="Entrega de analgesicos e orientacao para equipe da UBS.",
        ),
        Visit(
            visit_id="V07",
            patient_name="Paciente G",
            x=9.0,
            y=13.0,
            service_type=ServiceType.HORMONAL_MEDICATION,
            demand_units=3,
            service_minutes=15,
            earliest_start=9 * 60,
            latest_start=11 * 60 + 30,
            location_type=LocationType.HOSPITAL,
            requires_refrigeration=True,
            max_transport_minutes=50,
            notes="Insulina hormonal com controle estrito de temperatura.",
        ),
        Visit(
            visit_id="V08",
            patient_name="Paciente H",
            x=18.0,
            y=9.0,
            service_type=ServiceType.DOMESTIC_VIOLENCE,
            demand_units=1,
            service_minutes=20,
            earliest_start=9 * 60,
            latest_start=12 * 60,
            location_type=LocationType.HOME,
            requires_secure_protocol=True,
            notes="Acionamento sigiloso com rede de protecao local.",
        ),
        Visit(
            visit_id="V09",
            patient_name="Paciente I",
            x=6.0,
            y=17.0,
            service_type=ServiceType.POSTPARTUM,
            demand_units=2,
            service_minutes=25,
            earliest_start=10 * 60,
            latest_start=13 * 60,
            location_type=LocationType.HOME,
            notes="Janela segura confirmada com unidade de saude da familia.",
        ),
        Visit(
            visit_id="V10",
            patient_name="Paciente J",
            x=20.0,
            y=6.0,
            service_type=ServiceType.ONCOLOGY_SUPPORT,
            demand_units=2,
            service_minutes=15,
            earliest_start=8 * 60 + 30,
            latest_start=12 * 60 + 30,
            location_type=LocationType.HOSPITAL,
            notes="Kit de medicacao e acolhimento para consulta especializada.",
        ),
    ]
    if visit_overrides:
        visits = [
            replace(visit, **visit_overrides.get(visit.visit_id, {}))
            for visit in visits
        ]

    return ProblemInstance(
        name="Atendimento especializado a mulher - cenario demonstracao",
        depot=depot,
        vehicles=vehicles,
        visits=visits,
    )


def build_synthetic_problem(
    patient_count: int,
    seed: int = 42,
    depot_overrides: dict | None = None,
) -> ProblemInstance:
    rng = random.Random(seed)
    base = build_sample_problem(depot_overrides=depot_overrides)
    templates = [
        {
            "service_type": ServiceType.OBSTETRIC_EMERGENCY,
            "location_type": LocationType.CLINIC,
            "service_minutes": 25,
            "earliest_start": 7 * 60,
            "latest_start": 9 * 60,
            "demand_units": 2,
            "notes": "Prioridade clinica maxima. Validar encaminhamento imediato.",
        },
        {
            "service_type": ServiceType.DOMESTIC_VIOLENCE,
            "location_type": LocationType.HOME,
            "service_minutes": 20,
            "earliest_start": 8 * 60,
            "latest_start": 12 * 60,
            "demand_units": 1,
            "requires_secure_protocol": True,
            "notes": "Atendimento com sigilo reforcado.",
        },
        {
            "service_type": ServiceType.HORMONAL_MEDICATION,
            "location_type": LocationType.HOSPITAL,
            "service_minutes": 15,
            "earliest_start": 8 * 60,
            "latest_start": 12 * 60,
            "demand_units": 3,
            "requires_refrigeration": True,
            "max_transport_minutes": 45,
            "notes": "Medicacao com cadeia fria.",
        },
        {
            "service_type": ServiceType.POSTPARTUM,
            "location_type": LocationType.HOME,
            "service_minutes": 30,
            "earliest_start": 9 * 60,
            "latest_start": 14 * 60,
            "demand_units": 2,
            "notes": "Atendimento domiciliar pos-parto.",
        },
        {
            "service_type": ServiceType.ONCOLOGY_SUPPORT,
            "location_type": LocationType.CLINIC,
            "service_minutes": 20,
            "earliest_start": 8 * 60 + 30,
            "latest_start": 13 * 60,
            "demand_units": 2,
            "notes": "Apoio oncologico e entrega de insumos.",
        },
    ]
    visits: list[Visit] = []
    for index in range(patient_count):
        template = templates[index % len(templates)]
        earliest = template["earliest_start"] + rng.choice([0, 15, 30, 45])
        latest = max(earliest + 60, template["latest_start"] + rng.choice([0, 15, 30]))
        visits.append(
            Visit(
                visit_id=f"S{index + 1:03d}",
                patient_name=f"Paciente {index + 1}",
                x=round(rng.uniform(2.0, 24.0), 1),
                y=round(rng.uniform(2.0, 20.0), 1),
                service_type=template["service_type"],
                demand_units=template["demand_units"] + rng.choice([0, 0, 1]),
                service_minutes=template["service_minutes"],
                earliest_start=earliest,
                latest_start=latest,
                location_type=template["location_type"],
                requires_refrigeration=template.get("requires_refrigeration", False),
                requires_secure_protocol=template.get("requires_secure_protocol", False),
                max_transport_minutes=template.get("max_transport_minutes"),
                notes=template["notes"],
            )
        )
    return ProblemInstance(
        name=f"Atendimento especializado a mulher - cenario sintetico ({patient_count} pacientes)",
        depot=base.depot,
        vehicles=base.vehicles,
        visits=visits,
    )
