from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ServiceType(str, Enum):
    OBSTETRIC_EMERGENCY = "obstetric_emergency"
    DOMESTIC_VIOLENCE = "domestic_violence"
    HORMONAL_MEDICATION = "hormonal_medication"
    POSTPARTUM = "postpartum"
    ONCOLOGY_SUPPORT = "oncology_support"


class LocationType(str, Enum):
    HOME = "home"
    HOSPITAL = "hospital"
    CLINIC = "clinic"


class VehicleType(str, Enum):
    MOTORBIKE = "motorbike"
    VAN = "van"
    CAR = "car"
    DRONE = "drone"


SERVICE_LABELS = {
    ServiceType.OBSTETRIC_EMERGENCY: "Emergencia obstetrica",
    ServiceType.DOMESTIC_VIOLENCE: "Violencia domestica",
    ServiceType.HORMONAL_MEDICATION: "Medicacao hormonal",
    ServiceType.POSTPARTUM: "Pos-parto",
    ServiceType.ONCOLOGY_SUPPORT: "Apoio oncologico",
}


PRIORITY_WEIGHT = {
    ServiceType.OBSTETRIC_EMERGENCY: 7,
    ServiceType.DOMESTIC_VIOLENCE: 6,
    ServiceType.POSTPARTUM: 4,
    ServiceType.HORMONAL_MEDICATION: 3,
    ServiceType.ONCOLOGY_SUPPORT: 2,
}


@dataclass(frozen=True)
class Visit:
    visit_id: str
    patient_name: str
    x: float
    y: float
    service_type: ServiceType
    demand_units: int
    service_minutes: int
    earliest_start: int
    latest_start: int
    location_type: LocationType = LocationType.HOME
    requires_refrigeration: bool = False
    requires_secure_protocol: bool = False
    max_transport_minutes: int | None = None
    notes: str = ""

    @property
    def priority(self) -> int:
        return PRIORITY_WEIGHT[self.service_type]


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: str
    label: str
    vehicle_type: VehicleType
    max_distance_km: float
    max_stops: int
    max_supply_units: int
    speed_kmh: float
    cost_per_km: float
    allowed_service_types: tuple[ServiceType, ...] | None = None
    supports_refrigeration: bool = False
    supports_secure_protocol: bool = False


@dataclass(frozen=True)
class Depot:
    x: float
    y: float
    start_minute: int = 480
    safe_home_start: int = 420
    safe_home_end: int = 1140


@dataclass(frozen=True)
class ProblemInstance:
    name: str
    depot: Depot
    vehicles: list[Vehicle]
    visits: list[Visit]

    def visit_by_id(self) -> dict[str, Visit]:
        return {visit.visit_id: visit for visit in self.visits}


@dataclass(frozen=True)
class RouteStop:
    visit: Visit
    arrival_minute: int
    service_start_minute: int
    departure_minute: int
    distance_from_previous: float
    vehicle_id: str


@dataclass(frozen=True)
class Route:
    vehicle: Vehicle
    stops: list[RouteStop]
    total_distance_km: float
    total_cost: float


@dataclass(frozen=True)
class Evaluation:
    total_distance_km: float
    total_cost: float
    total_penalty: float
    fitness: float
    late_visits: int
    unassigned_visits: int
    delayed_priority_penalty: float
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Solution:
    chromosome: list[str]
    routes: list[Route]
    evaluation: Evaluation
