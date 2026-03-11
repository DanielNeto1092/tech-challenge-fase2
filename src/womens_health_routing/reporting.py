from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .domain import SERVICE_LABELS, ProblemInstance, Route, RouteStop, ServiceType, Solution


def minute_to_hhmm(value: int) -> str:
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"


@dataclass(frozen=True)
class ReportBundle:
    operations_manual: str
    visit_script: str
    qa_examples: list[str]


class NarrativeGenerator:
    def __init__(self, llm_provider: str | None = None) -> None:
        self.llm_provider = llm_provider or os.getenv("ROUTE_LLM_PROVIDER", "rule_based")

    def generate(self, problem: ProblemInstance, solution: Solution) -> ReportBundle:
        if self.llm_provider != "rule_based":
            generated = self._generate_with_http_llm(problem, solution)
            if generated is not None:
                return generated
        return ReportBundle(
            operations_manual=self._operations_manual(problem, solution),
            visit_script=self._visit_script(solution.routes),
            qa_examples=self._qa_examples(solution),
        )

    def _generate_with_http_llm(self, problem: ProblemInstance, solution: Solution) -> ReportBundle | None:
        endpoint = os.getenv("ROUTE_LLM_ENDPOINT")
        api_key = os.getenv("ROUTE_LLM_API_KEY")
        model = os.getenv("ROUTE_LLM_MODEL")
        if not endpoint or not model:
            return None

        route_context = self._route_context(solution.routes)
        manual_prompt = (
            "Voce e um assistente medico-logistico especializado em saude da mulher. "
            "Gere um manual operacional curto e objetivo para a equipe de transporte, com foco em seguranca, "
            "sigilo, prioridade clinica e cadeia fria quando aplicavel.\n\n"
            f"Cenario: {problem.name}\n{route_context}"
        )
        script_prompt = (
            "Transforme a rota otimizada em um roteiro operacional legivel para a equipe. "
            "Inclua ordem das visitas, tipo de atendimento, informacoes relevantes, estimativa de tempo e distancia.\n\n"
            f"Cenario: {problem.name}\n{route_context}"
        )
        manual = self._chat_completion(endpoint, api_key, model, manual_prompt)
        script = self._chat_completion(endpoint, api_key, model, script_prompt)
        if manual is None or script is None:
            return None
        return ReportBundle(
            operations_manual=manual,
            visit_script=script,
            qa_examples=self._qa_examples(solution),
        )

    def _chat_completion(self, endpoint: str, api_key: str | None, model: str, prompt: str) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Seja conciso, seguro e sensivel ao contexto de saude da mulher."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

        choices = data.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        return content.strip() if isinstance(content, str) and content.strip() else None

    def _operations_manual(self, problem: ProblemInstance, solution: Solution) -> str:
        sections = [
            f"Manual operacional para {problem.name}.",
            "Priorizar emergencias obstetricas e ocorrencias com risco social em qualquer replanejamento.",
            "Confirmar sigilo em visitas de violencia domestica e evitar contatos desnecessarios antes da chegada.",
            "Garantir cadeia fria em medicacoes hormonais transportadas pela van refrigerada.",
            "Respeitar janelas seguras de atendimento pos-parto e validar acolhimento local antes da entrega.",
        ]
        for route in solution.routes:
            if not route.stops:
                continue
            sections.append(f"Veiculo {route.vehicle.label} ({route.vehicle.vehicle_id}):")
            for stop in route.stops:
                sections.append(self._stop_instruction(stop))
        return "\n".join(sections)

    def _visit_script(self, routes: list[Route]) -> str:
        lines = ["Roteiro detalhado de visitas do dia."]
        for route in routes:
            if not route.stops:
                continue
            lines.append(
                f"{route.vehicle.label} | distancia {route.total_distance_km:.1f} km | custo estimado R$ {route.total_cost:.2f}"
            )
            for index, stop in enumerate(route.stops, start=1):
                lines.append(
                    f"{index}. {stop.visit.visit_id} - {stop.visit.patient_name} | {SERVICE_LABELS[stop.visit.service_type]} | "
                    f"chegada {minute_to_hhmm(stop.arrival_minute)} | inicio {minute_to_hhmm(stop.service_start_minute)} | "
                    f"janela {minute_to_hhmm(stop.visit.earliest_start)}-{minute_to_hhmm(stop.visit.latest_start)} | "
                    f"trecho {stop.distance_from_previous:.1f} km"
                )
        return "\n".join(lines)

    def _qa_examples(self, solution: Solution) -> list[str]:
        emergency_count = sum(
            1 for route in solution.routes for stop in route.stops if stop.visit.service_type == ServiceType.OBSTETRIC_EMERGENCY
        )
        next_priority = self._next_priority_stop(solution.routes)
        return [
            f"Qual o proximo atendimento prioritario? {next_priority}",
            f"Quantas paradas de emergencia temos hoje? {emergency_count}",
            f"Ha visitas fora da janela? {solution.evaluation.late_visits}",
        ]

    def _next_priority_stop(self, routes: list[Route]) -> str:
        ranked: list[RouteStop] = [stop for route in routes for stop in route.stops]
        if not ranked:
            return "Nenhuma visita planejada."
        ranked.sort(key=lambda stop: (-stop.visit.priority, stop.service_start_minute))
        top = ranked[0]
        return f"{top.visit.visit_id} ({SERVICE_LABELS[top.visit.service_type]}) as {minute_to_hhmm(top.service_start_minute)}"

    def _route_context(self, routes: list[Route]) -> str:
        lines = []
        for route in routes:
            if not route.stops:
                continue
            lines.append(f"Veiculo {route.vehicle.label} ({route.vehicle.vehicle_id})")
            for stop in route.stops:
                lines.append(
                    f"- {stop.visit.visit_id} | {SERVICE_LABELS[stop.visit.service_type]} | "
                    f"{stop.visit.patient_name} | chegada {minute_to_hhmm(stop.arrival_minute)} | "
                    f"inicio {minute_to_hhmm(stop.service_start_minute)} | trecho {stop.distance_from_previous:.1f} km | "
                    f"obs: {stop.visit.notes or 'sem observacoes'}"
                )
        return "\n".join(lines)

    def _stop_instruction(self, stop: RouteStop) -> str:
        prefix = (
            f"- {stop.visit.visit_id} {SERVICE_LABELS[stop.visit.service_type]} "
            f"para {stop.visit.patient_name} as {minute_to_hhmm(stop.service_start_minute)}:"
        )
        safety = []
        if stop.visit.requires_secure_protocol:
            safety.append("manter protocolo sigiloso")
        if stop.visit.requires_refrigeration:
            safety.append("preservar temperatura controlada")
        if not safety:
            safety.append("seguir protocolo assistencial padrao")
        if stop.visit.notes:
            safety.append(stop.visit.notes)
        return f"{prefix} {'; '.join(safety)}."
