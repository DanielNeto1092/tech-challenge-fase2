from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostico_saude_mulher.experiments.runner import executar_pipeline_completo, save_experiment_payload
from diagnostico_saude_mulher.visualization.plots import (
    plot_convergence,
    plot_experiment_comparison,
    plot_experiment_summary,
    plot_generation_metrics,
    plot_hyperparameter_behavior,
)


def main() -> None:
    """Executa os experimentos genéticos e gera relatórios visuais."""

    resultado = executar_pipeline_completo()
    output_dir = ROOT / "artifacts" / "graficos"
    graficos = [str(plot_experiment_comparison(resultado.baseline, resultado.experimentos_geneticos, output_dir))]
    graficos.append(str(plot_experiment_summary(resultado.experimentos_geneticos, output_dir)))
    graficos.append(str(plot_hyperparameter_behavior(resultado.experimentos_geneticos, output_dir)))
    for experimento in resultado.experimentos_geneticos:
        graficos.append(str(plot_convergence(experimento, output_dir)))
        graficos.append(str(plot_generation_metrics(experimento, output_dir)))

    payload = resultado.to_dict()
    payload["graficos"] = graficos
    report_path = save_experiment_payload(payload, ROOT / "artifacts" / "experimentos_geneticos.json")
    print(json.dumps({"relatorio": str(report_path), "graficos": graficos}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
