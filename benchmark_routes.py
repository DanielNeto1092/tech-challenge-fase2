from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from womens_health_routing.benchmarking import run_benchmark
from womens_health_routing.ga_solver import SolverConfig
from womens_health_routing.sample_data import build_sample_problem


if __name__ == "__main__":
    result = run_benchmark(build_sample_problem(), SolverConfig())
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
