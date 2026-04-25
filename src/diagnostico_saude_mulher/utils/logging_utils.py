from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(level: str, output_dir: Path | None = None) -> None:
    """Configura logging padronizado da aplicação."""

    root_logger = logging.getLogger()
    if getattr(root_logger, "_diagnostico_logging_configured", False):
        return

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [stream_handler]
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            output_dir / "aplicacao.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger.setLevel(resolved_level)
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger._diagnostico_logging_configured = True  # type: ignore[attr-defined]
