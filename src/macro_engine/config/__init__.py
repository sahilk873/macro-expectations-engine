"""Configuration and settings."""

import logging

from macro_engine.config.settings import EngineConfig, get_settings


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the macro engine."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-30s | %(levelname)-6s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


__all__ = ["EngineConfig", "get_settings", "setup_logging"]
