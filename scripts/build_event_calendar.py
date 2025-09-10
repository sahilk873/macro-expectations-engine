#!/usr/bin/env python3
"""Phase 2: Build the macro event calendar."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.events.calendar import build_event_calendar, save_event_calendar

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("Building macro event calendar...")
    df = build_event_calendar()
    path = save_event_calendar(df)
    logger.info("Saved %d events to %s", len(df), path)
    logger.info("Event types: %s", df["event_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
