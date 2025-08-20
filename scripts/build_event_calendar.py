#!/usr/bin/env python3
"""Phase 2: Build the macro event calendar."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.events.calendar import build_event_calendar, save_event_calendar


def main():
    print("Building macro event calendar...")
    df = build_event_calendar()
    path = save_event_calendar(df)
    print(f"Saved {len(df)} events to {path}")
    print(f"Event types: {df['event_type'].value_counts().to_dict()}")
    print("Done.")


if __name__ == "__main__":
    main()
