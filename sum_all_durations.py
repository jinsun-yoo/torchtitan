#!/usr/bin/env python3

"""Sum durations for all operation events in a Perfetto/Kineto trace JSON file.

This script scans ``traceEvents`` and adds every numeric ``dur`` value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum all event durations in a trace JSON file."
    )
    parser.add_argument("input", type=Path, help="Path to the trace JSON file")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")

    total_duration = 0.0
    op_count = 0

    for event in events:
        dur = event.get("dur")
        if isinstance(dur, (int, float)):
            total_duration += float(dur)
            op_count += 1

    print(f"operation_count={op_count}")
    print(f"total_duration={total_duration}")


if __name__ == "__main__":
    main()
