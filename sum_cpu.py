#!/usr/bin/env python3

"""Keep only top-level cpu_op events from a Kineto trace and sum their durations.

The input trace is expected to be a Perfetto-format Kineto JSON file with a
top-level ``traceEvents`` array. The script preserves metadata events (``ph`` ==
``M``) and keeps only those duration events with ``cat`` == ``cpu_op`` that are
not strictly contained inside another cpu_op interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any


def _event_start(event: dict[str, Any]) -> float:
	return float(event.get("ts", 0.0))


def _event_end(event: dict[str, Any]) -> float:
	return _event_start(event) + float(event.get("dur", 0.0))


def _is_metadata(event: dict[str, Any]) -> bool:
	return event.get("ph") == "M"


def _is_cpu_op(event: dict[str, Any]) -> bool:
	return event.get("cat") == "cpu_op" and event.get("ph") == "X"


def filter_top_level_cpu_ops(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
	cpu_ops_by_tid: dict[Any, list[tuple[int, dict[str, Any], float, float]]] = defaultdict(list)

	for index, event in enumerate(events):
		if _is_cpu_op(event):
			start = _event_start(event)
			end = _event_end(event)
			cpu_ops_by_tid[event.get("tid")].append((index, event, start, end))

	top_level_indices: set[int] = set()

	for cpu_ops in cpu_ops_by_tid.values():
		cpu_ops.sort(key=lambda item: (item[2], -item[3], item[0]))
		stack: list[tuple[float, float, int]] = []

		for index, _, start, end in cpu_ops:
			while stack and start >= stack[-1][1]:
				stack.pop()

			if stack and end <= stack[-1][1]:
				continue

			top_level_indices.add(index)
			stack.append((start, end, index))

	filtered_events = [
		event
		for index, event in enumerate(events)
		if _is_metadata(event) or index in top_level_indices
	]
	total_duration = sum((float(events[index].get("dur", 0.0)) * 1000) for index in top_level_indices)
	return filtered_events, total_duration


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Filter a Kineto trace down to top-level cpu_op events and metadata."
	)
	parser.add_argument("input", type=Path, help="Path to the Kineto trace JSON file")
	parser.add_argument(
		"output",
		type=Path,
		nargs="?",
		help="Path to write the filtered trace JSON file. Defaults to <input>.cpu_top.json",
	)
	args = parser.parse_args()

	output_path = args.output or args.input.with_name(f"{args.input.stem}.cpu_top.json")

	with args.input.open("r", encoding="utf-8") as handle:
		trace = json.load(handle)

	events = trace.get("traceEvents", [])
	if not isinstance(events, list):
		raise ValueError("traceEvents must be a list")

	filtered_events, total_duration = filter_top_level_cpu_ops(events)
	trace["traceEvents"] = filtered_events

	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(trace, handle, indent=2)
		handle.write("\n")

	print(f"top_level_cpu_op_count={sum(1 for event in filtered_events if _is_cpu_op(event))}")
	print(f"top_level_cpu_op_duration={total_duration}")
	print(f"wrote={output_path}")


if __name__ == "__main__":
	main()
