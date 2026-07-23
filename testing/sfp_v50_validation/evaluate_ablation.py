#!/usr/bin/env python3
"""Summarize staged ablation evidence and enforce per-stage promotion rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from common import write_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(payload)
    return records


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def validate_matrix(matrix: dict[str, Any]) -> None:
    stages = sorted(matrix["stages"], key=lambda stage: stage["order"])
    if [stage["order"] for stage in stages] != list(range(len(stages))):
        raise ValueError("Ablation stage order must be contiguous from zero")
    final = stages[-1]
    if final["id"] != "frozen_v50_full" or final["trials"] != 300:
        raise ValueError("Final ablation stage must be frozen_v50_full with 300 trials")
    expected_final = {
        "relative_plug_port_pose": True,
        "fixed_bias": False,
        "visual_contrast_rescue": True,
        "persistent_axial_seating": True,
        "lift_fresh_reperception": True,
        "recovery_visual_reentry": True,
    }
    if final["features"] != expected_final:
        raise ValueError("Frozen v50 feature ordering/invariants changed")


def summarize_stage(stage: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    correct_times = [
        float(record["correct_event_elapsed_s"])
        for record in records
        if record.get("correct_event_within_limit")
        and record.get("correct_event_elapsed_s") is not None
    ]
    correct = len(correct_times)
    summary = {
        "stage": stage["id"],
        "trials": len(records),
        "correct_events_within_limit": correct,
        "correct_event_rate": correct / len(records) if records else 0.0,
        "p95_correct_event_s": _percentile(correct_times, 0.95),
        "wrong_port_events": sum(bool(record.get("wrong_port_event")) for record in records),
        "offlimit_trials": sum(int(record.get("offlimit_count", 0)) > 0 for record in records),
        "force_penalty_trials": sum(bool(record.get("force_penalty")) for record in records),
    }
    promotion = stage["promotion"]
    failures = []
    comparisons = (
        ("trials", "min_trials", lambda actual, limit: actual >= limit),
        ("correct_event_rate", "min_correct_event_rate", lambda actual, limit: actual >= limit),
        ("p95_correct_event_s", "max_p95_correct_event_s", lambda actual, limit: actual is not None and actual <= limit),
        ("wrong_port_events", "max_wrong_port_events", lambda actual, limit: actual <= limit),
        ("offlimit_trials", "max_offlimit_trials", lambda actual, limit: actual <= limit),
        ("force_penalty_trials", "max_force_penalty_trials", lambda actual, limit: actual <= limit),
    )
    for metric, rule, comparator in comparisons:
        if rule in promotion and not comparator(summary[metric], promotion[rule]):
            failures.append(
                f"{metric}={summary[metric]!r} violates {rule}={promotion[rule]!r}"
            )
    summary["promotion_pass"] = not failures
    summary["promotion_failures"] = failures
    return summary


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=here / "ablation_matrix.yaml")
    parser.add_argument(
        "--results",
        action="append",
        default=[],
        metavar="STAGE=JSONL",
        help="Repeat for each completed stage",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    validate_matrix(matrix)
    stage_by_id = {stage["id"]: stage for stage in matrix["stages"]}
    summaries = []
    for value in args.results:
        if "=" not in value:
            parser.error(f"Expected STAGE=JSONL, got {value!r}")
        stage_id, raw_path = value.split("=", 1)
        if stage_id not in stage_by_id:
            parser.error(f"Unknown stage {stage_id!r}")
        summaries.append(summarize_stage(stage_by_id[stage_id], load_jsonl(Path(raw_path))))
    report = {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "stages": summaries,
        "all_completed_stages_pass": all(item["promotion_pass"] for item in summaries),
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["all_completed_stages_pass"] else 1)


if __name__ == "__main__":
    main()
