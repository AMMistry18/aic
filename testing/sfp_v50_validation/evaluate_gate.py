#!/usr/bin/env python3
"""Strictly evaluate the frozen 300/300 SFP insertion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import load_json, validate_content_sha256, write_json
from trial_evidence import canonical_insertion_event


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                results.append(payload)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                results.extend(payload)
            elif isinstance(payload, dict):
                results.append(payload)
            else:
                raise ValueError(f"Unsupported result JSON in {path}")
    return results


def evaluate_gate(gate: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    validate_content_sha256(gate, "frozen gate")
    if gate.get("task_scope") != "sfp_only" or gate.get("trial_count") != 300:
        raise ValueError("Gate is not the required 300-trial SFP suite")
    criteria = gate["criteria"]
    if criteria.get("required_correct_events") != 300:
        raise ValueError("Gate does not require 300 correct events")
    if criteria.get("max_correct_event_wall_seconds") != 45.0:
        raise ValueError("Gate wall-time limit is not exactly 45 seconds")
    if criteria.get("force_penalty_threshold_n") != 20.0:
        raise ValueError("Gate force threshold is not the scorer's 20 N")
    if criteria.get("force_penalty_duration_seconds") != 1.0:
        raise ValueError("Gate force duration is not the scorer's one second")

    expected_trials = {
        trial["trial_id"]: trial for trial in gate["scenario_suite"]["trials"]
    }
    if len(expected_trials) != 300:
        raise ValueError("Frozen gate does not contain 300 unique trials")
    expected_artifacts = {
        name: metadata["sha256"] for name, metadata in gate["artifacts"].items()
    }
    expected_images = {
        name: metadata["id"] for name, metadata in gate["runtime_images"].items()
    }
    seen: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    unexpected_ids: list[str] = []
    for result in results:
        trial_id = result.get("trial_id")
        if trial_id not in expected_trials:
            unexpected_ids.append(str(trial_id))
            continue
        if trial_id in seen:
            duplicate_ids.append(trial_id)
            continue
        seen[trial_id] = result

    failures: dict[str, list[str]] = {}
    correct_count = 0
    wrong_count = 0
    offlimit_trials = 0
    force_penalty_trials = 0
    official_wrong_count = 0
    official_offlimit_trials = 0
    official_force_penalty_trials = 0
    for trial_id, result in seen.items():
        expected = expected_trials[trial_id]
        reasons: list[str] = []
        if result.get("clock") != "time.monotonic_ns":
            reasons.append("non-monotonic or missing clock")
        if result.get("force_clock") != "wrench_header_stamp":
            reasons.append("force duration does not use scorer wrench timestamps")
        if result.get("force_tare") != "aic_controller/controller_state.fts_tare_offset":
            reasons.append("force evidence does not use scorer tare")
        if int(result.get("force_sample_count", 0)) <= 0:
            reasons.append("no tared force samples captured")
        if result.get("frozen_gate_sha256") != gate["content_sha256"]:
            reasons.append("frozen gate hash mismatch")
        if result.get("artifact_hashes") != expected_artifacts:
            reasons.append("artifact hashes mismatch")
        if result.get("runtime_image_ids") != expected_images:
            reasons.append("runtime Docker image ids mismatch")
        if result.get("config_sha256") != expected["config_sha256"]:
            reasons.append("scenario config hash mismatch")
        observed_expected = canonical_insertion_event(
            str(result.get("expected_insertion_event", ""))
        )
        if observed_expected != expected["expected_insertion_event"]:
            reasons.append("expected insertion namespace mismatch")

        elapsed = result.get("correct_event_elapsed_s")
        correct = bool(result.get("correct_insertion_event"))
        in_time = (
            correct
            and isinstance(elapsed, (int, float))
            and 0.0 <= float(elapsed) <= criteria["max_correct_event_wall_seconds"]
        )
        if in_time:
            correct_count += 1
        else:
            reasons.append("correct insertion event absent or later than 45 seconds")
        if result.get("wrong_port_event"):
            wrong_count += 1
            reasons.append("wrong-port insertion event")
        if int(result.get("offlimit_count", 0)) > 0:
            offlimit_trials += 1
            reasons.append("off-limit contact")
        if result.get("force_penalty"):
            force_penalty_trials += 1
            reasons.append("force penalty")
        scorer = result.get("scorer_truth", {})
        if not scorer.get("correct_full_insertion"):
            reasons.append("official scorer did not award full correct insertion")
        if scorer.get("wrong_port_penalty"):
            official_wrong_count += 1
            if not result.get("wrong_port_event"):
                wrong_count += 1
            reasons.append("official scorer reported wrong-port insertion")
        if scorer.get("offlimit_penalty"):
            official_offlimit_trials += 1
            if int(result.get("offlimit_count", 0)) <= 0:
                offlimit_trials += 1
            reasons.append("official scorer reported off-limit penalty")
        if scorer.get("force_penalty"):
            official_force_penalty_trials += 1
            if not result.get("force_penalty"):
                force_penalty_trials += 1
            reasons.append("official scorer reported force penalty")
        if reasons:
            failures[trial_id] = reasons

    missing_ids = sorted(set(expected_trials) - set(seen))
    infrastructure_failures = []
    if duplicate_ids:
        infrastructure_failures.append(f"duplicate trial ids: {sorted(set(duplicate_ids))}")
    if unexpected_ids:
        infrastructure_failures.append(f"unexpected trial ids: {sorted(set(unexpected_ids))}")
    if missing_ids:
        infrastructure_failures.append(f"missing {len(missing_ids)} trials")
    if len(results) != 300:
        infrastructure_failures.append(f"received {len(results)} results, expected exactly 300")

    passed = (
        not infrastructure_failures
        and not failures
        and correct_count == 300
        and wrong_count <= criteria["max_wrong_port_events"]
        and offlimit_trials <= criteria["max_offlimit_trials"]
        and force_penalty_trials <= criteria["max_force_penalty_trials"]
    )
    return {
        "schema_version": 1,
        "gate_id": gate["gate_id"],
        "frozen_gate_sha256": gate["content_sha256"],
        "passed": passed,
        "required_trials": 300,
        "received_results": len(results),
        "unique_expected_results": len(seen),
        "correct_events_within_45s": correct_count,
        "wrong_port_events": wrong_count,
        "offlimit_trials": offlimit_trials,
        "force_penalty_trials": force_penalty_trials,
        "official_scorer_wrong_port_events": official_wrong_count,
        "official_scorer_offlimit_trials": official_offlimit_trials,
        "official_scorer_force_penalty_trials": official_force_penalty_trials,
        "missing_trial_ids": missing_ids,
        "duplicate_trial_ids": sorted(set(duplicate_ids)),
        "unexpected_trial_ids": sorted(set(unexpected_ids)),
        "infrastructure_failures": infrastructure_failures,
        "trial_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gate = load_json(args.gate)
    report = evaluate_gate(gate, load_results(args.results))
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
