from __future__ import annotations

import json
from pathlib import Path

import yaml

from common import attach_content_sha256, write_json
from evaluate_ablation import summarize_stage, validate_matrix
from evaluate_gate import evaluate_gate
from freeze_gate import freeze_gate
from observe_trial import _gate_trial


VALIDATION = Path(__file__).resolve().parents[1]
MATRIX = VALIDATION / "ablation_matrix.yaml"


def _scenario_manifest(tmp_path: Path) -> Path:
    shard = tmp_path / "suite_00.yaml"
    shard.write_text("trials: {}\n")
    import hashlib

    shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
    trials = [
        {
            "trial_id": f"trial_{index:03d}",
            "scenario_seed": index,
            "cable_name": "cable_0",
            "expected_insertion_event": f"nic_card_mount_{index % 5}/sfp_port_{index % 2}",
            "config_path": shard.name,
            "config_sha256": shard_hash,
        }
        for index in range(300)
    ]
    manifest = attach_content_sha256(
        {
            "schema_version": 1,
            "suite_id": "test_suite",
            "master_seed": 42,
            "task_scope": "sfp_only",
            "trial_count": 300,
            "criteria": {
                "required_correct_events": 300,
                "max_correct_event_wall_seconds": 45.0,
                "max_wrong_port_events": 0,
                "max_offlimit_trials": 0,
                "max_force_penalty_trials": 0,
                "force_penalty_threshold_n": 20.0,
                "force_penalty_duration_seconds": 1.0,
                "clock": "time.monotonic_ns",
            },
            "shards": [{"path": shard.name, "sha256": shard_hash, "trial_ids": [item["trial_id"] for item in trials]}],
            "trials": trials,
        }
    )
    path = tmp_path / "scenario.json"
    write_json(path, manifest)
    return path


def _gate(tmp_path: Path) -> dict:
    artifacts = {}
    for name in ("controller_source", "plug_pose_model", "port_pose_model", "runtime_recipe"):
        path = tmp_path / name
        path.write_text(name)
        artifacts[name] = path
    output = tmp_path / "gate.json"
    return freeze_gate(
        scenario_manifest_path=_scenario_manifest(tmp_path),
        matrix_path=MATRIX,
        artifacts=artifacts,
        runtime_images={
            "evaluator": {"ref": "eval:test", "id": "sha256:" + "1" * 64},
            "model": {"ref": "model:test", "id": "sha256:" + "2" * 64},
        },
        output_path=output,
        gate_id="test_gate",
    )


def _passing_results(gate: dict) -> list[dict]:
    hashes = {name: metadata["sha256"] for name, metadata in gate["artifacts"].items()}
    images = {name: metadata["id"] for name, metadata in gate["runtime_images"].items()}
    return [
        {
            "trial_id": trial["trial_id"],
            "clock": "time.monotonic_ns",
            "force_clock": "wrench_header_stamp",
            "force_tare": "aic_controller/controller_state.fts_tare_offset",
            "force_sample_count": 1,
            "frozen_gate_sha256": gate["content_sha256"],
            "artifact_hashes": dict(hashes),
            "runtime_image_ids": dict(images),
            "config_sha256": trial["config_sha256"],
            "expected_insertion_event": trial["expected_insertion_event"],
            "correct_insertion_event": True,
            "correct_event_elapsed_s": 44.9,
            "wrong_port_event": False,
            "offlimit_count": 0,
            "force_penalty": False,
            "scorer_truth": {
                "correct_full_insertion": True,
                "wrong_port_penalty": False,
                "offlimit_penalty": False,
                "force_penalty": False,
            },
        }
        for trial in gate["scenario_suite"]["trials"]
    ]


def test_exact_300_results_pass(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    trial, hashes, config_hash = _gate_trial(
        gate, tmp_path / "gate.json", "trial_000"
    )
    assert trial["expected_insertion_event"] == "nic_card_mount_0/sfp_port_0"
    assert hashes == {
        name: metadata["sha256"] for name, metadata in gate["artifacts"].items()
    }
    assert config_hash == trial["config_sha256"]
    report = evaluate_gate(gate, _passing_results(gate))
    assert report["passed"]
    assert report["correct_events_within_45s"] == 300


def test_one_failure_or_hash_drift_fails_the_gate(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    results = _passing_results(gate)
    results[17]["correct_event_elapsed_s"] = 45.01
    results[18]["artifact_hashes"]["controller_source"] = "0" * 64
    report = evaluate_gate(gate, results)
    assert not report["passed"]
    assert set(report["trial_failures"]) == {"trial_017", "trial_018"}
    assert "trial_017" in report["trial_failures"]
    assert "trial_018" in report["trial_failures"]


def test_missing_duplicate_and_unexpected_results_fail(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    results = _passing_results(gate)
    results.pop()
    results.append(dict(results[0]))
    results.append({"trial_id": "not_frozen"})
    report = evaluate_gate(gate, results)
    assert not report["passed"]
    assert report["missing_trial_ids"]
    assert report["duplicate_trial_ids"] == ["trial_000"]
    assert report["unexpected_trial_ids"] == ["not_frozen"]


def test_ablation_order_and_promotion_summary() -> None:
    matrix = yaml.safe_load(MATRIX.read_text())
    validate_matrix(matrix)
    stage = next(item for item in matrix["stages"] if item["id"] == "visual_then_persistent_seating")
    records = [
        {
            "correct_event_within_limit": True,
            "correct_event_elapsed_s": 20.0,
            "wrong_port_event": False,
            "offlimit_count": 0,
            "force_penalty": False,
        }
        for _ in range(50)
    ]
    summary = summarize_stage(stage, records)
    assert summary["promotion_pass"]
    assert summary["correct_event_rate"] == 1.0
