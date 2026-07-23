#!/usr/bin/env python3
"""Bind the 300-scenario suite to exact runtime artifacts and hashes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from common import (
    attach_content_sha256,
    docker_image_id,
    load_json,
    parse_artifact,
    relative_path,
    sha256_file,
    validate_content_sha256,
    write_json,
)


def freeze_gate(
    *,
    scenario_manifest_path: Path,
    matrix_path: Path,
    artifacts: dict[str, Path],
    runtime_images: dict[str, dict[str, str]],
    output_path: Path,
    gate_id: str,
) -> dict[str, Any]:
    scenario = load_json(scenario_manifest_path)
    validate_content_sha256(scenario, "scenario manifest")
    if scenario.get("task_scope") != "sfp_only" or scenario.get("trial_count") != 300:
        raise ValueError("Frozen gate requires exactly 300 SFP-only scenarios")
    criteria = scenario.get("criteria", {})
    required_criteria = {
        "required_correct_events": 300,
        "max_correct_event_wall_seconds": 45.0,
        "max_wrong_port_events": 0,
        "max_offlimit_trials": 0,
        "max_force_penalty_trials": 0,
        "force_penalty_threshold_n": 20.0,
        "force_penalty_duration_seconds": 1.0,
        "clock": "time.monotonic_ns",
    }
    for key, expected in required_criteria.items():
        if criteria.get(key) != expected:
            raise ValueError(f"Scenario criterion {key} must be {expected!r}")

    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    final_stage = next(
        stage for stage in matrix["stages"] if stage["id"] == "frozen_v50_full"
    )
    required_artifacts = set(final_stage["required_artifacts"])
    missing = sorted(required_artifacts - set(artifacts))
    extra = sorted(set(artifacts) - required_artifacts)
    if missing or extra:
        raise ValueError(f"Artifact names mismatch; missing={missing}, extra={extra}")

    artifact_records = {
        name: {
            "path": relative_path(path, output_path.parent),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(artifacts.items())
    }
    if set(runtime_images) != {"evaluator", "model"}:
        raise ValueError("runtime_images must contain exactly evaluator and model")
    for name, metadata in runtime_images.items():
        if set(metadata) != {"ref", "id"}:
            raise ValueError(f"Runtime image {name} must contain ref and id")
        if not metadata["id"].startswith("sha256:"):
            raise ValueError(f"Runtime image {name} has an invalid Docker image id")
    gate = attach_content_sha256(
        {
            "schema_version": 1,
            "gate_id": gate_id,
            "bound_utc": datetime.now(timezone.utc).isoformat(),
            "task_scope": "sfp_only",
            "trial_count": 300,
            "criteria": criteria,
            "scenario_manifest": {
                "path": relative_path(scenario_manifest_path, output_path.parent),
                "sha256": sha256_file(scenario_manifest_path),
                "content_sha256": scenario["content_sha256"],
            },
            # Embed the trial contract so observation and evaluation never rely
            # on a mutable side file after this gate is signed.
            "scenario_suite": {
                "suite_id": scenario["suite_id"],
                "master_seed": scenario["master_seed"],
                "trials": scenario["trials"],
                "shards": scenario["shards"],
            },
            "ablation_matrix": {
                "path": relative_path(matrix_path, output_path.parent),
                "sha256": sha256_file(matrix_path),
                "final_stage": final_stage,
            },
            "artifacts": artifact_records,
            "runtime_images": runtime_images,
        }
    )
    write_json(output_path, gate)
    return gate


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=here / "ablation_matrix.yaml")
    parser.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--model-image", required=True)
    parser.add_argument(
        "--eval-image", default="ghcr.io/intrinsic-dev/aic/aic_eval:latest"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-id", default="sfp_v50_full_gate_v1")
    args = parser.parse_args()
    parsed = dict(parse_artifact(value) for value in args.artifact)
    runtime_images = {
        "evaluator": {"ref": args.eval_image, "id": docker_image_id(args.eval_image)},
        "model": {"ref": args.model_image, "id": docker_image_id(args.model_image)},
    }
    gate = freeze_gate(
        scenario_manifest_path=args.scenario_manifest.resolve(),
        matrix_path=args.matrix.resolve(),
        artifacts=parsed,
        runtime_images=runtime_images,
        output_path=args.output.resolve(),
        gate_id=args.gate_id,
    )
    print(f"{args.output}: {gate['content_sha256']}")


if __name__ == "__main__":
    main()
