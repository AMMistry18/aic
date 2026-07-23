#!/usr/bin/env python3
"""Record one live trial from ROS topics using monotonic receipt timestamps.

This observer intentionally ignores the policy's boolean action result as a
success oracle.  Physical success comes only from the correct
``/scoring/insertion_event`` namespace.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from common import load_json, sha256_file, validate_content_sha256, write_json
from trial_evidence import TrialEvidenceRecorder


TERMINAL_STATUS_NAMES = {4: "succeeded", 5: "canceled", 6: "aborted"}
ACTIVE_STATUSES = {1, 2}


def _uuid_bytes(status: Any) -> bytes:
    return bytes(status.goal_info.goal_id.uuid)


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _gate_trial(
    gate: dict[str, Any], gate_path: Path, trial_id: str
) -> tuple[dict[str, Any], dict[str, str], str]:
    scenario = gate["scenario_suite"]
    trial = next(
        (item for item in scenario["trials"] if item["trial_id"] == trial_id),
        None,
    )
    if trial is None:
        raise ValueError(f"Trial {trial_id!r} is not in the frozen gate")
    gate_dir = gate_path.resolve().parent
    scenario_manifest_path = _resolve_path(
        gate_dir, gate["scenario_manifest"]["path"]
    )
    if sha256_file(scenario_manifest_path) != gate["scenario_manifest"]["sha256"]:
        raise ValueError("Scenario manifest file hash changed after gate freeze")
    scenario_manifest = load_json(scenario_manifest_path)
    validate_content_sha256(scenario_manifest, "scenario manifest")
    if (
        scenario_manifest["content_sha256"]
        != gate["scenario_manifest"]["content_sha256"]
    ):
        raise ValueError("Scenario manifest content hash does not match frozen gate")
    config_path = _resolve_path(
        scenario_manifest_path.parent, trial["config_path"]
    )
    config_hash = sha256_file(config_path)
    if config_hash != trial["config_sha256"]:
        raise ValueError(f"Scenario config hash changed: {config_path}")

    artifacts = {}
    for name, metadata in gate["artifacts"].items():
        artifact_path = _resolve_path(gate_dir, metadata["path"])
        actual_hash = sha256_file(artifact_path)
        if actual_hash != metadata["sha256"]:
            raise ValueError(f"Frozen artifact hash changed: {name} ({artifact_path})")
        artifacts[name] = actual_hash
    return trial, artifacts, config_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--expected-event")
    parser.add_argument("--frozen-gate", type=Path)
    parser.add_argument(
        "--runtime-image-id",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="Actual image id, normally supplied by run_docker_gate.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-timeout-s", type=float, default=120.0)
    parser.add_argument("--terminal-timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    gate_hash = None
    artifact_hashes: dict[str, str] = {}
    config_hash = None
    expected_event = args.expected_event
    max_event_s = 45.0
    force_threshold = 20.0
    force_duration = 1.0
    runtime_image_ids: dict[str, str] = {}
    if args.frozen_gate:
        gate = load_json(args.frozen_gate)
        validate_content_sha256(gate, "frozen gate")
        trial, artifact_hashes, config_hash = _gate_trial(
            gate, args.frozen_gate, args.trial_id
        )
        gate_hash = gate["content_sha256"]
        expected_event = trial["expected_insertion_event"]
        criteria = gate["criteria"]
        max_event_s = float(criteria["max_correct_event_wall_seconds"])
        force_threshold = float(criteria["force_penalty_threshold_n"])
        force_duration = float(criteria["force_penalty_duration_seconds"])
        for value in args.runtime_image_id:
            if "=" not in value:
                parser.error(f"Expected NAME=SHA256 for --runtime-image-id, got {value!r}")
            name, image_id = value.split("=", 1)
            runtime_image_ids[name] = image_id
        expected_images = {
            name: metadata["id"] for name, metadata in gate["runtime_images"].items()
        }
        if runtime_image_ids != expected_images:
            parser.error(
                "--runtime-image-id values do not match the frozen evaluator/model images"
            )
    if not expected_event:
        parser.error("--expected-event is required without --frozen-gate")

    try:
        import rclpy
        from action_msgs.msg import GoalStatusArray
        from aic_control_interfaces.msg import ControllerState
        from geometry_msgs.msg import WrenchStamped
        from rclpy.node import Node
        from ros_gz_interfaces.msg import Contacts
        from std_msgs.msg import String
    except ImportError as error:
        raise SystemExit(
            "ROS 2 Python packages are required to observe a live trial"
        ) from error

    recorder = TrialEvidenceRecorder(
        trial_id=args.trial_id,
        expected_insertion_event=expected_event,
        max_event_wall_seconds=max_event_s,
        force_threshold_n=force_threshold,
        force_penalty_duration_seconds=force_duration,
        frozen_gate_sha256=gate_hash,
        artifact_hashes=artifact_hashes,
        runtime_image_ids=runtime_image_ids,
        config_sha256=config_hash,
    )

    class Observer(Node):
        def __init__(self) -> None:
            super().__init__(f"sfp_v50_gate_{args.trial_id[-12:]}")
            self.active_goal: bytes | None = None
            self.complete = False
            self.created_ns = time.monotonic_ns()
            self.tare_force: tuple[float, float, float] | None = None
            self.create_subscription(
                GoalStatusArray, "/insert_cable/_action/status", self.on_status, 10
            )
            self.create_subscription(
                String, "/scoring/insertion_event", self.on_event, 10
            )
            self.create_subscription(
                WrenchStamped, "/fts_broadcaster/wrench", self.on_wrench, 100
            )
            self.create_subscription(
                ControllerState,
                "/aic_controller/controller_state",
                self.on_controller_state,
                10,
            )
            self.create_subscription(
                Contacts, "/aic/gazebo/contacts/off_limit", self.on_contacts, 10
            )

        def on_status(self, message: Any) -> None:
            if self.active_goal is None:
                active = [status for status in message.status_list if status.status in ACTIVE_STATUSES]
                if active:
                    selected = active[-1]
                    self.active_goal = _uuid_bytes(selected)
                    recorder.start()
                    self.get_logger().info("Started monotonic gate timing")
                return
            for status in message.status_list:
                if _uuid_bytes(status) != self.active_goal:
                    continue
                terminal = TERMINAL_STATUS_NAMES.get(status.status)
                if terminal:
                    recorder.finish(terminal)
                    self.complete = True
                    return

        def on_event(self, message: Any) -> None:
            recorder.observe_insertion_event(message.data)

        def on_wrench(self, message: Any) -> None:
            # ScoringTier2 ignores wrench samples until it has a controller-state
            # tare, subtracts that tare, and uses the wrench header timestamps
            # for its one-second penalty calculation. Mirror that exactly;
            # monotonic time remains the separate event/deadline clock.
            if self.tare_force is None:
                return
            force = message.wrench.force
            stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
                message.header.stamp.nanosec
            )
            recorder.observe_force(
                force.x - self.tare_force[0],
                force.y - self.tare_force[1],
                force.z - self.tare_force[2],
                stamp_ns,
            )

        def on_controller_state(self, message: Any) -> None:
            force = message.fts_tare_offset.wrench.force
            self.tare_force = (force.x, force.y, force.z)

        def on_contacts(self, message: Any) -> None:
            if message.contacts:
                recorder.observe_offlimit()

        def poll_timeout(self) -> None:
            now_ns = time.monotonic_ns()
            if recorder.start_ns is None:
                if (now_ns - self.created_ns) / 1e9 > args.start_timeout_s:
                    raise TimeoutError("No insert_cable action became active")
            elif (now_ns - recorder.start_ns) / 1e9 > args.terminal_timeout_s:
                recorder.finish("observer_timeout", now_ns)
                self.complete = True

    rclpy.init()
    node = Observer()
    exit_code = 0
    try:
        while rclpy.ok() and not node.complete:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.poll_timeout()
    except TimeoutError as error:
        payload = {
            "schema_version": 1,
            "trial_id": args.trial_id,
            "observer_error": str(error),
            "local_gate_pass": False,
        }
        exit_code = 2
    else:
        payload = recorder.to_result()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
