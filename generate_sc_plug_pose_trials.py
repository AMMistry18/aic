#!/usr/bin/env python3
"""Generate strict simulator trials for SC plug-pose data collection.

Only the canonical held-SC setup is accepted: ``sfp_sc_cable_reversed`` with
the SC plug on gripped connection 0, an ``sc_tip`` task, and a present
``sc_rail_N`` backing the requested ``sc_port_N``.  The similarly named
``sc_mount_rail_N`` entries are distractor mounting rails, not insertion ports.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import yaml


CANONICAL_CABLE_ASSET = "sfp_sc_cable_reversed"
CANONICAL_PLUG_NAME = "sc_tip"
CANONICAL_PORT_NAME = "sc_port_base"
VALID_TARGETS = ("sc_port_0", "sc_port_1")
# The simulator publishes observable pose/TF data for cable_0 through cable_4.
# Each trial holds one cable, so generated trials must reuse those physical
# slots rather than inventing cable_5, cable_6, ... .
SIMULATOR_CABLE_SLOTS = 5


@dataclass(frozen=True)
class RandomizationRanges:
    offset_x_m: float = 0.0025
    offset_y_m: float = 0.0025
    offset_z_m: float = 0.0025
    roll_rad: float = math.radians(4.0)
    pitch_rad: float = math.radians(4.0)
    yaw_rad: float = math.radians(4.0)
    board_xy_m: float = 0.010
    board_z_m: float = 0.005
    board_yaw_rad: float = math.radians(8.0)


def _parse_args():
    repo_root = Path(__file__).resolve().parent
    default_output = (
        Path.home()
        / "aic_perception_data"
        / "sc_plug_pose"
        / "sc_plug_pose_trials.yaml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=repo_root / "aic_engine" / "config" / "sc_data_collect.yaml",
    )
    parser.add_argument("--out", type=Path, default=default_output)
    parser.add_argument("--trials", type=int, default=450)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--translation-jitter-mm", type=float, default=2.5)
    parser.add_argument("--rotation-jitter-deg", type=float, default=4.0)
    parser.add_argument("--board-xy-jitter-mm", type=float, default=10.0)
    parser.add_argument("--board-z-jitter-mm", type=float, default=5.0)
    parser.add_argument("--board-yaw-jitter-deg", type=float, default=8.0)
    return parser.parse_args()


def _canonical_task_items(
    trial: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    result = []
    cables = trial.get("scene", {}).get("cables", {})
    board = trial.get("scene", {}).get("task_board", {})
    for name, task in trial.get("tasks", {}).items():
        target = str(task.get("target_module_name", ""))
        if target not in VALID_TARGETS:
            continue
        port_index = int(target.rsplit("_", 1)[1])
        cable = cables.get(task.get("cable_name"), {})
        correct_task = (
            str(task.get("plug_type", "")).lower() == "sc"
            and str(task.get("port_type", "")).lower() == "sc"
            and str(task.get("plug_name", "")).lower() == CANONICAL_PLUG_NAME
            and str(task.get("port_name", "")).lower() == CANONICAL_PORT_NAME
        )
        correct_cable = (
            cable.get("cable_type") == CANONICAL_CABLE_ASSET
            and cable.get("attach_cable_to_gripper") is True
        )
        correct_port = (
            board.get(f"sc_rail_{port_index}", {}).get("entity_present") is True
        )
        if correct_task and correct_cable and correct_port:
            result.append((name, task))
    return result


def eligible_sc_trials(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, trial)
        for name, trial in config.get("trials", {}).items()
        if _canonical_task_items(trial)
    ]


def _jitter(rng: random.Random, radius: float) -> float:
    return rng.uniform(-float(radius), float(radius))


def randomize_trial(
    source_trial: dict[str, Any],
    rng: random.Random,
    ranges: RandomizationRanges,
    *,
    time_limit: int,
) -> dict[str, Any]:
    trial = deepcopy(source_trial)
    candidates = _canonical_task_items(trial)
    if not candidates:
        raise ValueError("source trial violates the canonical SC asset contract")
    task_name, task = rng.choice(candidates)
    task = deepcopy(task)
    task["time_limit"] = int(time_limit)
    trial["tasks"] = {task_name: task}

    cable_name = task["cable_name"]
    cable = deepcopy(trial["scene"]["cables"][cable_name])
    pose = cable["pose"]
    offset = pose["gripper_offset"]
    offset["x"] = float(offset["x"]) + _jitter(rng, ranges.offset_x_m)
    offset["y"] = float(offset["y"]) + _jitter(rng, ranges.offset_y_m)
    offset["z"] = float(offset["z"]) + _jitter(rng, ranges.offset_z_m)
    pose["roll"] = float(pose["roll"]) + _jitter(rng, ranges.roll_rad)
    pose["pitch"] = float(pose["pitch"]) + _jitter(rng, ranges.pitch_rad)
    pose["yaw"] = float(pose["yaw"]) + _jitter(rng, ranges.yaw_rad)
    trial["scene"]["cables"] = {cable_name: cable}

    board_pose = trial["scene"]["task_board"]["pose"]
    board_pose["x"] = float(board_pose["x"]) + _jitter(rng, ranges.board_xy_m)
    board_pose["y"] = float(board_pose["y"]) + _jitter(rng, ranges.board_xy_m)
    board_pose["z"] = float(board_pose["z"]) + _jitter(rng, ranges.board_z_m)
    board_pose["yaw"] = float(board_pose["yaw"]) + _jitter(
        rng, ranges.board_yaw_rad
    )
    return trial


def rebind_single_cable_slot(trial: dict[str, Any], slot: int) -> None:
    """Bind a one-cable trial to an observable simulator cable_N slot."""

    if not 0 <= slot < SIMULATOR_CABLE_SLOTS:
        raise ValueError(f"cable slot {slot} is outside the simulator range")
    task = next(iter(trial["tasks"].values()))
    source_name = str(task["cable_name"])
    cable = deepcopy(trial["scene"]["cables"][source_name])
    cable_name = f"cable_{slot}"
    task["cable_name"] = cable_name
    trial["scene"]["cables"] = {cable_name: cable}


def generate_config(
    template: dict[str, Any],
    *,
    trial_count: int,
    start_index: int,
    seed: int,
    ranges: RandomizationRanges,
    time_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if trial_count <= 0 or start_index <= 0:
        raise ValueError("trial_count and start_index must be positive")
    sources = eligible_sc_trials(template)
    if not sources:
        raise ValueError("template contains no canonical held-SC trials")
    output = {key: deepcopy(value) for key, value in template.items() if key != "trials"}
    output["trials"] = {}
    rng = random.Random(seed)
    manifest = []

    # A resumed range must use exactly the same randomized poses as that range
    # would have received in a single monolithic generation.  Advancing the
    # deterministic RNG also prevents resumed data from duplicating trials 1..N.
    for prior_index in range(start_index - 1):
        _, prior_source = sources[prior_index % len(sources)]
        randomize_trial(prior_source, rng, ranges, time_limit=time_limit)

    for local_index in range(trial_count):
        global_index = start_index + local_index
        source_name, source = sources[(global_index - 1) % len(sources)]
        generated_name = f"trial_{global_index:04d}"
        randomized = randomize_trial(
            source, rng, ranges, time_limit=time_limit
        )
        rebind_single_cable_slot(
            randomized, (global_index - 1) % SIMULATOR_CABLE_SLOTS
        )
        output["trials"][generated_name] = randomized
        task = next(iter(randomized["tasks"].values()))
        cable = randomized["scene"]["cables"][task["cable_name"]]
        target = task["target_module_name"]
        manifest.append(
            {
                "trial": generated_name,
                "global_trial_index": global_index,
                "source_trial": source_name,
                "cable_name": task["cable_name"],
                "physical_cable_slot": (global_index - 1) % SIMULATOR_CABLE_SLOTS,
                "cable_asset": cable["cable_type"],
                "plug_name": task["plug_name"],
                "plug_frame": f"{task['cable_name']}/sc_tip_link",
                "target_module_name": target,
                "port_frame": f"task_board/{target}/sc_port_base_link",
                "entrance_frame": (
                    f"task_board/{target}/sc_port_base_link_entrance"
                ),
                "grasp_pose": deepcopy(cable["pose"]),
                "board_pose": deepcopy(
                    randomized["scene"]["task_board"]["pose"]
                ),
            }
        )
    return output, manifest


def main():
    args = _parse_args()
    template_path = args.template.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    with template_path.open("r", encoding="utf-8") as stream:
        template = yaml.safe_load(stream)
    translation = args.translation_jitter_mm * 1e-3
    ranges = RandomizationRanges(
        offset_x_m=translation,
        offset_y_m=translation,
        offset_z_m=translation,
        roll_rad=math.radians(args.rotation_jitter_deg),
        pitch_rad=math.radians(args.rotation_jitter_deg),
        yaw_rad=math.radians(args.rotation_jitter_deg),
        board_xy_m=args.board_xy_jitter_mm * 1e-3,
        board_z_m=args.board_z_jitter_mm * 1e-3,
        board_yaw_rad=math.radians(args.board_yaw_jitter_deg),
    )
    generated, manifest = generate_config(
        template,
        trial_count=args.trials,
        start_index=args.start_index,
        seed=args.seed,
        ranges=ranges,
        time_limit=args.time_limit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(generated, stream, sort_keys=False)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "template": str(template_path),
                "output": str(output_path),
                "seed": args.seed,
                "start_index": args.start_index,
                "trial_count": args.trials,
                "expected_images": args.trials * 3 * 3,
                "asset_contract": {
                    "cable_asset": CANONICAL_CABLE_ASSET,
                    "plug_asset": "SC Plug",
                    "plug_frame": "cable_N/sc_tip_link",
                    "port_asset": "SC Port",
                    "valid_targets": list(VALID_TARGETS),
                    "port_rail_keys": ["sc_rail_0", "sc_rail_1"],
                    "excluded_distractor_rails": [
                        "sc_mount_rail_0",
                        "sc_mount_rail_1",
                    ],
                },
                "randomization_ranges": asdict(ranges),
                "trials": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.trials} canonical SC plug trials: {output_path}")
    print(f"Expected images at 3 cameras x 3 samples: {args.trials * 9}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
