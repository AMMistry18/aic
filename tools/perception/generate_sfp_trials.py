#!/usr/bin/env python3
"""Generate efficient simulator trials for SFP plug-pose data collection.

The held plug and wrist cameras are nearly rigid relative to one another, so
repeated arm viewpoints add little pose diversity.  This tool instead clones
valid SFP scenes and randomizes the cable's full six-degree-of-freedom spawn /
gripper attachment pose between trials.  Use the generated config with
``DataCollectorSfpPlugPoseGT``; its three synchronized cameras and three short
settled samples yield nine images per trial (about 4,050 images for 450 trials).
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


@dataclass(frozen=True)
class RandomizationRanges:
    offset_x_m: float = 0.003
    offset_y_m: float = 0.003
    offset_z_m: float = 0.003
    roll_rad: float = math.radians(6.0)
    pitch_rad: float = math.radians(6.0)
    yaw_rad: float = math.radians(6.0)
    board_xy_m: float = 0.012
    board_z_m: float = 0.006
    board_yaw_rad: float = math.radians(10.0)


def _parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_output = (
        Path.home()
        / "aic_perception_data"
        / "sfp_plug_pose"
        / "sfp_plug_pose_trials.yaml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=repo_root / "aic_engine" / "config" / "base_config.yaml",
        help="engine YAML containing at least one SFP trial",
    )
    parser.add_argument("--out", type=Path, default=default_output)
    parser.add_argument("--trials", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--translation-jitter-mm", type=float, default=3.0)
    parser.add_argument("--rotation-jitter-deg", type=float, default=6.0)
    parser.add_argument("--board-xy-jitter-mm", type=float, default=12.0)
    parser.add_argument("--board-z-jitter-mm", type=float, default=6.0)
    parser.add_argument("--board-yaw-jitter-deg", type=float, default=10.0)
    return parser.parse_args()


def _sfp_task_items(trial: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    tasks = trial.get("tasks", {})
    return [
        (name, task)
        for name, task in tasks.items()
        if str(task.get("port_type", "")).lower() == "sfp"
        and str(task.get("plug_type", "")).lower() == "sfp"
    ]


def eligible_sfp_trials(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    eligible = []
    for name, trial in config.get("trials", {}).items():
        if _sfp_task_items(trial):
            eligible.append((name, trial))
    return eligible


def _uniform_jitter(rng: random.Random, radius: float) -> float:
    return rng.uniform(-float(radius), float(radius))


def randomize_trial(
    source_trial: dict[str, Any],
    rng: random.Random,
    ranges: RandomizationRanges,
    *,
    time_limit: int,
) -> dict[str, Any]:
    """Clone one scene, retain one SFP task, and perturb grasp/background pose."""

    trial = deepcopy(source_trial)
    sfp_tasks = _sfp_task_items(trial)
    if not sfp_tasks:
        raise ValueError("source trial contains no SFP task")
    selected_task_name, selected_task = rng.choice(sfp_tasks)
    selected_task = deepcopy(selected_task)
    selected_task["time_limit"] = int(time_limit)
    trial["tasks"] = {selected_task_name: selected_task}

    cable_name = selected_task["cable_name"]
    cables = trial["scene"]["cables"]
    if cable_name not in cables:
        raise ValueError(f"task references missing cable {cable_name!r}")
    cable = deepcopy(cables[cable_name])
    pose = cable["pose"]
    offset = pose["gripper_offset"]
    offset["x"] = float(offset["x"]) + _uniform_jitter(rng, ranges.offset_x_m)
    offset["y"] = float(offset["y"]) + _uniform_jitter(rng, ranges.offset_y_m)
    offset["z"] = float(offset["z"]) + _uniform_jitter(rng, ranges.offset_z_m)
    pose["roll"] = float(pose["roll"]) + _uniform_jitter(rng, ranges.roll_rad)
    pose["pitch"] = float(pose["pitch"]) + _uniform_jitter(rng, ranges.pitch_rad)
    pose["yaw"] = float(pose["yaw"]) + _uniform_jitter(rng, ranges.yaw_rad)
    trial["scene"]["cables"] = {cable_name: cable}

    board_pose = trial["scene"]["task_board"]["pose"]
    board_pose["x"] = float(board_pose["x"]) + _uniform_jitter(rng, ranges.board_xy_m)
    board_pose["y"] = float(board_pose["y"]) + _uniform_jitter(rng, ranges.board_xy_m)
    board_pose["z"] = float(board_pose["z"]) + _uniform_jitter(rng, ranges.board_z_m)
    board_pose["yaw"] = float(board_pose["yaw"]) + _uniform_jitter(
        rng, ranges.board_yaw_rad
    )
    return trial


def generate_config(
    template: dict[str, Any],
    *,
    trial_count: int,
    seed: int,
    ranges: RandomizationRanges,
    time_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    sources = eligible_sfp_trials(template)
    if not sources:
        raise ValueError("template contains no SFP-to-SFP trials")
    output = {key: deepcopy(value) for key, value in template.items() if key != "trials"}
    output["trials"] = {}
    rng = random.Random(seed)
    manifest = []
    for index in range(trial_count):
        source_name, source = sources[index % len(sources)]
        generated_name = f"trial_{index + 1:04d}"
        randomized = randomize_trial(
            source,
            rng,
            ranges,
            time_limit=time_limit,
        )
        output["trials"][generated_name] = randomized
        task = next(iter(randomized["tasks"].values()))
        cable = randomized["scene"]["cables"][task["cable_name"]]
        manifest.append(
            {
                "trial": generated_name,
                "source_trial": source_name,
                "cable_name": task["cable_name"],
                "cable_type": cable["cable_type"],
                "grasp_pose": deepcopy(cable["pose"]),
                "board_pose": deepcopy(randomized["scene"]["task_board"]["pose"]),
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
                "trial_count": args.trials,
                "expected_images": args.trials * 3 * 3,
                "randomization_ranges": asdict(ranges),
                "trials": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.trials} randomized SFP trials: {output_path}")
    print(f"Expected images at 3 cameras x 3 samples: {args.trials * 9}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
