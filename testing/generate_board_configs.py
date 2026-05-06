#!/usr/bin/env python3
"""Generate randomized AIC engine board configs for robustness sweeps.

The generator keeps the official YAML schema but varies task-board pose, rail
translations, rail yaw, visible distractors, target type, and target side.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import yaml


NIC_RAILS = [f"nic_rail_{i}" for i in range(5)]
SC_RAILS = [f"sc_rail_{i}" for i in range(2)]
MOUNT_RAILS = [
    "lc_mount_rail_0",
    "sfp_mount_rail_0",
    "sc_mount_rail_0",
    "lc_mount_rail_1",
    "sfp_mount_rail_1",
    "sc_mount_rail_1",
]


def _rail_pose(rng: random.Random, lo: float, hi: float, yaw_abs: float) -> dict:
    return {
        "translation": round(rng.uniform(lo, hi), 4),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": round(rng.uniform(-yaw_abs, yaw_abs), 4),
    }


def _set_absent(task_board: dict, keys: list[str]) -> None:
    for key in keys:
        task_board[key] = {"entity_present": False}


def _randomize_common(task_board: dict, rng: random.Random, trial_idx: int) -> None:
    task_board["pose"] = {
        "x": round(rng.uniform(0.125, 0.175), 4),
        "y": round(rng.uniform(-0.165, 0.105), 4),
        "z": 1.14,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": round(rng.uniform(2.98, 3.34), 4),
    }
    _set_absent(task_board, NIC_RAILS + SC_RAILS + MOUNT_RAILS)

    for rail in rng.sample(NIC_RAILS, rng.randint(1, 4)):
        idx = rail.rsplit("_", 1)[1]
        task_board[rail] = {
            "entity_present": True,
            "entity_name": f"nic_card_{trial_idx}_{idx}",
            "entity_pose": _rail_pose(rng, -0.0215, 0.0234, 0.12),
        }

    for rail in SC_RAILS:
        idx = rail.rsplit("_", 1)[1]
        task_board[rail] = {
            "entity_present": True,
            "entity_name": f"sc_mount_{trial_idx}_{idx}",
            "entity_pose": _rail_pose(rng, -0.0600, 0.0550, 0.15),
        }

    for rail in rng.sample(MOUNT_RAILS, rng.randint(2, 4)):
        side = rail.rsplit("_", 1)[1]
        family = rail.split("_mount_rail_", 1)[0]
        task_board[rail] = {
            "entity_present": True,
            "entity_name": f"{family}_mount_{trial_idx}_{side}",
            "entity_pose": _rail_pose(rng, -0.09425, 0.09425, 0.0),
        }


def _choose_present_nic(task_board: dict, rng: random.Random) -> str:
    present = [rail for rail in NIC_RAILS if task_board[rail]["entity_present"]]
    rail = rng.choice(present)
    return f"nic_card_mount_{rail.rsplit('_', 1)[1]}"


def _make_trial(template: dict, rng: random.Random, trial_idx: int, mode: str) -> dict:
    trial = copy.deepcopy(template)
    task_board = trial["scene"]["task_board"]
    _randomize_common(task_board, rng, trial_idx)

    task = trial["tasks"]["task_1"]
    cable_name = f"cable_{trial_idx}"
    trial["scene"]["cables"] = {
        cable_name: {
            "pose": {
                "gripper_offset": {
                    "x": 0.0,
                    "y": 0.015385,
                    "z": round(rng.uniform(0.0380, 0.0480), 4),
                },
                "roll": 0.4432,
                "pitch": -0.4838,
                "yaw": 1.3303,
            },
            "attach_cable_to_gripper": True,
            "cable_type": "sfp_sc_cable_reversed" if mode == "sc" else "sfp_sc_cable",
        }
    }

    task["cable_type"] = "sfp_sc"
    task["cable_name"] = cable_name
    task["time_limit"] = 180
    if mode == "sc":
        side = rng.choice([0, 1])
        task["plug_type"] = "sc"
        task["plug_name"] = "sc_tip"
        task["port_type"] = "sc"
        task["port_name"] = "sc_port_base"
        task["target_module_name"] = f"sc_port_{side}"
    else:
        task["plug_type"] = "sfp"
        task["plug_name"] = "sfp_tip"
        task["port_type"] = "sfp"
        task["port_name"] = rng.choice(["sfp_port_0", "sfp_port_1"])
        task["target_module_name"] = _choose_present_nic(task_board, rng)
    return trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="aic_engine/config/sc_eval_config.yaml")
    parser.add_argument("--out-dir", default="aic_engine/config/generated_boards")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--configs", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    base_path = Path(args.base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(base_path.read_text())
    first_trial = next(iter(base["trials"].values()))

    rng = random.Random(args.seed)
    for config_idx in range(args.configs):
        cfg = copy.deepcopy(base)
        cfg["trials"] = {}
        modes = ["sc", "sfp", "sc"]
        while len(modes) < args.trials:
            modes.append(rng.choice(["sc", "sfp"]))
        rng.shuffle(modes)
        for trial_i, mode in enumerate(modes[: args.trials], start=1):
            cfg["trials"][f"trial_{trial_i}"] = _make_trial(
                first_trial, rng, config_idx * 100 + trial_i, mode
            )
        path = out_dir / f"robust_boards_seed{args.seed}_{config_idx:02d}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
