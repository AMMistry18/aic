#!/usr/bin/env python3
"""Generate deterministic, SFP-only v50 validation configurations.

The generated configs deliberately pin the model name to ``cable_0``.  The
repository's ROS-Gazebo bridge only maps insertion events for cable_0..cable_4;
older robustness configs used names such as cable_101 and silently lost the
authoritative insertion event.  Each engine trial is isolated, so reusing
cable_0 is safe and keeps physical success observable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import yaml

from common import attach_content_sha256, relative_path, sha256_file, write_json


GENERATOR_VERSION = "sfp-v50-heldout-v1"
FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
CABLE_NAME = "cable_0"
INSERTION_ROS_TOPIC = "/scoring/insertion_event"
INSERTION_GZ_TOPIC = "/cable_0/insertion_event"
NIC_RAILS = tuple(f"nic_rail_{index}" for index in range(5))
SC_RAILS = tuple(f"sc_rail_{index}" for index in range(2))
MOUNT_RAILS = tuple(
    f"{family}_mount_rail_{side}"
    for side in range(2)
    for family in ("lc", "sfp", "sc")
)
BOARD_YAW_BANDS = ((2.80, 2.95), (2.95, 3.05), (3.05, 3.20), (3.20, 3.35))


def _round(value: float) -> float:
    return round(float(value), 7)


def _balanced(values: Sequence[Any], count: int, rng: random.Random) -> list[Any]:
    pool = [values[index % len(values)] for index in range(count)]
    rng.shuffle(pool)
    return pool


def _trial_seed(master_seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{master_seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _cholesky(covariance: list[list[float]]) -> list[list[float]]:
    size = len(covariance)
    if size == 0 or any(len(row) != size for row in covariance):
        raise ValueError("covariance must be a non-empty square matrix")
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = covariance[row][column] - sum(
                lower[row][k] * lower[column][k] for k in range(column)
            )
            if row == column:
                if residual <= 0.0:
                    raise ValueError("covariance must be positive definite")
                lower[row][column] = math.sqrt(residual)
            else:
                lower[row][column] = residual / lower[column][column]
    return lower


class GraspSampler:
    """Samples a full six-dimensional gripper-to-plug pose distribution."""

    def __init__(self, config_path: Path):
        self.path = config_path.resolve()
        self.config = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported grasp distribution schema")
        if tuple(self.config.get("fields", ())) != FIELDS:
            raise ValueError(f"grasp fields must be exactly {FIELDS}")
        sampling = self.config.get("sampling", {})
        self.kind = sampling.get("kind")
        self.sampling = sampling
        self._empirical_rows: list[list[float]] = []
        if self.kind == "truncated_multivariate_normal":
            for key in ("mean", "covariance", "lower", "upper"):
                if key not in sampling:
                    raise ValueError(f"Missing grasp sampling.{key}")
            if any(len(sampling[key]) != 6 for key in ("mean", "lower", "upper")):
                raise ValueError("mean/lower/upper must each contain six values")
            self._cholesky = _cholesky(sampling["covariance"])
        elif self.kind == "empirical_csv":
            raw_samples = Path(sampling["samples_file"]).expanduser()
            if not raw_samples.is_absolute():
                raw_samples = self.path.parent / raw_samples
            with raw_samples.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != FIELDS:
                    raise ValueError(f"Empirical CSV columns must be exactly {FIELDS}")
                self._empirical_rows = [
                    [float(row[field]) for field in FIELDS] for row in reader
                ]
            if not self._empirical_rows:
                raise ValueError("Empirical grasp CSV contains no rows")
            jitter = sampling.get("jitter_std", [0.0] * 6)
            if len(jitter) != 6 or any(float(value) < 0.0 for value in jitter):
                raise ValueError("jitter_std must contain six non-negative values")
        else:
            raise ValueError(
                "sampling.kind must be truncated_multivariate_normal or empirical_csv"
            )

    def sample(self, rng: random.Random) -> dict[str, float]:
        if self.kind == "empirical_csv":
            values = list(rng.choice(self._empirical_rows))
            jitter = self.sampling.get("jitter_std", [0.0] * 6)
            values = [
                value + rng.gauss(0.0, float(sigma))
                for value, sigma in zip(values, jitter)
            ]
            lower = self.sampling.get("lower")
            upper = self.sampling.get("upper")
            if lower is not None:
                values = [max(float(bound), value) for bound, value in zip(lower, values)]
            if upper is not None:
                values = [min(float(bound), value) for bound, value in zip(upper, values)]
            return {field: _round(value) for field, value in zip(FIELDS, values)}

        mean = [float(value) for value in self.sampling["mean"]]
        lower = [float(value) for value in self.sampling["lower"]]
        upper = [float(value) for value in self.sampling["upper"]]
        for _ in range(10_000):
            normal = [rng.gauss(0.0, 1.0) for _ in range(6)]
            values = [
                mean[row]
                + sum(self._cholesky[row][column] * normal[column] for column in range(row + 1))
                for row in range(6)
            ]
            if all(lo <= value <= hi for value, lo, hi in zip(values, lower, upper)):
                return {field: _round(value) for field, value in zip(FIELDS, values)}
        raise RuntimeError("Unable to draw a grasp sample inside configured bounds")


def bridge_supports_cable_zero(path: Path) -> bool:
    mappings = yaml.safe_load(path.read_text(encoding="utf-8"))
    return any(
        mapping.get("ros_topic_name") == INSERTION_ROS_TOPIC
        and mapping.get("gz_topic_name") == INSERTION_GZ_TOPIC
        for mapping in mappings
    )


def _entity_pose(rng: random.Random, low: float, high: float, yaw: float) -> dict:
    return {
        "translation": _round(rng.uniform(low, high)),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": _round(rng.uniform(-yaw, yaw)),
    }


def _make_trial(
    scenario_id: str,
    scenario_seed: int,
    target_rail: int,
    target_port: int,
    nic_count: int,
    yaw_band: tuple[float, float],
    grasp_sampler: GraspSampler,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(scenario_seed)
    active_nics = {target_rail}
    candidates = [index for index in range(5) if index != target_rail]
    active_nics.update(rng.sample(candidates, k=max(0, nic_count - 1)))

    board: dict[str, Any] = {
        "pose": {
            "x": _round(rng.uniform(0.13, 0.20)),
            "y": _round(rng.uniform(-0.25, 0.10)),
            "z": 1.14,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": _round(rng.uniform(*yaw_band)),
        }
    }
    for index, rail in enumerate(NIC_RAILS):
        if index in active_nics:
            board[rail] = {
                "entity_present": True,
                "entity_name": f"nic_card_{scenario_id}_{index}",
                "entity_pose": _entity_pose(rng, -0.0215, 0.0234, 0.12),
            }
        else:
            board[rail] = {"entity_present": False}

    for index, rail in enumerate(SC_RAILS):
        if rng.random() < 0.75:
            board[rail] = {
                "entity_present": True,
                # Keep this distinct from the separate sc_mount_rail family.
                "entity_name": f"sc_port_mount_{scenario_id}_{index}",
                "entity_pose": _entity_pose(rng, -0.06, 0.055, 0.15),
            }
        else:
            board[rail] = {"entity_present": False}

    active_mounts = set(rng.sample(MOUNT_RAILS, k=rng.randint(2, 4)))
    for rail in MOUNT_RAILS:
        if rail in active_mounts:
            side = rail.rsplit("_", 1)[1]
            family = rail.split("_mount_rail_", 1)[0]
            board[rail] = {
                "entity_present": True,
                "entity_name": f"{family}_accessory_mount_{scenario_id}_{side}",
                "entity_pose": _entity_pose(rng, -0.09425, 0.09425, 0.0),
            }
        else:
            board[rail] = {"entity_present": False}

    grasp = grasp_sampler.sample(rng)
    target_module = f"nic_card_mount_{target_rail}"
    port_name = f"sfp_port_{target_port}"
    trial = {
        "scene": {
            "task_board": board,
            "cables": {
                CABLE_NAME: {
                    "pose": {
                        "gripper_offset": {
                            "x": grasp["x"],
                            "y": grasp["y"],
                            "z": grasp["z"],
                        },
                        "roll": grasp["roll"],
                        "pitch": grasp["pitch"],
                        "yaw": grasp["yaw"],
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable",
                }
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": CABLE_NAME,
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": port_name,
                "target_module_name": target_module,
                "time_limit": 45,
            }
        },
    }
    metadata = {
        "trial_id": scenario_id,
        "scenario_seed": scenario_seed,
        "cable_name": CABLE_NAME,
        "target_rail": target_rail,
        "target_port": target_port,
        "nic_count": nic_count,
        "board_yaw_band": list(yaw_band),
        "expected_insertion_event": f"{target_module}/{port_name}",
        "grasp_6dof": grasp,
        "board_pose": board["pose"],
    }
    return trial, metadata


def generate_suite(
    *,
    base_path: Path,
    bridge_path: Path,
    grasp_path: Path,
    output_dir: Path,
    suite_id: str,
    master_seed: int,
    trial_count: int,
    shard_size: int,
) -> Path:
    if trial_count <= 0 or shard_size <= 0:
        raise ValueError("trial_count and shard_size must be positive")
    if not bridge_supports_cable_zero(bridge_path):
        raise ValueError(
            f"{bridge_path} does not bridge {INSERTION_GZ_TOPIC} to "
            f"{INSERTION_ROS_TOPIC}"
        )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    for key in ("scoring", "task_board_limits"):
        if key not in base:
            raise ValueError(f"Base config is missing {key!r}")
    grasp_sampler = GraspSampler(grasp_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_rng = random.Random(master_seed)
    target_rails = _balanced(tuple(range(5)), trial_count, plan_rng)
    target_ports = _balanced((0, 1), trial_count, plan_rng)
    nic_counts = _balanced((1, 2, 3, 4, 5), trial_count, plan_rng)
    yaw_bands = _balanced(BOARD_YAW_BANDS, trial_count, plan_rng)

    generated: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for index in range(1, trial_count + 1):
        trial_id = f"{suite_id}_{index:04d}"
        trial, metadata = _make_trial(
            trial_id,
            _trial_seed(master_seed, index),
            target_rails[index - 1],
            target_ports[index - 1],
            nic_counts[index - 1],
            yaw_bands[index - 1],
            grasp_sampler,
        )
        generated.append((trial_id, trial, metadata))

    shard_records: list[dict[str, Any]] = []
    trial_records: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, trial_count, shard_size)):
        batch = generated[start : start + shard_size]
        config = {
            "scoring": deepcopy(base["scoring"]),
            "task_board_limits": deepcopy(base["task_board_limits"]),
            "trials": {trial_id: trial for trial_id, trial, _ in batch},
        }
        shard_name = f"{suite_id}_{shard_index:02d}.yaml"
        shard_path = output_dir / shard_name
        shard_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        shard_sha = sha256_file(shard_path)
        trial_ids = [trial_id for trial_id, _, _ in batch]
        shard_records.append(
            {
                "path": shard_name,
                "sha256": shard_sha,
                "trial_ids": trial_ids,
            }
        )
        for _, _, metadata in batch:
            trial_records.append(
                {
                    **metadata,
                    "config_path": shard_name,
                    "config_sha256": shard_sha,
                }
            )

    manifest = attach_content_sha256(
        {
            "schema_version": 1,
            "generator_version": GENERATOR_VERSION,
            "suite_id": suite_id,
            "master_seed": master_seed,
            "trial_count": trial_count,
            "shard_size": shard_size,
            "task_scope": "sfp_only",
            "bridge_strategy": {
                "description": "Pin every isolated trial to bridged cable_0",
                "cable_name": CABLE_NAME,
                "ros_topic": INSERTION_ROS_TOPIC,
                "gz_topic": INSERTION_GZ_TOPIC,
                "bridge_config": relative_path(bridge_path, output_dir),
                "bridge_config_sha256": sha256_file(bridge_path),
            },
            "base_config": {
                "path": relative_path(base_path, output_dir),
                "sha256": sha256_file(base_path),
            },
            "grasp_distribution": {
                "path": relative_path(grasp_path, output_dir),
                "sha256": sha256_file(grasp_path),
                "fields": list(FIELDS),
            },
            "criteria": {
                "required_correct_events": trial_count,
                "max_correct_event_wall_seconds": 45.0,
                "max_wrong_port_events": 0,
                "max_offlimit_trials": 0,
                "max_force_penalty_trials": 0,
                "force_penalty_threshold_n": 20.0,
                "force_penalty_duration_seconds": 1.0,
                "clock": "time.monotonic_ns",
            },
            "shards": shard_records,
            "trials": trial_records,
        }
    )
    manifest_path = output_dir / f"{suite_id}.manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("aic_engine/config/base_config.yaml"))
    parser.add_argument(
        "--bridge", type=Path, default=Path("aic_bringup/config/ros_gz_bridge_config.yaml")
    )
    parser.add_argument(
        "--grasp-distribution",
        type=Path,
        default=here / "grasp_distribution_qualification.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-id", default="sfp_v50_frozen_holdout_v1")
    parser.add_argument("--seed", type=int, default=2026071801)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--shard-size", type=int, default=30)
    args = parser.parse_args()
    manifest = generate_suite(
        base_path=args.base.resolve(),
        bridge_path=args.bridge.resolve(),
        grasp_path=args.grasp_distribution.resolve(),
        output_dir=args.output_dir.resolve(),
        suite_id=args.suite_id,
        master_seed=args.seed,
        trial_count=args.trials,
        shard_size=args.shard_size,
    )
    print(manifest)


if __name__ == "__main__":
    main()
