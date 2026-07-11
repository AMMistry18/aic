"""Identify which observation groups drive Flowstate/MuJoCo action mismatch."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch


FIELDS = (
    ("joint_offset", 0, 6),
    ("joint_velocity", 6, 12),
    ("tcp_pose_world_masked", 12, 19),
    ("tcp_velocity_port", 19, 25),
    ("port_pose_world_masked", 25, 32),
    ("tip_delta_port", 32, 35),
    ("tip_rotation_error_port", 35, 38),
    ("alignment_hint", 38, 44),
    ("scripted_hint", 44, 50),
    ("bias", 50, 51),
    ("wrench", 51, 57),
    ("last_action", 57, 63),
    ("tip_axes_port", 63, 69),
)
ACTIVE_GROUPS = tuple(field for field in FIELDS if not field[0].endswith("_masked"))


def _action(model, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(obs.astype(np.float32)[None])).numpy().reshape(-1)


def _named(obs: np.ndarray) -> dict[str, list[float]]:
    return {name: obs[start:end].astype(float).tolist() for name, start, end in FIELDS}


def _replace(base: np.ndarray, donor: np.ndarray, groups) -> np.ndarray:
    result = base.copy()
    for _name, start, end in groups:
        result[start:end] = donor[start:end]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowstate", required=True, type=Path)
    parser.add_argument("--mujoco", required=True, type=Path)
    parser.add_argument("--torchscript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_payload = json.loads(args.flowstate.read_text())
    mujoco_payload = json.loads(args.mujoco.read_text())
    flow = np.asarray(flow_payload["observation"], dtype=np.float32)
    records = mujoco_payload["records"]
    model = torch.jit.load(str(args.torchscript), map_location="cpu")
    model.eval()

    flow_action = _action(model, flow)
    best_record = min(
        records,
        key=lambda record: float(np.linalg.norm(
            np.asarray(record["observation"], dtype=np.float32)[32:38] - flow[32:38]
        )),
    )
    mujoco = np.asarray(best_record["observation"], dtype=np.float32)
    mujoco_action = _action(model, mujoco)
    base_error = float(np.linalg.norm(flow_action - mujoco_action))

    one_group = []
    for group in ACTIVE_GROUPS:
        action = _action(model, _replace(flow, mujoco, [group]))
        error = float(np.linalg.norm(action - mujoco_action))
        one_group.append({
            "group": group[0],
            "action": action.astype(float).tolist(),
            "distance_to_mujoco_action": error,
            "improvement": base_error - error,
        })
    one_group.sort(key=lambda row: row["distance_to_mujoco_action"])

    exhaustive = []
    for size in range(len(ACTIVE_GROUPS) + 1):
        for groups in itertools.combinations(ACTIVE_GROUPS, size):
            action = _action(model, _replace(flow, mujoco, groups))
            error = float(np.linalg.norm(action - mujoco_action))
            exhaustive.append({
                "groups": [group[0] for group in groups],
                "size": size,
                "distance_to_mujoco_action": error,
                "action": action.astype(float).tolist(),
            })
    exhaustive.sort(key=lambda row: (row["distance_to_mujoco_action"], row["size"]))

    result = {
        "flowstate_source": flow_payload.get("source"),
        "mujoco_seed": best_record["seed"],
        "flowstate_observation": _named(flow),
        "mujoco_observation": _named(mujoco),
        "field_delta_l2": {
            name: float(np.linalg.norm(flow[start:end] - mujoco[start:end]))
            for name, start, end in FIELDS
        },
        "flowstate_action": flow_action.astype(float).tolist(),
        "mujoco_action": mujoco_action.astype(float).tolist(),
        "base_action_distance": base_error,
        "single_group_substitutions": one_group,
        "best_subsets": exhaustive[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "mujoco_seed": best_record["seed"],
        "flowstate_action": result["flowstate_action"],
        "mujoco_action": result["mujoco_action"],
        "top_single_groups": one_group[:5],
        "best_subsets": exhaustive[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
