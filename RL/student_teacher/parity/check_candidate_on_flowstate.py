"""Gate a candidate student on the captured Flowstate handoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from RL.student_teacher.train_student_a import build_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--torchscript", required=True, type=Path)
    parser.add_argument("--flowstate", required=True, type=Path)
    parser.add_argument("--mujoco", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    config = dict(checkpoint.get("config", {}))
    hidden = int(config.get("hidden", state["net.0.weight"].shape[0]))
    feature_mode = config.get("feature_mode", "legacy")
    policy = build_policy(hidden=hidden, feature_mode=feature_mode)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    scripted = torch.jit.load(str(args.torchscript), map_location="cpu").eval()

    flow = np.asarray(json.loads(args.flowstate.read_text())["observation"], dtype=np.float32)
    mujoco_payload = json.loads(args.mujoco.read_text())
    mujoco = np.asarray(mujoco_payload["records"][0]["observation"], dtype=np.float32)

    def actions(obs):
        tensor = torch.from_numpy(obs[None])
        with torch.no_grad():
            eager = policy(tensor).numpy().reshape(-1)
            traced = scripted(tensor).numpy().reshape(-1)
        return eager, traced

    flow_eager, flow_scripted = actions(flow)
    mujoco_eager, mujoco_scripted = actions(mujoco)
    joint_swap = flow.copy()
    joint_swap[:6] = mujoco[:6]
    joint_swap_eager, _ = actions(joint_swap)

    result = {
        "feature_mode": feature_mode,
        "flowstate_action": flow_eager.astype(float).tolist(),
        "mujoco_action": mujoco_eager.astype(float).tolist(),
        "flowstate_with_mujoco_joints_action": joint_swap_eager.astype(float).tolist(),
        "joint_swap_action_delta": float(np.linalg.norm(flow_eager - joint_swap_eager)),
        "flowstate_torchscript_max_error": float(np.max(np.abs(flow_eager - flow_scripted))),
        "mujoco_torchscript_max_error": float(np.max(np.abs(mujoco_eager - mujoco_scripted))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    if result["flowstate_torchscript_max_error"] > 1e-6:
        raise SystemExit("Flowstate eager/TorchScript mismatch")
    if result["joint_swap_action_delta"] > 1e-6:
        raise SystemExit("candidate still depends on simulator-specific joint offsets")
    if float(flow_eager[2]) <= 0.0:
        raise SystemExit("candidate does not command inward motion at aligned Flowstate handoff")
    if float(np.max(np.abs(flow_eager[:2]))) >= 0.8:
        raise SystemExit("candidate saturates a lateral action at aligned Flowstate handoff")


if __name__ == "__main__":
    main()
