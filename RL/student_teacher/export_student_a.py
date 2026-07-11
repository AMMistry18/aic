"""Export a distilled state student to the AIC 69->6 TorchScript contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from RL.student_teacher.train_student_a import OBS_DIM, build_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=["auto", "legacy", "gazebo_v1", "flowstate_v1"],
        default="auto",
    )
    parser.add_argument("--hidden", type=int, default=0,
                        help="0 reads checkpoint config or infers the first layer")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    config = dict(checkpoint.get("config", {}))
    hidden = int(args.hidden or config.get("hidden", state["net.0.weight"].shape[0]))
    feature_mode = (
        config.get("feature_mode", "legacy")
        if args.feature_mode == "auto"
        else args.feature_mode
    )

    policy = build_policy(hidden=hidden, feature_mode=feature_mode)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    example = torch.zeros(OBS_DIM, dtype=torch.float32)
    traced = torch.jit.trace(policy, example, strict=True)
    traced = torch.jit.freeze(traced.eval())

    rng = np.random.default_rng(20260709)
    max_error = 0.0
    with torch.no_grad():
        for _ in range(16):
            sample = torch.from_numpy(rng.normal(size=OBS_DIM).astype(np.float32))
            error = torch.max(torch.abs(policy(sample) - traced(sample))).item()
            max_error = max(max_error, float(error))
    if max_error > 1e-6:
        raise RuntimeError(f"TorchScript parity failed: max_abs_error={max_error:g}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out))
    metadata = {
        "contract": f"sfp_{feature_mode}_69x6",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "feature_mode": feature_mode,
        "hidden": hidden,
        "wrench_mode": config.get("wrench_mode", "unknown"),
        "grasp_noise": config.get("grasp_noise", "unknown"),
        "action_convention": config.get("action_convention", "unknown"),
        "obs_dim": OBS_DIM,
        "action_dim": 6,
        "torchscript_max_abs_error": max_error,
    }
    out.with_suffix(out.suffix + ".contract.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
