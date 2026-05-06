"""Export a trained RSL-RL final-insertion actor to TorchScript.

The ROS-side PerceptionInsert policy expects a small callable module that maps
one 69-D observation vector to one 6-D action vector. RSL-RL checkpoints store
the full actor-critic state, so this script extracts just the actor MLP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"]

    actor = torch.nn.Sequential(
        torch.nn.Linear(69, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 6),
    )
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(actor_state)
    actor.eval()

    example = torch.zeros(1, 69, dtype=torch.float32)
    traced = torch.jit.trace(actor, example)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    print(out)


if __name__ == "__main__":
    main()
