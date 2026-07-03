"""Smoke test: prove W&B lands on entity=satya_anandh, project=1-inch-intrinsic-policy.

Reuses the existing `WandbLogger` from `RL/logging_utils.py`. Prints the
target entity+project to stdout at start so it's obvious where the run
will land. Then synthesizes a small fake training loop and logs the same
metric names the SAC trainer will emit (reward/episode, reward/mean,
loss/policy, loss/critic, train/global_step, train/learning_rate).

Usage:
    export WANDB_API_KEY=<your_key>
    pixi run python RL/scripts/connect_wandb.py --steps 50 --run-name mujoco_smoke

After it finishes, verify the run URL printed at the top points at
https://wandb.ai/satya_anandh/1-inch-intrinsic-policy/runs/<id>.

Other modes (all reuse the same WandbLogger):
    pixi run python RL/scripts/connect_wandb.py                 # fails if no API key
    WANDB_MODE=offline pixi run python RL/scripts/connect_wandb.py --steps 20
    WANDB_MODE=disabled pixi run python RL/scripts/connect_wandb.py
"""

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.logging_utils import WANDB_ENTITY, WANDB_PROJECT, WandbLogger


def main() -> int:
    p = argparse.ArgumentParser(
        description="Smoke test the WandbLogger against satya_anandh / 1-inch-intrinsic-policy."
    )
    p.add_argument("--steps", type=int, default=50,
                   help="number of fake training steps to log (default 50)")
    p.add_argument("--run-name", type=str, default="mujoco_smoke",
                   help="W&B run name (default: mujoco_smoke)")
    p.add_argument("--log-every", type=int, default=10,
                   help="print stdout every N steps (default 10)")
    args = p.parse_args()

    print(f"[wandb] target entity={WANDB_ENTITY!r} project={WANDB_PROJECT!r}",
          flush=True)

    # Reuse the same WandbLogger that's wired into RL/train.py.
    cb = WandbLogger(
        run_name=args.run_name,
        config={
            "smoke": True,
            "framework": "mujoco + wandb smoke",
            "steps": args.steps,
            "log_every": args.log_every,
        },
        enabled=True,
    )
    # BaseCallback needs a `model` attribute with a `logger.name_to_value`
    # dict, so we monkey-patch a stub. The SB3 SAC trainer will pass its
    # real model here in Step 3.
    cb.model = type("StubModel", (), {
        "logger": type("StubLogger", (), {
            "name_to_value": {
                "train/actor_loss": 0.42,
                "train/qf1_loss": 1.7,
                "train/qf2_loss": 1.8,
                "train/entropy_coef": 0.05,
                "train/learning_rate": 3e-4,
            },
        })(),
    })()

    # _try_init() reads WANDB_API_KEY + WANDB_MODE and refuses to log to
    # the wrong entity if the key is missing.
    if not cb._try_init():
        print("[wandb] init failed — see WandbLogger._init_error above. "
              "Refusing to log to the wrong account.", flush=True)
        return 1

    # Fake training loop — synthesize the same metric names the SAC
    # trainer will emit so we can verify the grouping in the W&B UI.
    for step in range(args.steps):
        payload = {
            "train/global_step": step,
            "reward/episode": 0.5 + 0.01 * step,
            "reward/mean": 0.5 + 0.005 * step,
            "loss/policy": 0.42 / (step + 1),
            "loss/critic": 1.7 / (step + 1),
            "loss/entropy_coef": 0.05,
            "train/learning_rate": 3e-4,
        }
        try:
            cb._wandb.log(payload, step=step)
        except Exception as exc:
            print(f"[wandb] log() raised: {exc}", flush=True)
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"[wandb] logged step {step}", flush=True)

    cb.finish()
    print("[wandb] finish() ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())