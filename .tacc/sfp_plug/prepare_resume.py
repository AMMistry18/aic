#!/usr/bin/env python3
"""Create a compact SFP collection config for unfinished trials."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-trial", type=int, required=True)
    return parser.parse_args()


def trial_number(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unexpected trial name: {name}") from exc


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    trials = payload.get("trials", {})
    payload["trials"] = {
        name: trial
        for name, trial in trials.items()
        if trial_number(name) >= args.start_trial
    }
    if not payload["trials"]:
        raise RuntimeError(f"no trials remain at or after {args.start_trial}")

    # The pose collector does not consume evaluation bags. Keep only the TF
    # topics required by aic_engine's scoring-readiness check; the batch job
    # removes completed temporary bags continuously.
    payload.setdefault("scoring", {})["topics"] = [
        {
            "topic": {
                "name": "/tf",
                "type": "tf2_msgs/msg/TFMessage",
            }
        },
        {
            "topic": {
                "name": "/tf_static",
                "type": "tf2_msgs/msg/TFMessage",
                "latched": True,
            }
        },
        {
            "topic": {
                "name": "/scoring/tf",
                "type": "tf2_msgs/msg/TFMessage",
            }
        },
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} with {len(payload['trials'])} trials "
        f"starting at {args.start_trial}"
    )


if __name__ == "__main__":
    main()
