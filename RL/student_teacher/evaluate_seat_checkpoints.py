"""Evaluate SB3 or actor-only seat checkpoints on one frozen reset suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from RL.student_teacher.train_seat import (
    checkpoint_selection_score,
    evaluate_seat,
)


class TorchScriptSeatModel:
    def __init__(self, path: Path):
        import torch

        self._torch = torch
        self.path = path
        self.actor = torch.jit.load(str(path), map_location="cpu").eval()
        contract_path = path.with_suffix(path.suffix + ".contract.json")
        self.contract = (
            json.loads(contract_path.read_text()) if contract_path.is_file()
            else None)
        if self.contract is not None:
            if self.contract.get("input_shape") != [8, 34]:
                raise ValueError(
                    f"{contract_path} has incompatible input_shape "
                    f"{self.contract.get('input_shape')}")
            if self.contract.get("output_shape") != [6]:
                raise ValueError(
                    f"{contract_path} has incompatible output_shape "
                    f"{self.contract.get('output_shape')}")

    def predict(self, observation, deterministic=True):
        del deterministic
        actor_history = np.asarray(observation["actor"], dtype=np.float32)
        if actor_history.shape != (8, 34):
            raise ValueError(f"actor history shape drifted: {actor_history.shape}")
        with self._torch.no_grad():
            action = self.actor(self._torch.from_numpy(actor_history))
        action = action.detach().cpu().numpy().astype(np.float32)
        if action.shape != (6,) or not bool(np.all(np.isfinite(action))):
            raise ValueError(f"invalid actor output from {self.path}: {action}")
        return action, None


def _load_model(path: Path):
    if path.suffix == ".ts":
        return TorchScriptSeatModel(path), "torchscript"
    if path.suffix == ".zip":
        from stable_baselines3 import SAC

        return SAC.load(path, device="auto"), "sb3_sac"
    raise ValueError(f"unsupported checkpoint extension: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--seed", type=int, default=190_000)
    args = parser.parse_args()
    for path in args.checkpoint:
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for path in args.checkpoint:
        model, checkpoint_type = _load_model(path)
        metrics, traces, outcomes = evaluate_seat(
            model, stage="deployment", episodes=args.episodes, seed=args.seed)
        result = {
            "checkpoint": str(path.resolve()),
            "checkpoint_type": checkpoint_type,
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "selection_score": list(checkpoint_selection_score(metrics)),
            "metrics": metrics,
            "outcomes": outcomes,
            "traces": traces,
        }
        results.append(result)
        print(json.dumps({key: value for key, value in result.items()
                          if key != "traces"}, indent=2), flush=True)

    ranked = sorted(
        results,
        key=lambda result: tuple(result["selection_score"]),
        reverse=True,
    )
    payload = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "ranking": ranked,
        "best_checkpoint": ranked[0]["checkpoint"],
    }
    (args.out / "checkpoint_evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    (args.out / "best_checkpoint.json").write_text(
        json.dumps(ranked[0], indent=2) + "\n")


if __name__ == "__main__":
    main()
