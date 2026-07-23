"""Collect replay-complete Student-v3 priors from preserved controllers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from RL.student_teacher.parity.evaluate_guided_controller import guided_action
from RL.student_teacher.student_env_a import DEPLOY_POS_SCALE, DEPLOY_ROT_SCALE
from RL.student_teacher.student_v3_env import StudentV3Env, make_student_v3_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--teacher-zip", type=Path,
                   default=Path("RL/student_teacher/weights/teacher_level1.zip"))
    p.add_argument("--old-student", type=Path,
                   default=Path("models/final_insert_sfp_flowstate_v1.ts"))
    p.add_argument("--teacher-transitions", type=int, default=30_000)
    p.add_argument("--student-transitions", type=int, default=15_000)
    p.add_argument("--failure-transitions", type=int, default=15_000)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--shard-size", type=int, default=20_000)
    p.add_argument("--max-episodes-per-source", type=int, default=5_000)
    return p.parse_args()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _full_action_to_residual(env: StudentV3Env, full_deploy: np.ndarray) -> np.ndarray:
    obs69 = env._current_obs69
    guide = np.asarray(guided_action(obs69), dtype=np.float64)
    full = np.clip(np.asarray(full_deploy, dtype=np.float64).reshape(6), -1.0, 1.0)
    desired_pos = np.clip(
        full[:3] * DEPLOY_POS_SCALE - guide[:3] * DEPLOY_POS_SCALE,
        -env._ACCUM_POS, env._ACCUM_POS)
    desired_rot = np.clip(
        full[3:] * DEPLOY_ROT_SCALE - guide[3:] * DEPLOY_ROT_SCALE,
        -env._ACCUM_ROT, env._ACCUM_ROT)
    delta_pos = desired_pos - env._residual_pos
    delta_rot = desired_rot - env._residual_rot
    return np.clip(np.concatenate([
        delta_pos / env._STEP_POS,
        delta_rot / env._STEP_ROT,
    ]), -1.0, 1.0).astype(np.float32)


def _episode_rows(env: StudentV3Env, action_fn, seed: int):
    obs, _ = env.reset(seed=seed)
    rows, prior_stall = [], False
    while True:
        action = np.asarray(action_fn(env), dtype=np.float32).reshape(6)
        next_obs, reward, term, trunc, info = env.step(action)
        stalled = bool(info.get("v3_stalled", False))
        category = 2 if (prior_stall or stalled) else 1
        rows.append({
            "obs_actor": obs["actor"].copy(),
            "obs_privileged": obs["privileged"].copy(),
            "actions": action.copy(),
            "rewards": np.float32(reward),
            "next_actor": next_obs["actor"].copy(),
            "next_privileged": next_obs["privileged"].copy(),
            "dones": np.float32(term or trunc),
            "category": np.int8(category),
            "bc_mask": np.int8(1),
        })
        prior_stall = prior_stall or stalled
        obs = next_obs
        if term or trunc:
            return rows, str(info.get("term_status") or "timeout"), info


def _collect_source(name, target, seed, max_episodes, controller_builder):
    domain_randomization = name == "failure_boundary"
    env = make_student_v3_env(
        seed=seed, stage="contact", domain_randomization=domain_randomization)
    action_fn, reset_fn = controller_builder(env)
    accepted, episodes = [], 0
    outcomes: dict[str, int] = {}
    try:
        while len(accepted) < target and episodes < max_episodes:
            if reset_fn is not None:
                reset_fn()
            rows, status, final_info = _episode_rows(env, action_fn, seed + episodes)
            outcomes[status] = outcomes.get(status, 0) + 1
            keep = status == "success" if name != "failure_boundary" else status != "success"
            if keep:
                if name == "failure_boundary":
                    for row in rows:
                        row["category"] = np.int8(0)
                        row["bc_mask"] = np.int8(0)
                accepted.extend(rows)
            episodes += 1
    finally:
        env.close()
    if len(accepted) < target:
        raise RuntimeError(
            f"{name}: collected {len(accepted)}/{target} after {episodes} episodes")
    return accepted[:target], {"episodes": episodes, "outcomes": outcomes}


def _teacher_builder(path: Path):
    from stable_baselines3 import SAC
    from RL.student_teacher.scripted_teacher_funnel import ScriptedTeacher
    from RL.student_teacher.train_student_a import (
        RESIDUAL_SCALE_DEFAULT, _teacher_target)
    teacher = SAC.load(path, device="cpu")

    def build(env):
        funnel = ScriptedTeacher(action_dim=6)

        def action_fn(v3):
            _sim, deploy = _teacher_target(
                v3.contract_env, funnel, teacher,
                RESIDUAL_SCALE_DEFAULT, "deploy")
            return _full_action_to_residual(v3, deploy)

        return action_fn, funnel.reset
    return build


def _student_builder(path: Path):
    import torch
    student = torch.jit.load(str(path), map_location="cpu").eval()

    def build(env):
        def action_fn(v3):
            with torch.no_grad():
                full = student(torch.as_tensor(
                    v3._current_obs69, dtype=torch.float32)).cpu().numpy()
            return _full_action_to_residual(v3, full)
        return action_fn, None
    return build


def _failure_builder(env):
    # Zero residual means the deterministic guided base drives into the
    # randomized failure boundary. These rows teach Q, never behavior cloning.
    return lambda _env: np.zeros(6, dtype=np.float32), None


def _write_shards(out: Path, rows: list[dict], shard_size: int):
    out.mkdir(parents=True, exist_ok=True)
    shards = []
    for index, start in enumerate(range(0, len(rows), shard_size)):
        chunk = rows[start:start + shard_size]
        arrays = {key: np.asarray([row[key] for row in chunk]) for key in chunk[0]}
        path = out / f"prior_{index:04d}.npz"
        np.savez_compressed(path, **arrays)
        shards.append({"path": path.name, "rows": len(chunk), "sha256": _sha(path)})
    return shards


def main():
    args = parse_args()
    if _sha(args.teacher_zip) != "fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4":
        raise SystemExit("frozen teacher hash mismatch")
    sources = [
        ("teacher_success", args.teacher_transitions, args.seed,
         _teacher_builder(args.teacher_zip)),
        ("old_student_success", args.student_transitions, args.seed + 100_000,
         _student_builder(args.old_student)),
        ("failure_boundary", args.failure_transitions, args.seed + 200_000,
         _failure_builder),
    ]
    all_rows, reports = [], {}
    for name, target, seed, builder in sources:
        rows, report = _collect_source(
            name, target, seed, args.max_episodes_per_source, builder)
        all_rows.extend(rows)
        reports[name] = {**report, "rows": len(rows)}
        print(name, reports[name], flush=True)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_rows)
    shards = _write_shards(args.out, all_rows, args.shard_size)
    manifest = {
        "schema": "student_v3_prior_v1",
        "rows": len(all_rows),
        "sampling_seed": args.seed,
        "teacher_sha256": _sha(args.teacher_zip),
        "old_student_sha256": _sha(args.old_student),
        "sources": reports,
        "shards": shards,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
