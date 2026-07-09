"""Distillation dataset generation for the deployable student (Phase-3).

Roll out the FROZEN PRIVILEGED teacher (funnel scripted base + learned residual,
loaded from ``teacher_level1.zip``) in the student env and, at each step, record a
pair:

    (student_deployable_obs,  teacher_EFFECTIVE_cartesian_action)

The BC TARGET is the EFFECTIVE action -- exactly what gets applied to
``env.step`` inside the teacher stack:

    priv       = scene._privileged_obs()                  # 21-D GT (teacher input)
    base       = funnel_teacher.act(scene, priv)          # scripted funnel base
    residual,_ = teacher.predict(priv, deterministic=True)
    effective  = clip(base + residual * residual_scale, -1, 1)   # <-- BC target

We log the COMBINED (base + residual) motion, not the residual alone, because at
deploy there is NO scripted base: the student must reproduce the TOTAL cartesian
command. This mirrors `ResidualTeacherWrapper.step` (which computes the same
`effective` internally) -- we re-derive it here so we can also capture the
student's deployable obs (image/ft/tcp/perceived-port), which the privileged
residual env does not render.

Data is written as sharded ``.npz`` files under ``<out>/dataset/``:
    image  (S, H, W, 3*n_cams) uint8
    vec    (S, 29)             float32   (force,tcp_pose,port_pose,tcp_pose_err,last_action)
    action (S, 6)              float32   effective cartesian residual (BC target)

Parallelism: N worker processes each roll out a share of the transitions on their
own env + teacher (CPU-only inference, so no GPU contention with evals).

DAgger-ready: `rollout_and_collect(..., student_policy=<callable>)` drives the env
with the STUDENT's action but still relabels the TARGET with the teacher's
effective action -- the exact aggregation step a later DAgger loop needs.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("MUJOCO_GL", os.environ.get("AIC_MUJOCO_GL") or "egl")

import numpy as np

TEACHER_ZIP_DEFAULT = "RL/student_teacher/weights/teacher_level1.zip"
RESIDUAL_SCALE_DEFAULT = 0.15


# --------------------------------------------------------------------------- #
# teacher effective-action helper
# --------------------------------------------------------------------------- #
def _load_teacher(zip_path: str):
    from stable_baselines3 import SAC
    # CPU-only: keep the GPU free for the concurrent recon evals.
    return SAC.load(zip_path, device="cpu")


def teacher_effective_action(env, funnel, teacher, residual_scale: float) -> np.ndarray:
    """funnel base + learned residual, combined and clipped -> the applied action."""
    scene = env.scene
    priv = env.priv_obs()
    base = np.asarray(funnel.act(scene, priv), dtype=np.float64).reshape(-1)
    residual, _ = teacher.predict(priv, deterministic=True)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    eff = np.clip(base + residual * float(residual_scale), -1.0, 1.0)
    return eff.astype(np.float32)


# --------------------------------------------------------------------------- #
# one-process rollout + shard writer
# --------------------------------------------------------------------------- #
def rollout_and_collect(*, worker: int, n_transitions: int, out_dir: str,
                        perception_noise: float, level: float, image_size: int,
                        residual_scale: float, teacher_zip: str, seed: int,
                        shard_size: int = 5000, student_policy=None):
    """Collect `n_transitions` (obs, effective_action) pairs; write shards.

    student_policy=None  -> BC: the teacher drives the env (teacher on-policy).
    student_policy=fn    -> DAgger: the STUDENT drives the env, target stays the
                            teacher's effective action (relabel/aggregate).
    """
    from RL.student_teacher.student_env import make_student_env, flatten_student_vec
    from RL.student_teacher.scripted_teacher_funnel import ScriptedTeacher

    env = make_student_env(perception_noise=perception_noise, level=level,
                           image_size=image_size, seed=seed)
    funnel = ScriptedTeacher(action_dim=6)
    teacher = _load_teacher(teacher_zip)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    buf_img, buf_vec, buf_act = [], [], []
    shard_k = 0
    collected = 0

    def _flush():
        nonlocal shard_k, buf_img, buf_vec, buf_act
        if not buf_img:
            return
        path = out / f"shard_w{worker:02d}_{shard_k:04d}.npz"
        np.savez_compressed(
            path,
            image=np.asarray(buf_img, dtype=np.uint8),
            vec=np.asarray(buf_vec, dtype=np.float32),
            action=np.asarray(buf_act, dtype=np.float32),
        )
        shard_k += 1
        buf_img, buf_vec, buf_act = [], [], []

    obs, _ = env.reset(seed=seed)
    funnel.reset()
    while collected < n_transitions:
        effective = teacher_effective_action(env, funnel, teacher, residual_scale)

        # record (student obs now, teacher effective action as BC target)
        buf_img.append(np.asarray(obs["image"], dtype=np.uint8))
        buf_vec.append(flatten_student_vec(obs))
        buf_act.append(effective)
        collected += 1

        # choose which action drives the env
        if student_policy is None:
            step_action = effective                    # BC: teacher on-policy
        else:
            step_action = np.asarray(student_policy(obs), dtype=np.float32).reshape(6)

        obs, _r, term, trunc, _info = env.step(step_action)

        if len(buf_img) >= shard_size:
            _flush()
        if term or trunc:
            obs, _ = env.reset()
            funnel.reset()

    _flush()
    env.close()
    return collected


def _worker_entry(kwargs):
    return rollout_and_collect(**kwargs)


# --------------------------------------------------------------------------- #
# parallel generation
# --------------------------------------------------------------------------- #
def generate(*, transitions: int, num_envs: int, out: str,
             perception_noise: float, level: float, image_size: int,
             residual_scale: float, teacher_zip: str, seed: int,
             shard_size: int = 5000, student_policy=None) -> str:
    """Generate `transitions` pairs across `num_envs` workers -> <out>/dataset/."""
    dataset_dir = Path(out) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # DAgger (student_policy given) is single-process: a torch policy does not
    # fork cleanly and CUDA must not be forked. BC parallelises freely (CPU).
    if student_policy is not None:
        num_envs = 1

    per = [transitions // num_envs] * num_envs
    for i in range(transitions % num_envs):
        per[i] += 1

    common = dict(out_dir=str(dataset_dir), perception_noise=perception_noise,
                  level=level, image_size=image_size, residual_scale=residual_scale,
                  teacher_zip=teacher_zip, shard_size=shard_size,
                  student_policy=student_policy)

    t0 = time.time()
    if num_envs == 1:
        rollout_and_collect(worker=0, n_transitions=per[0], seed=seed, **common)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")     # matches residual_env's proven approach
        procs = []
        for w in range(num_envs):
            kw = dict(worker=w, n_transitions=per[w], seed=seed + 1000 * w, **common)
            p = ctx.Process(target=_worker_entry, args=(kw,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        fail = [p.exitcode for p in procs if p.exitcode not in (0, None)]
        if fail:
            raise RuntimeError(f"dataset workers failed with exitcodes {fail}")

    shards = sorted(dataset_dir.glob("shard_*.npz"))
    print(f"[distill] generated ~{transitions} transitions across {num_envs} "
          f"workers in {time.time() - t0:.1f}s -> {len(shards)} shards in {dataset_dir}",
          flush=True)
    return str(dataset_dir)


# --------------------------------------------------------------------------- #
# CLI (standalone dataset build)
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=50000)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--out", type=str, default="RL/output/student_teacher/student_distill")
    p.add_argument("--perception-noise", type=float, default=1.0)
    p.add_argument("--level", type=float, default=1.0)
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE_DEFAULT)
    p.add_argument("--teacher-zip", type=str, default=TEACHER_ZIP_DEFAULT)
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    generate(transitions=a.transitions, num_envs=a.num_envs, out=a.out,
             perception_noise=a.perception_noise, level=a.level,
             image_size=a.image_size, residual_scale=a.residual_scale,
             teacher_zip=a.teacher_zip, seed=a.seed, shard_size=a.shard_size)


if __name__ == "__main__":
    main()
