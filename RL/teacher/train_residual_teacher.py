"""Train a residual RL policy on top of the scripted FUNNEL teacher (option b).

SAC + MLP policy over the PRIVILEGED flat-state obs, fixed curriculum level 1.0
(no curriculum advancement). The policy outputs a small residual that is added to
the funnel teacher's action inside ResidualTeacherWrapper. Writes checkpoints /
metrics ONLY under --out (default RL/output/teacher/residual_funnel_lvl1).

Does NOT touch the impedance controller (cart_kp_pos=100, cart_max_wrench=10 N).

Example (FULL run -- launch this yourself; the smoke uses tiny values):
  WANDB_MODE=offline .pixi/envs/default/bin/python -m RL.teacher.train_residual_teacher \
      --steps 150000 --num-envs 4 --out RL/output/teacher/residual_funnel_lvl1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("MUJOCO_GL", os.environ.get("AIC_MUJOCO_GL") or "egl")

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--out", type=str, default="RL/output/teacher/residual_funnel_lvl1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", type=float, default=1.0)
    p.add_argument("--residual-scale", type=float, default=0.15)
    # SAC hyperparams (sensible reuse of train.py)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=300000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=4)
    p.add_argument("--target-entropy", type=float, default=-3.0)
    p.add_argument("--ckpt-every", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=20000)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-only", type=str, default=None,
                   help="path to a trained model.zip: skip ALL training, just load "
                        "it and certify --eval-episodes at --level (deterministic).")
    p.add_argument("--obs-noise", type=float, default=0.0,
                   help="[--eval-only] scale a per-episode CONSISTENT perception bias "
                        "added to ONLY the policy's geometric obs dims [0:9] "
                        "(lateral/axial/axis/roll/depth/tip-vel); wrench+qvel [9:21] "
                        "stay clean. Does NOT affect physics, the funnel base action "
                        "(reads GT), or success. 0 = off.")
    return p.parse_args()


TERM_BUCKETS = ["success", "bad_collision", "force_abort", "off_limit", "timeout", "other"]

# Base 1-sigma perception error on the geometric obs dims [0:9] (scaled by --obs-noise).
# Dims [9:21] (6-axis FT wrench + arm_qvel) are measured accurately -> left CLEAN.
#   [0:2] lateral offset (m)   [2] axial (m)   [3] axis err (rad)   [4] roll err (rad)
#   [5] depth_norm             [6:9] tip linear velocity (m/s)
OBS_NOISE_BASE_SIGMA = np.array(
    [1.0e-3, 1.0e-3, 1.0e-3, 1.745e-2, 1.745e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3],
    dtype=np.float64)   # ~1mm lateral/axial, ~1deg axis/roll, 0.01 depth, 5mm/s vel


def _certify(model_path, episodes, seed, residual_scale, level, obs_noise=0.0):
    """Measurement-only certification: load the residual policy, run it in the SAME
    wrapper env used in training (level pinned to `level`), read term_status.

    obs_noise>0 adds a PER-EPISODE-CONSISTENT bias (drawn once per episode, held
    fixed) to ONLY the policy's geometric obs dims [0:9]; the env, the funnel base
    action (which reads exact GT, not obs), and success determination are untouched.
    => this is a LOWER BOUND on real perception degradation: the scripted base still
    runs on the exact answer key even when the policy's view is corrupted.
    """
    from stable_baselines3 import SAC
    from RL.teacher.residual_env import make_residual_teacher_env

    env = make_residual_teacher_env(residual_scale=residual_scale, level=level)
    model = SAC.load(model_path)

    sigma = OBS_NOISE_BASE_SIGMA * float(obs_noise)   # per-dim 1-sigma for dims [0:9]
    rng = np.random.default_rng(int(seed))
    obs_dim = int(np.asarray(env.observation_space.shape).prod())

    def _ep_bias():
        b = np.zeros(obs_dim, dtype=np.float64)
        if obs_noise > 0.0:
            b[:sigma.shape[0]] = rng.normal(0.0, sigma)   # geometric dims only
        return b

    counts = {k: 0 for k in TERM_BUCKETS}
    n_succ = 0
    score_succ = 0
    score_seen = 0
    obs, _ = env.reset(seed=int(seed))
    bias = _ep_bias()
    done_eps = 0
    while done_eps < int(episodes):
        obs_policy = (np.asarray(obs, dtype=np.float64) + bias).astype(np.float32)
        action, _ = model.predict(obs_policy, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        if term or trunc:
            ts = info.get("term_status")
            bucket = ts if ts in counts else "other"
            counts[bucket] += 1
            if ts == "success":
                n_succ += 1
            sc = info.get("score_success")
            if sc is not None:
                score_seen += 1
                score_succ += int(bool(sc))
            done_eps += 1
            obs, _ = env.reset()
            bias = _ep_bias()          # new per-episode consistent perception bias
    env.close()

    total = int(episodes)
    sr = n_succ / max(1, total)
    print("")
    print("=" * 56)
    print(" RESIDUAL-TEACHER CERTIFY (measurement only)")
    print("=" * 56)
    print(f" model          : {model_path}")
    print(f" level (pinned) : {level}")
    print(f" residual_scale : {residual_scale}")
    print(f" seed           : {seed}")
    print(f" obs_noise      : {obs_noise}")
    if obs_noise > 0.0:
        print(f" obs bias sigmas (dims[0:9], per-episode): "
              f"lat={sigma[0]*1e3:.2f}mm axial={sigma[2]*1e3:.2f}mm "
              f"axis={np.degrees(sigma[3]):.2f}deg roll={np.degrees(sigma[4]):.2f}deg "
              f"depth={sigma[5]:.3f} vel={sigma[6]*1e3:.1f}mm/s  (dims[9:21] CLEAN)")
        print(" caveat: LOWER BOUND -- funnel base action still uses exact GT; "
              "only the policy's obs is corrupted.")
    print("-" * 56)
    print(f" total episodes : {total}")
    print(f" success_rate   : {sr:.4f}  ({100.0 * sr:.2f}%)")
    print("-" * 56)
    print(" term_status breakdown:")
    for k in TERM_BUCKETS:
        c = counts[k]
        print(f"   {k:<14}: {c:>5d}  ({100.0 * c / max(1, total):6.2f}%)")
    print("-" * 56)
    if score_seen:
        print(f" score_success (cross-check): {score_succ}/{score_seen} "
              f"({100.0 * score_succ / score_seen:.2f}%)")
    print("=" * 56)
    return 0


def eval_success(make_env, action_fn, episodes, seed=0):
    """Roll out `episodes` on one env; success from info['term_status']."""
    env = make_env(seed=seed)
    obs, _ = env.reset(seed=seed)
    counts, n_succ, done_eps = {}, 0, 0
    while done_eps < episodes:
        a = action_fn(obs)
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            ts = str(info.get("term_status"))
            counts[ts] = counts.get(ts, 0) + 1
            n_succ += int(ts == "success")
            done_eps += 1
            obs, _ = env.reset()
    env.close()
    return n_succ / max(1, episodes), counts


def main():
    args = parse_args()

    # ---- eval-only: no training, no writes; just certify a saved model ----
    if args.eval_only:
        return _certify(args.eval_only, args.eval_episodes, args.seed,
                        args.residual_scale, args.level, args.obs_noise)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: F401
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    from RL.teacher.residual_env import make_residual_teacher_env

    def make_env(seed=None):
        return make_residual_teacher_env(
            residual_scale=args.residual_scale, level=args.level, seed=seed)

    def _thunk(rank):
        def _f():
            return Monitor(make_env())
        return _f

    n = int(args.num_envs)
    if n > 1:
        venv = SubprocVecEnv([_thunk(i) for i in range(n)], start_method="fork")
    else:
        venv = DummyVecEnv([_thunk(0)])
    venv.seed(int(args.seed))

    print(f"[residual] obs_space={venv.observation_space} act_space={venv.action_space} "
          f"num_envs={n} residual_scale={args.residual_scale} level={args.level}", flush=True)

    # --- SANITY: residual==0 must reproduce the funnel teacher (~70%) ---
    zero_sr, zero_counts = eval_success(
        make_env, lambda o: np.zeros(venv.action_space.shape[0], np.float32),
        args.eval_episodes, seed=1234)
    print(f"[residual] SANITY zero-residual success_rate = {zero_sr:.3f}  {zero_counts}",
          flush=True)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    model = SAC(
        "MlpPolicy", venv,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.warmup,
        tau=args.tau,
        gamma=args.gamma,
        ent_coef="auto",
        target_entropy=args.target_entropy,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        seed=int(args.seed),
        verbose=1,
    )

    # untrained-policy eval at step 0 (should be ~near the zero-residual base since
    # the untrained residual is small-ish noise on top of the teacher)
    sr0, c0 = eval_success(
        make_env, lambda o: model.predict(o, deterministic=True)[0],
        args.eval_episodes, seed=4321)
    print(f"[residual] step0 untrained-policy success_rate = {sr0:.3f}  {c0}", flush=True)

    metrics_path = out / "metrics.jsonl"
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"event": "sanity", "zero_residual_sr": zero_sr,
                            "zero_counts": zero_counts, "step0_policy_sr": sr0}) + "\n")

    class EvalCb(BaseCallback):
        def __init__(self, every, episodes):
            super().__init__()
            self.every = int(every); self.episodes = int(episodes); self._last = 0

        def _on_step(self):
            if self.num_timesteps - self._last >= self.every:
                self._last = self.num_timesteps
                sr, counts = eval_success(
                    make_env,
                    lambda o: self.model.predict(o, deterministic=True)[0],
                    self.episodes, seed=4321)
                print(f"[residual] step={self.num_timesteps} eval success_rate={sr:.3f} {counts}",
                      flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"event": "eval", "step": int(self.num_timesteps),
                                        "success_rate": sr, "counts": counts}) + "\n")
            return True

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.ckpt_every // n), save_path=str(out), name_prefix="model")
    eval_cb = EvalCb(args.eval_every, args.eval_episodes)

    t0 = time.time()
    model.learn(total_timesteps=int(args.steps), callback=[ckpt_cb, eval_cb],
                progress_bar=False)
    model.save(str(out / "model.zip"))
    print(f"[residual] DONE steps={args.steps} elapsed={time.time()-t0:.1f}s "
          f"-> {out/'model.zip'}", flush=True)

    with open(out / "train_meta.txt", "w") as f:
        f.write(f"residual_scale={args.residual_scale}\nlevel={args.level}\n"
                f"steps={args.steps}\nnum_envs={n}\nseed={args.seed}\n"
                f"lr={args.lr}\nbuffer_size={args.buffer_size}\nbatch_size={args.batch_size}\n"
                f"warmup={args.warmup}\ntrain_freq={args.train_freq}\n"
                f"gradient_steps={args.gradient_steps}\ntarget_entropy={args.target_entropy}\n")
    venv.close()


if __name__ == "__main__":
    main()
