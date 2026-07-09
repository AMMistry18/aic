"""Behavior-cloning trainer for the Contract-A STATE student (Phase-3 distillation).

Trains a small MLP over the EXACT 69-dim deploy obs (``student_env_a.py``) to
imitate the FROZEN privileged teacher's EFFECTIVE 6-D cartesian action, expressed
in the DEPLOY action convention (POS/ROT_SCALE). No CNN, no images -- the deploy
final-insert policy is a pure MLP over the 69-vector.

    priv       = scene._privileged_obs()                  # 21-D GT (teacher input)
    base       = funnel_teacher.act(scene, priv)          # scripted funnel base
    residual,_ = teacher.predict(priv, deterministic=True)
    eff_sim    = clip(base + residual * residual_scale, -1, 1)     # sim convention
    target     = sim_to_deploy_action(eff_sim)                     # <-- BC target

BC by MSE on tanh-bounded 6-D actions. Periodically certifies the student's
success_rate at level 1.0 on the deploy obs (via ``info['term_status']``).

Writes ONLY under --out:
    <out>/dataset/            teacher rollout shards (regenerated if missing/--regen)
    <out>/student_a.pt        latest student weights + config
    <out>/student_a_ep###.pt  periodic checkpoints
    <out>/metrics.jsonl       train loss + eval success_rate log

DAgger scaffold: --dagger-iters > 0 rolls the CURRENT student on-policy and
relabels the target with the teacher (single-process; torch policy doesn't fork).

Example (FULL run -- launch on the GPU pod):
  WANDB_MODE=offline PYTHONPATH=<repo> MUJOCO_GL=egl \
    python -m RL.student_teacher.train_student_a --transitions 150000 --epochs 40 \
    --num-envs 12 --out RL/output/student_teacher/student_a_distill --seed 0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("MUJOCO_GL", os.environ.get("AIC_MUJOCO_GL") or "egl")

import numpy as np

TEACHER_ZIP_DEFAULT = "RL/student_teacher/weights/teacher_level1.zip"
RESIDUAL_SCALE_DEFAULT = 0.15
OBS_DIM = 69
ACT_DIM = 6


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int, default=150000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--num-envs", type=int, default=12,
                   help="parallel rollout workers for dataset gen (CPU-bound)")
    p.add_argument("--perception-noise", type=float, default=1.0)
    p.add_argument("--level", type=float, default=1.0)
    p.add_argument("--action-convention", type=str, default="deploy",
                   choices=["deploy", "sim"])
    p.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE_DEFAULT)
    p.add_argument("--teacher-zip", type=str, default=TEACHER_ZIP_DEFAULT)
    p.add_argument("--out", type=str, default="RL/output/student_teacher/student_a_distill")
    p.add_argument("--seed", type=int, default=0)
    # optimisation
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256, help="MLP hidden width")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--regen", action="store_true",
                   help="force regenerate the dataset even if shards exist")
    p.add_argument("--shard-size", type=int, default=5000)
    # eval
    p.add_argument("--eval-every", type=int, default=5, help="epochs between evals")
    p.add_argument("--eval-episodes", type=int, default=30)
    # DAgger (scaffold; 0 = pure BC)
    p.add_argument("--dagger-iters", type=int, default=0)
    p.add_argument("--dagger-transitions", type=int, default=30000)
    p.add_argument("--dagger-epochs", type=int, default=10)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# student policy: MLP(69) -> tanh 6-D
# --------------------------------------------------------------------------- #
def build_policy(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
    import torch
    import torch.nn as nn

    class StudentMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, act_dim))

        def forward(self, obs):
            return torch.tanh(self.net(obs))

    return StudentMLP()


def _resolve_device(device: str):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------------------------------------------------------- #
# teacher effective action -> deploy-convention BC target
# --------------------------------------------------------------------------- #
def _teacher_target(env, funnel, teacher, residual_scale, convention):
    from RL.student_teacher.student_env_a import sim_to_deploy_action
    scene = env.scene
    priv = env.priv_obs()
    base = np.asarray(funnel.act(scene, priv), dtype=np.float64).reshape(-1)
    residual, _ = teacher.predict(priv, deterministic=True)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    eff_sim = np.clip(base + residual * float(residual_scale), -1.0, 1.0)
    target = eff_sim if convention == "sim" else sim_to_deploy_action(eff_sim)
    return eff_sim.astype(np.float32), np.asarray(target, dtype=np.float32)


# --------------------------------------------------------------------------- #
# dataset generation (state-only; light CPU env, no GL)
# --------------------------------------------------------------------------- #
def _rollout_worker(*, worker, n_transitions, out_dir, perception_noise, level,
                    action_convention, residual_scale, teacher_zip, seed,
                    shard_size, student_state_dict=None, hidden=256):
    from stable_baselines3 import SAC
    from RL.student_teacher.student_env_a import make_student_env_a
    from RL.student_teacher.scripted_teacher_funnel import ScriptedTeacher

    env = make_student_env_a(perception_noise=perception_noise, level=level,
                             action_convention=action_convention, seed=seed)
    funnel = ScriptedTeacher(action_dim=ACT_DIM)
    teacher = SAC.load(teacher_zip, device="cpu")

    student = None
    if student_state_dict is not None:
        import torch
        student = build_policy(hidden=hidden)
        student.load_state_dict(student_state_dict)
        student.eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buf_obs, buf_act = [], []
    shard_k, collected = 0, 0

    def _flush():
        nonlocal shard_k, buf_obs, buf_act
        if not buf_obs:
            return
        np.savez_compressed(
            out / f"shard_w{worker:02d}_{shard_k:04d}.npz",
            obs=np.asarray(buf_obs, dtype=np.float32),
            action=np.asarray(buf_act, dtype=np.float32))
        shard_k += 1
        buf_obs, buf_act = [], []

    obs, _ = env.reset(seed=seed)
    funnel.reset()
    while collected < n_transitions:
        eff_sim, target = _teacher_target(
            env, funnel, teacher, residual_scale, action_convention)
        buf_obs.append(np.asarray(obs, dtype=np.float32))
        buf_act.append(target)
        collected += 1

        if student is None:                     # BC: teacher drives (on-policy)
            nobs, _r, term, trunc, _ = env.step_sim(eff_sim)
        else:                                    # DAgger: student drives, teacher relabels
            import torch
            with torch.no_grad():
                a = student(torch.as_tensor(obs[None], dtype=torch.float32)
                            ).numpy().reshape(-1)
            nobs, _r, term, trunc, _ = env.step(a)   # action_convention convention
        obs = nobs

        if len(buf_obs) >= shard_size:
            _flush()
        if term or trunc:
            obs, _ = env.reset()
            funnel.reset()
    _flush()
    env.close()
    return collected


def _worker_entry(kw):
    return _rollout_worker(**kw)


def generate(*, transitions, num_envs, out, perception_noise, level,
             action_convention, residual_scale, teacher_zip, seed, shard_size,
             student_state_dict=None, hidden=256):
    dataset_dir = Path(out) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if student_state_dict is not None:       # DAgger is single-process (torch/CUDA)
        num_envs = 1

    per = [transitions // num_envs] * num_envs
    for i in range(transitions % num_envs):
        per[i] += 1
    common = dict(out_dir=str(dataset_dir), perception_noise=perception_noise,
                  level=level, action_convention=action_convention,
                  residual_scale=residual_scale, teacher_zip=teacher_zip,
                  shard_size=shard_size, student_state_dict=student_state_dict,
                  hidden=hidden)

    t0 = time.time()
    if num_envs == 1:
        _rollout_worker(worker=0, n_transitions=per[0], seed=seed, **common)
    else:
        ctx = mp.get_context("fork")
        procs = []
        for w in range(num_envs):
            kw = dict(worker=w, n_transitions=per[w], seed=seed + 1000 * w, **common)
            p = ctx.Process(target=_worker_entry, args=(kw,))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
        fail = [p.exitcode for p in procs if p.exitcode not in (0, None)]
        if fail:
            raise RuntimeError(f"dataset workers failed with exitcodes {fail}")

    shards = sorted(dataset_dir.glob("shard_*.npz"))
    print(f"[student_a] generated ~{transitions} transitions across {num_envs} "
          f"workers in {time.time()-t0:.1f}s -> {len(shards)} shards", flush=True)
    return str(dataset_dir)


def load_dataset(dataset_dir: str):
    shards = sorted(Path(dataset_dir).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {dataset_dir}")
    obs, acts = [], []
    for s in shards:
        z = np.load(s)
        obs.append(z["obs"]); acts.append(z["action"])
    return np.concatenate(obs, 0), np.concatenate(acts, 0)


# --------------------------------------------------------------------------- #
# eval (deploy obs, level 1.0, success from term_status)
# --------------------------------------------------------------------------- #
def eval_student(policy, *, episodes, perception_noise, level, action_convention,
                 seed, device):
    import torch
    from RL.student_teacher.student_env_a import make_student_env_a

    env = make_student_env_a(perception_noise=perception_noise, level=level,
                             action_convention=action_convention, seed=seed)
    policy.eval()

    def act(obs):
        with torch.no_grad():
            t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            return policy(t).cpu().numpy().reshape(-1)

    counts, n_succ, done = {}, 0, 0
    obs, _ = env.reset(seed=seed)
    while done < episodes:
        obs, _r, term, trunc, info = env.step(act(obs))
        if term or trunc:
            ts = str(info.get("term_status"))
            counts[ts] = counts.get(ts, 0) + 1
            n_succ += int(ts == "success")
            done += 1
            obs, _ = env.reset()
    env.close()
    policy.train()
    return n_succ / max(1, episodes), counts


# --------------------------------------------------------------------------- #
# BC fit
# --------------------------------------------------------------------------- #
def _wandb_log(data):
    """Log to wandb if an active run exists; never fatal."""
    try:
        import wandb
        if getattr(wandb, "run", None) is not None:
            wandb.log(data)
    except Exception:
        pass


def fit_bc(policy, dataset_dir, *, epochs, batch_size, lr, device, seed, out,
           metrics_path, eval_every, eval_episodes, perception_noise, level,
           action_convention, name="student_a", epoch_offset=0):
    import torch

    obs, action = load_dataset(dataset_dir)
    n = obs.shape[0]
    print(f"[student_a] dataset: {n} transitions  obs={obs.shape[1]} "
          f"action={action.shape[1]}", flush=True)

    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(action, device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = torch.as_tensor(perm[i:i + batch_size], device=device)
            pred = policy(obs_t[idx])
            loss = loss_fn(pred, act_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        mean_loss = tot / max(1, nb)
        gep = epoch_offset + ep + 1
        print(f"[student_a] epoch {gep} bc_mse={mean_loss:.6f}", flush=True)
        rec = {"event": "train", "epoch": gep, "bc_mse": mean_loss}
        _wandb_log({"bc/loss": mean_loss, "epoch": gep})

        if eval_every and (gep % eval_every == 0 or ep == epochs - 1):
            sr, counts = eval_student(
                policy, episodes=eval_episodes, perception_noise=perception_noise,
                level=level, action_convention=action_convention,
                seed=4321, device=device)
            print(f"[student_a] epoch {gep} eval success_rate={sr:.3f} {counts}",
                  flush=True)
            rec.update({"eval_success_rate": sr, "eval_counts": counts})
            _wandb_log({"eval/success_rate": sr, "epoch": gep})
            torch.save({"model": policy.state_dict(), "epoch": gep},
                       Path(out) / f"{name}_ep{gep:03d}.pt")

        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    torch.save({"model": policy.state_dict()}, Path(out) / f"{name}.pt")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dataset_dir = str(out / "dataset")
    metrics_path = out / "metrics.jsonl"

    # 1) dataset -- BEFORE any CUDA init (fork-based workers)
    have = bool(sorted(Path(dataset_dir).glob("shard_*.npz")))
    if args.regen or not have:
        generate(transitions=args.transitions, num_envs=args.num_envs, out=args.out,
                 perception_noise=args.perception_noise, level=args.level,
                 action_convention=args.action_convention,
                 residual_scale=args.residual_scale, teacher_zip=args.teacher_zip,
                 seed=args.seed, shard_size=args.shard_size, hidden=args.hidden)
    else:
        print(f"[student_a] reusing dataset in {dataset_dir} (pass --regen)", flush=True)

    # 1b) init wandb (main process only; AFTER fork-based dataset workers)
    try:
        import wandb
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "aic-contractA-distill"),
            name=os.environ.get("WANDB_RUN_NAME", f"student_a_seed{args.seed}"),
            config=vars(args))
        if run is not None and getattr(run, "url", None):
            print(f"[student_a] WANDB_URL {run.url}", flush=True)
    except Exception as e:
        print(f"[student_a] wandb init failed ({e}); continuing without wandb",
              flush=True)

    # 2) build policy
    import torch  # noqa: F401
    device = _resolve_device(args.device)
    policy = build_policy(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[student_a] MLP on {device}: obs={OBS_DIM} -> act={ACT_DIM} "
          f"hidden={args.hidden} params={n_params/1e6:.3f}M", flush=True)

    with open(out / "train_meta.txt", "w") as f:
        f.write(f"transitions={args.transitions}\nepochs={args.epochs}\n"
                f"num_envs={args.num_envs}\nperception_noise={args.perception_noise}\n"
                f"action_convention={args.action_convention}\n"
                f"residual_scale={args.residual_scale}\nteacher_zip={args.teacher_zip}\n"
                f"seed={args.seed}\nbatch_size={args.batch_size}\nlr={args.lr}\n"
                f"hidden={args.hidden}\nobs_dim={OBS_DIM}\nparams={n_params}\n")

    # 3) BC fit
    t0 = time.time()
    fit_bc(policy, dataset_dir, epochs=args.epochs, batch_size=args.batch_size,
           lr=args.lr, device=device, seed=args.seed, out=args.out,
           metrics_path=metrics_path, eval_every=args.eval_every,
           eval_episodes=args.eval_episodes, perception_noise=args.perception_noise,
           level=args.level, action_convention=args.action_convention)

    # 4) optional DAgger rounds (default 0 = pure BC)
    for r in range(int(args.dagger_iters)):
        print(f"[student_a] === DAgger round {r+1}/{args.dagger_iters} ===", flush=True)
        sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        generate(transitions=args.dagger_transitions, num_envs=1, out=args.out,
                 perception_noise=args.perception_noise, level=args.level,
                 action_convention=args.action_convention,
                 residual_scale=args.residual_scale, teacher_zip=args.teacher_zip,
                 seed=args.seed + 7000 + r, shard_size=args.shard_size,
                 student_state_dict=sd, hidden=args.hidden)
        fit_bc(policy, dataset_dir, epochs=args.dagger_epochs,
               batch_size=args.batch_size, lr=args.lr, device=device,
               seed=args.seed, out=args.out, metrics_path=metrics_path,
               eval_every=args.eval_every, eval_episodes=args.eval_episodes,
               perception_noise=args.perception_noise, level=args.level,
               action_convention=args.action_convention,
               epoch_offset=args.epochs + r * args.dagger_epochs)

    print(f"[student_a] DONE elapsed={time.time()-t0:.1f}s -> {out/'student_a.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
