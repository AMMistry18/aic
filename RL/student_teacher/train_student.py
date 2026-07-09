"""Behavior-cloning trainer for the DEPLOYABLE student (Phase-3 distillation).

Trains a small CNN(image) + MLP(vector) policy to imitate the FROZEN privileged
teacher's EFFECTIVE 6-D cartesian residual (see `distill_dataset.py`). BC by MSE
on tanh-bounded actions. Periodically certifies the student's success_rate at
level 1.0 on the DEPLOYABLE obs (via `info['term_status']`).

Writes ONLY under --out:
    <out>/dataset/            teacher rollout shards (regenerated if missing/--regen)
    <out>/student.pt          latest student weights + config
    <out>/student_ep###.pt    periodic checkpoints
    <out>/metrics.jsonl       train loss + eval success_rate log

Structured for a later DAgger loop: `dagger_round()` regenerates data with the
CURRENT student on-policy (teacher relabels the target) and appends new shards --
call it in a loop after the initial BC fit. BC is the default (dagger-iters 0).

Example (FULL run -- launch yourself once the GPU frees):
  WANDB_MODE=offline PYTHONPATH=/home/Anshul/AIC_Phase_1/aic_iso \
    /home/Anshul/AIC_Phase_1/aic_0/aic/.pixi/envs/default/bin/python \
    -m RL.student_teacher.train_student --transitions 150000 --epochs 40 --num-envs 12 \
    --out RL/output/student_teacher/student_distill --seed 0
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

from RL.student_teacher import distill_dataset
from RL.student_teacher.distill_dataset import RESIDUAL_SCALE_DEFAULT, TEACHER_ZIP_DEFAULT


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
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE_DEFAULT)
    p.add_argument("--teacher-zip", type=str, default=TEACHER_ZIP_DEFAULT)
    p.add_argument("--out", type=str, default="RL/output/student_teacher/student_distill")
    p.add_argument("--seed", type=int, default=0)
    # optimisation
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--feat", type=int, default=128, help="CNN/MLP feature width")
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
# student policy
# --------------------------------------------------------------------------- #
def build_policy(img_shape, vec_dim, act_dim=6, feat=128):
    import torch
    import torch.nn as nn

    class StudentPolicy(nn.Module):
        """small CNN(image) + MLP(vector) -> tanh 6-D cartesian residual."""

        def __init__(self):
            super().__init__()
            c, h, w = img_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
            self.img_fc = nn.Sequential(nn.Linear(n_flat, feat), nn.ReLU())
            self.vec_fc = nn.Sequential(
                nn.Linear(vec_dim, feat), nn.ReLU(),
                nn.Linear(feat, feat), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(2 * feat, feat), nn.ReLU(),
                nn.Linear(feat, act_dim))

        def forward(self, img, vec):
            # img: (B,C,H,W) uint8/float in [0,255]; vec: (B,vec_dim)
            f = self.img_fc(self.conv(img.float() / 255.0))
            g = self.vec_fc(vec)
            return torch.tanh(self.head(torch.cat([f, g], dim=1)))

    return StudentPolicy()


def _resolve_device(device: str):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------------------------------------------------------- #
# dataset loading
# --------------------------------------------------------------------------- #
def load_dataset(dataset_dir: str):
    """Load all shards into memory. Returns (image uint8 NHWC, vec f32, action f32)."""
    shards = sorted(Path(dataset_dir).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {dataset_dir}")
    imgs, vecs, acts = [], [], []
    for s in shards:
        z = np.load(s)
        imgs.append(z["image"]); vecs.append(z["vec"]); acts.append(z["action"])
    image = np.concatenate(imgs, axis=0)
    vec = np.concatenate(vecs, axis=0)
    action = np.concatenate(acts, axis=0)
    return image, vec, action


# --------------------------------------------------------------------------- #
# eval (deployable obs, level 1.0, success from term_status)
# --------------------------------------------------------------------------- #
def eval_student(policy, *, episodes, perception_noise, level, image_size,
                 seed, device):
    import torch
    from RL.student_teacher.student_env import make_student_env, flatten_student_vec

    env = make_student_env(perception_noise=perception_noise, level=level,
                           image_size=image_size, seed=seed)
    policy.eval()

    def act(obs):
        img = torch.as_tensor(np.ascontiguousarray(
            obs["image"].transpose(2, 0, 1)[None]), device=device)
        vec = torch.as_tensor(flatten_student_vec(obs)[None], device=device)
        with torch.no_grad():
            a = policy(img, vec).cpu().numpy().reshape(-1)
        return a

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
def fit_bc(policy, dataset_dir, *, epochs, batch_size, lr, device, seed,
           out, metrics_path, eval_every, eval_episodes, perception_noise,
           level, image_size, epoch_offset=0):
    import torch

    image, vec, action = load_dataset(dataset_dir)
    n = image.shape[0]
    print(f"[student] dataset: {n} transitions  image={image.shape[1:]} "
          f"vec={vec.shape[1]} action={action.shape[1]}", flush=True)

    vec_t = torch.as_tensor(vec, device=device)
    act_t = torch.as_tensor(action, device=device)
    # keep images on CPU uint8 (memory); move per-batch
    image_t = torch.as_tensor(image)                        # uint8 NHWC on CPU

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            idx_t = torch.as_tensor(idx)
            img_b = image_t[idx_t].permute(0, 3, 1, 2).to(device)   # -> (B,C,H,W)
            vec_b = vec_t[idx_t]
            act_b = act_t[idx_t]
            pred = policy(img_b, vec_b)
            loss = loss_fn(pred, act_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        mean_loss = tot / max(1, nb)
        gep = epoch_offset + ep + 1
        print(f"[student] epoch {gep} bc_mse={mean_loss:.5f}", flush=True)
        rec = {"event": "train", "epoch": gep, "bc_mse": mean_loss}

        if eval_every and (gep % eval_every == 0 or ep == epochs - 1):
            sr, counts = eval_student(
                policy, episodes=eval_episodes, perception_noise=perception_noise,
                level=level, image_size=image_size, seed=4321, device=device)
            print(f"[student] epoch {gep} eval success_rate={sr:.3f} {counts}",
                  flush=True)
            rec.update({"eval_success_rate": sr, "eval_counts": counts})
            torch.save({"model": policy.state_dict(), "epoch": gep},
                       Path(out) / f"student_ep{gep:03d}.pt")

        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    torch.save({"model": policy.state_dict()}, Path(out) / "student.pt")


# --------------------------------------------------------------------------- #
# DAgger round (scaffold -- regenerate on-policy, teacher relabels, append+fit)
# --------------------------------------------------------------------------- #
def dagger_round(policy, args, device, dataset_dir, metrics_path, round_idx):
    """One DAgger aggregation: roll the CURRENT student, relabel with teacher."""
    import torch
    from RL.student_teacher.student_env import flatten_student_vec

    policy.eval()

    def student_policy(obs):
        img = torch.as_tensor(np.ascontiguousarray(
            obs["image"].transpose(2, 0, 1)[None]), device=device)
        vec = torch.as_tensor(flatten_student_vec(obs)[None], device=device)
        with torch.no_grad():
            return policy(img, vec).cpu().numpy().reshape(-1)

    # single-process on-policy collection; teacher supplies the target action
    distill_dataset.generate(
        transitions=args.dagger_transitions, num_envs=1, out=args.out,
        perception_noise=args.perception_noise, level=args.level,
        image_size=args.image_size, residual_scale=args.residual_scale,
        teacher_zip=args.teacher_zip, seed=args.seed + 7000 + round_idx,
        shard_size=args.shard_size, student_policy=student_policy)
    policy.train()

    fit_bc(policy, dataset_dir, epochs=args.dagger_epochs,
           batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
           out=args.out, metrics_path=metrics_path, eval_every=args.eval_every,
           eval_episodes=args.eval_episodes, perception_noise=args.perception_noise,
           level=args.level, image_size=args.image_size,
           epoch_offset=args.epochs + round_idx * args.dagger_epochs)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset_dir = str(out / "dataset")
    metrics_path = out / "metrics.jsonl"

    # 1) dataset (regenerate if missing or --regen) -- BEFORE any CUDA init so
    #    the fork-based workers are safe.
    have_shards = bool(sorted(Path(dataset_dir).glob("shard_*.npz")))
    if args.regen or not have_shards:
        distill_dataset.generate(
            transitions=args.transitions, num_envs=args.num_envs, out=args.out,
            perception_noise=args.perception_noise, level=args.level,
            image_size=args.image_size, residual_scale=args.residual_scale,
            teacher_zip=args.teacher_zip, seed=args.seed, shard_size=args.shard_size)
    else:
        print(f"[student] reusing existing dataset in {dataset_dir} "
              f"(pass --regen to rebuild)", flush=True)

    # 2) build policy (probe shapes from one shard)
    import torch  # noqa: F401
    device = _resolve_device(args.device)
    z0 = np.load(sorted(Path(dataset_dir).glob("shard_*.npz"))[0])
    h, w, c = z0["image"].shape[1:]
    vec_dim = int(z0["vec"].shape[1])
    policy = build_policy((c, h, w), vec_dim, act_dim=6, feat=args.feat).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[student] policy on {device}: img=({c},{h},{w}) vec_dim={vec_dim} "
          f"params={n_params/1e6:.2f}M", flush=True)

    with open(out / "train_meta.txt", "w") as f:
        f.write(f"transitions={args.transitions}\nepochs={args.epochs}\n"
                f"num_envs={args.num_envs}\nperception_noise={args.perception_noise}\n"
                f"image_size={args.image_size}\nresidual_scale={args.residual_scale}\n"
                f"teacher_zip={args.teacher_zip}\nseed={args.seed}\n"
                f"batch_size={args.batch_size}\nlr={args.lr}\nfeat={args.feat}\n"
                f"vec_dim={vec_dim}\nparams={n_params}\n")

    # 3) BC fit
    t0 = time.time()
    fit_bc(policy, dataset_dir, epochs=args.epochs, batch_size=args.batch_size,
           lr=args.lr, device=device, seed=args.seed, out=args.out,
           metrics_path=metrics_path, eval_every=args.eval_every,
           eval_episodes=args.eval_episodes, perception_noise=args.perception_noise,
           level=args.level, image_size=args.image_size)

    # 4) optional DAgger rounds (default 0 = pure BC)
    for r in range(int(args.dagger_iters)):
        print(f"[student] === DAgger round {r + 1}/{args.dagger_iters} ===", flush=True)
        dagger_round(policy, args, device, dataset_dir, metrics_path, r)

    print(f"[student] DONE elapsed={time.time() - t0:.1f}s -> {out/'student.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
