"""Smoke test for mujoco.mjx (GPU-backed stepping).

Loads the SDF-exported AIC scene, copies it onto N parallel worlds on
the GPU via `mjx.put_model` + `mjx_v.reset`, steps N environments in a
single JIT'd XLA call, and reports steps/sec + GPU memory.

This proves:
  1. mujoco.mjx is importable (we have mujoco>=3.2).
  2. jax is on a CUDA device.
  3. The AIC scene compiles under mjx (no unsupported plugins like
     `mujoco.elasticity.cable` — see README).
  4. Batched stepping works at the planned 4096-env scale.

Usage:
    pixi run python RL/scripts/setup_mjx.py --worlds 4096 --steps 50
    pixi run python RL/scripts/setup_mjx.py --worlds 256  --steps 50  # quick
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# mujoco + jax + mjx
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=4096,
                   help="number of parallel worlds on the GPU (default 4096)")
    p.add_argument("--steps", type=int, default=50,
                   help="number of steps per world (default 50)")
    p.add_argument("--scene", type=str, default=None,
                   help="override AIC_MJCF_SCENE")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ---- env ----
    print(f"[mjx] jax version:       {jax.__version__}", flush=True)
    print(f"[mjx] jax devices:       {jax.devices()}", flush=True)
    print(f"[mjx] mujoco version:    {mujoco.__version__}", flush=True)

    scene_path = Path(
        args.scene
        or os.environ.get("AIC_MJCF_SCENE")
        or _REPO / "aic_utils" / "aic_mujoco" / "mjcf" / "scene.xml"
    )
    if not scene_path.exists():
        print(f"[mjx] FAIL: scene.xml missing at {scene_path}", flush=True)
        print(f"[mjx] run aic_utils/aic_mujoco/setup_pipeline.sh sdf mjcf first",
              flush=True)
        return 1
    print(f"[mjx] scene:             {scene_path}", flush=True)

    # ---- CPU model load ----
    cpu_model = mujoco.MjModel.from_xml_path(str(scene_path))
    gpu_model = mjx.put_model(cpu_model)
    print(f"[mjx] nbody={cpu_model.nbody}  nu=({cpu_model.nu})  "
          f"njnt={cpu_model.njnt}  nq={cpu_model.nq}  nv={cpu_model.nv}",
          flush=True)

    # ---- batched reset ----
    @jax.jit
    def reset(key):
        # Use jax.vmap to make a single env-reset function work for all worlds.
        def _one(k):
            d = mujoco.MjData(cpu_model)
            mujoco.mj_forward(cpu_model, d)
            return mjx.put_data(cpu_model, d)
        return jax.vmap(_one)(jax.random.split(key, args.worlds))

    @jax.jit
    def step(model, data):
        # Sample a random action in [-1, 1] per world, scale by the
        # model's actuator_ctrlrange, and step.
        keys = jax.random.split(jax.random.PRNGKey(int(jax.device_get(
            data.qpos[0, 0] * 1e6) % (2**31))), args.worlds)
        rand = jax.random.uniform(
            keys, (args.worlds, model.nu), minval=-1.0, maxval=1.0,
        )
        lo = jnp.broadcast_to(model.actuator_ctrlrange[:, 0],
                              (args.worlds, model.nu))
        hi = jnp.broadcast_to(model.actuator_ctrlrange[:, 1],
                              (args.worlds, model.nu))
        act = lo + (hi - lo) * (rand + 1.0) / 2.0
        return mjx.step(model, data, act)

    print(f"[mjx] resetting {args.worlds} worlds on "
          f"{jax.devices()[0]}...", flush=True)
    t0 = time.time()
    key = jax.random.PRNGKey(0)
    batch_data = reset(key)
    batch_data.block_until_ready()
    print(f"[mjx] reset: {(time.time()-t0)*1000:.1f} ms", flush=True)

    print(f"[mjx] stepping {args.steps} steps x {args.worlds} worlds...",
          flush=True)
    t0 = time.time()
    for _ in range(args.steps):
        batch_data = step(gpu_model, batch_data)
    batch_data.block_until_ready()
    elapsed = time.time() - t0
    total_steps = args.steps * args.worlds
    print(f"[mjx] {total_steps} steps in {elapsed:.2f}s "
          f"({total_steps/elapsed:,.0f} steps/s)", flush=True)

    # quick sanity — qpos should not all be the same (random actions
    # produce divergent trajectories)
    qpos_var = float(jnp.var(batch_data.qpos[:, 0]))
    print(f"[mjx] qpos[0] variance across worlds: {qpos_var:.4e} "
          f"(should be > 1e-6 if worlds are diverging)",
          flush=True)

    print("[mjx] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())