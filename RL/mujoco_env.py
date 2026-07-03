"""Last-inch residual-RL env that loads aic_utils/aic_mujoco/mjcf/scene.xml.

This is the Step-1 plumbing: scene loads, steps, exposes the obs+action
interfaces the future SAC loop will consume. Reward shaping is intentionally
omitted and will be added in Step 3.

Two MuJoCo training scenes are available in this repo:

  1. `RL/env.py:LastInchInsertEnv` — a *procedural* plug + port scene with
     a 6-DoF free joint for the plug and three wrist cameras. Fast to load,
     perfect for ablations and unit tests. ~1 ms per step.

  2. `aic_utils/aic_mujoco/mjcf/scene.xml` (this file's input) — the
     SDF-exported AIC scene (UR5e + gripper + cable + task board + ports)
     produced by `aic_utils/aic_mujoco/scripts/add_cable_plugin.py`. Matches
     the live AIC sim's contact model, cable flex, and `/scoring/insertion_event`.
     ~5 ms per step on a single env; this is the scene the 4096-env
     curriculum trainer will use.

This file is the path that matches the live AIC sim. See
`aic_utils/aic_mujoco/README.md` for the SDF -> MJCF export pipeline.

Usage:

    from RL.mujoco_env import MuJoCoEnvConfig, MuJoCoLastInchEnv
    env = MuJoCoLastInchEnv(MuJoCoEnvConfig())
    obs, info = env.reset(seed=0)
    obs, rew, term, trunc, info = env.step(env.action_space.sample())

Or override the scene path:

    AIC_MJCF_SCENE=/path/to/scene.xml pixi run python RL/scripts/setup_mujoco.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:  # pragma: no cover
    _HAS_MUJOCO = False

# Render backend — EGL for headless, OSMesa fallback, GLFW otherwise.
# Set BEFORE mujoco.Renderer is constructed.
os.environ.setdefault("MUJOCO_GL", "egl")

# Default scene path — the SDF-exported AIC MJCF, post
# `aic_utils/aic_mujoco/scripts/add_cable_plugin.py`. Override via the
# AIC_MJCF_SCENE env var.
_DEFAULT_SCENE = (
    Path(__file__).resolve().parent.parent
    / "aic_utils" / "aic_mujoco" / "mjcf" / "scene.xml"
)


@dataclass
class MuJoCoEnvConfig:
    """Config for the SDF-exported AIC scene + last-inch env wrapper."""

    scene_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AIC_MJCF_SCENE", str(_DEFAULT_SCENE))
        )
    )
    dt: float = 0.05
    image_h: int = 32
    image_w: int = 32
    n_cams: int = 3
    action_dim: int = 6
    max_episode_steps: int = 600  # 30 s at 20 Hz, matches REWARD_SPEC §6


class MuJoCoLastInchEnv(gym.Env):
    """`gym.Env` over the AIC SDF-exported MJCF scene.

    Observation space mirrors the schema the future SAC loop will consume
    (Dict of: image, force, tcp_pose, port_pose, tcp_pose_err, last_action),
    bit-compatible with what `RL/observation.py:build_obs_dict_from_arrays`
    and `RL/recorded_env.py:RecordedRolloutEnv` already produce. The reward
    in `RL/reward.py` will consume it unchanged in Step 3.

    Action space is `Box(action_dim,)` in `[-1, 1]` — a residual pose
    delta. The actual scaling and IK mapping is a Step-3 concern (see
    `RL/env.py:EnvConfig.pos_scale / rot_scale` for the procedural-scene
    reference that already does this).

    For Step 1 (this plan) the env's `step()` produces:
        obs      = the dict above, populated with the current scene state
        reward   = 0.0  (real reward is a Step-3 task)
        terminated = False  (no reward = no termination logic yet)
        truncated = (step_count >= max_episode_steps)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: MuJoCoEnvConfig = MuJoCoEnvConfig()):
        if not _HAS_MUJOCO:
            raise RuntimeError(
                "mujoco is not installed. Activate the pixi env: "
                "`pixi shell` (pixi.toml pins mujoco==3.5.0)."
            )
        super().__init__()
        self.cfg = cfg
        if not cfg.scene_path.exists():
            raise FileNotFoundError(
                f"AIC MJCF scene not found at {cfg.scene_path}. "
                "Run `aic_utils/aic_mujoco/scripts/add_cable_plugin.py` "
                "first (see aic_utils/aic_mujoco/README.md), or set "
                "AIC_MJCF_SCENE to a valid path."
            )
        self.model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
        self.data = mujoco.MjData(self.model)
        self._step_count = 0
        self._renderer: Optional["mujoco.Renderer"] = None  # lazy

        # Action space — residual pose delta in [-1, 1].
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(cfg.action_dim,),
            dtype=np.float32,
        )

        # Observation space — bit-compatible with RL/observation.py.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(cfg.image_h, cfg.image_w, cfg.n_cams * 3),
                dtype=np.uint8,
            ),
            "force": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32,
            ),
            "tcp_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32,
            ),
            "port_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32,
            ),
            "tcp_pose_err": spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32,
            ),
            "last_action": spaces.Box(
                low=-1.0, high=1.0, shape=(cfg.action_dim,),
                dtype=np.float32,
            ),
        })

    # ------------------------------------------------------------------ #
    # render plumbing
    # ------------------------------------------------------------------ #

    def _ensure_renderer(self) -> "mujoco.Renderer":
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.cfg.image_h,
                width=self.cfg.image_w,
            )
        return self._renderer

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._obs(), {}

    def _obs(self) -> dict[str, np.ndarray]:
        """Build the obs dict.

        Step-1 stub: zeros for everything that isn't trivially readable
        from the scene. Step-3 will populate tcp_pose, port_pose, force,
        tcp_pose_err from the AIC sim's body IDs (wrist_3_link,
        sc_port / nic_card mount, AtiForceTorqueSensor, etc.).
        """
        n_cams = self.cfg.n_cams
        return {
            "image": np.zeros(
                (self.cfg.image_h, self.cfg.image_w, n_cams * 3),
                dtype=np.uint8,
            ),
            "force": np.zeros(3, dtype=np.float32),
            "tcp_pose": np.zeros(7, dtype=np.float32),
            "port_pose": np.zeros(7, dtype=np.float32),
            "tcp_pose_err": np.zeros(6, dtype=np.float32),
            "last_action": np.zeros(self.cfg.action_dim, dtype=np.float32),
        }

    def step(self, action: np.ndarray):
        # Stub: no actuator wiring in Step 1 — just step the model forward.
        # Step-3 will map action -> MuJoCo ctrl vector via the UR5e joint
        # actuators in `aic_utils/aic_mujoco/mjcf/aic_robot.xml`.
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._obs()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        renderer = self._ensure_renderer()
        renderer.update_scene(self.data)
        return renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


__all__ = ["MuJoCoEnvConfig", "MuJoCoLastInchEnv"]