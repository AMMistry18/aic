"""
Offline training env for the last-inch residual SAC.

Consumes AIC-sim recordings (one .npz per episode, see `RL/recording.py`)
and exposes the same observation dict + action space as the live
`LastInchInsertEnv`. No physics in the loop — the env just plays back
recorded states, computes the reward, and the policy learns a residual
action that would have helped.

Why offline: the AIC Gazebo sim is the only physics source we trust
(matches the engine's contact model, the cable flex, and the
`/scoring/insertion_event` that the rubric uses). Training against
that physics — at the recorded timestep granularity — is faster than
running Gazebo in the loop and avoids any sim-to-real gap.

Reward (matches `RL/reward.py`):
    r_image       image L1 vs the recorded goal image
    r_force       piecewise-log force shaping
    r_xy          tip-port XY progress (small weight)
    r_action      action smoothness
    r_lateral     lateral force penalty
    r_done        +50 on the engine's `insertion_event` frame
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from .observation import ObsConfig, build_obs_dict_from_arrays
from .reward import (
    RewardConfig,
    TerminationConfig,
    check_termination,
    compute_reward,
)


@dataclass
class RecordedEnvConfig:
    obs: ObsConfig = ObsConfig()
    reward: RewardConfig = RewardConfig()
    term: TerminationConfig = TerminationConfig()
    # How many of the recorded frames count as the "last inch" — we
    # start the policy partway through a successful trajectory so the
    # policy only needs to learn the last few mm.
    last_inch_start_frame: int = 80
    # The curriculum level 0..1 maps to a frame range within
    # `last_inch_start_frame..end`. level=0 means start at frame 95% of
    # the trajectory (very close to the goal); level=1 means start at
    # frame `last_inch_start_frame` (furthest from the goal).
    port_type: str = "sc"


class RecordedRolloutEnv(gym.Env):
    """Offline env that streams (obs, action, reward, next_obs) from
    AIC-sim recordings. Supports the same reverse curriculum as
    `LastInchInsertEnv`.

    Observation space (Dict, identical to `LastInchInsertEnv`):
        image           (H, W, n_cams*ch) uint8
        force           (3,)      float32
        tcp_pose        (7,)      float32
        port_pose       (7,)      float32
        tcp_pose_err    (6,)      float32
        last_action     (6,)      float32

    Action space: Box(6,) float32 in [-1, 1] — residual pose delta.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, dataset_paths: list[str], cfg: RecordedEnvConfig = RecordedEnvConfig()):
        super().__init__()
        self.cfg = cfg
        if not dataset_paths:
            raise ValueError("RecordedRolloutEnv needs at least one .npz dataset file")
        self._rollouts: list[dict] = []
        for p in dataset_paths:
            self._rollouts.append(self._load_rollout(p))
        if not self._rollouts:
            raise ValueError("no usable rollouts loaded")
        # Pre-compute the goal image: the final frame of the first
        # successful rollout. (If we have multiple successful rollouts
        # we could pick the median; for first version the first works.)
        self._goal_image = self._pick_goal_image()
        # Reverse-curriculum state
        self._reset_mode = "curriculum"
        self._curriculum_level = 0.0
        self._level_file: Optional[Path] = None
        # Episode state
        self._t = 0
        self._rollout_idx = 0
        self._last_action = np.zeros(6, dtype=np.float32)
        self._f_z_buf: list[float] = []

        # Spaces
        H, W, ch = cfg.obs.image_h, cfg.obs.image_w, cfg.obs.image_channels
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(H, W, ch), dtype=np.uint8),
            "force": spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float32),
            "tcp_pose": spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float32),
            "port_pose": spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float32),
            "tcp_pose_err": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            "last_action": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    @staticmethod
    def _load_rollout(path: str) -> dict:
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}

    def _pick_goal_image(self) -> np.ndarray:
        # Prefer the final image of a successful rollout. Fall back to
        # the final image of any rollout.
        for r in self._rollouts:
            if bool(r["task_success"]):
                return r["images"][-1].copy()
        return self._rollouts[0]["images"][-1].copy()

    # ------------------------------------------------------------------ #
    # curriculum API (matches LastInchInsertEnv)
    # ------------------------------------------------------------------ #

    def set_curriculum_level(self, level: float) -> None:
        self._curriculum_level = float(np.clip(level, 0.0, 1.0))

    def get_curriculum_level(self) -> float:
        return float(self._curriculum_level)

    def set_reset_mode(self, mode: str) -> None:
        if mode not in ("curriculum", "random", "near_goal"):
            raise ValueError(f"unknown reset mode {mode!r}")
        self._reset_mode = mode

    def set_level_file(self, path: Optional[str]) -> None:
        self._level_file = Path(path) if path is not None else None

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # Re-read curriculum level from file (for SubprocVecEnv support)
        if self._level_file is not None and Path(self._level_file).exists():
            try:
                self._curriculum_level = float(Path(self._level_file).read_text().strip())
            except Exception:
                pass
        # Pick a random rollout
        rng = self.np_random
        self._rollout_idx = int(rng.integers(0, len(self._rollouts)))
        r = self._rollouts[self._rollout_idx]
        n_frames = r["images"].shape[0]
        # Pick a start frame based on the curriculum level.
        last_inch_start = min(self.cfg.last_inch_start_frame, n_frames - 2)
        if self._reset_mode == "near_goal":
            self._t = int(0.95 * (n_frames - 1))
        elif self._reset_mode == "curriculum":
            level = self._curriculum_level
            # level=0 -> start near goal (95% in); level=1 -> start at last_inch_start
            start_pct = 0.95 - 0.95 * level  # 0.95 -> 0.0
            self._t = int(np.clip(start_pct * (n_frames - 1),
                                  last_inch_start, n_frames - 2))
        else:  # random
            self._t = int(rng.integers(last_inch_start, n_frames - 1))
        self._last_action = np.zeros(6, dtype=np.float32)
        self._f_z_buf = []
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self._last_action = action

        r = self._rollouts[self._rollout_idx]
        n_frames = r["images"].shape[0]
        # advance by one step (the recorded action is what's already in
        # the dataset; we just walk forward through it). The policy's
        # action is "logged" for the trainer but doesn't change the
        # trajectory — that's the limitation of offline RL: the
        # policy's action doesn't affect next_obs. The reward still
        # reflects what the recorded trajectory did.
        self._t += 1
        done_in_episode = self._t >= n_frames - 1

        obs = self._obs()
        image_curr = obs["image"]
        f_xyz = obs["force"]
        self._f_z_buf.append(float(f_xyz[2]))

        # The engine's insertion_event is the authoritative success
        # signal. We use the recorded `insertion_event` flag at the
        # current frame.
        recorded_event = bool(r["insertion_event"][min(self._t, n_frames - 1)])
        # If the engine fired the event, the recorded action is
        # considered correct — the bonus fires at that frame.
        term_status = None
        if recorded_event:
            term_status = "success"
        # timeout when we reach the end of the recording
        if done_in_episode and term_status is None:
            term_status = "timeout"

        tip_xyz = r["tcp_xyzs"][min(self._t, n_frames - 1)]
        port_xyz = r["port_xyzs"][min(self._t, n_frames - 1)]
        image_l1_norm = float("nan")
        off_limit_contact = False
        total, breakdown = compute_reward(
            image_curr=image_curr,
            image_goal=self._goal_image,
            f_z=float(f_xyz[2]),
            f_xy=f_xyz[:2],
            tip_xy=tip_xyz[:2],
            port_xy=port_xyz[:2],
            a_t=action,
            a_prev=self._last_action,
            term_status=term_status,
            cfg=self.cfg.reward,
            bonus_eligible=(term_status == "success"),
        )
        image_l1_norm = breakdown.image_l1_norm

        # use the reward module's termination for safety cases
        if term_status is None:
            term_status = check_termination(
                image_l1_norm=image_l1_norm,
                f_z=float(f_xyz[2]),
                off_limit_contact=off_limit_contact,
                step=self._t,
                cfg_rew=self.cfg.reward,
                cfg_term=self.cfg.term,
                f_linger_dwell_s=0.0,
                bypass_image_success=True,
            )

        terminated = term_status in ("success", "force_abort", "off_limit")
        truncated = term_status == "timeout"
        f_z_mean = float(np.mean(self._f_z_buf)) if self._f_z_buf else float("nan")
        f_z_max = float(np.max(np.abs(self._f_z_buf))) if self._f_z_buf else float("nan")
        info = {
            "term_status": term_status,
            "breakdown": breakdown,
            "image_l1_norm": image_l1_norm,
            "f_z": float(f_xyz[2]),
            "f_z_mean": f_z_mean,
            "f_z_max": f_z_max,
            "wallclock": float(self._t) * 0.05,
            "curriculum_level": self._curriculum_level,
            "reset_mode": self._reset_mode,
        }
        return obs, float(total), terminated, truncated, info

    def _obs(self) -> dict:
        r = self._rollouts[self._rollout_idx]
        t = min(self._t, r["images"].shape[0] - 1)
        return build_obs_dict_from_arrays(
            image=r["images"][t],
            tcp_xyz=r["tcp_xyzs"][t],
            tcp_q_wxyz=r["tcp_qs"][t],
            port_xyz=r["port_xyzs"][t],
            port_q_wxyz=r["port_qs"][t],
            force_xyz=r["forces"][t],
            last_action=self._last_action,
            cfg=self.cfg.obs,
        )

    def render(self) -> np.ndarray:
        """Return the center-cam image of the current frame."""
        r = self._rollouts[self._rollout_idx]
        t = min(self._t, r["images"].shape[0] - 1)
        H, W, C = r["images"].shape[1:]
        n_cams = int(r["n_cams"])  # 0-dim array, hence the int() cast
        ch = C // n_cams
        center = r["images"][t, :, :, ch:ch*2]
        return np.ascontiguousarray(center)


def discover_dataset(directory: str, port_type: Optional[str] = None) -> list[str]:
    """Find all .npz rollout files in `directory`, optionally filtered
    by port_type stored inside the file."""
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.npz"), recursive=True))
    if port_type is None:
        return paths
    out = []
    for p in paths:
        try:
            d = np.load(p, allow_pickle=False)
            pt = str(d["port_type"])
        except Exception:
            continue
        if pt == port_type:
            out.append(p)
    return out


__all__ = ["RecordedEnvConfig", "RecordedRolloutEnv", "discover_dataset"]