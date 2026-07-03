"""
Gymnasium environment for residual-SAC last-inch insertion training.

We wrap a *MuJoCo* scene (not the AIC Gazebo sim) so we can step at 0.1 ms
per env step. This is required for residual SAC to converge in <2 GPU-hours
per port type — Gazebo would take ~80 days for the same number of steps.

Scene contents (see REWARD_SPEC.md §11):
    - 6-DoF free joint for the plug, controllable in (mx, my, mz, drx, dry, drz)
    - Fixed port box with chamfer inset at world origin
    - One overhead virtual camera rendering 32x32 grayscale at 20 Hz

Episode structure (REWARD_SPEC.md §7b):
    At reset, the plug is sampled from a random pose inside the last-inch
    envelope (±3 mm XY, −8/+4 mm Z, ±15° yaw). Force pre-contact is
    optional (random U(0, 8) N).

Reward:
    From `RL.reward.compute_reward`. The goal image is captured at the
    fully-inserted pose once during `__init__` (no per-reset capture —
    the port doesn't move in this sim).
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import xml.etree.ElementTree as ET
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

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:  # pragma: no cover
    _HAS_MUJOCO = False

# Render backend — EGL for headless, OSMesa fallback, GLFW otherwise.
# Set BEFORE mujoco.Renderer is constructed.
try:
    if os.environ.get("AIC_MUJOCO_GL", "egl").lower() == "egl":
        mujoco.GLContext_EGL = None  # placeholder to avoid lint complaints
        # modern mujoco picks backend based on env var MUJOCO_GL
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif os.environ.get("AIC_MUJOCO_GL", "egl").lower() == "osmesa":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
except Exception:
    pass

from .observation import ObsConfig, build_obs_dict_from_arrays
from .reward import (
    RewardConfig,
    RewardBreakdown,
    TerminationConfig,
    check_termination,
    compute_reward,
)


# --------------------------------------------------------------------------- #
# MJCF scene (built once, cached)
# --------------------------------------------------------------------------- #


# Port-type-specific MJCF geometry.
# Values are taken from the AIC engine's SDFs:
#   aic_assets/models/SFP Module/model.sdf   (SFP cage collision boxes)
#   aic_assets/models/SC Port/model.sdf      (SC port collision boxes)
#   aic_example_policies/.../PerceptionInsert.py: INSERTION_DEPTH
# Plug dimensions are the LC Plug (sfp_sc_cable) reduced to a simple
# box of half-extents (0.0035, 0.0035, 0.005) — close enough to the
# real plug for the dynamics to be representative without modeling the
# full mesh.

_PORT_GEOMETRY = {
    # SC: shallow, wider slot, 16 mm insertion depth
    "sc": {
        "entrance_radius_x": 0.00515,  # 10.3 mm wide / 2
        "entrance_radius_y": 0.003,    # chamfered
        "entrance_height": 0.004,      # the slot's vertical extent
        "insertion_depth": 0.016,      # matches INSERTION_DEPTH["sc"]
        "chamfer_radius": 0.0015,      # top-edge chamfer
        "wall_thickness": 0.001,
        "color": "0.3 0.3 0.4 1",      # bluish plastic
        "start_xy_max_m": 0.004,
        "start_z_min_m": -0.001,       # 1 mm above port entrance
        "start_z_max_m": 0.0005,
        "start_yaw_max_rad": 0.262,    # 15 deg
        "success_xy_max_m": 0.001,     # 1 mm radial error
        "success_z_max_m": -0.0015,    # 1.5 mm below (seated)
        "success_force_z_n": 1.0,
    },
    # SFP: deep narrow cage, 51 mm insertion depth
    "sfp": {
        "entrance_radius_x": 0.007,    # 14 mm wide / 2 (SFP cage opening)
        "entrance_radius_y": 0.0043,   # 8.6 mm tall / 2
        "entrance_height": 0.0086,     # full cage height
        "insertion_depth": 0.051,      # matches INSERTION_DEPTH["sfp"]
        "chamfer_radius": 0.001,
        "wall_thickness": 0.001,
        "color": "0.5 0.4 0.3 1",      # brownish metal
        "start_xy_max_m": 0.003,       # tighter — SFP is more sensitive
        "start_z_min_m": -0.0005,      # 0.5 mm above port entrance
        "start_z_max_m": 0.0003,
        "start_yaw_max_rad": 0.175,    # 10 deg
        "success_xy_max_m": 0.001,     # 1 mm radial error
        "success_z_max_m": -0.025,     # 25 mm below — past the top cap (z=0)
        "success_force_z_n": 1.0,      # require contact (not flying through)
    },
}


def _build_mjcf(port_type: str, meshdir: str, cam_res: int) -> str:
    """Generate the MJCF XML for a given port type. Cameras + plug are
    shared; only the port body changes between port types.
    """
    g = _PORT_GEOMETRY[port_type]
    color = g["color"]
    rx = g["entrance_radius_x"]
    ry = g["entrance_radius_y"]
    h = g["entrance_height"]
    chamfer = g["chamfer_radius"]
    wall = g["wall_thickness"]
    depth = g["insertion_depth"]

    # The port is a closed receptacle:
    #   * 4 side walls forming the U-shape opening at the top
    #   * a back wall at the bottom (the plug tip target at z=-insertion_depth)
    #   * a TOP CAP so the plug can't fall through the open top by gravity
    # The top cap is the small slab at z=0..+h/2 just inside the opening.
    return f"""\
<mujoco model="aic_last_inch_{port_type}">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <option timestep="0.05" gravity="0 0 0"/>
  <default>
    <geom rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <light pos="0 0 0.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="left_cam"   pos="-0.012  0     0.06" mode="fixed"
            fovy="60" resolution="{cam_res} {cam_res}"/>
    <camera name="center_cam" pos=" 0      0     0.06" mode="fixed"
            fovy="60" resolution="{cam_res} {cam_res}"/>
    <camera name="right_cam"  pos=" 0.012  0     0.06" mode="fixed"
            fovy="60" resolution="{cam_res} {cam_res}"/>
    <body name="port" pos="0 0 0">
      <!-- Back wall + receptacle cavity. The plug tip should reach
           z = -insertion_depth when fully inserted. -->
      <geom type="box" pos="0 0 {-depth/2:.4f}" size="{rx + wall:.5f} {ry + wall:.5f} {depth/2:.5f}"
            rgba="{color}" solref="0.02 1" friction="0.15 0.005 0.0001"/>
      <!-- 4 side walls forming the U opening (top half of the receptacle) -->
      <geom type="box" pos=" {rx + wall/2:.5f}  0           {-h/4:.5f}"
            size="{wall/2:.5f} {ry + wall:.5f} {h/4:.5f}"
            rgba="{color}" friction="0.30 0.005 0.0001"/>
      <geom type="box" pos="-{rx + wall/2:.5f}  0           {-h/4:.5f}"
            size="{wall/2:.5f} {ry + wall:.5f} {h/4:.5f}"
            rgba="{color}" friction="0.30 0.005 0.0001"/>
      <geom type="box" pos=" 0           {ry + wall/2:.5f} {-h/4:.5f}"
            size="{rx + wall:.5f} {wall/2:.5f} {h/4:.5f}"
            rgba="{color}" friction="0.30 0.005 0.0001"/>
      <geom type="box" pos=" 0          -{ry + wall/2:.5f} {-h/4:.5f}"
            size="{rx + wall:.5f} {wall/2:.5f} {h/4:.5f}"
            rgba="{color}" friction="0.30 0.005 0.0001"/>
      <!-- Top cap: closes the receptacle so the plug cannot fall through.
           Spans the full opening (rx, ry) at the top edge z=0. -->
      <geom type="box" pos="0 0 0" size="{rx + wall:.5f} {ry + wall:.5f} {wall/2:.5f}"
            rgba="{color}" friction="0.30 0.005 0.0001"/>
    </body>
    <body name="plug" pos="0 0 -0.005">
      <freejoint name="plug_free"/>
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
      <geom type="box" size="0.0035 0.0035 0.005" rgba="0.9 0.5 0.2 1"
            solref="0.02 1" friction="0.30 0.005 0.0001" mass="0.5"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="vx" joint="plug_free" gear="1 0 0 0 0 0" ctrlrange="-1 1"/>
    <motor name="vy" joint="plug_free" gear="0 1 0 0 0 0" ctrlrange="-1 1"/>
    <motor name="vz" joint="plug_free" gear="0 0 1 0 0 0" ctrlrange="-1 1"/>
    <motor name="wx" joint="plug_free" gear="0 0 0 1 0 0" ctrlrange="-1 1"/>
    <motor name="wy" joint="plug_free" gear="0 0 0 0 1 0" ctrlrange="-1 1"/>
    <motor name="wz" joint="plug_free" gear="0 0 0 0 0 1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


# Old single-template constant removed. Per-port MJCFs are now built
# dynamically by `_build_mjcf(port_type, ...)` and written to disk by
# `_make_mjcf_path(port_type, cam_res)`.


def _make_mjcf_path(port_type: str = "sc", cam_res: int = 32) -> Path:
    """Write the MJCF for a given port type to a tmpdir. Cache per-port.

    The cache key combines (port_type, cam_res) so changing either
    invalidates the cache.
    """
    cache_dir = Path(os.environ.get("AIC_MJCF_CACHE_DIR", tempfile.gettempdir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"aic_last_inch_{port_type}_cam{cam_res}.xml"
    if not f.exists():
        f.write_text(_build_mjcf(port_type, str(cache_dir), cam_res))
    return f


def _make_mjcf_path_legacy(cam_res: int = 32) -> Path:
    """Legacy stub - see the per-port version above. Kept for any
    stragglers that imported this name; the new _make_mjcf_path
    requires a port_type arg.
    """
    return _make_mjcf_path("sc", cam_res)


# --------------------------------------------------------------------------- #
# env
# --------------------------------------------------------------------------- #


@dataclass
class EnvConfig:
    obs: ObsConfig = ObsConfig()
    reward: RewardConfig = RewardConfig()
    term: TerminationConfig = TerminationConfig()
    pos_scale: tuple = (0.0015, 0.0015, 0.0035)
    rot_scale: tuple = (0.08, 0.08, 0.12)
    dt: float = 0.05
    # Port type — selects MJCF geometry + start envelope + success region.
    # If set, the start/success fields below are auto-populated from
    # _PORT_GEOMETRY. If left as None, the explicit values are used.
    port_type: Optional[str] = None
    # start envelope — must NOT overlap with the success region below
    start_xy_max_m: float = 0.004      # ±4 mm sideways miss
    start_z_min_m: float = -0.001      # 1 mm above port entrance (highest start)
    start_z_max_m: float = 0.0005      # 0.5 mm below — still well above success
    start_yaw_max_rad: float = np.deg2rad(15.0)
    pre_force_z_n_max: float = 2.0
    # success region — tightened so episodes are 200-600 steps, not 1-18
    success_xy_max_m: float = 0.001    # 1 mm radial error
    success_z_max_m: float = -0.0015   # 1.5 mm below port entrance (seated)
    success_force_z_n: float = 1.0     # must be in contact

    def for_port(self, port_type: str) -> "EnvConfig":
        """Return a new EnvConfig with the geometry params for `port_type`."""
        g = _PORT_GEOMETRY[port_type]
        return dataclasses.replace(self,
            port_type=port_type,
            start_xy_max_m=g["start_xy_max_m"],
            start_z_min_m=g["start_z_min_m"],
            start_z_max_m=g["start_z_max_m"],
            start_yaw_max_rad=g["start_yaw_max_rad"],
            success_xy_max_m=g["success_xy_max_m"],
            success_z_max_m=g["success_z_max_m"],
            success_force_z_n=g["success_force_z_n"],
        )


class LastInchInsertEnv(gym.Env):
    """MuJoCo gym env for residual SAC last-inch insertion training.

    Observation space (Dict):
        image           (H, W, N) uint8
        force           (3,)      float32
        tcp_pose        (7,)      float32
        port_pose       (7,)      float32
        tcp_pose_err    (7,)      float32
        last_action     (6,)      float32

    Action space: Box(6,) float32 in [-1, 1] - scaled by EnvConfig scales
    to a pose delta applied to the plug's free joint.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig = EnvConfig(), port_type: str = "sc"):
        super().__init__()
        if not _HAS_MUJOCO:
            raise RuntimeError("mujoco is required for LastInchInsertEnv")
        if port_type not in _PORT_GEOMETRY:
            raise ValueError(f"unknown port_type {port_type!r}; "
                             f"supported: {list(_PORT_GEOMETRY)}")
        # Auto-populate start envelope + success region from port type.
        # If cfg.port_type is None, the explicit cfg values are used.
        if cfg.port_type is None:
            cfg = cfg.for_port(port_type)
        self.cfg = cfg
        self.port_type = port_type
        # Always use 3 wrist cams (matching AIC Observation.msg). The image
        # is RGB; the channel count is n_cams*3 (left, center, right stacked).
        if cfg.obs.n_cams != 3:
            self.cfg = dataclasses.replace(cfg, obs=dataclasses.replace(
                cfg.obs, n_cams=3, image_ch_per_cam=3))
        self._mjcf_path = _make_mjcf_path(port_type, cam_res=cfg.obs.image_h)
        self._model = mujoco.MjModel.from_xml_path(str(self._mjcf_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=cfg.obs.image_h, width=cfg.obs.image_w)
        self._port_q = np.array([1.0, 0, 0, 0], dtype=np.float64)  # identity port frame
        self._step_count = 0
        # curriculum / reset-mode state
        self._reset_mode: str = "curriculum"
        self._curriculum_level: float = 0.0
        self._level_file: Optional[Path] = None  # set by set_level_file()
        self._last_action = np.zeros(6, dtype=np.float32)
        self._goal_image = None  # set during _capture_goal_image
        self._f_linger_dwell_s = 0.0
        # insertion depth for this port type (used by success check, video)
        self._insertion_depth = _PORT_GEOMETRY[port_type]["insertion_depth"]

        # spaces
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

        # capture the goal image once at the canonical fully-inserted pose
        self._capture_goal_image()

    # ------------------------------------------------------------------ #
    # goal image
    # ------------------------------------------------------------------ #

    def _capture_goal_image(self):
        """Render the image at the canonical fully-inserted pose.

        For both SFP and SC the goal is: tip at port centre (XY), seated
        at the bottom of the receptacle (z = -insertion_depth). We let
        the plug settle for 20 steps so contact forces reach steady state.
        """
        goal_xyz = np.array([0.0, 0.0, -self._insertion_depth * 0.95], dtype=np.float64)
        self._reset_to_pose(tip_xyz=goal_xyz, tip_q_wxyz=np.array([1, 0, 0, 0]))
        # let it settle
        for _ in range(20):
            mujoco.mj_step(self._model, self._data)
        self._goal_image = self._render_image()

    def _reset_to_pose(self, tip_xyz: np.ndarray, tip_q_wxyz: np.ndarray):
        # Plug body is positioned with its tip at `tip_xyz`. Since the
        # plug's geometry is a 0.005 m half-extent box centred at the body
        # origin, we shift the body by -tip_xyz_world_origin + tip_offset.
        # For simplicity, set the body position to `tip_xyz` and the
        # orientation to `tip_q_wxyz` (plug orientation == tip orientation
        # in this simplified model).
        self._data.qpos[:3] = tip_xyz
        self._data.qpos[3:7] = tip_q_wxyz
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def _render_image(self) -> np.ndarray:
        """Render the 3 wrist cameras and stack along the channel axis.

        Returns (H, W, n_cams*3) uint8 — the policy's image observation.
        Each cam is rendered independently, so the policy gets a proper
        stereo / multi-view signal matching the AIC wrist mount.
        """
        frames = []
        for cam_name in ("left_cam", "center_cam", "right_cam"):
            self._renderer.update_scene(self._data, camera=cam_name)
            rgb = self._renderer.render()  # (H, W, 3) uint8
            frames.append(rgb)
        # Concatenate along channel axis → (H, W, n_cams*3) uint8
        return np.concatenate(frames, axis=2)

    def render(self) -> np.ndarray:
        """Public render — returns the center-cam RGB frame for video logging."""
        self._renderer.update_scene(self._data, camera="center_cam")
        return self._renderer.render()

    # ------------------------------------------------------------------ #
    # state extraction
    # ------------------------------------------------------------------ #

    def _plug_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._data.qpos[:3].copy(), self._data.qpos[3:7].copy()

    def _force_xyz(self) -> np.ndarray:
        # Aggregate contact forces on the plug body for the z-axis
        # reading. Mujoco exposes `data.cfrc_int` (6-vectors per body).
        # We pull the world-frame force on the plug body.
        # body id 1 == plug (after worldbody)
        try:
            plug_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "plug")
        except Exception:
            return np.zeros(3, dtype=np.float32)
        cfrc = self._data.cfrc_int[plug_body_id]
        return np.array(cfrc[:3], dtype=np.float32)

    def _tip_xy(self) -> np.ndarray:
        # In this simplified model, tip == plug body origin
        return self._data.qpos[:2].copy()

    def _port_xy(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # start-pose sampling (curriculum / random / near_goal)
    # ------------------------------------------------------------------ #

    def capture_initial_state(self) -> dict:
        """Sample a start pose for a new episode.

        The distribution is governed by `self._curriculum_level` (0..1) and
        `self._reset_mode`:

            mode="curriculum", level=0   → near the goal (small jitter)
            mode="curriculum", level=1   → full last-inch envelope
            mode="curriculum", level=0.5 → half the envelope
            mode="random"                → always full envelope
            mode="near_goal"             → always near the goal

        Curriculum progression is driven externally by the trainer (see
        `CurriculumScheduler`). The env just samples the start pose.
        """
        rng = self.np_random
        if self._reset_mode == "near_goal":
            # Tighter distribution: plug starts very close to the goal.
            xy_err = rng.uniform(-0.001, 0.001, size=2)
            z_err = rng.uniform(-0.001, 0.0005)
            yaw = rng.uniform(-np.deg2rad(5.0), np.deg2rad(5.0))
        elif self._reset_mode == "curriculum":
            level = float(np.clip(self._curriculum_level, 0.0, 1.0))
            # Interpolate envelope radii between (near_goal) and (full).
            near_xy = 0.0015      # 1.5 mm at level=0
            near_z = (0.0005, -0.0015)  # z range at level=0
            near_yaw = np.deg2rad(5.0)
            full_xy = self.cfg.start_xy_max_m
            full_z_lo, full_z_hi = self.cfg.start_z_min_m, self.cfg.start_z_max_m
            full_yaw = self.cfg.start_yaw_max_rad

            # If start_z_max_m is positive and start_z_min_m is negative,
            # we interpolate each endpoint separately. For our config:
            # start_z_min_m = -0.001, start_z_max_m = 0.0005. So we
            # treat z_low as the "deepest" (most negative) and z_high as
            # the "highest" (least negative) start. The near-goal range
            # is (z_high=-0.0015, z_low=0.0005) — actually inverted,
            # we want near-goal to be *close to 0*. Adjust:
            z_high = near_z[0] + level * (full_z_hi - near_z[0])
            z_low  = near_z[1] + level * (full_z_lo - near_z[1])
            xy_r   = near_xy + level * (full_xy - near_xy)
            yaw_r  = near_yaw + level * (full_yaw - near_yaw)
            xy_err = rng.uniform(-xy_r, xy_r, size=2)
            z_err  = rng.uniform(z_low, z_high)
            yaw    = rng.uniform(-yaw_r, yaw_r)
        else:  # "random"
            xy_err = rng.uniform(-self.cfg.start_xy_max_m, self.cfg.start_xy_max_m, size=2)
            z_err = rng.uniform(self.cfg.start_z_min_m, self.cfg.start_z_max_m)
            yaw = rng.uniform(-self.cfg.start_yaw_max_rad, self.cfg.start_yaw_max_rad)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        return {
            "tip_xyz": np.array([xy_err[0], xy_err[1], z_err], dtype=np.float64),
            "tip_q_wxyz": np.array([cy, 0, 0, sy], dtype=np.float64),
            "qvel": np.zeros(6, dtype=np.float64),
            "step": 0,
            "last_action": np.zeros(6, dtype=np.float32),
            "f_linger_dwell_s": 0.0,
            "curriculum_level": float(self._curriculum_level),
        }

    def set_curriculum_level(self, level: float) -> None:
        """Set the start-pose curriculum level (0=near goal, 1=full envelope)."""
        self._curriculum_level = float(np.clip(level, 0.0, 1.0))

    def get_curriculum_level(self) -> float:
        return float(self._curriculum_level)

    def set_reset_mode(self, mode: str) -> None:
        """Set the reset distribution mode: 'curriculum' / 'random' / 'near_goal'."""
        if mode not in ("curriculum", "random", "near_goal"):
            raise ValueError(f"unknown reset mode {mode!r}")
        self._reset_mode = mode

    def set_level_file(self, path: Optional[str]) -> None:
        """Make the env re-read its curriculum level from `path` on every reset.

        Used by `CurriculumScheduler` to communicate level changes to
        SubprocVecEnv workers (which can't call methods on the parent).
        """
        self._level_file = Path(path) if path is not None else None

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # Re-read curriculum level from file (if set). This lets a
        # SubprocVecEnv worker pick up level changes from the trainer
        # without needing in-process access to the env.
        if self._level_file is not None and Path(self._level_file).exists():
            try:
                self._curriculum_level = float(Path(self._level_file).read_text().strip())
            except Exception:
                pass
        state = self.capture_initial_state()
        self._data.qpos[:3] = state["tip_xyz"]
        self._data.qpos[3:7] = state["tip_q_wxyz"]
        self._data.qvel[:] = state["qvel"]
        self._step_count = state["step"]
        self._last_action = state["last_action"]
        self._f_linger_dwell_s = state["f_linger_dwell_s"]
        self._f_z_buf = []
        # settle for a couple of steps so contact forces are visible
        for _ in range(2):
            mujoco.mj_step(self._model, self._data)
        self._step_count = 0
        self._last_action = np.zeros(6, dtype=np.float32)
        self._f_linger_dwell_s = 0.0
        self._f_z_buf: list[float] = []   # per-episode Fz samples for stats

        # settle for a couple of steps so contact forces are visible
        for _ in range(2):
            mujoco.mj_step(self._model, self._data)

        obs = self._obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)

        # Apply the action as a *velocity target* via the 6 actuators.
        # We set qvel directly to the action-derived velocity (no force
        # integration), which is the simplest stable mapping for a
        # free joint at our coarse dt=0.05s. With gravity=0 the plug
        # doesn't accumulate unwanted drift between actions.
        pos_scale = np.asarray(self.cfg.pos_scale)
        rot_scale = np.asarray(self.cfg.rot_scale)
        scale = np.concatenate([pos_scale, rot_scale])
        # Action in [-1, 1] -> velocity in scale / dt m/s
        target_vel = (action * scale) / self.cfg.dt
        self._data.ctrl[:] = np.clip(target_vel, -10.0, 10.0)
        # The free-joint accepts a 6-vector of velocity directly
        self._data.qvel[:6] = target_vel
        mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        self._last_action = action.astype(np.float32)

        # build obs + reward
        obs = self._obs()
        image_curr = obs["image"]
        f_xyz = obs["force"]
        self._f_z_buf.append(float(f_xyz[2]))
        tip_xyz, tip_q = self._plug_pose()

        term_status = None
        # Success = tip within success_xy_max_m of port XY, seated below
        # port entrance by success_z_max_m, AND in light contact (so the
        # policy can't get "success" by hovering below the port).
        xy_close = np.linalg.norm(tip_xyz[:2]) < self.cfg.success_xy_max_m
        seated = tip_xyz[2] < self.cfg.success_z_max_m
        in_contact = abs(f_xyz[2]) > self.cfg.success_force_z_n
        if xy_close and seated and in_contact:
            term_status = "success"

        # dwell timer for sustained over-force
        if abs(f_xyz[2]) > self.cfg.term.f_linger_n:
            self._f_linger_dwell_s += self.cfg.dt
        else:
            self._f_linger_dwell_s = 0.0

        image_l1_norm = float("nan")
        off_limit_contact = False  # not modelled in this minimal sim
        total, breakdown = compute_reward(
            image_curr=image_curr,
            image_goal=self._goal_image,
            f_z=float(f_xyz[2]),
            f_xy=f_xyz[:2],
            tip_xy=tip_xyz[:2],
            port_xy=self._port_xy(),
            a_t=action.astype(np.float32),
            a_prev=self._last_action,
            term_status=term_status,
            cfg=self.cfg.reward,
            # Gate the image success bonus on the geometric success check.
            # Without this, a "fall through the cage" state would have
            # image ≈ goal image and flood the reward with +β_s per step.
            bonus_eligible=(term_status == "success"),
        )
        image_l1_norm = breakdown.image_l1_norm

        # termination check (only force_abort/timeout/off_limit here).
        # Success is determined above by the 3-condition geometry+contact
        # check; the image-based success in check_termination is bypassed
        # because the goal image matches the empty cage as well as the
        # seated plug.
        if term_status is None:
            term_status = check_termination(
                image_l1_norm=image_l1_norm,
                f_z=float(f_xyz[2]),
                off_limit_contact=off_limit_contact,
                step=self._step_count,
                cfg_rew=self.cfg.reward,
                cfg_term=self.cfg.term,
                f_linger_dwell_s=self._f_linger_dwell_s,
                bypass_image_success=True,
            )

        terminated = term_status in ("success", "force_abort", "off_limit")
        truncated = term_status == "timeout"
        # per-episode F/T stats (only meaningful at episode end, but cheap to ship)
        f_z_mean = float(np.mean(self._f_z_buf)) if self._f_z_buf else float("nan")
        f_z_max = float(np.max(np.abs(self._f_z_buf))) if self._f_z_buf else float("nan")
        info = {
            "term_status": term_status,
            "breakdown": breakdown,
            "image_l1_norm": image_l1_norm,
            "f_z": float(f_xyz[2]),
            "f_z_mean": f_z_mean,
            "f_z_max": f_z_max,
            "wallclock": float(self._step_count) * self.cfg.dt,
            "curriculum_level": float(self._curriculum_level),
            "reset_mode": self._reset_mode,
        }
        return obs, float(total), terminated, truncated, info

    def _obs(self) -> dict:
        tip_xyz, tip_q = self._plug_pose()
        image = self._render_image()
        return build_obs_dict_from_arrays(
            image=image,
            tcp_xyz=tip_xyz,
            tcp_q_wxyz=tip_q,
            port_xyz=np.zeros(3, dtype=np.float32),
            port_q_wxyz=np.array([1.0, 0, 0, 0], dtype=np.float32),
            force_xyz=self._force_xyz(),
            last_action=self._last_action,
            cfg=self.cfg.obs,
        )

    # gym required helpers
    def render(self):
        return self._render_image()


__all__ = ["EnvConfig", "LastInchInsertEnv"]