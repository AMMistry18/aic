"""
Gymnasium environment for residual-SAC last-inch insertion training.

We wrap a *MuJoCo* scene (not the AIC Gazebo sim) so we can step at ~0.1 ms
per physics tick. This is required for image-SAC to converge in a few
GPU-hours per port type — Gazebo would take weeks for the same step count.

Scene contents (see REWARD_SPEC.md §11):
    - 6-DoF free joint for the plug, actuated in (mx, my, mz, drx, dry, drz)
      via velocity-servo actuators (one per DoF).
    - A real receptacle: 4 side walls + a bottom wall forming an OPEN-top
      rectangular cavity sized to the plug plus a small clearance. The plug
      can actually be driven into it; the bottom wall stops it at full depth.
    - Three overhead virtual cameras (left/centre/right) rendering H×W RGB
      at 20 Hz, stacked on the channel axis to match the AIC 3-wrist-cam obs.

Physics / actuation (v0.3 — fixed):
    - Action ∈ [-1, 1]^6 → target pose-delta per control step, converted to a
      target velocity `action * scale / dt` and tracked by velocity actuators.
    - The velocity servo lets contact forces actually resist the plug, so
      `mj_contactForce`-summed wrench on the plug is a real, bounded signal.
    - Control rate 20 Hz (dt=0.05 s); physics timestep 0.005 s → 10 substeps.

Episode structure — reverse curriculum (REWARD_SPEC.md §7c):
    level 0  → plug starts *near the seated goal* (deep in the cavity,
               centred): a small nudge seats it, so success is dense from
               the very first episodes and bootstraps SAC.
    level 1  → plug starts *above the mouth* with up to ±full xy miss and
               ±full yaw: the policy must align and insert the whole way.
    The trainer's CurriculumScheduler advances the level as the success
    rate rises; the env reads the level from a file on every reset (so it
    works across SubprocVecEnv workers).

Reward:
    From `RL.reward.compute_reward`. The goal image is captured once at the
    fully-seated pose during `__init__` (the port never moves in this sim).
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

# Render backend — EGL for headless (the only one that works on this box).
# Set BEFORE mujoco.Renderer is constructed.
try:
    _gl = os.environ.get("AIC_MUJOCO_GL", "egl").lower()
    if _gl == "egl":
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif _gl == "osmesa":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
except Exception:
    pass

from .observation import ObsConfig, build_obs_dict_from_arrays
from .reward import (
    RewardConfig,
    TerminationConfig,
    check_termination,
    compute_reward,
)


# --------------------------------------------------------------------------- #
# scene geometry
# --------------------------------------------------------------------------- #

# The plug is a simple box (half-extents in metres). Everything else is
# derived from the plug so the cavity always actually fits the plug.
_PLUG_HALF = (0.0035, 0.0035, 0.005)   # 7 x 7 x 10 mm plug

# Per-port-type geometry. Only the depth, clearance, and start/success
# envelopes differ; the wall construction is shared. Values are informed by
# the AIC SDFs (SFP cage / SC port) and PerceptionInsert.INSERTION_DEPTH.
#
#   depth        : cavity depth (m). Plug tip reaches z=-depth when seated.
#   clearance    : radial gap between plug and interior wall (m). Tighter =
#                  harder / more contact-rich (aligned with ISO 286 upper
#                  bound ~0.5-0.6 mm from IndustReal).
#   wall         : wall thickness (m).
#   full_xy_max  : level-1 start xy miss radius (m).
#   full_yaw_max : level-1 start yaw range (rad).
#   success_frac : insertion fraction (of depth) that counts as success.
#   success_xy   : xy radial tolerance for success (m).
_PORT_GEOMETRY = {
    # SC: shallow, 16 mm insertion depth.
    "sc": {
        "depth": 0.016,
        "clearance": 0.0006,
        "wall": 0.002,
        "color": "0.30 0.30 0.40 1",   # bluish plastic
        "full_xy_max": 0.003,          # ±3 mm sideways miss at level 1
        "full_yaw_max": np.deg2rad(15.0),
        "success_frac": 0.70,          # tip ≥70 % of depth in
        "success_xy": 0.0015,          # 1.5 mm radial tolerance
    },
    # SFP: deep narrow cage, 51 mm insertion depth.
    "sfp": {
        "depth": 0.051,
        "clearance": 0.0008,
        "wall": 0.002,
        "color": "0.50 0.40 0.30 1",   # brownish metal
        "full_xy_max": 0.003,
        "full_yaw_max": np.deg2rad(10.0),
        "success_frac": 0.65,
        "success_xy": 0.0015,
    },
}


def _derive_geometry(port_type: str) -> dict:
    """Compute all derived scene/curriculum anchors for a port type."""
    g = _PORT_GEOMETRY[port_type]
    ph_x, ph_y, ph_z = _PLUG_HALF
    depth = g["depth"]
    interior_half_x = ph_x + g["clearance"]
    interior_half_y = ph_y + g["clearance"]
    # Plug body-origin z when fully seated (tip at cavity bottom z=-depth).
    seated_center_z = -depth + ph_z
    # z (of plug body origin) at which the tip has entered by `success_frac`.
    # tip_z = center_z - ph_z ; tip at -success_frac*depth  → center threshold:
    success_z_max = -g["success_frac"] * depth + ph_z
    # Level-1 start: tip a hair above the mouth (center z slightly positive).
    mouth_start_z = ph_z - 0.002   # tip ~2 mm above entrance
    return {
        **g,
        "interior_half_x": interior_half_x,
        "interior_half_y": interior_half_y,
        "seated_center_z": seated_center_z,
        "success_z_max": success_z_max,
        "mouth_start_z": mouth_start_z,
    }


def _build_mjcf(port_type: str, meshdir: str, cam_res: int) -> str:
    """Generate the MJCF for a port type: an open-top rectangular socket the
    plug can actually be inserted into, plus 3 wrist cameras and a
    velocity-actuated free-joint plug.
    """
    d = _derive_geometry(port_type)
    color = d["color"]
    depth = d["depth"]
    wall = d["wall"]
    ox = d["interior_half_x"]   # interior half-opening x
    oy = d["interior_half_y"]   # interior half-opening y
    ph_x, ph_y, ph_z = _PLUG_HALF

    # Max target velocities (for actuator ctrlrange) from the action scales.
    # pos ±(1.5,1.5,3.5) mm and rot ±(0.08,0.08,0.12) rad per dt=0.05 s.
    return f"""\
<mujoco model="aic_last_inch_{port_type}">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <option timestep="0.005" gravity="0 0 0" integrator="implicitfast"/>
  <default>
    <geom contype="1" conaffinity="1" solref="0.008 1" solimp="0.95 0.99 0.001"
          friction="0.4 0.01 0.001"/>
  </default>
  <worldbody>
    <light pos="0 0 0.3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="left_cam"   pos="-0.012 0 0.06" mode="fixed" fovy="60"
            resolution="{cam_res} {cam_res}"/>
    <camera name="center_cam" pos=" 0     0 0.06" mode="fixed" fovy="60"
            resolution="{cam_res} {cam_res}"/>
    <camera name="right_cam"  pos=" 0.012 0 0.06" mode="fixed" fovy="60"
            resolution="{cam_res} {cam_res}"/>
    <body name="port" pos="0 0 0">
      <!-- bottom wall: the plug tip seats against this at z=-depth -->
      <geom name="port_bottom" type="box"
            pos="0 0 {-depth - wall/2:.5f}"
            size="{ox + wall:.5f} {oy + wall:.5f} {wall/2:.5f}" rgba="{color}"/>
      <!-- 4 side walls forming an OPEN-top rectangular cavity -->
      <geom name="port_wx_pos" type="box"
            pos="{ox + wall/2:.5f} 0 {-depth/2:.5f}"
            size="{wall/2:.5f} {oy + wall:.5f} {depth/2:.5f}" rgba="{color}"/>
      <geom name="port_wx_neg" type="box"
            pos="{-(ox + wall/2):.5f} 0 {-depth/2:.5f}"
            size="{wall/2:.5f} {oy + wall:.5f} {depth/2:.5f}" rgba="{color}"/>
      <geom name="port_wy_pos" type="box"
            pos="0 {oy + wall/2:.5f} {-depth/2:.5f}"
            size="{ox + wall:.5f} {wall/2:.5f} {depth/2:.5f}" rgba="{color}"/>
      <geom name="port_wy_neg" type="box"
            pos="0 {-(oy + wall/2):.5f} {-depth/2:.5f}"
            size="{ox + wall:.5f} {wall/2:.5f} {depth/2:.5f}" rgba="{color}"/>
    </body>
    <body name="plug" pos="0 0 0.006">
      <freejoint name="plug_free"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
      <geom name="plug" type="box" size="{ph_x:.5f} {ph_y:.5f} {ph_z:.5f}"
            rgba="0.9 0.5 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="plug_free" gear="1 0 0 0 0 0" kv="100" ctrlrange="-0.06 0.06"/>
    <velocity name="vy" joint="plug_free" gear="0 1 0 0 0 0" kv="100" ctrlrange="-0.06 0.06"/>
    <velocity name="vz" joint="plug_free" gear="0 0 1 0 0 0" kv="100" ctrlrange="-0.12 0.12"/>
    <velocity name="wx" joint="plug_free" gear="0 0 0 1 0 0" kv="5"   ctrlrange="-2.5 2.5"/>
    <velocity name="wy" joint="plug_free" gear="0 0 0 0 1 0" kv="5"   ctrlrange="-2.5 2.5"/>
    <velocity name="wz" joint="plug_free" gear="0 0 0 0 0 1" kv="5"   ctrlrange="-3.0 3.0"/>
  </actuator>
</mujoco>
"""


def _make_mjcf_path(port_type: str = "sc", cam_res: int = 32) -> Path:
    """Write the MJCF for a port type to a tmpdir and return its path.

    Cache key is (port_type, cam_res, mjcf-content-hash) so any scene edit
    invalidates the cached file.
    """
    import hashlib
    cache_dir = Path(os.environ.get("AIC_MJCF_CACHE_DIR", tempfile.gettempdir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    xml = _build_mjcf(port_type, str(cache_dir), cam_res)
    h = hashlib.sha1(xml.encode()).hexdigest()[:8]
    f = cache_dir / f"aic_last_inch_{port_type}_cam{cam_res}_{h}.xml"
    if not f.exists():
        f.write_text(xml)
    return f


def _make_mjcf_path_legacy(cam_res: int = 32) -> Path:
    """Legacy stub — kept for any importer of the old name."""
    return _make_mjcf_path("sc", cam_res)


# --------------------------------------------------------------------------- #
# env config
# --------------------------------------------------------------------------- #


@dataclass
class EnvConfig:
    obs: ObsConfig = ObsConfig()
    reward: RewardConfig = RewardConfig()
    term: TerminationConfig = TerminationConfig()
    pos_scale: tuple = (0.0015, 0.0015, 0.0035)
    rot_scale: tuple = (0.08, 0.08, 0.12)
    dt: float = 0.05
    physics_timestep: float = 0.005     # → n_substeps = dt / physics_timestep
    # Port type — selects MJCF geometry + envelopes. Auto-populated by
    # for_port(); left None uses the explicit fields below.
    port_type: Optional[str] = None
    # start envelope (level-1 / "full") — derived from port geometry by for_port()
    start_xy_max_m: float = 0.003
    start_yaw_max_rad: float = np.deg2rad(15.0)
    # success region — derived from port geometry by for_port()
    success_xy_max_m: float = 0.0015
    success_z_max_m: float = -0.006     # plug-origin z below which = seated
    success_force_z_n: float = 0.2      # min contact force to confirm a seat

    def for_port(self, port_type: str) -> "EnvConfig":
        """Return a copy with start/success fields derived from geometry."""
        d = _derive_geometry(port_type)
        return dataclasses.replace(self,
            port_type=port_type,
            start_xy_max_m=d["full_xy_max"],
            start_yaw_max_rad=d["full_yaw_max"],
            success_xy_max_m=d["success_xy"],
            success_z_max_m=d["success_z_max"],
        )


class LastInchInsertEnv(gym.Env):
    """MuJoCo gym env for image-SAC last-inch insertion training.

    Observation space (Dict):
        image           (H, W, n_cams*3) uint8
        force           (3,)             float32   world-frame contact force
        tcp_pose        (7,)             float32   plug pose [xyz, wxyz]
        port_pose       (7,)             float32   port entrance pose (origin)
        tcp_pose_err    (6,)             float32   [Δxyz_port, axis·angle]
        last_action     (6,)             float32   previous action

    Action space: Box(6,) in [-1, 1], scaled to a per-step pose delta and
    applied as a velocity target to the plug's free joint.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig = EnvConfig(), port_type: str = "sc"):
        super().__init__()
        if not _HAS_MUJOCO:
            raise RuntimeError("mujoco is required for LastInchInsertEnv")
        if port_type not in _PORT_GEOMETRY:
            raise ValueError(f"unknown port_type {port_type!r}; "
                             f"supported: {list(_PORT_GEOMETRY)}")
        if cfg.port_type is None:
            cfg = cfg.for_port(port_type)
        # Always 3 wrist cams (matching AIC Observation.msg), RGB per cam.
        if cfg.obs.n_cams != 3 or cfg.obs.image_ch_per_cam != 3:
            cfg = dataclasses.replace(cfg, obs=dataclasses.replace(
                cfg.obs, n_cams=3, image_ch_per_cam=3))
        self.cfg = cfg
        self.port_type = port_type
        self._geom = _derive_geometry(port_type)

        self._mjcf_path = _make_mjcf_path(port_type, cam_res=cfg.obs.image_h)
        self._model = mujoco.MjModel.from_xml_path(str(self._mjcf_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=cfg.obs.image_h,
                                         width=cfg.obs.image_w)
        self._plug_gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "plug")
        self._n_substeps = max(1, int(round(cfg.dt / cfg.physics_timestep)))

        self._port_q = np.array([1.0, 0, 0, 0], dtype=np.float64)
        self._step_count = 0
        self._reset_mode: str = "curriculum"
        self._curriculum_level: float = 0.0
        self._level_file: Optional[Path] = None
        self._last_action = np.zeros(6, dtype=np.float32)
        self._goal_image = None
        self._f_linger_dwell_s = 0.0
        self._f_z_buf: list[float] = []
        self._prev_depth_norm: float = 0.0
        self._insertion_depth = self._geom["depth"]

        # action → velocity scale (concat pos+rot)
        self._vel_scale = np.concatenate([
            np.asarray(cfg.pos_scale), np.asarray(cfg.rot_scale)]) / cfg.dt

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

        self._capture_goal_image()

    # ------------------------------------------------------------------ #
    # goal image
    # ------------------------------------------------------------------ #

    def _capture_goal_image(self):
        """Render the image at the fully-seated pose (tip at cavity bottom)."""
        self._reset_to_pose(
            tip_xyz=np.array([0.0, 0.0, self._geom["seated_center_z"]]),
            tip_q_wxyz=np.array([1.0, 0, 0, 0]))
        for _ in range(40):
            mujoco.mj_step(self._model, self._data)
        self._goal_image = self._render_image()

    def _reset_to_pose(self, tip_xyz: np.ndarray, tip_q_wxyz: np.ndarray):
        self._data.qpos[:3] = tip_xyz
        self._data.qpos[3:7] = tip_q_wxyz
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def _render_image(self) -> np.ndarray:
        """Render the 3 wrist cameras stacked along the channel axis →
        (H, W, n_cams*3) uint8."""
        frames = []
        for cam_name in ("left_cam", "center_cam", "right_cam"):
            self._renderer.update_scene(self._data, camera=cam_name)
            frames.append(self._renderer.render())
        return np.concatenate(frames, axis=2)

    def render(self) -> np.ndarray:
        """Public render — center-cam RGB frame for video logging."""
        self._renderer.update_scene(self._data, camera="center_cam")
        return self._renderer.render()

    # ------------------------------------------------------------------ #
    # state extraction
    # ------------------------------------------------------------------ #

    def _plug_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._data.qpos[:3].copy(), self._data.qpos[3:7].copy()

    def _force_xyz(self) -> np.ndarray:
        """Sum world-frame contact force on the plug via mj_contactForce.

        cfrc_int / cfrc_ext do not report contact for a free-joint body in a
        usable way here, so we iterate the active contacts, pull the 6-D
        contact wrench (in the contact frame), rotate the force part to world,
        and accumulate with the correct sign for the plug geom.
        """
        d = self._data
        m = self._model
        total = np.zeros(3, dtype=np.float64)
        buf = np.zeros(6, dtype=np.float64)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != self._plug_gid and c.geom2 != self._plug_gid:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            frame = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
            f_world = frame.T @ buf[:3]      # contact-frame force → world
            # contact normal points from geom1 to geom2; force on geom2 is
            # +f_world. Flip if the plug is geom1.
            sign = 1.0 if c.geom2 == self._plug_gid else -1.0
            total += sign * f_world
        return total.astype(np.float32)

    def _tip_xy(self) -> np.ndarray:
        return self._data.qpos[:2].copy()

    def _port_xy(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def _depth_norm(self, z: float) -> float:
        """Insertion fraction in [0, 1]: 0 at/above the entrance, 1 fully seated."""
        denom = abs(self._geom["seated_center_z"]) + 1e-9
        return float(np.clip(-z / denom, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # start-pose sampling (reverse curriculum)
    # ------------------------------------------------------------------ #

    def capture_initial_state(self) -> dict:
        """Sample a start pose. Reverse curriculum by distance-from-goal:

            level 0  → near the SEATED goal (deep, centred, tiny yaw)
            level 1  → above the mouth, ±full xy miss, ±full yaw
            random   → always level-1 envelope
            near_goal→ always level-0 envelope

        Interpolating the *start z* from near-seated (level 0) up to
        above-the-mouth (level 1) is what makes the curriculum "reverse":
        early episodes begin almost inserted so success is dense.
        """
        rng = self.np_random
        g = self._geom
        seated_z = g["seated_center_z"]
        mouth_z = g["mouth_start_z"]

        if self._reset_mode == "near_goal":
            level = 0.0
        elif self._reset_mode == "random":
            level = 1.0
        else:  # "curriculum"
            level = float(np.clip(self._curriculum_level, 0.0, 1.0))

        # near-goal (level 0) anchors
        near_xy = 0.0005
        near_yaw = np.deg2rad(2.0)
        near_z = seated_z + 0.002       # just above fully seated
        # full (level 1) anchors
        full_xy = self.cfg.start_xy_max_m
        full_yaw = self.cfg.start_yaw_max_rad
        full_z = mouth_z                # above the entrance

        xy_r = near_xy + level * (full_xy - near_xy)
        yaw_r = near_yaw + level * (full_yaw - near_yaw)
        z_center = near_z + level * (full_z - near_z)

        xy_err = rng.uniform(-xy_r, xy_r, size=2)
        z = z_center + rng.uniform(-0.0005, 0.0005)   # ±0.5 mm z jitter
        yaw = rng.uniform(-yaw_r, yaw_r)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        return {
            "tip_xyz": np.array([xy_err[0], xy_err[1], z], dtype=np.float64),
            "tip_q_wxyz": np.array([cy, 0, 0, sy], dtype=np.float64),
            "qvel": np.zeros(6, dtype=np.float64),
            "step": 0,
            "last_action": np.zeros(6, dtype=np.float32),
            "f_linger_dwell_s": 0.0,
            "curriculum_level": float(self._curriculum_level),
        }

    def set_curriculum_level(self, level: float) -> None:
        self._curriculum_level = float(np.clip(level, 0.0, 1.0))

    def get_curriculum_level(self) -> float:
        return float(self._curriculum_level)

    def set_reset_mode(self, mode: str) -> None:
        if mode not in ("curriculum", "random", "near_goal"):
            raise ValueError(f"unknown reset mode {mode!r}")
        self._reset_mode = mode

    def set_level_file(self, path: Optional[str]) -> None:
        """Re-read the curriculum level from `path` on every reset (so a
        SubprocVecEnv worker picks up trainer-side level changes)."""
        self._level_file = Path(path) if path is not None else None

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if self._level_file is not None and Path(self._level_file).exists():
            try:
                self._curriculum_level = float(
                    Path(self._level_file).read_text().strip())
            except Exception:
                pass
        state = self.capture_initial_state()
        self._data.qpos[:3] = state["tip_xyz"]
        self._data.qpos[3:7] = state["tip_q_wxyz"]
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        self._last_action = np.zeros(6, dtype=np.float32)
        self._f_linger_dwell_s = 0.0
        self._f_z_buf = []
        # settle a couple of physics ticks so contact forces are visible
        for _ in range(2):
            mujoco.mj_step(self._model, self._data)
        self._prev_depth_norm = self._depth_norm(float(self._data.qpos[2]))
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)

        # Action → per-step pose delta → target velocity, tracked by the
        # velocity-servo actuators. Contact can now genuinely resist the plug.
        target_vel = action * self._vel_scale
        self._data.ctrl[:] = target_vel
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        obs = self._obs()
        image_curr = obs["image"]
        f_xyz = obs["force"]
        self._f_z_buf.append(float(f_xyz[2]))
        tip_xyz, tip_q = self._plug_pose()
        depth_norm = self._depth_norm(float(tip_xyz[2]))

        # Success = seated deep enough AND centred (bottom wall guarantees a
        # z below the seated threshold is genuinely inside the cavity). A tiny
        # contact-force check confirms the plug is actually pressing, not
        # ghosting through.
        term_status = None
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
            bonus_eligible=(term_status == "success"),
            depth_norm=depth_norm,
            prev_depth_norm=self._prev_depth_norm,
        )
        image_l1_norm = breakdown.image_l1_norm
        self._prev_depth_norm = depth_norm

        if term_status is None:
            term_status = check_termination(
                image_l1_norm=image_l1_norm,
                f_z=float(f_xyz[2]),
                off_limit_contact=False,
                step=self._step_count,
                cfg_rew=self.cfg.reward,
                cfg_term=self.cfg.term,
                f_linger_dwell_s=self._f_linger_dwell_s,
                bypass_image_success=True,
            )

        # last_action is updated AFTER the reward (so r_action sees a_{t-1}).
        self._last_action = action.astype(np.float32)

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
            "tip_z_mm": float(tip_xyz[2] * 1000.0),
            "depth_norm": float(depth_norm),
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

    def close(self):
        """Free the MuJoCo renderer explicitly (idempotent).

        Without this, the renderer is only released at GC time via
        Renderer.__del__, which throws noisy AttributeError / EGLError
        during interpreter teardown under SubprocVecEnv.
        """
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
            self._renderer = None


__all__ = ["EnvConfig", "LastInchInsertEnv", "_PORT_GEOMETRY", "_build_mjcf"]
