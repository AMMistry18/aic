"""Gymnasium env wrapping the REAL MuJoCo scene (`aic_utils/aic_mujoco/mjcf/scene.xml`).

Sim-side backend for last-inch insertion RL, built on the Gazebo->MuJoCo pipeline scene
(UR5e + Robotiq gripper + welded plug + elastic cable + task board). Replaces the earlier
procedural abstraction in `env.py` (free-joint plug + box port) with the actual robot.

Design (per user, 2026-07-02):
  * REVERSE CURRICULUM: reset starts the plug *inserted* in the port (curriculum_level=0)
    and, as level -> 1, retracts it outward along the insertion axis (+ lateral/yaw jitter)
    to span the full last inch. The residual policy is what we train.
  * IMAGE SAC: reward is image-distance to a goal image captured at the inserted pose
    (crude for now; reward shaping is a later step -- currently a simple negative L2).
  * Two models (SFP / SC): `insert_target_body` selects the insertion frame; this scene
    (sfp_sc_cable) has the gripper over the SFP module, so that is the default target.

Two scene subtleties handled here:
  1. Stable reset despite the tool<->plug weld: place the arm (via IK) so gripper_tcp is at
     the sampled pose, then rigidly move the straight cable so the welded plug starts at
     `tool ∘ relpose` (zero weld violation) -> no QACC blow-up. Settle with a
     gravity-compensated PD (arm actuators are torque motors).
  2. Home MUST be the real AIC config, else the wrist buries into the enclosure wall and the
     FT sensor reads ~20 kN. With it, FT reads the true ~10 N gripped-plug load.

NOTE (pending): action/control scheme (joint residual here), reward shaping, and success
termination are placeholders. See `RL/reward.py` for the reward components to port.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces  # type: ignore

import mujoco

os.environ.setdefault("MUJOCO_GL", os.environ.get("AIC_MUJOCO_GL", "egl"))

_DEFAULT_SCENE = str(
    Path(__file__).resolve().parents[1]
    / "aic_utils" / "aic_mujoco" / "mjcf" / "scene.xml"
)

ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
# Real AIC home (aic_engine/config/base_config.yaml home_joint_positions).
AIC_HOME = (-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110)
CAMERAS = ("center_camera", "left_camera", "right_camera")


@dataclass
class SceneEnvConfig:
    scene_path: str = _DEFAULT_SCENE
    home_qpos: tuple = AIC_HOME
    settle_steps: int = 150
    control_substeps: int = 5
    max_episode_steps: int = 200
    # base PD controller (torque motors) + residual action scaling
    kp: float = 200.0
    kd: float = 25.0
    action_joint_scale: float = 0.03   # rad; residual delta per joint at |action|=1
    gripper_ctrl: float = 0.0
    # --- reverse curriculum (start inserted, retract outward) ---
    insert_target_body: str = "sfp_module_link"  # insertion goal frame in the scene
    insert_goal_offset: tuple = (0.0, 0.0, 0.0)  # extra offset on goal tcp (m, world)
    curriculum_level: float = 0.0        # 0 = inserted (goal), 1 = full retract (home)
    jitter_xy_m: float = 0.004           # lateral jitter magnitude at level 1
    jitter_yaw_rad: float = 0.20         # yaw jitter magnitude at level 1
    ik_iters: int = 200
    ik_tol: float = 1e-4
    ik_damping: float = 0.05       # DLS damping (stable near UR5e wrist singularities)
    ik_step_max: float = 0.15      # max |dq| per IK iteration (rad)
    # observation
    include_images: bool = True
    cameras: tuple = CAMERAS   # subset of wrist cams; single cam -> 3ch (clean SB3 CNN)
    image_h: int = 84
    image_w: int = 84


class SceneInsertEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: SceneEnvConfig = SceneEnvConfig()):
        super().__init__()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.scene_path)
        self.data = mujoco.MjData(self.model)

        jid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        self._arm_qadr = np.array([self.model.jnt_qposadr[jid(j)] for j in ARM_JOINTS])
        self._arm_vadr = np.array([self.model.jnt_dofadr[jid(j)] for j in ARM_JOINTS])
        self._tool_id = bid("ati/tool_link")
        self._plug_id = bid("lc_plug_link")
        self._cend_id = bid("cable_end_0")
        self._cfree_adr = self.model.jnt_qposadr[jid("cable_end_0_free")]
        self._tcp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
        self._target_bid = bid(cfg.insert_target_body)
        self._weld_id = [i for i in range(self.model.neq)
                         if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD][0]
        ed = self.model.eq_data[self._weld_id]
        self._rp_pos, self._rp_quat = ed[3:6].copy(), ed[6:10].copy()
        self._ft_force_adr = self._sensor_adr("AtiForceTorqueSensor_force")
        self._ft_torque_adr = self._sensor_adr("AtiForceTorqueSensor_torque")

        self._home = np.asarray(cfg.home_qpos, dtype=np.float64)
        self._curriculum_level = float(cfg.curriculum_level)
        self._level_file: Optional[Path] = None
        self._step_count = 0
        self._last_action = np.zeros(6, np.float32)

        # renderer
        self._renderer, self._cams = None, []
        if cfg.include_images:
            try:
                self._renderer = mujoco.Renderer(self.model, height=cfg.image_h, width=cfg.image_w)
                self._cams = [c for c in cfg.cameras
                              if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c) >= 0]
            except Exception as e:  # pragma: no cover
                print(f"[SceneInsertEnv] renderer unavailable ({e}); images disabled")
                self._renderer = None

        # --- home & goal tcp poses (FK at home, welded plug settled) ---
        self._rigid_home(self._home)
        self._home_tcp = self.data.site_xpos[self._tcp_sid].copy()
        self._home_quat = self._site_quat()
        goal = self.data.xpos[self._target_bid].copy() + np.asarray(cfg.insert_goal_offset)
        self._goal_tcp = goal
        self._goal_quat = self._home_quat.copy()   # keep plug orientation through insertion

        # spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        obs_spaces = {
            "arm_qpos": spaces.Box(-np.pi, np.pi, (6,), np.float32),
            "arm_qvel": spaces.Box(-50.0, 50.0, (6,), np.float32),
            "tcp_pose": spaces.Box(-3.0, 3.0, (7,), np.float32),
            "ft": spaces.Box(-500.0, 500.0, (6,), np.float32),
            "last_action": spaces.Box(-1.0, 1.0, (6,), np.float32),
        }
        if self._renderer is not None and self._cams:
            obs_spaces["image"] = spaces.Box(0, 255, (cfg.image_h, cfg.image_w, 3 * len(self._cams)), np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        # goal image (inserted pose) for the image reward
        self._goal_image = None
        if self._renderer is not None and self._cams:
            self._reset_to_level(0.0, jitter=False)
            self._goal_image = self._render_image().astype(np.float32)

    # ------------------------------------------------------------------ #
    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else None

    def _site_quat(self):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self._tcp_sid].reshape(9))
        return q

    @staticmethod
    def _qmul(a, b):
        r = np.zeros(4); mujoco.mju_mulQuat(r, a, b); return r

    @staticmethod
    def _qinv(a):
        r = np.zeros(4); mujoco.mju_negQuat(r, a); return r

    @staticmethod
    def _qrot(q, v):
        r = np.zeros(3); mujoco.mju_rotVecQuat(r, v, q); return r

    # ------------------------------------------------------------------ #
    def _rigid_home(self, home):
        """Arm to `home`; rigidly move straight cable so welded plug starts at tool∘relpose."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qadr] = home
        mujoco.mj_forward(self.model, self.data)
        xpt, xqt = self.data.xpos[self._tool_id].copy(), self.data.xquat[self._tool_id].copy()
        des_q = self._qmul(xqt, self._rp_quat)
        des_p = xpt + self._qrot(xqt, self._rp_pos)
        xpp, xqp = self.data.xpos[self._plug_id].copy(), self.data.xquat[self._plug_id].copy()
        xpc, xqc = self.data.xpos[self._cend_id].copy(), self.data.xquat[self._cend_id].copy()
        Tq = self._qmul(des_q, self._qinv(xqp))
        Tp = des_p - self._qrot(Tq, xpp)
        self.data.qpos[self._cfree_adr:self._cfree_adr + 3] = Tp + self._qrot(Tq, xpc)
        self.data.qpos[self._cfree_adr + 3:self._cfree_adr + 7] = self._qmul(Tq, xqc)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _ik(self, target_pos, target_quat, q_init):
        """Damped-least-squares 6-DoF IK for gripper_tcp; returns arm qpos."""
        q = np.array(q_init, dtype=np.float64)
        Jp, Jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(self.cfg.ik_iters):
            self.data.qpos[self._arm_qadr] = q
            mujoco.mj_forward(self.model, self.data)
            perr = target_pos - self.data.site_xpos[self._tcp_sid]
            qerr = np.zeros(3)
            mujoco.mju_subQuat(qerr, target_quat, self._site_quat())
            err = np.concatenate([perr, qerr])
            if np.linalg.norm(err) < self.cfg.ik_tol:
                break
            mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self._tcp_sid)
            J = np.vstack([Jp[:, self._arm_vadr], Jr[:, self._arm_vadr]])  # 6x6
            dq = J.T @ np.linalg.solve(J @ J.T + self.cfg.ik_damping * np.eye(6), err)
            n = np.linalg.norm(dq)
            if n > self.cfg.ik_step_max:      # clamp step near singularities
                dq *= self.cfg.ik_step_max / n
            q = q + dq
            for k, adr in enumerate(self._arm_qadr):
                lo, hi = self.model.jnt_range[k]
                q[k] = np.clip(q[k], lo, hi) if self.model.jnt_limited[k] else q[k]
        return q

    def _sample_start_tcp(self, level, jitter=True, rng=None):
        """Interpolate tcp from goal (inserted, level 0) toward home (retracted, level 1)."""
        rng = rng or np.random
        tcp = self._goal_tcp + level * (self._home_tcp - self._goal_tcp)
        quat = self._home_quat.copy()
        if jitter and level > 0:
            tcp = tcp + np.array([rng.uniform(-1, 1) * self.cfg.jitter_xy_m * level,
                                  rng.uniform(-1, 1) * self.cfg.jitter_xy_m * level, 0.0])
            yaw = rng.uniform(-1, 1) * self.cfg.jitter_yaw_rad * level
            dq = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            quat = self._qmul(dq, quat)
        return tcp, quat

    def _reset_to_level(self, level, jitter=True, settle=None):
        tcp, quat = self._sample_start_tcp(level, jitter=jitter, rng=self.np_random)
        q_arm = self._ik(tcp, quat, self._home)
        self._rigid_home(q_arm)
        n = self.cfg.settle_steps if settle is None else settle
        for _ in range(n):
            self.data.ctrl[:6] = self._base_torque(q_arm)
            self.data.ctrl[6] = self.cfg.gripper_ctrl
            mujoco.mj_step(self.model, self.data)
        self._reset_arm_target = q_arm

    def _base_torque(self, target):
        q = self.data.qpos[self._arm_qadr]; qd = self.data.qvel[self._arm_vadr]
        return self.cfg.kp * (target - q) - self.cfg.kd * qd + self.data.qfrc_bias[self._arm_vadr]

    # ------------------------------------------------------------------ #
    def set_curriculum_level(self, level: float):
        self._curriculum_level = float(np.clip(level, 0.0, 1.0))

    def get_curriculum_level(self) -> float:
        return self._curriculum_level

    def set_level_file(self, path: Optional[str]):
        self._level_file = Path(path) if path else None

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if self._level_file is not None and self._level_file.exists():
            try:
                self._curriculum_level = float(self._level_file.read_text().strip())
            except Exception:
                pass
        self._reset_to_level(self._curriculum_level, jitter=True)
        self._step_count = 0
        self._last_action = np.zeros(6, np.float32)
        return self._obs(), {"curriculum_level": self._curriculum_level}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64).reshape(6), -1.0, 1.0)
        target = self._reset_arm_target + action * self.cfg.action_joint_scale
        for _ in range(self.cfg.control_substeps):
            self.data.ctrl[:6] = self._base_torque(target)
            self.data.ctrl[6] = self.cfg.gripper_ctrl
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._last_action = action.astype(np.float32)
        obs = self._obs()
        reward = self._reward(obs)
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        info = {"tcp_pos": obs["tcp_pose"][:3], "curriculum_level": self._curriculum_level}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _reward(self, obs) -> float:
        # Placeholder "image SAC" reward: negative normalized L2 to the goal image.
        # Reward shaping (image/force/xy/action/lateral/done) is a later step.
        if self._goal_image is None or "image" not in obs:
            return 0.0
        diff = obs["image"].astype(np.float32) - self._goal_image
        return float(-np.mean(diff * diff) / (255.0 ** 2))

    def _render_image(self):
        frames = []
        for cam in self._cams:
            self._renderer.update_scene(self.data, camera=cam)
            frames.append(self._renderer.render())
        return np.concatenate(frames, axis=2)

    def _obs(self) -> dict:
        d = self.data
        tcp_pos = d.site_xpos[self._tcp_sid].copy()
        ft = np.zeros(6)
        if self._ft_force_adr is not None:
            ft[:3] = d.sensordata[self._ft_force_adr:self._ft_force_adr + 3]
        if self._ft_torque_adr is not None:
            ft[3:] = d.sensordata[self._ft_torque_adr:self._ft_torque_adr + 3]
        obs = {
            "arm_qpos": d.qpos[self._arm_qadr].astype(np.float32),
            "arm_qvel": d.qvel[self._arm_vadr].astype(np.float32),
            "tcp_pose": np.concatenate([tcp_pos, self._site_quat()]).astype(np.float32),
            "ft": ft.astype(np.float32),
            "last_action": self._last_action.copy(),
        }
        if self._renderer is not None and self._cams:
            obs["image"] = self._render_image().astype(np.uint8)
        return obs

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera=self._cams[0] if self._cams else -1)
        return self._renderer.render()[:, :, :3]

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


__all__ = ["SceneEnvConfig", "SceneInsertEnv"]


if __name__ == "__main__":
    cfg = SceneEnvConfig(include_images=bool(int(os.environ.get("AIC_IMAGES", "1"))))
    env = SceneInsertEnv(cfg)
    print("obs keys:", list(env.observation_space.spaces.keys()))
    print("goal_tcp:", np.round(env._goal_tcp, 4), "home_tcp:", np.round(env._home_tcp, 4))
    for lvl in (0.0, 0.5, 1.0):
        env.set_curriculum_level(lvl)
        obs, info = env.reset(seed=0)
        maxv = 0.0
        for _ in range(40):
            obs, r, term, trunc, _ = env.step(env.action_space.sample() * 0.3)
            maxv = max(maxv, float(np.abs(env.data.qvel).max()))
        img = obs.get("image")
        print(f"  level={lvl}: tcp={np.round(obs['tcp_pose'][:3],3)} |ft|={np.linalg.norm(obs['ft'][:3]):.2f} "
              f"max|qvel|={maxv:.2f} reward={r:.4f} image={None if img is None else img.shape}")
    env.close()
    print("CURRICULUM SMOKE OK")
