"""Isaac Lab DirectRLEnv for GPU-vectorized AIC SFP insertion."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from pxr import Gf, Sdf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from ..asset_contract import load_asset_manifest, reset_bank_path
from .task_core import (
    RewardWeights,
    TaskThresholds,
    compute_reward,
    insertion_geometry,
    termination_masks,
)


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HOME_Q = (-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110)
WELD_RELPOSE = (
    -0.000711,
    0.001759,
    0.168213,
    0.577301,
    0.816105,
    -0.021418,
    -0.015395,
)


def _join_prim(root: str, relative: str) -> str:
    return root.rstrip("/") + (relative if relative.startswith("/") else f"/{relative}")


_ASSETS = load_asset_manifest()
_ROBOT_ROOT = "/World/envs/env_.*/Robot"
_WORLD_ROOT = "/World/envs/env_.*/AICWorld"


@configclass
class AICLastInchEnvCfg(DirectRLEnvCfg):
    # 500 Hz physics and controller, 20 Hz policy.
    decimation = 25
    episode_length_s = 10.0
    action_space = 6
    observation_space = 31
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path=_ROBOT_ROOT,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ASSETS["robot_usd"],
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # AIC/MuJoCo add qfrc_bias to the impedance torque. Disabling
                # gravity on robot links is the PhysX-drive equivalent; cable
                # and plug gravity remain active in the world asset.
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=dict(zip(ARM_JOINTS, HOME_Q)),
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINTS[:3]),
                effort_limit_sim=150.0,
                stiffness=100.0,
                damping=40.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINTS[3:]),
                effort_limit_sim=28.0,
                stiffness=50.0,
                damping=15.0,
            ),
        },
    )
    world: AssetBaseCfg = AssetBaseCfg(
        prim_path=_WORLD_ROOT,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ASSETS["world_usd"], activate_contact_sensors=True
        ),
    )
    frames: FrameTransformerCfg = FrameTransformerCfg(
        # FrameTransformer requires a rigid body. gripper_tcp is an MJCF site,
        # so track its parent tool body with the site's exact local offset.
        prim_path=_join_prim(_ROBOT_ROOT, _ASSETS["robot_tool_relpath"]),
        source_frame_offset=OffsetCfg(
            pos=tuple(_ASSETS["robot_tcp_offset_pos"]),
            rot=tuple(_ASSETS["robot_tcp_offset_quat_wxyz"]),
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=_join_prim(_WORLD_ROOT, _ASSETS["world_tip_relpath"]), name="tip"
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=_join_prim(_WORLD_ROOT, _ASSETS["world_tail_relpath"]), name="tail"
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=_join_prim(_WORLD_ROOT, _ASSETS["world_target_relpath"]), name="port"
            ),
        ],
    )
    plug_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=_join_prim(_WORLD_ROOT, _ASSETS["world_tip_relpath"]),
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )

    action_joint_scale = 0.01
    action_joint_limit = 0.35
    reset_mode = os.environ.get("AIC_ISAAC_RESET_MODE", "curriculum")
    curriculum_steps = 500_000
    thresholds = TaskThresholds()
    reward_weights = RewardWeights()


class AICLastInchEnv(DirectRLEnv):
    cfg: AICLastInchEnvCfg

    def __init__(self, cfg: AICLastInchEnvCfg, render_mode: str | None = None, **kwargs):
        self._manifest = load_asset_manifest()
        super().__init__(cfg, render_mode, **kwargs)
        self._arm_ids = torch.as_tensor(
            [self._robot.find_joints(name)[0][0] for name in ARM_JOINTS],
            device=self.device,
            dtype=torch.long,
        )
        frame_names = self._frames.data.target_frame_names
        self._tip_frame_id = frame_names.index("tip")
        self._tail_frame_id = frame_names.index("tail")
        self._port_frame_id = frame_names.index("port")
        tool_body_name = self._manifest["robot_tool_relpath"].rstrip("/").split("/")[-1]
        tool_ids, _ = self._robot.find_bodies(tool_body_name)
        if len(tool_ids) != 1:
            raise RuntimeError(f"expected one robot tool body named {tool_body_name!r}, got {tool_ids}")
        self._tool_body_id = tool_ids[0]
        self._joint_lower = self._robot.data.soft_joint_pos_limits[0, self._arm_ids, 0]
        self._joint_upper = self._robot.data.soft_joint_pos_limits[0, self._arm_ids, 1]
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._reset_anchor = self._joint_targets[:, self._arm_ids].clone()
        self._actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_depth = torch.zeros(self.num_envs, device=self.device)
        self._force_over_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._needs_depth_init = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._state: dict[str, torch.Tensor] = {}
        self._success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._bad_collision = torch.zeros_like(self._success)
        self._force_abort = torch.zeros_like(self._success)
        self._load_reset_bank()

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        # Spawn the converted static world/cable once under env_0, then clone it.
        world_path = self.cfg.world.prim_path.replace("env_.*/", "env_0/")
        self.cfg.world.spawn.func(world_path, self.cfg.world.spawn)
        self._author_plug_weld()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        self._frames = FrameTransformer(self.cfg.frames)
        self._plug_contact = ContactSensor(self.cfg.plug_contact)
        self.scene.sensors["frames"] = self._frames
        self.scene.sensors["plug_contact"] = self._plug_contact
        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _author_plug_weld(self) -> None:
        robot_root = "/World/envs/env_0/Robot"
        world_root = "/World/envs/env_0/AICWorld"
        tool = _join_prim(robot_root, self._manifest["robot_tool_relpath"])
        plug = _join_prim(world_root, self._manifest["world_plug_relpath"])
        joint = UsdPhysics.FixedJoint.Define(
            get_current_stage(), Sdf.Path("/World/envs/env_0/AICPlugWeld")
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(tool)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(plug)])
        p = WELD_RELPOSE
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*p[:3]))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(p[3], Gf.Vec3f(*p[4:])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    def _load_reset_bank(self) -> None:
        path = reset_bank_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"MuJoCo reset bank not found: {path}. Run scripts/export_reset_bank.py "
                "inside the AIC Pixi environment before launching Isaac training."
            )
        bank = np.load(path, allow_pickle=False)
        qpos = np.asarray(bank["qpos"], dtype=np.float32)
        levels = np.asarray(bank["level"], dtype=np.float32)
        bank_source_hash = str(bank["source_sha256"].item())
        manifest_source_hash = str(self._manifest.get("source_sha256", ""))
        if not manifest_source_hash or bank_source_hash != manifest_source_hash:
            raise ValueError(
                "reset bank and imported USD were generated from different MJCF sources; "
                "regenerate both artifacts"
            )
        if qpos.ndim != 2 or qpos.shape[1] != 6 or levels.shape != (qpos.shape[0],):
            raise ValueError(f"invalid reset bank shapes: qpos={qpos.shape}, level={levels.shape}")
        self._bank_qpos = torch.as_tensor(qpos, device=self.device)
        self._bank_levels = torch.as_tensor(levels, device=self.device)

    def _current_curriculum_level(self) -> float:
        if self.cfg.reset_mode == "near_goal":
            return 0.0
        if self.cfg.reset_mode == "random":
            return 1.0
        return min(1.0, float(self.common_step_counter) / max(float(self.cfg.curriculum_steps), 1.0))

    def _sample_bank(self, count: int) -> torch.Tensor:
        level = self._current_curriculum_level()
        if self.cfg.reset_mode == "random":
            candidates = torch.arange(self._bank_qpos.shape[0], device=self.device)
        else:
            distance = (self._bank_levels - level).abs()
            tolerance = max(0.026, float(distance.min().item()) + 1.0e-5)
            candidates = torch.nonzero(distance <= tolerance, as_tuple=False).squeeze(-1)
        picks = candidates[torch.randint(candidates.numel(), (count,), device=self.device)]
        return self._bank_qpos[picks]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone().clamp(-1.0, 1.0)
        current = self._joint_targets[:, self._arm_ids]
        proposed = current + self.cfg.action_joint_scale * self._actions
        low = torch.maximum(self._reset_anchor - self.cfg.action_joint_limit, self._joint_lower)
        high = torch.minimum(self._reset_anchor + self.cfg.action_joint_limit, self._joint_upper)
        self._joint_targets[:, self._arm_ids] = torch.maximum(torch.minimum(proposed, high), low)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(
            self._joint_targets[:, self._arm_ids], joint_ids=self._arm_ids
        )

    def _read_task_state(self) -> dict[str, torch.Tensor]:
        targets_pos = self._frames.data.target_pos_w
        targets_quat = self._frames.data.target_quat_w
        tip_pos = targets_pos[:, self._tip_frame_id]
        tail_pos = targets_pos[:, self._tail_frame_id]
        port_pos = targets_pos[:, self._port_frame_id]
        tip_quat = targets_quat[:, self._tip_frame_id]
        port_quat = targets_quat[:, self._port_frame_id]
        geometry = insertion_geometry(
            tip_pos, tail_pos, tip_quat, port_pos, port_quat, self.cfg.thresholds.seated_depth_m
        )
        # MuJoCo subtracts the gripped-plug F/T baseline. PhysX contact sensors
        # already report contact forces only, so the plug-tip signal is the
        # closest baseline-free equivalent for insertion contact.
        force_w = self._plug_contact.data.net_forces_w
        if force_w.ndim == 3:
            force_w = force_w[:, -1]
        return {
            **geometry,
            "tip_pos_w": tip_pos,
            "tail_pos_w": tail_pos,
            "port_pos_w": port_pos,
            "tip_quat_w": tip_quat,
            "port_quat_w": port_quat,
            "tcp_pos_w": self._frames.data.source_pos_w,
            "tcp_quat_w": self._frames.data.source_quat_w,
            "force_w": force_w,
            "wrench_tcp": self._robot.data.body_incoming_joint_wrench_b[:, self._tool_body_id],
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._state = self._read_task_state()
        self._success, self._bad_collision, self._force_abort, self._force_over_count = termination_masks(
            self._state, self._state["force_w"], self._force_over_count, self.cfg.thresholds
        )
        # Match MuJoCo's requirement for contact at final seating.
        self._success &= torch.linalg.vector_norm(self._state["force_w"], dim=-1) > 1.0e-3
        terminated = self._success | self._bad_collision | self._force_abort
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        previous_depth = torch.where(
            self._needs_depth_init, self._state["depth_norm"], self._previous_depth
        )
        reward, terms = compute_reward(
            self._state,
            self._state["force_w"],
            self._actions,
            self._previous_actions,
            previous_depth,
            self._success,
            self._bad_collision,
            self._force_abort,
            timeout,
            self.cfg.reward_weights,
        )
        self._previous_depth.copy_(self._state["depth_norm"])
        self._needs_depth_init.zero_()
        self.extras["log"] = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.extras["log"].update(
            {
                "task/success_rate": self._success.float().mean(),
                "task/depth_norm": self._state["depth_norm"].mean(),
                "task/lateral_error_m": self._state["lateral_error"].mean(),
                "task/axis_error_rad": self._state["axis_error"].mean(),
                "task/roll_error_rad": self._state["roll_error"].mean(),
                "task/curriculum_level": torch.tensor(
                    self._current_curriculum_level(), device=self.device
                ),
            }
        )
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if not self._state:
            self._state = self._read_task_state()
        qpos = self._robot.data.joint_pos[:, self._arm_ids]
        qvel = self._robot.data.joint_vel[:, self._arm_ids]
        tcp_pos = self._state["tcp_pos_w"] - self.scene.env_origins
        tcp_quat = self._state["tcp_quat_w"]
        wrench = self._state["wrench_tcp"]
        obs = torch.cat((qpos, qvel, tcp_pos, tcp_quat, wrench, self._actions), dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=1.0e3, neginf=-1.0e3)}

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        q_arm = self._sample_bank(len(env_ids))
        qpos = self._robot.data.default_joint_pos[env_ids].clone()
        qvel = torch.zeros_like(qpos)
        qpos[:, self._arm_ids] = q_arm
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(qpos, qvel, None, env_ids)
        self._joint_targets[env_ids] = qpos
        self._reset_anchor[env_ids] = q_arm
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._force_over_count[env_ids] = 0
        self._needs_depth_init[env_ids] = True
        self._state = {}


__all__ = ["AICLastInchEnv", "AICLastInchEnvCfg"]
