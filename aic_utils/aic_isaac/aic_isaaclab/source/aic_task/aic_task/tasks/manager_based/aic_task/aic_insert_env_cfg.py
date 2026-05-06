# AIC insertion env. Extends AIC-Task-v0 scene + actions, retargets
# command/observation/reward to port-frame insertion.
#
# NOTE: does NOT call super().__post_init__() — the parent AICTaskEnvCfg
# hardcodes references to reward fields our InsertRewardsCfg doesn't have.

import math

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .aic_task_env_cfg import AICTaskEnvCfg
from .mdp.events import randomize_board_and_parts, randomize_dome_light

INSERTION_DEPTH = {"sc": 0.016, "sfp": 0.046}

TARGETS = {
    "sc": {"scene_name": "sc_port", "depth": 0.016},
    "sc2": {"scene_name": "sc_port_2", "depth": 0.016},
    "sfp": {"scene_name": "nic_card", "depth": 0.046},
}

# Plug-tip transform in wrist_3_link frame from PerceptionInsert priors.
_PLUG_TIP_IN_WRIST = {
    "sc": ((0.04026, 0.00907, 0.14939), (0.85472, 0.01261, -0.51889, -0.00920)),
    "sc2": ((0.04026, 0.00907, 0.14939), (0.85472, 0.01261, -0.51889, -0.00920)),
    "sfp": ((0.05631, 0.00137, 0.14857), (0.86526, 0.01717, -0.50059, -0.02774)),
}


@configclass
class InsertObservationsCfg:
    """Privileged teacher observations for final alignment and insertion."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        tip_pose_error = ObsTerm(
            func=mdp.tip_pose_error_port,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        tip_axes_port = ObsTerm(
            func=mdp.tip_axes_port,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        tcp_vel_port = ObsTerm(
            func=mdp.tcp_velocity_port,
            params={"target_key": "${TARGET_KEY}"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
        )

        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class InsertRewardsCfg:
    approach_xy_coarse = RewTerm(
        func=mdp.tip_to_port_xy_l2, weight=-2.0,
        params={"target_key": "${TARGET_KEY}"},
    )
    approach_xy_progress = RewTerm(
        func=mdp.tip_to_port_xy_progress,
        weight=24.0,
        params={"target_key": "${TARGET_KEY}", "clip": 0.02},
    )
    approach_xy_fine = RewTerm(
        func=mdp.tip_to_port_xy_exp,
        weight=18.0,
        params={"target_key": "${TARGET_KEY}", "sigma": 0.004},
    )
    axis_alignment = RewTerm(
        func=mdp.plug_port_axis_alignment,
        weight=4.0,
        params={"target_key": "${TARGET_KEY}"},
    )
    twist_alignment = RewTerm(
        func=mdp.plug_port_twist_alignment,
        weight=8.0,
        params={"target_key": "${TARGET_KEY}"},
    )
    rotation_dense = RewTerm(
        func=mdp.tip_rotation_exp,
        weight=12.0,
        params={"target_key": "${TARGET_KEY}", "sigma": 0.16},
    )
    rotation_progress = RewTerm(
        func=mdp.tip_rotation_progress,
        weight=16.0,
        params={"target_key": "${TARGET_KEY}", "clip": 0.05},
    )
    approach_z_dense = RewTerm(
        func=mdp.tip_to_port_z_inv,
        weight=8.0,
        params={"target_key": "${TARGET_KEY}", "xy_gate": 0.01},
    )
    approach_descend = RewTerm(
        func=mdp.tip_to_port_descend_reward,
        weight=12.0,
        params={"target_key": "${TARGET_KEY}", "sigma_z": 0.015, "xy_gate": 0.006},
    )
    approach_z_progress = RewTerm(
        func=mdp.tip_to_port_z_progress,
        weight=20.0,
        params={"target_key": "${TARGET_KEY}", "xy_gate": 0.006, "clip": 0.01},
    )
    insertion_depth = RewTerm(
        func=mdp.insertion_depth_reward,
        weight=160.0,
        params={"target_key": "${TARGET_KEY}", "xy_gate": 0.0035, "rot_gate": 0.16},
    )
    success_bonus = RewTerm(
        func=mdp.insertion_success_bonus,
        weight=220.0,
        params={"target_key": "${TARGET_KEY}", "xy_thresh": 0.002, "rot_thresh": 0.12},
    )
    alive = RewTerm(func=mdp.alive_bonus, weight=0.02)
    force_penalty = RewTerm(
        func=mdp.wrist_force_l2, weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)


@configclass
class InsertTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    insertion_success = DoneTerm(
        func=mdp.insertion_success,
        params={"target_key": "${TARGET_KEY}"},
    )
    drift_failure = DoneTerm(
        func=mdp.drift_failure,
        params={"target_key": "${TARGET_KEY}", "xy_thresh": 0.015},
    )
    force_failure = DoneTerm(
        func=mdp.force_failure,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]),
            "force_thresh": 20.0,
            "duration_steps": 30,
        },
    )


@configclass
class InsertEventCfg:
    randomize_light = EventTerm(
        func=randomize_dome_light,
        mode="reset",
        params={
            "intensity_range": (1500.0, 3500.0),
            "color_range": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        },
    )

    randomize_board_and_parts = EventTerm(
        func=randomize_board_and_parts,
        mode="reset",
        params={
            "board_scene_name": "task_board",
            "board_default_pos": (0.2837, 0.229, 0.0),
            "board_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005)},
            "parts": [
                {"scene_name": "sc_port", "offset": (0.0067, -0.0362, 0.005), "pose_range": {"x": (-0.005, 0.02)}},
                {"scene_name": "sc_port_2", "offset": (0.0076, -0.0783, 0.005), "pose_range": {"x": (-0.005, 0.02)}},
                {"scene_name": "nic_card", "offset": (-0.03235, 0.02329, 0.0743), "pose_range": {"y": (0.0, 0.12)}, "snap_step": {"y": 0.04}},
            ],
            "sync_usd_xforms": False,
        },
    )

    reset_to_preinsertion = EventTerm(
        func=mdp.reset_to_preinsertion_pose,
        mode="reset",
        params={
            "target_key": "${TARGET_KEY}",
            "curriculum_steps": 150000,
            "joint_noise_start": 0.008,
            "joint_noise_end": 0.003,
            "xy_error_start": 0.008,
            "xy_error_end": 0.0025,
            "z_above_start": 0.020,
            "z_above_end": 0.006,
            "roll_pitch_start": 0.18,
            "roll_pitch_end": 0.05,
            "yaw_start": 0.30,
            "yaw_end": 0.08,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_to_preinsertion_pose,
        mode="reset",
        params={
            "target_key": "${TARGET_KEY}",
            "curriculum_steps": 150000,
            "joint_noise_start": 0.008,
            "joint_noise_end": 0.003,
            "xy_error_start": 0.008,
            "xy_error_end": 0.0025,
            "z_above_start": 0.020,
            "z_above_end": 0.006,
            "roll_pitch_start": 0.18,
            "roll_pitch_end": 0.05,
            "yaw_start": 0.30,
            "yaw_end": 0.08,
        },
    )


@configclass
class AICInsertEnvCfg(AICTaskEnvCfg):
    """SC port insertion env. Inherits scene + actions from AICTaskEnvCfg.
    Overrides __post_init__ to skip parent logic that references reward/command
    fields which don't exist in our subclass."""

    observations: InsertObservationsCfg = InsertObservationsCfg()
    rewards: InsertRewardsCfg = InsertRewardsCfg()
    terminations: InsertTerminationsCfg = InsertTerminationsCfg()
    events: InsertEventCfg = InsertEventCfg()

    def __post_init__(self):
        # Intentionally NOT calling super().__post_init__() — it references
        # reward fields (end_effector_position_tracking, etc.) that we don't
        # define in InsertRewardsCfg.

        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        target_key = getattr(self, "_target_key", "sc")

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=_PLUG_TIP_IN_WRIST[target_key][0],
                rot=_PLUG_TIP_IN_WRIST[target_key][1],
            ),
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="svd",
                ik_params={"k_val": 1.0, "min_singular_value": 1e-5},
            ),
            scale=(0.008, 0.008, 0.012, 0.16, 0.16, 0.20),
        )

        # Command manager still needs the ee_pose command registered even
        # though rewards ignore it.
        self.commands.ee_pose.body_name = "wrist_3_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        self._resolve_target_key(target_key)

    def _resolve_target_key(self, target_key: str):
        """Fill ${TARGET_KEY} placeholders in obs/reward/termination params."""
        for group in [self.observations.policy, self.rewards, self.terminations, self.events]:
            for name in dir(group):
                term = getattr(group, name, None)
                params = getattr(term, "params", None)
                if isinstance(params, dict) and params.get("target_key") == "${TARGET_KEY}":
                    params["target_key"] = target_key


@configclass
class AICInsertSCEnvCfg(AICInsertEnvCfg):
    def __post_init__(self):
        self._target_key = "sc"
        super().__post_init__()


@configclass
class AICInsertSC2EnvCfg(AICInsertEnvCfg):
    def __post_init__(self):
        self._target_key = "sc2"
        super().__post_init__()


@configclass
class AICInsertSFPEnvCfg(AICInsertEnvCfg):
    def __post_init__(self):
        self._target_key = "sfp"
        super().__post_init__()
