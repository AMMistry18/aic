"""Kinematic terminal-insertion environment for last-mile TCP policies."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, quat_rotate


class KinematicLastMileRLEnv(ManagerBasedRLEnv):
    """Manager env that advances the insertion TCP directly in SE(3).

    The upstream controller already delivers the connector tip to the port
    neighborhood. This env keeps Isaac Lab's managers/rewards/observations, but
    treats the final policy action as a small relative TCP motion instead of
    asking the full robot/cable articulation to settle from a synthetic reset.
    """

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        self.action_manager.process_action(action)
        self._apply_kinematic_tip_action(action)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _apply_kinematic_tip_action(self, action: torch.Tensor) -> None:
        if not hasattr(self, "_kinematic_tip_pos_w") or not hasattr(self, "_kinematic_tip_quat_w"):
            return

        scale = torch.tensor(self.cfg.actions.arm_action.scale, device=self.device, dtype=action.dtype).unsqueeze(0)
        command = torch.clamp(action, -1.0, 1.0) * scale

        robot = self.scene["robot"]
        root_quat_w = robot.data.root_quat_w
        delta_pos_w = quat_rotate(root_quat_w, command[:, :3])
        self._kinematic_tip_pos_w[:] = self._kinematic_tip_pos_w + delta_pos_w

        rot_vec_w = quat_rotate(root_quat_w, command[:, 3:])
        angle = torch.norm(rot_vec_w, dim=-1)
        axis = rot_vec_w / torch.clamp(angle.unsqueeze(-1), min=1.0e-6)
        delta_quat = quat_from_angle_axis(angle, axis)
        small = angle < 1.0e-8
        if torch.any(small):
            delta_quat[small] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device, dtype=action.dtype)
        self._kinematic_tip_quat_w[:] = quat_mul(delta_quat, self._kinematic_tip_quat_w)
        self._kinematic_tip_quat_w[:] = self._kinematic_tip_quat_w / torch.clamp(
            torch.norm(self._kinematic_tip_quat_w, dim=-1, keepdim=True),
            min=1.0e-6,
        )
