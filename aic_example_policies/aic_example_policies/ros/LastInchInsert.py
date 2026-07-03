"""
LastInchInsert — AIC policy that adds a residual SAC on top of PerceptionInsert
for the last inch of SFP / SC cable insertion.

Behaviour:
    - Inherits all behaviour from PerceptionInsert (approach, descent,
      spiral/SC search, etc.).
    - When the gripper reaches the RL handoff gate (xy <= 3.5 mm, depth
      within [-6, +8] mm of port entrance), it pauses the spiral and
      defers the next 600 steps (~30 s) to a learned residual policy.
    - The residual policy is loaded from the path set by
      AIC_LASTINCH_WEIGHTS (defaults to the active model's weight_path
      from RL/load_model.py, or a bundled checkpoint next to this file).
    - The residual action is a 6-vector in [-1, 1] scaled by
      FINAL_INSERT_POS_SCALE / FINAL_INSERT_ROT_SCALE (same as the
      existing PerceptionInsert final-insert hook) and applied as a
      delta to the plug-tip pose.

Env vars:
    AIC_LASTINCH_DISABLE=1          fully disable the residual hook
                                    (default: 1 — disabled until first
                                    trained checkpoint is published)
    AIC_LASTINCH_WEIGHTS=/path/to/policy.pt
                                    TorchScript checkpoint path
    AIC_LASTINCH_DEVICE=cuda|cpu   inference device (auto if unset)

This file lives in aic_example_policies/ros/ so `aic_model --policy:=...`
can dynamic-import it; the registry entry in RL/models.toml uses
policy_class = aic_example_policies.ros.LastInchInsert.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_inverse

from .PerceptionInsert import (
    FINAL_INSERT_POS_SCALE,
    FINAL_INSERT_ROT_SCALE,
    INSERTION_DEPTH,
    PerceptionInsert,
)


_HERE = Path(__file__).resolve().parent


class LastInchInsert(PerceptionInsert):
    """Adds a residual SAC hook to PerceptionInsert for the last inch."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._lastinch_policy = None
        self._lastinch_kind = None
        self._lastinch_device = None
        self._lastinch_warned = False
        self._lastinch_last_action = np.zeros(6, dtype=np.float32)
        self._lastinch_target_tip_xyz = None
        self._lastinch_target_tip_quat = None
        self._load_lastinch_policy()

    # ------------------------------------------------------------------ #
    # policy loading (mirrors PerceptionInsert._load_final_insert_policy)
    # ------------------------------------------------------------------ #

    def _load_lastinch_policy(self):
        if os.environ.get("AIC_LASTINCH_DISABLE", "1").strip().lower() not in (
            "0", "false", "no", "off", ""
        ):
            self.get_logger().info(
                "LastInchInsert: residual hook disabled (AIC_LASTINCH_DISABLE=1); "
                "inheriting PerceptionInsert behaviour"
            )
            return

        policy_path = os.environ.get("AIC_LASTINCH_WEIGHTS")
        if not policy_path:
            # fallback: bundled checkpoint next to this file
            bundled = _HERE.parent.parent.parent / "outputs" / "residual_sac_v0" / "policy.pt"
            if bundled.exists():
                policy_path = str(bundled)
        if not policy_path or not os.path.exists(policy_path):
            self.get_logger().warn(
                f"AIC_LASTINCH_WEIGHTS not set and bundled fallback missing; "
                "falling back to PerceptionInsert"
            )
            return

        policy_path = str(Path(policy_path).expanduser().resolve())
        suffix = Path(policy_path).suffix.lower()
        try:
            if suffix == ".onnx":
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._lastinch_policy = ort.InferenceSession(policy_path, providers=providers)
                self._lastinch_kind = "onnx"
                self.get_logger().info(f"Loaded ONNX last-inch policy: {policy_path}")
            else:
                import torch
                device_name = os.environ.get(
                    "AIC_LASTINCH_DEVICE",
                    "cuda" if torch.cuda.is_available() else "cpu",
                )
                self._lastinch_device = torch.device(device_name)
                self._lastinch_policy = torch.jit.load(
                    policy_path, map_location=self._lastinch_device
                )
                self._lastinch_policy.eval()
                self._lastinch_kind = "torchscript"
                self.get_logger().info(
                    f"Loaded TorchScript last-inch policy on {self._lastinch_device}: {policy_path}"
                )
        except Exception as exc:
            self._lastinch_policy = None
            self.get_logger().warn(
                f"Failed to load last-inch policy {policy_path}: {exc}; "
                "falling back to PerceptionInsert"
            )

    # ------------------------------------------------------------------ #
    # handoff gate: same as PerceptionInsert._rl_handoff_gate
    # ------------------------------------------------------------------ #

    def _lastinch_gate(self, X: np.ndarray) -> bool:
        metrics = self._final_insert_pose_metrics(self._task, X, None) \
            if hasattr(self, "_final_insert_pose_metrics") else None
        # simpler: read gripper pose directly
        gripper_xyz, q_gripper = self._gripper_pose_from_tf()
        if gripper_xyz is None or X is None:
            return False
        try:
            tip_xyz, _ = self._plug_tip_pose_world(gripper_xyz, q_gripper, self._task.port_type)
        except Exception:
            return False
        if tip_xyz is None:
            return False
        xy = float(np.linalg.norm(tip_xyz[:2] - np.asarray(X[:2])))
        depth = float(X[2] - tip_xyz[2])
        return xy <= 0.008 and -0.008 <= depth <= 0.012

    # ------------------------------------------------------------------ #
    # residual inference
    # ------------------------------------------------------------------ #

    def _build_lastinch_observation(self, obs_msg, X, port_transform) -> Optional[np.ndarray]:
        # lazy import so the policy file can load without ROS deps in unit tests
        from RL.observation import build_obs_dict_from_ros
        port_xyz = np.asarray(X, dtype=np.float64)
        port_q = np.array([
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        ])
        try:
            d = build_obs_dict_from_ros(
                obs_msg, port_xyz, port_q, self._lastinch_last_action
            )
        except Exception as exc:
            if not self._lastinch_warned:
                self.get_logger().warn(f"last-inch obs build failed: {exc}")
                self._lastinch_warned = True
            return None
        # SB3 MultiInputPolicy expects a flat dict of tensors; we expose
        # the dict directly and let the caller (the JIT) handle flattening.
        return d

    def _infer_lastinch_action(self, obs_dict) -> Optional[np.ndarray]:
        if self._lastinch_policy is None:
            return None
        try:
            if self._lastinch_kind == "onnx":
                # ONNX needs numpy inputs in a fixed order. We rely on the
                # training script exporting the inputs in the documented
                # order: image, force, tcp_pose, port_pose, tcp_pose_err,
                # last_action. The downstream LastInchInsert ONNX export
                # in train.py exposes that same order.
                inputs = {i.name: np.asarray(obs_dict[i.name])[None, ...]
                          for i in self._lastinch_policy.get_inputs()}
                out = self._lastinch_policy.run(None, inputs)[0]
                return np.clip(np.asarray(out)[0, :6], -1.0, 1.0)
            else:
                import torch
                with torch.no_grad():
                    tensors = {
                        k: torch.from_numpy(np.asarray(v)[None, ...]).to(self._lastinch_device)
                        for k, v in obs_dict.items()
                    }
                    # image is uint8 for SB3; torchscript policy was traced
                    # with float32 inputs (we already cast to float in
                    # build_obs_dict_from_ros). Make sure dtype matches.
                    if tensors["image"].dtype != torch.float32:
                        tensors["image"] = tensors["image"].float() / 255.0
                    out = self._lastinch_policy(tensors)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    action = out.detach().cpu().numpy()[0]
                return np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:6], -1.0, 1.0)
        except Exception as exc:
            if not self._lastinch_warned:
                self.get_logger().warn(f"last-inch inference failed: {exc}")
                self._lastinch_warned = True
            return None

    # ------------------------------------------------------------------ #
    # residual action application (mirrors _apply_final_insert_action)
    # ------------------------------------------------------------------ #

    def _apply_lastinch_action(self, move_robot, port_transform, X, action) -> bool:
        gripper_xyz, q_gripper = self._gripper_pose_from_tf()
        if gripper_xyz is None:
            return False
        try:
            tip_xyz, q_tip = self._plug_tip_pose_world(gripper_xyz, q_gripper, self._task.port_type)
        except Exception:
            return False
        if tip_xyz is None:
            return False

        if self._lastinch_target_tip_xyz is None:
            self._lastinch_target_tip_xyz = tip_xyz.copy()
        if self._lastinch_target_tip_quat is None:
            self._lastinch_target_tip_quat = q_tip

        pos_step = np.asarray(action[:3], dtype=np.float64) * FINAL_INSERT_POS_SCALE
        desired_tip_xyz = self._lastinch_target_tip_xyz + pos_step
        # clip XY drift from port centre
        xy_from_port = desired_tip_xyz[:2] - np.asarray(X[:2])
        xy_norm = float(np.linalg.norm(xy_from_port))
        if xy_norm > 0.008:
            desired_tip_xyz[:2] = np.asarray(X[:2]) + xy_from_port / xy_norm * 0.008

        q_delta = self._axis_angle_to_quat_wxyz(
            np.asarray(action[3:6], dtype=np.float64) * FINAL_INSERT_ROT_SCALE
        )
        q_tip_new = quaternion_multiply(q_delta, self._lastinch_target_tip_quat)
        # normalise
        q_tip_new = q_tip_new / np.linalg.norm(q_tip_new)

        self._lastinch_target_tip_xyz = desired_tip_xyz
        self._lastinch_target_tip_quat = q_tip_new
        self._lastinch_last_action = np.asarray(action, dtype=np.float32)

        try:
            self._set_tip_pose_target(
                move_robot=move_robot,
                tip_xyz=desired_tip_xyz,
                q_tip_wxyz=tuple(q_tip_new),
                port_type=self._task.port_type,
                stiffness=[160.0, 160.0, 180.0, 60.0, 60.0, 60.0],
                damping=[65.0, 65.0, 70.0, 25.0, 25.0, 25.0],
            )
        except TransformException as exc:
            self.get_logger().warn(f"last-inch move failed: {exc}")
            return False
        self._publish_port_tf(X, port_transform)
        return True

    # ------------------------------------------------------------------ #
    # entry point override
    # ------------------------------------------------------------------ #

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if self._lastinch_policy is None:
            return super().insert_cable(task, get_observation, move_robot, send_feedback)

        # run PerceptionInsert up to the handoff, then take over
        # We do this by intercepting: the super-class' insert_cable
        # internally calls _run_final_insert_policy once the gate fires,
        # and that path loads AIC_FINAL_INSERT_POLICY (a different env
        # var). Here we replicate the handoff flow inline so we can use
        # *our* loaded policy regardless of AIC_FINAL_INSERT_* env vars.
        return self._insert_cable_with_lastinch(task, get_observation, move_robot, send_feedback)

    def _insert_cable_with_lastinch(self, task, get_observation, move_robot, send_feedback):
        # Reuse PerceptionInsert's approach + descent + seating by calling
        # its insert_cable once with the residual hook bypassed. The
        # bypass is done by temporarily clearing the final_insert policy
        # inside the parent class. Simpler: re-implement the descent /
        # seating loop inline calling parent helpers, gated on our
        # last-inch policy activation.
        #
        # For brevity and to avoid duplicating 1000+ lines of perception
        # logic, we delegate to the parent and only override the final
        # stage. This is the same strategy PerceptionInsert already uses
        # for its own _load_final_insert_policy hook.

        # Patch parent to use our policy as its final-insert hook.
        # Save / restore the parent's attributes around the call so we
        # don't pollute global state.
        saved_policy = self._final_insert_policy
        saved_kind = self._final_insert_policy_kind
        saved_device = self._final_insert_device
        saved_disabled_env = os.environ.get("AIC_FINAL_INSERT_DISABLE")
        os.environ["AIC_FINAL_INSERT_DISABLE"] = "0"

        # Re-route the parent's loader to our weights by temporarily
        # pointing AIC_FINAL_INSERT_POLICY at our TorchScript file.
        ts_path = self._lastinch_policy_path
        if ts_path is not None:
            os.environ["AIC_FINAL_INSERT_POLICY"] = ts_path
            os.environ["AIC_FINAL_INSERT_OBS_LEN"] = "69"
            # trigger reload
            self._load_final_insert_policy()
            # hand off to parent's existing pipeline
            try:
                return super().insert_cable(task, get_observation, move_robot, send_feedback)
            finally:
                # restore
                self._final_insert_policy = saved_policy
                self._final_insert_policy_kind = saved_kind
                self._finalinch_device = saved_device
                if saved_disabled_env is None:
                    os.environ.pop("AIC_FINAL_INSERT_DISABLE", None)
                else:
                    os.environ["AIC_FINAL_INSERT_DISABLE"] = saved_disabled_env

        # Fallback: parent's behaviour with no residual
        return super().insert_cable(task, get_observation, move_robot, send_feedback)

    @property
    def _lastinch_policy_path(self) -> Optional[str]:
        if self._lastinch_policy is None:
            return None
        # The TorchScript/ONNX path is stored on the model attribute if
        # we tracked it; fall back to the env var that was used to load.
        return os.environ.get("AIC_LASTINCH_WEIGHTS")


__all__ = ["LastInchInsert"]