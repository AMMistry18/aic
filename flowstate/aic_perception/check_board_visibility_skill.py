#!/usr/bin/env python3
"""Flowstate skill that searches for a fully visible task board.

The skill consumes only documented wrist-camera, measured joint-state,
controller-state, wrist-force, and robot-mounted TF data. It performs
image-feedback shoulder-pan centering followed by fixed-orientation upward
clearance through the documented AIC controller interface.
TF lookups are hard-coded to the gripper TCP and the three robot-mounted camera
optical frames relative to ``base_link``; object and scoring frames are never
requested.
"""

from __future__ import annotations

from concurrent import futures
from dataclasses import replace
import math
import threading
import time
import traceback

from absl import app, flags, logging
import grpc
import numpy as np

from intrinsic.skills.python import skill_interface


FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 8003, "Port to listen on.", allow_override=True)
flags.DEFINE_string(
    "skill_service_config_filename",
    "",
    "Path to the generated skill config.",
    allow_override=True,
)


class CheckBoardVisibilitySkill(skill_interface.Skill):
    """Search camera views and perform bounded internal robot motion."""

    def __init__(self):
        super().__init__()
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros.buffer import Buffer
        from tf2_ros.transform_listener import TransformListener

        from aic_perception.config import PerceptionConfig
        from aic_perception.camera_rig import CameraRig
        from aic_perception.robot_motion import RobotMotion

        if not rclpy.ok():
            rclpy.init()
        self.config = PerceptionConfig()
        self.node = rclpy.create_node("check_board_visibility_node")
        self.camera_rig = CameraRig(self.node, self.config)
        self.robot_motion = RobotMotion(self.node, self.camera_rig, self.config)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self.node, spin_thread=False
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="check-board-visibility-ros",
            daemon=True,
        )
        self._spin_thread.start()
        self._execute_lock = threading.Lock()
        logging.info(
            "CheckBoardVisibilitySkill ready: image_topics=%s wrench_topic=%s "
            "joint_state_topic=%s",
            self.config.image_topics,
            self.config.wrench_topic,
            self.config.joint_state_topic,
        )

    def execute(self, request, context):
        from aic_perception import check_board_visibility_skill_pb2 as pb2

        result = pb2.CheckBoardVisibilitySkillResult(success=False)
        if not self._execute_lock.acquire(blocking=False):
            result.message = "another board-visibility invocation is still running"
            return result
        try:
            context.canceller.ready()
            self._execute_inner(
                request.params,
                result,
                cancelled=lambda: context.canceller.cancelled,
            )
        except skill_interface.SkillCancelledError:
            raise
        except Exception:
            result.message = "crashed: " + traceback.format_exc(limit=8)
            logging.error(result.message)
        finally:
            self._execute_lock.release()
        logging.info(
            "board visibility: success=%s seen=%s done=%s edges=%s cam=%s "
            "target=%s force=%.2fN msg=%s",
            result.success,
            result.seen,
            result.done,
            result.edges,
            result.steer_camera,
            result.target_valid,
            result.force_n,
            result.message,
        )
        if not result.success or not result.done:
            raise skill_interface.SkillError(
                4,
                result.message
                or "board search ended before a complete view was reached",
            )
        return result

    def _execute_inner(self, params, result, cancelled) -> None:
        from aic_perception.board_visibility import (
            analyze_board,
            combine_cameras,
            view_quality,
        )
        from aic_perception.robot_motion import (
            base_yaw_target_pose,
            normalize_quaternion,
            quaternion_angular_distance,
        )
        from aic_perception.viewpoint_search import (
            ActionKind,
            AdaptiveViewpointPlanner,
        )

        min_contrast = int(params.min_contrast or 30)
        margin_px = int(params.margin_px or 15)
        ignore_bottom = float(params.ignore_bottom_frac or 0.15)
        step_m = float(params.step_m or 0.04)
        backoff_step_m = float(params.backoff_step_m or step_m)
        timeout_sec = float(params.timeout_seconds or 10)
        min_area_frac = float(params.min_area_frac or 0.005)
        # Official scoring penalizes >20 N sustained for >1 second. Keep a
        # 2 N margin while allowing the observed unloaded ~14 N wrist norm.
        max_force_n = float(params.max_force_n or 18.0)
        max_speed_mps = float(params.max_speed_mps or 0.04)
        publish_hz = float(params.publish_hz or 20.0)
        # The AIC controller continues small corrective motion after a profile
        # completes.  A 6 mm / 8 s default accepts a reached, held viewpoint
        # without treating that harmless residual correction as a collision.
        settle_tolerance_m = float(params.settle_tolerance_m or 0.006)
        move_timeout_sec = float(params.move_timeout_seconds or 8.0)
        max_travel_m = float(params.max_travel_m or 0.80)
        force_delta_n = float(params.force_delta_n or 5.0)
        search_timeout_sec = float(params.search_timeout_seconds or 90.0)
        max_displacement_m = float(params.max_displacement_m or 0.50)
        angular_step_rad = float(params.angular_step_rad or 0.10)
        max_angular_displacement_rad = float(
            params.max_angular_displacement_rad or 1.60
        )
        max_angular_travel_rad = float(params.max_angular_travel_rad or 2.20)
        # Reserve a real edge margin, but do not require 20% of the projected
        # board size on every side.  That old dynamic pad rejected centered,
        # fully visible views as simultaneously clipped at top and bottom.
        context_margin_frac = float(params.context_margin_frac or 0.05)
        min_detail_area_frac = float(params.min_detail_area_frac or 0.06)
        min_rectangularity = float(params.min_rectangularity or 0.50)
        stable_frames = int(params.stable_frames or 2)
        max_angular_speed_rps = float(params.max_angular_speed_rps or 0.20)
        settle_orientation_tolerance_rad = float(
            params.settle_orientation_tolerance_rad or 0.05
        )
        self._validate_parameters(
            min_contrast=min_contrast,
            margin_px=margin_px,
            ignore_bottom=ignore_bottom,
            step_m=step_m,
            backoff_step_m=backoff_step_m,
            timeout_sec=timeout_sec,
            min_area_frac=min_area_frac,
            max_force_n=max_force_n,
            max_speed_mps=max_speed_mps,
            publish_hz=publish_hz,
            settle_tolerance_m=settle_tolerance_m,
            move_timeout_sec=move_timeout_sec,
            max_travel_m=max_travel_m,
            force_delta_n=force_delta_n,
            search_timeout_sec=search_timeout_sec,
            max_displacement_m=max_displacement_m,
            angular_step_rad=angular_step_rad,
            max_angular_displacement_rad=max_angular_displacement_rad,
            max_angular_travel_rad=max_angular_travel_rad,
            context_margin_frac=context_margin_frac,
            min_detail_area_frac=min_detail_area_frac,
            min_rectangularity=min_rectangularity,
            stable_frames=stable_frames,
            max_angular_speed_rps=max_angular_speed_rps,
            settle_orientation_tolerance_rad=settle_orientation_tolerance_rad,
        )
        logging.info(
            "active search parameters: cameras=%s margin_px=%d context=%.3f "
            "ignore_bottom=%.3f step=%.3fm angular_step=%.3frad "
            "settle=%.3fm/%.3frad move_timeout=%.1fs stable_frames=%d "
            "single_camera_completion=true",
            sorted(self.config.camera_frames),
            margin_px,
            context_margin_frac,
            ignore_bottom,
            step_m,
            angular_step_rad,
            settle_tolerance_m,
            settle_orientation_tolerance_rad,
            move_timeout_sec,
            stable_frames,
        )

        planner = AdaptiveViewpointPlanner(
            min_goal_area_frac=min_detail_area_frac,
            max_goal_area_frac=0.45,
            expected_cameras=tuple(sorted(self.config.camera_frames)),
        )
        started_at = time.monotonic()
        deadline = started_at + search_timeout_sec
        baseline_force_xyz = None
        initial_pose = None
        initial_joint1 = None
        saved_action_poses = {}
        complete_camera = None
        complete_streak = 0
        iteration = 0

        while True:
            result.elapsed_seconds = max(0.0, time.monotonic() - started_at)
            if cancelled():
                raise skill_interface.SkillCancelledError(
                    "board search cancelled before the next move"
                )
            remaining_search_sec = deadline - time.monotonic()
            if remaining_search_sec <= 0.0:
                result.success = False
                result.message = (
                    f"adaptive viewpoint search reached its "
                    f"{search_timeout_sec:.1f}s safety deadline"
                )
                return

            snapshot = self.camera_rig.grab(
                timeout_sec=min(timeout_sec, remaining_search_sec)
            )
            if snapshot is None:
                result.success = False
                result.message = (
                    "no fresh wrist-camera frame received from approved topics"
                )
                return
            if baseline_force_xyz is None:
                baseline_force_xyz = snapshot.force_xyz

            result.target_valid = False
            result.target_frame = ""
            result.dx = result.dy = result.dz = 0.0
            result.backoff = False
            result.num_cameras = len(snapshot.frames)
            force_norm = snapshot.force_norm
            result.force_n = float(force_norm or 0.0)
            if snapshot.force_xyz is None:
                result.success = False
                result.message = (
                    "no fresh wrist-force sample; refusing Cartesian motion"
                )
                return
            if not snapshot.frames:
                result.success = False
                result.message = "no supported fresh camera images were decoded"
                return

            reports = {}
            for camera_name, frame in snapshot.frames.items():
                camera_report = analyze_board(
                    frame["image"],
                    margin_px=margin_px,
                    min_area_frac=min_area_frac,
                    ignore_bottom_frac=ignore_bottom,
                    min_contrast=float(min_contrast),
                    min_rectangularity=min_rectangularity,
                    min_detail_area_frac=min_detail_area_frac,
                    context_pad_frac=context_margin_frac,
                )
                reports[camera_name] = camera_report
                logging.info(
                    "iteration=%d %s: seen=%s ready=%s edges=%s area=%.3f "
                    "rect=%.2f quality=%.3f center=(%.3f,%.3f) reasons=%s "
                    "stamp=%s",
                    iteration,
                    camera_name,
                    camera_report.seen,
                    camera_report.full,
                    sorted(camera_report.edges),
                    camera_report.area_frac,
                    camera_report.rectangularity,
                    camera_report.quality_score,
                    camera_report.center_error[0],
                    camera_report.center_error[1],
                    camera_report.failure_reasons,
                    frame["stamp_ns"],
                )

            result.seen = any(item.seen for item in reports.values())
            done, camera, report = combine_cameras(reports)
            result.done = False
            result.success = False
            if camera is not None and report is not None:
                result.steer_camera = camera
                result.edges = ",".join(sorted(report.edges))
                result.area_frac = float(report.area_frac)
                result.rectangularity = float(report.rectangularity)
                result.view_quality = float(view_quality(report))

            # ``full`` now means the dark plate, a dynamic context envelope for
            # upright NIC/SC hardware, adequate pixel scale, and a clean shape
            # are all inside the usable frame. Require the same camera to hold
            # that result across fresh images so one segmentation fluctuation
            # cannot release downstream IVM.
            ready_candidates = [
                (name, item)
                for name, item in reports.items()
                if item.full
                and item.area_frac <= 0.45
            ]
            if ready_candidates:
                ready_camera, ready_report = max(
                    ready_candidates, key=lambda item: view_quality(item[1])
                )
                if complete_camera == ready_camera:
                    complete_streak += 1
                else:
                    complete_camera = ready_camera
                    complete_streak = 1
                result.component_coverage_ready = True
                result.steer_camera = ready_camera
                result.edges = ""
                result.area_frac = float(ready_report.area_frac)
                result.rectangularity = float(ready_report.rectangularity)
                result.view_quality = float(view_quality(ready_report))
            else:
                complete_camera = None
                complete_streak = 0
                result.component_coverage_ready = False

            if complete_streak >= stable_frames:
                result.done = True
                result.success = True
                result.elapsed_seconds = max(0.0, time.monotonic() - started_at)
                result.message = (
                    f"board and padded NIC/SC component envelope held fully "
                    f"visible in {complete_camera} for {complete_streak} fresh "
                    f"frames after {result.moves_executed} adaptive moves"
                )
                return
            if complete_streak:
                logging.info(
                    "coverage-ready stability %d/%d in %s; capturing again",
                    complete_streak,
                    stable_frames,
                    complete_camera,
                )
                continue

            if self._force_exceeded(
                snapshot.force_xyz,
                baseline_force_xyz,
                max_force_n,
                force_delta_n,
            ):
                result.success = False
                result.force_abort = True
                result.message = (
                    f"wrist force guard active at {result.force_n:.2f}N; "
                    "refusing another move"
                )
                return

            # ``analyze_board.full`` already requires physical and dynamic
            # context clearance on every edge.  Do not add a second centroid
            # gate here: a fully visible board need not be image-centered, and
            # the live workcell's base +Z motion does not materially change its
            # normalized vertical centroid.
            planning_reports = {
                name: replace(
                    item,
                    full=bool(
                        item.full
                        and item.area_frac <= 0.45
                    ),
                )
                for name, item in reports.items()
            }
            action = planner.next_action(
                planning_reports,
                deadline_reached=time.monotonic() >= deadline,
            )
            result.last_action = action.kind.value
            if action.terminal:
                result.success = False
                result.message = action.reason
                return

            if action.kind == ActionKind.OBSERVE:
                # J1 uses this no-motion action to require alignment in two
                # independent center-camera frames. The next loop iteration
                # grabs a fresh synchronized camera/force snapshot.
                logging.info(
                    "iteration=%d action=%s id=%d camera=%s reason=%s",
                    iteration,
                    action.kind.value,
                    action.action_id,
                    action.camera,
                    action.reason,
                )
                iteration += 1
                continue

            if action.kind in {
                ActionKind.BASE_YAW,
                ActionKind.HORIZONTAL_SCAN,
            }:
                # Acquisition and visible-board horizontal alignment use the
                # Cartesian pose exactly induced by a small shoulder-pan
                # rotation: rotate both TCP position and orientation about the
                # base-Z axis.  The live controller rejects MODE_JOINT while
                # this Flowstate execution owns Cartesian control, so do not
                # make a doomed target-mode request.  This rigid base-yaw arc
                # is distinct from camera-axis pitch/aim and follows joint 1's
                # local FK while retaining the proven Cartesian safety path.
                try:
                    (px, py, pz), (qx, qy, qz, qw) = self._gripper_pose(
                        timeout_sec
                    )
                except Exception as error:
                    result.success = False
                    result.message = f"permitted gripper TF unavailable: {error}"
                    return
                current_position = np.asarray((px, py, pz), dtype=float)
                current_orientation = normalize_quaternion((qx, qy, qz, qw))
                if initial_pose is None:
                    initial_pose = (current_position.copy(), current_orientation)

                joint1 = self.robot_motion.current_joint1(min(timeout_sec, 2.0))
                if joint1 is None:
                    result.success = False
                    result.message = (
                        "fresh measured /joint_states arm pose unavailable "
                        "for horizontal board centering"
                    )
                    return
                if initial_joint1 is None:
                    initial_joint1 = joint1
                joint_delta = (
                    float(action.aim_direction[0])
                    * angular_step_rad
                    * float(action.angular_scale)
                )
                target_joint1 = joint1 + joint_delta
                if (
                    abs(target_joint1 - initial_joint1)
                    > max_angular_displacement_rad + 1e-9
                ):
                    rejection_reason = (
                        "horizontal board centering exhausted its "
                        f"{max_angular_displacement_rad:.3f}rad start-relative "
                        "joint-1 envelope"
                    )
                    logging.warning(
                        "iteration=%d rejecting action=%s id=%d before motion: %s",
                        iteration,
                        action.kind.value,
                        action.action_id,
                        rejection_reason,
                    )
                    planner.mark_yaw_unavailable(
                        action,
                        reason=rejection_reason,
                    )
                    result.last_action = "base_yaw_preflight_rejected"
                    continue
                if (
                    result.angular_travel_rad + abs(joint_delta)
                    > max_angular_travel_rad + 1e-9
                ):
                    rejection_reason = (
                        "horizontal board centering exhausted its "
                        f"{max_angular_travel_rad:.3f}rad cumulative joint-1 "
                        "travel envelope"
                    )
                    logging.warning(
                        "iteration=%d rejecting action=%s id=%d before motion: %s",
                        iteration,
                        action.kind.value,
                        action.action_id,
                        rejection_reason,
                    )
                    planner.mark_yaw_unavailable(
                        action,
                        reason=rejection_reason,
                        global_unavailable=True,
                    )
                    result.last_action = "base_yaw_preflight_rejected"
                    continue

                # Joint 1 is the base-Z shoulder-pan axis.  Predict its TCP
                # sweep before publishing so a rejected direction never first
                # crosses a start-relative or cumulative Cartesian envelope.
                predicted_position_values, target_orientation = (
                    base_yaw_target_pose(
                        current_position,
                        current_orientation,
                        joint_delta,
                    )
                )
                predicted_position = np.asarray(
                    predicted_position_values, dtype=float
                )
                predicted_step_distance = float(
                    np.linalg.norm(predicted_position - current_position)
                )
                predicted_start_displacement = float(
                    np.linalg.norm(predicted_position - initial_pose[0])
                )
                if predicted_start_displacement > max_displacement_m + 1e-9:
                    rejection_reason = (
                        "predicted joint-1 TCP sweep would exceed the "
                        f"{max_displacement_m:.3f}m start-relative workspace "
                        "envelope"
                    )
                    logging.warning(
                        "iteration=%d rejecting action=%s id=%d before motion: %s",
                        iteration,
                        action.kind.value,
                        action.action_id,
                        rejection_reason,
                    )
                    planner.mark_yaw_unavailable(
                        action,
                        reason=rejection_reason,
                    )
                    result.last_action = "base_yaw_preflight_rejected"
                    continue
                if (
                    result.travel_m + predicted_step_distance
                    > max_travel_m + 1e-9
                ):
                    rejection_reason = (
                        "predicted joint-1 TCP sweep would exceed the "
                        f"{max_travel_m:.3f}m cumulative translation envelope"
                    )
                    logging.warning(
                        "iteration=%d rejecting action=%s id=%d before motion: %s",
                        iteration,
                        action.kind.value,
                        action.action_id,
                        rejection_reason,
                    )
                    planner.mark_yaw_unavailable(
                        action,
                        reason=rejection_reason,
                        global_unavailable=True,
                    )
                    result.last_action = "base_yaw_preflight_rejected"
                    continue

                logging.info(
                    "iteration=%d action=%s id=%d camera=%s "
                    "reason=%s joint1=%.4f nominal_target_joint1=%.4f "
                    "delta=%.4frad control=cartesian_base_yaw_arc",
                    iteration,
                    action.kind.value,
                    action.action_id,
                    action.camera,
                    action.reason,
                    joint1,
                    target_joint1,
                    joint_delta,
                )
                outcome = self.robot_motion.move_smooth(
                    tuple(float(value) for value in predicted_position),
                    target_orientation=target_orientation,
                    max_speed_mps=min(max_speed_mps, 0.04),
                    max_angular_speed_radps=min(max_angular_speed_rps, 0.20),
                    publish_hz=publish_hz,
                    settle_tolerance_m=settle_tolerance_m,
                    settle_angular_tolerance_rad=min(
                        settle_orientation_tolerance_rad,
                        0.015,
                        max(0.006, 0.25 * abs(joint_delta)),
                    ),
                    settle_angular_speed_radps=0.08,
                    timeout_sec=move_timeout_sec,
                    baseline_force_xyz=baseline_force_xyz,
                    max_force_n=max_force_n,
                    force_delta_n=force_delta_n,
                    cancelled=cancelled,
                )
                if outcome.cancelled:
                    raise skill_interface.SkillCancelledError(outcome.message)
                if not outcome.success:
                    result.success = False
                    result.force_abort = outcome.force_abort
                    result.message = outcome.message
                    return

                try:
                    post_position, post_orientation = self._gripper_pose(timeout_sec)
                except Exception as error:
                    result.success = False
                    result.message = (
                        f"permitted gripper TF unavailable after joint-1 yaw: {error}"
                    )
                    return
                post_position_array = np.asarray(post_position, dtype=float)
                step_distance = float(
                    np.linalg.norm(post_position_array - current_position)
                )
                start_displacement = float(
                    np.linalg.norm(post_position_array - initial_pose[0])
                )
                if start_displacement > max_displacement_m + 1e-9:
                    result.success = False
                    result.message = (
                        "joint-1 centering reached the "
                        f"{max_displacement_m:.3f}m workspace envelope"
                    )
                    return
                if result.travel_m + step_distance > max_travel_m + 1e-9:
                    result.success = False
                    result.message = (
                        "joint-1 centering reached the "
                        f"{max_travel_m:.3f}m cumulative translation envelope"
                    )
                    return

                delta = post_position_array - current_position
                result.dx, result.dy, result.dz = (
                    float(value) for value in delta
                )
                result.target_valid = True
                result.target_frame = self.config.base_frame
                result.target.x, result.target.y, result.target.z = post_position
                (
                    result.target.qx,
                    result.target.qy,
                    result.target.qz,
                    result.target.qw,
                ) = post_orientation
                result.moves_executed += 1
                result.travel_m += step_distance
                post_joint1 = self.robot_motion.current_joint1(
                    min(timeout_sec, 0.5)
                )
                measured_joint_delta = (
                    float(post_joint1 - joint1)
                    if post_joint1 is not None
                    else float(joint_delta)
                )
                result.angular_travel_rad += max(
                    abs(measured_joint_delta),
                    float(outcome.angular_distance_rad),
                )
                result.moved = True
                logging.info(
                    "base-yaw arc %d completed: requested=%.4frad "
                    "measured_joint1=%.4frad tcp=%.4fm "
                    "travel=%.4fm angular_travel=%.4frad",
                    result.moves_executed,
                    joint_delta,
                    measured_joint_delta,
                    step_distance,
                    result.travel_m,
                    result.angular_travel_rad,
                )
                iteration += 1
                continue

            if action.kind == ActionKind.ROLLBACK:
                saved_pose = saved_action_poses.get(action.rollback_of)
                if saved_pose is None:
                    result.success = False
                    result.message = (
                        f"planner requested missing rollback pose for action "
                        f"{action.rollback_of}"
                    )
                    return
                (px, py, pz), (qx, qy, qz, qw) = self._gripper_pose(timeout_sec)
                target_position, target_orientation = saved_pose
                delta = np.asarray(target_position, dtype=float) - np.asarray(
                    (px, py, pz), dtype=float
                )
                result.rollback_count += 1
                camera_for_log = action.camera or "saved_pose"
                result.backoff = False
            else:
                if action.camera is None:
                    result.success = False
                    result.message = "adaptive planner produced motion without a camera"
                    return
                try:
                    (px, py, pz), (qx, qy, qz, qw) = self._gripper_pose(
                        timeout_sec
                    )
                    if action.kind != ActionKind.UP_CLEARANCE:
                        image_right, image_down, back_away = (
                            self._camera_axes_in_base(action.camera, timeout_sec)
                        )
                except Exception as error:
                    result.success = False
                    result.message = f"permitted robot-mounted TF unavailable: {error}"
                    return

                current_position = np.asarray((px, py, pz), dtype=float)
                current_orientation = normalize_quaternion((qx, qy, qz, qw))
                if initial_pose is None:
                    initial_pose = (current_position.copy(), current_orientation)

                if action.kind == ActionKind.UP_CLEARANCE:
                    # This phase is deliberately not optical-axis aiming.  In
                    # the AIC workcell base +Z is vertical clearance from the
                    # task board.  Holding the measured quaternion prevents IK
                    # from folding the elbow merely to rotate the wrist camera.
                    delta = np.asarray(
                        (
                            0.0,
                            0.0,
                            backoff_step_m
                            * float(action.translation_scale)
                            * float(action.axial_direction),
                        ),
                        dtype=float,
                    )
                else:
                    image_direction = np.asarray(
                        action.image_direction, dtype=float
                    )
                    image_delta = (
                        step_m
                        * float(action.translation_scale)
                        * (
                            image_direction[0]
                            * np.asarray(image_right, dtype=float)
                            + image_direction[1]
                            * np.asarray(image_down, dtype=float)
                        )
                    )
                    axial_delta = (
                        backoff_step_m
                        * float(action.translation_scale)
                        * float(action.axial_direction)
                        * np.asarray(back_away, dtype=float)
                    )
                    delta = image_delta + axial_delta
                target_position = tuple(
                    float(value) for value in current_position + delta
                )

                target_orientation = current_orientation
                if action.kind != ActionKind.UP_CLEARANCE:
                    aim_direction = np.asarray(
                        action.aim_direction, dtype=float
                    )
                    aim_norm = float(np.linalg.norm(aim_direction))
                    if aim_norm > 1e-9 and action.angular_scale > 0.0:
                        aim_direction /= aim_norm
                        camera_forward = -np.asarray(back_away, dtype=float)
                        if action.kind == ActionKind.CAMERA_ROLL:
                            # J4 is a roll about the optical axis, not a
                            # pitch motion about a camera-plane axis.
                            rotation_axis = camera_forward * float(aim_direction[0])
                        else:
                            desired_ray_shift = (
                                aim_direction[0]
                                * np.asarray(image_right, dtype=float)
                                + aim_direction[1]
                                * np.asarray(image_down, dtype=float)
                            )
                            rotation_axis = np.cross(
                                camera_forward, desired_ray_shift
                            )
                        axis_norm = float(np.linalg.norm(rotation_axis))
                        if axis_norm > 1e-9:
                            target_orientation = self._rotate_orientation_in_base(
                                current_orientation,
                                rotation_axis / axis_norm,
                                angular_step_rad * float(action.angular_scale),
                            )

                saved_action_poses[action.action_id] = (
                    tuple(float(value) for value in current_position),
                    current_orientation,
                )
                camera_for_log = action.camera
                result.backoff = (
                    action.kind == ActionKind.UP_CLEARANCE
                    or action.axial_direction > 0.0
                )

            if initial_pose is None:
                # A rollback cannot be the first planner action, but keep this
                # guard explicit for corrupted state.
                result.success = False
                result.message = "adaptive search has no initial TCP pose"
                return

            target_position_array = np.asarray(target_position, dtype=float)
            step_distance = float(np.linalg.norm(np.asarray(delta, dtype=float)))
            step_angle = quaternion_angular_distance(
                (qx, qy, qz, qw), target_orientation
            )
            start_displacement = float(
                np.linalg.norm(target_position_array - initial_pose[0])
            )
            start_angle = quaternion_angular_distance(
                initial_pose[1], target_orientation
            )
            if start_displacement > max_displacement_m + 1e-9:
                result.success = False
                result.message = (
                    f"adaptive search exhausted its {max_displacement_m:.3f}m "
                    "start-relative workspace envelope"
                )
                return
            if result.travel_m + step_distance > max_travel_m + 1e-9:
                result.success = False
                result.message = (
                    f"adaptive search exhausted its {max_travel_m:.3f}m "
                    "cumulative translation envelope"
                )
                return
            if start_angle > max_angular_displacement_rad + 1e-9:
                result.success = False
                result.message = (
                    f"adaptive search exhausted its "
                    f"{max_angular_displacement_rad:.3f}rad start-relative "
                    "orientation envelope"
                )
                return
            if (
                result.angular_travel_rad + step_angle
                > max_angular_travel_rad + 1e-9
            ):
                result.success = False
                result.message = (
                    f"adaptive search exhausted its {max_angular_travel_rad:.3f}rad "
                    "cumulative angular envelope"
                )
                return

            result.dx, result.dy, result.dz = (float(value) for value in delta)
            result.target_valid = True
            result.target_frame = self.config.base_frame
            result.target.x, result.target.y, result.target.z = target_position
            (
                result.target.qx,
                result.target.qy,
                result.target.qz,
                result.target.qw,
            ) = target_orientation

            logging.info(
                "iteration=%d action=%s id=%d camera=%s reason=%s "
                "delta=(%.4f,%.4f,%.4f)m angle=%.4frad "
                "start=(%.4f,%.4f,%.4f)m target=(%.4f,%.4f,%.4f)m",
                iteration,
                action.kind.value,
                action.action_id,
                camera_for_log,
                action.reason,
                result.dx,
                result.dy,
                result.dz,
                step_angle,
                px,
                py,
                pz,
                result.target.x,
                result.target.y,
                result.target.z,
            )

            outcome = self.robot_motion.move_smooth(
                (result.target.x, result.target.y, result.target.z),
                target_orientation=target_orientation,
                max_speed_mps=max_speed_mps,
                max_angular_speed_radps=max_angular_speed_rps,
                publish_hz=publish_hz,
                settle_tolerance_m=settle_tolerance_m,
                settle_angular_tolerance_rad=settle_orientation_tolerance_rad,
                timeout_sec=move_timeout_sec,
                baseline_force_xyz=baseline_force_xyz,
                max_force_n=max_force_n,
                force_delta_n=force_delta_n,
                cancelled=cancelled,
            )
            if outcome.cancelled:
                raise skill_interface.SkillCancelledError(outcome.message)
            if not outcome.success:
                result.success = False
                result.force_abort = outcome.force_abort
                result.message = outcome.message
                return
            result.moves_executed += 1
            result.travel_m += float(outcome.distance_m)
            result.angular_travel_rad += float(outcome.angular_distance_rad)
            result.moved = result.moved or (
                outcome.distance_m > 0.0 or outcome.angular_distance_rad > 0.0
            )
            logging.info(
                "adaptive move %d completed: distance=%.4fm angle=%.4frad "
                "travel=%.4fm angular_travel=%.4frad",
                result.moves_executed,
                outcome.distance_m,
                outcome.angular_distance_rad,
                result.travel_m,
                result.angular_travel_rad,
            )
            iteration += 1

    @staticmethod
    def _validate_parameters(**values) -> None:
        if not 0 <= values["min_contrast"] <= 255:
            raise ValueError("min_contrast must be in [0, 255]")
        if not 0 <= values["margin_px"] <= 4096:
            raise ValueError("margin_px must be in [0, 4096]")
        if not 0.0 <= values["ignore_bottom"] < 0.5:
            raise ValueError("ignore_bottom_frac must be in [0, 0.5)")
        if not 0.001 <= values["step_m"] <= 0.05:
            raise ValueError("step_m must be in [0.001, 0.05] meters")
        if not 0.001 <= values["backoff_step_m"] <= 0.05:
            raise ValueError("backoff_step_m must be in [0.001, 0.05] meters")
        if not 0.1 <= values["timeout_sec"] <= 60.0:
            raise ValueError("timeout_seconds must be in [0.1, 60]")
        if not 0.0001 <= values["min_area_frac"] <= 0.5:
            raise ValueError("min_area_frac must be in [0.0001, 0.5]")
        if not 1.0 <= values["max_force_n"] <= 100.0:
            raise ValueError("max_force_n must be in [1, 100]")
        if not 0.005 <= values["max_speed_mps"] <= 0.05:
            raise ValueError("max_speed_mps must be in [0.005, 0.05]")
        if not 10.0 <= values["publish_hz"] <= 50.0:
            raise ValueError("publish_hz must be in [10, 50]")
        if not 0.001 <= values["settle_tolerance_m"] <= 0.01:
            raise ValueError("settle_tolerance_m must be in [0.001, 0.01]")
        if not 1.0 <= values["move_timeout_sec"] <= 15.0:
            raise ValueError("move_timeout_seconds must be in [1, 15]")
        if not 0.01 <= values["max_travel_m"] <= 1.0:
            raise ValueError("max_travel_m must be in [0.01, 1.0]")
        if not 1.0 <= values["force_delta_n"] <= 15.0:
            raise ValueError("force_delta_n must be in [1, 15]")
        if not 10.0 <= values["search_timeout_sec"] <= 300.0:
            raise ValueError("search_timeout_seconds must be in [10, 300]")
        if not 0.02 <= values["max_displacement_m"] <= 0.5:
            raise ValueError("max_displacement_m must be in [0.02, 0.5]")
        if not 0.005 <= values["angular_step_rad"] <= 0.2:
            raise ValueError("angular_step_rad must be in [0.005, 0.2]")
        if not 0.05 <= values["max_angular_displacement_rad"] <= 2.0:
            raise ValueError(
                "max_angular_displacement_rad must be in [0.05, 2.0]"
            )
        if not 0.05 <= values["max_angular_travel_rad"] <= 3.0:
            raise ValueError("max_angular_travel_rad must be in [0.05, 3.0]")
        if not 0.0 <= values["context_margin_frac"] <= 0.3:
            raise ValueError("context_margin_frac must be in [0, 0.3]")
        if not 0.005 <= values["min_detail_area_frac"] <= 0.4:
            raise ValueError("min_detail_area_frac must be in [0.005, 0.4]")
        if values["min_detail_area_frac"] < values["min_area_frac"]:
            raise ValueError("min_detail_area_frac must be >= min_area_frac")
        if not 0.2 <= values["min_rectangularity"] <= 1.0:
            raise ValueError("min_rectangularity must be in [0.2, 1.0]")
        if not 1 <= values["stable_frames"] <= 5:
            raise ValueError("stable_frames must be in [1, 5]")
        if not 0.02 <= values["max_angular_speed_rps"] <= 0.5:
            raise ValueError("max_angular_speed_rps must be in [0.02, 0.5]")
        if not 0.005 <= values["settle_orientation_tolerance_rad"] <= 0.1:
            raise ValueError(
                "settle_orientation_tolerance_rad must be in [0.005, 0.1]"
            )

    @staticmethod
    def _rotate_orientation_in_base(
        orientation, axis_base, angle_rad: float
    ) -> tuple[float, float, float, float]:
        """Pre-multiply a ROS-order quaternion by a base-frame axis rotation."""
        from aic_perception.robot_motion import normalize_quaternion

        axis = np.asarray(axis_base, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("rotation axis must be a finite three-vector")
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9 or not math.isfinite(angle_rad):
            raise ValueError("rotation axis/angle is invalid")
        axis /= norm
        half = 0.5 * float(angle_rad)
        sine = math.sin(half)
        delta = (
            float(axis[0] * sine),
            float(axis[1] * sine),
            float(axis[2] * sine),
            math.cos(half),
        )
        current = normalize_quaternion(orientation)
        x1, y1, z1, w1 = delta
        x2, y2, z2, w2 = current
        return normalize_quaternion(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            )
        )

    @staticmethod
    def _force_exceeded(
        force_xyz,
        baseline_xyz,
        max_force_n: float,
        force_delta_n: float,
    ) -> bool:
        if force_xyz is None:
            return False
        force = np.asarray(force_xyz, dtype=float)
        if float(np.linalg.norm(force)) >= max_force_n:
            return True
        if baseline_xyz is None:
            return False
        baseline = np.asarray(baseline_xyz, dtype=float)
        return float(np.linalg.norm(force - baseline)) >= force_delta_n

    def _camera_axes_in_base(
        self, camera: str, timeout_sec: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Look up only an allowlisted robot-mounted camera optical frame."""
        from aic_perception.board_visibility import optical_axes_in_base
        from rclpy.duration import Duration
        from rclpy.time import Time

        if camera not in self.config.camera_frames:
            raise ValueError(f"camera {camera!r} is outside the TF allowlist")
        camera_frame = self.config.camera_frames[camera]
        transform = self.tf_buffer.lookup_transform(
            self.config.base_frame,
            camera_frame,
            Time(),
            timeout=Duration(seconds=min(timeout_sec, 3.0)),
        )
        quaternion = transform.transform.rotation
        return optical_axes_in_base(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )

    def _gripper_pose(
        self, timeout_sec: float
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Look up only the allowlisted robot TCP pose in ``base_link``."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        transform = self.tf_buffer.lookup_transform(
            self.config.base_frame,
            self.config.gripper_frame,
            Time(),
            timeout=Duration(seconds=min(timeout_sec, 3.0)),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        values = np.asarray(
            (
                translation.x,
                translation.y,
                translation.z,
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("TCP transform contains non-finite values")
        if float(np.linalg.norm(values[3:])) < 0.5:
            raise ValueError("TCP transform quaternion is not initialized")
        return (
            (float(values[0]), float(values[1]), float(values[2])),
            (float(values[3]), float(values[4]), float(values[5]), float(values[6])),
        )

    # SkillRepository plumbing used by the SDK Python skill service.
    def configure_runtime(self, service_config) -> None:
        from intrinsic.skills.internal import runtime_data
        from aic_perception import check_board_visibility_skill_pb2 as pb2

        self._skill_alias = service_config.skill_description.skill_name
        self._runtime_data = runtime_data.get_runtime_data_from(
            service_config,
            pb2.CheckBoardVisibilitySkillParams.DESCRIPTOR,
        )

    def _check_alias(self, name: str) -> None:
        if name != self._skill_alias:
            from intrinsic.skills.internal import skill_repository

            raise skill_repository.InvalidSkillAliasError(
                f"unknown skill alias: {name}"
            )

    def get_skill(self, name):
        self._check_alias(name)
        return self

    def get_skill_execute(self, name):
        self._check_alias(name)
        return self

    def get_skill_project(self, name):
        self._check_alias(name)
        return self

    def get_skill_runtime_data(self, name):
        self._check_alias(name)
        return self._runtime_data

    def get_skill_aliases(self):
        return [self._skill_alias]


def start_runner(argv):
    logging.info("CheckBoardVisibilitySkill service starting")
    from intrinsic.skills.internal import skill_service_impl
    from intrinsic.skills.proto import skill_service_config_pb2
    from intrinsic.skills.proto import skill_service_pb2_grpc

    if not FLAGS.skill_service_config_filename:
        raise ValueError("--skill_service_config_filename is required")
    service_config = skill_service_config_pb2.SkillServiceConfig()
    with open(FLAGS.skill_service_config_filename, "rb") as config_file:
        service_config.ParseFromString(config_file.read())
    skill_instance = CheckBoardVisibilitySkill()
    skill_instance.configure_runtime(service_config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    projector = skill_service_impl.SkillProjectorServicer(
        skill_instance, None, None, None
    )
    executor = skill_service_impl.SkillExecutorServicer(
        skill_instance, None, None, None
    )
    information = skill_service_impl.SkillInformationServicer(
        service_config.skill_description
    )
    skill_service_pb2_grpc.add_ProjectorServicer_to_server(projector, server)
    skill_service_pb2_grpc.add_ExecutorServicer_to_server(executor, server)
    skill_service_pb2_grpc.add_SkillInformationServicer_to_server(
        information, server
    )

    server.add_insecure_port(f"[::]:{FLAGS.port}")
    server.start()
    logging.info("gRPC server listening on port %s", FLAGS.port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("CheckBoardVisibilitySkill stopped")


if __name__ == "__main__":
    app.run(start_runner, flags_parser=lambda argv: FLAGS(argv, known_only=True))
