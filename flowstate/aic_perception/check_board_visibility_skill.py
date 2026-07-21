#!/usr/bin/env python3
"""Flowstate skill that searches for an IVM-ready task-component view.

The skill consumes only documented wrist-camera, measured joint-state,
controller-state, wrist-force, and robot-mounted TF data. It performs
image-feedback shoulder-pan centering, wrist-3 long-axis alignment, center-
camera oblique survey positioning, and upward clearance through the documented AIC
controller interface.
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

    @staticmethod
    def _resolve_survey_target(pb2, params) -> tuple[int, str]:
        """Return a safe protobuf enum number/name pair for one invocation."""

        raw_value = int(getattr(params, "survey_target", 0) or 0)
        enum_wrapper = getattr(
            pb2.CheckBoardVisibilitySkillParams, "SurveyTarget", None
        )
        if enum_wrapper is not None:
            try:
                return raw_value, str(enum_wrapper.Name(raw_value))
            except ValueError:
                logging.warning(
                    "unknown survey_target enum value %d; using UNSPECIFIED",
                    raw_value,
                )
        # The fallback also keeps the wrapper importable with a stale generated
        # pb2 while a development workspace is being rebuilt.
        return 0, "UNSPECIFIED"

    def __init__(self):
        super().__init__()
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros.buffer import Buffer
        from tf2_ros.transform_listener import TransformListener

        from aic_perception.config import PerceptionConfig
        from aic_perception.camera_rig import CameraRig
        from aic_perception.gripper_masks import GripperMaskBank
        from aic_perception.robot_motion import RobotMotion

        if not rclpy.ok():
            rclpy.init()
        self.config = PerceptionConfig()
        self.node = rclpy.create_node("check_board_visibility_node")
        self.camera_rig = CameraRig(self.node, self.config)
        self.gripper_masks = GripperMaskBank()
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
        survey_target_value, survey_target_name = self._resolve_survey_target(
            pb2, request.params
        )
        if not self._execute_lock.acquire(blocking=False):
            result.message = (
                f"survey_target={survey_target_name}: another "
                "board-visibility invocation is still running"
            )
            return result
        try:
            context.canceller.ready()
            self._execute_inner(
                request.params,
                result,
                cancelled=lambda: context.canceller.cancelled,
                survey_target_value=survey_target_value,
                survey_target_name=survey_target_name,
            )
        except skill_interface.SkillCancelledError:
            raise
        except Exception:
            result.message = "crashed: " + traceback.format_exc(limit=8)
            logging.error(result.message)
        finally:
            try:
                handoff_ready = self.robot_motion.prepare_controller_handoff()
                logging.info(
                    "board visibility command stream stopped at measured "
                    "state; handoff_ready=%s",
                    handoff_ready,
                )
            except Exception as error:
                # Never replace the search result or prevent the process-level
                # controller cleanup.  The bridge lease itself is released by
                # Switch To Default Controller, not by this ROS publisher.
                logging.warning(
                    "could not publish final measured-state handoff target: %s",
                    error,
                )
            self._execute_lock.release()
        if not result.message.startswith("survey_target="):
            result.message = (
                f"survey_target={survey_target_name}: {result.message}"
            )
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
        logging.info(
            "board visibility returning normally; "
            "controller_handoff=Switch To Default Controller required"
        )
        # Search/sensor failures are represented by the result rather than a
        # gRPC skill error.  The Flowstate process must always execute
        # ``Switch To Default Controller`` immediately after this skill and
        # only then gate on ``result.success && result.done``.  Raising here
        # aborted a normal Sequence before that cleanup node, leaving the AIC
        # controller bridge's ICON session holding ``arm`` and causing the
        # following Move Robot skill to fail with "Part: 'arm' is already in
        # use."  Cancellation still propagates through SkillCancelledError.
        return result

    def _execute_inner(
        self,
        params,
        result,
        cancelled,
        *,
        survey_target_value: int,
        survey_target_name: str,
    ) -> None:
        from aic_perception.board_visibility import (
            SurveyTargetMode,
            analyze_board,
            combine_cameras,
            ivm_survey_rejection_reasons,
            normalize_survey_target,
            rotation_matrix_from_quaternion,
            survey_view_requirements,
            view_quality,
        )
        from aic_perception.robot_motion import (
            MotionFailure,
            base_yaw_target_pose,
            normalize_quaternion,
            quaternion_angular_distance,
        )
        from aic_perception.viewpoint_search import (
            ActionKind,
            AdaptiveViewpointPlanner,
            survey_tilt_correction,
        )

        min_contrast = int(params.min_contrast or 30)
        margin_px = int(params.margin_px or 15)
        # The calibrated per-camera silhouette replaces the old blanket 15%
        # crop.  A non-zero request value remains available as an additional
        # conservative exclusion band.
        ignore_bottom = float(params.ignore_bottom_frac)
        step_m = float(params.step_m or 0.04)
        backoff_step_m = float(params.backoff_step_m or step_m)
        timeout_sec = float(params.timeout_seconds or 10)
        min_area_frac = float(params.min_area_frac or 0.005)
        # Official scoring penalizes >20 N sustained for >1 second. Keep a
        # 2 N margin while allowing the observed unloaded ~14 N wrist norm.
        max_force_n = float(params.max_force_n or 18.0)
        max_speed_mps = float(params.max_speed_mps or 0.05)
        publish_hz = float(params.publish_hz or 20.0)
        # The AIC controller continues small corrective motion after a profile
        # completes.  A 6 mm / 8 s default accepts a reached, held viewpoint
        # without treating that harmless residual correction as a collision.
        settle_tolerance_m = float(params.settle_tolerance_m or 0.008)
        move_timeout_sec = float(params.move_timeout_seconds or 6.0)
        max_travel_m = float(params.max_travel_m or 0.80)
        force_delta_n = float(params.force_delta_n or 5.0)
        # The clean live trace spent roughly 25 seconds leveling/centering and
        # then consumed the old 90-second budget on 8-second settling cycles.
        # Workspace, travel, force, and per-move limits remain independently
        # bounded, so extend only the overall observation/motion budget.
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
        context_margin_frac = float(params.context_margin_frac or 0.02)
        min_detail_area_frac = float(params.min_detail_area_frac or 0.06)
        min_rectangularity = float(params.min_rectangularity or 0.50)
        stable_frames = int(params.stable_frames or 2)
        max_angular_speed_rps = float(params.max_angular_speed_rps or 0.30)
        settle_orientation_tolerance_rad = float(
            params.settle_orientation_tolerance_rad or 0.05
        )
        target_center_tilt_deg = float(params.target_center_tilt_deg or 32.0)
        center_tilt_tolerance_deg = float(
            params.center_tilt_tolerance_deg or 2.0
        )
        ivm_min_center_board_area_frac = float(
            params.ivm_min_center_board_area_frac or 0.32
        )
        ivm_max_center_board_area_frac = float(
            params.ivm_max_center_board_area_frac or 0.50
        )

        # Explicit component surveys own their camera geometry.  The old
        # proto controls remain the legacy UNSPECIFIED escape hatch; applying
        # a saved 32deg whole-board parameter to a staged SFP source would
        # hide the module top in the lateral views that IVM fuses.
        target_mode = normalize_survey_target(survey_target_name)
        target_geometry_source = "legacy proto/default"
        if target_mode is not SurveyTargetMode.UNSPECIFIED:
            target_requirements = survey_view_requirements(target_mode)
            target_center_tilt_deg = target_requirements.target_tilt_deg
            center_tilt_tolerance_deg = target_requirements.tilt_tolerance_deg
            target_geometry_source = "target-specific IVM profile"
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
            target_center_tilt_deg=target_center_tilt_deg,
            center_tilt_tolerance_deg=center_tilt_tolerance_deg,
            ivm_min_center_board_area_frac=ivm_min_center_board_area_frac,
            ivm_max_center_board_area_frac=ivm_max_center_board_area_frac,
        )
        # Legacy start-relative and cumulative envelopes repeatedly rejected
        # useful viewpoint corrections even though the controller already
        # enforces URDF joint/velocity limits.  Keep accepting the proto fields
        # for backward-compatible Flowstate nodes, but do not use them as
        # policy termination conditions.  Motion remains incremental, measured,
        # cancellable, deadline-bound, and guarded by fresh wrist force.
        max_travel_m = math.inf
        max_displacement_m = math.inf
        max_angular_displacement_rad = math.inf
        max_angular_travel_rad = math.inf
        planner = AdaptiveViewpointPlanner(
            min_goal_area_frac=max(
                ivm_min_center_board_area_frac, min_detail_area_frac
            ),
            max_goal_area_frac=ivm_max_center_board_area_frac,
            min_gripper_clearance_px=32.0,
            auxiliary_min_area_frac=0.12,
            auxiliary_min_rectangularity=0.55,
            auxiliary_min_gripper_clearance_px=32.0,
            auxiliary_context_scale=1.50,
            auxiliary_max_center_error_x=0.25,
            auxiliary_max_center_error_y=0.25,
            max_auxiliary_translates=10,
            survey_confirmation_frames=2,
            expected_cameras=tuple(sorted(self.config.camera_frames)),
            survey_target=survey_target_name,
        )
        logging.info(
            "active search parameters: survey_target=%s(%d) cameras=%s "
            "margin_px=%d context=%.3f "
            "ignore_bottom=%.3f step=%.3fm angular_step=%.3frad "
            "settle=%.3fm/%.3frad move_timeout=%.1fs search_timeout=%.1fs "
            "stable_frames_configured=%d j6_min_ratio=%.2f "
            "j6_confirm_frames=%d j6_tolerance=%.1fdeg "
            "center_survey_tilt=%.1f+/-%.1fdeg center_area=%.2f..%.2f "
            "geometry_source=%s "
            "motion_envelopes=controller_native "
            "completion=two_fresh_synchronized_three_camera_survey_frames",
            survey_target_name,
            survey_target_value,
            sorted(self.config.camera_frames),
            margin_px,
            context_margin_frac,
            ignore_bottom,
            step_m,
            angular_step_rad,
            settle_tolerance_m,
            settle_orientation_tolerance_rad,
            move_timeout_sec,
            search_timeout_sec,
            stable_frames,
            planner.min_long_axis_ratio,
            planner.roll_confirmation_frames,
            planner.roll_align_threshold_deg,
            target_center_tilt_deg,
            center_tilt_tolerance_deg,
            planner.survey_view.min_area_frac,
            planner.survey_view.max_area_frac,
            target_geometry_source,
        )
        started_at = time.monotonic()
        deadline = started_at + search_timeout_sec
        baseline_force_xyz = None
        initial_pose = None
        initial_joint1 = None
        initial_joint6 = None
        saved_action_poses = {}
        iteration = 0
        joint_yaw_available = True
        target_center_tilt_rad = math.radians(target_center_tilt_deg)
        center_tilt_tolerance_rad = math.radians(center_tilt_tolerance_deg)
        min_level_progress_rad = 0.01
        leveling_moves = 0
        level_clearance_applied = False
        level_anchor_joint1 = None
        level_anchor_joint6 = None
        level_vertical_polarity = 1.0
        pending_level_vertical_sample = None

        def motion_cancelled() -> bool:
            return bool(cancelled() or time.monotonic() >= deadline)

        def require_motion_force(current_snapshot):
            """Return a snapshot with fresh force, but only when motion needs it.

            A final complete image or a confirmation-only OBSERVE action does
            not command the robot and must not fail merely because the
            independent wrench subscriber missed that camera instant.  Every
            actual motion still requires a genuinely fresh sample here and in
            ``RobotMotion`` itself.
            """

            nonlocal baseline_force_xyz
            if current_snapshot.force_xyz is not None:
                if baseline_force_xyz is None:
                    baseline_force_xyz = current_snapshot.force_xyz
                return current_snapshot
            force_wait_sec = min(
                timeout_sec, max(0.0, deadline - time.monotonic())
            )
            logging.warning(
                "robot motion requested without force in the camera snapshot; "
                "waiting up to %.1fs for fresh wrist-force feedback",
                force_wait_sec,
            )
            fresh_force = self.camera_rig.wait_for_force_xyz(
                timeout_sec=force_wait_sec,
                max_age_sec=0.5,
            )
            if fresh_force is None:
                return None
            if baseline_force_xyz is None:
                baseline_force_xyz = fresh_force
            logging.info("fresh wrist-force feedback recovered before motion")
            return replace(current_snapshot, force_xyz=fresh_force)

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

            # Images and wrench arrive independently.  Perception-only work is
            # allowed to continue without a force sample; the sample becomes a
            # hard prerequisite only after the planner requests real motion.
            if snapshot.force_xyz is None:
                logging.info(
                    "camera snapshot has no simultaneous fresh wrist-force "
                    "sample; evaluating completion before requesting one"
                )
            elif baseline_force_xyz is None:
                baseline_force_xyz = snapshot.force_xyz

            result.target_valid = False
            result.target_frame = ""
            result.dx = result.dy = result.dz = 0.0
            result.backoff = False
            result.num_cameras = len(snapshot.frames)
            force_norm = snapshot.force_norm
            result.force_n = float(force_norm or 0.0)
            if not snapshot.frames:
                result.success = False
                result.message = "no supported fresh camera images were decoded"
                return

            reports = {}
            for camera_name, frame in snapshot.frames.items():
                gripper_ignore = self.gripper_masks.ignored_pixels(
                    camera_name, frame["image"].shape
                )
                camera_report = analyze_board(
                    frame["image"],
                    margin_px=margin_px,
                    min_area_frac=min_area_frac,
                    ignore_bottom_frac=ignore_bottom,
                    min_contrast=float(min_contrast),
                    min_rectangularity=min_rectangularity,
                    min_detail_area_frac=min_detail_area_frac,
                    context_pad_frac=context_margin_frac,
                    ignore_mask=gripper_ignore,
                    survey_target=survey_target_name,
                )
                reports[camera_name] = camera_report
                survey_reasons = ivm_survey_rejection_reasons(
                    camera_report,
                    require_full_plate=False,
                    max_area_frac=ivm_max_center_board_area_frac,
                    clearance_scale=0.20,
                )
                logging.info(
                    "iteration=%d %s: seen=%s plate_full=%s "
                    "diagnostic_survey_reasons=%s edges=%s area=%.3f "
                    "rect=%.2f quality=%.3f center=(%.3f,%.3f) "
                    "long_axis_error=%+.1fdeg long_ratio=%.2f logo=%s "
                    "logo_center=(%.3f,%.3f) logo_area=%.4f reasons=%s "
                    "clearance=(%.0f,%.0f,%.0f,%.0f)px pad=%.0fpx "
                    "gripper_mask_contact=%s gripper_overlap=%dpx "
                    "gripper_clearance=%.1fpx escape=(%+.2f,%+.2f) stamp=%s",
                    iteration,
                    camera_name,
                    camera_report.seen,
                    camera_report.full,
                    survey_reasons,
                    sorted(camera_report.edges),
                    camera_report.area_frac,
                    camera_report.rectangularity,
                    camera_report.quality_score,
                    camera_report.center_error[0],
                    camera_report.center_error[1],
                    camera_report.orientation_deg,
                    camera_report.long_axis_ratio,
                    camera_report.logo_seen,
                    camera_report.logo_center_error[0],
                    camera_report.logo_center_error[1],
                    camera_report.logo_area_frac,
                    camera_report.failure_reasons,
                    camera_report.clearance_px[0],
                    camera_report.clearance_px[1],
                    camera_report.clearance_px[2],
                    camera_report.clearance_px[3],
                    camera_report.context_pad_px,
                    camera_report.artificial_bottom_contact,
                    camera_report.gripper_overlap_px,
                    camera_report.gripper_clearance_px,
                    camera_report.gripper_escape_direction[0],
                    camera_report.gripper_escape_direction[1],
                    frame["stamp_ns"],
                )
                if target_mode is not SurveyTargetMode.UNSPECIFIED:
                    logging.info(
                        "iteration=%d %s target_roi=%s seen=%s full=%s "
                        "visible=%.3f(min center/side %.2f/%.2f) "
                        "area=%.3f(range %.3f..%.3f) "
                        "center=(%+.3f,%+.3f) "
                        "edges=%s mask_overlap=%dpx mask_clearance=%.1fpx",
                        iteration,
                        camera_name,
                        camera_report.survey_target,
                        camera_report.target_region_seen,
                        camera_report.target_region_full,
                        camera_report.target_region_visible_frac,
                        target_requirements.center_min_visible_frac,
                        target_requirements.side_min_visible_frac,
                        camera_report.target_region_area_frac,
                        target_requirements.min_area_frac,
                        target_requirements.max_area_frac,
                        camera_report.target_region_center_error[0],
                        camera_report.target_region_center_error[1],
                        sorted(camera_report.target_region_edges),
                        camera_report.target_region_gripper_overlap_px,
                        camera_report.target_region_gripper_clearance_px,
                    )

            # Validate the image-plane component of the preceding leveling
            # move against this genuinely fresh center frame.  If the board
            # moved farther toward the same vertical edge, invert the camera
            # image-Y polarity for the next J2--J4 correction.  This makes the
            # controller independent of a mount/TF sign convention and avoids
            # repeating the wrong-way J4 roll seen in the live trace.
            center_report = reports.get("center_camera")
            if (
                pending_level_vertical_sample is not None
                and center_report is not None
                and center_report.seen
            ):
                feedback_mode = pending_level_vertical_sample[0]
                if feedback_mode == "gripper":
                    _, previous_overlap, previous_clearance, commanded_level_y = (
                        pending_level_vertical_sample
                    )
                    current_overlap = center_report.gripper_overlap_px
                    current_clearance = center_report.gripper_clearance_px
                    worsened = (
                        current_overlap > previous_overlap + 100
                        or (
                            previous_overlap == 0
                            and current_overlap == 0
                            and current_clearance + 2.0 < previous_clearance
                        )
                    )
                    if worsened:
                        level_vertical_polarity *= -1.0
                        logging.warning(
                            "leveling mask escape worsened separation "
                            "(overlap %d->%dpx, clearance %.1f->%.1fpx, "
                            "commanded image-y %+.1f); reversing image-y "
                            "polarity",
                            previous_overlap,
                            current_overlap,
                            previous_clearance,
                            current_clearance,
                            commanded_level_y,
                        )
                    else:
                        logging.info(
                            "leveling mask escape validated by fresh center "
                            "frame: overlap %d->%dpx clearance %.1f->%.1fpx",
                            previous_overlap,
                            current_overlap,
                            previous_clearance,
                            current_clearance,
                        )
                else:
                    _, previous_level_y, commanded_level_y = (
                        pending_level_vertical_sample
                    )
                    current_level_y = float(center_report.center_error[1])
                    if (
                        previous_level_y * current_level_y > 0.0
                        and abs(current_level_y)
                        > abs(previous_level_y) + 0.03
                    ):
                        level_vertical_polarity *= -1.0
                        logging.warning(
                            "leveling vertical correction moved the board "
                            "farther toward the frame edge (%+.3f -> %+.3f, "
                            "commanded image-y %+.1f); reversing image-y "
                            "polarity",
                            previous_level_y,
                            current_level_y,
                            commanded_level_y,
                        )
                    else:
                        logging.info(
                            "leveling vertical correction validated by fresh "
                            "center frame: %+.3f -> %+.3f",
                            previous_level_y,
                            current_level_y,
                        )
                pending_level_vertical_sample = None

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

            # Side cameras may seed acquisition and are mandatory supporting
            # evidence for IVM, but can never finish the search by themselves.
            # The planner requires center-camera J6/survey-tilt geometry plus
            # simultaneous context and gripper clearance in all three views.
            result.component_coverage_ready = False

            # Check any available force sample before success.  A missing
            # sample may not block a no-motion terminal frame, but every
            # movement path below calls ``require_motion_force`` first.
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

            # Completion needs a fresh physical survey-tilt check, not a
            # sticky acknowledgement from an earlier positioning iteration.
            # If later clearance IK leaves the configured oblique band, return
            # to LEVEL immediately.
            center_tilt_ready = False
            if planner.phase == "ascend_clearance":
                try:
                    completion_image_right, _, completion_back_away = (
                        self._camera_axes_in_base(
                            "center_camera", timeout_sec
                        )
                    )
                except Exception as error:
                    result.success = False
                    result.message = (
                        "permitted center-camera TF unavailable for fresh "
                        f"survey-tilt completion check: {error}"
                    )
                    return
                completion_tilt_rad = math.acos(
                    float(
                        np.clip(
                            np.dot(
                                np.asarray(completion_back_away, dtype=float),
                                np.array([0.0, 0.0, 1.0]),
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
                _, completion_tilt_step, completion_survey_error = (
                    survey_tilt_correction(
                        completion_back_away,
                        completion_image_right,
                        target_center_tilt_rad,
                        center_tilt_tolerance_rad,
                    )
                )
                center_tilt_ready = completion_tilt_step == 0.0
                logging.info(
                    "fresh center-camera completion tilt=%.3frad "
                    "target=%.3frad survey_vector_error=%.3frad "
                    "tolerance=%.3frad ready=%s",
                    completion_tilt_rad,
                    target_center_tilt_rad,
                    completion_survey_error,
                    center_tilt_tolerance_rad,
                    center_tilt_ready,
                )
                if not center_tilt_ready:
                    planner.request_relevel()
                    logging.warning(
                        "center camera left the IVM survey-tilt band during "
                        "clearance; returning to joints 2-4 positioning"
                    )

            # Attach the fresh physical survey-tilt gate to the center report.
            # Do not overload ``full``: that image-only field includes the old
            # padded whole-plate context which is intentionally unnecessary at
            # the closer IVM component view.
            planning_reports = {
                name: replace(
                    item,
                    survey_tilt_ready=bool(
                        name != "center_camera" or center_tilt_ready
                    ),
                )
                for name, item in reports.items()
            }
            action = planner.next_action(
                planning_reports,
                deadline_reached=time.monotonic() >= deadline,
            )
            result.last_action = action.kind.value

            if action.kind == ActionKind.DONE:
                result.done = True
                result.success = True
                result.component_coverage_ready = True
                result.steer_camera = "center_camera"
                result.elapsed_seconds = max(0.0, time.monotonic() - started_at)
                result.message = (
                    "all three cameras confirmed synchronized target context "
                    "and gripper clearance while the center retained its "
                    "aligned oblique IVM survey view in two fresh frames after "
                    f"{result.moves_executed} adaptive moves"
                )
                return
            if action.terminal:
                result.success = False
                result.message = action.reason
                return

            # Position the survey tilt only after J1 centering and J6 long-axis
            # alignment. The terminal view belongs to the center camera, so
            # control its optical axis rather than TCP +Z (the camera is
            # mounted about 15 degrees off the tool axis). A controlled
            # oblique angle preserves module depth cues that disappear in a
            # perfectly vertical view. The bounded Cartesian correction and
            # small +Z clearance are realized primarily through joints 2-4
            # while the measured J1/J6 references remain phase anchors.
            if planner.phase == "j2_4_level":
                try:
                    level_position, level_orientation = self._gripper_pose(
                        timeout_sec
                    )
                    (
                        level_image_right,
                        level_image_down,
                        camera_back_away,
                    ) = self._camera_axes_in_base("center_camera", timeout_sec)
                except Exception as error:
                    result.success = False
                    result.message = (
                        "permitted center-camera/TCP TF unavailable for "
                        f"leveling: {error}"
                    )
                    return
                current_level_joint1 = self.robot_motion.current_joint1(
                    min(timeout_sec, 2.0)
                )
                current_level_joint6 = self.robot_motion.current_joint(
                    5, min(timeout_sec, 2.0)
                )
                if current_level_joint1 is None or current_level_joint6 is None:
                    result.success = False
                    result.message = (
                        "fresh measured J1/J6 references unavailable at the "
                        "joints 2-4 leveling boundary"
                    )
                    return
                if level_anchor_joint1 is None:
                    level_anchor_joint1 = current_level_joint1
                if level_anchor_joint6 is None:
                    level_anchor_joint6 = current_level_joint6
                if initial_joint1 is None:
                    initial_joint1 = current_level_joint1
                if initial_joint6 is None:
                    initial_joint6 = current_level_joint6
                camera_back_away = np.asarray(camera_back_away, dtype=float)
                straight_up = np.array([0.0, 0.0, 1.0])
                tilt_rad = math.acos(
                    float(
                        np.clip(
                            np.dot(camera_back_away, straight_up), -1.0, 1.0
                        )
                    )
                )
                if tilt_rad > 1.30:
                    result.success = False
                    result.message = (
                        f"center camera is {tilt_rad:.2f}rad from "
                        "vertical; the survey pitch is outside the supported "
                        "positioning range"
                    )
                    return
                (
                    level_axis,
                    tilt_correction_rad,
                    survey_error_before,
                ) = survey_tilt_correction(
                    camera_back_away,
                    level_image_right,
                    target_center_tilt_rad,
                    center_tilt_tolerance_rad,
                )
                if tilt_correction_rad != 0.0:
                    if initial_pose is None:
                        initial_pose = (
                            np.asarray(level_position, dtype=float),
                            normalize_quaternion(level_orientation),
                        )
                    pre_level_joint1 = current_level_joint1
                    pre_level_joint6 = current_level_joint6
                    # Fixed-position orientation-only IK stalled in an earlier
                    # live run. Give the first pitch request one small base-Z
                    # singularity escape, then never accumulate clearance on
                    # later pitch steps. Repeated 20 mm lifts were the reason
                    # the previous survey finished above IVM's useful range.
                    level_target_orientation = self._rotate_orientation_in_base(
                        level_orientation,
                        level_axis,
                        tilt_correction_rad,
                    )
                    level_clearance_m = (
                        0.0
                        if level_clearance_applied
                        else min(0.015, backoff_step_m)
                    )
                    level_center_y = 0.0
                    level_image_x_direction = 0.0
                    level_image_y_direction = 0.0
                    level_feedback_mode = "center"
                    level_gripper_overlap = 0
                    level_gripper_clearance = float("inf")
                    level_center_delta = np.zeros(3, dtype=float)
                    if center_report is not None and center_report.seen:
                        level_center_y = float(center_report.center_error[1])
                        level_gripper_blocked = (
                            center_report.artificial_bottom_contact
                            or center_report.gripper_overlap_px > 0
                            or center_report.gripper_clearance_px < 20.0
                        )
                        if level_gripper_blocked:
                            level_feedback_mode = "gripper"
                            level_gripper_overlap = (
                                center_report.gripper_overlap_px
                            )
                            level_gripper_clearance = (
                                center_report.gripper_clearance_px
                            )
                            escape_x, escape_y = (
                                center_report.gripper_escape_direction
                            )
                            escape_norm = math.hypot(escape_x, escape_y)
                            if escape_norm < 1e-9:
                                escape_x, escape_y, escape_norm = 0.0, -1.0, 1.0
                            # Static board image displacement is opposite the
                            # camera-plane motion.  Apply the learned vertical
                            # polarity to Y; X is directly calibrated by TF.
                            level_image_x_direction = -escape_x / escape_norm
                            level_image_y_direction = (
                                -escape_y
                                / escape_norm
                                * level_vertical_polarity
                            )
                            level_center_scale = min(
                                1.5,
                                max(
                                    1.0,
                                    1.0
                                    + center_report.gripper_overlap_px / 20000.0,
                                ),
                            )
                            level_center_delta = (
                                np.asarray(level_image_right, dtype=float)
                                * step_m
                                * level_center_scale
                                * level_image_x_direction
                                + np.asarray(level_image_down, dtype=float)
                                * step_m
                                * level_center_scale
                                * level_image_y_direction
                            )
                        elif abs(level_center_y) > 0.35:
                            level_image_y_direction = (
                                math.copysign(1.0, level_center_y)
                                * level_vertical_polarity
                            )
                            level_center_scale = min(
                                1.5, max(1.0, abs(level_center_y) / 0.20)
                            )
                            level_center_delta = (
                                np.asarray(level_image_down, dtype=float)
                                * step_m
                                * level_center_scale
                                * level_image_y_direction
                            )
                    level_target_position_array = (
                        np.asarray(level_position, dtype=float)
                        + np.array((0.0, 0.0, level_clearance_m), dtype=float)
                        + level_center_delta
                    )
                    level_target_position = tuple(
                        float(value) for value in level_target_position_array
                    )
                    logging.info(
                        "joints 2-4 IVM survey positioning after J6: "
                        "center-camera tilt=%.3frad target=%.3frad "
                        "survey_error=%.3frad step=%.3frad clearance=%.3fm "
                        "vertical_error=%+.3f image_direction=(%+.2f,%+.2f) "
                        "camera_plane_delta=(%+.4f,%+.4f,%+.4f)m; keeping "
                        "J1/J6 phase references fixed",
                        tilt_rad,
                        target_center_tilt_rad,
                        survey_error_before,
                        tilt_correction_rad,
                        level_clearance_m,
                        level_center_y,
                        level_image_x_direction,
                        level_image_y_direction,
                        float(level_center_delta[0]),
                        float(level_center_delta[1]),
                        float(level_center_delta[2]),
                    )
                    motion_snapshot = require_motion_force(snapshot)
                    if motion_snapshot is None:
                        result.success = False
                        result.message = (
                            "no fresh wrist-force sample after waiting; "
                            "refusing joints 2-4 camera-leveling motion"
                        )
                        return
                    snapshot = motion_snapshot
                    result.force_n = float(snapshot.force_norm or 0.0)
                    outcome = self.robot_motion.move_smooth(
                        level_target_position,
                        target_orientation=level_target_orientation,
                        max_speed_mps=max_speed_mps,
                        max_angular_speed_radps=max_angular_speed_rps,
                        publish_hz=publish_hz,
                        settle_tolerance_m=settle_tolerance_m,
                        settle_angular_tolerance_rad=settle_orientation_tolerance_rad,
                        timeout_sec=move_timeout_sec,
                        baseline_force_xyz=baseline_force_xyz,
                        max_force_n=max_force_n,
                        force_delta_n=force_delta_n,
                        cancelled=motion_cancelled,
                    )
                    if outcome.cancelled:
                        if cancelled():
                            raise skill_interface.SkillCancelledError(
                                outcome.message
                            )
                        result.success = False
                        result.message = (
                            "adaptive viewpoint search reached its safety "
                            "deadline during camera leveling"
                        )
                        return
                    if not outcome.success:
                        result.success = False
                        result.force_abort = outcome.force_abort
                        result.message = outcome.message
                        return
                    next_travel_m = result.travel_m + float(
                        outcome.distance_m
                    )
                    next_angular_travel_rad = (
                        result.angular_travel_rad
                        + float(outcome.angular_distance_rad)
                    )
                    if next_travel_m > max_travel_m + 1e-9:
                        result.success = False
                        result.message = (
                            "joints 2-4 leveling reached the "
                            f"{max_travel_m:.3f}m cumulative translation envelope"
                        )
                        return
                    if (
                        next_angular_travel_rad
                        > max_angular_travel_rad + 1e-9
                    ):
                        result.success = False
                        result.message = (
                            "joints 2-4 leveling reached the "
                            f"{max_angular_travel_rad:.3f}rad cumulative angular "
                            "envelope"
                        )
                        return
                    result.moves_executed += 1
                    leveling_moves += 1
                    level_clearance_applied = True
                    result.travel_m = next_travel_m
                    result.angular_travel_rad = next_angular_travel_rad
                    result.moved = True
                    result.last_action = "camera_level"
                    if abs(level_image_y_direction) > 0.0:
                        if level_feedback_mode == "gripper":
                            pending_level_vertical_sample = (
                                "gripper",
                                level_gripper_overlap,
                                level_gripper_clearance,
                                level_image_y_direction,
                            )
                        else:
                            pending_level_vertical_sample = (
                                "center",
                                level_center_y,
                                level_image_y_direction,
                            )
                    try:
                        (
                            residual_image_right,
                            _,
                            residual_back_away,
                        ) = self._camera_axes_in_base(
                            "center_camera", timeout_sec
                        )
                    except Exception as error:
                        result.success = False
                        result.message = (
                            "permitted center-camera TF unavailable after "
                            f"leveling: {error}"
                        )
                        return
                    residual_tilt = math.acos(
                        float(
                            np.clip(
                                np.dot(
                                    np.asarray(residual_back_away, dtype=float),
                                    straight_up,
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    (
                        _,
                        residual_tilt_step,
                        residual_tilt_error,
                    ) = survey_tilt_correction(
                        residual_back_away,
                        residual_image_right,
                        target_center_tilt_rad,
                        center_tilt_tolerance_rad,
                    )
                    post_level_joint1 = self.robot_motion.current_joint1(
                        min(timeout_sec, 0.5)
                    )
                    post_level_joint6 = self.robot_motion.current_joint(
                        5, min(timeout_sec, 0.5)
                    )
                    if post_level_joint1 is None or post_level_joint6 is None:
                        result.success = False
                        result.message = (
                            "fresh measured J1/J6 references unavailable after "
                            "joints 2-4 camera leveling"
                        )
                        return
                    level_step_joint1_drift = (
                        post_level_joint1 - pre_level_joint1
                    )
                    level_step_joint6_drift = (
                        post_level_joint6 - pre_level_joint6
                    )
                    level_joint1_drift = (
                        post_level_joint1 - level_anchor_joint1
                    )
                    level_joint6_drift = (
                        post_level_joint6 - level_anchor_joint6
                    )
                    logging.info(
                        "joints 2-4 IVM survey stage complete: residual center "
                        "camera tilt=%.3frad target=%.3frad error=%.3frad "
                        "angle_moved=%.3frad "
                        "target_reached=%s step_j1_drift=%+.4frad "
                        "step_j6_drift=%+.4frad cumulative_j1_drift=%+.4frad "
                        "cumulative_j6_drift=%+.4frad",
                        residual_tilt,
                        target_center_tilt_rad,
                        residual_tilt_error,
                        outcome.angular_distance_rad,
                        outcome.target_reached,
                        level_step_joint1_drift,
                        level_step_joint6_drift,
                        level_joint1_drift,
                        level_joint6_drift,
                    )
                    # Cartesian IK may redistribute J1/J6 while realizing the
                    # requested camera pose.  Restarting CENTER/ALIGN here made
                    # the live policy repeat J6 and leveling until the deadline.
                    # Keep progressing from the measured pose: the fresh center
                    # image remains the authority for horizontal centering and
                    # the <=2-degree long-axis gate, with a final confirmed J6
                    # trim available after component framing.
                    if (
                        residual_tilt_step != 0.0
                        and residual_tilt_error
                        >= survey_error_before - min_level_progress_rad
                    ):
                        result.success = False
                        result.message = (
                            "joints 2-4 positioning made less than 0.01rad "
                            "progress toward the IVM survey-tilt band; "
                            "refusing to repeat the same Cartesian request"
                        )
                        return
                    if residual_tilt_step == 0.0:
                        planner.mark_level_complete()
                    iteration += 1
                    continue

                # Even when no leveling motion is needed, transition the
                # planner and capture one genuinely fresh terminal frame.
                planner.mark_level_complete()
                logging.info(
                    "center camera already inside the IVM survey-tilt band "
                    "after J6; capturing the first fresh completion frame"
                )
                iteration += 1
                continue

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

            if action.moves_robot:
                motion_snapshot = require_motion_force(snapshot)
                if motion_snapshot is None:
                    result.success = False
                    result.message = (
                        "no fresh wrist-force sample after waiting; refusing "
                        f"{action.kind.value} motion"
                    )
                    return
                snapshot = motion_snapshot
                result.force_n = float(snapshot.force_norm or 0.0)

            if action.kind == ActionKind.CAMERA_ROLL:
                # J6 is UR5e wrist_3_joint. A Cartesian orientation request can
                # be distributed across several IK joints and therefore never
                # guarantees that the camera module actually yaws. Switch to
                # documented joint mode, preserve joints 1-5 exactly, command
                # only measured joint index 5, then restore Cartesian mode.
                try:
                    (px, py, pz), (qx, qy, qz, qw) = self._gripper_pose(
                        timeout_sec
                    )
                except Exception as error:
                    result.success = False
                    result.message = (
                        f"permitted gripper TF unavailable for J6 yaw: {error}"
                    )
                    return
                current_position = np.asarray((px, py, pz), dtype=float)
                current_orientation = normalize_quaternion((qx, qy, qz, qw))
                current_joint1 = self.robot_motion.current_joint1(
                    min(timeout_sec, 2.0)
                )
                current_joint6 = self.robot_motion.current_joint(
                    5, min(timeout_sec, 2.0)
                )
                if current_joint1 is None or current_joint6 is None:
                    result.success = False
                    result.message = (
                        "fresh measured joint 1/J6 state unavailable before "
                        "camera-module yaw"
                    )
                    return
                if initial_pose is None:
                    initial_pose = (
                        current_position.copy(),
                        current_orientation,
                    )
                if initial_joint1 is None:
                    initial_joint1 = current_joint1
                if initial_joint6 is None:
                    initial_joint6 = current_joint6

                joint_delta = (
                    float(action.aim_direction[0])
                    * angular_step_rad
                    * float(action.angular_scale)
                )
                target_joint6 = current_joint6 + joint_delta
                if (
                    abs(target_joint6 - initial_joint6)
                    > max_angular_displacement_rad + 1e-9
                ):
                    rejection_reason = (
                        "J6 long-edge alignment exhausted its "
                        f"{max_angular_displacement_rad:.3f}rad start-relative "
                        "wrist_3_joint envelope"
                    )
                    logging.warning(
                        "iteration=%d rejecting direct J6 id=%d before "
                        "motion: %s; enabling J1/zoom fallback",
                        iteration,
                        action.action_id,
                        rejection_reason,
                    )
                    planner.mark_roll_unavailable(
                        action,
                        reason=rejection_reason,
                    )
                    result.last_action = "camera_roll_preflight_rejected"
                    iteration += 1
                    continue
                if (
                    result.angular_travel_rad + abs(joint_delta)
                    > max_angular_travel_rad + 1e-9
                ):
                    rejection_reason = (
                        "J6 long-edge alignment exhausted its "
                        f"{max_angular_travel_rad:.3f}rad cumulative angular "
                        "travel envelope"
                    )
                    logging.warning(
                        "iteration=%d rejecting direct J6 id=%d before "
                        "motion: %s; enabling joints 2-4 zoom fallback",
                        iteration,
                        action.action_id,
                        rejection_reason,
                    )
                    planner.mark_roll_unavailable(
                        action,
                        reason=rejection_reason,
                        allow_j1_fallback=False,
                    )
                    result.last_action = "camera_roll_preflight_rejected"
                    iteration += 1
                    continue

                tool_axis = rotation_matrix_from_quaternion(
                    *current_orientation
                )[:, 2]
                predicted_orientation = self._rotate_orientation_in_base(
                    current_orientation, tool_axis, joint_delta
                )
                predicted_start_angle = quaternion_angular_distance(
                    initial_pose[1], predicted_orientation
                )
                if (
                    predicted_start_angle
                    > max_angular_displacement_rad + 1e-9
                ):
                    rejection_reason = (
                        "predicted J6 yaw would exceed the "
                        f"{max_angular_displacement_rad:.3f}rad TCP orientation "
                        "envelope"
                    )
                    logging.warning(
                        "iteration=%d rejecting direct J6 id=%d before "
                        "motion: %s; enabling joints 2-4 zoom fallback",
                        iteration,
                        action.action_id,
                        rejection_reason,
                    )
                    planner.mark_roll_unavailable(
                        action,
                        reason=rejection_reason,
                        allow_j1_fallback=False,
                    )
                    result.last_action = "camera_roll_preflight_rejected"
                    iteration += 1
                    continue

                logging.info(
                    "iteration=%d action=camera_roll id=%d camera=%s "
                    "reason=%s joint6=%.4f target_joint6=%.4f "
                    "delta=%.4frad control=direct_wrist_3_joint",
                    iteration,
                    action.action_id,
                    action.camera,
                    action.reason,
                    current_joint6,
                    target_joint6,
                    joint_delta,
                )
                outcome = self.robot_motion.move_joint6_yaw(
                    joint_delta,
                    max_speed_radps=min(max_angular_speed_rps, 0.20),
                    publish_hz=publish_hz,
                    settle_tolerance_rad=min(
                        0.015, max(0.006, 0.20 * abs(joint_delta))
                    ),
                    settle_tcp_speed_mps=0.02,
                    timeout_sec=move_timeout_sec,
                    baseline_force_xyz=baseline_force_xyz,
                    max_force_n=max_force_n,
                    force_delta_n=force_delta_n,
                    cancelled=motion_cancelled,
                )
                if outcome.cancelled:
                    if cancelled():
                        raise skill_interface.SkillCancelledError(
                            outcome.message
                        )
                    result.success = False
                    result.message = (
                        "adaptive viewpoint search reached its safety "
                        "deadline during direct J6 yaw"
                    )
                    return
                if not outcome.success:
                    recoverable_j6_failure = outcome.failure in {
                        MotionFailure.CONTROLLER_UNAVAILABLE,
                        MotionFailure.MODE_UNAVAILABLE,
                        MotionFailure.MODE_CHANGED,
                        MotionFailure.TARGET_TIMEOUT,
                    }
                    cartesian_restore_failed = (
                        "Cartesian target mode" in outcome.message
                        or "Cartesian target-mode" in outcome.message
                    )
                    if (
                        recoverable_j6_failure
                        and not outcome.force_abort
                        and not cartesian_restore_failed
                    ):
                        logging.warning(
                            "iteration=%d direct J6 id=%d unavailable after "
                            "safe stop/reversal: %s; enabling J1/zoom "
                            "fallback",
                            iteration,
                            action.action_id,
                            outcome.message,
                        )
                        planner.mark_roll_unavailable(
                            action,
                            reason=outcome.message,
                        )
                        result.last_action = "camera_roll_unavailable"
                        iteration += 1
                        continue
                    result.success = False
                    result.force_abort = outcome.force_abort
                    result.message = outcome.message
                    return

                try:
                    post_position, post_orientation = self._gripper_pose(
                        timeout_sec
                    )
                except Exception as error:
                    result.success = False
                    result.message = (
                        f"permitted gripper TF unavailable after J6 yaw: {error}"
                    )
                    return
                post_joint6 = self.robot_motion.current_joint(
                    5, min(timeout_sec, 0.5)
                )
                post_joint1 = self.robot_motion.current_joint1(
                    min(timeout_sec, 0.5)
                )
                if post_joint6 is None or post_joint1 is None:
                    result.success = False
                    result.message = (
                        "fresh measured joint 1/J6 state unavailable after "
                        "camera-module yaw"
                    )
                    return
                if (
                    abs(post_joint1 - current_joint1) > 0.02
                ):
                    result.success = False
                    result.message = (
                        "direct J6 command unexpectedly moved joint 1; "
                        "camera yaw rejected"
                    )
                    return
                measured_joint_delta = post_joint6 - current_joint6
                angular_step = max(
                    abs(measured_joint_delta), outcome.joint_distance_rad
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
                result.dx = result.dy = result.dz = 0.0
                result.moves_executed += 1
                result.angular_travel_rad += angular_step
                result.moved = True
                logging.info(
                    "direct J6 yaw %d completed: requested=%.4frad "
                    "measured_wrist_3=%.4frad joint1_drift=%.4frad "
                    "angular_travel=%.4frad",
                    result.moves_executed,
                    joint_delta,
                    measured_joint_delta,
                    post_joint1 - current_joint1,
                    result.angular_travel_rad,
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
                # base-Z axis. Prior live runs rejected MODE_JOINT on this
                # base-yaw path, so J1 retains the proven Cartesian safety
                # route. Direct J6 above attempts documented joint mode only
                # because no Cartesian IK request can guarantee wrist_3 motion,
                # and it has an explicit J1/zoom fallback if unavailable. This
                # rigid base-yaw arc is distinct from camera-axis pitch/aim and
                # follows joint 1's local FK.
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
                    (
                        result.angular_travel_rad + abs(joint_delta)
                    )
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
                predicted_start_angle = quaternion_angular_distance(
                    initial_pose[1], target_orientation
                )
                if predicted_start_angle > max_angular_displacement_rad + 1e-9:
                    rejection_reason = (
                        "predicted joint-1 yaw would exceed the "
                        f"{max_angular_displacement_rad:.3f}rad start-relative "
                        "TCP orientation envelope"
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
                    (
                        result.travel_m + predicted_step_distance
                    )
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

                if joint_yaw_available:
                    # Strict one-joint-at-a-time centering: command measured
                    # shoulder_pan directly.  The Cartesian arc below remains
                    # only as a fallback because, at the leveled pose, the
                    # base-Z and wrist_3 axes are both vertical and Cartesian
                    # IK is free to satisfy the orientation change with J6 -
                    # the observed roll drift during centering.
                    pre_joint6 = self.robot_motion.current_joint(
                        5, min(timeout_sec, 2.0)
                    )
                    logging.info(
                        "iteration=%d action=%s id=%d camera=%s reason=%s "
                        "joint1=%.4f target_joint1=%.4f delta=%.4frad "
                        "control=direct_shoulder_pan_joint",
                        iteration,
                        action.kind.value,
                        action.action_id,
                        action.camera,
                        action.reason,
                        joint1,
                        target_joint1,
                        joint_delta,
                    )
                    outcome = self.robot_motion.move_joint1_yaw(
                        joint_delta,
                        max_speed_radps=min(max_angular_speed_rps, 0.20),
                        publish_hz=publish_hz,
                        settle_tolerance_rad=min(
                            0.015, max(0.006, 0.20 * abs(joint_delta))
                        ),
                        settle_tcp_speed_mps=0.02,
                        timeout_sec=move_timeout_sec,
                        baseline_force_xyz=baseline_force_xyz,
                        max_force_n=max_force_n,
                        force_delta_n=force_delta_n,
                        cancelled=motion_cancelled,
                    )
                    if outcome.cancelled:
                        if cancelled():
                            raise skill_interface.SkillCancelledError(
                                outcome.message
                            )
                        result.success = False
                        result.message = (
                            "adaptive viewpoint search reached its safety "
                            "deadline during direct J1 yaw"
                        )
                        return
                    if not outcome.success:
                        recoverable_j1_failure = outcome.failure in {
                            MotionFailure.CONTROLLER_UNAVAILABLE,
                            MotionFailure.MODE_UNAVAILABLE,
                            MotionFailure.MODE_CHANGED,
                        }
                        cartesian_restore_failed = (
                            "Cartesian target mode" in outcome.message
                            or "Cartesian target-mode" in outcome.message
                        )
                        if (
                            recoverable_j1_failure
                            and not outcome.force_abort
                            and not cartesian_restore_failed
                        ):
                            joint_yaw_available = False
                            logging.warning(
                                "iteration=%d direct J1 unavailable (%s); "
                                "falling back to the Cartesian base-yaw arc",
                                iteration,
                                outcome.message,
                            )
                        else:
                            result.success = False
                            result.force_abort = outcome.force_abort
                            result.message = outcome.message
                            return
                    else:
                        try:
                            post_position, post_orientation = (
                                self._gripper_pose(timeout_sec)
                            )
                        except Exception as error:
                            result.success = False
                            result.message = (
                                "permitted gripper TF unavailable after "
                                f"direct J1 yaw: {error}"
                            )
                            return
                        post_joint1 = self.robot_motion.current_joint1(
                            min(timeout_sec, 0.5)
                        )
                        post_joint6 = self.robot_motion.current_joint(
                            5, min(timeout_sec, 0.5)
                        )
                        if post_joint1 is None:
                            result.success = False
                            result.message = (
                                "fresh measured /joint_states arm pose "
                                "unavailable after direct J1 yaw"
                            )
                            return
                        if (
                            pre_joint6 is not None
                            and post_joint6 is not None
                            and abs(post_joint6 - pre_joint6) > 0.02
                        ):
                            result.success = False
                            result.message = (
                                "direct J1 command unexpectedly moved joint "
                                "6; base yaw rejected"
                            )
                            return
                        if (
                            abs(float(post_joint1) - initial_joint1)
                            > max_angular_displacement_rad + 1e-9
                        ):
                            result.success = False
                            result.message = (
                                "measured joint-1 pose exceeded the "
                                "start-relative "
                                f"{max_angular_displacement_rad:.3f}rad "
                                "envelope"
                            )
                            return
                        post_position_array = np.asarray(
                            post_position, dtype=float
                        )
                        step_distance = float(
                            np.linalg.norm(
                                post_position_array - current_position
                            )
                        )
                        start_displacement = float(
                            np.linalg.norm(
                                post_position_array - initial_pose[0]
                            )
                        )
                        if start_displacement > max_displacement_m + 1e-9:
                            result.success = False
                            result.message = (
                                "joint-1 centering reached the "
                                f"{max_displacement_m:.3f}m workspace envelope"
                            )
                            return
                        if (
                            result.travel_m + step_distance
                            > max_travel_m + 1e-9
                        ):
                            result.success = False
                            result.message = (
                                "joint-1 centering reached the "
                                f"{max_travel_m:.3f}m cumulative translation "
                                "envelope"
                            )
                            return
                        measured_joint_delta = float(post_joint1 - joint1)
                        angular_step = max(
                            abs(measured_joint_delta),
                            float(outcome.joint_distance_rad),
                        )
                        if (
                            result.angular_travel_rad + angular_step
                            > max_angular_travel_rad + 1e-9
                        ):
                            result.success = False
                            result.message = (
                                "measured joint-1 motion exceeded the "
                                f"cumulative {max_angular_travel_rad:.3f}rad "
                                "envelope"
                            )
                            return
                        delta = post_position_array - current_position
                        result.dx, result.dy, result.dz = (
                            float(value) for value in delta
                        )
                        result.target_valid = True
                        result.target_frame = self.config.base_frame
                        (
                            result.target.x,
                            result.target.y,
                            result.target.z,
                        ) = post_position
                        (
                            result.target.qx,
                            result.target.qy,
                            result.target.qz,
                            result.target.qw,
                        ) = post_orientation
                        result.moves_executed += 1
                        result.travel_m += step_distance
                        result.angular_travel_rad += angular_step
                        result.moved = True
                        logging.info(
                            "direct J1 yaw %d completed: requested=%.4frad "
                            "measured_joint1=%.4frad joint6_drift=%.4frad "
                            "tcp=%.4fm travel=%.4fm angular_travel=%.4frad",
                            result.moves_executed,
                            joint_delta,
                            measured_joint_delta,
                            (
                                float(post_joint6 - pre_joint6)
                                if pre_joint6 is not None
                                and post_joint6 is not None
                                else 0.0
                            ),
                            step_distance,
                            result.travel_m,
                            result.angular_travel_rad,
                        )
                        iteration += 1
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
                # Preserve the exact pre-yaw pose. If the center-camera mask
                # discontinuously switches between disconnected board
                # fragments, the planner restores this pose once and changes
                # viewpoint with J2 instead of entering a yaw limit cycle.
                saved_action_poses[action.action_id] = (
                    tuple(float(value) for value in current_position),
                    current_orientation,
                )
                outcome = self.robot_motion.move_smooth(
                    tuple(float(value) for value in predicted_position),
                    target_orientation=target_orientation,
                    max_speed_mps=min(max_speed_mps, 0.05),
                    max_angular_speed_radps=min(max_angular_speed_rps, 0.30),
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
                    cancelled=motion_cancelled,
                )
                if outcome.cancelled:
                    if cancelled():
                        raise skill_interface.SkillCancelledError(outcome.message)
                    result.success = False
                    result.message = (
                        "adaptive viewpoint search reached its safety deadline "
                        "during joint-1 motion"
                    )
                    return
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
                start_angle = quaternion_angular_distance(
                    initial_pose[1], post_orientation
                )
                if start_displacement > max_displacement_m + 1e-9:
                    result.success = False
                    result.message = (
                        "joint-1 centering reached the "
                        f"{max_displacement_m:.3f}m workspace envelope"
                    )
                    return
                if start_angle > max_angular_displacement_rad + 1e-9:
                    result.success = False
                    result.message = (
                        "joint-1 centering reached the "
                        f"{max_angular_displacement_rad:.3f}rad TCP orientation envelope"
                    )
                    return
                if (
                    result.travel_m + step_distance
                    > max_travel_m + 1e-9
                ):
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
                if post_joint1 is None:
                    result.success = False
                    result.message = (
                        "fresh measured /joint_states arm pose unavailable "
                        "after horizontal board centering"
                    )
                    return
                if (
                    abs(float(post_joint1) - initial_joint1)
                    > max_angular_displacement_rad + 1e-9
                ):
                    result.success = False
                    result.message = (
                        "measured joint-1 pose exceeded the start-relative "
                        f"{max_angular_displacement_rad:.3f}rad envelope"
                    )
                    return
                measured_joint_delta = float(post_joint1 - joint1)
                angular_step = max(
                    abs(measured_joint_delta),
                    float(outcome.angular_distance_rad),
                )
                if (
                    result.angular_travel_rad + angular_step
                    > max_angular_travel_rad + 1e-9
                ):
                    result.success = False
                    result.message = (
                        "measured joint-1 motion exceeded the cumulative "
                        f"{max_angular_travel_rad:.3f}rad envelope"
                    )
                    return
                result.angular_travel_rad += angular_step
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
                try:
                    (px, py, pz), (qx, qy, qz, qw) = self._gripper_pose(
                        timeout_sec
                    )
                except Exception as error:
                    result.success = False
                    result.message = (
                        "permitted gripper TF unavailable during rollback: "
                        f"{error}"
                    )
                    return
                target_position, target_orientation = saved_pose
                delta = np.asarray(target_position, dtype=float) - np.asarray(
                    (px, py, pz), dtype=float
                )
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
                    initial_joint1 = self.robot_motion.current_joint1(
                        min(timeout_sec, 2.0)
                    )
                    if initial_joint1 is None:
                        result.success = False
                        result.message = (
                            "fresh measured /joint_states arm pose unavailable "
                            "at adaptive-search start"
                        )
                        return

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
                        # CAMERA_ROLL has already been handled by the direct
                        # wrist_3_joint branch above.  Remaining aim actions
                        # are Cartesian ray shifts only.
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
            if (
                action.kind != ActionKind.ROLLBACK
                and result.travel_m + step_distance
                > max_travel_m + 1e-9
            ):
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
                action.kind != ActionKind.ROLLBACK
                and (
                    result.angular_travel_rad + step_angle
                )
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
                cancelled=motion_cancelled,
            )
            if outcome.cancelled:
                if cancelled():
                    raise skill_interface.SkillCancelledError(outcome.message)
                result.success = False
                result.message = (
                    "adaptive viewpoint search reached its safety deadline "
                    f"during {action.kind.value} motion"
                )
                return
            if not outcome.success:
                result.success = False
                result.force_abort = outcome.force_abort
                result.message = outcome.message
                return
            if (
                action.kind == ActionKind.UP_CLEARANCE
                and not outcome.target_reached
            ):
                planner.mark_clearance_partial(
                    action,
                    reason=outcome.message,
                )
                logging.warning(
                    "base +Z clearance stopped short after %.4fm; replanning "
                    "from the measured pose and a fresh center image",
                    outcome.distance_m,
                )
            post_motion_joint1 = self.robot_motion.current_joint1(
                min(timeout_sec, 0.5)
            )
            if post_motion_joint1 is None:
                result.success = False
                result.message = (
                    "fresh measured /joint_states arm pose unavailable after "
                    f"{action.kind.value} motion"
                )
                return
            if (
                level_anchor_joint1 is not None
                and level_anchor_joint6 is not None
                and planner.phase == "ascend_clearance"
            ):
                post_motion_joint6 = self.robot_motion.current_joint(
                    5, min(timeout_sec, 0.5)
                )
                if post_motion_joint6 is None:
                    result.success = False
                    result.message = (
                        "fresh measured joint 6 unavailable after post-level "
                        f"{action.kind.value} motion"
                    )
                    return
                post_level_joint1_drift = (
                    float(post_motion_joint1) - level_anchor_joint1
                )
                post_level_joint6_drift = (
                    float(post_motion_joint6) - level_anchor_joint6
                )
                logging.info(
                    "post-level joints 2-4 motion measured against phase anchors: "
                    "j1_drift=%+.4frad j6_drift=%+.4frad",
                    post_level_joint1_drift,
                    post_level_joint6_drift,
                )
                # Do not restart J1/J6 solely from joint redistribution.  The
                # next synchronized images decide whether centering/alignment
                # actually changed and the ASCEND phase repairs only the failed
                # image predicate at the final survey pose.
            if (
                initial_joint1 is not None
                and abs(float(post_motion_joint1) - initial_joint1)
                > max_angular_displacement_rad + 1e-9
            ):
                result.success = False
                result.message = (
                    f"{action.kind.value} motion moved measured joint 1 beyond "
                    f"the {max_angular_displacement_rad:.3f}rad start-relative envelope"
                )
                return
            if action.kind == ActionKind.ROLLBACK:
                try:
                    restored_position, restored_orientation = self._gripper_pose(
                        timeout_sec
                    )
                except Exception as error:
                    result.success = False
                    result.message = (
                        "permitted gripper TF unavailable after rollback: "
                        f"{error}"
                    )
                    return
                rollback_position_error = float(
                    np.linalg.norm(
                        np.asarray(restored_position, dtype=float)
                        - np.asarray(target_position, dtype=float)
                    )
                )
                rollback_angle_error = quaternion_angular_distance(
                    restored_orientation, target_orientation
                )
                if (
                    rollback_position_error > settle_tolerance_m
                    or rollback_angle_error > settle_orientation_tolerance_rad
                ):
                    result.success = False
                    result.message = (
                        "anti-cycle rollback did not restore its saved pose "
                        f"(position error {rollback_position_error:.4f}m, "
                        f"orientation error {rollback_angle_error:.4f}rad)"
                    )
                    return
                result.rollback_count += 1
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
        if not 5.0 <= values["target_center_tilt_deg"] <= 45.0:
            raise ValueError("target_center_tilt_deg must be in [5, 45]")
        if not 1.0 <= values["center_tilt_tolerance_deg"] <= 10.0:
            raise ValueError("center_tilt_tolerance_deg must be in [1, 10]")
        if (
            values["center_tilt_tolerance_deg"]
            >= values["target_center_tilt_deg"]
        ):
            raise ValueError(
                "center_tilt_tolerance_deg must be below target_center_tilt_deg"
            )
        if not 0.05 <= values["ivm_min_center_board_area_frac"] <= 0.4:
            raise ValueError(
                "ivm_min_center_board_area_frac must be in [0.05, 0.4]"
            )
        if not 0.1 <= values["ivm_max_center_board_area_frac"] <= 0.6:
            raise ValueError(
                "ivm_max_center_board_area_frac must be in [0.1, 0.6]"
            )
        if (
            values["ivm_min_center_board_area_frac"]
            >= values["ivm_max_center_board_area_frac"]
        ):
            raise ValueError(
                "ivm_min_center_board_area_frac must be below "
                "ivm_max_center_board_area_frac"
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
        # Magnitude comparison: the static gravity/bias load rotates in the
        # wrist sensor frame during J5/J6 reorientation, so a vector delta
        # falsely trips in free space.  A constant load's norm is invariant.
        baseline = np.asarray(baseline_xyz, dtype=float)
        return (
            abs(
                float(np.linalg.norm(force)) - float(np.linalg.norm(baseline))
            )
            >= force_delta_n
        )

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
