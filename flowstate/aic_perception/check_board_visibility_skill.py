#!/usr/bin/env python3
"""Flowstate skill that searches for a fully visible task board.

The skill consumes only documented wrist-camera, measured joint-state,
controller-state, wrist-force, and robot-mounted TF data. It performs
image-feedback shoulder-pan centering, wrist-3 long-axis alignment, center-
camera top-down leveling, and upward clearance through the documented AIC
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

    def _execute_inner(self, params, result, cancelled) -> None:
        from aic_perception.board_visibility import (
            analyze_board,
            combine_cameras,
            ivm_survey_rejection_reasons,
            rotation_matrix_from_quaternion,
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
        # Stage 2 is designed around two settled post-motion triplets.  Keep
        # the complete skill bounded to the requested 60-second headroom;
        # accepting an old 90-second default made a Flowstate timeout look
        # like an inexplicable controller hang.
        search_timeout_sec = float(params.search_timeout_seconds or 60.0)
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
        max_angular_speed_rps = float(params.max_angular_speed_rps or 0.30)
        settle_orientation_tolerance_rad = float(
            params.settle_orientation_tolerance_rad or 0.05
        )
        # `survey_target` was present in the deployed v4 descriptor before it
        # was accidentally dropped from this source branch.  Use numeric enum
        # values here so an older generated Python stub remains safe while the
        # rebuilt descriptor is rolling out: 0/1 are loose staged SFP, and
        # 2/3 deliberately retain the legacy NIC/SC completion contract.
        survey_target = int(getattr(params, "survey_target", 0))
        staged_sfp_target = self._uses_staged_sfp_stage2(survey_target)
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
            min_goal_area_frac=max(0.26, min_detail_area_frac),
            max_goal_area_frac=0.36,
            min_gripper_clearance_px=20.0,
            auxiliary_min_area_frac=0.08,
            auxiliary_min_rectangularity=0.55,
            auxiliary_min_gripper_clearance_px=12.0,
            auxiliary_context_scale=0.75,
            max_auxiliary_translates=8,
            survey_confirmation_frames=2,
            expected_cameras=tuple(sorted(self.config.camera_frames)),
        )
        logging.info(
            "active search parameters: cameras=%s margin_px=%d context=%.3f "
            "ignore_bottom=%.3f step=%.3fm angular_step=%.3frad "
            "settle=%.3fm/%.3frad move_timeout=%.1fs search_timeout=%.1fs "
            "stable_frames_configured=%d j6_min_ratio=%.2f "
            "j6_confirm_frames=%d j6_tolerance=%.1fdeg "
            "motion_envelopes=controller_native "
            "completion=two_fresh_synchronized_three_camera_survey_frames",
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
        )
        started_at = time.monotonic()
        # Reserve a bounded portion of the configured total for the geometric
        # stage. Stage 1 may not consume the entire invocation and then ask
        # Stage 2 to move against an already-expired deadline.
        stage2_reserve_sec = min(
            20.0,
            max(4.0, 0.25 * search_timeout_sec),
            0.40 * search_timeout_sec,
        )
        overall_deadline = started_at + search_timeout_sec
        deadline = overall_deadline - stage2_reserve_sec
        baseline_force_xyz = None
        initial_pose = None
        initial_joint1 = None
        initial_joint6 = None
        saved_action_poses = {}
        iteration = 0
        joint_yaw_available = True
        top_down_tolerance_rad = 0.06
        level_joint_drift_tolerance_rad = 0.02
        min_level_progress_rad = 0.01
        leveling_moves = 0
        level_anchor_joint1 = None
        level_anchor_joint6 = None
        level_vertical_polarity = 1.0
        pending_level_vertical_sample = None
        # Stage 1 is only an acquisition phase.  It must not abort the SFP
        # survey just because the legacy board-centroid heuristic is unhappy
        # with a gripper-clipped initial image.  Give the measured logo/board
        # acquisition policy a finite but useful budget, then let Stage 2 make
        # the final fail-closed pose decision from calibration and all cameras.
        # This is a budget of *all* Stage-1 robot moves, not just the
        # logo-specific fallback moves.  In particular, the legacy planner's
        # initial backoffs must not silently consume extra attempts.
        max_logo_acquisition_moves = 5
        logo_acquisition_moves = 0

        def motion_cancelled() -> bool:
            return bool(cancelled() or time.monotonic() >= deadline)

        def stage2_motion_cancelled() -> bool:
            return bool(cancelled() or time.monotonic() >= overall_deadline)

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
                )
                reports[camera_name] = camera_report
                survey_reasons = ivm_survey_rejection_reasons(camera_report)
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
            # The planner requires center-camera J6/top-down geometry plus
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

            # Keep a real time reserve for the geometric stage.  Do this
            # before asking the legacy planner for another action, so a
            # non-terminal legacy backoff cannot become an uncounted sixth
            # Stage-1 move.  Stage 2 makes its own fail-closed landmark check.
            if (
                staged_sfp_target
                and result.moves_executed >= max_logo_acquisition_moves
            ):
                logging.info(
                    "Stage 1 used its %d total robot-move budget; handing "
                    "the current landmark evidence to Stage 2",
                    max_logo_acquisition_moves,
                )
                self._run_sfp_geometric_stage2(
                    snapshot=snapshot,
                    reports=reports,
                    result=result,
                    timeout_sec=timeout_sec,
                    deadline=overall_deadline,
                    started_at=started_at,
                    max_speed_mps=max_speed_mps,
                    max_angular_speed_rps=max_angular_speed_rps,
                    publish_hz=publish_hz,
                    settle_tolerance_m=settle_tolerance_m,
                    settle_orientation_tolerance_rad=(
                        settle_orientation_tolerance_rad
                    ),
                    move_timeout_sec=move_timeout_sec,
                    baseline_force_xyz=baseline_force_xyz,
                    max_force_n=max_force_n,
                    force_delta_n=force_delta_n,
                    cancelled=cancelled,
                    motion_cancelled=stage2_motion_cancelled,
                )
                return

            # Completion needs a fresh physical top-down check, not a sticky
            # acknowledgement from an earlier leveling iteration.  If later
            # clearance IK tilts the camera, return to LEVEL immediately.
            center_top_down = False
            if planner.phase == "ascend_clearance":
                try:
                    _, _, completion_back_away = self._camera_axes_in_base(
                        "center_camera", timeout_sec
                    )
                except Exception as error:
                    result.success = False
                    result.message = (
                        "permitted center-camera TF unavailable for fresh "
                        f"top-down completion check: {error}"
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
                center_top_down = (
                    completion_tilt_rad <= top_down_tolerance_rad
                )
                logging.info(
                    "fresh center-camera completion tilt=%.3frad "
                    "top_down=%s",
                    completion_tilt_rad,
                    center_top_down,
                )
                if not center_top_down:
                    planner.request_relevel()
                    logging.warning(
                        "center camera lost top-down alignment during "
                        "clearance; returning to joints 2-4 leveling"
                    )

            # Gate only the center report on physical top-down TF. Side reports
            # retain their own full/context evidence so the planner can require
            # a synchronized three-camera survey before releasing the arm.
            planning_reports = {
                name: replace(
                    item,
                    full=bool(
                        item.full
                        and (
                            name != "center_camera"
                            or center_top_down
                        )
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
                if not staged_sfp_target:
                    # This shared skill is also called before the NIC and SC
                    # branches.  Do not send those target modes through the
                    # loose-SFP CAD/PnP survey gate; it would change their
                    # downstream semantics and break the active process.
                    result.done = True
                    result.success = True
                    result.component_coverage_ready = True
                    result.steer_camera = "center_camera"
                    result.elapsed_seconds = max(
                        0.0, time.monotonic() - started_at
                    )
                    result.message = (
                        "legacy synchronized board-visibility survey verified "
                        f"for target mode {survey_target} after "
                        f"{result.moves_executed} adaptive moves"
                    )
                    return
                # Stage 1 has only acquired a coarse board view.  Do not expose
                # that as terminal success: Stage 2 estimates the board's full
                # 6-DoF pose and chooses one deterministic board-relative TCP
                # pose that frames the complete loose-SFP envelope in all
                # three calibrated cameras.
                if not self._stage2_has_complete_landmark(snapshot, reports):
                    if logo_acquisition_moves >= max_logo_acquisition_moves:
                        logging.info(
                            "Stage 1 used its %d measured acquisition moves; "
                            "handing the current landmark evidence to Stage 2 "
                            "for a calibrated fail-closed decision",
                            max_logo_acquisition_moves,
                        )
                        self._run_sfp_geometric_stage2(
                            snapshot=snapshot,
                            reports=reports,
                            result=result,
                            timeout_sec=timeout_sec,
                            deadline=overall_deadline,
                            started_at=started_at,
                            max_speed_mps=max_speed_mps,
                            max_angular_speed_rps=max_angular_speed_rps,
                            publish_hz=publish_hz,
                            settle_tolerance_m=settle_tolerance_m,
                            settle_orientation_tolerance_rad=(
                                settle_orientation_tolerance_rad
                            ),
                            move_timeout_sec=move_timeout_sec,
                            baseline_force_xyz=baseline_force_xyz,
                            max_force_n=max_force_n,
                            force_delta_n=force_delta_n,
                            cancelled=cancelled,
                            motion_cancelled=stage2_motion_cancelled,
                        )
                        return
                    acquired = self._move_to_acquire_complete_logo(
                        snapshot=snapshot,
                        reports=reports,
                        result=result,
                        timeout_sec=timeout_sec,
                        step_m=step_m,
                        max_speed_mps=max_speed_mps,
                        max_angular_speed_rps=max_angular_speed_rps,
                        publish_hz=publish_hz,
                        settle_tolerance_m=settle_tolerance_m,
                        settle_orientation_tolerance_rad=(
                            settle_orientation_tolerance_rad
                        ),
                        move_timeout_sec=move_timeout_sec,
                        baseline_force_xyz=baseline_force_xyz,
                        max_force_n=max_force_n,
                        force_delta_n=force_delta_n,
                        cancelled=cancelled,
                        motion_cancelled=stage2_motion_cancelled,
                    )
                    if not acquired:
                        return
                    logo_acquisition_moves += 1
                    iteration += 1
                    continue
                self._run_sfp_geometric_stage2(
                    snapshot=snapshot,
                    reports=reports,
                    result=result,
                    timeout_sec=timeout_sec,
                    deadline=overall_deadline,
                    started_at=started_at,
                    max_speed_mps=max_speed_mps,
                    max_angular_speed_rps=max_angular_speed_rps,
                    publish_hz=publish_hz,
                    settle_tolerance_m=settle_tolerance_m,
                    settle_orientation_tolerance_rad=(
                        settle_orientation_tolerance_rad
                    ),
                    move_timeout_sec=move_timeout_sec,
                    baseline_force_xyz=baseline_force_xyz,
                    max_force_n=max_force_n,
                    force_delta_n=force_delta_n,
                    cancelled=cancelled,
                    motion_cancelled=stage2_motion_cancelled,
                )
                return
            if action.terminal:
                if staged_sfp_target:
                    # A clipped board centroid is not proof that the purple
                    # logo is unusable.  Continue with measured logo/board
                    # acquisition instead of exposing the legacy planner's
                    # two-backoff STAGNATED result as a terminal failure.
                    if logo_acquisition_moves < max_logo_acquisition_moves:
                        acquired = self._move_to_acquire_complete_logo(
                            snapshot=snapshot,
                            reports=reports,
                            result=result,
                            timeout_sec=timeout_sec,
                            step_m=step_m,
                            max_speed_mps=max_speed_mps,
                            max_angular_speed_rps=max_angular_speed_rps,
                            publish_hz=publish_hz,
                            settle_tolerance_m=settle_tolerance_m,
                            settle_orientation_tolerance_rad=(
                                settle_orientation_tolerance_rad
                            ),
                            move_timeout_sec=move_timeout_sec,
                            baseline_force_xyz=baseline_force_xyz,
                            max_force_n=max_force_n,
                            force_delta_n=force_delta_n,
                            cancelled=cancelled,
                            motion_cancelled=stage2_motion_cancelled,
                        )
                        if not acquired:
                            return
                        logo_acquisition_moves += 1
                        iteration += 1
                        continue
                    logging.info(
                        "legacy Stage-1 planner stagnated after %d measured "
                        "logo/board acquisition moves; handing off to Stage 2",
                        logo_acquisition_moves,
                    )
                    self._run_sfp_geometric_stage2(
                        snapshot=snapshot,
                        reports=reports,
                        result=result,
                        timeout_sec=timeout_sec,
                        deadline=overall_deadline,
                        started_at=started_at,
                        max_speed_mps=max_speed_mps,
                        max_angular_speed_rps=max_angular_speed_rps,
                        publish_hz=publish_hz,
                        settle_tolerance_m=settle_tolerance_m,
                        settle_orientation_tolerance_rad=(
                            settle_orientation_tolerance_rad
                        ),
                        move_timeout_sec=move_timeout_sec,
                        baseline_force_xyz=baseline_force_xyz,
                        max_force_n=max_force_n,
                        force_delta_n=force_delta_n,
                        cancelled=cancelled,
                        motion_cancelled=stage2_motion_cancelled,
                    )
                    return
                result.success = False
                result.message = action.reason
                return

            # Level only after J1 centering and J6 long-axis alignment.  The
            # terminal view belongs to the center camera, so level its optical
            # axis rather than the TCP +Z axis (the camera is mounted about
            # 15 degrees off the tool axis).  The bounded Cartesian correction
            # and small +Z clearance are realized primarily through joints
            # 2-4 while the already-measured J1/J6 references remain the search
            # envelope anchors.
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
                        "straight down; the survey pitch is outside the "
                        "leveling range"
                    )
                    return
                if tilt_rad > top_down_tolerance_rad:
                    if initial_pose is None:
                        initial_pose = (
                            np.asarray(level_position, dtype=float),
                            normalize_quaternion(level_orientation),
                        )
                    pre_level_joint1 = current_level_joint1
                    pre_level_joint6 = current_level_joint6
                    level_axis = np.cross(camera_back_away, straight_up)
                    axis_norm = float(np.linalg.norm(level_axis))
                    if axis_norm < 1e-9:
                        result.success = False
                        result.message = (
                            "center-camera leveling axis is degenerate; aborting"
                        )
                        return
                    # Fixed-position orientation-only IK stalled in the live
                    # run before J6 was ever reached. Make the pitch correction
                    # in small bounded increments while adding a little base-Z
                    # clearance so joints 2-4 can move away from the singular
                    # posture instead of forcing the wrist to solve it alone.
                    level_step_rad = min(0.12, tilt_rad)
                    level_target_orientation = self._rotate_orientation_in_base(
                        level_orientation,
                        level_axis / axis_norm,
                        level_step_rad,
                    )
                    level_clearance_m = min(0.02, backoff_step_m)
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
                        "joints 2-4 top-view leveling after J6: center-camera "
                        "tilt=%.3frad step=%.3frad clearance=%.3fm "
                        "vertical_error=%+.3f image_direction=(%+.2f,%+.2f) "
                        "camera_plane_delta=(%+.4f,%+.4f,%+.4f)m; keeping "
                        "J1/J6 phase references fixed",
                        tilt_rad,
                        level_step_rad,
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
                        _, _, residual_back_away = self._camera_axes_in_base(
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
                        "joints 2-4 top-view stage complete: residual center "
                        "camera tilt=%.3frad angle_moved=%.3frad "
                        "target_reached=%s step_j1_drift=%+.4frad "
                        "step_j6_drift=%+.4frad cumulative_j1_drift=%+.4frad "
                        "cumulative_j6_drift=%+.4frad",
                        residual_tilt,
                        outcome.angular_distance_rad,
                        outcome.target_reached,
                        level_step_joint1_drift,
                        level_step_joint6_drift,
                        level_joint1_drift,
                        level_joint6_drift,
                    )
                    if (
                        abs(level_joint1_drift)
                        > level_joint_drift_tolerance_rad
                        or abs(level_joint6_drift)
                        > level_joint_drift_tolerance_rad
                    ):
                        planner.request_recenter()
                        logging.warning(
                            "joints 2-4 leveling changed J1/J6 by "
                            "%+.4f/%+.4frad; returning to visual J1/J6 "
                            "correction instead of failing",
                            level_joint1_drift,
                            level_joint6_drift,
                        )
                        iteration += 1
                        continue
                    if (
                        residual_tilt > top_down_tolerance_rad
                        and residual_tilt
                        >= tilt_rad - min_level_progress_rad
                    ):
                        result.success = False
                        result.message = (
                            "joints 2-4 leveling made less than 0.01rad "
                            "top-down progress; refusing to repeat the same "
                            "Cartesian request until the deadline"
                        )
                        return
                    if residual_tilt <= top_down_tolerance_rad:
                        planner.mark_level_complete()
                    iteration += 1
                    continue

                # Even when no leveling motion is needed, transition the
                # planner and capture one genuinely fresh terminal frame.
                planner.mark_level_complete()
                logging.info(
                    "center camera already top-down after J6; capturing the "
                    "first fresh completion frame"
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
                    "post-level joints 2-4 motion preserved phase anchors: "
                    "j1_drift=%+.4frad j6_drift=%+.4frad",
                    post_level_joint1_drift,
                    post_level_joint6_drift,
                )
                if (
                    abs(post_level_joint1_drift)
                    > level_joint_drift_tolerance_rad
                    or abs(post_level_joint6_drift)
                    > level_joint_drift_tolerance_rad
                ):
                    planner.request_recenter()
                    logging.warning(
                        "post-level %s motion changed J1/J6 by "
                        "%+.4f/%+.4frad; returning to visual correction",
                        action.kind.value,
                        post_level_joint1_drift,
                        post_level_joint6_drift,
                    )
                    iteration += 1
                    continue
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

    def _stage2_has_complete_landmark(self, snapshot, reports) -> bool:
        """Whether any calibrated camera has a complete Stage-2 landmark."""
        for camera_name in ("center_camera", "left_camera", "right_camera"):
            if (
                camera_name not in snapshot.frames
                or camera_name not in snapshot.calibrations
                or camera_name not in reports
            ):
                continue
            image = snapshot.frames[camera_name]["image"]
            ignored = self.gripper_masks.ignored_pixels(
                camera_name, image.shape
            )
            observation, _ = self._stage2_landmarks(
                image, reports[camera_name], ignored
            )
            if observation is not None:
                return True
        return False

    def _move_to_acquire_complete_logo(
        self,
        *,
        snapshot,
        reports,
        result,
        timeout_sec: float,
        step_m: float,
        max_speed_mps: float,
        max_angular_speed_rps: float,
        publish_hz: float,
        settle_tolerance_m: float,
        settle_orientation_tolerance_rad: float,
        move_timeout_sec: float,
        baseline_force_xyz,
        max_force_n: float,
        force_delta_n: float,
        cancelled,
        motion_cancelled,
    ) -> bool:
        """Make one measured board-fit or camera-plane logo correction.

        When the plate still fills or clips the image, a board-normal retreat
        is more informative than sliding the logo around the same cropped
        view.  Once scale is usable, use the logo direction for a bounded
        camera-plane correction.  Both choices are measured from the fresh
        frame; neither is a blind sweep.
        """
        import cv2

        from aic_perception.board_visibility import detect_purple_logo

        selected = None
        for camera_name in ("center_camera", "left_camera", "right_camera"):
            if camera_name not in snapshot.frames or camera_name not in reports:
                continue
            detected = detect_purple_logo(
                snapshot.frames[camera_name]["image"]
            )
            if detected is not None:
                selected = (camera_name, detected)
                break
        if selected is None:
            self._stage2_not_done(
                result,
                "purple pixels are unavailable, so a logo-acquisition move "
                "would be blind",
            )
            return False

        camera_name, (logo_mask, logo_centroid, _, _) = selected
        image = snapshot.frames[camera_name]["image"]
        height, width = image.shape[:2]
        ignored = self.gripper_masks.ignored_pixels(camera_name, image.shape)
        uncertainty = cv2.dilate(
            ignored.astype(np.uint8), np.ones((9, 9), np.uint8)
        ).astype(bool)
        center = np.array((0.5 * (width - 1), 0.5 * (height - 1)), dtype=float)
        desired_image = center - np.asarray(logo_centroid, dtype=float)
        desired_image /= np.array(
            (max(1.0, 0.5 * width), max(1.0, 0.5 * height)), dtype=float
        )
        if np.any(logo_mask.astype(bool) & uncertainty):
            logo_y, logo_x = np.nonzero(logo_mask)
            mask_y, mask_x = np.nonzero(uncertainty)
            if logo_x.size and mask_x.size:
                escape = np.array(
                    (
                        float(logo_x.mean() - mask_x.mean()) / max(1.0, width),
                        float(logo_y.mean() - mask_y.mean()) / max(1.0, height),
                    ),
                    dtype=float,
                )
                if float(np.linalg.norm(escape)) > 1e-6:
                    desired_image = escape
        source_report = reports[camera_name]
        prefer_backoff = bool(
            source_report.area_frac >= 0.32
            or "context_clipped" in source_report.failure_reasons
        )
        desired_norm = float(np.linalg.norm(desired_image))
        if desired_norm < 1e-4 and not prefer_backoff:
            self._stage2_not_done(
                result,
                f"{camera_name} logo is visible but its complete outline "
                "cannot be recovered; refusing an unmeasured direction",
            )
            return False
        if desired_norm >= 1e-4:
            desired_image /= desired_norm
        try:
            position, orientation = self._gripper_pose(timeout_sec)
            image_right, image_down, camera_back_away = self._camera_axes_in_base(
                camera_name, timeout_sec
            )
        except Exception as error:
            self._stage2_not_done(
                result, f"logo-acquisition camera/TCP TF unavailable: {error}"
            )
            return False
        if prefer_backoff:
            # The camera back-away axis is derived from the current optical
            # TF.  This is the same physical zoom-out direction as the
            # legacy BACKOFF action, but it remains available after that
            # planner's two-backoff centroid limit.
            correction_m = min(0.045, max(0.020, 0.9 * step_m))
            delta = correction_m * np.asarray(camera_back_away, dtype=float)
            acquisition_mode = "board_fit_backoff"
        else:
            # An object moves opposite camera translation in the image.
            correction_m = min(0.025, max(0.010, 0.5 * step_m))
            delta = -correction_m * (
                float(desired_image[0]) * np.asarray(image_right, dtype=float)
                + float(desired_image[1]) * np.asarray(image_down, dtype=float)
            )
            acquisition_mode = "logo_plane_shift"
        target_position_array = np.asarray(position, dtype=float) + delta
        target_position = tuple(float(value) for value in target_position_array)
        fresh_force = snapshot.force_xyz
        if fresh_force is None:
            fresh_force = self.camera_rig.wait_for_force_xyz(
                timeout_sec=timeout_sec, max_age_sec=0.5
            )
        if fresh_force is None:
            self._stage2_not_done(
                result,
                "no fresh wrist-force sample for bounded logo acquisition",
            )
            return False
        if baseline_force_xyz is None:
            baseline_force_xyz = fresh_force
        if cancelled():
            raise skill_interface.SkillCancelledError(
                "board search cancelled before logo acquisition"
            )

        result.last_action = "acquire_complete_purple_logo"
        result.target_valid = True
        result.target_frame = self.config.base_frame
        result.target.x, result.target.y, result.target.z = target_position
        result.dx, result.dy, result.dz = (
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
        )
        logging.info(
            "bounded Stage-1 acquisition mode=%s camera=%s "
            "desired_image=(%+.3f,%+.3f) delta=(%+.4f,%+.4f,%+.4f)m",
            acquisition_mode,
            camera_name,
            desired_image[0],
            desired_image[1],
            delta[0],
            delta[1],
            delta[2],
        )
        outcome = self.robot_motion.move_smooth(
            target_position,
            target_orientation=orientation,
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
            self._stage2_not_done(
                result, "search deadline reached during logo acquisition"
            )
            return False
        if not outcome.success:
            result.force_abort = outcome.force_abort
            self._stage2_not_done(
                result, f"bounded logo-acquisition move failed: {outcome.message}"
            )
            return False
        result.moves_executed += 1
        result.travel_m += float(outcome.distance_m)
        result.angular_travel_rad += float(outcome.angular_distance_rad)
        result.moved = result.moved or (
            outcome.distance_m > 0.0 or outcome.angular_distance_rad > 0.0
        )
        return True

    @staticmethod
    def _uses_staged_sfp_stage2(survey_target: int) -> bool:
        """Whether this invocation owns the loose staged-SFP survey gate."""
        # UNSPECIFIED is retained as the historical pre-enum SFP default.
        return int(survey_target) in (0, 1)

    @staticmethod
    def _stage2_not_done(result, reason: str) -> None:
        """Return a geometric-stage rejection without aborting the flowchart."""
        result.success = True
        result.done = False
        result.component_coverage_ready = False
        result.target_valid = False
        result.last_action = "sfp_geometric_stage2_rejected"
        result.message = f"SFP geometric Stage 2 not ready: {reason}"

    @staticmethod
    def _stage2_landmarks(image, report, ignored_pixels):
        """Extract a complete logo and a four-corner board observation.

        Stage 2 deliberately refuses a centroid from a cropped or
        gripper-covered magenta fragment.  The board quad is recovered from
        the full dark component anchored by that complete logo; the downstream
        PnP solver, rather than an image scale heuristic, then recovers the
        board's arbitrary 6-DoF pose.
        """
        import cv2

        from aic_perception.board_visibility import detect_purple_logo

        if not report.seen or not report.full:
            return None, "source camera does not contain the complete board"
        logo = detect_purple_logo(image)
        if logo is None:
            return None, "complete purple logo was not detected"
        logo_mask, logo_centroid, logo_area, logo_bbox = logo
        height, width = image.shape[:2]
        x0, y0, x1, y1 = logo_bbox
        logo_margin = min(x0, y0, width - 1 - x1, height - 1 - y1)
        if logo_margin < 8:
            return None, "purple logo touches the physical image boundary"
        ignored = np.asarray(ignored_pixels, dtype=bool)
        if ignored.shape != (height, width):
            return None, "gripper mask dimensions do not match the image"
        uncertainty = cv2.dilate(
            ignored.astype(np.uint8), np.ones((9, 9), np.uint8)
        ).astype(bool)
        if np.any(logo_mask.astype(bool) & uncertainty):
            return None, "purple logo intersects the gripper uncertainty mask"
        logo_width = x1 - x0 + 1
        logo_height = y1 - y0 + 1
        logo_box_area = float(logo_width * logo_height)
        logo_fill = float(logo_area) / max(1.0, logo_box_area)
        logo_aspect = float(logo_width) / max(1.0, float(logo_height))
        if (
            min(logo_width, logo_height) < 12
            or not 0.05 <= logo_fill <= 0.85
            or not 0.35 <= logo_aspect <= 2.8
        ):
            return None, "purple logo shape is clipped or too small for pose"

        source = np.asarray(image)
        if source.ndim == 3 and source.shape[2] == 4:
            gray = cv2.cvtColor(source, cv2.COLOR_BGRA2GRAY)
        elif source.ndim == 3:
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            gray = source
        _, dark = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        dark[ignored] = 0
        dark[logo_mask.astype(bool)] = 255
        # Remove thin cable/card protrusions while preserving the plate core.
        kernel_size = max(5, int(round(0.012 * min(height, width))) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        plate = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        plate = cv2.morphologyEx(plate, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(plate, 8)
        if count <= 1:
            return None, "dark board component could not be recovered"
        logo_x = int(round(logo_centroid[0]))
        logo_y = int(round(logo_centroid[1]))
        label = int(labels[logo_y, logo_x])
        if label == 0:
            # The logo can sit on a bright inlay. Choose the plate component
            # with the greatest overlap with the Stage-1 board bounding box.
            if report.bbox is None:
                return None, "logo is not connected to a board component"
            bx0, by0, bx1, by1 = report.bbox
            overlap_scores = []
            for index in range(1, count):
                sx, sy, sw, sh, _ = stats[index]
                overlap = max(0, min(sx + sw, bx1 + 1) - max(sx, bx0))
                overlap *= max(0, min(sy + sh, by1 + 1) - max(sy, by0))
                overlap_scores.append((overlap, index))
            label = max(overlap_scores)[1]
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, "board outline contour is unavailable"
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        quad = None
        for epsilon_frac in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
            approximation = cv2.approxPolyDP(
                hull, epsilon_frac * perimeter, True
            )
            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                quad = approximation.reshape(4, 2).astype(float)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
        if abs(float(cv2.contourArea(quad.astype(np.float32)))) < 0.08 * (
            height * width
        ):
            return None, "board outline is too small for stable planar PnP"
        quad_margin = min(
            float(quad[:, 0].min()),
            float(quad[:, 1].min()),
            float(width - 1 - quad[:, 0].max()),
            float(height - 1 - quad[:, 1].max()),
        )
        if quad_margin < 3.0:
            return None, "board outline touches the physical image boundary"
        return (quad, np.asarray(logo_centroid, dtype=float)), "ok"

    def _base_transform_at(
        self, child_frame: str, stamp_ns: int, timeout_sec: float
    ):
        """Return ``base_link_T_child`` at an image timestamp, never latest.

        Camera projection is only meaningful when the base-to-camera and
        base-to-TCP transforms describe the same captured image.  Static TFs
        are allowed to report timestamp zero.  Dynamic transforms must either
        be returned at the requested time or within 50 ms of it.
        """
        from aic_perception.board_stage2 import Transform
        from rclpy.duration import Duration
        from rclpy.time import Time

        permitted = set(self.config.camera_frames.values())
        permitted.add(self.config.gripper_frame)
        if child_frame not in permitted:
            raise ValueError(f"TF frame {child_frame!r} is outside the allowlist")
        if not isinstance(stamp_ns, (int, np.integer)) or int(stamp_ns) <= 0:
            raise ValueError("image timestamp is invalid for TF lookup")
        requested = Time(nanoseconds=int(stamp_ns))
        stamped = self.tf_buffer.lookup_transform(
            self.config.base_frame,
            child_frame,
            requested,
            timeout=Duration(seconds=min(timeout_sec, 3.0)),
        )
        header_stamp = stamped.header.stamp
        returned_ns = int(header_stamp.sec) * 1_000_000_000 + int(
            header_stamp.nanosec
        )
        # Static transforms conventionally have a zero header stamp.  For a
        # dynamic edge, do not silently accept a stale transform.
        if returned_ns and abs(returned_ns - int(stamp_ns)) > 50_000_000:
            raise ValueError(
                f"TF for {child_frame} is {abs(returned_ns - int(stamp_ns)) / 1e6:.1f}ms "
                "away from the image timestamp"
            )
        translation = stamped.transform.translation
        rotation = stamped.transform.rotation
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
            raise ValueError(f"TF for {child_frame} contains non-finite values")
        if float(np.linalg.norm(values[3:])) < 0.5:
            raise ValueError(f"TF quaternion for {child_frame} is uninitialized")
        return Transform.from_quaternion(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
            (float(translation.x), float(translation.y), float(translation.z)),
        )

    def _run_sfp_geometric_stage2(
        self,
        *,
        snapshot,
        reports,
        result,
        timeout_sec: float,
        deadline: float,
        started_at: float,
        max_speed_mps: float,
        max_angular_speed_rps: float,
        publish_hz: float,
        settle_tolerance_m: float,
        settle_orientation_tolerance_rad: float,
        move_timeout_sec: float,
        baseline_force_xyz,
        max_force_n: float,
        force_delta_n: float,
        cancelled,
        motion_cancelled,
    ) -> None:
        """Estimate, execute, and verify one board-relative loose-SFP pose."""
        from aic_perception.board_stage2 import (
            CameraModel,
            GripperExclusion,
            board_pose_set_is_consistent,
            estimate_board_pose,
            quaternion_from_matrix,
            sampled_cartesian_path_is_safe,
            search_survey_pose,
            verify_survey_view,
        )

        expected = tuple(sorted(self.config.camera_frames))
        if snapshot.force_xyz is None:
            force_wait_sec = min(
                timeout_sec, max(0.0, deadline - time.monotonic())
            )
            fresh_force = self.camera_rig.wait_for_force_xyz(
                timeout_sec=force_wait_sec,
                max_age_sec=0.5,
            )
            if fresh_force is None:
                self._stage2_not_done(
                    result,
                    "no fresh wrist-force sample after waiting; refusing the "
                    "geometric SFP survey move",
                )
                return
            if baseline_force_xyz is None:
                baseline_force_xyz = fresh_force
        missing_frames = sorted(set(expected) - set(snapshot.frames))
        missing_calibration = sorted(set(expected) - set(snapshot.calibrations))
        if missing_frames:
            self._stage2_not_done(
                result, f"fresh Stage-1 images missing {missing_frames}"
            )
            return
        if missing_calibration:
            self._stage2_not_done(
                result, f"approved CameraInfo missing {missing_calibration}"
            )
            return

        camera_models = {}
        for camera_name in expected:
            calibration = snapshot.calibrations[camera_name]
            frame = snapshot.frames[camera_name]
            from aic_perception.camera_rig import (
                frames_are_approved_camera_pair,
            )

            if not frames_are_approved_camera_pair(
                str(frame.get("frame_id", "")),
                calibration.frame_id,
                self.config.camera_frames[camera_name],
            ):
                self._stage2_not_done(
                    result,
                    f"{camera_name} image/CameraInfo frame is outside the "
                    "sensor-link/optical allowlist",
                )
                return
            if (
                calibration.height,
                calibration.width,
            ) != frame["image"].shape[:2]:
                self._stage2_not_done(
                    result, f"{camera_name} CameraInfo/image dimensions differ"
                )
                return
            try:
                camera_models[camera_name] = CameraModel(
                    name=camera_name,
                    K=calibration.camera_matrix,
                    width=calibration.width,
                    height=calibration.height,
                    distortion=np.asarray(
                        calibration.distortion, dtype=float
                    ),
                    distortion_model=calibration.distortion_model,
                )
            except ValueError as error:
                self._stage2_not_done(
                    result, f"{camera_name} calibration rejected: {error}"
                )
                return

        # Prefer the center, but a complete logo in either calibrated side
        # camera is sufficient. This is a *seed* for Stage 2, not a Stage-1
        # handoff criterion: a somewhat noisy outline may still be adequate
        # to plan a bounded move whose path, IK, force, and final all-camera
        # visibility verification remain independently fail-closed.
        observations = {}
        for camera_name in ("center_camera", "left_camera", "right_camera"):
            if camera_name not in snapshot.frames:
                continue
            image = snapshot.frames[camera_name]["image"]
            ignored = self.gripper_masks.ignored_pixels(
                camera_name, image.shape
            )
            observations[camera_name] = self._stage2_landmarks(
                image, reports[camera_name], ignored
            )
        complete_cameras = [
            camera_name
            for camera_name in (
                "center_camera",
                "left_camera",
                "right_camera",
            )
            if camera_name in observations
            and observations[camera_name][0] is not None
        ]
        if not complete_cameras:
            reasons = "; ".join(
                f"{name}: {reason}"
                for name, (_, reason) in observations.items()
            )
            self._stage2_not_done(
                result,
                "no calibrated camera contains a complete unobstructed purple "
                f"logo and board outline ({reasons})",
            )
            return

        try:
            # All transforms are evaluated at the actual corresponding image
            # time.  Images can legally name sensor_link; their calibrated
            # intrinsics are nevertheless optical, so only the fixed optical
            # child is projected through and no message-provided TF frame is
            # ever queried.
            base_T_tcp_by_camera = {
                name: self._base_transform_at(
                    self.config.gripper_frame,
                    int(snapshot.frames[name]["stamp_ns"]),
                    timeout_sec,
                )
                for name in expected
            }
            base_T_cam = {
                name: self._base_transform_at(
                    self.config.camera_frames[name],
                    int(snapshot.frames[name]["stamp_ns"]),
                    timeout_sec,
                )
                for name in expected
            }
        except Exception as error:
            self._stage2_not_done(
                result,
                "timestamp-bound permitted camera/TCP TF unavailable: "
                f"{error}",
            )
            return
        tcp_T_cam = {
            name: base_T_tcp_by_camera[name].inverse().compose(transform)
            for name, transform in base_T_cam.items()
        }

        pose_estimates = []
        pose_failures = {}
        for camera_name in complete_cameras:
            (board_quad, logo_centroid), _ = observations[camera_name]
            estimate, pose_reason = estimate_board_pose(
                board_quad,
                logo_centroid,
                camera_models[camera_name],
                base_T_cam[camera_name],
                # The default 6 px threshold is appropriate for accepting a
                # final measured board pose.  It was incorrectly used as a
                # pre-motion handoff gate, causing Stage 2 to return without
                # trying a safe survey pose in otherwise usable scenes.  Use
                # this estimate only as a bounded motion seed; completion
                # still uses fresh, strict all-camera verification below.
                max_reprojection_error_px=20.0,
                max_logo_error_px=120.0,
            )
            if estimate is None:
                pose_failures[camera_name] = pose_reason
            else:
                pose_estimates.append(estimate)
        if not pose_estimates:
            self._stage2_not_done(
                result,
                "board pose confidence rejected in every complete-logo camera: "
                + "; ".join(
                    f"{name}={reason}"
                    for name, reason in pose_failures.items()
                ),
            )
            return
        # Use all accepted cameras as a consistency check. A rejected center
        # hypothesis does not hide a valid side-camera estimate, but two
        # mutually contradictory accepted estimates may not be guessed
        # between. Select the largest consistent cluster and prefer center
        # within it.
        clusters = []
        for candidate_estimate in pose_estimates:
            cluster = []
            for other in pose_estimates:
                translation_error = float(
                    np.linalg.norm(
                        candidate_estimate.base_T_board.translation
                        - other.base_T_board.translation
                    )
                )
                rotation_delta = (
                    candidate_estimate.base_T_board.rotation.T
                    @ other.base_T_board.rotation
                )
                angle_error = math.acos(
                    float(
                        np.clip(
                            0.5 * (np.trace(rotation_delta) - 1.0),
                            -1.0,
                            1.0,
                        )
                    )
                )
                if translation_error <= 0.05 and angle_error <= math.radians(8):
                    cluster.append(other)
            clusters.append(cluster)
        consistent = max(
            clusters,
            key=lambda cluster: (
                len(cluster),
                any(item.camera_name == "center_camera" for item in cluster),
            ),
        )
        if len(pose_estimates) > 1 and len(consistent) < 2:
            self._stage2_not_done(
                result,
                "accepted camera pose estimates disagree by more than "
                "5 cm / 8 degrees",
            )
            return
        board_pose = next(
            (
                item
                for item in consistent
                if item.camera_name == "center_camera"
            ),
            min(
                consistent,
                key=lambda item: (
                    item.reprojection_error_px,
                    item.logo_error_px,
                ),
            ),
        )
        source_camera = board_pose.camera_name
        base_T_tcp = base_T_tcp_by_camera[source_camera]

        grippers = {}
        for camera_name in expected:
            shape = snapshot.frames[camera_name]["image"].shape
            gripper_mask = self.gripper_masks.ignored_pixels(
                camera_name, shape
            )
            grippers[camera_name] = GripperExclusion(
                mask=gripper_mask, margin_px=32.0
            )
        candidate, search_reason = search_survey_pose(
            board_pose,
            tcp_T_cam,
            camera_models,
            grippers,
            reference_camera="center_camera",
            current_base_T_tcp=base_T_tcp,
        )
        if candidate is None:
            self._stage2_not_done(
                result, f"no safe all-camera SFP survey pose: {search_reason}"
            )
            return
        target = candidate.base_T_tcp
        displacement = target.translation - base_T_tcp.translation
        distance_m = float(np.linalg.norm(displacement))
        if distance_m > 0.65:
            self._stage2_not_done(
                result,
                f"computed survey pose is {distance_m:.3f}m away; refusing "
                "an unsafe one-shot Cartesian move",
            )
            return
        target_rotation_delta = base_T_tcp.rotation.T @ target.rotation
        orientation_distance = math.acos(
            float(
                np.clip(
                    0.5 * (np.trace(target_rotation_delta) - 1.0),
                    -1.0,
                    1.0,
                )
            )
        )
        if (
            not math.isfinite(orientation_distance)
            or orientation_distance > math.radians(45.0) + 1e-6
        ):
            self._stage2_not_done(
                result,
                "computed survey orientation is "
                f"{orientation_distance:.2f}rad from the acquired pose; "
                "refusing an unsafe one-shot rotation",
            )
            return
        if (
            float(target.translation[2]) < 0.02
            or float(np.linalg.norm(target.translation)) > 1.20
        ):
            self._stage2_not_done(
                result, "computed survey TCP lies outside the workspace guard"
            )
            return
        board_normal = np.asarray(
            board_pose.base_T_board.rotation[:, 2], dtype=float
        )
        board_origin = np.asarray(
            board_pose.base_T_board.translation, dtype=float
        )
        current_clearance = float(
            np.dot(base_T_tcp.translation - board_origin, board_normal)
        )
        target_clearance = float(
            np.dot(target.translation - board_origin, board_normal)
        )
        lateral_delta = displacement - np.dot(
            displacement, board_normal
        ) * board_normal

        def path_is_safe(
            start: np.ndarray,
            end: np.ndarray,
            *,
            minimum_clearance: float,
            allow_outward_retreat: bool = False,
        ) -> bool:
            return sampled_cartesian_path_is_safe(
                start,
                end,
                board_origin=board_origin,
                board_normal=board_normal,
                minimum_clearance=minimum_clearance,
                allow_outward_retreat=allow_outward_retreat,
            )

        if target_clearance < 0.12:
            self._stage2_not_done(
                result,
                "computed SFP survey pose is too close to the board plane",
            )
            return
        needs_orientation_waypoint = (
            orientation_distance > settle_orientation_tolerance_rad
        )
        rotation_clearance_m = 0.40
        needs_retreat = current_clearance < 0.12
        # A lateral leg near the board is never sent directly, even when the
        # endpoints are individually legal.  First establish a 16-cm normal
        # standoff, then transit laterally.  This is the one and only retreat
        # Stage 2 may insert.
        needs_retreat = needs_retreat or (
            float(np.linalg.norm(lateral_delta)) > 0.08
            and min(current_clearance, target_clearance) < 0.16
        )
        # The repository exposes no supported IK/collision-query service.
        # Never sweep the ~35 cm wrist-camera rig beside the board.  If a
        # meaningful orientation change is required, retreat with the acquired
        # orientation held, rotate in place beyond the rig's conservative
        # bounding radius, then translate with the final orientation fixed.
        needs_retreat = needs_retreat or (
            needs_orientation_waypoint
            and current_clearance < rotation_clearance_m
        )
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            self._stage2_not_done(
                result, "search deadline left no time for the geometric move"
            )
            return
        if cancelled():
            raise skill_interface.SkillCancelledError(
                "board search cancelled before geometric SFP motion"
            )

        if needs_retreat:
            required_retreat_clearance = (
                rotation_clearance_m
                if needs_orientation_waypoint
                else 0.16
            )
            retreat_distance = max(
                0.0, required_retreat_clearance - current_clearance
            )
            retreat_position_array = (
                base_T_tcp.translation + retreat_distance * board_normal
            )
            retreat_position = tuple(
                float(value) for value in retreat_position_array
            )
            if not path_is_safe(
                base_T_tcp.translation,
                retreat_position_array,
                minimum_clearance=0.12,
                allow_outward_retreat=True,
            ) or not path_is_safe(
                retreat_position_array,
                target.translation,
                minimum_clearance=0.12,
            ):
                self._stage2_not_done(
                    result,
                    "retreat/direct geometric SFP path violates sampled "
                    "workspace or board-normal clearance",
                )
                return
            logging.info(
                "SFP Stage 2 inserting board-normal retreat %.3fm before "
                "%.3fm lateral transit (clearance %.3fm)",
                retreat_distance,
                float(np.linalg.norm(lateral_delta)),
                current_clearance,
            )
            retreat = self.robot_motion.move_smooth(
                retreat_position,
                target_orientation=quaternion_from_matrix(base_T_tcp.rotation),
                max_speed_mps=max_speed_mps,
                max_angular_speed_radps=max_angular_speed_rps,
                publish_hz=publish_hz,
                settle_tolerance_m=settle_tolerance_m,
                settle_angular_tolerance_rad=(
                    settle_orientation_tolerance_rad
                ),
                timeout_sec=min(remaining, max(move_timeout_sec, 8.0)),
                baseline_force_xyz=baseline_force_xyz,
                max_force_n=max_force_n,
                force_delta_n=force_delta_n,
                cancelled=motion_cancelled,
            )
            if retreat.cancelled:
                if cancelled():
                    raise skill_interface.SkillCancelledError(retreat.message)
                self._stage2_not_done(
                    result,
                    "search deadline reached during board-normal retreat",
                )
                return
            if not retreat.success or not retreat.target_reached:
                result.force_abort = retreat.force_abort
                self._stage2_not_done(
                    result,
                    "board-normal retreat did not reach its safe waypoint: "
                    f"{retreat.message}",
                )
                return
            result.moves_executed += 1
            result.travel_m += float(retreat.distance_m)
            result.angular_travel_rad += float(retreat.angular_distance_rad)
            result.moved = True
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                self._stage2_not_done(
                    result,
                    "search deadline expired after the board-normal retreat",
                )
                return
        elif not path_is_safe(
            base_T_tcp.translation,
            target.translation,
            minimum_clearance=0.12,
        ):
            self._stage2_not_done(
                result,
                "direct geometric SFP path fails sampled clearance guard",
            )
            return

        target_position = tuple(float(value) for value in target.translation)
        target_orientation = quaternion_from_matrix(target.rotation)
        orientation_position_array = (
            retreat_position_array
            if needs_retreat
            else base_T_tcp.translation
        )
        if needs_orientation_waypoint:
            orientation_clearance = float(
                np.dot(
                    orientation_position_array - board_origin,
                    board_normal,
                )
            )
            if orientation_clearance < rotation_clearance_m:
                self._stage2_not_done(
                    result,
                    "no conservative board clearance for the planned wrist "
                    "orientation waypoint",
                )
                return
            orientation_position = tuple(
                float(value) for value in orientation_position_array
            )
            orientation_outcome = self.robot_motion.move_smooth(
                orientation_position,
                target_orientation=target_orientation,
                max_speed_mps=max_speed_mps,
                max_angular_speed_radps=max_angular_speed_rps,
                publish_hz=publish_hz,
                settle_tolerance_m=settle_tolerance_m,
                settle_angular_tolerance_rad=(
                    settle_orientation_tolerance_rad
                ),
                timeout_sec=min(remaining, max(move_timeout_sec, 8.0)),
                baseline_force_xyz=baseline_force_xyz,
                max_force_n=max_force_n,
                force_delta_n=force_delta_n,
                cancelled=motion_cancelled,
            )
            if orientation_outcome.cancelled:
                if cancelled():
                    raise skill_interface.SkillCancelledError(
                        orientation_outcome.message
                    )
                self._stage2_not_done(
                    result,
                    "search deadline reached during the safe orientation "
                    "waypoint",
                )
                return
            if (
                not orientation_outcome.success
                or not orientation_outcome.target_reached
            ):
                result.force_abort = orientation_outcome.force_abort
                self._stage2_not_done(
                    result,
                    "safe orientation waypoint did not complete: "
                    f"{orientation_outcome.message}",
                )
                return
            result.moves_executed += 1
            result.travel_m += float(orientation_outcome.distance_m)
            result.angular_travel_rad += float(
                orientation_outcome.angular_distance_rad
            )
            result.moved = True
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                self._stage2_not_done(
                    result,
                    "search deadline expired after the safe orientation "
                    "waypoint",
                )
                return
        result.last_action = "sfp_geometric_stage2_move"
        result.target_valid = True
        result.target_frame = self.config.base_frame
        result.target.x, result.target.y, result.target.z = target_position
        result.dx, result.dy, result.dz = (
            float(displacement[0]),
            float(displacement[1]),
            float(displacement[2]),
        )
        logging.info(
            "SFP Stage 2 source=%s reprojection=%.2fpx target=(%.4f,%.4f,"
            "%.4f)m standoff=%.3fm yaw=%+.3frad min_clearance=%.1fpx",
            source_camera,
            board_pose.reprojection_error_px,
            target_position[0],
            target_position[1],
            target_position[2],
            candidate.standoff_m,
            candidate.yaw_rad,
            candidate.min_clearance_px,
        )
        outcome = self.robot_motion.move_smooth(
            target_position,
            target_orientation=target_orientation,
            max_speed_mps=max_speed_mps,
            max_angular_speed_radps=max_angular_speed_rps,
            publish_hz=publish_hz,
            settle_tolerance_m=settle_tolerance_m,
            settle_angular_tolerance_rad=settle_orientation_tolerance_rad,
            timeout_sec=min(remaining, max(move_timeout_sec, 12.0)),
            baseline_force_xyz=baseline_force_xyz,
            max_force_n=max_force_n,
            force_delta_n=force_delta_n,
            cancelled=motion_cancelled,
        )
        if outcome.cancelled:
            if cancelled():
                raise skill_interface.SkillCancelledError(outcome.message)
            self._stage2_not_done(
                result, "search deadline reached during geometric SFP motion"
            )
            return
        if not outcome.success:
            result.force_abort = outcome.force_abort
            self._stage2_not_done(
                result, f"geometric SFP motion did not complete: {outcome.message}"
            )
            return
        result.moves_executed += 1
        result.travel_m += float(outcome.distance_m)
        result.angular_travel_rad += float(outcome.angular_distance_rad)
        result.moved = result.moved or (
            outcome.distance_m > 0.0 or outcome.angular_distance_rad > 0.0
        )
        if not outcome.target_reached:
            self._stage2_not_done(
                result,
                "controller stopped safely before the computed SFP survey pose",
            )
            return

        verification_timeout = min(
            timeout_sec, max(0.0, deadline - time.monotonic())
        )
        if verification_timeout <= 0.0:
            self._stage2_not_done(
                result, "search deadline expired before fresh verification"
            )
            return
        fresh = self.camera_rig.grab(
            timeout_sec=verification_timeout,
            min_cameras=len(expected),
            collection_grace_sec=0.0,
        )
        if fresh is None or set(fresh.frames) != set(expected):
            self._stage2_not_done(
                result, "fresh three-camera verification triplet unavailable"
            )
            return
        if not fresh.frames_within_skew(50_000_000):
            self._stage2_not_done(
                result,
                "fresh three-camera verification exceeds 50 ms timestamp skew",
            )
            return
        from aic_perception.board_visibility import analyze_board

        image_rejections = {}
        fresh_reports = {}
        for camera_name in expected:
            image = fresh.frames[camera_name]["image"]
            ignored = self.gripper_masks.ignored_pixels(
                camera_name, image.shape
            )
            fresh_report = analyze_board(
                image,
                margin_px=3,
                min_area_frac=0.001,
                ignore_bottom_frac=0.0,
                min_contrast=20.0,
                min_rectangularity=0.20,
                min_detail_area_frac=0.005,
                context_pad_frac=0.0,
                ignore_mask=ignored,
            )
            fresh_reports[camera_name] = fresh_report
            reasons = []
            if not fresh_report.seen:
                reasons.append("board_context_not_detected")
            if fresh_report.artificial_bottom_contact:
                reasons.append("board_contacts_gripper_mask")
            if fresh_report.gripper_overlap_px > 0:
                reasons.append(
                    f"board_gripper_overlap={fresh_report.gripper_overlap_px}px"
                )
            if reasons:
                image_rejections[camera_name] = reasons
        if image_rejections:
            details = "; ".join(
                f"{name}={','.join(reasons)}"
                for name, reasons in image_rejections.items()
            )
            self._stage2_not_done(
                result,
                "fresh camera pixels rejected the predicted SFP survey view: "
                f"{details}",
            )
            return
        try:
            # Do not project a fresh image using the pose at verification
            # wall-clock time.  Each board PnP and each camera projection is
            # tied to that camera's frame timestamp.
            fresh_base_T_tcp = {
                name: self._base_transform_at(
                    self.config.gripper_frame,
                    int(fresh.frames[name]["stamp_ns"]),
                    timeout_sec,
                )
                for name in expected
            }
            fresh_base_T_cam = {
                name: self._base_transform_at(
                    self.config.camera_frames[name],
                    int(fresh.frames[name]["stamp_ns"]),
                    timeout_sec,
                )
                for name in expected
            }
        except Exception as error:
            self._stage2_not_done(
                result,
                "timestamp-bound post-move camera/TCP TF unavailable: "
                f"{error}",
            )
            return
        fresh_estimates = {}
        fresh_pose_failures = {}
        for camera_name in (
            "center_camera",
            "left_camera",
            "right_camera",
        ):
            image = fresh.frames[camera_name]["image"]
            ignored = self.gripper_masks.ignored_pixels(
                camera_name, image.shape
            )
            observation, reason = self._stage2_landmarks(
                image, fresh_reports[camera_name], ignored
            )
            if observation is None:
                fresh_pose_failures[camera_name] = reason
                continue
            fresh_quad, fresh_logo = observation
            estimate, reason = estimate_board_pose(
                fresh_quad,
                fresh_logo,
                camera_models[camera_name],
                fresh_base_T_cam[camera_name],
            )
            if estimate is None:
                fresh_pose_failures[camera_name] = reason
            else:
                fresh_estimates[camera_name] = estimate
        fresh_consistent, fresh_consistency_reason = (
            board_pose_set_is_consistent(
                fresh_estimates,
                board_pose,
                expected,
            )
        )
        if not fresh_consistent:
            self._stage2_not_done(
                result,
                "fresh triplet board poses are incomplete or inconsistent: "
                f"{fresh_consistency_reason}; "
                + "; ".join(
                    f"{name}={reason}"
                    for name, reason in fresh_pose_failures.items()
                )
            )
            return
        # Verify each projection against the TCP and optical TF captured at
        # *that* camera's image time.  `verify_survey_view` intentionally
        # accepts a mapping, so one-camera calls avoid smearing the rig pose
        # across a 50-ms synchronized triplet.
        verification_by_camera = {}
        for estimate in fresh_estimates.values():
            name = estimate.camera_name
            verification_by_camera[name] = verify_survey_view(
                estimate,
                fresh_base_T_tcp[name],
                {name: tcp_T_cam[name]},
                {name: camera_models[name]},
                {name: grippers[name]},
                {name: int(fresh.frames[name]["stamp_ns"])},
                max_skew_ns=50_000_000,
            )
        missing_verified = set(expected) - set(verification_by_camera)
        verification_failed = {
            name: check
            for name, check in verification_by_camera.items()
            if not check.passed
        }
        if missing_verified or verification_failed:
            details = "; ".join(
                f"{name}="
                f"{','.join(coverage.reasons) or coverage.reason}"
                for name, coverage in verification_by_camera.items()
            )
            self._stage2_not_done(
                result,
                f"fresh all-camera SFP verification failed: "
                f"missing={sorted(missing_verified)} ({details})",
            )
            return

        confirmation_timeout = min(
            timeout_sec, max(0.0, deadline - time.monotonic())
        )
        if confirmation_timeout <= 0.0:
            self._stage2_not_done(
                result, "deadline expired before the settled confirmation triplet"
            )
            return
        confirmation = self.camera_rig.grab(
            timeout_sec=confirmation_timeout,
            min_cameras=len(expected),
            collection_grace_sec=0.0,
        )
        if (
            confirmation is None
            or set(confirmation.frames) != set(expected)
            or not confirmation.frames_within_skew(50_000_000)
        ):
            self._stage2_not_done(
                result, "second settled three-camera triplet is unavailable"
            )
            return
        confirmation_rejections = {}
        confirmation_reports = {}
        for camera_name in expected:
            image = confirmation.frames[camera_name]["image"]
            ignored = self.gripper_masks.ignored_pixels(
                camera_name, image.shape
            )
            confirmation_report = analyze_board(
                image,
                margin_px=3,
                min_area_frac=0.001,
                ignore_bottom_frac=0.0,
                min_contrast=20.0,
                min_rectangularity=0.20,
                min_detail_area_frac=0.005,
                context_pad_frac=0.0,
                ignore_mask=ignored,
            )
            confirmation_reports[camera_name] = confirmation_report
            reasons = []
            if not confirmation_report.seen:
                reasons.append("board_context_not_detected")
            if confirmation_report.artificial_bottom_contact:
                reasons.append("board_contacts_gripper_mask")
            if confirmation_report.gripper_overlap_px > 0:
                reasons.append(
                    "board_gripper_overlap="
                    f"{confirmation_report.gripper_overlap_px}px"
                )
            if reasons:
                confirmation_rejections[camera_name] = reasons
        try:
            confirmation_base_T_tcp = {
                name: self._base_transform_at(
                    self.config.gripper_frame,
                    int(confirmation.frames[name]["stamp_ns"]),
                    timeout_sec,
                )
                for name in expected
            }
            confirmation_base_T_cam = {
                name: self._base_transform_at(
                    self.config.camera_frames[name],
                    int(confirmation.frames[name]["stamp_ns"]),
                    timeout_sec,
                )
                for name in expected
            }
        except Exception as error:
            self._stage2_not_done(
                result,
                "timestamp-bound confirmation camera/TCP TF unavailable: "
                f"{error}",
            )
            return
        confirmation_estimates = {}
        for camera_name in expected:
            image = confirmation.frames[camera_name]["image"]
            ignored = self.gripper_masks.ignored_pixels(camera_name, image.shape)
            observation, reason = self._stage2_landmarks(
                image, confirmation_reports[camera_name], ignored
            )
            if observation is None:
                confirmation_rejections.setdefault(camera_name, []).append(reason)
                continue
            estimate, reason = estimate_board_pose(
                observation[0],
                observation[1],
                camera_models[camera_name],
                confirmation_base_T_cam[camera_name],
            )
            if estimate is None:
                confirmation_rejections.setdefault(camera_name, []).append(
                    f"PnP={reason}"
                )
                continue
            confirmation_estimates[camera_name] = estimate

        confirmation_consistent, confirmation_consistency_reason = (
            board_pose_set_is_consistent(
                confirmation_estimates,
                board_pose,
                expected,
            )
        )
        if not confirmation_consistent:
            self._stage2_not_done(
                result,
                "confirmation board poses are incomplete or inconsistent: "
                f"{confirmation_consistency_reason}",
            )
            return

        confirmation_projection = {}
        for name, estimate in confirmation_estimates.items():
            confirmation_projection[name] = verify_survey_view(
                estimate,
                confirmation_base_T_tcp[name],
                {name: tcp_T_cam[name]},
                {name: camera_models[name]},
                {name: grippers[name]},
                {name: int(confirmation.frames[name]["stamp_ns"])},
                max_skew_ns=50_000_000,
            )
        missing_confirmation = set(expected) - set(confirmation_projection)
        failed_confirmation = {
            name: projection
            for name, projection in confirmation_projection.items()
            if not projection.passed
        }
        if confirmation_rejections or missing_confirmation or failed_confirmation:
            details = "; ".join(
                f"{name}={','.join(reasons)}"
                for name, reasons in confirmation_rejections.items()
            )
            self._stage2_not_done(
                result,
                "second settled triplet failed image/projection verification"
                + (
                    f": missing={sorted(missing_confirmation)}; {details}"
                    if details or missing_confirmation
                    else ""
                ),
            )
            return

        result.done = True
        result.success = True
        result.component_coverage_ready = True
        result.steer_camera = source_camera
        result.last_action = "sfp_geometric_stage2_verified"
        result.elapsed_seconds = max(0.0, time.monotonic() - started_at)
        result.message = (
            "geometric Stage 2 verified the complete loose-SFP envelope "
            "inside all three fresh calibrated camera views with conservative "
            f"gripper clearance after {result.moves_executed} total moves"
        )

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
        if not 10.0 <= values["search_timeout_sec"] <= 60.0:
            raise ValueError("search_timeout_seconds must be in [10, 60]")
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
