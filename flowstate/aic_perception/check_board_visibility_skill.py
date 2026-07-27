#!/usr/bin/env python3
"""Flowstate skill for deterministic insignia acquisition and board survey.

Stage 1 observes first and, only when needed, executes one offline-swept,
force-guarded joint path to a known observation posture. Stage 2 is
perception-only and computes the target-specific survey pose from the complete
purple insignia. Inputs remain restricted to documented wrist cameras,
measured robot state, wrist force, and robot-mounted TF; object, simulation,
and scoring frames are never requested.
"""

from __future__ import annotations

from concurrent import futures
from dataclasses import replace
import math
import signal
import threading
import time
import traceback

from absl import app, flags, logging
import grpc
import numpy as np

from intrinsic.skills.python import skill_interface


# Tallest structure standing on the task board: the NIC card tips / SFP cage
# entrances, at board Z 0.1793 (measured from the workcell model).  Survey poses
# are checked against this rather than against the board plane -- a pose that
# clears the plate can still be inside the cards.
BOARD_TALLEST_COMPONENT_Z = 0.1793
# Margin the tool keeps above that.  Move Robot does its own collision checking,
# but a pose it has to refuse fails the task outright, so poses that cut it fine
# are never published in the first place.
TOOL_COMPONENT_CLEARANCE_M = 0.06

# SC bore geometry (``sc_sector_corners``): the mouth is 7.6 x 22.4 mm over a
# 15.64 mm recess, and the camera is displaced along board X, the normal of the
# adapter's board-Y long face.
SC_BORE_DEPTH_M = 0.01564
SC_BORE_HALF_WIDTH_Y_M = 0.0112
# How far the ray may walk across the *narrow* axis before the pose is refused.
#
# This number is the acceptance criterion, not the adapter, and it is what caps
# the achievable depth cue.  At the mouth *half* width (3.8 mm) the ray must
# still reach the back-plane **centre**, which puts a hard 13.66 deg ceiling on
# the long-face angle -- and the depth cue is f*depth*tan(theta)/dist, so that
# ceiling capped the cue at 3.3-4.5 px.  That is most of why the SC view was
# fragile: there was no headroom left to angle into.
#
# At the *full* mouth width the criterion instead becomes "a displaced dark
# strip is still visible", which is what the estimator actually keys on.  The
# back plane is then partly occluded rather than centred: at 18 deg the strip
# is 7.6 - 15.64*tan(18) = 2.5 mm wide by 22.4 mm long, and its displacement --
# the cue itself -- more than doubles.  Measured worst mouth over the whole
# legal rail, 24 board placements:
#
#   band     criterion            worst cue   dark strip
#   10-13    back centre           3.34 px      4.4 mm
#   14-18    displaced strip       7.13 px      3.1 mm
#   16-20    displaced strip       7.99 px      2.5 mm
#
# Do not read this as a relaxed *safety* gate: framing, gripper clearance,
# live-seeded IK and arm-in-view are all unchanged and still hard.
SC_BORE_X_TOLERANCE_M = 0.0076

# The deployed aic_controller rejects `/aic_controller/change_target_mode`
# around in-flight executions, so the first request routinely times out and is
# retried.  That happens before the profile starts and must not be charged to
# the motion budget.
JOINT_MODE_SWITCH_ALLOWANCE_SEC = 3.0
# Measured settling after the profile ends.
JOINT_SETTLE_ALLOWANCE_SEC = 2.0


def _contact_force_n(force_xyz) -> float:
    """Wrist force the untared static load cannot explain, for diagnostics.

    Logging the raw magnitude alone is what made the force failures so hard to
    read: 25.72 N looks alarming and is in fact a free-space reading at one
    wrist orientation.  Always log this alongside it.
    """
    from aic_perception.robot_motion import contact_force_n

    return contact_force_n(force_xyz)


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
        from rclpy.signals import SignalHandlerOptions
        from tf2_ros.buffer import Buffer
        from tf2_ros.transform_listener import TransformListener

        from aic_perception.config import PerceptionConfig
        from aic_perception.camera_rig import CameraRig
        from aic_perception.gripper_masks import GripperMaskBank
        from aic_perception.robot_motion import RobotMotion

        if not rclpy.ok():
            # The gRPC service owns SIGINT/SIGTERM.  rclpy's default signal
            # handler shuts down the context underneath the executor thread
            # while the gRPC main thread is still waiting, producing an
            # RCLError and forcing Kubernetes to SIGKILL the container.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
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

    def close(self) -> None:
        """Stop ROS callbacks before invalidating their rclpy context."""
        import rclpy

        try:
            self._executor.shutdown(timeout_sec=2.0)
        finally:
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
            try:
                self._executor.remove_node(self.node)
            except Exception:
                pass
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

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
        """Expose the insignia from one deterministic observation posture.

        Stage 2 remains the only authority for declaring a usable insignia and
        for producing the downstream survey target.  Stage 1 observes first,
        executes one offline-swept joint path only when necessary, observes
        once more, then either hands the fresh triplet to Stage 2 or returns a
        normal not-done result while holding the known-safe observation pose.
        """

        from aic_perception.arm_ik import UR5eArm
        from aic_perception.board_stage2 import CameraModel
        from aic_perception.board_visibility import analyze_board, view_quality
        from aic_perception.purple_insignia import analyze_purple
        from aic_perception.board_seek import (
            SEEK_HARD_MOVE_CEILING,
            SEEK_STALL_MOVES,
            seek_progress_score,
            select_work_target,
        )

        min_contrast = int(params.min_contrast or 30)
        margin_px = int(params.margin_px or 15)
        ignore_bottom = float(params.ignore_bottom_frac)
        step_m = float(params.step_m or 0.04)
        backoff_step_m = float(params.backoff_step_m or step_m)
        timeout_sec = float(params.timeout_seconds or 10)
        min_area_frac = float(params.min_area_frac or 0.005)
        max_force_n = float(params.max_force_n or 18.0)
        max_speed_mps = float(params.max_speed_mps or 0.05)
        publish_hz = float(params.publish_hz or 20.0)
        settle_tolerance_m = float(params.settle_tolerance_m or 0.008)
        move_timeout_sec = float(params.move_timeout_seconds or 6.0)
        max_travel_m = float(params.max_travel_m or 0.80)
        force_delta_n = float(params.force_delta_n or 5.0)
        search_timeout_sec = float(params.search_timeout_seconds or 60.0)
        max_displacement_m = float(params.max_displacement_m or 0.50)
        angular_step_rad = float(params.angular_step_rad or 0.10)
        max_angular_displacement_rad = float(
            params.max_angular_displacement_rad or 1.60
        )
        max_angular_travel_rad = float(
            params.max_angular_travel_rad or 2.20
        )
        context_margin_frac = float(params.context_margin_frac or 0.05)
        min_detail_area_frac = float(params.min_detail_area_frac or 0.06)
        min_rectangularity = float(params.min_rectangularity or 0.50)
        stable_frames = int(params.stable_frames or 2)
        max_angular_speed_rps = float(
            params.max_angular_speed_rps or 0.30
        )
        settle_orientation_tolerance_rad = float(
            params.settle_orientation_tolerance_rad or 0.05
        )
        survey_target = int(getattr(params, "survey_target", 0))
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
            settle_orientation_tolerance_rad=(
                settle_orientation_tolerance_rad
            ),
        )

        started_at = time.monotonic()
        baseline_force_xyz = None

        def motion_cancelled() -> bool:
            return bool(cancelled())

        def handoff_to_stage2(snapshot, reports) -> None:
            self._run_sfp_geometric_stage2(
                snapshot=snapshot,
                reports=reports,
                result=result,
                survey_target=survey_target,
                timeout_sec=timeout_sec,
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
                motion_cancelled=motion_cancelled,
            )

        def observe(label: str):
            nonlocal baseline_force_xyz
            if cancelled():
                raise skill_interface.SkillCancelledError(
                    f"board acquisition cancelled before {label}"
                )
            snapshot = self.camera_rig.grab(timeout_sec=timeout_sec)
            if snapshot is None or not snapshot.frames:
                result.success = False
                result.done = False
                result.target_valid = False
                result.message = (
                    "no fresh wrist-camera frame received from approved topics"
                )
                return None, None
            if snapshot.force_xyz is not None and baseline_force_xyz is None:
                baseline_force_xyz = snapshot.force_xyz

            result.elapsed_seconds = max(
                0.0, time.monotonic() - started_at
            )
            result.target_valid = False
            result.target_frame = ""
            result.component_coverage_ready = False
            result.dx = result.dy = result.dz = 0.0
            result.backoff = False
            result.num_cameras = len(snapshot.frames)
            result.force_n = float(snapshot.force_norm or 0.0)
            reports = {}
            purple_reports = {}
            for camera_name, frame in snapshot.frames.items():
                ignored = self.gripper_masks.ignored_pixels(
                    camera_name, frame["image"].shape
                )
                reports[camera_name] = analyze_board(
                    frame["image"],
                    margin_px=margin_px,
                    min_area_frac=min_area_frac,
                    ignore_bottom_frac=ignore_bottom,
                    min_contrast=float(min_contrast),
                    min_rectangularity=min_rectangularity,
                    min_detail_area_frac=min_detail_area_frac,
                    context_pad_frac=context_margin_frac,
                    ignore_mask=ignored,
                )
                masked = frame["image"].copy()
                masked[ignored] = 0
                purple_reports[camera_name] = analyze_purple(
                    masked, margin_px=margin_px
                )
                board_report = reports[camera_name]
                purple_report = purple_reports[camera_name]
                logging.info(
                    "deterministic Stage 1 %s %s: board_seen=%s "
                    "edges=%s area=%.3f long_ratio=%.2f "
                    "purple_seen=%s purple_full=%s purple_edges=%s "
                    "purple_area=%.4f stamp=%s",
                    label,
                    camera_name,
                    board_report.seen,
                    sorted(board_report.edges),
                    board_report.area_frac,
                    board_report.long_axis_ratio,
                    purple_report.seen,
                    purple_report.full,
                    sorted(purple_report.edges),
                    purple_report.area_frac,
                    frame["stamp_ns"],
                )

            result.seen = any(report.seen for report in reports.values())
            if reports:
                camera_name, report = max(
                    reports.items(),
                    key=lambda item: float(view_quality(item[1])),
                )
                result.steer_camera = camera_name
                result.edges = ",".join(sorted(report.edges))
                result.area_frac = float(report.area_frac)
                result.rectangularity = float(report.rectangularity)
                result.view_quality = float(view_quality(report))
            return snapshot, reports

        snapshot, reports = observe("initial")
        if snapshot is None:
            return
        if self._force_exceeded(
            snapshot.force_xyz,
            baseline_force_xyz,
            max_force_n,
            force_delta_n,
        ):
            result.success = False
            result.done = False
            result.force_abort = True
            result.message = (
                f"wrist force guard active: raw={result.force_n:.2f}N "
                f"unexplained={_contact_force_n(snapshot.force_xyz):.2f}N "
                f"(free-space envelope 12-27N); "
                "deterministic acquisition refused"
            )
            return
        if self._stage2_has_complete_landmark(snapshot, reports):
            result.last_action = "initial_insignia_handoff"
            logging.info(
                "Stage-2-valid insignia already exposed; no Stage-1 motion"
            )
            handoff_to_stage2(snapshot, reports)
            return

        expected = tuple(sorted(self.config.camera_frames))
        missing_frames = sorted(set(expected) - set(snapshot.frames))
        missing_calibration = sorted(
            set(expected) - set(snapshot.calibrations)
        )
        if missing_frames or missing_calibration:
            result.success = False
            result.done = False
            result.message = (
                "deterministic acquisition requires all approved cameras; "
                f"missing_frames={missing_frames} "
                f"missing_calibration={missing_calibration}"
            )
            return

        fresh_force = snapshot.force_xyz
        if fresh_force is None:
            fresh_force = self.camera_rig.wait_for_force_xyz(
                timeout_sec=timeout_sec, max_age_sec=0.5
            )
        if fresh_force is None:
            result.success = False
            result.done = False
            result.message = (
                "no fresh wrist-force sample for deterministic acquisition"
            )
            return
        if baseline_force_xyz is None:
            baseline_force_xyz = fresh_force

        purple_reports = self._scan_purple(snapshot)


        # ------------------------------------------------------------------
        # Stage 1: image-plane seek (ported from move_to_board_skill v3).
        #
        # Small Cartesian translations at fixed orientation, steering on
        # whichever camera most needs work.  No joint target mode, no phase
        # sequencing, and no use of the board's orientation -- which is
        # degenerate (`long_ratio=1.00`) in exactly the clipped views where
        # Stage 1 is needed.  Purple takes over from the board mask the moment
        # any camera sees it.
        # ------------------------------------------------------------------
        baseline_force_xyz = snapshot.force_xyz or baseline_force_xyz
        mode, steer_camera, steer_report = select_work_target(
            reports, purple_reports
        )
        if steer_camera is None or steer_report is None:
            self._stage2_not_done(
                result,
                "no board or purple insignia detected after gripper masking",
            )
            result.last_action = "seek_no_target"
            return

        result.last_action = "board_seek"
        # No fixed hop budget: a corner start legitimately needs far more moves
        # than a near-framed one.  The search runs while it is still improving
        # and stops when it stalls.  The ceiling only guarantees termination --
        # Stage 1 has no aggregate wall clock, so an unbounded loop would hang
        # the skill.
        best_score = seek_progress_score(reports, purple_reports)
        stalled_moves = 0
        step = 0
        while step < SEEK_HARD_MOVE_CEILING:
            step += 1
            if cancelled():
                raise skill_interface.SkillCancelledError(
                    "board acquisition cancelled during seek"
                )
            logging.info(
                "seek step %d mode=%s on %s edges=%s area=%.3f "
                "center=(%+.3f,%+.3f) best_score=%.3f",
                step,
                mode,
                steer_camera,
                ",".join(sorted(steer_report.edges)) or "none",
                steer_report.area_frac,
                float(steer_report.center_error[0]),
                float(steer_report.center_error[1]),
                best_score,
            )
            outcome, skip_reason = self._seek_step(
                steer_camera,
                steer_report,
                timeout_sec=timeout_sec,
                baseline_force_xyz=baseline_force_xyz,
                max_force_n=max_force_n,
                force_delta_n=force_delta_n,
                publish_hz=publish_hz,
                cancelled=motion_cancelled,
                target_label=mode,
            )
            if skip_reason is not None:
                self._stage2_not_done(
                    result,
                    f"{mode} seen on {steer_camera} after "
                    f"{result.moves_executed} moves but no centering signal "
                    f"remains: {skip_reason}",
                )
                result.last_action = "seek_no_signal"
                return
            if outcome.cancelled:
                raise skill_interface.SkillCancelledError(outcome.message)
            if not outcome.success:
                result.success = False
                result.done = False
                result.target_valid = False
                result.force_abort = outcome.force_abort
                result.message = (
                    f"board seek move {result.moves_executed + 1} failed: "
                    f"{outcome.message}"
                )
                result.last_action = "seek_move_failed"
                return

            result.moves_executed += 1
            result.moved = True
            result.travel_m += float(outcome.distance_m)

            snapshot, reports = observe(f"seek_{step}")
            if snapshot is None:
                return
            purple_reports = self._scan_purple(snapshot)
            if self._stage2_has_complete_landmark(snapshot, reports):
                result.last_action = "seek_insignia_handoff"
                logging.info(
                    "board seek exposed a Stage-2-valid insignia after "
                    "%d move(s)",
                    result.moves_executed,
                )
                handoff_to_stage2(snapshot, reports)
                return

            score = seek_progress_score(reports, purple_reports)
            if score > best_score + 1e-9:
                best_score = score
                stalled_moves = 0
            else:
                stalled_moves += 1
                logging.info(
                    "seek made no progress (%.3f <= best %.3f), %d/%d "
                    "consecutive",
                    score,
                    best_score,
                    stalled_moves,
                    SEEK_STALL_MOVES,
                )
                if stalled_moves >= SEEK_STALL_MOVES:
                    result.last_action = "seek_stalled"
                    self._stage2_not_done(
                        result,
                        f"board seek stalled after {result.moves_executed} "
                        f"move(s): {SEEK_STALL_MOVES} consecutive moves "
                        "produced no progress toward exposing the insignia",
                    )
                    return

            mode, steer_camera, steer_report = select_work_target(
                reports, purple_reports, preferred=steer_camera
            )
            if steer_camera is None or steer_report is None:
                self._stage2_not_done(
                    result,
                    "lost the board and the insignia after seek move "
                    f"{result.moves_executed}",
                )
                result.last_action = "seek_target_lost"
                return

        result.last_action = "seek_ceiling_reached"
        self._stage2_not_done(
            result,
            f"board seek hit its {SEEK_HARD_MOVE_CEILING}-move termination "
            "backstop without exposing the insignia; the stall detector "
            "should have ended this first",
        )

    def _scan_purple(self, snapshot) -> dict:
        """Purple-insignia report per fresh camera, after gripper masking."""
        from aic_perception.board_seek import BOARD_MARGIN_PX
        from aic_perception.purple_insignia import analyze_purple

        reports = {}
        for name, frame in snapshot.frames.items():
            masked = self.gripper_masks.apply(name, frame["image"])
            reports[name] = analyze_purple(masked, margin_px=BOARD_MARGIN_PX)
        return reports

    def _seek_step(
        self,
        camera: str,
        report,
        *,
        timeout_sec: float,
        baseline_force_xyz,
        max_force_n: float,
        force_delta_n: float,
        publish_hz: float,
        cancelled,
        target_label: str = "target",
    ):
        """One bounded image-plane translation toward centring ``report``.

        Cartesian and orientation-preserving on purpose.  Holding orientation
        fixed is what removes the J1/J6 coupling that broke the phase machine
        at the levelled pose, and staying in Cartesian mode avoids the joint
        target-mode negotiation that the deployed controller drops
        ("controller left joint target mode") mid-segment.
        """
        from aic_perception.board_seek import (
            CENTER_STEP_M,
            image_plane_direction,
            MAX_SPEED_MPS,
            MOVE_TIMEOUT_SEC,
            SETTLE_TOLERANCE_M,
        )
        from aic_perception.board_visibility import world_delta
        from aic_perception.robot_motion import normalize_quaternion

        direction = image_plane_direction(report)
        if direction is None:
            return None, (
                f"{target_label} is already near the image centre on {camera}"
            )

        position, orientation = self._gripper_pose(timeout_sec)
        image_right, image_down, back_away = self._camera_axes_in_base(
            camera, timeout_sec
        )
        delta = world_delta(
            np.asarray(direction, dtype=float),
            backoff=False,
            step_m=CENTER_STEP_M,
            base_image_right=image_right,
            base_image_down=image_down,
            base_backoff=back_away,
        )
        target = tuple(
            float(value)
            for value in np.asarray(position, dtype=float) + delta
        )
        logging.info(
            "seek %s on %s: image_dir=(%+.2f,%+.2f) step=%.3fm "
            "delta=(%+.4f,%+.4f,%+.4f)",
            target_label,
            camera,
            direction[0],
            direction[1],
            CENTER_STEP_M,
            delta[0],
            delta[1],
            delta[2],
        )
        outcome = self.robot_motion.move_smooth(
            target,
            target_orientation=normalize_quaternion(orientation),
            max_speed_mps=MAX_SPEED_MPS,
            publish_hz=publish_hz,
            settle_tolerance_m=SETTLE_TOLERANCE_M,
            timeout_sec=MOVE_TIMEOUT_SEC,
            baseline_force_xyz=baseline_force_xyz,
            max_force_n=max_force_n,
            force_delta_n=force_delta_n,
            cancelled=cancelled,
        )
        return outcome, None

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
    def _uses_geometric_survey(survey_target: int) -> bool:
        """Whether this invocation uses the insignia-driven sector survey.

        All three deployed target modes now take the geometric path; each frames
        its own board sector.  The legacy adaptive board search remains only as
        the Stage-1 fallback that moves until the insignia is exposed.
        """
        # UNSPECIFIED is retained as the historical pre-enum SFP default.
        return int(survey_target) in (0, 1, 2, 3)

    @staticmethod
    def _coverage_targets_for_target(survey_target: int) -> tuple:
        """Board-frame coverage ladder framed by this survey target.

        ``search_survey_pose`` tries these in order and commits to the first
        that yields a pose, so a ladder asks for the most coverage a given
        board placement can actually afford instead of failing outright.

        **SFP (0/1) frames the module strip centred, not one rail.**  The
        superseded ``sfp_sector_corners`` covered the +Y rail alone
        (Y 0.0 .. 0.225), which put the aimed centre 112.5 mm off the middle of
        the staged modules.  The five staged modules run Y -0.15625 .. +0.15625
        across *both* rails, so all of the search's framing slack was banked on
        the +Y side and the outer -Y module fell out of frame.  That is the
        observed 4-of-5 hardware failure, and the offline sweep reproduces it
        in every case: at identical search settings the old box clips a module
        in 96 of its 96 found poses, 35 of them holding only four of the six
        seats.  The centred box frames every module in all 92 of its poses.

        The replacement box is the *same size*, straddling Y=0.  Enlarging it
        was measured and rejected: a wider box pushes the selected standoff
        from 0.64 m to 0.85-0.90 m and shrinks every module in the image, and
        past ~0.85 m the arm's own links enter a wrist camera at every roll, so
        full-strip containment is not reachable at all.

        NIC and SC keep their own sectors: their geometry is decided by bore
        aperture, not by strip length, and both are already validated.
        """
        from aic_perception.board_stage2 import (
            nic_sector_corners,
            sc_sector_corners,
            sfp_module_strip_corners,
        )

        target = int(survey_target)
        if target == 2:  # NIC_SFP_DESTINATION
            return (nic_sector_corners(),)
        if target == 3:  # SC_DESTINATION_PORT
            return (sc_sector_corners(),)
        # 0 UNSPECIFIED / 1 STAGED_SFP_MODULE
        return (sfp_module_strip_corners(),)

    @staticmethod
    def _arm_clear_of_own_cameras(base_T_tcp, joints, arm, tcp_T_cam, cameras):
        """True when no arm link stands in any wrist camera's view.

        A survey pose can be perfectly top-down, collision-free and fully
        framed, and still be useless because the robot's own upper arm or
        forearm lies across the picture -- which is exactly what a field run
        produced (obliquity 0.0 deg, yet the view was blocked by the arm).  The
        gripper keep-out cannot catch this: it is a fixed image-space silhouette,
        correct only for what is rigidly attached to wrist_3, while these links
        move independently of the wrist.

        Approximate by construction: the configuration checked is the branch
        nearest the current joints, and Move Robot may choose another.  That is
        still far better than assuming the arm is never in frame.
        """
        from aic_perception.board_stage2 import project_points

        for start, end, radius in arm.link_segments(joints):
            samples = np.array(
                [start + (end - start) * t for t in np.linspace(0.0, 1.0, 25)]
            )
            for name, camera in cameras.items():
                camera_pose = base_T_tcp.compose(tcp_T_cam[name])
                local = camera_pose.inverse().apply(samples)
                pixels, in_front = project_points(local, camera)
                for pixel, ahead, point in zip(pixels, in_front, local):
                    if not ahead or not np.all(np.isfinite(pixel)):
                        continue
                    # Grow the segment by its own radius at that depth, so a
                    # tube grazing the frame edge still counts as intruding.
                    margin = radius * float(camera.K[0, 0]) / max(
                        float(point[2]), 1e-6
                    )
                    if (
                        -margin <= pixel[0] <= camera.width + margin
                        and -margin <= pixel[1] <= camera.height + margin
                    ):
                        return False
        return True

    @staticmethod
    def _survey_view_settings(survey_target: int) -> dict:
        """Per-sector view geometry passed to ``search_survey_pose``.

        **NIC (2)** is decided entirely by the SFP port bores on the card tips.
        Each is a 16 x 12 mm aperture at the top of a 45.8 mm recess whose axis
        points straight *up* -- 0.7 deg off the board normal, measured from the
        workcell model -- so a port only shows the black depth the IVM keys on to
        a ray within ``atan(6/45.8) = 7.5 deg`` of that axis.  Two consequences,
        both the opposite of what this sector used to ask for:

        * **Look straight down, never tilted.** A cross-rail tilt reads the cages
          edge-on and the near wall occludes the bore, so the ports render as flat
          grey rectangles and the IVM finds nothing.  The old committed 12-22 deg
          bore band did exactly that: at board yaws where it was reachable it
          resolved **0 of 10 ports**.  Hence no tilt band and a 2 deg obliquity
          cap, which also stops the ranking trading the overhead view away.
        * **Stand as far off as the arm can reach.** The ten ports span 160 mm, so
          the outermost sits ``atan(0.081/d)`` off the optical axis -- it needs
          ``d >= 0.62 m`` above the port plane to stay inside the 7.5 deg cone.
          This is why ``prefer_far_standoff`` is right here for a real reason, not
          for "undistorted" framing.

        All three cameras must frame the sector.  That needs a smaller gripper
        keep-out margin than the default 40 px -- at 40 px no all-camera pose is
        reachable at board yaws near 0.  25 px still leaves 57 px between the
        cards and true gripper pixels, because ``GripperExclusion`` already
        dilates the silhouette by 32 px before this margin applies.

        **Reorientation budget.** The reachability gate also keeps the wrist
        cameras clear of the forearm (``UR5eArm.flange_T_probes`` -- a purely
        kinematic gate published a pose the workcell planner then refused
        outright as a self-collision). At some board yaws *every* candidate
        within the default 45 deg reorientation cap and 7-roll sample sits too
        close to the forearm; only a wider roll sweep finds the camera-cluster
        orientation that swings clear. NIC therefore samples 24 rolls (15 deg
        steps) instead of the default 7, and allows up to 90 deg of
        reorientation instead of 45 -- Move Robot's own Cartesian planner, not
        this geometric search, is what actually executes the move and picks the
        joint path, so the wider cap only widens which *poses* are offered, not
        how the arm gets there.

        **SC (3)** also opens along the board normal.  Hardware showed the
        previous rail-derived offset approached the wrong side of each adapter:
        one camera saw the mouths from their short end and failed.  The mouth's
        transformed long face runs along board Y (22.4 mm); to stand off that
        long face, the camera is displaced along its board-X normal, explicitly
        supplied rather than inferred from the cluster bounding box.

        The long-face direction is right and hardware-proven; what was wrong
        was how far it was allowed to go.  The angle used to stop at 13 deg
        because the bore gate demanded the ray still reach the back-plane
        **centre**, a hard 13.66 deg ceiling on the narrow 7.6 mm axis (see
        ``SC_BORE_X_TOLERANCE_M``).  Since the cue is
        ``f*depth*tan(theta)/dist``, that ceiling *was* the flakiness: the
        displacement never exceeded 3.3-4.5 px, and an adapter that slid along
        its rail lost even that, because the search aims at the sector centroid
        and a mouth offset ``delta`` is seen at ``atan(tan(theta) - delta/d)``
        -- about 10 deg of swing over the 115 mm of legal travel, most of the
        whole cone.

        The gate now accepts a *displaced dark strip* instead of a centred back
        plane, and the band moves out to 16-20 deg.  Measured worst mouth over
        the whole legal rail, 24 board placements: 3.34 px -> 7.99 px.  The
        144-case sweep reports 7.36-8.55 px where it used to report 3.34-4.45,
        and the bore margin's worst case *improves* from 0.013 to 0.031.

        Standoff follows the angle down to a 0.55-0.62 m ladder -- closest
        feasible wins, and on this axis closer is also deeper.  Do not go below
        0.55 m: 0.45 m is the pose that put the tool on top of the ports in
        both side cameras.

        The along-long-face component still stays below 2 deg.  Tilting there
        was measured and rejected: the wide axis has a 35.61 deg cone and would
        carry more cue still, but it is not the face the detector was validated
        against.  All three cameras must frame the entire sector and remain
        gripper-clear, and for each mouth at least two cameras must see through
        the physical bore and retain at least 3.0 px of displacement.

        **SFP (0/1)** pick modules read fine from the standard close all-camera
        near-overhead view.
        """
        target = int(survey_target)
        if target == 2:  # NIC_SFP_DESTINATION
            return dict(
                cross_rail_tilt_band_rad=None,
                cross_rail_sign=0.0,
                require_all_cameras_frame=True,
                prefer_far_standoff=True,
                max_obliquity_rad=math.radians(2.0),
                min_required_clearance_px=25.0,
                max_angular_motion_rad=math.radians(90.0),
                yaws_rad=tuple(
                    math.radians(deg) for deg in range(-180, 180, 15)
                ),
            )
        if target == 3:  # SC_DESTINATION_PORT
            return dict(
                # Approach the adapter from its long face.  The face runs along
                # board Y, so its outward in-plane normal is board X.  Do not
                # infer this from the SC cluster extent: the three/two-port rail
                # is longer along X and caused the exact axis swap seen in the
                # failed camera image.
                cross_rail_tilt_band_rad=(
                    math.radians(16.0),
                    math.radians(20.0),
                ),
                directional_tilt_axis_board=(1.0, 0.0, 0.0),
                max_along_rail_tilt_rad=math.radians(2.0),
                cross_rail_sign=0.0,
                # All three cameras must frame the sector AND stay gripper-clear.
                # Dropping this to chase a closer, higher-resolution view was
                # tried and reverted: with only the reference camera checked, the
                # tool sat *on top of* the ports in both side cameras (gripper
                # clearance -13 to -32 px at every board yaw) while the centre
                # camera reported a healthy +58 px, and the 0.45 m pose put the
                # TCP at base z 0.24 m, which the arm could only reach through a
                # contorted configuration.  "All five ports project inside the
                # image" is NOT the same as "all five are unoccluded" -- that
                # was the flawed check behind the regression.
                require_all_cameras_frame=True,
                prefer_far_standoff=False,
                min_required_clearance_px=25.0,
                # Joint motion is measured from the live six-joint state.  A
                # 90-degree Cartesian reorientation cap suppresses
                # camera-clear wrist rolls after some Stage-1 exits, so SC
                # searches the full finite roll family and lets the live-seeded
                # IK motion gate reject excessive relative travel.
                max_angular_motion_rad=math.pi,
                yaws_rad=tuple(
                    math.radians(deg) for deg in range(-180, 180, 15)
                ),
                # Standoff is a short ladder, not a pin.  ``prefer_far_standoff``
                # is False so the closest feasible rung wins, and on this axis
                # closer is also deeper: the cue is f*depth*tan(theta)/dist.
                # It stays a ladder rather than the full grid because the rungs
                # below 0.55 m never survive all-camera framing and gripper
                # clearance -- the 0.45 m pose is the one that put the tool on
                # top of the ports in both side cameras.
                standoffs_m=(0.55, 0.58, 0.60, 0.62),
            )
        # SFP (0/1/anything else): the close, all-camera, near-overhead view.
        # The keep-out margin and reorientation budget are NIC's already-proven
        # values, adopted here for availability, not for framing.  With the
        # centred coverage box the 144-case sweep finds a pose in 58 cases at
        # the 40 px / 45 deg defaults and in 92 at 25 px / 90 deg -- and all 92
        # still frame every module, so the extra 34 are genuine gains rather
        # than weaker views.  25 px is measured against a silhouette
        # ``GripperExclusion`` has already dilated by 32 px.
        return dict(
            cross_rail_tilt_band_rad=None,
            cross_rail_sign=0.0,
            require_all_cameras_frame=True,
            prefer_far_standoff=False,
            min_required_clearance_px=25.0,
            max_angular_motion_rad=math.radians(90.0),
        )

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
        """Extract the Stage-2 insignia seed from the handoff image.

        Returns ``((insignia_quad, insignia_centroid), "ok")`` or
        ``(None, reason)``.  Pose is driven by the large asymmetric purple
        insignia rather than the board outline: the insignia stays fully in frame
        at survey standoffs where the plate clips, so this no longer requires a
        recoverable outline or a "full" Stage-1 report.  The only seed
        requirements are a credible insignia that is fully inside the image and
        unobstructed by the gripper.  ``report`` is retained for signature
        compatibility and is unused.
        """
        import cv2

        from aic_perception.board_visibility import detect_insignia_polygon

        detected = detect_insignia_polygon(image)
        if detected is None:
            return None, "purple insignia was not detected"
        quad, centroid = detected
        height, width = image.shape[:2]
        ignored = np.asarray(ignored_pixels, dtype=bool)
        if ignored.shape != (height, width):
            return None, "gripper mask dimensions do not match the image"
        # The four detected corners must sit inside the frame for a stable PnP.
        quad_margin = min(
            float(quad[:, 0].min()),
            float(quad[:, 1].min()),
            float(width - 1 - quad[:, 0].max()),
            float(height - 1 - quad[:, 1].max()),
        )
        if quad_margin < 3.0:
            return None, "insignia touches the physical image boundary"
        # A degenerate/near-collinear detection cannot yield a pose.
        if abs(float(cv2.contourArea(quad.astype(np.float32)))) < 0.001 * (
            height * width
        ):
            return None, "insignia quad is too small or degenerate for PnP"
        # Refuse an occluded insignia: rasterise its convex region and require it
        # clear of the dilated gripper keep-out.
        uncertainty = cv2.dilate(
            ignored.astype(np.uint8), np.ones((9, 9), np.uint8)
        ).astype(bool)
        insignia_fill = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(
            insignia_fill,
            cv2.convexHull(quad.round().astype(np.int32)),
            1,
        )
        if np.any(insignia_fill.astype(bool) & uncertainty):
            return None, "insignia intersects the gripper uncertainty mask"
        return (quad, np.asarray(centroid, dtype=float)), "ok"

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
        survey_target: int,
        timeout_sec: float,
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
        """Estimate and publish one board-relative survey pose.

        Perception-only for every target mode: it estimates the board pose from
        the insignia, computes one board-relative survey pose that frames this
        target's board sector (SFP +Y rail / NIC cards / SC ports) in all three
        cameras, and publishes it as ``result.survey_pose`` for a downstream Move
        Robot skill.  It commands no motion, so it needs no wrist force and does
        no controller work.
        """
        from aic_perception.board_stage2 import (
            CameraModel,
            GripperExclusion,
            estimate_board_pose_from_insignia,
            quaternion_from_matrix,
            rectangular_bore_depth_cue_px,
            rectangular_bore_visibility_margin,
            sc_bore_sample_points,
            search_survey_pose,
        )
        from aic_perception.arm_ik import JOINT_LIMITS, UR5eArm

        expected = tuple(sorted(self.config.camera_frames))
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
                "no calibrated camera contains an unobstructed, fully framed "
                f"insignia ({reasons})",
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
            (insignia_quad, insignia_centroid), _ = observations[camera_name]
            estimate, pose_reason = estimate_board_pose_from_insignia(
                insignia_quad,
                insignia_centroid,
                camera_models[camera_name],
                base_T_cam[camera_name],
            )
            if estimate is None:
                pose_failures[camera_name] = pose_reason
            else:
                pose_estimates.append(estimate)
        if not pose_estimates:
            self._stage2_not_done(
                result,
                "insignia pose rejected in every insignia camera: "
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

        # Real reachability gate.  A UR5e IK check (the arm's own kinematics --
        # no task-board TF) replaces the base-origin sphere so the search commits
        # to poses the arm can actually achieve -- including reaching to the far,
        # bore-facing side when the ports face away -- and never publishes one
        # Move Robot cannot solve.  The flange->TCP offset is self-calibrated
        # from the live, static (joint-state, TCP) sample; if the recovered
        # offset is implausible (an unexpected base-frame convention) we fall
        # back to the sphere rather than trusting a bad model.
        joint_motion_fn = None
        joint_motion_preference_fn = None
        ik_arm = None
        ik_seed = None
        preferred_j6_target = None
        # With live-relative winding, SC's stronger 10-13 degree view has a
        # camera-clear branch in every swept scenario under a 185-degree
        # worst-joint cap. 182 degrees loses cases, so retain a small measured
        # margin around the required half-turn. Keep the previous 225-degree
        # budget for the other sectors, whose reach policies were not retuned.
        joint_motion_limit_rad = math.radians(
            185.0 if int(survey_target) == 3 else 225.0
        )
        try:
            measured_joints = [
                self.robot_motion.current_joint(index) for index in range(6)
            ]
            if all(value is not None for value in measured_joints):
                # Recover the base_link<->model frame convention and the tool
                # offset together from this one static sample (the workcell
                # base_link classically differs from the UR kinematic base by a
                # 180-deg-about-Z flip).  The failure string lists every
                # candidate's offset for diagnosis.
                arm, cal_desc = UR5eArm.autocalibrate(measured_joints, base_T_tcp)
                if arm is not None:
                    # Teach the gate where the wrist cameras ride, so it rejects
                    # poses whose only configurations fold a camera into the
                    # forearm.  Without this the gate is purely kinematic and
                    # publishes poses the workcell planner then refuses with "IK
                    # could not find a collision free configuration"
                    # (robot.forearm_link vs left_camera.camera_link) -- a hard
                    # move failure, not a graceful one.  The camera extrinsics
                    # are the ones already recovered from the permitted TF.
                    arm = replace(
                        arm,
                        flange_T_probes=tuple(
                            arm.flange_T_tcp.compose(extrinsic)
                            for extrinsic in tcp_T_cam.values()
                        ),
                    )
                    seed = np.asarray(measured_joints, dtype=float)
                    ik_arm = arm
                    ik_seed = seed
                    if int(survey_target) == 3:
                        # Ask for a half-turn wrist orientation relative to the
                        # live start.  Choose an exact +/-180 degree target
                        # inside the modeled physical J6 limits; there is no
                        # artificial absolute Move Robot position window.
                        j6_low, j6_high = JOINT_LIMITS[5]
                        requested_flips = (
                            seed[5] + math.pi,
                            seed[5] - math.pi,
                        )
                        legal_flips = tuple(
                            float(value)
                            for value in requested_flips
                            if j6_low - 1e-9 <= value <= j6_high + 1e-9
                        )
                        if legal_flips:
                            preferred_j6_target = legal_flips[0]
                        else:  # Defensive: a valid physical seed always has one.
                            preferred_j6_target = float(
                                np.clip(requested_flips[0], j6_low, j6_high)
                            )

                        def joint_motion_preference_fn(
                            delta,
                            _seed=seed,
                            _preferred=preferred_j6_target,
                        ):
                            return abs(
                                float(_seed[5] + delta[5] - _preferred)
                            )
                    # Try every exact shoulder/elbow/wrist branch.  Choosing the
                    # nearest branch *before* the arm-in-view test falsely
                    # rejects a pose whenever that one branch blocks a camera,
                    # even if another finite branch leaves all three clear.
                    def select_clear_ik_solution(
                        pose,
                        _arm=arm,
                        _seed=seed,
                        _extrinsics=tcp_T_cam,
                        _cameras=camera_models,
                    ):
                        clear = [
                            joints
                            for joints in _arm.solve_ranked(
                                pose,
                                _seed,
                            )
                            if self._arm_clear_of_own_cameras(
                                pose, joints, _arm, _extrinsics, _cameras
                            )
                        ]
                        if not clear:
                            return None
                        return min(
                            clear,
                            key=lambda joints: (
                                float(np.abs(joints - _seed).max()),
                                (
                                    abs(float(joints[5] - preferred_j6_target))
                                    if preferred_j6_target is not None
                                    else 0.0
                                ),
                                float(np.abs(joints - _seed).sum()),
                            ),
                        )

                    # Return physical (unwrapped) joint travel only when the
                    # target has an arm-clear branch.  The survey selector uses
                    # this vector both as the reachability gate and to prefer
                    # the least-contorted roll among camera-equivalent poses.
                    def joint_motion_fn(pose):
                        joints = select_clear_ik_solution(pose)
                        return None if joints is None else joints - seed
                    logging.info(
                        "arm IK joint-motion gate active: %s wrist-camera "
                        "keep-out %.0fmm over %d probes; arm-in-view rejection "
                        "over %d cameras; max predicted joint move %ddeg",
                        cal_desc,
                        arm.min_self_clearance_m * 1000.0,
                        len(arm.flange_T_probes),
                        len(camera_models),
                        round(math.degrees(joint_motion_limit_rad)),
                    )
                else:
                    logging.info(
                        "arm IK calibration failed (%s); using sphere reach",
                        cal_desc,
                    )
            else:
                logging.info(
                    "measured arm joints unavailable; using sphere reach"
                )
        except Exception as ik_error:  # pragma: no cover - defensive
            logging.info(
                "arm IK reachability unavailable (%s); using sphere reach",
                ik_error,
            )

        # NIC keeps its measured straight-down bore view. SC approaches from
        # the long adapter face with a mandatory 10-13 degree offset along that
        # face's board-X normal.
        #
        # SC needs one additional constraint.  The three camera origins are
        # separated by ~115 mm, and wrist roll rotates that baseline relative to
        # the adapter's rectangular 7.6 x 22.4 mm mouth.  A pose can therefore
        # frame every mouth in every image while a side camera looks across the
        # narrow dimension and a wall hides the bore depth.  That is exactly the
        # split seen in hardware: the axis-aligned case found all five, while a
        # diagonal-board case selected the first IK-clear pose even though both
        # side-camera rays lay outside the narrow-axis cone.  Filter those poses
        # geometrically, and within the fixed standoff maximize the worst
        # projected mouth-to-back-centre pixel displacement before minimizing
        # joint motion. All three cameras still pass the independent framing and
        # gripper gates, while two physically open depth views per mouth are
        # sufficient for the fused IVM. Requiring all three was the hidden
        # constraint that forced hardware back to a visibly head-on 7-8 degrees.
        view_quality_fn = None
        min_view_quality = -math.inf
        view_quality_motion_tolerance = 0.0
        if int(survey_target) == 3:  # SC_DESTINATION_PORT
            # SC must not publish a Cartesian target unless the live-seeded IK
            # gate finds an arm-clear configuration whose physical joint delta
            # from the current state stays under the relative motion cap.
            if joint_motion_fn is None:
                self._stage2_not_done(
                    result,
                    "SC survey requires live finite IK for relative joint-motion "
                    "validation",
                )
                return

            sc_view_samples = sc_bore_sample_points()

            def _sc_view_quality(pose):
                bore_margin = rectangular_bore_visibility_margin(
                    sc_view_samples,
                    board_pose.base_T_board,
                    pose,
                    tcp_T_cam,
                    half_width_x_m=SC_BORE_X_TOLERANCE_M,
                    half_width_y_m=SC_BORE_HALF_WIDTH_Y_M,
                    depth_m=SC_BORE_DEPTH_M,
                    camera_names=expected,
                    required_camera_count=2,
                )
                if bore_margin < 0.0:
                    return -math.inf
                return rectangular_bore_depth_cue_px(
                    sc_view_samples,
                    board_pose.base_T_board,
                    pose,
                    tcp_T_cam,
                    camera_models,
                    depth_m=SC_BORE_DEPTH_M,
                    camera_names=expected,
                    required_camera_count=2,
                )

            view_quality_fn = _sc_view_quality
            # Both recent failures contain four real ports in an exact 40 mm
            # lattice while the camera view remains visibly head-on. The prior
            # all-three-camera bore gate made every >=10 degree pose impossible.
            # Requiring two open, strongly oblique views keeps every swept case:
            # selected cues are 3.34-4.45 px. A minimum angle of 11 degrees
            # loses 2/96, so 10-13 is the strongest fully reachable band.
            min_view_quality = 3.0
            view_quality_motion_tolerance = 0.1
        candidate, search_reason = search_survey_pose(
            board_pose,
            tcp_T_cam,
            camera_models,
            grippers,
            reference_camera="center_camera",
            current_base_T_tcp=base_T_tcp,
            # Frame a reachable sector, not the whole board: framing the whole
            # board in all three canted cameras needs a standoff beyond the
            # UR5e's ~0.85 m reach.  The reach guard is the real UR5e envelope
            # (base_link origin); min-motion then picks the closest reachable
            # pose that frames the sector in all cameras.  SFP supplies a
            # centred ladder rather than one box -- see
            # ``_coverage_targets_for_target`` for why an off-centre sector
            # cropped a physically present module.
            coverage_targets=self._coverage_targets_for_target(survey_target),
            # Framing, obliquity and standoff preference are per-sector; see
            # ``_survey_view_settings`` for the port geometry that decides each.
            **self._survey_view_settings(survey_target),
            view_quality=view_quality_fn,
            min_view_quality=min_view_quality,
            view_quality_motion_tolerance=view_quality_motion_tolerance,
            # Reachability is now decided by the live-seeded UR5e IK motion
            # gate, not this base-origin sphere.  In addition to filtering
            # impossible/arm-occluded poses it ranks camera-equivalent rolls by
            # their physical joint travel from the live state and refuses any
            # candidate over the configured relative motion cap.
            # The base-origin sphere wrongly rejected reachable far,
            # bore-facing poses and admitted unsolvable ones; it survives only
            # as a loose fallback when IK is unavailable.  Standoff is still
            # bounded by each sector's configured search band.
            joint_motion=joint_motion_fn,
            max_joint_motion_rad=joint_motion_limit_rad,
            joint_motion_preference=joint_motion_preference_fn,
            # Let the requested J6 half-turn influence the selected roll only
            # inside a bounded motion plateau.  It may buy at most 30 degrees
            # of additional worst-joint travel, never the old violent route.
            joint_preference_motion_tolerance_rad=math.radians(30.0),
            max_reach_m=0.85,
            min_height_m=0.02,
        )
        if candidate is None:
            self._stage2_not_done(
                result, f"no safe all-camera survey pose: {search_reason}"
            )
            return
        if int(survey_target) == 3:  # SC_DESTINATION_PORT
            selected_bore_margin = rectangular_bore_visibility_margin(
                sc_view_samples,
                board_pose.base_T_board,
                candidate.base_T_tcp,
                tcp_T_cam,
                half_width_x_m=SC_BORE_X_TOLERANCE_M,
                half_width_y_m=SC_BORE_HALF_WIDTH_Y_M,
                depth_m=SC_BORE_DEPTH_M,
                camera_names=expected,
                required_camera_count=2,
            )
            logging.info(
                "SC survey image geometry required_depth_cameras=2 "
                "bore_margin_2cam=%+.3f depth_cue_2cam=%.3fpx",
                selected_bore_margin,
                candidate.view_quality,
            )
        # The board-frame region this pose frames in all cameras (whole board or
        # the module region); the post-move confirm re-checks exactly this.
        coverage_target = candidate.coverage_target
        target = candidate.base_T_tcp
        displacement = target.translation - base_T_tcp.translation
        distance_m = float(np.linalg.norm(displacement))
        board_normal = np.asarray(
            board_pose.base_T_board.rotation[:, 2], dtype=float
        )
        board_origin = np.asarray(
            board_pose.base_T_board.translation, dtype=float
        )
        # The survey search already enforced reach, height, <=45deg orientation
        # change, and all-camera coverage.  Add cheap fail-closed guards, then
        # publish the pose -- this skill no longer executes the SFP survey move.
        if (
            float(target.translation[2]) < 0.02
            or float(np.linalg.norm(target.translation)) > 1.20
        ):
            self._stage2_not_done(
                result, "computed survey TCP lies outside the workspace guard"
            )
            return
        # Clearance is measured over the tallest thing standing on the board,
        # not over the board plane.  The NIC card tips reach board Z 0.1793, so
        # the old 0.12 m plane guard sat 59 mm *below* them and never protected
        # them at all -- it would happily pass a pose that puts the tool into the
        # cards.  That matters most for the SC sector, whose ports sit low and
        # whose survey pose is deliberately close.  Move Robot has its own
        # collision checking, but a pose it must refuse is a hard task failure,
        # so keep it out of the published set.
        target_clearance = float(
            np.dot(target.translation - board_origin, board_normal)
        )
        if target_clearance < BOARD_TALLEST_COMPONENT_Z + TOOL_COMPONENT_CLEARANCE_M:
            self._stage2_not_done(
                result,
                "computed survey pose does not clear the board's tallest "
                f"components ({target_clearance:.3f} m over the board plane; "
                f"need {BOARD_TALLEST_COMPONENT_Z + TOOL_COMPONENT_CLEARANCE_M:.3f} m)",
            )
            return

        predicted_joints = None
        if ik_arm is not None and ik_seed is not None:
            predicted_joints = select_clear_ik_solution(target)
        if int(survey_target) == 3 and predicted_joints is None:
            self._stage2_not_done(
                result,
                "selected SC Cartesian pose has no arm-clear IK branch inside "
                "the relative joint-motion budget",
            )
            return
        if predicted_joints is not None:
            predicted_delta = predicted_joints - ik_seed
            logging.info(
                "survey IK motion current_deg=%s target_deg=%s "
                "delta_deg=%s max=%.1fdeg total=%.1fdeg "
                "preferred_j6_deg=%s j6_error=%.1fdeg "
                "relative_origin=live_joints",
                np.round(np.degrees(ik_seed), 1).tolist(),
                np.round(np.degrees(predicted_joints), 1).tolist(),
                np.round(np.degrees(predicted_delta), 1).tolist(),
                math.degrees(float(np.abs(predicted_delta).max())),
                math.degrees(float(np.abs(predicted_delta).sum())),
                (
                    round(math.degrees(preferred_j6_target), 1)
                    if preferred_j6_target is not None
                    else None
                ),
                (
                    math.degrees(
                        abs(float(predicted_joints[5] - preferred_j6_target))
                    )
                    if preferred_j6_target is not None
                    else 0.0
                ),
            )

        # Preserve the deployed interface: Flowstate reads the seven scalar
        # fields in result.target, a Python code node packs them into a
        # Cartesian TCP pose, and Move Robot plans that constrained pose.
        quaternion = quaternion_from_matrix(target.rotation)
        result.target_valid = True
        result.target_frame = self.config.base_frame
        result.target.x = float(target.translation[0])
        result.target.y = float(target.translation[1])
        result.target.z = float(target.translation[2])
        result.target.qx = float(quaternion[0])
        result.target.qy = float(quaternion[1])
        result.target.qz = float(quaternion[2])
        result.target.qw = float(quaternion[3])
        result.survey_pose.position.x = float(target.translation[0])
        result.survey_pose.position.y = float(target.translation[1])
        result.survey_pose.position.z = float(target.translation[2])
        result.survey_pose.orientation.x = float(quaternion[0])
        result.survey_pose.orientation.y = float(quaternion[1])
        result.survey_pose.orientation.z = float(quaternion[2])
        result.survey_pose.orientation.w = float(quaternion[3])
        result.dx = float(displacement[0])
        result.dy = float(displacement[1])
        result.dz = float(displacement[2])
        logging.info(
            "SFP Stage 2 published survey pose source=%s reprojection=%.2fpx "
            "target=(%.4f,%.4f,%.4f)m standoff=%.3fm yaw=%+.3frad "
            "min_clearance=%.1fpx view_quality=%+.3f move=%.3fm "
            "obliquity=%.1fdeg cross_tilt=%.1fdeg along_tilt=%.1fdeg "
            "joint_max=%.1fdeg joint_total=%.1fdeg",
            source_camera,
            board_pose.reprojection_error_px,
            target.translation[0],
            target.translation[1],
            target.translation[2],
            candidate.standoff_m,
            candidate.yaw_rad,
            candidate.min_clearance_px,
            candidate.view_quality,
            distance_m,
            # How far off the board normal the published view ended up.  Every
            # recessed port on this board is read down its own axis, so this is
            # the number that says whether the bores will show their depth.
            # Measured on the reference camera's optical axis, NOT the TCP +Z:
            # the wrist cameras are pitched 15 deg off the tool axis, so the TCP
            # axis reads ~15 deg high and does not correspond to the obliquity
            # limit the search actually enforces.
            math.degrees(
                math.acos(
                    max(
                        -1.0,
                        min(
                            1.0,
                            abs(
                                float(
                                    np.dot(
                                        target.compose(
                                            tcp_T_cam["center_camera"]
                                        ).rotation[:, 2],
                                        board_normal,
                                    )
                                )
                            ),
                        ),
                    )
                )
            ),
            math.degrees(candidate.cross_rail_tilt_rad),
            math.degrees(candidate.along_rail_tilt_rad),
            math.degrees(candidate.max_joint_motion_rad),
            math.degrees(candidate.total_joint_motion_rad),
        )
        result.done = True
        result.success = True
        result.component_coverage_ready = True
        result.steer_camera = source_camera
        result.last_action = "sfp_survey_pose_published"
        result.elapsed_seconds = max(0.0, time.monotonic() - started_at)
        result.message = (
            "geometric Stage 2 published a safe board-relative Cartesian "
            "target framing the SFP/SC modules in all three cameras; pack "
            "result.target x/y/z/qx/qy/qz/qw for Move Robot"
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
        # Single source of truth; see ``robot_motion.force_guard_tripped``.
        # The wrist reading is untared, so its magnitude sweeps a ~12 N
        # free-space envelope as the wrist reorients.  Neither the 18 N
        # absolute ceiling nor a 5 N magnitude-delta survives that, and both
        # fired in free space on hardware.
        from aic_perception.robot_motion import force_guard_tripped

        return force_guard_tripped(
            force_xyz, baseline_xyz, max_force_n, force_delta_n
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

    def stop_service(signum, _frame):
        logging.info(
            "CheckBoardVisibilitySkill stopping on signal %s", signum
        )
        server.stop(grace=1.0)

    signal.signal(signal.SIGINT, stop_service)
    signal.signal(signal.SIGTERM, stop_service)
    try:
        server.wait_for_termination()
    finally:
        server.stop(grace=0)
        skill_instance.close()
        logging.info("CheckBoardVisibilitySkill stopped")


if __name__ == "__main__":
    app.run(start_runner, flags_parser=lambda argv: FLAGS(argv, known_only=True))
