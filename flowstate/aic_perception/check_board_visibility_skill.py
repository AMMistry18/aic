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
# Ceiling on summed six-joint travel for a survey move.
#
# Every successful field move totalled 91-208 deg.  The contorted one that
# reached hardware totalled 561.8 deg with a worst joint of only 166 deg, so
# the worst-joint cap passed it.  The offline distribution is bimodal -- poses
# are either under ~300 deg or over 450 -- so anything in 300..450 selects the
# same set; 400 sits in the gap with margin over the 208 deg field maximum.
#
# Applied to **SC only** (see ``_run_sfp_geometric_stage2``).  Gating SFP as
# well costs 26 of 144 staged-SFP placements -- those are placements whose only
# survey pose is a whole-arm reconfiguration -- and SFP has not reported the
# problem in the field, so it keeps its previous behaviour.
#
# Caveat worth keeping: the 561.8 deg move quoted above was logged on an *SFP*
# run (``view_quality=+inf``, ``cross_tilt=0.0``, 225 deg gate).  An SC-only
# cap therefore would not have refused that particular pose.
TOTAL_JOINT_MOTION_LIMIT_RAD = math.radians(400.0)

# How many calibrated cameras must hold a complete, unobstructed insignia before
# Stage 2 will reconstruct the board.
#
# **One.  Two was tried on hardware on 2026-07-28 and reverted the same day.**
#
# The motivation was real: a single-view PnP of one small quad is a weak *range*
# measurement, the insignia held 0.45% of the centre image, and two invocations
# 7 s apart at the same arm pose disagreed -- one published at 0.640 m standoff,
# the next framed a single candidate at 0.837 m reach and refused.  Selected
# poses sit at 25.3-26.7 px against a 25 px clearance floor, so a few
# millimetres of range error flips the near-standoff family across it.
#
# But requiring two complete views refuses far too many real start poses.  In
# the field it rejected five consecutive invocations with "0 have one" -- at
# poses where a side camera held a *partial* insignia and the board was plainly
# in view -- before an arm position finally exposed it to all three.  Stage 1
# acquisition no longer exists, so every one of those is a dead stop for the
# operator rather than something the skill can recover from.  Availability at a
# usable start pose beats a tighter board pose that is never computed.
#
# What is kept is the free half: whenever two or more cameras *do* accept an
# estimate, they must agree within 5 cm / 8 deg and their board origins are
# averaged (see ``_run_sfp_geometric_stage2``).  That is strictly better than
# the old single-source pick and costs nothing when only one view exists.
REQUIRED_INSIGNIA_CAMERAS = 1

# Last-resort tolerance for an insignia clipped by a sliver.
#
# The Stage-2 landmark is the bracket's complete bounding rectangle, so a
# clipped extreme shrinks it and biases the recovered range -- which is why the
# contract wants the quad 3 px *inside* the frame and why that is still what
# runs whenever any camera provides it.  But "complete in no camera" was
# aborting sequences that were plainly recoverable, with the bracket readable
# in every picture and only a corner of one arm across a border.  12 px is
# about 10% of the bracket's projected size at survey standoffs, so the induced
# range error stays inside the 5 cm agreement window that still has to pass.
SLIVER_EDGE_MARGIN_PX = 12.0

JOINT_MODE_SWITCH_ALLOWANCE_SEC = 3.0
# Measured settling after the profile ends.
JOINT_SETTLE_ALLOWANCE_SEC = 2.0


class InsigniaNotExposedError(RuntimeError):
    """The start pose does not expose the insignia to any calibrated camera.

    Raised out of ``execute`` as a real skill error so the Flowstate process
    fails loudly instead of quietly branching on a result field.  It is
    deliberately raised **after** the controller handoff has been published --
    see ``execute`` -- because throwing before that cleanup leaves the AIC
    controller bridge holding ``arm`` and the next Move Robot fails with
    "Part: 'arm' is already in use."
    """


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
        pending_error: InsigniaNotExposedError | None = None
        try:
            context.canceller.ready()
            self._execute_inner(
                request.params,
                result,
                cancelled=lambda: context.canceller.cancelled,
            )
        except skill_interface.SkillCancelledError:
            raise
        except InsigniaNotExposedError as error:
            # Held, not raised here: the ``finally`` below must publish the
            # controller handoff first.
            pending_error = error
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
        # A missing insignia is a real skill error, not a result field: with
        # Stage 1 removed there is nothing for the process to retry, so it
        # should fail loudly rather than be silently branched around.
        #
        # It is raised *here*, after the ``finally`` above has published the
        # measured-state handoff, precisely because raising before that
        # cleanup is what previously left the AIC controller bridge's ICON
        # session holding ``arm`` and made the following Move Robot fail with
        # "Part: 'arm' is already in use."  The Flowstate process must still
        # run ``Switch To Default Controller`` after this skill; that node
        # releases the bridge lease and is unaffected by the raise.
        #
        # Sensor failures and Stage-2 rejections still come back as results.
        if pending_error is not None:
            # SkillError takes (status_code, message) -- two positional args.
            # 9 is FAILED_PRECONDITION, which is what this is: the caller was
            # required to hand us a start pose that already exposes the
            # insignia, and did not.
            raise skill_interface.SkillError(9, str(pending_error))
        return result

    def _execute_inner(self, params, result, cancelled) -> None:
        """Observe once, then either run Stage 2 or fail.

        There is no Stage 1 any more.  Three successive acquisition designs
        failed on hardware -- a phase machine steering on a board orientation
        that is degenerate in clipped views, a joint plan the deployed
        controller refused to execute, and an image-plane servo with no
        gradient once the board overflowed the frame -- so the search was
        removed rather than tuned again.

        The skill is now pure perception and commands no motion at all.  Stage
        2 remains the only authority for declaring a usable insignia and for
        producing the downstream survey target.  If the start pose does not
        already expose a complete unobstructed insignia to a calibrated
        camera, this reports a hard failure and the caller must fix the start
        pose.
        """

        from aic_perception.arm_ik import UR5eArm
        from aic_perception.board_stage2 import CameraModel
        from aic_perception.board_visibility import analyze_board, view_quality
        from aic_perception.purple_insignia import analyze_purple

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

        purple_reports: dict = {}

        def observe(label: str):
            nonlocal baseline_force_xyz, purple_reports
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

        if self._stage2_has_complete_landmark(snapshot, reports):
            result.last_action = "insignia_handoff"
            logging.info(
                "insignia exposed in a calibrated camera; handing off to "
                "geometric Stage 2"
            )
            handoff_to_stage2(snapshot, reports)
            return

        # No search, no motion.  Stage 1 was removed after three successive
        # designs failed on hardware -- a phase machine steering on a
        # degenerate orientation cue, a joint plan the controller would not
        # execute, and an image-plane servo with no gradient once the board
        # overflowed the frame.  The skill is now pure perception: either the
        # start pose already exposes the insignia or the caller must fix the
        # start pose.
        #
        # This is a hard failure, not the usual expected-rejection result,
        # because there is nothing left for the process to retry.  It still
        # returns normally rather than raising: throwing before cleanup leaves
        # `arm` in use and breaks the next Move Robot node.
        seen_anywhere = sorted(
            name for name, report in purple_reports.items() if report.seen
        )
        complete = sorted(
            self._cameras_with_usable_landmark(snapshot, reports)[0]
        )
        result.success = False
        result.done = False
        result.target_valid = False
        result.last_action = "insignia_not_exposed"
        result.message = (
            "no calibrated camera contains a complete unobstructed purple "
            "insignia"
            + (
                f" ({len(complete)} of {REQUIRED_INSIGNIA_CAMERAS} required)"
                if REQUIRED_INSIGNIA_CAMERAS > 1
                else ""
            )
            + " and Stage 1 acquisition has been removed; "
            + (
                f"partial insignia visible in {','.join(seen_anywhere)}"
                if seen_anywhere
                else "no camera sees the insignia at all"
            )
            + ". Move the arm to a start pose that already exposes it."
        )
        logging.error(result.message)
        raise InsigniaNotExposedError(result.message)


    def _cameras_with_complete_landmark(
        self, snapshot, reports, min_edge_margin_px=3.0
    ) -> list[str]:
        """Calibrated cameras holding a complete Stage-2 landmark."""
        found = []
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
                image,
                reports[camera_name],
                ignored,
                min_edge_margin_px=min_edge_margin_px,
            )
            if observation is not None:
                found.append(camera_name)
        return found

    def _cameras_with_usable_landmark(self, snapshot, reports):
        """Cameras usable for Stage 2, and the edge margin that admitted them.

        The contract is unchanged: a *complete* insignia, 3 px clear of the
        frame, is what Stage 2 wants and what it uses whenever it exists.

        The fallback exists because "complete in no camera" was aborting runs
        that were plainly recoverable -- hardware reported ``partial insignia
        visible in center_camera,left_camera,right_camera`` while the bracket
        was fully readable in the picture and only a sliver of one arm crossed
        a border.  Refusing there throws away a working sequence.

        So when nothing passes the real contract, retry once allowing the quad
        to sit up to ``SLIVER_EDGE_MARGIN_PX`` *outside* the frame.  This is a
        genuinely weaker measurement -- the landmark is the bracket's bounding
        rectangle, so a clipped extreme shrinks it and biases the PnP range --
        hence it is last-resort, logged loudly, and the multi-camera agreement
        check still has to pass on top of it.
        """
        strict = self._cameras_with_complete_landmark(snapshot, reports)
        if strict:
            return strict, 3.0
        relaxed = self._cameras_with_complete_landmark(
            snapshot, reports, min_edge_margin_px=-SLIVER_EDGE_MARGIN_PX
        )
        return relaxed, -SLIVER_EDGE_MARGIN_PX

    def _stage2_has_complete_landmark(self, snapshot, reports) -> bool:
        """Whether enough calibrated cameras have a complete Stage-2 landmark.

        ``REQUIRED_INSIGNIA_CAMERAS`` views, not one: a single small quad's PnP
        range is the jitter that made two identical-input invocations disagree.
        """
        usable, _margin = self._cameras_with_usable_landmark(snapshot, reports)
        return len(usable) >= REQUIRED_INSIGNIA_CAMERAS

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
            SFP_FALLBACK_COVERAGE_HALF_Y,
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
        #
        # Two rungs, widest first, and the wide one is the one that is *correct*
        # rather than merely aspirational.
        #
        # This used to be a three-rung ladder reaching board Y +/-0.2575 -- sized
        # from the high-mix argument that Zones 3/4 fixtures mount on any rail in
        # any order over +/-0.09425 m of travel, so the survey should hold the
        # whole legal pick region.  Measured over 81 cases at the hardware board
        # distance, **no span at or beyond +/-0.17825 is feasible at all**: the
        # box cannot be framed and gripper-cleared from any pose the arm can
        # reach.  Both wide rungs therefore failed in every field run, and
        # because ``search_survey_pose`` runs a full standoff x offset x roll grid
        # per rung, they were charged for as full searches -- roughly two thirds
        # of the 154.86 s tier the field reported.  They are gone.
        #
        # What is left is the span that is actually measurable
        # (``SFP_COVERAGE_HALF_Y``, +/-0.145: 81/81 with all six seats framed and
        # clear of the tool) and, behind it, the +/-0.1125 box that shipped.  The
        # fallback exists purely so availability cannot regress below today; it
        # is known to hide an outer seat behind the gripper, so the caller logs a
        # warning when it is the rung that produced the pose rather than letting
        # it pass as a normal success.
        return (
            sfp_module_strip_corners(),
            sfp_module_strip_corners(SFP_FALLBACK_COVERAGE_HALF_Y),
        )

    @staticmethod
    def _warn_if_degraded_coverage(survey_target: int, candidate) -> None:
        """Announce a survey pose that only framed the fallback coverage box.

        ``_coverage_targets_for_target`` keeps the previously shipped +/-0.1125
        strip as a last rung so availability cannot regress.  On its own that box
        is measured to hide an outer module behind the tool in 81 of 81 swept
        cases, and a pose from it looks clean in every other diagnostic -- all six
        seats 122-159 px inside every image, the box itself gripper-clear,
        obliquity and joint travel healthy -- with one module quietly missing from
        IVM downstream.

        It can no longer publish blind: ``_staged_seats_are_visible`` gates every
        candidate on the seat bodies themselves, so a fallback pose that reaches
        here has been verified seat by seat.  What is left to report is that it got
        there on the thinner box, which means less slack against board-pose error
        than the primary rung carries -- worth knowing when a run is being
        debugged, not worth failing over.
        """
        from aic_perception.board_stage2 import SFP_COVERAGE_HALF_Y

        if int(survey_target) not in (0, 1):
            return
        target = getattr(candidate, "coverage_target", None)
        if target is None:
            return
        half_y = float(np.asarray(target)[:, 1].max())
        if half_y >= SFP_COVERAGE_HALF_Y - 1e-9:
            return
        logging.warning(
            "survey pose framed only the FALLBACK staged-SFP coverage box "
            "(board Y +/-%.4f m instead of +/-%.4f m): no pose framing the wider "
            "box was reachable at this board placement. Every legal module seat "
            "was still independently verified inside all three images and clear "
            "of the tool, so expect all five modules -- but the margin against "
            "board-pose error is thinner than usual. Moving the board closer to "
            "the base restores the primary view.",
            half_y,
            SFP_COVERAGE_HALF_Y,
        )

    @staticmethod
    def _arm_clear_of_own_cameras(
        base_T_tcp,
        joints,
        arm,
        tcp_T_cam,
        cameras,
        sector_regions=None,
        clearance_px=25.0,
    ):
        """True when no arm link occludes the *target sector* in any camera.

        A survey pose can be perfectly top-down, collision-free and fully
        framed, and still be useless because the robot's own upper arm or
        forearm lies across the picture -- which is exactly what a field run
        produced (obliquity 0.0 deg, yet the view was blocked by the arm).  The
        gripper keep-out cannot catch this: it is a fixed image-space silhouette,
        correct only for what is rigidly attached to wrist_3, while these links
        move independently of the wrist.

        **This is a view-quality gate, not a collision gate.**  The collision
        check is ``UR5eArm.self_clearance`` (the 140 mm wrist-camera keep-out),
        which is untouched and still hard.  Nothing here can drive the arm into
        anything.

        The rule used to be "reject if any arm sample lands anywhere in any
        image", which is far stronger than the thing that actually matters and
        was refusing sound poses.  On the 2026-07-28 hardware board it rejected
        29 of 435 NIC candidates at board yaw 135 and 19 of 427 at yaw 180 --
        enough to leave no pose at all, while the published SFP pose at the very
        same board succeeded.  An arm limb clipping a far corner of a splayed
        side image does not occlude ten recessed ports on the other side of the
        frame; only a limb lying *across the ports* does.

        So for the bore sectors the test mirrors how the gripper is already
        handled: the sector's projected region, grown by ``clearance_px``, is the
        keep-out, and the arm merely has to stay off it.  Measured on that board,
        this recovers a pose at all eight swept yaws with zero arm rejections, and
        a limb over the sector is still refused -- which is the SC yaw-70 case the
        gate was built for.

        ``sector_regions=None`` is the whole-image rule: **no arm limb anywhere in
        any image**.  Staged SFP uses it, because there it costs nothing.  Once the
        coverage box reaches the outer seats the search settles at 0.80-0.85 m
        instead of 0.64 m, and over 144 cases the keep-out set to the coverage box,
        the rail column, the whole board face, or nowhere-in-any-image all select
        138/144 with identical standoff and joint-travel ranges.  NIC and SC keep
        the region rule; they sit close to their bore cones and cannot afford it.

        Approximate by construction: the configuration checked is one physical
        branch, and Move Robot may choose another -- hence the caller's
        all-branches-clear rule.
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
                region = (
                    sector_regions.get(name)
                    if sector_regions is not None
                    else None
                )
                if sector_regions is not None and region is None:
                    # The sector does not project into this camera at all, so
                    # there is nothing here for the arm to occlude.
                    continue
                for pixel, ahead, point in zip(pixels, in_front, local):
                    if not ahead or not np.all(np.isfinite(pixel)):
                        continue
                    # Grow the segment by its own radius at that depth, so a
                    # tube grazing the boundary still counts as intruding.
                    margin = radius * float(camera.K[0, 0]) / max(
                        float(point[2]), 1e-6
                    )
                    if region is None:
                        if (
                            -margin <= pixel[0] <= camera.width + margin
                            and -margin <= pixel[1] <= camera.height + margin
                        ):
                            return False
                        continue
                    u_min, v_min, u_max, v_max = region
                    pad = margin + clearance_px
                    if (
                        u_min - pad <= pixel[0] <= u_max + pad
                        and v_min - pad <= pixel[1] <= v_max + pad
                    ):
                        return False
        return True

    def _search_survey_pose_tier(
        self,
        board_pose,
        tcp_T_cam,
        camera_models,
        grippers,
        base_T_tcp,
        survey_target,
        view_settings,
        view_quality_fn,
        min_view_quality,
        view_quality_motion_tolerance,
        joint_motion_fn,
        max_joint_motion_rad,
        max_total_joint_motion_rad,
        joint_motion_preference_fn,
    ):
        """One pass of the survey search at a given relaxation tier.

        ``view_settings`` and the coverage targets carry the per-sector view
        policy and are supplied by the caller unchanged from
        ``_survey_view_settings`` except for the clearance margin, which the
        last tier may reduce.  The joint-travel caps and the arm-branch rule are
        what the tiers actually vary; see the ladder in
        ``_run_sfp_geometric_stage2``.
        """
        from aic_perception.board_stage2 import search_survey_pose

        return search_survey_pose(
            board_pose,
            tcp_T_cam,
            camera_models,
            grippers,
            reference_camera="center_camera",
            current_base_T_tcp=base_T_tcp,
            # Frame a reachable sector, not the whole board: framing the whole
            # board in all three canted cameras needs a standoff beyond the
            # UR5e's ~0.85 m reach.  SFP supplies a centred ladder rather than
            # one box -- see ``_coverage_targets_for_target`` for why an
            # off-centre sector cropped a physically present module.
            coverage_targets=self._coverage_targets_for_target(survey_target),
            **view_settings,
            view_quality=view_quality_fn,
            min_view_quality=min_view_quality,
            view_quality_motion_tolerance=view_quality_motion_tolerance,
            # Reachability is decided by the live-seeded UR5e IK motion gate,
            # not the base-origin sphere, which both admitted unsolvable poses
            # and rejected reachable far ones.
            joint_motion=joint_motion_fn,
            max_joint_motion_rad=max_joint_motion_rad,
            max_total_joint_motion_rad=max_total_joint_motion_rad,
            joint_motion_preference=joint_motion_preference_fn,
            # Let the requested J6 half-turn influence the selected roll only
            # inside a bounded motion plateau.
            joint_preference_motion_tolerance_rad=math.radians(30.0),
            max_reach_m=0.85,
            min_height_m=0.02,
        )

    # There is deliberately no Stage-2 insignia check.
    #
    # A tier once required the published survey pose to keep the purple insignia
    # readable, so that the *next* call in an SFP -> NIC -> SC chain could still
    # localize the board.  It is removed: Stage 1 already gates on a complete
    # insignia before Stage 2 runs, which is the check that matters, and pushing
    # the same requirement onto the survey endpoint bought nothing measurable
    # while costing a whole extra grid search per invocation.  The field trace it
    # was meant to help shows its rejection counter at zero -- it never refused a
    # single candidate.  If a chained call cannot find the insignia from where the
    # previous survey left the arm, the fix belongs in how the process sequences
    # its Move Robot poses, not in narrowing every survey view to protect a
    # measurement that has already been taken.

    @staticmethod
    def _staged_seats_are_visible(base_T_tcp, base_T_board, tcp_T_cam, cameras,
                                  grippers, edge_margin_px=12.0):
        """Is every legal staged-module seat actually visible from this pose?

        **Framing the coverage box is necessary and not sufficient**, the same way
        it is not sufficient for the SC bore or the NIC cage cone.  The box sets
        what the survey aims at and how far off it stands; the modules are what
        IVM has to see, and a seat outside the box is checked by nothing.  Both
        staged-SFP hardware failures were exactly that gap -- the one-rail box
        cropped an outer module out of the image, and the +/-0.1125 box left the
        +Y module 122 px inside the frame and squarely behind the centre camera's
        tool silhouette.

        So ask the real question directly: all six legal seat bodies
        (``sfp_seat_bodies`` -- mount origin through protruding tip), inside the
        usable image and clear of the gripper mask, in all three cameras.  Which
        seat is empty does not matter; the outermost two are occupied and are what
        bind.

        This runs in the **IK gate**, which sees ~68 poses per search, not in
        ``view_quality``, which sees ~10k.  Putting per-candidate projection work
        in the latter is what took a single SFP tier from 64 s to 160 s.
        """
        from aic_perception.board_stage2 import project_points, sfp_seat_bodies

        for seat in sfp_seat_bodies():
            for name, camera in cameras.items():
                cam_from_board = (
                    base_T_tcp.compose(tcp_T_cam[name])
                    .inverse()
                    .compose(base_T_board)
                )
                pixels, in_front = project_points(
                    cam_from_board.apply(seat), camera
                )
                if not np.all(in_front) or not np.all(np.isfinite(pixels)):
                    return False
                if (
                    pixels[:, 0].min() < edge_margin_px
                    or pixels[:, 1].min() < edge_margin_px
                    or pixels[:, 0].max() > camera.width - 1 - edge_margin_px
                    or pixels[:, 1].max() > camera.height - 1 - edge_margin_px
                ):
                    return False
                exclusion = grippers.get(name)
                mask = getattr(exclusion, "mask", None)
                if mask is None:
                    continue
                for u, v in pixels:
                    row = int(round(float(v)))
                    col = int(round(float(u)))
                    if (
                        0 <= row < mask.shape[0]
                        and 0 <= col < mask.shape[1]
                        and mask[row, col]
                    ):
                        return False
        return True

    @staticmethod
    def _sector_image_regions(base_T_tcp, base_T_board, target_board,
                              tcp_T_cam, cameras):
        """Per-camera bounding box of the coverage target's projection."""
        from aic_perception.board_stage2 import project_points

        regions = {}
        for name, camera in cameras.items():
            cam_from_board = (
                base_T_tcp.compose(tcp_T_cam[name])
                .inverse()
                .compose(base_T_board)
            )
            pixels, in_front = project_points(
                cam_from_board.apply(target_board), camera
            )
            if not np.all(in_front) or not np.all(np.isfinite(pixels)):
                continue
            regions[name] = (
                float(pixels[:, 0].min()),
                float(pixels[:, 1].min()),
                float(pixels[:, 0].max()),
                float(pixels[:, 1].max()),
            )
        return regions

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
                # Full finite roll family, as SC and SFP.  A cap measured from
                # the current TCP selects the candidate set rather than
                # bounding motion, so a rolled-wrist Stage-1 exit could leave
                # only unreachable candidates in scope; the live-seeded joint
                # gate bounds the actual travel.  See the SFP block below for
                # the measured start-pose table.
                max_angular_motion_rad=math.pi,
                yaws_rad=tuple(
                    math.radians(deg) for deg in range(-180, 180, 15)
                ),
                # Standoff floor, measured 2026-07-28 (`test/nic_sweep_runner`).
                #
                # Framing is not sufficiency here, exactly as for the SC bore.
                # Over 144 placements the search published 21 poses that framed
                # all ten ports in all three cameras while the outermost ports
                # sat *outside* the 7.5 deg cage cone -- a view that returns
                # about 6 of 10 ports and looks like a success.  Every one of
                # them had been driven below 0.66 m because the arm's envelope
                # would not reach farther at that placement.
                #
                # The poses that do resolve all ten ports occupy a tight and
                # very stable band -- 0.66 .. 0.76 m standoff, worst cone
                # 7.27 deg against the 7.46 deg limit -- and it is the *same*
                # band the previous 90 deg cap and the older harness board
                # position produced whenever they worked.  So this floor is not
                # a new preference; it is the measured signature of the view
                # that has always been correct.
                #
                # Cost is honesty, not coverage: the passing set is unchanged at
                # 105/144, and the 21 misleading successes become `done=false`
                # refusals the process can branch on.  The rungs above 0.76 m
                # are retained because `prefer_far_standoff` should still climb
                # if a nearer board placement ever allows it.
                standoffs_m=(
                    0.66,
                    0.68,
                    0.70,
                    0.73,
                    0.76,
                    0.80,
                    0.85,
                    0.90,
                    1.00,
                    1.15,
                    1.25,
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
        #
        # **Reorientation budget.**  ``max_angular_motion_rad`` is measured from
        # the *current* TCP, so it does not merely bound how far the arm turns
        # -- it decides which candidates are ever scored.  At 90 deg that made
        # availability a function of the Stage-1 exit wrist roll.  Measured at
        # the real hardware board distance (0.558 m horizontal), 8 board yaws:
        #
        #     live start pose      cap=45   cap=90   cap=180
        #     field 01:29            1/8      5/8      7/8
        #     sweep home             3/8      6/8      7/8
        #     home + J6 +90 deg      0/8      0/8      7/8
        #     chained start          5/8      7/8      7/8
        #
        # From the rolled-wrist start the 90 deg cap admitted 1036 framed
        # candidates of which *zero* had any IK solution, which is the field
        # "BINDING GATE = reachability" refusal.  SC already searches the full
        # finite roll family for exactly this reason; SFP and NIC now do too,
        # and the live-seeded joint-travel gate -- which measures real motion
        # from the real start -- remains the authority on how far the arm moves.
        #
        # The roll count goes 7 -> 24 with it: at cap=180 the 24-roll sweep
        # takes every one of those start poses from 7/8 to 8/8, because the
        # surviving pose at some board yaws needs a camera-cluster orientation
        # that the coarse 7-sample family skips.  It costs ~3.7x search time
        # (~2.1 s -> ~8 s per case offline), the same trade NIC and SC already
        # make.
        return dict(
            cross_rail_tilt_band_rad=None,
            cross_rail_sign=0.0,
            require_all_cameras_frame=True,
            prefer_far_standoff=False,
            min_required_clearance_px=25.0,
            max_angular_motion_rad=math.pi,
            # 30 deg steps.  24 rolls (15 deg) bought the last 1-of-8 board yaws
            # over the default 7, but the grid is 21 standoffs x 25 offsets x
            # rolls, so each roll costs real time: 12 halves an SFP search that
            # measured 160 s in the field while keeping most of the coverage.
            yaws_rad=tuple(
                math.radians(deg) for deg in range(-180, 180, 30)
            ),
            # Standoff band, trimmed at both ends.  Closest still wins inside it
            # (letting joint travel outrank distance was measured and rejected --
            # see ``search_survey_pose``); this is purely about not searching rungs
            # that cannot be selected.
            #
            # * 0.70 floor -- ``SFP_COVERAGE_HALF_Y`` (+/-0.145) cannot be framed
            #   and gripper-cleared in three canted cameras nearer than ~0.75 m.
            #   The default ladder opens at 0.30 m, so nine rungs below the floor
            #   each ran a full 25-offset x 12-roll grid of full-resolution
            #   gripper-mask work that could not produce a pose.  Dropping them
            #   halves the search and selects the **bit-identical** TCP pose,
            #   standoff, clearance and roll at every yaw tested.  0.70 rather than
            #   0.76 keeps one spare rung in case a placement frames slightly
            #   closer than any swept one.
            # * 0.90 ceiling -- closest-first means the rungs above 0.90 can never
            #   be selected while anything nearer is feasible, and the 144-case
            #   sweep never selects beyond 0.90 (0.76-0.90, median 0.80).  Keeping
            #   1.00/1.15/1.25 only paid for grid work that could not win.  Handoff
            #   8.2 also measured the arm's own links entering a wrist camera at
            #   every roll past ~0.85 m.
            standoffs_m=(0.70, 0.73, 0.76, 0.80, 0.85, 0.90),
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
    def _stage2_landmarks(
        image, report, ignored_pixels, min_edge_margin_px=3.0
    ):
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
        if quad_margin < min_edge_margin_px:
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
            Transform,
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
        # Same two-pass rule the Stage-1 gate uses: the real contract first,
        # and only if no camera satisfies it, the sliver tolerance.  Stage 1
        # and Stage 2 must agree on which cameras are usable or the gate lets a
        # triplet through that Stage 2 then rejects.
        _usable, insignia_edge_margin_px = self._cameras_with_usable_landmark(
            snapshot, reports
        )
        if insignia_edge_margin_px < 3.0:
            logging.warning(
                "no camera holds a fully-framed insignia; accepting one "
                "clipped by up to %.0f px. The landmark is the bracket's "
                "bounding rectangle, so a clipped extreme biases the recovered "
                "range -- the multi-camera agreement check still applies, but "
                "treat this board pose as lower confidence.",
                SLIVER_EDGE_MARGIN_PX,
            )
        observations = {}
        for camera_name in ("center_camera", "left_camera", "right_camera"):
            if camera_name not in snapshot.frames:
                continue
            image = snapshot.frames[camera_name]["image"]
            ignored = self.gripper_masks.ignored_pixels(
                camera_name, image.shape
            )
            observations[camera_name] = self._stage2_landmarks(
                image,
                reports[camera_name],
                ignored,
                min_edge_margin_px=insignia_edge_margin_px,
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
        if len(pose_estimates) < REQUIRED_INSIGNIA_CAMERAS:
            accepted = sorted(item.camera_name for item in pose_estimates)
            self._stage2_not_done(
                result,
                f"board reconstruction needs {REQUIRED_INSIGNIA_CAMERAS} "
                f"accepted insignia pose estimates and has {len(accepted)}"
                + (f" ({','.join(accepted)})" if accepted else "")
                + (
                    "; rejected: "
                    + "; ".join(
                        f"{name}={reason}"
                        for name, reason in pose_failures.items()
                    )
                    if pose_failures
                    else ""
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
        # Two mutually contradictory accepted estimates may not be guessed
        # between.  This only bites when a second view exists; a lone accepted
        # estimate is its own cluster and passes, which is what keeps the
        # one-camera start poses usable.
        if len(pose_estimates) > 1 and len(consistent) < 2:
            self._stage2_not_done(
                result,
                f"{len(pose_estimates)} accepted camera pose estimates but "
                f"only {len(consistent)} agree within 5 cm / 8 degrees",
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
        # Average the agreeing views' board origin.  This is the point of
        # demanding two of them: each is a single-view PnP whose weakest axis is
        # range, the cameras are ~115 mm apart so their range errors are largely
        # independent, and the survey search runs against a 25 px clearance
        # floor that a few millimetres of range error can cross.  Rotation is
        # left to the preferred view rather than averaged -- an orientation mean
        # over a near-square landmark can interpolate between two different
        # mirror hypotheses, and the 8 deg cluster test already bounds the
        # disagreement.
        cluster_translations = np.array(
            [item.base_T_board.translation for item in consistent], dtype=float
        )
        mean_translation = cluster_translations.mean(axis=0)
        translation_spread_m = float(
            np.linalg.norm(
                cluster_translations - mean_translation, axis=1
            ).max()
        )
        logging.info(
            "board pose fused over %d agreeing cameras (%s): source=%s "
            "origin_spread=%.4fm shift_from_source=%.4fm",
            len(consistent),
            ",".join(sorted(item.camera_name for item in consistent)),
            source_camera,
            translation_spread_m,
            float(
                np.linalg.norm(
                    mean_translation - board_pose.base_T_board.translation
                )
            ),
        )
        board_pose = replace(
            board_pose,
            base_T_board=Transform(
                board_pose.base_T_board.rotation, mean_translation
            ),
        )
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
        # Which gate actually rejected the candidates.  "none had a reachable,
        # joint-motion-valid IK solution" conflates three very different
        # failures -- unreachable, arm-in-view, and over the travel cap -- and
        # they need opposite fixes, so count them separately.
        ik_stats = {
            "probed": 0,
            "no_ik": 0,
            # Analytic IK existed but every branch swung a wrist camera inside
            # the forearm keep-out.  Counted apart from ``no_ik`` because the
            # fixes are opposite: one needs a different standoff/board
            # placement, the other a different keep-out or roll.
            "keepout": 0,
            # Staged SFP only: the pose framed its coverage box but at least one
            # legal module seat left an image or landed behind the tool.
            "seat_hidden": 0,
            "arm_blocked": 0,
            "clear": 0,
            "best_worst_joint_rad": math.inf,
        }
        # Per-pose detail for the near-miss table.  Bounded by the framed count
        # (a few hundred), so holding it all and logging the best handful is
        # cheap and is the only way to see *why* a whole sweep was refused.
        ik_records: list[dict] = []
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
        # Summed six-joint travel is gated for **SC and SFP**; NIC still has no
        # report of the problem.
        #
        # The 185/225 deg cap is on the worst *single* joint, so it says nothing
        # about three joints swinging 170 deg at once -- a whole-arm
        # reconfiguration rather than a survey move, and what the field describes
        # as "the arm contorts and gets in between the cameras and board".  It was
        # SC-only on the grounds that gating SFP cost 26 of 144 placements and SFP
        # had not reported it.  Both halves of that have since failed: a field SFP
        # run published `joint_max=175.5 total=616.5deg`, and the 26-of-144 was
        # measured under the old 90 deg reorientation cap and one-rail coverage.
        #
        # Re-measured at the current policy, capping SFP at 400 deg selects 123 of
        # 144 instead of 138 and pulls worst summed travel from 640 deg to 342.
        # Those 15 are **not** lost: the next relaxation tier lifts the total cap,
        # so a placement with no civilised pose still gets the contorted one --
        # logged as a relaxation instead of published silently as if it were
        # normal.  That is the whole point of the ladder.
        total_joint_motion_limit_rad = (
            TOTAL_JOINT_MOTION_LIMIT_RAD
            if int(survey_target) in (0, 1, 3)
            else math.inf
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
                    # The sector this survey is for.  The arm-in-view gate needs
                    # it to ask the right question -- "is the arm across the
                    # ports" rather than "is the arm anywhere in frame".  The
                    # coverage ladder's first entry is the sector proper; SFP is
                    # the only multi-entry ladder and its entries share a centre.
                    _sector_board = self._coverage_targets_for_target(
                        survey_target
                    )[0]
                    # **SFP takes the absolute rule: no arm limb anywhere in any
                    # image.**
                    #
                    # The sector-region rule exists because the whole-image one
                    # was refusing sound NIC poses -- an arm clipping a far corner
                    # of a splayed side image does not occlude ten ports on the
                    # other side of the frame.  That reasoning still holds for NIC
                    # and SC, which are pinned near their bore cones and cannot
                    # afford it.  It does not hold for SFP any more, and the field
                    # instruction is that the arm getting between the cameras and
                    # the board cannot happen -- the same failure SC hit at board
                    # yaw 70 (handoff 10.6), where the upper arm occupied the
                    # centre camera behind a nominally perfect top-down view.
                    #
                    # It is free here, which is the whole reason to take it.  Now
                    # that the coverage box reaches the outer seats the search
                    # settles at 0.80-0.85 m instead of 0.64 m, and from there the
                    # arm is simply not in frame: measured over 144 cases, keep-out
                    # = coverage box, = rail column, = whole board and = nowhere in
                    # any image all select 138/144 with the same standoff range and
                    # the same joint travel.  The higher view removed the problem
                    # structurally rather than trading anything for it.
                    _whole_image_arm_rule = int(survey_target) in (0, 1)
                    # Staged SFP is the only sector whose components can sit
                    # outside the framed box: NIC cards and SC adapters are bolted
                    # inside their own sector geometry, while a module seat lives
                    # on a rail that extends past anything the search can frame.
                    _seat_gate = (
                        (
                            lambda pose: self._staged_seats_are_visible(
                                pose,
                                board_pose.base_T_board,
                                tcp_T_cam,
                                camera_models,
                                grippers,
                            )
                        )
                        if int(survey_target) in (0, 1)
                        else None
                    )

                    def select_clear_ik_solution(
                        pose,
                        _arm=arm,
                        _seed=seed,
                        _extrinsics=tcp_T_cam,
                        _cameras=camera_models,
                        _sector_board=_sector_board,
                    ):
                        ik_stats["probed"] += 1
                        if _seat_gate is not None and not _seat_gate(pose):
                            ik_stats["seat_hidden"] += 1
                            return None
                        reach = float(np.linalg.norm(pose.translation))
                        tcp_z = float(pose.translation[2])
                        solutions = _arm.solve_ranked(pose, _seed)
                        record = {
                            "reach_m": reach,
                            "tcp_z_m": tcp_z,
                            "n_ik": len(solutions),
                            "n_clear": 0,
                            "worst_joint_rad": math.inf,
                            "gate": "unreachable",
                        }
                        if not solutions:
                            # ``solve_ranked`` filters the wrist-camera/forearm
                            # keep-out *before* returning, so an empty list has
                            # two completely different meanings and the old
                            # code reported both as "no analytic IK solution at
                            # all".  That verdict sent debugging after the arm's
                            # workspace when the pose was in fact reachable and
                            # only the 140 mm keep-out refused it -- measured at
                            # the hardware board distance, 231 of 926 "no IK"
                            # verdicts were keep-out rejections.  Re-solve
                            # without the keep-out (only on the failure path, so
                            # the hot path is unchanged) and name the real gate.
                            if _arm.solve_all(pose):
                                ik_stats["keepout"] += 1
                                record["gate"] = "camera_keepout"
                            else:
                                ik_stats["no_ik"] += 1
                                record["gate"] = "unreachable"
                            ik_records.append(record)
                            return None
                        # Cheapest branch travel regardless of arm-in-view, so
                        # the table can separate "no branch exists" from "the
                        # only clear branch is too far".
                        record["worst_joint_rad"] = min(
                            float(np.abs(joints - _seed).max())
                            for joints in solutions
                        )
                        arm_keepout_regions = (
                            None
                            if _whole_image_arm_rule
                            else self._sector_image_regions(
                                pose,
                                board_pose.base_T_board,
                                _sector_board,
                                _extrinsics,
                                _cameras,
                            )
                        )
                        clear = [
                            joints
                            for joints in solutions
                            if self._arm_clear_of_own_cameras(
                                pose,
                                joints,
                                _arm,
                                _extrinsics,
                                _cameras,
                                arm_keepout_regions,
                            )
                        ]
                        record["n_clear"] = len(clear)
                        # HARD STOP on the arm occluding its own view.
                        #
                        # The skill publishes a *Cartesian* pose; Move Robot
                        # re-solves it and may take any co-terminal branch, not
                        # the one checked here.  Accepting a pose because *one*
                        # branch is arm-clear therefore guarantees nothing --
                        # and offline most accepted poses have only some
                        # branches clear, which is how the arm keeps landing
                        # under the camera in the field.
                        #
                        # Require every branch to be clear, so whichever one
                        # Move Robot picks, the arm is out of frame.
                        # The last tiers accept a pose with *an* arm-clear
                        # branch rather than demanding every branch be clear.
                        # Preferred remains all-clear -- Move Robot picks the
                        # branch -- but a view that exists on one branch beats
                        # publishing nothing, and the caller logs which tier
                        # paid for it.
                        if not clear or (
                            len(clear) != len(solutions)
                            and not relax["any_branch"]
                        ):
                            ik_stats["arm_blocked"] += 1
                            record["gate"] = (
                                "arm_in_view"
                                if not clear
                                else "arm_in_view_some_branches"
                            )
                            ik_records.append(record)
                            return None
                        ik_stats["clear"] += 1
                        clear_worst = min(
                            float(np.abs(joints - _seed).max())
                            for joints in clear
                        )
                        record["worst_joint_rad"] = clear_worst
                        record["gate"] = "passed_ik"
                        ik_records.append(record)
                        ik_stats["best_worst_joint_rad"] = min(
                            ik_stats["best_worst_joint_rad"], clear_worst
                        )
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
        # --- relaxation ladder -------------------------------------------
        #
        # Two things are non-negotiable and appear in *no* tier below:
        #
        #   1. the view.  Each sector's coverage box must be framed in all
        #      three cameras and clear of the gripper silhouette, with its own
        #      geometry -- SFP's near-overhead strip view, NIC's <=2 deg
        #      straight-down look into the cages, SC's 16-20 deg long-face
        #      depth band and two-camera bore gate.  ``coverage_targets``,
        #      ``require_all_cameras_frame``, the obliquity/tilt bands and
        #      ``min_view_quality`` are therefore fixed.
        #   2. collisions.  ``UR5eArm.self_clearance``'s 140 mm wrist-camera
        #      keep-out is applied inside ``solve_ranked`` and is never
        #      loosened by anything here.
        #
        # Everything else is comfort, not correctness, and comfort should not
        # be the reason the skill returns no pose at all.  The field kept
        # hitting refusals whose binding gate was joint travel or the
        # all-branches arm rule while a perfectly good view existed.  So try
        # the strict policy first and, only if it finds nothing, step down one
        # constraint at a time and log which tier paid for the result.
        #
        # Note the gripper is *allowed* in frame -- it always is, that is what
        # the calibrated silhouette is for.  What must stay clear of the
        # sector is the gripper's keep-out and the arm's own limbs.
        # The last two tiers trade *view angle* for existence, and only for
        # sectors using an isotropic obliquity cap (NIC, SFP) -- never for SC,
        # whose directional tilt band is the depth measurement itself.
        #
        # This is deliberately last.  Section 9.3 records that tilting NIC
        # across the rail resolved 0 of 10 ports, because the cages only show
        # their black interior near the board normal.  But that experiment
        # traded a good view for a tilted one; here the alternative is no pose
        # at all, and a degraded view strictly beats nothing.  Measured at the
        # 2026-07-28 04:34 refusal (NIC ports 0.715 m out, every straight-down
        # candidate needing >=0.914 m of reach against an envelope of ~0.86 m):
        #
        #     obliquity cap   found   standoff   reach    ports in cone
        #      2 deg (ship)    no       -          -         -
        #      5 deg           no       -          -         -
        #      8 deg          YES      0.66      0.862      5/10
        #     15 deg          YES      0.66      0.862      5/10
        #
        # 8 deg is the threshold and buys 5 ports instead of 0.  Beyond it
        # nothing improves: the ranking still prefers the most overhead
        # feasible pose, so the wider cap is permission, not a worse view.
        #
        # There is **no insignia tier**.  One used to sit in front of "strict",
        # requiring the published pose to keep the purple insignia readable so a
        # chained NIC/SC call could still localize the board.  It never rejected a
        # candidate in the field (its counter read zero on the trace it was added
        # for), Stage 1 already gates on a complete insignia, and every tier costs
        # a full standoff x offset x roll grid -- so it was pure latency.  The
        # full reasoning sits above ``_arm_clear_of_own_cameras``, where the
        # projection helper it used to call was defined.
        relax = {"any_branch": False}
        search_tiers = (
            ("strict", joint_motion_limit_rad, total_joint_motion_limit_rad,
             False, None, None),
            ("joint-travel caps lifted", math.radians(360.0), math.inf,
             False, None, None),
            ("any arm-clear IK branch", math.radians(360.0), math.inf,
             True, None, None),
            ("reduced clearance margin", math.radians(360.0), math.inf,
             True, 12.0, None),
            ("angled view (8deg off normal)", math.radians(360.0), math.inf,
             True, 12.0, 8.0),
            ("angled view (15deg off normal)", math.radians(360.0), math.inf,
             True, 12.0, 15.0),
        )
        # Verbose by design.  A refusal has to be explainable from the log
        # alone: which sector policy ran, against which board, from which live
        # joints, how many candidates each gate consumed, and at which tier.
        # Three field sessions were spent re-deriving that from a single line.
        base_settings = self._survey_view_settings(survey_target)
        coverage_boxes = self._coverage_targets_for_target(survey_target)
        board_origin = board_pose.base_T_board.translation
        logging.info(
            "survey policy target=%d sector_boxes=%d coverage_x=[%.3f,%.3f] "
            "coverage_y=[%.3f,%.3f] coverage_z=[%.3f,%.3f] all_cameras=%s "
            "prefer_far=%s obliquity_max=%.1fdeg clearance_floor=%.1fpx "
            "rolls=%d standoffs=%s tilt_band=%s min_view_quality=%s",
            int(survey_target),
            len(coverage_boxes),
            float(coverage_boxes[0][:, 0].min()),
            float(coverage_boxes[0][:, 0].max()),
            float(coverage_boxes[0][:, 1].min()),
            float(coverage_boxes[0][:, 1].max()),
            float(coverage_boxes[0][:, 2].min()),
            float(coverage_boxes[0][:, 2].max()),
            base_settings.get("require_all_cameras_frame"),
            base_settings.get("prefer_far_standoff"),
            math.degrees(
                base_settings.get("max_obliquity_rad", math.radians(20.0))
            ),
            base_settings.get("min_required_clearance_px", math.nan),
            len(base_settings.get("yaws_rad", ()) or ()) or 7,
            (
                "%.2f..%.2f"
                % (
                    min(base_settings["standoffs_m"]),
                    max(base_settings["standoffs_m"]),
                )
                if base_settings.get("standoffs_m")
                else "default 0.30..1.25"
            ),
            (
                "%.1f..%.1fdeg"
                % tuple(
                    math.degrees(v)
                    for v in base_settings["cross_rail_tilt_band_rad"]
                )
                if base_settings.get("cross_rail_tilt_band_rad")
                else "none"
            ),
            (
                f"{min_view_quality:.2f}"
                if math.isfinite(min_view_quality)
                else "none"
            ),
        )
        logging.info(
            "survey inputs board_origin=(%.4f,%.4f,%.4f) board_normal=(%.3f,"
            "%.3f,%.3f) source=%s reprojection=%.2fpx tcp=(%.4f,%.4f,%.4f) "
            "seed_deg=%s live_ik=%s",
            board_origin[0],
            board_origin[1],
            board_origin[2],
            *board_pose.base_T_board.rotation[:, 2],
            source_camera,
            board_pose.reprojection_error_px,
            base_T_tcp.translation[0],
            base_T_tcp.translation[1],
            base_T_tcp.translation[2],
            (
                np.round(np.degrees(ik_seed), 1).tolist()
                if ik_seed is not None
                else "n/a"
            ),
            joint_motion_fn is not None,
        )

        candidate, search_reason, search_tier = None, "", "strict"
        tier_reports: list[str] = []
        for tier_index, (
            tier_label,
            tier_joint_cap,
            tier_total_cap,
            tier_any_branch,
            tier_clearance_px,
            tier_obliquity_deg,
        ) in enumerate(search_tiers):
            relax["any_branch"] = tier_any_branch
            for key in ("probed", "no_ik", "keepout", "seat_hidden",
                        "arm_blocked", "clear"):
                ik_stats[key] = 0
            ik_stats["best_worst_joint_rad"] = math.inf
            ik_records.clear()
            view_settings = self._survey_view_settings(survey_target)
            if tier_clearance_px is not None:
                view_settings["min_required_clearance_px"] = tier_clearance_px
            if tier_obliquity_deg is not None:
                if view_settings.get("cross_rail_tilt_band_rad") is not None:
                    # SC: the tilt band *is* the depth measurement.  There is
                    # no meaningful "slightly worse angle" here, so this tier
                    # has nothing to offer and would only repeat tier 3.
                    continue
                # Only ever widen.  SFP already allows 20 deg, so these tiers
                # are a no-op there and bite only on NIC's 2 deg cap.
                view_settings["max_obliquity_rad"] = max(
                    view_settings.get(
                        "max_obliquity_rad", math.radians(20.0)
                    ),
                    math.radians(tier_obliquity_deg),
                )
            # Rule worth keeping even though the insignia check that motivated
            # it is gone: never put per-candidate work in ``view_quality``.  It
            # is evaluated on every candidate surviving the cheap prunes (~10k
            # per search) while only the framed handful (68 in the field trace)
            # reaches the IK gate.  Three camera projections plus a mask lookup
            # there took a single SFP tier from 64 s to 160 s.
            tier_view_quality = view_quality_fn
            tier_min_view_quality = min_view_quality
            tier_started_at = time.monotonic()
            candidate, search_reason = self._search_survey_pose_tier(
                board_pose,
                tcp_T_cam,
                camera_models,
                grippers,
                base_T_tcp,
                survey_target,
                view_settings,
                tier_view_quality,
                tier_min_view_quality,
                view_quality_motion_tolerance,
                joint_motion_fn,
                tier_joint_cap,
                tier_total_cap,
                joint_motion_preference_fn,
            )
            tier_summary = (
                "tier='%s' joint_cap=%.0fdeg total_cap=%s any_branch=%s "
                "clearance=%.1fpx -> %s | probed=%d unreachable=%d "
                "camera_keepout=%d seat_hidden=%d arm_in_view=%d arm_clear=%d "
                "best_worst_joint=%s took=%.2fs"
                % (
                    tier_label,
                    math.degrees(tier_joint_cap),
                    (
                        f"{math.degrees(tier_total_cap):.0f}deg"
                        if math.isfinite(tier_total_cap)
                        else "none"
                    ),
                    tier_any_branch,
                    view_settings.get("min_required_clearance_px", math.nan),
                    "FOUND" if candidate is not None else "none",
                    ik_stats["probed"],
                    ik_stats["no_ik"],
                    ik_stats["keepout"],
                    ik_stats["seat_hidden"],
                    ik_stats["arm_blocked"],
                    ik_stats["clear"],
                    (
                        f"{math.degrees(ik_stats['best_worst_joint_rad']):.1f}deg"
                        if math.isfinite(ik_stats["best_worst_joint_rad"])
                        else "n/a"
                    ),
                    time.monotonic() - tier_started_at,
                )
            )
            tier_reports.append(tier_summary)
            logging.info("survey search %s", tier_summary)
            if candidate is not None:
                search_tier = tier_label
                if tier_obliquity_deg is not None:
                    # The one relaxation that genuinely degrades the picture.
                    logging.warning(
                        "survey pose required an ANGLED view (tier '%s'): no "
                        "pose within the sector's normal-view cap was "
                        "reachable at this board placement. The bores are read "
                        "down their own axis, so expect fewer than all ports "
                        "to resolve -- offline this recovered 5 of 10 where "
                        "the straight-down view had none. Moving the board "
                        "closer to the base restores the full view. "
                        "Strict tier said: %s",
                        tier_label,
                        tier_reports[0],
                    )
                elif tier_index > 0:
                    logging.warning(
                        "survey pose required relaxation tier '%s'; the view "
                        "requirements and the %.0fmm collision keep-out were "
                        "NOT relaxed. Reason the strict tier failed: %s",
                        tier_label,
                        (ik_arm.min_self_clearance_m if ik_arm else 0.0) * 1000.0,
                        tier_reports[0],
                    )
                self._warn_if_degraded_coverage(survey_target, candidate)
                break
            if joint_motion_fn is None:
                # Without a live IK model the tiers differ in nothing that
                # matters; do not sweep the same search four times.
                break
        if candidate is None:
            search_reason = (
                f"{search_reason} (all {len(search_tiers)} relaxation tiers "
                "exhausted; view requirements and the collision keep-out are "
                "never relaxed)"
            )
        if candidate is None:
            # Full attribution: every tier that was tried and what consumed its
            # candidates.  Read top-down -- if `probed` is 0 the search never
            # reached IK and the loss is framing/clearance/obliquity; if
            # `unreachable` dominates the board is simply too far for the
            # standoff this sector needs; if `arm_in_view` or `camera_keepout`
            # dominate the pose was reachable and something else refused it.
            for line in tier_reports:
                logging.error("survey search exhausted %s", line)
            best_worst = ik_stats["best_worst_joint_rad"]
            cap_deg = math.degrees(joint_motion_limit_rad)
            logging.error(
                "survey IK rejection breakdown: probed=%d unreachable=%d "
                "camera_keepout=%d seat_hidden=%d "
                "arm_in_view=%d arm_clear=%d best_worst_joint=%s cap=%.1fdeg",
                ik_stats["probed"],
                ik_stats["no_ik"],
                ik_stats["keepout"],
                ik_stats["seat_hidden"],
                ik_stats["arm_blocked"],
                ik_stats["clear"],
                (
                    f"{math.degrees(best_worst):.1f}deg"
                    if math.isfinite(best_worst)
                    else "n/a"
                ),
                cap_deg,
            )
            # Name the binding gate outright rather than leaving it to be
            # inferred from the counts.
            if ik_stats["probed"] == 0:
                verdict = (
                    "no pose reached the IK gate at all -- everything died on "
                    "framing, clearance, obliquity or the 0.85 m reach prune"
                )
            elif ik_stats["clear"] > 0:
                verdict = (
                    f"BINDING GATE = joint-travel cap: {ik_stats['clear']} "
                    f"pose(s) had an arm-clear branch but the cheapest needed "
                    f"{math.degrees(best_worst):.1f}deg against a "
                    f"{cap_deg:.1f}deg cap"
                )
            # Rank the gates by what is *actionable*, not by raw count.
            #
            # ``no_ik`` dominates almost every refusal simply because the grid
            # sweeps standoffs the arm was never going to reach -- NIC probes
            # the farthest rungs first by design, so it racks up hundreds of
            # unreachable poses on the way to the ones that matter.  Reporting
            # that as the binding gate sent the 2026-07-28 NIC failure after the
            # workspace when all eight of its near-misses were `arm_in_view`
            # with 2-4 valid IK branches.  Among candidates the arm *can* reach,
            # whichever gate refused them is the one worth naming.
            elif ik_stats["arm_blocked"] >= max(1, ik_stats["keepout"]):
                verdict = (
                    f"BINDING GATE = arm-in-view: {ik_stats['arm_blocked']} "
                    "pose(s) had valid IK but the arm sat in a wrist camera "
                    "on every branch"
                )
            elif ik_stats["keepout"] > 0:
                verdict = (
                    f"BINDING GATE = wrist-camera keep-out: "
                    f"{ik_stats['keepout']} pose(s) were kinematically "
                    "reachable but every branch put a wrist camera inside the "
                    f"{(ik_arm.min_self_clearance_m if ik_arm else 0.0) * 1000:.0f}"
                    "mm forearm keep-out"
                )
            else:
                verdict = (
                    f"BINDING GATE = reachability: {ik_stats['no_ik']} "
                    "pose(s) had no analytic IK solution at all"
                )
            logging.error("survey IK verdict: %s", verdict)
            logging.error(
                "survey IK seed_deg=%s reach_span=%s tcp_z_span=%s",
                (
                    np.round(np.degrees(ik_seed), 1).tolist()
                    if ik_seed is not None
                    else "n/a"
                ),
                (
                    f"{min(r['reach_m'] for r in ik_records):.3f}.."
                    f"{max(r['reach_m'] for r in ik_records):.3f}m"
                    if ik_records
                    else "n/a"
                ),
                (
                    f"{min(r['tcp_z_m'] for r in ik_records):.3f}.."
                    f"{max(r['tcp_z_m'] for r in ik_records):.3f}m"
                    if ik_records
                    else "n/a"
                ),
            )
            # Nearest misses: most IK branches first, then least travel.
            for index, record in enumerate(
                sorted(
                    ik_records,
                    key=lambda item: (
                        -item["n_clear"],
                        -item["n_ik"],
                        item["worst_joint_rad"],
                    ),
                )[:8],
                start=1,
            ):
                logging.error(
                    "  near-miss %d: gate=%-12s reach=%.3fm tcp_z=%.3fm "
                    "ik_branches=%d arm_clear=%d worst_joint=%s",
                    index,
                    record["gate"],
                    record["reach_m"],
                    record["tcp_z_m"],
                    record["n_ik"],
                    record["n_clear"],
                    (
                        f"{math.degrees(record['worst_joint_rad']):.1f}deg"
                        if math.isfinite(record["worst_joint_rad"])
                        else "n/a"
                    ),
                )
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
            "Stage 2 published survey pose target=%d tier='%s' source=%s "
            "reprojection=%.2fpx "
            "target=(%.4f,%.4f,%.4f)m standoff=%.3fm yaw=%+.3frad "
            "min_clearance=%.1fpx view_quality=%+.3f move=%.3fm "
            "obliquity=%.1fdeg cross_tilt=%.1fdeg along_tilt=%.1fdeg "
            "joint_max=%.1fdeg joint_total=%.1fdeg",
            int(survey_target),
            search_tier,
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
