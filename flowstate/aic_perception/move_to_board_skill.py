#!/usr/bin/env python3
"""Minimal camera-ready Flowstate scaffold for a new board-survey skill.

This intentionally does *not* command robot motion yet. Its job is to give
the new implementation the same tested ROS/Flowstate plumbing as the older
skill: three permitted wrist-camera subscriptions, force feedback, guarded
controller helpers, and robot-mounted TF access. Add the simple
image-feedback search inside :meth:`MoveToBoardSkill._execute_inner` once the
camera-only smoke test works in Flowstate.
"""

from __future__ import annotations

from concurrent import futures
import threading
import traceback

from absl import app, flags, logging
import grpc

from intrinsic.skills.python import skill_interface


FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 8003, "Port to listen on.", allow_override=True)
flags.DEFINE_string(
    "skill_service_config_filename",
    "",
    "Path to the generated skill config.",
    allow_override=True,
)

# Looser than IVM-ready full: board inside frame with a small fixed margin.
BOARD_MARGIN_PX = 10
BOARD_CONTEXT_PAD_FRAC = 0.0
CENTER_STEP_M = 0.03
CENTER_ERROR_TRIGGER = 0.10
MAX_FORCE_N = 18.0
FORCE_DELTA_N = 5.0
MAX_SPEED_MPS = 0.04
PUBLISH_HZ = 20.0
SETTLE_TOLERANCE_M = 0.006
MOVE_TIMEOUT_SEC = 8.0


def board_view_ready(report) -> bool:
    """True when the board is in frame with our survey margin (not friend's IVM pad)."""
    return bool(report.seen and not report.edges)


def pick_best_camera(reports: dict) -> tuple[str | None, object | None]:
    """Prefer any seen board, then largest area, then view quality."""
    from aic_perception.board_visibility import view_quality

    seen = [(name, report) for name, report in reports.items() if report.seen]
    if not seen:
        return None, None
    name, report = max(
        seen,
        key=lambda item: (item[1].area_frac, view_quality(item[1])),
    )
    return name, report


def image_plane_direction(report) -> tuple[float, float] | None:
    """Unit (image-right, image-down) direction that centers the board.

    Prefer single-edge cues; if opposite edges are clipped, fall back to
    center_error on that axis. Raise/standoff is intentionally not handled here.
    """
    import math

    def axis_sign(neg_edge: str, pos_edge: str, center_value: float) -> float:
        has_neg = neg_edge in report.edges
        has_pos = pos_edge in report.edges
        if has_neg and not has_pos:
            return -1.0
        if has_pos and not has_neg:
            return 1.0
        if abs(center_value) >= CENTER_ERROR_TRIGGER:
            return 1.0 if center_value > 0.0 else -1.0
        return 0.0

    dx = axis_sign("left", "right", float(report.center_error[0]))
    dy = axis_sign("top", "bottom", float(report.center_error[1]))
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    return (dx / norm, dy / norm)


class MoveToBoardSkill(skill_interface.Skill):
    """Fresh, camera-ready skill shell for the board survey experiment."""

    def __init__(self):
        super().__init__()

        # Keep ROS imports here so importing this file for syntax/tests does
        # not require a running ROS environment.
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros.buffer import Buffer
        from tf2_ros.transform_listener import TransformListener

        from aic_perception.camera_rig import CameraRig
        from aic_perception.config import PerceptionConfig
        from aic_perception.gripper_masks import GripperMaskBank
        from aic_perception.robot_motion import RobotMotion

        if not rclpy.ok():
            rclpy.init()

        # These are shared, well-tested helpers. They subscribe only to the
        # permitted camera/force/controller topics declared in config.py.
        self.config = PerceptionConfig()
        self.node = rclpy.create_node("move_to_board_node")
        self.camera_rig = CameraRig(self.node, self.config)
        self.gripper_masks = GripperMaskBank()
        self.robot_motion = RobotMotion(self.node, self.camera_rig, self.config)

        # The future move loop may use only the robot-mounted camera frames
        # and TCP frame from config.py; it must never query board/world TF.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self.node, spin_thread=False
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="move-to-board-ros",
            daemon=True,
        )
        self._spin_thread.start()
        self._execute_lock = threading.Lock()

        logging.info(
            "MoveToBoardSkill ready: image_topics=%s wrench_topic=%s",
            self.config.image_topics,
            self.config.wrench_topic,
        )

    def execute(self, request, context):
        """Run one safe camera-only smoke test until movement is implemented."""
        from aic_perception import move_to_board_skill_pb2 as pb2

        result = pb2.MoveToBoardSkillResult(success=False)
        if not self._execute_lock.acquire(blocking=False):
            result.message = "another move-to-board invocation is still running"
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

        if not result.success:
            raise skill_interface.SkillError(
                4, result.message or "move-to-board camera setup failed"
            )
        return result

    def _scan_cameras(self, snapshot) -> dict:
        """Phase 0.1: mask + analyze every fresh wrist camera."""
        from aic_perception.board_visibility import analyze_board

        reports = {}
        for name, frame in snapshot.frames.items():
            masked = self.gripper_masks.apply(name, frame["image"])
            reports[name] = analyze_board(
                masked,
                margin_px=BOARD_MARGIN_PX,
                ignore_bottom_frac=0.0,
                context_pad_frac=BOARD_CONTEXT_PAD_FRAC,
            )
        return reports

    def _fill_camera_summaries(self, snapshot, reports, result) -> None:
        result.camera_names[:] = []
        result.camera_summaries[:] = []
        result.camera_names.extend(sorted(snapshot.frames))
        for name in sorted(reports):
            report = reports[name]
            height, width = snapshot.frames[name]["image"].shape[:2]
            ready = board_view_ready(report)
            result.camera_summaries.append(
                f"{name}={width}x{height} seen={report.seen} ready={ready} "
                f"area={report.area_frac:.3f} edges={','.join(sorted(report.edges)) or 'none'}"
            )
            logging.info(
                "%s board scan: seen=%s ready=%s area=%.3f edges=%s center=(%.3f,%.3f)",
                name,
                report.seen,
                ready,
                report.area_frac,
                sorted(report.edges),
                report.center_error[0],
                report.center_error[1],
            )

    def _gripper_pose(self, timeout_sec: float):
        from rclpy.duration import Duration
        from rclpy.time import Time
        import numpy as np

        transform = self.tf_buffer.lookup_transform(
            self.config.base_frame,
            self.config.gripper_frame,
            Time(),
            timeout=Duration(seconds=min(timeout_sec, 3.0)),
        )
        t = transform.transform.translation
        r = transform.transform.rotation
        values = np.asarray(
            (t.x, t.y, t.z, r.x, r.y, r.z, r.w), dtype=float
        )
        if not np.all(np.isfinite(values)) or float(np.linalg.norm(values[3:])) < 0.5:
            raise ValueError("TCP transform invalid")
        return (
            (float(values[0]), float(values[1]), float(values[2])),
            (float(values[3]), float(values[4]), float(values[5]), float(values[6])),
        )

    def _camera_axes_in_base(self, camera: str, timeout_sec: float):
        from aic_perception.board_visibility import optical_axes_in_base
        from rclpy.duration import Duration
        from rclpy.time import Time

        if camera not in self.config.camera_frames:
            raise ValueError(f"camera {camera!r} outside TF allowlist")
        transform = self.tf_buffer.lookup_transform(
            self.config.base_frame,
            self.config.camera_frames[camera],
            Time(),
            timeout=Duration(seconds=min(timeout_sec, 3.0)),
        )
        q = transform.transform.rotation
        return optical_axes_in_base(q.x, q.y, q.z, q.w)

    def _center_in_image(
        self,
        camera: str,
        report,
        *,
        timeout_sec: float,
        baseline_force_xyz,
        cancelled,
    ):
        """One camera-plane 2D step toward centering the board."""
        import numpy as np
        from aic_perception.board_visibility import world_delta
        from aic_perception.robot_motion import normalize_quaternion

        direction = image_plane_direction(report)
        if direction is None:
            return None, "no 2D centering signal (board already near image center)"

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
        target = tuple(float(v) for v in np.asarray(position, dtype=float) + delta)
        logging.info(
            "2D center on %s: image_dir=(%+.2f,%+.2f) step=%.3fm "
            "delta=(%.4f,%.4f,%.4f)",
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
            publish_hz=PUBLISH_HZ,
            settle_tolerance_m=SETTLE_TOLERANCE_M,
            timeout_sec=MOVE_TIMEOUT_SEC,
            baseline_force_xyz=baseline_force_xyz,
            max_force_n=MAX_FORCE_N,
            force_delta_n=FORCE_DELTA_N,
            cancelled=cancelled,
        )
        return outcome, None

    def _execute_inner(self, params, result, cancelled) -> None:
        """Observe, then one 2D image-plane centering move if not ready."""
        timeout_sec = float(params.camera_timeout_seconds or 3.0)
        if timeout_sec <= 0.0:
            raise ValueError("camera_timeout_seconds must be positive")
        if cancelled():
            raise skill_interface.SkillCancelledError("move-to-board cancelled")

        snapshot = self.camera_rig.grab(timeout_sec=timeout_sec)
        if snapshot is None or not snapshot.frames:
            result.message = "no fresh wrist-camera image received"
            return
        if snapshot.force_xyz is None:
            result.message = "no fresh wrist-force sample; refusing motion"
            return

        reports = self._scan_cameras(snapshot)
        result.num_cameras = len(snapshot.frames)
        result.force_n = float(snapshot.force_norm or 0.0)
        self._fill_camera_summaries(snapshot, reports, result)

        camera, report = pick_best_camera(reports)
        if camera is None or report is None:
            result.success = False
            result.done = False
            result.message = "no board detected after gripper masking"
            return

        if board_view_ready(report):
            result.success = True
            result.done = True
            result.message = (
                f"board already in view on {camera} "
                f"(area={report.area_frac:.3f}, margin={BOARD_MARGIN_PX}px)"
            )
            logging.info(result.message)
            return

        outcome, skip_reason = self._center_in_image(
            camera,
            report,
            timeout_sec=timeout_sec,
            baseline_force_xyz=snapshot.force_xyz,
            cancelled=cancelled,
        )
        if skip_reason is not None:
            result.success = True
            result.done = False
            result.message = (
                f"board seen on {camera} but 2D center skipped: {skip_reason}"
            )
            return
        if outcome.cancelled:
            raise skill_interface.SkillCancelledError(outcome.message)
        if not outcome.success:
            result.success = False
            result.done = False
            result.message = outcome.message
            return

        if cancelled():
            raise skill_interface.SkillCancelledError("move-to-board cancelled")
        snapshot = self.camera_rig.grab(timeout_sec=timeout_sec)
        if snapshot is None or not snapshot.frames:
            result.success = False
            result.message = "no fresh wrist-camera image after 2D center move"
            return

        reports = self._scan_cameras(snapshot)
        result.num_cameras = len(snapshot.frames)
        result.force_n = float(snapshot.force_norm or 0.0)
        self._fill_camera_summaries(snapshot, reports, result)
        camera, report = pick_best_camera(reports)
        if camera is None or report is None:
            result.success = False
            result.done = False
            result.message = "lost board after 2D center move"
            return
        if board_view_ready(report):
            result.success = True
            result.done = True
            result.message = (
                f"board framed on {camera} after 2D center move "
                f"(area={report.area_frac:.3f})"
            )
            logging.info(result.message)
            return

        result.success = True
        result.done = False
        result.message = (
            f"2D center move done on {camera}; still not framed "
            f"(edges={','.join(sorted(report.edges)) or 'none'}, "
            f"area={report.area_frac:.3f})"
        )
        logging.info(result.message)

    # SkillRepository plumbing required by the SDK Python skill service.
    def configure_runtime(self, service_config) -> None:
        from intrinsic.skills.internal import runtime_data
        from aic_perception import move_to_board_skill_pb2 as pb2

        self._skill_alias = service_config.skill_description.skill_name
        self._runtime_data = runtime_data.get_runtime_data_from(
            service_config,
            pb2.MoveToBoardSkillParams.DESCRIPTOR,
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
    """Start the standard Intrinsic Python skill gRPC server."""
    logging.info("MoveToBoardSkill service starting")
    from intrinsic.skills.internal import skill_service_impl
    from intrinsic.skills.proto import skill_service_config_pb2
    from intrinsic.skills.proto import skill_service_pb2_grpc

    if not FLAGS.skill_service_config_filename:
        raise ValueError("--skill_service_config_filename is required")
    service_config = skill_service_config_pb2.SkillServiceConfig()
    with open(FLAGS.skill_service_config_filename, "rb") as config_file:
        service_config.ParseFromString(config_file.read())

    skill_instance = MoveToBoardSkill()
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
        logging.info("MoveToBoardSkill stopped")


if __name__ == "__main__":
    app.run(start_runner, flags_parser=lambda argv: FLAGS(argv, known_only=True))
