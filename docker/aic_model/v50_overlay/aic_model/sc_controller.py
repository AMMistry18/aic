"""Scripted SC (duplex fibre) insertion for the ``insert_cable`` skill.

This is the SC counterpart to :mod:`v50_controller`.  It reuses that module's
proven, geometry-agnostic seating primitives -- persistent force-lead setpoint,
wall-time stall detection, bounded overtravel, and the proportional wrench
alignment law -- but every axial constant is re-derived for SC, because SC seats
at 15.64 mm against SFP's 45.8 mm and a naive constant copy would command the
plug straight through the back of the port on the first stall.

It is deliberately *fixed-grasp*: unlike v50 there is no SC plug-pose model, so
the tip is located from the TCP through a static transform rather than from
vision.  See "UNCALIBRATED" below.

Ground truth, all derived from the shipped assets (not measured, not guessed):

``aic_assets/models/SC Port/model.sdf``
  * ``sc_port_base_link_entrance`` sits 15.64 mm along -Z of ``sc_port_base_link``
    => 15.64 mm mouth-to-seated.
  * ``sc_port_base_link`` is posed ``0 -0.002 0`` rpy ``(pi/2, pi, 0)`` relative
    to ``sc_port_link``, so the seat point is 2 mm behind the port origin and the
    insertion axis is ``sc_port_link`` -Y.
  * Opening walls: side walls centred at x=+/-12.047 mm (1.687 mm thick) give
    inner faces at +/-11.204 mm; the top/bottom plates at z=+/-4.35 mm (0.6 mm
    thick) give inner faces at +/-4.05 mm.  The duplex opening is therefore
    22.41 mm x 8.10 mm, split by a 2.99 mm centre divider into two 9.71 mm bores
    on a 12.7 mm pitch (the standard SC duplex pitch -- a useful sanity check
    that this reading of the collision set is right).

``aic_description/urdf/task_board.urdf.xacro``
  * SC ports are posed rpy ``(1.57 + roll, pitch, 1.57 + yaw)`` on the board, so
    ``R_board_from_port = Rz(90) Rx(90) = [[0,0,1],[1,0,0],[0,1,0]]``.
  * Composing that with the port-base rotation above maps the insertion axis to
    board **-Z**: SC descends straight down, exactly like SFP.  This is why the
    SFP entrance-frame estimator (which hardcodes "insertion axis is world -Z")
    is reusable here unchanged.
  * ``sc_port_link`` +X maps to board +Y, so the duplex opening's long axis lies
    along board Y.
  * The two board slots sit at y=0.0295 and y=0.0705, i.e. **41 mm apart** --
    nearly double the SFP cage pitch, so nearest-tip port selection has far more
    margin here than it does for SFP.

``aic_model/aic_model/sc_plug_pose_geometry.py``
  * The plug body is 20.0 mm x 6.4 mm.  Against the 22.41 x 8.10 mm opening that
    is 1.2 mm of lateral and 0.85 mm of vertical clearance per side -- a much
    looser fit than SFP, which is the main reason a script is plausible here.

TWO THINGS ARE UNCALIBRATED AND WILL NOT WORK UNTIL RESOLVED:

1. ``SC_TIP_IN_TCP_*`` defaults to the SFP grasp transform.  It is almost
   certainly wrong -- it is the same gripper and cable but the other connector.
   Re-solve it with ``RL_INSERT_CALIB_DUMP=1`` exactly as the SFP transform was
   solved, then set ``RL_INSERT_SC_TIP_IN_TCP_POS`` / ``_QUAT``.
2. The keypoint convention of the legacy ``best_sc_pose.pt`` model is not
   recorded anywhere in this repo, and ``sc_plug_pose_geometry`` states its
   keypoints are "unrelated" to it.  Rather than guess, this module *measures*
   the triangulated keypoint rectangle at runtime and logs which convention it
   matches (see ``classify_opening``).  Read one run's ``SC_OPENING`` lines
   before trusting any pose it produces.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .rl_insert_contract import (
    SFP_TIP_IN_TCP_POS,
    SFP_TIP_IN_TCP_QUAT,
    quat_to_matrix,
)
from .v50_controller import (
    HARD_FAILURE,
    SEATED,
    STALLED,
    WallProgressWatch,
    axis_angle,
    clamp_vector_norm,
    rotation_from_axis_angle,
)


# Deliberately local rather than imported from v50_controller: these are that
# module's private helpers, and the SFP build is mid-deploy.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalize_event(value: object) -> str:
    """Reduce an insertion-event name to its trailing ``module/port``.

    The scoring topic publishes e.g. ``cable_0#0#nic_card_mount_0/sfp_port_1``
    while the task names only ``nic_card_mount_0/sfp_port_0``.  v50's normaliser
    strips whitespace and slashes but not the ``cable_N#0#`` prefix, so its
    equality test cannot match even on a correct port; strip the prefix here.
    """
    text = str(value or "").strip().strip("/")
    return text.rsplit("#", 1)[-1]


def _env_vector(name: str, default: np.ndarray) -> np.ndarray:
    raw = os.environ.get(name)
    if not raw:
        return np.asarray(default, dtype=np.float64).copy()
    try:
        value = np.array([float(part) for part in raw.replace(",", " ").split()],
                         dtype=np.float64)
    except ValueError:
        return np.asarray(default, dtype=np.float64).copy()
    if value.shape != np.asarray(default).shape:
        return np.asarray(default, dtype=np.float64).copy()
    return value


# --------------------------------------------------------------------------
# Geometry derived from the assets (see module docstring for provenance).
# --------------------------------------------------------------------------
SC_INSERT_DEPTH_M = 0.01564          # sc_port_base_link_entrance -> sc_port_base_link
SC_OPENING_WIDTH_M = 0.02241         # duplex inner width  (port +X / board +Y)
SC_OPENING_HEIGHT_M = 0.00810        # duplex inner height (port +Z / board +X)
SC_BORE_WIDTH_M = 0.00971            # one bore, between side wall and divider
SC_BORE_PITCH_M = 0.01270            # standard SC duplex pitch
SC_PLUG_WIDTH_M = 0.02000            # from sc_plug_pose_geometry keypoints
SC_PLUG_HEIGHT_M = 0.00640

# The four YOLO keypoints may bound either the whole duplex opening or a single
# bore.  Both hypotheses are checked at runtime; ``classify_opening`` returns the
# label and the lateral offset that maps the perceived centroid onto the duplex
# centre (which is what the duplex plug actually inserts into).
SC_OPENING_HYPOTHESES = (
    ("duplex", SC_OPENING_WIDTH_M, SC_OPENING_HEIGHT_M),
    ("single_bore", SC_BORE_WIDTH_M, SC_OPENING_HEIGHT_M),
)

# Local rectangle for the single-view PnP fallback, in the same corner order the
# SFP estimator uses (KP0 top-left, KP1 top-right, KP2 bottom-right, KP3
# bottom-left).  Defaults to the duplex hypothesis; override if a run's
# SC_OPENING lines say otherwise.
LOCAL_SC_PORT_KPS = np.array(
    [
        [+SC_OPENING_WIDTH_M / 2.0, +SC_OPENING_HEIGHT_M / 2.0, 0.0],
        [-SC_OPENING_WIDTH_M / 2.0, +SC_OPENING_HEIGHT_M / 2.0, 0.0],
        [-SC_OPENING_WIDTH_M / 2.0, -SC_OPENING_HEIGHT_M / 2.0, 0.0],
        [+SC_OPENING_WIDTH_M / 2.0, -SC_OPENING_HEIGHT_M / 2.0, 0.0],
    ],
    dtype=np.float64,
)

# UNCALIBRATED -- see module docstring item 1.
SC_TIP_IN_TCP_POS = _env_vector("RL_INSERT_SC_TIP_IN_TCP_POS", SFP_TIP_IN_TCP_POS)
SC_TIP_IN_TCP_QUAT = _env_vector("RL_INSERT_SC_TIP_IN_TCP_QUAT", SFP_TIP_IN_TCP_QUAT)
SC_TIP_CALIBRATED = _env_bool("RL_INSERT_SC_TIP_CALIBRATED", False)

SC_WRENCH_LOG_PERIOD_S = _env_float("RL_INSERT_SC_WRENCH_LOG_PERIOD_S", 0.25)
SC_PERCEPT_SAMPLES = _env_int("RL_INSERT_SC_PERCEPT_SAMPLES", 7)
SC_PERCEPT_MIN_AGREE = _env_int("RL_INSERT_SC_PERCEPT_MIN_AGREE", 3)
SC_PERCEPT_SAMPLE_DT = _env_float("RL_INSERT_SC_PERCEPT_SAMPLE_DT", 0.10)
SC_PERCEPT_AGREE_TOL_M = _env_float("RL_INSERT_SC_PERCEPT_AGREE_TOL_M", 0.004)
SC_MAX_PORT_REPROJ_PX = _env_float("RL_INSERT_SC_MAX_PORT_REPROJ_PX", 6.0)
SC_MAX_SELECT_REPROJ_PX = _env_float("RL_INSERT_SC_MAX_SELECT_REPROJ_PX", 5.0)
# The board slots are 41 mm apart, so a generous handoff gate still cannot
# select the neighbouring port.
SC_MAX_HANDOFF_SELECT_M = _env_float("RL_INSERT_SC_MAX_HANDOFF_SELECT_M", 0.030)
SC_HANDOFF_MAX_DIST_M = _env_float("RL_INSERT_SC_HANDOFF_MAX_DIST_M", 0.120)
# Dimensional gate on the triangulated rectangle, wide enough to admit either
# keypoint-convention hypothesis and reject anything that is not an SC opening.
SC_MIN_OPENING_M = _env_float("RL_INSERT_SC_MIN_OPENING_M", 0.005)
SC_MAX_OPENING_M = _env_float("RL_INSERT_SC_MAX_OPENING_M", 0.030)


@dataclass
class SCConfig:
    """SC seating limits.

    Every axial value is re-derived for the 15.64 mm bore.  Where a value is a
    fraction of the SFP equivalent the fraction is stated, because the SFP
    numbers are the only ones with field evidence behind them.
    """

    command_dt_sim_s: float = 0.05
    align_timeout_wall_s: float = 15.0
    align_lateral_tol_m: float = 0.001
    align_rotation_tol_rad: float = np.deg2rad(2.0)
    align_max_lateral_step_m: float = 0.0015
    align_max_rotation_step_rad: float = np.deg2rad(1.5)

    # 0.8 mm on a 45.8 mm bore is 1.7%; the same fraction here is 0.27 mm, which
    # is near TF noise, so this sits above the proportional value deliberately.
    stall_progress_m: float = 0.0005
    stall_timeout_wall_s: float = 2.5
    seat_stall_grace_s: float = 1.5

    # SC has a lighter spring latch than the SFP cage, and the whole insertion is
    # a third as deep, so the force ladder is scaled down across the board.
    contact_force_n: float = 2.0
    target_axial_force_n: float = 5.0
    seat_force_cap_n: float = 7.0
    force_abort_n: float = 12.0
    force_abort_wall_s: float = 0.25
    axial_stiffness_n_m: float = 500.0

    # THE constant that must not be copied from SFP: 8 N / 500 N/m = 16 mm of
    # axial lead is longer than the entire SC insertion.  Capping the lead at
    # 5 mm (32% of the bore) keeps the impedance demand bounded without the
    # setpoint ever running past the back of the port.
    max_axial_lead_m: float = 0.005

    free_speed_m_s: float = 0.008     # halved vs SFP; SC approach must stay slow
    contact_speed_m_s: float = 0.004

    # Clearance is 1.2 mm/side laterally, so the safety envelope is tighter than
    # SFP's 6 mm in absolute terms but looser relative to the fit.
    lateral_safety_m: float = 0.003
    rotation_safety_rad: float = np.deg2rad(10.0)

    # Proportional wrench alignment, same law as the fixed v50 one: a low-pass of
    # a clamped proportional target, NOT an accumulator.  The clamp is slightly
    # larger than SFP's because the SC fit is looser.
    seat_align_enable: bool = True
    seat_align_force_gain: float = 0.00003
    seat_align_moment_gain: float = 0.004
    seat_align_max_lat_m: float = 0.0006
    seat_align_max_tilt_rad: float = 0.0087
    seat_align_release_decay: float = 0.7

    seat_mouth_zone_m: float = 0.002          # 13% of bore, as SFP's 6 mm is
    seat_mouth_speed_scale: float = 0.25
    seat_overtravel_m: float = 0.0015         # 10% of bore, as SFP's 5 mm is
    seat_candidate_depth_m: float = 0.0152    # 97% of bore, as SFP's 44.5 mm is
    insertion_event_timeout_wall_s: float = 6.0

    @property
    def force_lead_m(self) -> float:
        return min(
            self.max_axial_lead_m,
            self.target_axial_force_n / self.axial_stiffness_n_m,
        )

    @classmethod
    def from_env(cls) -> "SCConfig":
        return cls(
            command_dt_sim_s=_env_float("RL_INSERT_SC_COMMAND_DT_S", 0.05),
            align_timeout_wall_s=_env_float("RL_INSERT_SC_ALIGN_TIMEOUT_S", 15.0),
            stall_timeout_wall_s=_env_float("RL_INSERT_SC_STALL_TIMEOUT_S", 2.5),
            stall_progress_m=_env_float("RL_INSERT_SC_STALL_PROGRESS_M", 0.0005),
            free_speed_m_s=_env_float("RL_INSERT_SC_FREE_SPEED_M_S", 0.008),
            contact_speed_m_s=_env_float("RL_INSERT_SC_CONTACT_SPEED_M_S", 0.004),
            contact_force_n=_env_float("RL_INSERT_SC_CONTACT_FORCE_N", 2.0),
            target_axial_force_n=_env_float("RL_INSERT_SC_TARGET_FORCE_N", 5.0),
            seat_force_cap_n=_env_float("RL_INSERT_SC_SEAT_FORCE_CAP_N", 7.0),
            force_abort_n=_env_float("RL_INSERT_SC_FORCE_ABORT_N", 12.0),
            axial_stiffness_n_m=_env_float("RL_INSERT_SC_AXIAL_STIFFNESS_N_M", 500.0),
            max_axial_lead_m=_env_float("RL_INSERT_SC_MAX_AXIAL_LEAD_M", 0.005),
            seat_align_enable=_env_bool("RL_INSERT_SC_SEAT_ALIGN_ENABLE", True),
            seat_align_force_gain=_env_float("RL_INSERT_SC_SEAT_ALIGN_FORCE_GAIN", 0.00003),
            seat_align_moment_gain=_env_float("RL_INSERT_SC_SEAT_ALIGN_MOMENT_GAIN", 0.004),
            seat_align_max_lat_m=_env_float("RL_INSERT_SC_SEAT_ALIGN_MAX_LAT_M", 0.0006),
            seat_align_max_tilt_rad=_env_float("RL_INSERT_SC_SEAT_ALIGN_MAX_TILT_RAD", 0.0087),
            seat_align_release_decay=_env_float("RL_INSERT_SC_SEAT_ALIGN_RELEASE_DECAY", 0.7),
            seat_mouth_zone_m=_env_float("RL_INSERT_SC_SEAT_MOUTH_ZONE_M", 0.002),
            seat_mouth_speed_scale=_env_float("RL_INSERT_SC_SEAT_MOUTH_SPEED_SCALE", 0.25),
            seat_stall_grace_s=_env_float("RL_INSERT_SC_SEAT_STALL_GRACE_S", 1.5),
            seat_overtravel_m=_env_float("RL_INSERT_SC_SEAT_OVERTRAVEL_M", 0.0015),
            seat_candidate_depth_m=_env_float("RL_INSERT_SC_SEAT_CANDIDATE_DEPTH_M", 0.0152),
            insertion_event_timeout_wall_s=_env_float("RL_INSERT_SC_EVENT_TIMEOUT_S", 6.0),
        ).validated()

    def validated(self) -> "SCConfig":
        if not 0.0 < self.target_axial_force_n < self.seat_force_cap_n:
            raise ValueError("sc target force must be below the seat force cap")
        if not self.seat_force_cap_n < self.force_abort_n <= 18.0:
            raise ValueError("sc force cap must be below the <=18 N hard abort")
        if self.axial_stiffness_n_m <= 0.0:
            raise ValueError("sc axial stiffness must be positive")
        # The whole point of the SC rescale: a lead longer than the bore would
        # command the plug through the back of the port on the first stall.
        if not 0.0 < self.force_lead_m < SC_INSERT_DEPTH_M:
            raise ValueError(
                f"sc axial lead {self.force_lead_m*1000:.1f}mm must stay inside the "
                f"{SC_INSERT_DEPTH_M*1000:.2f}mm bore"
            )
        if not 0.0 <= self.seat_overtravel_m <= 0.003:
            raise ValueError("sc seat overtravel must stay within 0-3 mm")
        if not 0.0 < self.seat_candidate_depth_m <= SC_INSERT_DEPTH_M:
            raise ValueError("sc candidate depth must lie inside the bore")
        if self.seat_align_max_lat_m < 0.0 or self.seat_align_max_tilt_rad < 0.0:
            raise ValueError("sc alignment correction caps must be non-negative")
        if not 0.0 <= self.seat_align_release_decay <= 1.0:
            raise ValueError("sc alignment release decay must be within 0-1")
        if self.seat_mouth_zone_m < 0.0 or self.seat_mouth_speed_scale < 0.0:
            raise ValueError("sc mouth slowdown parameters must be non-negative")
        return self


def sc_tip_pose_from_tcp(tcp_pos, tcp_quat):
    """Fixed-grasp tip pose.  See UNCALIBRATED item 1 in the module docstring."""
    R_tcp = quat_to_matrix(tcp_quat)
    R_tip = R_tcp @ quat_to_matrix(SC_TIP_IN_TCP_QUAT)
    tip_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(3) + R_tcp @ SC_TIP_IN_TCP_POS
    return tip_pos, R_tip


def tcp_pose_for_sc_tip(tip_pos, tip_rotation):
    R_tip = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    R_tcp = R_tip @ quat_to_matrix(SC_TIP_IN_TCP_QUAT).T
    tcp_pos = np.asarray(tip_pos, dtype=np.float64).reshape(3) - R_tcp @ SC_TIP_IN_TCP_POS
    return tcp_pos, R_tcp


def next_sc_depth(
    current_depth: float,
    commanded_depth: float,
    elapsed_wall_s: float,
    force_n: float,
    config: SCConfig,
) -> float:
    """Advance the absolute seat setpoint, bounded by the SC bore.

    Deliberately a separate function from ``v50_controller.next_persistent_depth``
    rather than a parameterisation of it: that one closes over the SFP
    ``INSERT_DEPTH_M`` module constant, and the SFP build is mid-deploy.
    """
    current_depth = float(current_depth)
    commanded_depth = max(float(commanded_depth), current_depth)
    if np.isfinite(force_n) and force_n >= config.seat_force_cap_n:
        candidate = commanded_depth
    else:
        speed = (
            config.contact_speed_m_s
            if np.isfinite(force_n) and force_n >= config.contact_force_n
            else config.free_speed_m_s
        )
        candidate = commanded_depth + speed * max(0.0, float(elapsed_wall_s))
    return min(
        SC_INSERT_DEPTH_M + config.seat_overtravel_m,
        candidate,
        current_depth + config.force_lead_m,
    )


def classify_opening(width_m: float, height_m: float):
    """Name the keypoint convention a triangulated rectangle matches.

    Returns ``(label, residual_m, lateral_offset_m)``.  ``lateral_offset_m`` is
    the shift, along the opening's long (+X) axis, from the perceived centroid to
    the duplex centre that the duplex plug actually enters -- zero for the duplex
    hypothesis, and half a bore pitch for the single-bore one, though its sign
    cannot be known without also knowing which bore was detected.

    This exists because the keypoint convention of ``best_sc_pose.pt`` is not
    recorded anywhere; measuring it beats guessing it.
    """
    best = None
    for label, expect_w, expect_h in SC_OPENING_HYPOTHESES:
        residual = abs(width_m - expect_w) + abs(height_m - expect_h)
        if best is None or residual < best[1]:
            best = (label, residual)
    label, residual = best
    offset = 0.0 if label == "duplex" else SC_BORE_PITCH_M / 2.0
    return label, float(residual), float(offset)


class ScInsertionController:
    """Align to the perceived SC opening, then force-regulate to the seat."""

    STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
    DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
    HOLD_STIFFNESS = [200.0, 200.0, 200.0, 80.0, 80.0, 80.0]
    HOLD_DAMPING = [80.0, 80.0, 80.0, 30.0, 30.0, 30.0]

    def __init__(self, policy, task, get_observation, move_robot, send_feedback,
                 *, port_pos, port_quat, Rp, config=None):
        self.policy = policy
        self.task = task
        self.get_observation = get_observation
        self.move_robot = move_robot
        self.send_feedback = send_feedback
        self.config = (config or SCConfig.from_env()).validated()
        self.port_pos = np.asarray(port_pos, dtype=np.float64).reshape(3)
        self.port_quat = np.asarray(port_quat, dtype=np.float64).reshape(4)
        self.Rp = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
        self.log = policy.get_logger()
        parent = policy._parent_node
        self.event_generation = int(getattr(parent, "_insertion_event_generation", 0))
        self.expected_event = _normalize_event(
            f"{getattr(task, 'target_module_name', '')}/{getattr(task, 'port_name', '')}"
        )

    # ------------------------------------------------------------- geometry
    def _tip_pose(self):
        tcp_pos, tcp_quat = self.policy._tcp()
        return sc_tip_pose_from_tcp(tcp_pos, tcp_quat)

    def _tcp_target(self, tip_pos, tip_rotation):
        # geometry_msgs is imported lazily so the pure geometry in this module
        # stays importable (and unit-testable) without a ROS environment.
        from geometry_msgs.msg import Point, Pose, Quaternion

        from .rl_insert_contract import matrix_to_quat

        tcp_pos, R_tcp = tcp_pose_for_sc_tip(tip_pos, tip_rotation)
        q_tcp = matrix_to_quat(R_tcp)
        return Pose(
            position=Point(x=float(tcp_pos[0]), y=float(tcp_pos[1]), z=float(tcp_pos[2])),
            orientation=Quaternion(
                w=float(q_tcp[0]), x=float(q_tcp[1]),
                y=float(q_tcp[2]), z=float(q_tcp[3])),
        )

    def _errors(self):
        tip_pos, tip_rotation = self._tip_pose()
        delta = self.Rp.T @ (tip_pos - self.port_pos)
        rotation_error = axis_angle(self.Rp.T @ tip_rotation)
        return float(delta[2]), delta[:2], rotation_error, tip_pos, tip_rotation

    def _hold_tip(self, tip_pos, tip_rotation) -> None:
        self.policy.set_pose_target(
            self.move_robot,
            self._tcp_target(tip_pos, tip_rotation),
            stiffness=self.HOLD_STIFFNESS,
            damping=self.HOLD_DAMPING,
        )

    # --------------------------------------------------------------- wrench
    def _force_magnitude(self, observation) -> float:
        if observation is None:
            return float("nan")
        wrench = self.policy._wrench_vector(observation) - self.policy._wrench_baseline
        return float(np.linalg.norm(wrench[:3]))

    def _wrench_plug_frame(self, observation):
        if observation is None:
            nan3 = np.full(3, np.nan, dtype=np.float64)
            return nan3, nan3.copy()
        wrench_wrist = (
            self.policy._wrench_vector(observation) - self.policy._wrench_baseline
        )
        _, R_tip = self._tip_pose()
        wrist_to_plug = self.Rp.T @ R_tip
        return wrist_to_plug @ wrench_wrist[:3], wrist_to_plug @ wrench_wrist[3:]

    def _alignment_sample(self, observation, depth, force, acc_lat, acc_tilt):
        """Low-passed proportional wrench correction -- not an accumulator.

        Identical in form to the fixed v50 law.  The SFP field logs showed an
        accumulator saturates at its clamp within a few samples of first chamfer
        contact and then jams the plug; do not reintroduce one here.
        """
        f_plug, m_plug = self._wrench_plug_frame(observation)
        contact = bool(np.isfinite(force) and force >= self.config.contact_force_n)
        finite = bool(np.all(np.isfinite(f_plug[:2])) and np.all(np.isfinite(m_plug[:2])))
        if contact and finite:
            target_lat = clamp_vector_norm(
                -self.config.seat_align_force_gain * f_plug[:2],
                self.config.seat_align_max_lat_m,
            )
            target_tilt = clamp_vector_norm(
                -self.config.seat_align_moment_gain * m_plug[:2],
                self.config.seat_align_max_tilt_rad,
            )
        else:
            target_lat = np.zeros(2, dtype=np.float64)
            target_tilt = np.zeros(2, dtype=np.float64)
        decay = self.config.seat_align_release_decay
        acc_lat = clamp_vector_norm(
            decay * np.asarray(acc_lat, dtype=np.float64).reshape(2)
            + (1.0 - decay) * target_lat,
            self.config.seat_align_max_lat_m,
        )
        acc_tilt = clamp_vector_norm(
            decay * np.asarray(acc_tilt, dtype=np.float64).reshape(2)
            + (1.0 - decay) * target_tilt,
            self.config.seat_align_max_tilt_rad,
        )
        return acc_lat, acc_tilt, (depth, f_plug, m_plug)

    def _log_wrench(self, sample, acc_lat, acc_tilt, *, summary: Optional[str] = None):
        depth, f_plug, m_plug = sample
        suffix = f" summary={summary}" if summary else ""
        self.log.info(
            f"[sc] SEAT_WRENCH depth={depth*1000.0:.2f}mm "
            f"axial_N={f_plug[2]:.2f} "
            f"lat_N={np.round(f_plug[:2], 2).tolist()} "
            f"|lat|={np.linalg.norm(f_plug[:2]):.2f} "
            f"|M|={np.linalg.norm(m_plug[:2]):.3f} "
            f"nudge_applied_mm={np.round(acc_lat * 1000.0, 3).tolist()} "
            f"tilt_applied_deg={np.degrees(np.linalg.norm(acc_tilt)):.3f}"
            f"{suffix}"
        )

    # ---------------------------------------------------------------- event
    def _event_status(self):
        parent = self.policy._parent_node
        generation = int(getattr(parent, "_insertion_event_generation", 0))
        if generation <= self.event_generation:
            return None
        value = _normalize_event(getattr(parent, "_insertion_event_value", ""))
        if value == self.expected_event:
            return SEATED
        self.log.error(
            f"[sc] insertion event was for wrong port '{value}', expected "
            f"'{self.expected_event}'"
        )
        return HARD_FAILURE

    def _wait_for_insertion_event(self, fixed_tip) -> str:
        deadline = time.monotonic() + self.config.insertion_event_timeout_wall_s
        hard_force_since = None
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            status = self._event_status()
            if status is not None:
                return status
            observation = self.get_observation()
            force = self._force_magnitude(observation)
            tip_pos, _ = self._tip_pose()
            if np.isfinite(force) and force > self.config.force_abort_n:
                self._hold_tip(tip_pos, self.Rp)
                hard_force_since = hard_force_since or time.monotonic()
                if time.monotonic() - hard_force_since >= self.config.force_abort_wall_s:
                    self.log.error(
                        f"[sc] >{self.config.force_abort_n:.1f}N during event dwell"
                    )
                    return HARD_FAILURE
            else:
                hard_force_since = None
                self.policy.set_pose_target(
                    self.move_robot,
                    self._tcp_target(fixed_tip, self.Rp),
                    stiffness=self.STIFFNESS,
                    damping=self.DAMPING,
                )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.warn("[sc] seated geometry produced no matching insertion event")
        return STALLED

    # ----------------------------------------------------------------- run
    def _align(self) -> bool:
        start = time.monotonic()
        depth, _, _, _, _ = self._errors()
        align_depth = depth
        while time.monotonic() - start < self.config.align_timeout_wall_s:
            self.policy._enforce_action_deadline(self.move_robot)
            depth, lateral_xy, rotation_error, tip_pos, _ = self._errors()
            lateral = float(np.linalg.norm(lateral_xy))
            rotation = float(np.linalg.norm(rotation_error))
            if (
                lateral <= self.config.align_lateral_tol_m
                and rotation <= self.config.align_rotation_tol_rad
            ):
                self.log.info(
                    f"[sc] aligned: lateral={lateral*1000:.2f}mm "
                    f"rot={np.degrees(rotation):.2f}deg depth={depth*1000:.2f}mm"
                )
                return True
            # Bounded increments from the CURRENT tip, as v50 does. Commanding
            # the corrected pose outright is a large uncommanded motion at the
            # mouth of an 8 mm opening.
            lateral_step = -lateral_xy
            lateral_norm = float(np.linalg.norm(lateral_step))
            if lateral_norm > self.config.align_max_lateral_step_m:
                lateral_step *= self.config.align_max_lateral_step_m / lateral_norm
            target_tip = (
                tip_pos
                + self.Rp[:, 0] * lateral_step[0]
                + self.Rp[:, 1] * lateral_step[1]
                + self.Rp[:, 2] * float(np.clip(align_depth - depth, -0.001, 0.001))
            )
            if rotation > self.config.align_max_rotation_step_rad:
                remaining = 1.0 - self.config.align_max_rotation_step_rad / rotation
                target_rotation = self.Rp @ rotation_from_axis_angle(remaining * rotation_error)
            else:
                target_rotation = self.Rp
            self.policy.set_pose_target(
                self.move_robot,
                self._tcp_target(target_tip, target_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.error(
            f"[sc] alignment did not converge in {self.config.align_timeout_wall_s:.1f}s"
        )
        return False

    def _seat(self) -> str:
        depth, _, _, _, _ = self._errors()
        command_depth = depth
        now = time.monotonic()
        last_command_time = now
        progress = WallProgressWatch.start(depth, now, self.config)
        hard_force_since = None
        acc_lat = np.zeros(2, dtype=np.float64)
        acc_tilt = np.zeros(2, dtype=np.float64)
        last_wrench_log_time = 0.0
        stall_grace_deadline = None

        while True:
            self.policy._enforce_action_deadline(self.move_robot)
            status = self._event_status()
            if status is not None:
                return status
            observation = self.get_observation()
            depth, lateral_xy, rotation_error, tip_pos, _ = self._errors()
            lateral = float(np.linalg.norm(lateral_xy))
            rotation = float(np.linalg.norm(rotation_error))
            force = self._force_magnitude(observation)
            now = time.monotonic()

            sample = None
            if self.config.seat_align_enable:
                acc_lat, acc_tilt, sample = self._alignment_sample(
                    observation, depth, force, acc_lat, acc_tilt
                )
                if now - last_wrench_log_time >= SC_WRENCH_LOG_PERIOD_S:
                    self._log_wrench(sample, acc_lat, acc_tilt)
                    last_wrench_log_time = now

            if np.isfinite(force) and force > self.config.force_abort_n:
                self._hold_tip(tip_pos, self.Rp)
                hard_force_since = hard_force_since or now
                if now - hard_force_since >= self.config.force_abort_wall_s:
                    self.log.error(
                        f"[sc] sustained force {force:.1f}N exceeds "
                        f"{self.config.force_abort_n:.1f}N; held and aborted"
                    )
                    return HARD_FAILURE
                self.policy.sleep_for(self.config.command_dt_sim_s)
                continue
            hard_force_since = None

            if lateral > self.config.lateral_safety_m or rotation > self.config.rotation_safety_rad:
                self.log.warn(
                    f"[sc] wedge geometry: lateral={lateral*1000:.2f}mm "
                    f"rotation={np.degrees(rotation):.1f}deg"
                )
                if sample is not None:
                    self._log_wrench(sample, acc_lat, acc_tilt, summary="stall")
                return STALLED

            if depth >= self.config.seat_candidate_depth_m:
                fixed_tip = self.port_pos + self.Rp[:, 2] * SC_INSERT_DEPTH_M
                return self._wait_for_insertion_event(fixed_tip)

            stalled_now = progress.stalled(depth, now)
            if stall_grace_deadline is not None and not stalled_now:
                self.log.info(
                    f"[sc] stall grace recovered: depth={depth*1000:.2f}mm "
                    f"best={progress.best_depth*1000:.2f}mm"
                )
                stall_grace_deadline = None
            if stalled_now:
                if self.config.seat_stall_grace_s > 0.0 and stall_grace_deadline is None:
                    stall_grace_deadline = now + self.config.seat_stall_grace_s
                    self.log.warn(
                        f"[sc] wall-time stall grace: depth={depth*1000:.2f}mm "
                        f"best={progress.best_depth*1000:.2f}mm force={force:.2f}N"
                    )
                if (
                    self.config.seat_stall_grace_s <= 0.0
                    or now >= stall_grace_deadline
                ):
                    self.log.warn(
                        f"[sc] wall-time stall: depth={depth*1000:.2f}mm "
                        f"best={progress.best_depth*1000:.2f}mm force={force:.2f}N"
                    )
                    if sample is not None:
                        self._log_wrench(sample, acc_lat, acc_tilt, summary="stall")
                    return STALLED

            depth_config = self.config
            if (
                self.config.seat_mouth_speed_scale != 1.0
                and depth < self.config.seat_mouth_zone_m
            ):
                depth_config = replace(
                    self.config,
                    free_speed_m_s=self.config.free_speed_m_s * self.config.seat_mouth_speed_scale,
                    contact_speed_m_s=self.config.contact_speed_m_s * self.config.seat_mouth_speed_scale,
                )
            command_depth = next_sc_depth(
                depth, command_depth, now - last_command_time, force, depth_config
            )
            last_command_time = now

            target_tip = self.port_pos + self.Rp[:, 2] * command_depth
            target_rotation = self.Rp
            if self.config.seat_align_enable and (
                np.any(acc_lat != 0.0) or np.any(acc_tilt != 0.0)
            ):
                target_tip = (
                    target_tip + self.Rp[:, 0] * acc_lat[0] + self.Rp[:, 1] * acc_lat[1]
                )
                target_rotation = self.Rp @ rotation_from_axis_angle(
                    np.array([acc_tilt[0], acc_tilt[1], 0.0], dtype=np.float64)
                )
            self.policy.set_pose_target(
                self.move_robot,
                self._tcp_target(target_tip, target_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)

    def run(self) -> bool:
        if not SC_TIP_CALIBRATED:
            self.log.warn(
                "[sc] SC_TIP_IN_TCP is the UNCALIBRATED SFP default -- re-solve it "
                "with RL_INSERT_CALIB_DUMP=1 and set RL_INSERT_SC_TIP_IN_TCP_POS/_QUAT "
                "plus RL_INSERT_SC_TIP_CALIBRATED=1 before trusting this run"
            )
        self.send_feedback("sc align to perceived opening")
        if not self._align():
            return False
        self.send_feedback("sc force-regulated seating")
        outcome = self._seat()
        if outcome == SEATED:
            self.log.info(f"[sc] insertion event confirmed for {self.expected_event}")
            return True
        self.log.warn(f"[sc] seating ended without confirmation: {outcome}")
        return False


# --------------------------------------------------------------------------
# Perception: SC opening pose from the legacy best_sc_pose.pt keypoints.
# --------------------------------------------------------------------------
def sc_multiview_candidates(policy, per_cam):
    """Triangulate the four SC keypoints across cameras.

    Mirrors the SFP flow.  The orientation estimator is reused verbatim: it
    assumes the insertion axis is world -Z, which the asset geometry confirms is
    true for SC as well (see module docstring).
    """
    cams = [cam for cam, dets in per_cam.items() if dets]
    if len(cams) < 2:
        return []
    for cam in cams:
        per_cam[cam] = per_cam[cam][:5]

    candidates = []
    for picks in itertools.product(*[per_cam[cam] for cam in cams]):
        kp_3d = []
        try:
            for i in range(4):
                pts_2d = [tuple(pick["kps"][i]) for pick in picks]
                Ps = [pick["P"] for pick in picks]
                kp_3d.append(policy._pc.triangulate(pts_2d, Ps))
        except Exception:
            continue
        kp_3d = np.array(kp_3d, dtype=np.float64)
        X = kp_3d.mean(axis=0)
        if X[2] < -0.05 or X[2] > 0.25:
            continue

        q_wxyz, yaw = policy._estimate_sfp_port_orientation(kp_3d)
        if q_wxyz is None:
            continue

        width = float(np.linalg.norm(((kp_3d[0] + kp_3d[3]) * 0.5) - ((kp_3d[1] + kp_3d[2]) * 0.5)))
        height = float(np.linalg.norm(((kp_3d[0] + kp_3d[1]) * 0.5) - ((kp_3d[2] + kp_3d[3]) * 0.5)))
        if not (SC_MIN_OPENING_M <= width <= SC_MAX_OPENING_M):
            continue
        if not (SC_MIN_OPENING_M <= height <= SC_MAX_OPENING_M):
            continue

        errors = []
        for pick in picks:
            for i in range(4):
                err = policy._reproject_error_px(kp_3d[i], pick["K"], pick["T"], pick["kps"][i])
                if err is not None:
                    errors.append(err)
        if not errors:
            continue
        reproj = float(np.mean(errors))
        label, residual, offset = classify_opening(width, height)
        score = reproj + residual * 250.0 - 0.02 * float(
            np.mean([pick.get("conf", 0.0) for pick in picks])
        )
        candidates.append({
            "X": X, "kp_3d": kp_3d, "q_wxyz": q_wxyz, "yaw": yaw,
            "score": float(score), "reproj_px": reproj,
            "width": width, "height": height,
            "opening": label, "opening_residual_m": residual,
            "bore_offset_m": offset,
        })

    candidates.sort(key=lambda c: c["score"])
    return candidates


def perceive_sc_port_pose(policy, task, obs):
    """One-frame SC opening pose: ``(pos, quat_wxyz, reproj_px)`` or ``None``."""
    log = policy.get_logger()
    views = policy._build_views(obs)
    if len(views) < 2:
        log.warn(f"[sc] only {len(views)} camera views usable")
        return None

    per_cam = {}
    for cam, (bgr, K, T) in views.items():
        try:
            dets = policy._pc.detect_sc_pose(bgr, conf_thresh=0.2)
        except Exception as exc:
            log.warn(f"[sc] {cam}: detect_sc_pose failed: {exc}")
            continue
        usable = []
        for det in dets[:5]:
            kps = np.asarray(det.get("kps"), dtype=np.float64)
            if kps.shape[0] < 4:
                continue
            usable.append({
                "kps": kps[:4], "conf": det.get("conf", 0.0), "K": K, "T": T,
                "P": policy._pc.build_projection_matrix(K, T),
            })
        if usable:
            per_cam[cam] = usable

    candidates = sc_multiview_candidates(policy, per_cam)
    if not candidates:
        log.warn("[sc] multiview matching found no SC opening candidates")
        return None

    clean = [c for c in candidates if c["reproj_px"] <= SC_MAX_SELECT_REPROJ_PX]
    if not clean:
        best = min(candidates, key=lambda c: c["reproj_px"])
        log.warn(
            f"[sc] no candidate under {SC_MAX_SELECT_REPROJ_PX:.1f}px select gate "
            f"(best {best['reproj_px']:.1f}px) -- rejecting frame"
        )
        return None

    try:
        tcp_pos, tcp_quat = policy._tcp()
        tip_pos, _ = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
    except Exception:
        chosen = clean[0]
    else:
        in_range = [
            c for c in clean
            if float(np.linalg.norm(c["X"] - tip_pos)) <= SC_MAX_HANDOFF_SELECT_M
        ]
        if not in_range:
            nearest = min(clean, key=lambda c: float(np.linalg.norm(c["X"] - tip_pos)))
            log.warn(
                f"[sc] all candidates beyond {SC_MAX_HANDOFF_SELECT_M*1000:.0f}mm handoff "
                f"gate (nearest {np.linalg.norm(nearest['X']-tip_pos)*1000:.1f}mm)"
            )
            return None
        chosen = min(in_range, key=lambda c: float(np.linalg.norm(c["X"] - tip_pos)))

    log.info(
        f"[sc] SC_OPENING convention={chosen['opening']} "
        f"width={chosen['width']*1000:.2f}mm height={chosen['height']*1000:.2f}mm "
        f"residual={chosen['opening_residual_m']*1000:.2f}mm "
        f"bore_offset={chosen['bore_offset_m']*1000:.2f}mm "
        f"reproj={chosen['reproj_px']:.1f}px"
    )
    if chosen["opening"] != "duplex":
        log.warn(
            "[sc] keypoints match a SINGLE BORE, not the duplex opening -- the "
            "duplex plug enters the opening centre, so this pose is offset by "
            f"{chosen['bore_offset_m']*1000:.2f}mm along the opening's long axis "
            "and its sign is unknown. Resolve the keypoint convention before use."
        )
    return (
        np.asarray(chosen["X"], dtype=np.float64),
        np.asarray(chosen["q_wxyz"], dtype=np.float64),
        float(chosen["reproj_px"]),
    )


def perceive_sc_port_pose_consensus(policy, task, get_observation):
    """Median of an agreeing cluster of single-frame SC poses."""
    log = policy.get_logger()
    samples = []
    for _ in range(SC_PERCEPT_SAMPLES):
        obs = get_observation()
        if obs is not None:
            res = perceive_sc_port_pose(policy, task, obs)
            if res is not None:
                X, q, reproj = res
                if np.isfinite(reproj) and reproj <= SC_MAX_PORT_REPROJ_PX:
                    samples.append((np.asarray(X, float), np.asarray(q, float), float(reproj)))
        policy.sleep_for(SC_PERCEPT_SAMPLE_DT)

    if len(samples) < SC_PERCEPT_MIN_AGREE:
        log.error(
            f"[sc] perception consensus failed: only {len(samples)}/{SC_PERCEPT_SAMPLES} "
            f"frames passed reproj (need {SC_PERCEPT_MIN_AGREE})"
        )
        return None

    positions = np.array([s[0] for s in samples])
    med = np.median(positions, axis=0)
    keep = [s for s in samples
            if float(np.linalg.norm(s[0] - med)) <= SC_PERCEPT_AGREE_TOL_M]
    if len(keep) < SC_PERCEPT_MIN_AGREE:
        spread = float(np.max(np.linalg.norm(positions - med, axis=1))) * 1000
        log.error(
            f"[sc] perception consensus failed: {len(keep)}/{len(samples)} agree within "
            f"{SC_PERCEPT_AGREE_TOL_M*1000:.1f}mm (spread={spread:.1f}mm)"
        )
        return None

    port_pos = np.median(np.array([s[0] for s in keep]), axis=0)
    reproj = float(np.median([s[2] for s in keep]))
    best = min(keep, key=lambda s: float(np.linalg.norm(s[0] - port_pos)))
    log.info(
        f"[sc] perception consensus: {len(keep)}/{len(samples)} agree, "
        f"port={np.round(port_pos, 5).tolist()} reproj={reproj:.2f}px"
    )
    return port_pos, best[1], reproj


# --------------------------------------------------------------------------
# Skill entry point.
# --------------------------------------------------------------------------
def run_sc_insertion(policy, task, get_observation, move_robot, send_feedback) -> bool:
    """Perceive the SC opening, sanity-check the handoff, align, and seat."""
    from .rl_insert_contract import port_frame

    log = policy.get_logger()
    send_feedback("sc opening perception")

    perceived = perceive_sc_port_pose_consensus(policy, task, get_observation)
    if perceived is None:
        log.error("[sc] perception failed to produce an SC port pose")
        return False
    port_pos, port_quat, reproj_px = perceived
    port_quat = np.asarray(port_quat, dtype=np.float64)
    port_quat /= max(float(np.linalg.norm(port_quat)), 1e-12)
    log.info(
        f"[sc] perceived port p={np.round(port_pos, 5).tolist()} "
        f"q_wxyz={np.round(port_quat, 5).tolist()} reproj={reproj_px:.2f}px"
    )

    obs = get_observation()
    if obs is None:
        log.error("[sc] no observation available for the wrench baseline")
        return False
    policy._wrench_baseline = policy._wrench_vector(obs)

    Rp = port_frame(port_quat)
    tcp_pos, tcp_quat = policy._tcp()
    tip_pos, R_tip = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
    dist = float(np.linalg.norm(tip_pos - port_pos))
    handoff_delta = Rp.T @ (tip_pos - port_pos)
    handoff_rot = axis_angle(Rp.T @ R_tip)
    log.info(
        f"[sc] handoff check: |tip-mouth|={dist*1000:.1f}mm "
        f"delta_port_mm={(handoff_delta*1000).round(2).tolist()} "
        f"rot_err_deg={np.degrees(handoff_rot).round(2).tolist()}"
    )
    if dist > SC_HANDOFF_MAX_DIST_M:
        log.error(
            f"[sc] tip is {dist*1000:.0f}mm from the mouth -- outside the "
            "last-inch envelope; the upstream macro must hand off closer"
        )
        return False

    return ScInsertionController(
        policy, task, get_observation, move_robot, send_feedback,
        port_pos=port_pos, port_quat=port_quat, Rp=Rp,
    ).run()


__all__ = [
    "LOCAL_SC_PORT_KPS",
    "SCConfig",
    "SC_BORE_PITCH_M",
    "SC_INSERT_DEPTH_M",
    "SC_OPENING_HEIGHT_M",
    "SC_OPENING_WIDTH_M",
    "ScInsertionController",
    "classify_opening",
    "next_sc_depth",
    "perceive_sc_port_pose",
    "perceive_sc_port_pose_consensus",
    "run_sc_insertion",
    "sc_multiview_candidates",
    "sc_tip_pose_from_tcp",
    "tcp_pose_for_sc_tip",
]
