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
    inner faces at +/-11.204 mm.  Height is depth-dependent: the full-depth
    ``cube_collider_box_mid*`` rails put the ceiling at +4.050, but
    ``cube_collider_box.001`` is a 10.8 mm-deep lip across the middle of the
    channel that drops it to **+3.800**; the floor is ``.002`` at -4.050
    throughout.  The plug passes through the lip, so the binding opening is
    22.41 mm x **7.85 mm**, asymmetric about z=0.  It is split by a 2.99 mm
    centre divider into two 9.71 mm bores on a 12.7 mm pitch (the standard SC
    duplex pitch -- a useful sanity check that this reading is right).
    8.10 mm is the height clear of the lip and is a real dimension; it is just
    not the one a clearance budget can be built on.

``aic_description/urdf/task_board.urdf.xacro``
  * SC ports are posed rpy ``(1.57 + roll, pitch, 1.57 + yaw)`` on the board, so
    ``R_board_from_port = Rz(90) Rx(90) = [[0,0,1],[1,0,0],[0,1,0]]``.
  * Composing that with the port-base rotation above maps the insertion axis to
    board **-Z**: SC descends straight down, exactly like SFP.  This is why the
    SFP entrance-frame estimator (which hardcodes "insertion axis is world -Z")
    is reusable here unchanged.
  * ``sc_port_link`` +X maps to board +Y, so the duplex opening's long axis lies
    along board Y.
  * STALE -- do not rely on this: the qualification xacro placed two slots at
    y=0.0295 and y=0.0705, i.e. 41 mm apart.  ``task_board.urdf.xacro`` now
    declares ``sc_port_0..4`` (3 on rail 0, 2 on rail 1), matching the ~6
    adapters visible in field imagery, so neighbours can be far closer than
    41 mm.  Adapters cannot overlap, so their own 25.78 mm outer width is the
    only safe lower bound on port-to-port spacing.

``aic_model/aic_model/sc_plug_pose_geometry.py``
  * The plug body is 20.0 mm x 6.4 mm.  Against the 22.41 x 7.85 mm opening that
    is 1.2 mm of lateral and **0.725 mm** of vertical clearance per side -- still
    a much looser fit than SFP, which is the main reason a script is plausible
    here, but vertical is the binding axis and it is tighter than once recorded.
    Budget grasp repeatability against 0.725 mm, not 1.2 mm.

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
# MINIMUM clear height along the insertion path.  The channel is neither
# symmetric about z=0 nor constant along its depth, so there are two defensible
# numbers here and only one of them is a clearance budget:
#   * cube_collider_box_mid / _mid02 / _mid03 run the FULL 27.432 mm depth at
#     z 4.050..4.650, so the ceiling is +4.050 over most of the channel.
#   * cube_collider_box.001 is a LIP, not a plate: full width (x 25.781) but
#     only 10.8 mm deep, centred at y=0, spanning z 3.800..4.650.  Through that
#     band the ceiling drops to +3.800.
#   * Floor is .002 at -4.050, full depth.
# The plug traverses the lip -- 15.64 mm of insertion from the y=+13.716 face
# reaches y=-1.92, and the lip spans y=-5.4..+5.4 -- so the binding height is
# 3.800 + 4.050 = 7.850 mm.  8.10 mm is the height everywhere else and is not a
# misreading; it is simply not the number a tolerance budget can use.
SC_OPENING_HEIGHT_M = 0.00785        # duplex inner height (port +Z / board +X)
SC_BORE_WIDTH_M = 0.00971            # one bore, between side wall and divider
SC_BORE_PITCH_M = 0.01270            # standard SC duplex pitch
SC_PLUG_WIDTH_M = 0.02000            # from sc_plug_pose_geometry keypoints
SC_PLUG_HEIGHT_M = 0.00640

# The model does not outline any physical feature, so the hypotheses must be the
# LABEL CONVENTIONS the two collectors in this repo project, not SDF geometry.
# There are exactly two, and they disagree:
#
#   DataCollectorScPoseGT.py   SC_HALF_WIDTH_M 0.0044 / SC_HALF_HEIGHT_M 0.0030
#                              ->  8.8  x 6.00 mm   (aspect 1.467)
#   DataCollectorPoseSC.py     SC_FULL_WIDTH_M 0.02578 / SC_FULL_LENGTH_M 0.00927
#                              -> 25.78 x 9.27 mm   (aspect 2.781)
#
# The shipped best_sc_pose.pt follows the FIRST.  Measured, not assumed: running
# the weights over testing/check_sc_previews and scaling by the SC duplex bore
# pitch (12.70 mm, a fixed mechanical dimension and so independent of segmenting
# anything) gives 9.00 x 5.94 and 8.73 x 6.01 mm -- 8.8 x 6.0 to about 1%.  The
# quad's diagonals agree to 1-3% while adjacent sides differ 1.44-1.51, so it is
# a rectangle and not a diamond with those diagonals.  Its centroid sits on the
# DUPLEX CENTRE (1.2-1.9 mm from the bore-pair midpoint, against 5.3-5.4 mm from
# either bore), which is why both conventions carry a zero bore offset below.
#
# Note train_sc.py's --data default points at pose_sc, which is the SECOND
# collector's output directory -- so the shipped weights were not produced by the
# default invocation.  Keep both entries: classify_opening then reports which
# model is actually loaded, and one build works with the current weights and any
# retrain, A/B-able through AIC_SC_POSE_WEIGHTS without a code change.
#
# The SDF-derived guesses that used to live here (duplex 22.41 x 8.10, bore
# 9.71 x 8.10) matched neither, and cost two field runs: they made every genuine
# detection classify as "single_bore" and emit a warning claiming the pose needed
# a 6.35 mm correction it does not need.
SC_GT_LABEL_WIDTH_M = 0.0088         # DataCollectorScPoseGT, 2 * SC_HALF_WIDTH_M
SC_GT_LABEL_HEIGHT_M = 0.0060        # DataCollectorScPoseGT, 2 * SC_HALF_HEIGHT_M
SC_FACE_LABEL_WIDTH_M = 0.02578      # DataCollectorPoseSC, outer face
SC_FACE_LABEL_HEIGHT_M = 0.00927     # DataCollectorPoseSC, outer face
SC_OPENING_HYPOTHESES = (
    ("gt_label", SC_GT_LABEL_WIDTH_M, SC_GT_LABEL_HEIGHT_M),
    ("outer_face", SC_FACE_LABEL_WIDTH_M, SC_FACE_LABEL_HEIGHT_M),
)

# Local rectangle for the single-view PnP fallback, in the same corner order the
# SFP estimator uses (KP0 top-left, KP1 top-right, KP2 bottom-right, KP3
# bottom-left).  This MUST be the convention the loaded weights emit, not the
# port's physical opening: PnP scales the pose by the ratio between this
# rectangle and the observed one, so the old 22.41 mm entry would have placed the
# port ~2.5x too far away.  Nothing calls solvePnP for SC today (the only SC pose
# path is multi-view triangulation, which needs no model rectangle), so this is a
# trap for whoever wires that fallback up rather than a live bug -- but wire it to
# the hypothesis ``classify_opening`` actually selects, not to this default.
LOCAL_SC_PORT_KPS = np.array(
    [
        [+SC_GT_LABEL_WIDTH_M / 2.0, +SC_GT_LABEL_HEIGHT_M / 2.0, 0.0],
        [-SC_GT_LABEL_WIDTH_M / 2.0, +SC_GT_LABEL_HEIGHT_M / 2.0, 0.0],
        [-SC_GT_LABEL_WIDTH_M / 2.0, -SC_GT_LABEL_HEIGHT_M / 2.0, 0.0],
        [+SC_GT_LABEL_WIDTH_M / 2.0, -SC_GT_LABEL_HEIGHT_M / 2.0, 0.0],
    ],
    dtype=np.float64,
)

# UNCALIBRATED -- see module docstring item 1.
SC_TIP_IN_TCP_POS = _env_vector("RL_INSERT_SC_TIP_IN_TCP_POS", SFP_TIP_IN_TCP_POS)
SC_TIP_IN_TCP_QUAT = _env_vector("RL_INSERT_SC_TIP_IN_TCP_QUAT", SFP_TIP_IN_TCP_QUAT)
SC_TIP_CALIBRATED = _env_bool("RL_INSERT_SC_TIP_CALIBRATED", False)

# Take the insertion axis from perception but keep the twist the macro handed us,
# instead of rotating the plug onto the perceived port yaw.
#
# 2026-07-25 field run: `rot_err_deg=[3.19, -4.37, -89.55]` -- 89.71 deg about an
# axis 3.46 deg off the insertion axis, stable across all 7 consensus frames.
# That is a frame-convention offset, not a perception error:
#   * `_estimate_sfp_port_orientation` builds its in-plane axis from exactly the
#     vector `sc_multiview_candidates` calls `width`, and width (7.39mm) >
#     height (3.97mm), so Rp[:,0] really is the opening's long axis;
#   * a non-square rectangle pins that axis to within 180 deg, never 90 deg;
#   * the duplex plug is 20.0mm across and the opening 7.85mm tall, so a plug
#     genuinely turned 90 deg could not enter at all -- yet the handoff was only
#     3.74/1.75mm off laterally.
# It is the SFP grasp transform standing in for the uncalibrated SC one (6c).
#
# The cost of chasing it was the whole run: `_align` slews at
# ``align_max_rotation_step_rad`` (1.5 deg) per iteration against a 15 s wall
# budget, so 89.7 deg needs >=60 iterations and times out.  Removing an exact
# -90 deg about the port Z leaves 4.89 deg, i.e. the macro's handoff twist is
# already inside the 6.9 deg budget the 1.2mm lateral clearance allows.
#
# Do NOT "fix" the timeout by raising the budget or the step cap: that lets the
# robot complete a 90 deg turn it should never make, and drive a 20mm plug at a
# 7.85mm opening.  Once 6c lands this offset collapses to ~0 and the flag stops
# mattering; it stays as the way to A/B the two behaviours in sim.
SC_PRESERVE_HANDOFF_YAW = _env_bool("RL_INSERT_SC_PRESERVE_HANDOFF_YAW", True)

SC_WRENCH_LOG_PERIOD_S = _env_float("RL_INSERT_SC_WRENCH_LOG_PERIOD_S", 0.25)
SC_PERCEPT_SAMPLES = _env_int("RL_INSERT_SC_PERCEPT_SAMPLES", 7)
SC_PERCEPT_MIN_AGREE = _env_int("RL_INSERT_SC_PERCEPT_MIN_AGREE", 3)
SC_PERCEPT_SAMPLE_DT = _env_float("RL_INSERT_SC_PERCEPT_SAMPLE_DT", 0.10)
SC_PERCEPT_AGREE_TOL_M = _env_float("RL_INSERT_SC_PERCEPT_AGREE_TOL_M", 0.004)
SC_MAX_PORT_REPROJ_PX = _env_float("RL_INSERT_SC_MAX_PORT_REPROJ_PX", 6.0)
SC_MAX_SELECT_REPROJ_PX = _env_float("RL_INSERT_SC_MAX_SELECT_REPROJ_PX", 5.0)
# Pixel radius for the pre-triangulation detection filter. Measured from the
# projected gripper TCP, not the plug tip -- the SC tip transform is still
# uncalibrated, so it must not be what a perception gate is centred on.
SC_MAX_DETECT_PX_FROM_TIP = _env_float("RL_INSERT_SC_MAX_DETECT_PX_FROM_TIP", 250.0)
SC_MAX_DETS_PER_CAM = max(1, int(_env_float("RL_INSERT_SC_MAX_DETS_PER_CAM", 8)))
# WARNING -- this gate no longer has the margin its value was chosen for, and it
# is a 3D distance, so handoff height is mixed into a lateral decision.
# It was sized against 41 mm slot spacing, which is stale (see the ground truth
# above: the board now carries sc_port_0..4).  Upstream
# docs/task_board_description.md is explicit that the board "supports up to five
# SC ports, distributed across two rails" and that ports "slide along their rails
# to allow for randomized positional offsets" over [0, 0.115] m -- so there is no
# fixed pitch to lean on at all, and two ports on one rail can end up adjacent.
# The only safe bound is that
# adapters cannot overlap, so neighbours are >= 25.78 mm apart laterally -- and
# at a 15 mm handoff height a shoulder-to-shoulder neighbour sits at
# sqrt(25.78^2 + 15^2) ~= 29.8 mm, i.e. just INSIDE this 30 mm gate.
# Tightening the number alone is not the fix: it would start rejecting the real
# target whenever the macro hands off higher.  The fix is to gate on lateral
# (board XY) distance, since insertion is straight down board -Z -- then the
# target is ~0-3 mm and any neighbour >= 25.78 mm regardless of height.
# Left as-is deliberately: changing selection behaviour belongs in its own
# change, with its own field run.
SC_MAX_HANDOFF_SELECT_M = _env_float("RL_INSERT_SC_MAX_HANDOFF_SELECT_M", 0.030)
SC_HANDOFF_MAX_DIST_M = _env_float("RL_INSERT_SC_HANDOFF_MAX_DIST_M", 0.120)
# How far "inside" the port the tip may compute to at handoff before the run is
# refused.  Zero is the physical truth -- the plug is outside until it is pushed
# in -- so this is pure tolerance for perception noise and a macro that hands off
# right at the mouth, not a licence to start partly inserted.
SC_MAX_HANDOFF_DEPTH_M = _env_float("RL_INSERT_SC_MAX_HANDOFF_DEPTH_M", 0.002)
# Dimensional gate on the triangulated rectangle.  The floor is deliberately NOT
# the port's physical size, and NOT the label size either -- see below.
#
# The quad spans 0.25-0.26 of the adapter in every preview frame, but read that
# fraction carefully: it is 0.25 of the FULL VISIBLE BODY, which is the 34.67 mm
# FOA-005A including its mounting flanges (sc_port_visual.glb node 16, extents
# 34.671 x 27.432 x 9.271 mm), NOT of the 25.78 mm outer face.  0.256 x 34.671 =
# 8.88 mm, which is the label.  Against 25.78 mm the same fraction computes to
# 6.6 mm and sends you looking for a convention that does not exist.  Two
# separate sessions made exactly that substitution before the bore-pitch ruler
# settled it; the absolute numbers are in SC_OPENING_HYPOTHESES above.
#
# The floor must clear the TRIANGULATED size, which is materially smaller than
# the label: the 2026-07-25 field runs measured 7.09-7.40 mm x 3.95-4.07 mm
# against a 8.8 x 6.0 mm label, because the target port is detected at only
# ~0.32-0.45 confidence in the left and right cameras (a different adapter wins
# 0.91 there) and weak corners pull the triangulated quad in towards its own
# centroid.  A ~4.0 mm short axis is a full 1 mm BELOW the old 0.005 floor, so
# that gate was rejecting every genuine candidate deterministically, not
# marginally -- which is why the failure reproduced exactly x7.
#
# Derive any future raise from the *matched* hypothesis and the observed
# shrinkage, never from the label or the SDF alone.
SC_MIN_OPENING_M = _env_float("RL_INSERT_SC_MIN_OPENING_M", 0.002)
SC_MAX_OPENING_M = _env_float("RL_INSERT_SC_MAX_OPENING_M", 0.030)

# Warn when the triangulated rectangle is this far from its closest known label
# convention.  The 2026-07-25 field runs sat at ~3.65 mm (7.09x4.06 against the
# 8.8x6.0 gt_label) purely from weak outer-camera detections, so a threshold at
# or below that would fire every single frame and train people to ignore it.
# Matching NEITHER convention looks like ~19-24 mm, so 6 mm separates "known
# shrinkage" from "the loaded weights are not one of the two we know about".
SC_OPENING_RESIDUAL_WARN_M = _env_float("RL_INSERT_SC_OPENING_RESIDUAL_WARN_M", 0.006)


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
    """Name the label convention a triangulated rectangle matches.

    Returns ``(label, residual_m, lateral_offset_m)``.

    ``lateral_offset_m`` is always 0.0 and is kept only so callers and logs do
    not have to change.  Both collectors project their rectangle from
    ``sc_port_base_link_entrance`` -- the duplex centre -- so a detection is
    never half a bore off, whichever convention produced it.  Measurement backs
    this: the predicted quad's centroid sits 1.2-1.9 mm from the bore-pair
    midpoint against 5.3-5.4 mm from either individual bore.  The SC port also
    declares ONE TouchPlugin where the NIC declares two, and the duplex plug
    enters as a single unit, so there is no bore to choose.

    This exists because the keypoint convention of ``best_sc_pose.pt`` is not
    recorded anywhere; measuring it beats guessing it.
    """
    best = None
    for label, expect_w, expect_h in SC_OPENING_HYPOTHESES:
        residual = abs(width_m - expect_w) + abs(height_m - expect_h)
        if best is None or residual < best[1]:
            best = (label, residual)
    label, residual = best
    return label, float(residual), 0.0


def seat_frame(Rp, R_tip):
    """Insertion axis from perception, in-plane twist preserved from the plug.

    ``Rp`` columns are ``[lat_x, lat_y, insert_axis]``.  This returns a frame
    with the SAME third column -- perception's insertion axis is trusted, and it
    is a constant (``_estimate_sfp_port_orientation`` hardcodes world -Z) -- but
    whose in-plane axes are rotated to sit under the plug's current twist rather
    than under the perceived port yaw.  See ``SC_PRESERVE_HANDOFF_YAW``.

    The result is still orthonormal and right-handed, so it is a drop-in for
    ``Rp`` anywhere a rotation target is commanded.  Falls back to ``Rp`` when
    the plug's own x-axis is parallel to the insertion axis and there is no
    in-plane direction to preserve.
    """
    Rp = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
    R_tip = np.asarray(R_tip, dtype=np.float64).reshape(3, 3)
    z = Rp[:, 2]
    z_norm = float(np.linalg.norm(z))
    if z_norm < 1e-12:
        return Rp.copy()
    z = z / z_norm
    x = R_tip[:, 0] - float(np.dot(R_tip[:, 0], z)) * z
    x_norm = float(np.linalg.norm(x))
    if x_norm < 1e-6:
        return Rp.copy()
    x = x / x_norm
    return np.column_stack([x, np.cross(z, x), z])


class ScInsertionController:
    """Align to the perceived SC opening, then force-regulate to the seat."""

    STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
    DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
    HOLD_STIFFNESS = [200.0, 200.0, 200.0, 80.0, 80.0, 80.0]
    HOLD_DAMPING = [80.0, 80.0, 80.0, 30.0, 30.0, 30.0]

    def __init__(self, policy, task, get_observation, move_robot, send_feedback,
                 *, port_pos, port_quat, Rp, Rs=None, config=None):
        self.policy = policy
        self.task = task
        self.get_observation = get_observation
        self.move_robot = move_robot
        self.send_feedback = send_feedback
        self.config = (config or SCConfig.from_env()).validated()
        self.port_pos = np.asarray(port_pos, dtype=np.float64).reshape(3)
        self.port_quat = np.asarray(port_quat, dtype=np.float64).reshape(4)
        self.Rp = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
        # Rp drives POSITION (and the wrench frame); Rs drives ROTATION targets.
        # They share column 2, so the lateral plane -- and therefore every
        # position correction -- is identical either way; only the in-plane
        # twist differs.  Rs defaults to Rp, which is the pre-2026-07-25
        # behaviour and what the geometry tests construct.
        self.Rs = self.Rp.copy() if Rs is None else np.asarray(
            Rs, dtype=np.float64).reshape(3, 3)
        # Constant in-plane rotation from Rp to Rs.  Corrections measured in the
        # Rp frame are re-expressed through this before being commanded on Rs.
        self.R_yaw = self.Rp.T @ self.Rs
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
        rotation_error = axis_angle(self.Rs.T @ tip_rotation)
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
                self._hold_tip(tip_pos, self.Rs)
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
                    self._tcp_target(fixed_tip, self.Rs),
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
                target_rotation = self.Rs @ rotation_from_axis_angle(remaining * rotation_error)
            else:
                target_rotation = self.Rs
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
                self._hold_tip(tip_pos, self.Rs)
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
            target_rotation = self.Rs
            if self.config.seat_align_enable and (
                np.any(acc_lat != 0.0) or np.any(acc_tilt != 0.0)
            ):
                target_tip = (
                    target_tip + self.Rp[:, 0] * acc_lat[0] + self.Rp[:, 1] * acc_lat[1]
                )
                # acc_tilt is measured about Rp's lateral axes (_wrench_plug_frame
                # resolves the wrench through Rp), so apply it in the Rp frame and
                # only then carry the constant twist across to the seat frame.
                # Commanding it on Rs directly would rotate the correction by the
                # very offset this frame exists to absorb.
                target_rotation = self.Rp @ rotation_from_axis_angle(
                    np.array([acc_tilt[0], acc_tilt[1], 0.0], dtype=np.float64)
                ) @ self.R_yaw
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
def _sc_detection_centroid(det):
    kps = np.asarray(det.get("kps"), dtype=np.float64)
    if kps.shape[0] < 4:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.mean(kps[:4, :2], axis=0)


def _project_point_px(P, X):
    x = np.asarray(P, dtype=np.float64) @ np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    if x[2] <= 1e-6:
        return None
    return np.array([x[0] / x[2], x[1] / x[2]], dtype=np.float64)


def _rank_sc_detections(dets):
    return sorted(dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True)[:SC_MAX_DETS_PER_CAM]


def _round_list(values, decimals=1):
    return np.round(np.asarray(values, dtype=np.float64), decimals).tolist()


def _sc_detection_diag(dets):
    diag = []
    for i, det in enumerate(dets):
        diag.append({
            "i": i,
            "conf": round(float(det.get("conf", 0.0)), 3),
            "centroid": _round_list(_sc_detection_centroid(det), 1),
        })
    return diag


def _sc_tip_projections(policy, per_cam):
    """Project the gripper TCP into each camera as the detection-filter centre.

    Deliberately the TCP and not ``sc_tip_pose_from_tcp``: the SC tip transform
    is the uncalibrated SFP default (see UNCALIBRATED item 1), and centring a
    perception gate on a constant we know is wrong couples this filter to that
    error.  The TCP comes straight from TF, and it sits ~58 mm from the tip --
    far inside a gate whose radius is hundreds of pixels, so the coarse
    proximity test loses nothing by using it.
    """
    try:
        anchor_pos, _ = policy._tcp()
        anchor_pos = np.asarray(anchor_pos, dtype=np.float64).reshape(3)
    except Exception:
        return None, None

    anchor_uv = {}
    for cam, dets in per_cam.items():
        if not dets:
            continue
        try:
            P = policy._pc.build_projection_matrix(dets[0]["K"], dets[0]["T"])
            uv = _project_point_px(P, anchor_pos)
        except Exception:
            return anchor_pos, None
        if uv is None or not np.all(np.isfinite(uv)):
            return anchor_pos, None
        anchor_uv[cam] = uv
    return anchor_pos, anchor_uv


def _select_sc_detections_for_triangulation(policy, per_cam, log=None):
    cams = [cam for cam, dets in per_cam.items() if dets]
    if not cams:
        return {}

    candidates = {cam: list(per_cam[cam]) for cam in cams}
    _anchor_pos, anchor_uv = _sc_tip_projections(policy, candidates)
    use_tip_filter = anchor_uv is not None and all(cam in anchor_uv for cam in cams)

    selected = {}
    for cam in cams:
        dets = candidates[cam]
        if use_tip_filter:
            uv = anchor_uv[cam]
            survivors = [
                det for det in dets
                if float(np.linalg.norm(_sc_detection_centroid(det) - uv)) <= SC_MAX_DETECT_PX_FROM_TIP
            ]
            selected[cam] = _rank_sc_detections(survivors)
            tip_txt = _round_list(uv, 1)
            mode = "tcp_filter"
        else:
            selected[cam] = _rank_sc_detections(dets)
            tip_txt = None
            mode = "fallback_confidence"

        if log is not None:
            log.info(
                f"[sc] SC_PERCEPT_CAMERA cam={cam} mode={mode} "
                f"radius={SC_MAX_DETECT_PX_FROM_TIP:.1f}px "
                f"before={len(dets)} after={len(selected[cam])} "
                f"tcp_px={tip_txt} dets={_sc_detection_diag(dets)}"
            )
            # An emptied camera makes triangulation impossible, and the caller's
            # "no candidates" message does not say why. Name the cause here.
            if use_tip_filter and dets and not selected[cam]:
                log.warn(
                    f"[sc] SC_PERCEPT_CAMERA cam={cam} pre-filter removed all "
                    f"{len(dets)} detection(s): none within "
                    f"{SC_MAX_DETECT_PX_FROM_TIP:.0f}px of the TCP projection at "
                    f"{tip_txt}. Raise RL_INSERT_SC_MAX_DETECT_PX_FROM_TIP if the "
                    "target port is genuinely further out in this view."
                )

    return selected


def _log_sc_best_candidate(log, candidate):
    cam_diag = []
    for diag in candidate.get("camera_diagnostics", []):
        cam_diag.append({
            "cam": diag["cam"],
            "conf": round(float(diag["conf"]), 3),
            "centroid": _round_list(diag["centroid"], 1),
            "kps": _round_list(diag["kps"], 1),
            "reproj_px": [
                None if err is None else round(float(err), 2)
                for err in diag["reproj_px"]
            ],
        })
    log.info(
        f"[sc] SC_PERCEPT_BEST score={candidate['score']:.2f} "
        f"reproj={candidate['reproj_px']:.2f}px "
        f"width={candidate['width']*1000:.2f}mm height={candidate['height']*1000:.2f}mm "
        f"opening={candidate['opening']} "
        f"X={_round_list(candidate['X'], 5)} cams={cam_diag}"
    )


def _log_sc_rejections(log, rejects, limit=8):
    """Report the combinations that never became candidates, and why.

    Without this, a perception failure says only "no candidates" and the cause
    has to be inferred from the source -- which cost a field run on 2026-07-25.
    Every ``continue`` in ``sc_multiview_candidates`` records here instead.
    """
    counts = {}
    for reason, _ in rejects:
        counts[reason] = counts.get(reason, 0) + 1
    detail = [f"{reason}({info})" for reason, info in rejects[:limit] if info]
    log.warn(
        f"[sc] SC_PERCEPT_REJECT {len(rejects)} combination(s) discarded "
        f"counts={counts} "
        f"size_gate=[{SC_MIN_OPENING_M * 1000:.1f}, {SC_MAX_OPENING_M * 1000:.1f}]mm "
        f"sample={detail}"
    )


def sc_multiview_candidates(policy, per_cam):
    """Triangulate the four SC keypoints across cameras.

    Mirrors the SFP flow.  The orientation estimator is reused verbatim: it
    assumes the insertion axis is world -Z, which the asset geometry confirms is
    true for SC as well (see module docstring).
    """
    log = policy.get_logger()
    per_cam = _select_sc_detections_for_triangulation(policy, per_cam, log=log)
    cams = [cam for cam, dets in per_cam.items() if dets]
    if len(cams) < 2:
        return []

    candidates = []
    rejects = []
    for picks in itertools.product(*[per_cam[cam] for cam in cams]):
        kp_3d = []
        try:
            for i in range(4):
                pts_2d = [tuple(pick["kps"][i]) for pick in picks]
                Ps = [pick["P"] for pick in picks]
                kp_3d.append(policy._pc.triangulate(pts_2d, Ps))
        except Exception as exc:
            rejects.append(("triangulate_error", f"{type(exc).__name__}"))
            continue
        kp_3d = np.array(kp_3d, dtype=np.float64)
        X = kp_3d.mean(axis=0)
        if X[2] < -0.05 or X[2] > 0.25:
            rejects.append(("depth", f"z={X[2] * 1000:.0f}mm"))
            continue

        q_wxyz, yaw = policy._estimate_sfp_port_orientation(kp_3d)
        if q_wxyz is None:
            rejects.append(("degenerate_axis", "in-plane axis vertical or zero"))
            continue

        width = float(np.linalg.norm(((kp_3d[0] + kp_3d[3]) * 0.5) - ((kp_3d[1] + kp_3d[2]) * 0.5)))
        height = float(np.linalg.norm(((kp_3d[0] + kp_3d[1]) * 0.5) - ((kp_3d[2] + kp_3d[3]) * 0.5)))
        if not (SC_MIN_OPENING_M <= width <= SC_MAX_OPENING_M) or not (
            SC_MIN_OPENING_M <= height <= SC_MAX_OPENING_M
        ):
            rejects.append(("size", f"{width * 1000:.1f}x{height * 1000:.1f}mm"))
            continue

        errors = []
        camera_diagnostics = []
        for cam, pick in zip(cams, picks):
            reproj_px = []
            for i in range(4):
                err = policy._reproject_error_px(kp_3d[i], pick["K"], pick["T"], pick["kps"][i])
                if err is not None:
                    errors.append(err)
                    reproj_px.append(float(err))
                else:
                    reproj_px.append(None)
            camera_diagnostics.append({
                "cam": cam,
                "conf": float(pick.get("conf", 0.0)),
                "centroid": _sc_detection_centroid(pick),
                "kps": np.asarray(pick["kps"][:4], dtype=np.float64),
                "reproj_px": reproj_px,
            })
        if not errors:
            rejects.append(("no_reproj", "every keypoint failed to reproject"))
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
            "camera_diagnostics": camera_diagnostics,
        })

    # Log whenever anything was discarded, not only on total failure: a run that
    # produces one candidate out of eight is also worth knowing about.
    if rejects:
        _log_sc_rejections(log, rejects)

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
        for det in dets:
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

    # Diagnose the same candidate the select gate reports on. candidates[0] is
    # best by *score* (reproj + shape residual - confidence bonus), which is not
    # necessarily best by reproj, and logging two different candidates under the
    # word "best" is how a log costs you a field run to interpret.
    best_by_reproj = min(candidates, key=lambda c: c["reproj_px"])
    _log_sc_best_candidate(log, best_by_reproj)

    clean = [c for c in candidates if c["reproj_px"] <= SC_MAX_SELECT_REPROJ_PX]
    if not clean:
        log.warn(
            f"[sc] no candidate under {SC_MAX_SELECT_REPROJ_PX:.1f}px select gate "
            f"(best {best_by_reproj['reproj_px']:.1f}px, {len(candidates)} candidates) "
            "-- rejecting frame"
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

    expected = dict((label, (w, h)) for label, w, h in SC_OPENING_HYPOTHESES)
    exp_w, exp_h = expected[chosen["opening"]]
    log.info(
        f"[sc] SC_OPENING convention={chosen['opening']} "
        f"width={chosen['width']*1000:.2f}mm height={chosen['height']*1000:.2f}mm "
        f"expected={exp_w*1000:.2f}x{exp_h*1000:.2f}mm "
        f"residual={chosen['opening_residual_m']*1000:.2f}mm "
        f"reproj={chosen['reproj_px']:.1f}px"
    )
    # Both conventions project from the duplex centre, so there is no bore to
    # choose and no offset to apply -- the old warning here claimed a 6.35 mm
    # correction that would have pushed the plug half a bore off-centre.  What is
    # worth warning about is a rectangle that matches NEITHER convention, which
    # means the loaded weights are not one of the two this repo knows how to
    # label, or triangulation has degraded badly.
    if chosen["opening_residual_m"] > SC_OPENING_RESIDUAL_WARN_M:
        log.warn(
            f"[sc] triangulated opening {chosen['width']*1000:.2f}x"
            f"{chosen['height']*1000:.2f}mm is {chosen['opening_residual_m']*1000:.2f}mm "
            f"from its closest known label convention ({chosen['opening']}, "
            f"{exp_w*1000:.2f}x{exp_h*1000:.2f}mm). Expect shrinkage when the "
            "outer cameras detect the target weakly; check the SC_PERCEPT_BEST "
            "per-camera confidences before trusting the scale."
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

    # The plug cannot already be inside a port it has not been pushed into, so a
    # positive handoff depth is physically impossible and means the computed tip
    # is not where the plug is.  Nothing downstream can recover from that: depth
    # feeds seat_candidate_depth_m, so an inflated reading makes _seat skip the
    # entire approach and wait for an insertion event that cannot arrive, which
    # RL_INSERT_REPORT_MISS_AS_SUCCESS then reports as success.
    #
    # 2026-07-25 both field runs read +7.04 and +6.99 mm here BEFORE any motion,
    # and the second went on to "seat" at 21.13 mm -- deeper than the 15.64 mm
    # fully-seated depth -- with 0.22 N axial, i.e. touching nothing at all.
    # Fail loudly instead, and name the cause: this is the uncalibrated tip
    # transform (6c), not perception, which agreed 6/6 at 4.50 px.
    if handoff_delta[2] > SC_MAX_HANDOFF_DEPTH_M:
        log.error(
            f"[sc] handoff depth is {handoff_delta[2]*1000:+.2f}mm -- the plug tip "
            "is computed to be INSIDE the port before any motion, which is "
            f"impossible (gate {SC_MAX_HANDOFF_DEPTH_M*1000:.1f}mm). "
            f"SC_TIP_IN_TCP_POS is {'CALIBRATED' if SC_TIP_CALIBRATED else 'the UNCALIBRATED SFP default'}"
            "; re-solve it with RL_INSERT_CALIB_DUMP=1 over ~10 grasps. Refusing "
            "to seat against a tip position this wrong."
        )
        return False

    Rs = seat_frame(Rp, R_tip) if SC_PRESERVE_HANDOFF_YAW else Rp
    # Rp.T @ Rs is a pure rotation about the insertion axis, so this is the
    # single number that says how far the perceived port yaw sits from the twist
    # the macro handed us.  Expect ~-90 deg until SC_TIP_IN_TCP is calibrated;
    # once it is, this should collapse towards zero.  If it is ever large AND
    # not near a right angle, the convention story here is wrong -- re-measure
    # before trusting either frame.
    twist = float(np.degrees(axis_angle(Rp.T @ Rs)[2]))
    log.info(
        f"[sc] seat frame: preserve_handoff_yaw={SC_PRESERVE_HANDOFF_YAW} "
        f"twist_vs_perceived_yaw_deg={twist:.2f} "
        f"(rotation the controller is NOT commanding)"
    )

    return ScInsertionController(
        policy, task, get_observation, move_robot, send_feedback,
        port_pos=port_pos, port_quat=port_quat, Rp=Rp, Rs=Rs,
    ).run()


__all__ = [
    "LOCAL_SC_PORT_KPS",
    "SCConfig",
    "SC_BORE_PITCH_M",
    "SC_INSERT_DEPTH_M",
    "SC_OPENING_HEIGHT_M",
    "SC_OPENING_WIDTH_M",
    "SC_PRESERVE_HANDOFF_YAW",
    "ScInsertionController",
    "classify_opening",
    "next_sc_depth",
    "perceive_sc_port_pose",
    "perceive_sc_port_pose_consensus",
    "run_sc_insertion",
    "sc_multiview_candidates",
    "sc_tip_pose_from_tcp",
    "seat_frame",
    "tcp_pose_for_sc_tip",
]
