"""Scripted SC (duplex fibre) insertion for the ``insert_cable`` skill.

This is the SC counterpart to :mod:`v50_controller`.  It reuses that module's
proven, geometry-agnostic seating primitives -- persistent force-lead setpoint,
wall-time stall detection, bounded overtravel, and the proportional wrench
alignment law -- but every axial constant is re-derived for SC, because SC seats
at 15.64 mm against SFP's 45.8 mm and a naive constant copy would command the
plug straight through the back of the port on the first stall.

The tip is located from vision, exactly as v50 does for SFP: every run starts
by measuring ``sc_tip_link`` with :class:`~aic_model.sc_plug_pose.ScPlugPoseEstimator`
(``prime_sc_plug_pose``), converting it to a per-grasp TCP->tip transform, and
refusing to insert when that measurement cannot be obtained.  The historic
fixed-grasp constant survives only for calibration diagnostics and for unit
tests that exercise geometry without a camera; the control path never falls
back to it once a run has started (see item 1 below).

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

CALIBRATION NOTES:

1. RESOLVED for the control path: the tip transform is measured per run by
   ``prime_sc_plug_pose`` from the trained SC plug-pose model, so the
   ``SC_TIP_IN_TCP_*`` constants (still the SFP grasp transform by default, and
   still almost certainly wrong for SC) no longer steer the robot.  They remain
   for ``RL_INSERT_CALIB_DUMP=1`` diagnostics and for camera-less unit tests;
   ``run_sc_insertion`` aborts before any motion if the measurement fails.
2. The deployed port model is ``best_sc_mouth_pose.pt``.  It predicts the four
   physical front-mouth corners plus an explicit centre using the single
   geometry contract in ``aic_example_policies.ros.sc_mouth_pose_geometry``.
   The controller rejects legacy four-keypoint detections rather than silently
   reinterpreting their virtual 8.8 x 6.0 mm rectangle as a physical mouth.
"""

from __future__ import annotations

import itertools
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from aic_example_policies.ros.sc_mouth_pose_geometry import (
    LOCAL_SC_FRONT_MOUTH_KPS_M,
    SC_FRONT_MOUTH_HEIGHT_M,
    SC_FRONT_MOUTH_WIDTH_M,
    SC_MOUTH_KEYPOINT_COUNT,
)

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - deployment validation covers availability
    least_squares = None

from .rl_insert_contract import (
    SFP_TIP_IN_TCP_POS,
    SFP_TIP_IN_TCP_QUAT,
    matrix_to_quat,
    port_frame,
    quat_to_matrix,
)
from .sc_visual_alignment import (
    ScBlueSideSignature,
    aggregate_sc_blue_side_signatures,
    bounded_recovery_offset_update,
    bounded_visual_port_update,
    detect_sc_duplex_opening,
    detect_sc_recovery_direction,
    fuse_sc_opening_hits,
    fuse_sc_recovery_evidence,
    measure_sc_blue_side_signature,
    project_point_px as project_sc_visual_point_px,
    ray_to_plane as sc_visual_ray_to_plane,
)
from .v50_controller import (
    HARD_FAILURE,
    SEATED,
    STALLED,
    WallProgressWatch,
    axis_angle,
    clamp_vector_norm,
    rotation_from_axis_angle,
    solve_tip_in_tcp,
    tip_from_tcp_transform,
)

# Distinct from both True and False: `ScInsertionController._align` returns
# this when the alignment timeout fires but `SC_STRICT_PERCEPTION` is off, so
# `run()` proceeds to seating instead of throwing the pose away.  Aborting
# before an attempt scores proximity only (~25 max); attempting -- even from
# an unconverged pose -- can still score a partial insertion (38-50), so
# discarding it is strictly worse (docs/scoring.md:106-114).  `_seat()`
# remains the safety net for a genuinely bad pose: it trips
# `lateral_safety_m` / `rotation_safety_rad` and returns STALLED rather than
# driving a wedge deeper.
ALIGN_TIMED_OUT = "align_timed_out"


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


def _sc_camera_frame_stamps(observation) -> dict[str, float]:
    """Extract finite ROS-domain image stamps without decoding the images."""

    from .sfp_plug_pose import stamp_to_seconds

    stamps = {}
    messages = {
        "left_camera": getattr(observation, "left_image", None),
        "center_camera": getattr(observation, "center_image", None),
        "right_camera": getattr(observation, "right_image", None),
    }
    for camera, message in messages.items():
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        try:
            stamp_s = float(stamp_to_seconds(stamp))
        except (TypeError, ValueError):
            continue
        if np.isfinite(stamp_s):
            stamps[camera] = stamp_s
    return stamps


def _sc_time_seconds(value) -> float:
    """Best-effort conversion of ROS, rclpy, or test-clock time to seconds."""

    if value is None:
        return float("nan")
    try:
        nanoseconds = getattr(value, "nanoseconds")
        if nanoseconds is not None:
            seconds = float(nanoseconds) * 1e-9
            if np.isfinite(seconds):
                return seconds
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        seconds = float(getattr(value, "sec")) + 1e-9 * float(
            getattr(value, "nanosec", 0.0)
        )
        if np.isfinite(seconds):
            return seconds
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return seconds if np.isfinite(seconds) else float("nan")


def _sc_sim_time_seconds(policy) -> float:
    """Return the policy's simulation-clock time without making it mandatory."""

    for getter in (
        lambda: policy.time_now(),
        lambda: policy.get_clock().now(),
    ):
        try:
            seconds = _sc_time_seconds(getter())
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(seconds):
            return seconds
    return float("nan")


def _sc_action_budget_remaining_s(policy, *, now_wall_s: float | None = None) -> float:
    """Expose the deployed policy deadline when present, otherwise NaN."""

    try:
        deadline = float(getattr(policy, "_action_deadline_wall"))
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    now = time.monotonic() if now_wall_s is None else float(now_wall_s)
    remaining = deadline - now
    return remaining if np.isfinite(remaining) else float("nan")


def _sc_enforce_action_deadline(policy, move_robot) -> None:
    """Check the deployment deadline from long-running SC perception loops."""

    enforce = getattr(policy, "_enforce_action_deadline", None)
    if callable(enforce):
        enforce(move_robot)


def _sc_format_time_map(values: dict[str, float]) -> str:
    """Render a compact, stable map for operator-facing SC timing logs."""

    pieces = []
    for camera in sorted(values):
        try:
            value = float(values[camera])
        except (TypeError, ValueError):
            value = float("nan")
        pieces.append(
            f"{camera}:{value:.3f}" if np.isfinite(value) else f"{camera}:na"
        )
    return "{" + ",".join(pieces) + "}"


def _sc_frame_timing_fields(policy, frame_stamps: dict[str, float]) -> str:
    """Describe image header age in its ROS/simulation clock domain."""

    now_sim_s = _sc_sim_time_seconds(policy)
    ages = {
        camera: now_sim_s - float(stamp_s)
        if np.isfinite(now_sim_s) and np.isfinite(float(stamp_s))
        else float("nan")
        for camera, stamp_s in frame_stamps.items()
    }
    return (
        f"frame_stamps_s={_sc_format_time_map(frame_stamps)} "
        f"frame_age_s={_sc_format_time_map(ages)}"
    )


def _sc_recovery_support_summary(diagnostics: dict) -> str:
    """Keep paired-mask diagnostics readable in one live log line."""

    def values(name: str) -> str:
        try:
            array = np.asarray(diagnostics[name], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return "na"
        return np.round(array, 3).tolist().__repr__()

    pieces = []
    mode = diagnostics.get("support_mode")
    if mode is not None:
        pieces.append(f"mode={mode}")
    for name, label in (
        ("baseline_support_fractions", "baseline"),
        ("current_support_fractions", "current"),
        ("common_support_fractions", "common"),
        ("new_occlusion_fractions", "new_occlusion"),
        ("support_change_fractions", "changed"),
    ):
        if name in diagnostics:
            pieces.append(f"{label}={values(name)}")
    for name, label in (
        ("baseline_corridor_support_fraction", "corridor_baseline"),
        ("current_corridor_support_fraction", "corridor_current"),
        ("common_corridor_support_fraction", "corridor_common"),
        ("new_corridor_occlusion_fraction", "corridor_new_occlusion"),
        ("corridor_support_change_fraction", "corridor_changed"),
    ):
        if name in diagnostics:
            try:
                pieces.append(f"{label}={float(diagnostics[name]):.3f}")
            except (TypeError, ValueError):
                pieces.append(f"{label}=na")
    direct_summary = " ".join(pieces) if pieces else "unavailable"
    by_band = diagnostics.get("support_by_band")
    if isinstance(by_band, dict):
        band_summaries = []
        for band, band_diagnostics in sorted(by_band.items()):
            if isinstance(band_diagnostics, dict):
                band_summaries.append(
                    f"band{band}[{_sc_recovery_support_summary(band_diagnostics)}]"
                )
        if band_summaries:
            return ";".join(band_summaries)
    return direct_summary


class _ScTiming:
    """Small, wall-clock truthful trace shared by the SC insertion phases."""

    def __init__(self, policy, log):
        self.policy = policy
        self.log = log
        self.start_wall_s = time.monotonic()
        self.start_sim_s = _sc_sim_time_seconds(policy)
        self.phase_start_wall_s = self.start_wall_s
        self.phase = "start"

    @staticmethod
    def _number(value: float) -> str:
        return f"{value:.3f}" if np.isfinite(value) else "na"

    def mark(self, phase: str, **fields) -> None:
        now_wall_s = time.monotonic()
        now_sim_s = _sc_sim_time_seconds(self.policy)
        phase_elapsed_s = now_wall_s - self.phase_start_wall_s
        total_elapsed_s = now_wall_s - self.start_wall_s
        sim_elapsed_s = (
            now_sim_s - self.start_sim_s
            if np.isfinite(now_sim_s) and np.isfinite(self.start_sim_s)
            else float("nan")
        )
        budget_remaining_s = _sc_action_budget_remaining_s(
            self.policy, now_wall_s=now_wall_s
        )
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        suffix = f" {details}" if details else ""
        self.log.info(
            f"[sc] SC_TIMING phase={phase} previous_phase={self.phase} "
            f"phase_wall_s={self._number(phase_elapsed_s)} "
            f"total_wall_s={self._number(total_elapsed_s)} "
            f"sim_elapsed_s={self._number(sim_elapsed_s)} "
            f"action_budget_remaining_s={self._number(budget_remaining_s)}"
            f"{suffix}"
        )
        self.phase = str(phase)
        self.phase_start_wall_s = now_wall_s

    def summary(self, outcome: bool, *, reason: str) -> None:
        now_wall_s = time.monotonic()
        now_sim_s = _sc_sim_time_seconds(self.policy)
        sim_elapsed_s = (
            now_sim_s - self.start_sim_s
            if np.isfinite(now_sim_s) and np.isfinite(self.start_sim_s)
            else float("nan")
        )
        self.log.info(
            f"[sc] SC_TIMING_SUMMARY outcome={bool(outcome)} reason={reason} "
            f"total_wall_s={self._number(now_wall_s - self.start_wall_s)} "
            f"sim_elapsed_s={self._number(sim_elapsed_s)} "
            "action_budget_remaining_s="
            f"{self._number(_sc_action_budget_remaining_s(self.policy, now_wall_s=now_wall_s))}"
        )


def _normalize_event(value: object) -> str:
    """Reduce an insertion-event name to its trailing ``module/port``.

    The scoring topic publishes e.g. ``cable_0#0#nic_card_mount_0/sfp_port_1``
    while the task names only ``nic_card_mount_0/sfp_port_0``.  v50's normaliser
    strips whitespace and slashes but not the ``cable_N#0#`` prefix, so its
    equality test cannot match even on a correct port; strip the prefix here.

    Order matters.  SC publishes ``cable_0#1#/sc_port_0/sc_port_base`` -- the
    prefix is followed by a *slash*.  Stripping slashes before splitting on
    ``#`` leaves the one the split then exposes, so the result kept a leading
    ``/`` and compared unequal to the task's ``sc_port_0/sc_port_base``.  The
    2026-07-27 03:33 run seated the correct port and was logged as "SEATED IN A
    DIFFERENT PORT"; under ``RL_INSERT_SC_STRICT_PORT_EVENT=1`` that same run
    would have been a HARD_FAILURE.  Strip slashes *after* the split.
    """
    text = str(value or "").strip()
    return text.rsplit("#", 1)[-1].strip("/")


def _env_vector(name: str, default: np.ndarray) -> np.ndarray:
    """Parse a vector env var, tolerating the format the calib dump prints.

    ``dump_sc_grasp_calibration`` logs the solved transform via ``.tolist()``,
    i.e. ``[0.1, 0.2, 0.3]``.  Without stripping brackets ``float("[0.1")``
    raises and this falls back to the SFP default -- silently, and the SFP
    default is exactly the value that produces the +7 mm fake seat.  So the
    natural copy-paste out of the log used to be indistinguishable from never
    having set the variable at all.  Accept both bracketed and bare forms, and
    say so out loud when a value is present but unusable.
    """
    raw = os.environ.get(name)
    if not raw:
        return np.asarray(default, dtype=np.float64).copy()
    cleaned = raw.strip().strip("[]()").replace(",", " ")
    try:
        value = np.array([float(part) for part in cleaned.split()], dtype=np.float64)
    except ValueError:
        print(f"[sc] {name}={raw!r} is not a number list -- IGNORED, using the "
              f"uncalibrated default {np.asarray(default).tolist()}", file=sys.stderr)
        return np.asarray(default, dtype=np.float64).copy()
    if value.shape != np.asarray(default).shape:
        print(f"[sc] {name}={raw!r} has {value.shape} entries, expected "
              f"{np.asarray(default).shape} -- IGNORED, using the uncalibrated "
              f"default {np.asarray(default).tolist()}", file=sys.stderr)
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

# Port-pose model contract.  These are the externally visible FRONT-mouth
# dimensions (22.407 x 8.10 mm), not the 7.85 mm binding clearance used by the
# seating safety calculations above.  Keeping model geometry in the collector's
# module makes checkpoint, labels, multiview fit, and deployment share one source
# of truth.
SC_POSE_WIDTH_M = SC_FRONT_MOUTH_WIDTH_M
SC_POSE_HEIGHT_M = SC_FRONT_MOUTH_HEIGHT_M
SC_POSE_LOCAL_KPS_M = LOCAL_SC_FRONT_MOUTH_KPS_M
SC_POSE_CONVENTION = "physical_front_mouth"

# UNCALIBRATED -- see module docstring item 1.
SC_TIP_IN_TCP_POS = _env_vector("RL_INSERT_SC_TIP_IN_TCP_POS", SFP_TIP_IN_TCP_POS)
SC_TIP_IN_TCP_QUAT = _env_vector("RL_INSERT_SC_TIP_IN_TCP_QUAT", SFP_TIP_IN_TCP_QUAT)
SC_TIP_CALIBRATED = _env_bool("RL_INSERT_SC_TIP_CALIBRATED", False)

# Deployment policy (docs/INSERTION_EVENT_POLICY.md): any fresh scoring
# insertion event is a successful insertion, even when it names a different
# cable or port than the task asked for.  Flowstate owns task/port bookkeeping;
# this controller only confirms that the held cable physically seated.
# Nearest-to-tip remains the perception association rule, so a port-ID mismatch
# is expected behaviour rather than an anomaly, and rejecting the event aborts
# the process *after* a successful insertion.  Opt in to the old strict
# behaviour with RL_INSERT_SC_STRICT_PORT_EVENT=1; do not change this default.
SC_STRICT_PORT_EVENT = _env_bool("RL_INSERT_SC_STRICT_PORT_EVENT", False)

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

# Extra ground-truth TF frames to probe when solving SC_TIP_IN_TCP_*, appended
# after the task-derived names built by ``sc_calib_frame_candidates``.
#
# Do NOT reuse RLInsert's CALIB_PLUG_FRAMES: it is SFP-only
# ("sfp_tip,sfp_plug,plug,cable_0,sfp,gripper/sfp_tip,...") and, more subtly,
# every name in it is missing both the ``cable_N/`` prefix and the ``_link``
# suffix that the sim actually publishes.  The naming that demonstrably works is
# in DataCollectorScPlugPoseGT._tip_frame_candidates and
# DataCollectorSfpPlugPoseGT: ``{task.cable_name}/{task.plug_name}_link``, e.g.
# ``cable_0/sc_tip_link`` -- matching ``sc_tip_link`` in the SC Plug model.sdf,
# merged into the cable model.  Guessing "sc_tip" resolves nothing.
# Field enumeration (2026-07-25) showed the sim publishes neither of those.  It
# does publish ``selected_sc/sc_tip_link`` and ``selected_sc/sc_plug_link``, and
# they are a real SC plug -- their separation is 11.65 mm, exactly the SC Plug
# SDF's sc_tip_joint offset.  But they are STATIC: between two runs whose TCP
# moved 39 mm, they moved 0.7 mm, sitting ~30 cm away at table height.  That is a
# second plug, or a spawn-time pose, and either way it does not track the grasp.
# Deliberately NOT listed as candidates -- they resolve, which is exactly what
# makes them dangerous.
SC_CALIB_PLUG_FRAMES = [
    f.strip() for f in os.environ.get(
        "RL_INSERT_SC_CALIB_PLUG_FRAMES",
        "cable_0/sc_tip_link,sc_tip_link",
    ).split(",") if f.strip()
]

# A plug held in the gripper: the SFP default already puts its tip 58 mm out, and
# an SC plug is comparable.  Outside this band the frame is the wrist itself
# (nearer) or furniture (further), never the grasped plug.
SC_HELD_PLUG_MIN_M = _env_float("RL_INSERT_SC_HELD_PLUG_MIN_M", 0.02)
SC_HELD_PLUG_MAX_M = _env_float("RL_INSERT_SC_HELD_PLUG_MAX_M", 0.12)


def sc_calib_frame_candidates(task) -> list[str]:
    """Ground-truth tip frames to probe, most task-specific first.

    Mirrors ``DataCollectorScPlugPoseGT._tip_frame_candidates`` so calibration
    and plug-pose collection agree on where the truth lives.
    """
    cable = str(getattr(task, "cable_name", "") or "").strip()
    plug = str(getattr(task, "plug_name", "") or "").strip()
    candidates = []
    if cable and plug:
        candidates.append(f"{cable}/{plug}_link")
    if cable:
        candidates.append(f"{cable}/sc_tip_link")
    candidates.extend(SC_CALIB_PLUG_FRAMES)
    return list(dict.fromkeys(c for c in candidates if c))

SC_WRENCH_LOG_PERIOD_S = _env_float("RL_INSERT_SC_WRENCH_LOG_PERIOD_S", 0.25)
# 2026-07-27: an SC run aborted in 8.6s of a 144s budget having never moved
# toward the port, because all 7 perception frames landed 5.30-5.38px against
# the 5.0px SC_MAX_SELECT_REPROJ_PX gate -- measured as the 5-keypoint mean,
# which the near-symmetric, viewpoint-ambiguous corners dominate. Position
# comes solely from KP4 (the explicit mouth centre), which is roll-invariant
# and triangulated independently of the corners, so a noisy corner mean says
# nothing about whether the centre -- the actual motion target -- is right.
# Aborting before an attempt scores proximity only (~25 max); attempting and
# reaching even 1.39mm of insertion scores partial credit (38-50), so
# perception gates below default to RANKING candidates (by the trustworthy
# centre-only reprojection) rather than rejecting the frame outright when
# nothing clears the strict gate. Set this to restore the old reject-on-
# gate-miss behaviour everywhere it applies (select gate, consensus
# fallback, and the alignment-timeout-proceeds-to-seating below). The
# SC_MAX_HANDOFF_LATERAL_M wrong-port gate is NOT affected by this flag --
# it stays fatal regardless (a wrong-port insertion scores -12, far worse
# than aborting).
SC_STRICT_PERCEPTION = _env_bool("RL_INSERT_SC_STRICT_PERCEPTION", False)
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
SC_JOINT_FIT_TOP_K = max(1, _env_int("RL_INSERT_SC_JOINT_FIT_TOP_K", 12))
# Candidate selection answers only which mouth is under the plug.  Express the
# displacement in each candidate's port frame and gate its two lateral axes;
# handoff height belongs to the separate 120 mm safety gate below.
SC_MAX_HANDOFF_LATERAL_M = _env_float("RL_INSERT_SC_MAX_HANDOFF_LATERAL_M", 0.010)
SC_HANDOFF_MAX_DIST_M = _env_float("RL_INSERT_SC_HANDOFF_MAX_DIST_M", 0.120)
# How far "inside" the port the tip may compute to at handoff before the run is
# refused.  Zero is the physical truth -- the plug is outside until it is pushed
# in -- so this is pure tolerance for perception noise and a macro that hands off
# right at the mouth, not a licence to start partly inserted.
SC_MAX_HANDOFF_DEPTH_M = _env_float("RL_INSERT_SC_MAX_HANDOFF_DEPTH_M", 0.002)
# Broad corruption gate. Candidate scoring below compares against the one active
# 22.407 x 8.10 mm model contract; these bounds only reject degenerate DLT.
SC_MIN_OPENING_M = _env_float("RL_INSERT_SC_MIN_OPENING_M", 0.002)
SC_MAX_OPENING_M = _env_float("RL_INSERT_SC_MAX_OPENING_M", 0.030)

# Warn when the triangulated corners drift materially from the active physical
# mouth geometry. The explicit KP4 centre remains the motion target.
SC_OPENING_RESIDUAL_WARN_M = _env_float("RL_INSERT_SC_OPENING_RESIDUAL_WARN_M", 0.006)

# Search cyclic relabellings of each camera's four keypoints before triangulating
# (see _best_keypoint_correspondence).  Costs 4^(n_cameras-1) triangulations per
# combination -- 16 for three cameras -- against a failure that otherwise rejects
# every frame.  Set to 0 to get the old index-to-index behaviour back.
SC_KEYPOINT_ROLL_SEARCH = _env_bool("RL_INSERT_SC_KEYPOINT_ROLL_SEARCH", True)


@dataclass
class SCConfig:
    """SC seating limits.

    Every axial value is re-derived for the 15.64 mm bore.  Where a value is a
    fraction of the SFP equivalent the fraction is stated, because the SFP
    numbers are the only ones with field evidence behind them.
    """

    command_dt_sim_s: float = 0.05
    align_timeout_wall_s: float = 15.0
    # The binding vertical half-clearance is 0.725 mm.  The old 1.0 mm stop
    # could therefore declare success outside the physical opening even with a
    # perfect port estimate.  0.30 mm remains just above the observed ~0.27 mm
    # TF/noise floor while preserving 0.425 mm of binding-axis margin.
    align_lateral_tol_m: float = 0.0003
    align_rotation_tol_rad: float = np.deg2rad(2.0)
    align_max_lateral_step_m: float = 0.005
    align_mouth_max_lateral_step_m: float = 0.0015
    align_max_rotation_step_rad: float = np.deg2rad(1.5)

    # Legacy absolute colour correction.  This compensated for the virtual
    # 8.8 x 6.0 mm model's common-centre bias.  The active model directly
    # predicts the physical mouth and its centre, so the extra correction is
    # disabled by default but remains available for a controlled A/B.
    visual_align_enable: bool = False
    visual_align_interval_wall_s: float = 0.15
    visual_align_max_step_m: float = 0.0005
    visual_align_single_view_scale: float = 0.5
    visual_align_max_total_m: float = 0.003
    visual_align_max_view_disagree_m: float = 0.001
    visual_align_max_force_n: float = 1.5
    visual_align_search_scale: float = 1.65
    visual_align_consensus_samples: int = 7
    visual_align_consensus_min_agree: int = 4
    visual_align_consensus_spread_m: float = 0.00075
    visual_align_sample_dt_s: float = 0.10

    # Stall-only relative visual servo.  Unlike visual_align above, this never
    # replaces the frozen YOLO-derived target.  It compares blue side-band
    # occlusion against a pre-contact reference from the same camera and carries
    # a separate bounded offset through seating.  Motion requires two agreeing
    # cameras and light contact; weak or contradictory evidence stops recovery.
    visual_recovery_enable: bool = True
    visual_recovery_step_m: float = 0.00025
    visual_recovery_max_total_m: float = 0.002
    visual_recovery_min_force_n: float = 0.4
    visual_recovery_max_force_n: float = 1.5
    visual_recovery_force_n: float = 1.0
    visual_recovery_settle_wall_s: float = 0.20
    visual_recovery_frame_timeout_wall_s: float = 0.75
    visual_recovery_contact_timeout_wall_s: float = 1.0
    visual_recovery_meaningful_depth_m: float = 0.0005
    visual_recovery_min_views: int = 2
    visual_recovery_baseline_samples: int = 3
    visual_recovery_baseline_min_samples: int = 2

    # 0.8 mm on a 45.8 mm bore is 1.7%; the same fraction here is 0.27 mm, which
    # is near TF noise, so this sits above the proportional value deliberately.
    stall_progress_m: float = 0.0005
    stall_timeout_wall_s: float = 5.0
    seat_stall_grace_s: float = 3.0

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

    # Field evidence 2026-07-27 19:31: the scoring event fired 0.66 s AFTER
    # _seat() had already returned STALLED for the same run.  The plug seats
    # when the push stops, so bailing out of the stall path the instant
    # progress halts can discard a success that is only a beat away.
    stall_event_dwell_wall_s: float = 3.0

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
            align_lateral_tol_m=_env_float(
                "RL_INSERT_SC_ALIGN_LATERAL_TOL_M", 0.0003
            ),
            align_max_lateral_step_m=_env_float(
                "RL_INSERT_SC_ALIGN_MAX_LATERAL_STEP_M", 0.005
            ),
            visual_align_enable=_env_bool("RL_INSERT_SC_VISUAL_ALIGN_ENABLE", False),
            visual_align_interval_wall_s=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_INTERVAL_S", 0.15
            ),
            visual_align_max_step_m=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_MAX_STEP_M", 0.0005
            ),
            visual_align_single_view_scale=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_SINGLE_VIEW_SCALE", 0.5
            ),
            visual_align_max_total_m=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_MAX_TOTAL_M", 0.003
            ),
            visual_align_max_view_disagree_m=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_MAX_VIEW_DISAGREE_M", 0.001
            ),
            visual_align_max_force_n=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_MAX_FORCE_N", 1.5
            ),
            visual_align_search_scale=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_SEARCH_SCALE", 1.65
            ),
            visual_align_consensus_samples=_env_int(
                "RL_INSERT_SC_VISUAL_ALIGN_CONSENSUS_SAMPLES", 7
            ),
            visual_align_consensus_min_agree=_env_int(
                "RL_INSERT_SC_VISUAL_ALIGN_CONSENSUS_MIN_AGREE", 4
            ),
            visual_align_consensus_spread_m=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_CONSENSUS_SPREAD_M", 0.00075
            ),
            visual_align_sample_dt_s=_env_float(
                "RL_INSERT_SC_VISUAL_ALIGN_SAMPLE_DT_S", 0.10
            ),
            visual_recovery_enable=_env_bool(
                "RL_INSERT_SC_VISUAL_RECOVERY_ENABLE", True
            ),
            visual_recovery_step_m=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_STEP_M", 0.00025
            ),
            visual_recovery_max_total_m=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_MAX_TOTAL_M", 0.002
            ),
            visual_recovery_min_force_n=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_MIN_FORCE_N", 0.4
            ),
            visual_recovery_max_force_n=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_MAX_FORCE_N", 1.5
            ),
            visual_recovery_force_n=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_FORCE_N", 1.0
            ),
            visual_recovery_settle_wall_s=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_SETTLE_S", 0.20
            ),
            visual_recovery_frame_timeout_wall_s=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_FRAME_TIMEOUT_S", 0.75
            ),
            visual_recovery_contact_timeout_wall_s=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_CONTACT_TIMEOUT_S", 1.0
            ),
            visual_recovery_meaningful_depth_m=_env_float(
                "RL_INSERT_SC_VISUAL_RECOVERY_DEPTH_GAIN_M", 0.0005
            ),
            visual_recovery_min_views=_env_int(
                "RL_INSERT_SC_VISUAL_RECOVERY_MIN_VIEWS", 2
            ),
            visual_recovery_baseline_samples=_env_int(
                "RL_INSERT_SC_VISUAL_RECOVERY_BASELINE_SAMPLES", 3
            ),
            visual_recovery_baseline_min_samples=_env_int(
                "RL_INSERT_SC_VISUAL_RECOVERY_BASELINE_MIN_SAMPLES", 2
            ),
            stall_timeout_wall_s=_env_float("RL_INSERT_SC_STALL_TIMEOUT_S", 5.0),
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
            seat_stall_grace_s=_env_float("RL_INSERT_SC_SEAT_STALL_GRACE_S", 3.0),
            seat_overtravel_m=_env_float("RL_INSERT_SC_SEAT_OVERTRAVEL_M", 0.0015),
            seat_candidate_depth_m=_env_float("RL_INSERT_SC_SEAT_CANDIDATE_DEPTH_M", 0.0152),
            insertion_event_timeout_wall_s=_env_float("RL_INSERT_SC_EVENT_TIMEOUT_S", 6.0),
            stall_event_dwell_wall_s=_env_float(
                "RL_INSERT_SC_STALL_EVENT_DWELL_WALL_S", 3.0
            ),
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
        binding_half_clearance = (SC_OPENING_HEIGHT_M - SC_PLUG_HEIGHT_M) * 0.5
        if not 0.0 < self.align_lateral_tol_m <= binding_half_clearance:
            raise ValueError(
                "sc align tolerance must be positive and no larger than the "
                f"{binding_half_clearance*1000:.3f}mm binding half-clearance"
            )
        if self.align_max_lateral_step_m <= 0.0:
            raise ValueError("sc precontact align step must be positive")
        if self.align_mouth_max_lateral_step_m <= 0.0:
            raise ValueError("sc at-mouth align step must be positive")
        if self.visual_align_interval_wall_s < 0.0:
            raise ValueError("sc visual alignment interval must be non-negative")
        if not 0.0 < self.visual_align_max_step_m <= self.visual_align_max_total_m:
            raise ValueError("sc visual step must be positive and within its total cap")
        if not 0.0 < self.visual_align_max_total_m <= self.lateral_safety_m:
            raise ValueError("sc visual total cap must fit inside lateral safety")
        if not 0.0 < self.visual_align_single_view_scale <= 1.0:
            raise ValueError("sc visual single-view scale must be within (0, 1]")
        if self.visual_align_max_view_disagree_m <= 0.0:
            raise ValueError("sc visual view-disagreement gate must be positive")
        if self.visual_align_max_force_n < 0.0:
            raise ValueError("sc visual force gate must be non-negative")
        if not 1.0 < self.visual_align_search_scale <= 3.0:
            raise ValueError("sc visual search scale must be within (1, 3]")
        if self.visual_align_consensus_samples < 1:
            raise ValueError("sc visual consensus needs at least one sample")
        if not (
            1
            <= self.visual_align_consensus_min_agree
            <= self.visual_align_consensus_samples
        ):
            raise ValueError("sc visual min-agree must fit inside its sample count")
        if self.visual_align_consensus_spread_m <= 0.0:
            raise ValueError("sc visual consensus spread must be positive")
        if self.visual_align_sample_dt_s < 0.0:
            raise ValueError("sc visual sample interval must be non-negative")
        if not (
            0.0
            < self.visual_recovery_step_m
            <= self.visual_recovery_max_total_m
        ):
            raise ValueError(
                "sc visual recovery step must be positive and within its total cap"
            )
        if not (
            0.0
            < self.visual_recovery_max_total_m
            + self.seat_align_max_lat_m
            <= self.lateral_safety_m
        ):
            raise ValueError(
                "sc visual recovery plus wrench correction must fit lateral safety"
            )
        if not (
            0.0
            <= self.visual_recovery_min_force_n
            < self.visual_recovery_force_n
            <= self.visual_recovery_max_force_n
            < self.force_abort_n
        ):
            raise ValueError(
                "sc visual recovery force must lie between its contact gates"
            )
        if (
            self.visual_recovery_settle_wall_s < 0.0
            or self.visual_recovery_frame_timeout_wall_s <= 0.0
            or self.visual_recovery_contact_timeout_wall_s <= 0.0
        ):
            raise ValueError("sc visual recovery timing must be non-negative")
        if self.visual_recovery_meaningful_depth_m <= 0.0:
            raise ValueError("sc visual recovery depth gain must be positive")
        if self.visual_recovery_min_views < 2:
            raise ValueError("sc visual recovery requires at least two views")
        if self.visual_recovery_baseline_samples < 1:
            raise ValueError("sc visual recovery baseline needs at least one sample")
        if not (
            1
            <= self.visual_recovery_baseline_min_samples
            <= self.visual_recovery_baseline_samples
        ):
            raise ValueError(
                "sc visual recovery baseline min samples must fit its sample count"
            )
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


def sc_grasp_transform(policy):
    """Per-run measured ``(tip_in_tcp_pos, R_tcp_from_tip)``, or ``None``.

    Populated by ``prime_sc_plug_pose`` from the SC plug-pose model, exactly as
    ``prime_v50_plug_pose`` populates ``_v50_grasp_transform`` for SFP.
    """

    return getattr(policy, "_sc_grasp_transform", None)


def sc_tip_from_tcp(policy, tcp_pos, tcp_quat):
    """Tip pose through the measured per-grasp transform once it exists.

    Mirrors ``v50_tip_from_tcp``: before priming (unit tests, or the calib-dump
    diagnostics) the fixed-grasp constant keeps geometry helpers total, but
    ``run_sc_insertion`` refuses to start control until ``prime_sc_plug_pose``
    has succeeded, so no field motion is ever driven by the constant.
    """

    transform = sc_grasp_transform(policy)
    if transform is None:
        return sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
    return tip_from_tcp_transform(tcp_pos, tcp_quat, *transform)


def sc_tcp_pose_for_tip(policy, tip_pos, tip_rotation):
    """Inverse of ``sc_tip_from_tcp`` -- must use the same transform source.

    Returns ``(tcp_pos, R_tcp)`` with a rotation MATRIX like
    ``tcp_pose_for_sc_tip`` does.  Deliberately not v50's
    ``tcp_for_tip_transform``, which returns a quaternion -- handing that to
    ``_tcp_target``'s ``matrix_to_quat`` would silently corrupt every
    commanded pose on the primed path.
    """

    transform = sc_grasp_transform(policy)
    if transform is None:
        return tcp_pose_for_sc_tip(tip_pos, tip_rotation)
    tip_in_tcp_pos, R_tcp_from_tip = transform
    R_tip = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    R_tcp = R_tip @ np.asarray(R_tcp_from_tip, dtype=np.float64).reshape(3, 3).T
    tcp_pos = (
        np.asarray(tip_pos, dtype=np.float64).reshape(3)
        - R_tcp @ np.asarray(tip_in_tcp_pos, dtype=np.float64).reshape(3)
    )
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


def _sc_calib_dump_enabled() -> bool:
    """Read the flag at call time, not import time.

    RLInsert snapshots CALIB_DUMP into a module constant on import, which makes
    it untestable and unsettable by anything that loads after this module.
    """
    return _env_bool("RL_INSERT_CALIB_DUMP", False) or _env_bool(
        "RL_INSERT_SC_CALIB_DUMP", False)


def parse_tf_frame_names(text: str) -> list[str]:
    """Frame names out of either tf2 dump format.

    ``all_frames_as_yaml()`` puts each frame at column 0; ``all_frames_as_string()``
    emits ``Frame X exists with parent Y.``  Accept both rather than depending on
    which one a given tf2 build provides.
    """
    names = {line.split(":", 1)[0].strip()
             for line in text.splitlines()
             if ":" in line and line[:1] not in ("", " ", "\t", "-")}
    names |= set(re.findall(r"Frame (\S+?) exists", text))
    return sorted(n for n in names if n)


def _tf_frame_names(policy, log) -> list[str]:
    """Every frame the TF buffer knows about, or ``[]`` if it cannot be read."""
    node = getattr(policy, "_parent_node", None)
    buffer = getattr(node, "_tf_buffer", None)
    if buffer is None:
        log.warn("[sc-calib] no TF buffer reachable; cannot enumerate frames")
        return []

    text = ""
    for method in ("all_frames_as_yaml", "all_frames_as_string"):
        fn = getattr(buffer, method, None)
        if fn is None:
            continue
        try:
            text = str(fn() or "")
        except Exception as ex:
            log.warn(f"[sc-calib] {method}() failed: {ex}")
            continue
        if text.strip():
            break
    if not text.strip():
        log.warn("[sc-calib] TF buffer reported no frames at all")
        return []

    names = parse_tf_frame_names(text)
    if not names:
        log.warn(f"[sc-calib] could not parse frame names; raw TF dump follows:\n"
                 f"{text[:2000]}")
    return names


def _probe_tf_frames_for_tip(policy, log, lookup, tcp_pos, R_tcp, assumed_tip) -> bool:
    """Solve the tip transform against every plausible frame, best guess first.

    Printing a list of names and asking for another run wastes a grasp -- the
    name is only interesting because of the number behind it.  So resolve every
    candidate the sim actually publishes, solve the transform for each, and rank
    by distance to the assumed tip: the true tip is a few mm from where the SFP
    default puts it, while a wrist or board frame is tens of cm away.  One run
    then yields the answer whatever the frame turns out to be called.
    """
    names = _tf_frame_names(policy, log)
    if not names:
        return False

    # Identify the held plug by GEOMETRY, not by name.  Name matching failed
    # twice: 'cable_0/sc_tip_link' does not exist, and filtering to sc/tip/plug
    # names narrowed 63 frames to 9 -- none of which is the grasped plug, while
    # the other 54 were never even printed.  A plug in the gripper is a few cm
    # from the TCP no matter what it is called, so probe everything and sort by
    # distance to the TCP.  Ports and arm links are excluded only from the final
    # recommendation, never from the listing.
    log.warn(f"[sc-calib] probing all {len(names)} TF frames for the held plug")

    solved = []
    for frame in names:
        try:
            tf = lookup("base_link", frame, timeout_sec=0.05)
        except Exception:
            continue
        tr, ro = tf.transform.translation, tf.transform.rotation
        gt_pos = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        gt_quat = np.array([ro.w, ro.x, ro.y, ro.z], dtype=np.float64)
        stamp = getattr(getattr(tf, "header", None), "stamp", None)
        stamp_s = (float(getattr(stamp, "sec", 0))
                   + float(getattr(stamp, "nanosec", 0)) * 1e-9) if stamp else float("nan")
        solved.append({
            "frame": frame,
            "pos": gt_pos,
            "tcp_dist": float(np.linalg.norm(gt_pos - tcp_pos)),
            "p_rel": R_tcp.T @ (gt_pos - tcp_pos),
            "q_rel": matrix_to_quat(R_tcp.T @ quat_to_matrix(gt_quat)),
            "stamp": stamp_s,
        })

    if not solved:
        log.warn(f"[sc-calib] not one frame resolved. all frames: {names}")
        return False

    solved.sort(key=lambda s: s["tcp_dist"])
    log.warn(f"[sc-calib] === ALL FRAMES BY DISTANCE FROM TCP ({len(solved)} total) ===")
    for s in solved:
        log.warn(f"[sc-calib]   {s['tcp_dist'] * 1000:8.1f}mm  {s['frame']}  "
                 f"pos={np.round(s['pos'], 4).tolist()} stamp={s['stamp']:.3f}")

    # A held SC plug tip sits roughly 40-90 mm from the TCP: closer is the wrist
    # or the gripper itself, further is furniture.  Anything named for a port or
    # an arm link is the thing we insert into or the thing doing the inserting.
    def _is_candidate(s) -> bool:
        low = s["frame"].lower()
        if any(k in low for k in ("port", "arm_", "flange", "wrist", "gripper",
                                  "finger", "camera", "base_link", "world")):
            return False
        return SC_HELD_PLUG_MIN_M <= s["tcp_dist"] <= SC_HELD_PLUG_MAX_M

    candidates = [s for s in solved if _is_candidate(s)]
    if not candidates:
        log.warn("[sc-calib] >>> NO frame sits 20-120mm from the TCP, so the sim "
                 "does not publish a pose that tracks the grasped plug. "
                 "Ground-truth calibration is not available this way -- use the "
                 "SC plug-pose model instead, or derive the tip from the plug "
                 "body: the SC Plug SDF fixes sc_tip_link at plug-frame "
                 "x=+11.65mm rpy=(-pi/2, 0, -pi/2) (see sc_plug_pose_geometry).")
        return False

    log.warn("[sc-calib] === HELD-PLUG CANDIDATES (20-120mm from TCP) ===")
    for s in candidates:
        log.warn(f"[sc-calib]   {s['frame']}  {s['tcp_dist'] * 1000:.1f}mm from TCP")
        log.warn(f"[sc-calib]     RL_INSERT_SC_TIP_IN_TCP_POS ={np.round(s['p_rel'], 10).tolist()}")
        log.warn(f"[sc-calib]     RL_INSERT_SC_TIP_IN_TCP_QUAT={np.round(s['q_rel'], 10).tolist()}")
    log.warn(f"[sc-calib] >>> USE: RL_INSERT_SC_CALIB_PLUG_FRAMES={candidates[0]['frame']}")
    return True


def dump_sc_grasp_calibration(policy, task) -> bool:
    """Log everything needed to re-solve ``SC_TIP_IN_TCP_*``.  Logs only.

    RLInsert's ``_dump_grasp_calibration`` is unreachable on the SC path -- the
    ``plug_type == "sc"`` branch returns into ``run_sc_insertion`` well before
    it -- and it probes SFP frames and prints SFP constant names.  So
    ``RL_INSERT_CALIB_DUMP=1`` on an SC run produced nothing at all, which would
    have made 6c impossible to actually perform.

    Returns True if a ground-truth frame resolved.
    """
    log = policy.get_logger()
    try:
        tcp_pos, tcp_quat = policy._tcp()
    except Exception as ex:
        log.error(f"[sc-calib] cannot read TCP: {ex}")
        return False
    tcp_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
    tcp_quat = np.asarray(tcp_quat, dtype=np.float64).reshape(4)
    R_tcp = quat_to_matrix(tcp_quat)
    tip_pos, R_tip = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)

    log.info("[sc-calib] === SC GRASP CALIBRATION DUMP (base_link) ===")
    log.info(f"[sc-calib] TCP pos={np.round(tcp_pos, 6).tolist()} "
             f"quat_wxyz={np.round(tcp_quat, 6).tolist()}")
    log.info(f"[sc-calib] ASSUMED tip (current transform) "
             f"pos={np.round(tip_pos, 6).tolist()} "
             f"quat_wxyz={np.round(matrix_to_quat(R_tip), 6).tolist()}")
    log.info(f"[sc-calib] current SC_TIP_IN_TCP_POS={SC_TIP_IN_TCP_POS.tolist()} "
             f"QUAT={SC_TIP_IN_TCP_QUAT.tolist()} calibrated={SC_TIP_CALIBRATED}")

    frames = sc_calib_frame_candidates(task)

    lookup = getattr(policy, "_lookup_transform", None)
    if lookup is None:
        log.error("[sc-calib] policy has no _lookup_transform; cannot resolve ground truth")
        return False

    found = False
    for frame in frames:
        try:
            tf = lookup("base_link", frame, timeout_sec=0.3)
        except Exception:
            continue
        tr, ro = tf.transform.translation, tf.transform.rotation
        gt_pos = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        gt_quat = np.array([ro.w, ro.x, ro.y, ro.z], dtype=np.float64)
        # Resolving is not the same as being right.  'selected_sc/sc_tip_link'
        # resolves fine and is a genuine SC plug, but it is a STATIC frame 30 cm
        # away: across two runs the TCP moved 39 mm and it moved 0.7 mm.  A frame
        # that cannot be the plug in the gripper must never be printed as SOLVED,
        # or the number gets baked in and the fake seat survives calibration.
        tcp_dist = float(np.linalg.norm(gt_pos - tcp_pos))
        if not (SC_HELD_PLUG_MIN_M <= tcp_dist <= SC_HELD_PLUG_MAX_M):
            log.warn(f"[sc-calib] REJECTED '{frame}': {tcp_dist * 1000:.0f}mm from the "
                     f"TCP, outside the {SC_HELD_PLUG_MIN_M * 1000:.0f}-"
                     f"{SC_HELD_PLUG_MAX_M * 1000:.0f}mm a held plug can occupy. "
                     f"pos={np.round(gt_pos, 6).tolist()} -- not the grasped plug.")
            continue
        q_rel = matrix_to_quat(R_tcp.T @ quat_to_matrix(gt_quat))
        p_rel = R_tcp.T @ (gt_pos - tcp_pos)
        found = True
        log.info(f"[sc-calib] GROUND-TRUTH frame '{frame}' RESOLVED: "
                 f"pos={np.round(gt_pos, 6).tolist()} "
                 f"{tcp_dist * 1000:.1f}mm from TCP "
                 f"quat_wxyz={np.round(gt_quat, 6).tolist()}")
        log.info(f"[sc-calib]   >>> SOLVED RL_INSERT_SC_TIP_IN_TCP_POS ="
                 f"{np.round(p_rel, 10).tolist()}")
        log.info(f"[sc-calib]   >>> SOLVED RL_INSERT_SC_TIP_IN_TCP_QUAT="
                 f"{np.round(q_rel, 10).tolist()}")
        log.info("[sc-calib]   >>> collect these over ~10 grasps; the SPREAD is the "
                 "measurement that decides whether a fixed transform is enough "
                 "(budget against 0.725 mm vertical clearance, not 1.2 mm lateral), "
                 "then set both defaults in BOTH copies of sc_controller.py")
    if not found:
        log.warn(f"[sc-calib] no ground-truth frame resolved from {frames}.")
        found = _probe_tf_frames_for_tip(policy, log, lookup, tcp_pos, R_tcp, tip_pos)
    log.info("[sc-calib] === END DUMP ===")
    return found


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

    # 0 means "bounded by the action deadline alone", matching the SFP ladder
    # (v50_controller.max_wedge_retries).  The ladder alternates cheap and full
    # recovery -- unload, full retry, unload, full retry, ... -- for as long as
    # the remaining budget can afford the next rung, so the attempt count is a
    # consequence of ACTION_TIME_BUDGET_S rather than a separate cap.
    MAX_SEAT_ATTEMPTS = 0
    # Relative back-off from the stalled depth, NOT a return to standoff: 1.5 mm
    # is enough to unload the wedge in place while keeping the plug in the mouth.
    # The full-retry rung is the one that goes all the way back to standoff.
    RETRY_UNLOAD_M = 0.0015
    RETRY_UNLOAD_HOLD_WALL_S = 2.0
    # Port re-perception cost, measured: the 2026-07-28 log spent 9.58 s wall on
    # a 7-sample consensus.  Budgeted explicitly so a full retry cannot start
    # without room for it.
    RETRY_REPERCEIVE_BUDGET_S = 12.0
    # Rejects a mid-retry jump to the neighbouring port -- see
    # _retry_reperceive_port.
    RETRY_REPERCEIVE_MAX_SHIFT_M = 0.010
    # Alignment reservation for a retry.  NOT align_timeout_wall_s: that is 60 s
    # deployed, while post-ba01eda alignment converges in 2-6 s of wall time, so
    # reserving the timeout would refuse retries that comfortably fit.  Under-
    # reserving is the safe direction here -- _align calls
    # _enforce_action_deadline every iteration, so the global budget still hard-
    # bounds it, and a retry cut short scores no worse than one never started.
    RETRY_ALIGN_BUDGET_S = 15.0

    STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
    DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
    # Precontact alignment is still free-space motion, so it can use a stiffer
    # translational impedance to overcome the cable bias. Seating and recovery
    # use separate contact-phase stiffness values below.
    SEAT_LATERAL_STIFFNESS_N_M = 90.0
    PRECONTACT_ALIGN_STIFFNESS = [
        _env_float("RL_INSERT_SC_PRECONTACT_STIFFNESS_N_M", 800.0),
        _env_float("RL_INSERT_SC_PRECONTACT_STIFFNESS_N_M", 800.0),
        _env_float("RL_INSERT_SC_PRECONTACT_STIFFNESS_N_M", 800.0),
        80.0,
        80.0,
        80.0,
    ]
    PRECONTACT_ALIGN_DAMPING = [
        _env_float("RL_INSERT_SC_PRECONTACT_DAMPING_N_S_M", 160.0),
        _env_float("RL_INSERT_SC_PRECONTACT_DAMPING_N_S_M", 160.0),
        _env_float("RL_INSERT_SC_PRECONTACT_DAMPING_N_S_M", 160.0),
        30.0,
        30.0,
        30.0,
    ]
    HOLD_STIFFNESS = [200.0, 200.0, 200.0, 80.0, 80.0, 80.0]
    HOLD_DAMPING = [80.0, 80.0, 80.0, 30.0, 30.0, 30.0]

    def __init__(self, policy, task, get_observation, move_robot, send_feedback,
                 *, port_pos, port_quat, Rp, Rs=None, config=None, timing=None):
        self.policy = policy
        self.task = task
        self.get_observation = get_observation
        self.move_robot = move_robot
        self.send_feedback = send_feedback
        self.config = (config or SCConfig.from_env()).validated()
        self.port_pos = np.asarray(port_pos, dtype=np.float64).reshape(3)
        self._visual_origin_port_pos = self.port_pos.copy()
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
        self._visual_last_attempt_wall_s = float("-inf")
        self._visual_last_miss_log_wall_s = float("-inf")
        self._visual_mask_bank = None
        self._visual_last_correction_xy = np.zeros(2, dtype=np.float64)
        self._visual_samples_xy = []
        self._visual_target_locked = False
        self._visual_recovery_last_frame_stamps = {}
        self._visual_recovery_last_failure_reason = None
        # Frozen only after the visual target is frozen, so recovery rectifies
        # both the clean reference and the stall frame in exactly one port frame.
        self._visual_recovery_baseline = {}
        self._visual_recovery_baseline_last_frame_stamps = {}
        self.timing = timing
        # A pre-contact segment remains fixed until the arm reaches it.
        # Recomputing a target from the current tip every 50 ms caps steady
        # lateral force at 90 N/m * 1.5 mm and made large handoff errors crawl
        # into the 15 s alignment timeout.
        self._align_segment_lateral_xy = None

    def _seating_stiffness(self) -> np.ndarray:
        """Return a base-frame matrix with 500 N/m along the port axis.

        The policy serializes full 6x6 matrices directly.  Rotating only the
        translational block preserves 90 N/m lateral compliance while matching
        SCConfig's axial force-lead model.
        """

        translation_port = np.diag(
            [
                self.SEAT_LATERAL_STIFFNESS_N_M,
                self.SEAT_LATERAL_STIFFNESS_N_M,
                self.config.axial_stiffness_n_m,
            ]
        )
        translation_base = self.Rp @ translation_port @ self.Rp.T
        translation_base = 0.5 * (translation_base + translation_base.T)
        stiffness = np.zeros((6, 6), dtype=np.float64)
        stiffness[:3, :3] = translation_base
        stiffness[3:, 3:] = np.diag(self.STIFFNESS[3:])
        return stiffness

    def _seating_axial_stiffness_n_m(self) -> float:
        """Return the scalar stiffness seen along the insertion axis."""

        stiffness = self._seating_stiffness()
        axis = self.Rp[:, 2]
        return float(axis @ stiffness[:3, :3] @ axis)

    # --------------------------------------------------------- visual target
    def _visual_due(self, now: Optional[float] = None) -> bool:
        if (
            not self.config.visual_align_enable
            or getattr(self, "_visual_target_locked", False)
            or not hasattr(self, "get_observation")
        ):
            return False
        now = time.monotonic() if now is None else float(now)
        return (
            now - getattr(self, "_visual_last_attempt_wall_s", float("-inf"))
            >= self.config.visual_align_interval_wall_s
        )

    def _visual_bore_quads_world(self) -> np.ndarray:
        """Projected CAD rectangles for the two physical SC bores."""

        quads = []
        for center_x in (+SC_BORE_PITCH_M * 0.5, -SC_BORE_PITCH_M * 0.5):
            local = np.array(
                [
                    [center_x + SC_BORE_WIDTH_M * 0.5, +SC_OPENING_HEIGHT_M * 0.5, 0.0],
                    [center_x - SC_BORE_WIDTH_M * 0.5, +SC_OPENING_HEIGHT_M * 0.5, 0.0],
                    [center_x - SC_BORE_WIDTH_M * 0.5, -SC_OPENING_HEIGHT_M * 0.5, 0.0],
                    [center_x + SC_BORE_WIDTH_M * 0.5, -SC_OPENING_HEIGHT_M * 0.5, 0.0],
                ],
                dtype=np.float64,
            )
            quads.append(self.port_pos + (self.Rp @ local.T).T)
        return np.asarray(quads, dtype=np.float64)

    def _visual_ignored_pixels(self, camera: str, image_shape):
        if self._visual_mask_bank is None:
            try:
                from aic_perception.gripper_masks import GripperMaskBank

                self._visual_mask_bank = GripperMaskBank()
            except Exception as exc:
                self._visual_mask_bank = False
                self.log.warn(
                    f"[sc] SC_VISUAL gripper masks unavailable ({type(exc).__name__}); "
                    "continuing with geometry/color gates"
                )
        if self._visual_mask_bank is False:
            return None
        try:
            return self._visual_mask_bank.ignored_pixels(camera, image_shape)
        except Exception:
            return None

    def _visual_miss(self, now: float, phase: str, reason: str) -> None:
        # Missing imagery is expected as the plug occludes the mouth.  Throttle
        # it and, critically, retain the last accepted target without aborting.
        if now - self._visual_last_miss_log_wall_s >= 0.75:
            self.log.warn(
                f"[sc] SC_VISUAL_NO_CORRECTION phase={phase} reason={reason} "
                "action=retain_last_target_and_continue"
            )
            self._visual_last_miss_log_wall_s = now

    def _visual_refine_port(
        self,
        observation,
        *,
        force_n: float = float("nan"),
        phase: str,
        force_sample: bool = False,
    ) -> bool:
        """Collect one physical-mouth sample; never raises or aborts control."""

        now = time.monotonic()
        if getattr(self, "_visual_target_locked", False):
            return False
        if not force_sample and not self._visual_due(now):
            return False
        self._visual_last_attempt_wall_s = now
        if observation is None:
            self._visual_miss(now, phase, "no_observation")
            return False
        if np.isfinite(force_n) and force_n > self.config.visual_align_max_force_n:
            self._visual_miss(now, phase, f"contact_force_{force_n:.2f}N")
            return False
        try:
            views = self.policy._build_views(observation)
        except Exception as exc:
            self._visual_miss(now, phase, f"build_views_{type(exc).__name__}")
            return False
        if not views:
            self._visual_miss(now, phase, "no_camera_views")
            return False

        bore_world = self._visual_bore_quads_world()
        hits = []
        bore_counts = {}
        rejection_reasons = {}
        for camera, (bgr, K, T_cam_from_world) in views.items():
            try:
                P = self.policy._pc.build_projection_matrix(K, T_cam_from_world)
                projected = [
                    project_sc_visual_point_px(P, point)
                    for point in bore_world.reshape(-1, 3)
                ]
                if any(uv is None for uv in projected):
                    rejection_reasons[camera] = "projection"
                    continue
                bore_quads_uv = np.asarray(projected, dtype=np.float64).reshape(2, 4, 2)
                ignored = self._visual_ignored_pixels(camera, bgr.shape)
                diagnostics = {}
                detection = detect_sc_duplex_opening(
                    bgr,
                    bore_quads_uv,
                    ignored,
                    diagnostics=diagnostics,
                    search_scale=self.config.visual_align_search_scale,
                )
                if detection is None:
                    rejection_reasons[camera] = diagnostics.get("reason", "not_found")
                    continue
                plane_point = sc_visual_ray_to_plane(
                    detection.center_uv,
                    K,
                    T_cam_from_world,
                    plane_point=self.port_pos,
                    plane_normal=self.Rp[:, 2],
                )
                if plane_point is None:
                    rejection_reasons[camera] = "ray_plane"
                    continue
                hits.append({"camera": camera, "plane_point": plane_point})
                bore_counts[camera] = detection.detected_bores
            except Exception as exc:
                rejection_reasons[camera] = f"exception_{type(exc).__name__}"

        estimate = fuse_sc_opening_hits(
            hits,
            origin_port_pos=self._visual_origin_port_pos,
            Rp=self.Rp,
            max_view_disagreement_m=self.config.visual_align_max_view_disagree_m,
            max_total_offset_m=self.config.visual_align_max_total_m,
            allow_single_view=True,
        )
        if estimate is None:
            reason = (
                "view_disagreement"
                if len(hits) >= 2
                else f"visible_views_{len(hits)}:{rejection_reasons}"
            )
            self._visual_miss(now, phase, reason)
            return False

        sample_xy = (
            self.Rp.T
            @ (
                np.asarray(estimate.point_world, dtype=np.float64)
                - self._visual_origin_port_pos
            )
        )[:2]
        if not np.all(np.isfinite(sample_xy)):
            self._visual_miss(now, phase, "nonfinite_sample")
            return False
        if estimate.single_view:
            sample_xy *= self.config.visual_align_single_view_scale
        self._visual_samples_xy.append(np.asarray(sample_xy, dtype=np.float64))
        disagree_mm = (
            "single"
            if estimate.single_view
            else f"{estimate.disagreement_m * 1000.0:.2f}"
        )
        self.log.info(
            f"[sc] SC_VISUAL_SAMPLE phase={phase} "
            f"sample={len(self._visual_samples_xy)}/"
            f"{self.config.visual_align_consensus_samples} "
            f"cameras={list(estimate.cameras)} "
            f"bores={bore_counts} disagreement_mm={disagree_mm} "
            f"offset_from_pose_mm={np.round(sample_xy * 1000.0, 3).tolist()}"
        )
        return True

    def _prime_visual_recovery_baseline(self) -> bool:
        """Freeze per-camera blue side-band references before approach motion.

        This deliberately runs after ``_finalize_visual_target`` while the arm
        is still stationary.  A baseline from the raw YOLO pose would be
        rectified in a different frame when visual pre-alignment corrects the
        target, which can itself look like a plug offset at the later stall.
        """

        self._visual_recovery_baseline = {}
        self._visual_recovery_baseline_last_frame_stamps = {}
        if (
            not self.config.visual_recovery_enable
            or not hasattr(self, "get_observation")
        ):
            return False

        # Keep the immutable, rectified support masks as well as the scalar
        # fractions.  A baseline made from fractions alone cannot distinguish a
        # stable calibrated gripper mask from a newly occluded side band at a
        # later stall.  The detector pairs this frozen support with the current
        # frame and fails closed if it changed.
        samples: dict[str, dict[int, list[ScBlueSideSignature]]] = {}
        last_stamps: dict[str, float] = {}
        for index in range(self.config.visual_recovery_baseline_samples):
            self.policy._enforce_action_deadline(self.move_robot)
            observation = self.get_observation()
            rejections = {}
            accepted = []
            if observation is None:
                rejections["observation"] = "missing"
            else:
                try:
                    views = self.policy._build_views(observation)
                except Exception as exc:
                    views = {}
                    rejections["views"] = f"build_{type(exc).__name__}"
                if views:
                    frame_stamps = _sc_camera_frame_stamps(observation)
                    bore_world = self._visual_bore_quads_world()
                    for camera, (bgr, K, T_cam_from_world) in views.items():
                        try:
                            stamp_s = frame_stamps.get(camera)
                            if stamp_s is None:
                                rejections[camera] = "missing_frame_stamp"
                                continue
                            if stamp_s <= last_stamps.get(camera, float("-inf")) + 1e-9:
                                rejections[camera] = "stale_frame"
                                continue
                            projection = self.policy._pc.build_projection_matrix(
                                K, T_cam_from_world
                            )
                            projected = [
                                project_sc_visual_point_px(projection, point)
                                for point in bore_world.reshape(-1, 3)
                            ]
                            if any(uv is None for uv in projected):
                                rejections[camera] = "projection"
                                continue
                            bore_quads_uv = np.asarray(
                                projected, dtype=np.float64
                            ).reshape(2, 4, 2)
                            ignored = self._visual_ignored_pixels(camera, bgr.shape)
                            if ignored is None:
                                rejections[camera] = "gripper_mask_unavailable"
                                continue
                            signatures = {}
                            for band in (10, 12):
                                diagnostics = {}
                                signature = measure_sc_blue_side_signature(
                                    bgr,
                                    bore_quads_uv,
                                    ignored,
                                    band_half_width_px=band,
                                    diagnostics=diagnostics,
                                )
                                if signature is None:
                                    rejections[camera] = diagnostics.get(
                                        "reason", "side_measurement_failed"
                                    )
                                    break
                                signatures[band] = signature
                            if len(signatures) != 2:
                                continue
                            camera_samples = samples.setdefault(
                                camera, {10: [], 12: []}
                            )
                            for band, signature in signatures.items():
                                camera_samples[band].append(signature)
                            last_stamps[camera] = stamp_s
                            accepted.append(camera)
                        except Exception as exc:
                            rejections[camera] = f"exception_{type(exc).__name__}"
            self.log.info(
                f"[sc] SC_VISUAL_RECOVERY_BASELINE_SAMPLE "
                f"sample={index + 1}/{self.config.visual_recovery_baseline_samples} "
                f"accepted={accepted} rejected={rejections}"
            )
            if (
                index + 1 < self.config.visual_recovery_baseline_samples
                and self.config.visual_align_sample_dt_s > 0.0
            ):
                self.policy.sleep_for(self.config.visual_align_sample_dt_s)

        baseline = {}
        insufficient = {}
        for camera, per_band in samples.items():
            counts = {band: len(values) for band, values in per_band.items()}
            if any(
                count < self.config.visual_recovery_baseline_min_samples
                for count in counts.values()
            ):
                insufficient[camera] = f"samples={counts}"
                continue
            aggregate = {
                band: aggregate_sc_blue_side_signatures(values)
                for band, values in per_band.items()
            }
            if any(signature is None for signature in aggregate.values()):
                insufficient[camera] = "incompatible_or_missing_support"
                continue
            # The downstream detector also checks this.  Reject early so the
            # operator can see that the reference itself was not blue enough.
            if any(
                np.any(signature.blue_fractions < 0.50)
                for signature in aggregate.values()
            ):
                insufficient[camera] = "weak_blue_side_reference"
                continue
            baseline[camera] = aggregate
            self.log.info(
                f"[sc] SC_VISUAL_RECOVERY_BASELINE camera={camera} "
                f"samples={counts} side_visibility="
                f"{{10: {np.round(aggregate[10].blue_fractions, 3).tolist()}, "
                f"12: {np.round(aggregate[12].blue_fractions, 3).tolist()}}}"
            )

        self._visual_recovery_baseline_last_frame_stamps = last_stamps
        if len(baseline) < self.config.visual_recovery_min_views:
            self.log.warn(
                f"[sc] SC_VISUAL_RECOVERY_BASELINE_UNAVAILABLE "
                f"usable_views={len(baseline)} "
                f"need={self.config.visual_recovery_min_views} "
                f"rejected={insufficient} action=disable_recovery_only"
            )
            return False
        self._visual_recovery_baseline = baseline
        self.log.info(
            f"[sc] SC_VISUAL_RECOVERY_BASELINE_READY "
            f"cameras={list(baseline)}"
        )
        return True

    def _visual_recovery_estimate(self, observation):
        """Return a relative port-frame escape direction from agreeing views."""

        self._visual_recovery_last_failure_reason = None
        if observation is None:
            self.log.warn(
                "[sc] SC_VISUAL_RECOVERY_NO_MOTION reason=no_observation"
            )
            return None
        try:
            views = self.policy._build_views(observation)
        except Exception as exc:
            self.log.warn(
                f"[sc] SC_VISUAL_RECOVERY_NO_MOTION "
                f"reason=build_views_{type(exc).__name__}"
            )
            return None
        if not views:
            self.log.warn(
                "[sc] SC_VISUAL_RECOVERY_NO_MOTION reason=no_camera_views"
            )
            return None

        # port_pos is the frozen pre-alignment target.  Recovery never mutates
        # it, so each image is rectified in one stable physical frame.
        frame_stamps = _sc_camera_frame_stamps(observation)
        frame_timing = _sc_frame_timing_fields(self.policy, frame_stamps)
        last_frame_stamps = dict(
            getattr(self, "_visual_recovery_last_frame_stamps", {})
        )
        baseline_by_camera = dict(
            getattr(self, "_visual_recovery_baseline", {})
        )
        bore_world = self._visual_bore_quads_world()
        evidence_by_view = []
        rejection_reasons = {}
        fresh_cameras = []
        for camera, (bgr, K, T_cam_from_world) in views.items():
            try:
                stamp_s = frame_stamps.get(camera)
                if stamp_s is None:
                    rejection_reasons[camera] = "missing_frame_stamp"
                    continue
                previous_stamp_s = last_frame_stamps.get(camera)
                if (
                    previous_stamp_s is not None
                    and stamp_s <= previous_stamp_s + 1e-9
                ):
                    rejection_reasons[camera] = "stale_frame"
                    continue
                fresh_cameras.append(camera)
                projection = self.policy._pc.build_projection_matrix(
                    K, T_cam_from_world
                )
                projected = [
                    project_sc_visual_point_px(projection, point)
                    for point in bore_world.reshape(-1, 3)
                ]
                if any(uv is None for uv in projected):
                    rejection_reasons[camera] = "projection"
                    continue
                bore_quads_uv = np.asarray(
                    projected, dtype=np.float64
                ).reshape(2, 4, 2)
                diagnostics = {}
                ignored = self._visual_ignored_pixels(camera, bgr.shape)
                if ignored is None:
                    rejection_reasons[camera] = "gripper_mask_unavailable"
                    continue
                baseline = baseline_by_camera.get(camera)
                if baseline is None:
                    rejection_reasons[camera] = "baseline_unavailable"
                    continue
                evidence = detect_sc_recovery_direction(
                    bgr,
                    bore_quads_uv,
                    ignored,
                    baseline_blue_fractions=baseline,
                    diagnostics=diagnostics,
                )
                self.log.info(
                    f"[sc] SC_VISUAL_RECOVERY_SUPPORT camera={camera} "
                    f"reason={diagnostics.get('reason', 'unknown')} "
                    f"{_sc_recovery_support_summary(diagnostics)}"
                )
                if evidence is None:
                    rejection_reasons[camera] = diagnostics.get(
                        "reason", "not_found"
                    )
                    continue
                evidence_by_view.append(
                    {"camera": camera, "evidence": evidence}
                )
                self.log.info(
                    f"[sc] SC_VISUAL_RECOVERY_VIEW camera={camera} "
                    f"direction={np.round(evidence.direction_xy, 3).tolist()} "
                    f"side_visibility_ratio="
                    f"{np.round(evidence.margins, 3).tolist()} "
                    f"confidence={evidence.confidence:.2f} "
                    f"balanced={evidence.balanced}"
                )
            except Exception as exc:
                rejection_reasons[camera] = f"exception_{type(exc).__name__}"

        self.log.info(
            f"[sc] SC_VISUAL_RECOVERY_VIEWS "
            f"accepted={[item['camera'] for item in evidence_by_view]} "
            f"rejected={rejection_reasons} {frame_timing} "
            "action_budget_remaining_s="
            f"{_sc_action_budget_remaining_s(self.policy):.3f}"
        )
        estimate = fuse_sc_recovery_evidence(
            evidence_by_view,
            min_views=self.config.visual_recovery_min_views,
        )
        if estimate is None:
            if (
                len(fresh_cameras) < self.config.visual_recovery_min_views
                and rejection_reasons
                and all(
                    reason == "stale_frame"
                    for reason in rejection_reasons.values()
                )
            ):
                self._visual_recovery_last_failure_reason = "stale_frames"
            else:
                self._visual_recovery_last_failure_reason = "weak_evidence"
            self.log.warn(
                f"[sc] SC_VISUAL_RECOVERY_NO_MOTION "
                f"reason={self._visual_recovery_last_failure_reason} "
                f"accepted_views={len(evidence_by_view)} "
                f"rejections={rejection_reasons}"
            )
            return None
        for camera, stamp_s in frame_stamps.items():
            if camera in views:
                last_frame_stamps[camera] = stamp_s
        self._visual_recovery_last_frame_stamps = last_frame_stamps
        self.log.info(
            f"[sc] SC_VISUAL_RECOVERY_FUSED cameras={list(estimate.cameras)} "
            f"direction={np.round(estimate.direction_xy, 3).tolist()} "
            f"confidence={estimate.confidence:.2f} "
            f"resultant={estimate.resultant:.2f} balanced={estimate.balanced} "
            f"{frame_timing}"
        )
        return estimate

    def _finalize_visual_target(self, *, phase: str) -> bool:
        """Robustly choose one target, then freeze it for this insertion."""

        self._visual_target_locked = True
        samples = np.asarray(
            getattr(self, "_visual_samples_xy", []), dtype=np.float64
        ).reshape(-1, 2)
        minimum = self.config.visual_align_consensus_min_agree
        if len(samples) < minimum:
            self.log.warn(
                f"[sc] SC_VISUAL_LOCK phase={phase} accepted={len(samples)}/"
                f"{self.config.visual_align_consensus_samples} need={minimum} "
                "action=freeze_raw_pose_and_continue"
            )
            return False

        provisional = np.median(samples, axis=0)
        deviations = np.linalg.norm(samples - provisional, axis=1)
        inlier_mask = deviations <= self.config.visual_align_consensus_spread_m
        degraded = int(np.count_nonzero(inlier_mask)) < minimum
        if degraded:
            # Still work with what the cameras provided: select the samples
            # closest to the temporal median instead of chasing every frame or
            # abandoning the insertion.
            closest = np.argsort(deviations)[:minimum]
            kept = samples[closest]
        else:
            kept = samples[inlier_mask]
        locked_xy = np.median(kept, axis=0)
        spread = float(np.max(np.linalg.norm(kept - locked_xy, axis=1)))
        observed = (
            self._visual_origin_port_pos
            + self.Rp[:, 0] * locked_xy[0]
            + self.Rp[:, 1] * locked_xy[1]
        )
        # The target changes once while the robot is stationary.  Subsequent
        # robot motion is still bounded by align_max_lateral_step_m; using the
        # total visual cap here avoids leaving half of a stable measured bias.
        update = bounded_visual_port_update(
            self.port_pos,
            self._visual_origin_port_pos,
            observed,
            self.Rp,
            max_step_m=self.config.visual_align_max_total_m,
            max_total_m=self.config.visual_align_max_total_m,
        )
        if update is None:
            self.log.warn(
                f"[sc] SC_VISUAL_LOCK phase={phase} invalid consensus "
                "action=freeze_raw_pose_and_continue"
            )
            return False
        target, step_xy = update
        self.port_pos = target
        self._visual_last_correction_xy = np.asarray(step_xy, dtype=np.float64)
        total_xy = (self.Rp.T @ (self.port_pos - self._visual_origin_port_pos))[:2]
        self.log.info(
            f"[sc] SC_VISUAL_LOCK phase={phase} accepted={len(samples)}/"
            f"{self.config.visual_align_consensus_samples} kept={len(kept)} "
            f"degraded_temporal={degraded} spread_mm={spread*1000.0:.3f} "
            f"locked_offset_mm={np.round(total_xy * 1000.0, 3).tolist()} "
            "target_frozen=true"
        )
        return True

    def _prime_visual_target(self) -> bool:
        """Collect a stationary temporal batch and lock one visual target."""

        if not hasattr(self, "get_observation"):
            self._visual_target_locked = True
            return False
        if getattr(self, "_visual_target_locked", False):
            return bool(np.any(getattr(
                self, "_visual_last_correction_xy", np.zeros(2, dtype=np.float64)
            )))
        corrected = False
        if self.config.visual_align_enable:
            self._visual_samples_xy = []
            for index in range(self.config.visual_align_consensus_samples):
                self.policy._enforce_action_deadline(self.move_robot)
                self._visual_refine_port(
                    self.get_observation(),
                    phase="prealign",
                    force_sample=True,
                )
                if (
                    index + 1 < self.config.visual_align_consensus_samples
                    and self.config.visual_align_sample_dt_s > 0.0
                ):
                    self.policy.sleep_for(self.config.visual_align_sample_dt_s)
            corrected = self._finalize_visual_target(phase="prealign")
        else:
            # Recovery remains useful when absolute visual pre-alignment is
            # disabled for an experiment, but it still needs one immutable
            # rectification frame for its own clean reference.
            self._visual_target_locked = True
        self._prime_visual_recovery_baseline()
        return corrected

    # ------------------------------------------------------------- geometry
    def _tip_pose(self):
        # Pure kinematics through the per-grasp transform prime_sc_plug_pose
        # measured -- total and non-failing, so the align/seat hot loops never
        # see a vision dropout mid-motion.  The estimator is NOT re-run here.
        tcp_pos, tcp_quat = self.policy._tcp()
        return sc_tip_from_tcp(self.policy, tcp_pos, tcp_quat)

    def _tcp_target(self, tip_pos, tip_rotation):
        # geometry_msgs is imported lazily so the pure geometry in this module
        # stays importable (and unit-testable) without a ROS environment.
        from geometry_msgs.msg import Point, Pose, Quaternion

        from .rl_insert_contract import matrix_to_quat

        tcp_pos, R_tcp = sc_tcp_pose_for_tip(self.policy, tip_pos, tip_rotation)
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
    def _event_status(self, depth_m: float):
        """Accept any fresh matching insertion event as seated.

        ``/scoring/insertion_event`` is a ~1 mm tip-proximity trigger, not a
        full-bore contact signal -- true since the 2026-07-09 assets
        (aic#593) and field-confirmed on 2026-07-27 19:31, when an SC run
        stalled at 1.39 mm depth and the event still fired 0.66 s later. This
        method used to defer any event below ``seat_candidate_depth_m``
        (15.2 mm, "97% of the bore"): that was correct reasoning for a
        contact-sensor trigger and is wrong for a proximity trigger, so it
        was rejecting successes the robot had already earned. Do not
        reinstate a depth gate here. ``depth_m`` is retained purely so the
        acceptance log below is the permanent record of depth-at-event;
        ``seat_candidate_depth_m`` still separately governs when ``_seat()``
        stops driving (see ``SCConfig.seat_candidate_depth_m``).
        """

        parent = self.policy._parent_node
        generation = int(getattr(parent, "_insertion_event_generation", 0))
        if generation <= self.event_generation:
            return None
        value = _normalize_event(getattr(parent, "_insertion_event_value", ""))
        matching = value == self.expected_event
        if not matching and SC_STRICT_PORT_EVENT:
            self.log.error(
                f"[sc] insertion event was for wrong port '{value}', expected "
                f"'{self.expected_event}'"
            )
            return HARD_FAILURE

        depth = float(depth_m)
        depth_text = f"{depth * 1000.0:.2f}" if np.isfinite(depth) else "nonfinite"
        if matching:
            self.log.info(
                f"[sc] SC_EVENT_ACCEPTED generation={generation} "
                f"event='{value}' depth_mm={depth_text}"
            )
            return SEATED
        self.log.error(
            f"[sc] SEATED IN A DIFFERENT PORT: event '{value}', task asked for "
            f"'{self.expected_event}'. Counting it as seated because port "
            f"selection is nearest-to-tip by design -- but scoring credits only "
            f"the requested port, so the macro parked over the wrong one. "
            f"RL_INSERT_SC_STRICT_PORT_EVENT=1 to treat this as a failure."
        )
        return SEATED

    def _wait_for_insertion_event(self, fixed_tip) -> str:
        deadline = time.monotonic() + self.config.insertion_event_timeout_wall_s
        hard_force_since = None
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            observation = self.get_observation()
            force = self._force_magnitude(observation)
            depth, _, _, tip_pos, _ = self._errors()
            status = self._event_status(depth)
            if status is not None:
                return status
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
                    stiffness=self._seating_stiffness(),
                    damping=self.DAMPING,
                )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.warn("[sc] seated geometry produced no matching insertion event")
        return STALLED

    def _stall_event_dwell(self, tip_pos, tip_rotation, depth) -> Optional[str]:
        """Hold position briefly after a stall in case a late event lands.

        Field evidence 2026-07-27 19:31: the scoring event fired 0.66 s AFTER
        ``_seat()`` had already returned STALLED for the same run.  Since
        ``/scoring/insertion_event`` is a ~1 mm tip-proximity trigger
        (aic#593), not full-bore contact, a push that just stalled can still
        be a beat away from a credited event.  This does not drive the arm
        any deeper -- it holds the tip at ``(tip_pos, tip_rotation)`` and
        gives ``stall_event_dwell_wall_s`` for the event to arrive before the
        caller concedes STALLED.
        """
        dwell_s = self.config.stall_event_dwell_wall_s
        if dwell_s <= 0.0:
            return None
        depth_text = f"{depth * 1000.0:.2f}" if np.isfinite(depth) else "nonfinite"
        self.log.info(
            f"[sc] SC_STALL_EVENT_DWELL_START depth_mm={depth_text} "
            f"dwell_s={dwell_s:.2f}"
        )
        deadline = time.monotonic() + dwell_s
        hard_force_since = None
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            self._hold_tip(tip_pos, tip_rotation)
            observation = self.get_observation()
            depth, _, _, _, _ = self._errors()
            status = self._event_status(depth)
            if status is not None:
                return status
            force = self._force_magnitude(observation)
            if np.isfinite(force) and force > self.config.force_abort_n:
                hard_force_since = hard_force_since or time.monotonic()
                if time.monotonic() - hard_force_since >= self.config.force_abort_wall_s:
                    self.log.error(
                        f"[sc] >{self.config.force_abort_n:.1f}N during stall-event dwell"
                    )
                    return HARD_FAILURE
            else:
                hard_force_since = None
            self.policy.sleep_for(self.config.command_dt_sim_s)
        depth_text = f"{depth * 1000.0:.2f}" if np.isfinite(depth) else "nonfinite"
        self.log.warn(
            f"[sc] SC_STALL_EVENT_DWELL_TIMEOUT depth_mm={depth_text} "
            f"dwell_s={dwell_s:.2f}"
        )
        return None

    # ----------------------------------------------------------------- run
    def _align(self):
        """Drive to the perceived opening.  Returns ``True``, ``False``, or ``ALIGN_TIMED_OUT``.

        ``True`` means converged within tolerance.  ``False`` means the
        caller must abort -- returned on timeout only when
        ``SC_STRICT_PERCEPTION`` is set.  ``ALIGN_TIMED_OUT`` (the default
        timeout outcome) tells ``run()`` to proceed to seating anyway; see the
        comment at the ``return ALIGN_TIMED_OUT`` site below for why that is
        safe.
        """
        # The cameras are wrist-mounted, so gather the complete temporal batch
        # before commanding motion.  Once selected, this target is immutable:
        # the field run showed per-frame target replacement oscillating by
        # 1.45 x 0.94 mm and making a 0.3 mm convergence gate impossible.
        timing = getattr(self, "timing", None)
        if timing is not None:
            timing.mark("visual_prealign_start")
        self._prime_visual_target()
        if timing is not None:
            timing.mark("alignment_start")
        start = time.monotonic()
        depth, lateral_xy, rotation_error, _, _ = self._errors()
        align_depth = depth
        last_depth = depth
        last_lateral = float(np.linalg.norm(lateral_xy))
        last_rotation = float(np.linalg.norm(rotation_error))
        command_count = 0
        while time.monotonic() - start < self.config.align_timeout_wall_s:
            self.policy._enforce_action_deadline(self.move_robot)
            depth, lateral_xy, rotation_error, tip_pos, _ = self._errors()
            lateral = float(np.linalg.norm(lateral_xy))
            rotation = float(np.linalg.norm(rotation_error))
            last_depth = depth
            last_lateral = lateral
            last_rotation = rotation
            if (
                lateral <= self.config.align_lateral_tol_m
                and rotation <= self.config.align_rotation_tol_rad
            ):
                self.log.info(
                    f"[sc] aligned: lateral={lateral*1000:.2f}mm "
                    f"rot={np.degrees(rotation):.2f}deg depth={depth*1000:.2f}mm"
                )
                if timing is not None:
                    timing.mark(
                        "alignment_converged",
                        lateral_mm=f"{lateral * 1000.0:.3f}",
                        rotation_deg=f"{np.degrees(rotation):.3f}",
                        depth_mm=f"{depth * 1000.0:.3f}",
                        commands=command_count,
                    )
                return True
            # Farther than the mouth zone the plug is in free space.  Use a
            # *persistent*, still-bounded segment there: replacing the
            # target from the current tip every loop capped force at 0.135 N
            # and made 4-6 mm handoff errors crawl into the alignment timeout.
            # At the mouth preserve the former compliant current-tip increments.
            precontact = depth < -self.config.seat_mouth_zone_m
            segment_xy = getattr(self, "_align_segment_lateral_xy", None)
            if precontact:
                if (
                    segment_xy is None
                    or float(np.linalg.norm(lateral_xy - segment_xy))
                    <= self.config.align_lateral_tol_m
                ):
                    lateral_step = -lateral_xy
                    lateral_norm = float(np.linalg.norm(lateral_step))
                    if lateral_norm > self.config.align_max_lateral_step_m:
                        lateral_step *= (
                            self.config.align_max_lateral_step_m / lateral_norm
                        )
                    segment_xy = np.asarray(
                        lateral_xy + lateral_step, dtype=np.float64
                    )
                    self._align_segment_lateral_xy = segment_xy
                    self.log.info(
                        f"[sc] SC_ALIGN_SEGMENT mode=precontact "
                        f"goal_lateral_mm="
                        f"{np.round(segment_xy * 1000.0, 3).tolist()} "
                        f"remaining_mm={lateral * 1000.0:.3f} "
                        f"stiffness_axial_N_m="
                        f"{self.PRECONTACT_ALIGN_STIFFNESS[2]:.1f}"
                    )
                target_tip = (
                    self.port_pos
                    + self.Rp[:, 0] * segment_xy[0]
                    + self.Rp[:, 1] * segment_xy[1]
                    + self.Rp[:, 2]
                    * float(np.clip(align_depth, depth - 0.001, depth + 0.001))
                )
                stiffness = self.PRECONTACT_ALIGN_STIFFNESS
                damping = self.PRECONTACT_ALIGN_DAMPING
            else:
                self._align_segment_lateral_xy = None
                lateral_step = -lateral_xy
                lateral_norm = float(np.linalg.norm(lateral_step))
                if lateral_norm > self.config.align_mouth_max_lateral_step_m:
                    lateral_step *= (
                        self.config.align_mouth_max_lateral_step_m / lateral_norm
                    )
                target_tip = (
                    tip_pos
                    + self.Rp[:, 0] * lateral_step[0]
                    + self.Rp[:, 1] * lateral_step[1]
                    + self.Rp[:, 2]
                    * float(np.clip(align_depth - depth, -0.001, 0.001))
                )
                stiffness = self.STIFFNESS
                damping = self.DAMPING
            if rotation > self.config.align_max_rotation_step_rad:
                remaining = 1.0 - self.config.align_max_rotation_step_rad / rotation
                target_rotation = self.Rs @ rotation_from_axis_angle(remaining * rotation_error)
            else:
                target_rotation = self.Rs
            self.policy.set_pose_target(
                self.move_robot,
                self._tcp_target(target_tip, target_rotation),
                stiffness=stiffness,
                damping=damping,
            )
            command_count += 1
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.error(
            f"[sc] alignment did not converge in {self.config.align_timeout_wall_s:.1f}s "
            f"SC_ALIGN_TIMEOUT final_depth_mm={last_depth * 1000.0:.3f} "
            f"final_lateral_mm={last_lateral * 1000.0:.3f} "
            f"final_rotation_deg={np.degrees(last_rotation):.3f} "
            f"commands={command_count} action_budget_remaining_s="
            f"{_sc_action_budget_remaining_s(self.policy):.3f}"
        )
        if timing is not None:
            timing.mark(
                "alignment_timeout",
                lateral_mm=f"{last_lateral * 1000.0:.3f}",
                rotation_deg=f"{np.degrees(last_rotation):.3f}",
                depth_mm=f"{last_depth * 1000.0:.3f}",
                commands=command_count,
            )
        if SC_STRICT_PERCEPTION:
            return False
        # Do not abort here: aborting before an attempt scores proximity only
        # (~25 max), while attempting -- even from this unconverged pose --
        # can still land a partial insertion (38-50), so throwing the pose
        # away is strictly worse (docs/scoring.md:106-114).  This is safe
        # because _seat() is the backstop for a genuinely bad pose: it trips
        # lateral_safety_m (3mm) / rotation_safety_rad and returns STALLED
        # rather than driving a wedge deeper.  ALIGN_TIMED_OUT keeps this
        # outcome distinguishable from a converged alignment for both the
        # caller and the logs.
        self.log.warn(
            "[sc] SC_ALIGN_TIMEOUT_PROCEEDING seating despite the unconverged "
            "alignment above; RL_INSERT_SC_STRICT_PERCEPTION=1 restores the "
            "old abort-on-timeout behaviour"
        )
        return ALIGN_TIMED_OUT

    def _seat(self) -> str:
        depth, _, _, _, _ = self._errors()
        command_depth = depth
        seat_stiffness = self._seating_stiffness()
        seat_axial_stiffness = self._seating_axial_stiffness_n_m()
        self.log.info(
            f"[sc] SC_SEAT_IMPEDANCE lateral_N_m="
            f"{self.SEAT_LATERAL_STIFFNESS_N_M:.1f} "
            f"axial_N_m={seat_axial_stiffness:.1f} "
            f"force_lead_mm={self.config.force_lead_m * 1000.0:.3f} "
            f"nominal_force_N="
            f"{seat_axial_stiffness * self.config.force_lead_m:.3f} "
            f"recovery_lead_mm="
            f"{self.config.visual_recovery_force_n / seat_axial_stiffness * 1000.0:.3f} "
            f"recovery_nominal_force_N={self.config.visual_recovery_force_n:.3f}"
        )
        now = time.monotonic()
        last_command_time = now
        progress = WallProgressWatch.start(depth, now, self.config)
        hard_force_since = None
        acc_lat = np.zeros(2, dtype=np.float64)
        acc_tilt = np.zeros(2, dtype=np.float64)
        last_wrench_log_time = 0.0
        stall_grace_deadline = None
        recovery_offset_xy = np.zeros(2, dtype=np.float64)
        recovery_path_m = 0.0
        recovery_active = False
        recovery_start_depth = float("nan")
        recovery_next_step_wall_s = float("inf")
        recovery_low_force_since = None
        recovery_frame_wait_since = None
        recovery_steps_this_episode = 0

        while True:
            self.policy._enforce_action_deadline(self.move_robot)
            observation = self.get_observation()
            depth, lateral_xy, rotation_error, tip_pos, tip_rotation = self._errors()
            lateral = float(np.linalg.norm(lateral_xy))
            rotation = float(np.linalg.norm(rotation_error))
            force = self._force_magnitude(observation)
            now = time.monotonic()
            status = self._event_status(depth)
            if status is not None:
                return status

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
                return STALLED

            if depth >= self.config.seat_candidate_depth_m:
                fixed_tip = (
                    self.port_pos
                    + self.Rp[:, 0] * recovery_offset_xy[0]
                    + self.Rp[:, 1] * recovery_offset_xy[1]
                    + self.Rp[:, 2] * SC_INSERT_DEPTH_M
                )
                return self._wait_for_insertion_event(fixed_tip)

            if recovery_active:
                depth_gain = depth - recovery_start_depth
                if depth_gain >= self.config.visual_recovery_meaningful_depth_m:
                    self.log.info(
                        f"[sc] SC_VISUAL_RECOVERY_EXIT reason=depth_advanced "
                        f"gain_mm={depth_gain * 1000.0:.2f} "
                        f"offset_mm="
                        f"{np.round(recovery_offset_xy * 1000.0, 3).tolist()} "
                        f"path_mm={recovery_path_m * 1000.0:.2f}"
                    )
                    recovery_active = False
                    recovery_low_force_since = None
                    recovery_frame_wait_since = None
                    command_depth = max(command_depth, depth)
                    last_command_time = now
                    progress = WallProgressWatch.start(depth, now, self.config)
                    stall_grace_deadline = None
                else:
                    if (
                        not np.isfinite(force)
                        or force > self.config.visual_recovery_max_force_n
                    ):
                        self._hold_tip(tip_pos, tip_rotation)
                        self.log.warn(
                            f"[sc] SC_VISUAL_RECOVERY_EXIT reason=force_gate "
                            f"force_N={force:.2f} "
                            f"max_N={self.config.visual_recovery_max_force_n:.2f}"
                        )
                        return STALLED

                    if force < self.config.visual_recovery_min_force_n:
                        if recovery_low_force_since is None:
                            recovery_low_force_since = now
                        if (
                            now - recovery_low_force_since
                            >= self.config.visual_recovery_contact_timeout_wall_s
                        ):
                            self._hold_tip(tip_pos, tip_rotation)
                            self.log.warn(
                                f"[sc] SC_VISUAL_RECOVERY_EXIT reason=contact_lost "
                                f"force_N={force:.2f}"
                            )
                            return STALLED
                    else:
                        recovery_low_force_since = None
                        if now >= recovery_next_step_wall_s:
                            estimate = self._visual_recovery_estimate(observation)
                            if estimate is None:
                                if (
                                    getattr(
                                        self,
                                        "_visual_recovery_last_failure_reason",
                                        None,
                                    )
                                    == "stale_frames"
                                ):
                                    if recovery_frame_wait_since is None:
                                        recovery_frame_wait_since = now
                                        self.log.warn(
                                            "[sc] SC_VISUAL_RECOVERY_WAIT "
                                            "reason=stale_frames "
                                            "action=hold_without_lateral_step"
                                        )
                                    if (
                                        now - recovery_frame_wait_since
                                        >= self.config.visual_recovery_frame_timeout_wall_s
                                    ):
                                        self._hold_tip(tip_pos, tip_rotation)
                                        self.log.warn(
                                            "[sc] SC_VISUAL_RECOVERY_EXIT "
                                            "reason=fresh_frame_timeout"
                                        )
                                        return STALLED
                                else:
                                    self._hold_tip(tip_pos, tip_rotation)
                                    self.log.warn(
                                        "[sc] SC_VISUAL_RECOVERY_EXIT "
                                        "reason=weak_or_conflicting_vision"
                                    )
                                    return STALLED
                            elif estimate.balanced:
                                recovery_frame_wait_since = None
                                if recovery_steps_this_episode == 0:
                                    self._hold_tip(tip_pos, tip_rotation)
                                    self.log.warn(
                                        "[sc] SC_VISUAL_RECOVERY_EXIT "
                                        "reason=clearances_already_balanced"
                                    )
                                    return STALLED
                                self.log.info(
                                    "[sc] SC_VISUAL_RECOVERY_EXIT "
                                    "reason=clearances_balanced_after_step "
                                    "action=resume_seating"
                                )
                                recovery_active = False
                                recovery_low_force_since = None
                                recovery_frame_wait_since = None
                                command_depth = depth
                                last_command_time = now
                                progress = WallProgressWatch.start(
                                    depth, now, self.config
                                )
                                stall_grace_deadline = None
                            else:
                                recovery_frame_wait_since = None
                                update = bounded_recovery_offset_update(
                                    recovery_offset_xy,
                                    estimate.direction_xy,
                                    recovery_path_m,
                                    max_step_m=(
                                        self.config.visual_recovery_step_m
                                    ),
                                    max_total_m=(
                                        self.config.visual_recovery_max_total_m
                                    ),
                                )
                                if update is None:
                                    self._hold_tip(tip_pos, tip_rotation)
                                    self.log.warn(
                                        "[sc] SC_VISUAL_RECOVERY_EXIT "
                                        "reason=travel_cap_reached"
                                    )
                                    return STALLED
                                (
                                    recovery_offset_xy,
                                    applied_step_xy,
                                    recovery_path_m,
                                ) = update
                                recovery_steps_this_episode += 1
                                recovery_next_step_wall_s = (
                                    now
                                    + self.config.visual_recovery_settle_wall_s
                                )
                                self.log.info(
                                    f"[sc] SC_VISUAL_RECOVERY_STEP "
                                    f"step_mm="
                                    f"{np.round(applied_step_xy * 1000.0, 3).tolist()} "
                                    f"offset_mm="
                                    f"{np.round(recovery_offset_xy * 1000.0, 3).tolist()} "
                                    f"path_mm={recovery_path_m * 1000.0:.2f} "
                                    f"force_N={force:.2f}"
                                )

                    if recovery_active:
                        recovery_command_depth = min(
                            SC_INSERT_DEPTH_M
                            + self.config.seat_overtravel_m,
                            recovery_start_depth
                            + self.config.visual_recovery_force_n
                            / self.config.axial_stiffness_n_m,
                        )
                        target_tip = (
                            self.port_pos
                            + self.Rp[:, 0] * recovery_offset_xy[0]
                            + self.Rp[:, 1] * recovery_offset_xy[1]
                            + self.Rp[:, 2] * recovery_command_depth
                        )
                        self.policy.set_pose_target(
                            self.move_robot,
                            self._tcp_target(target_tip, self.Rs),
                            stiffness=seat_stiffness,
                            damping=self.DAMPING,
                        )
                        self.policy.sleep_for(self.config.command_dt_sim_s)
                        continue

            sample = None
            if self.config.seat_align_enable:
                acc_lat, acc_tilt, sample = self._alignment_sample(
                    observation, depth, force, acc_lat, acc_tilt
                )
                if now - last_wrench_log_time >= SC_WRENCH_LOG_PERIOD_S:
                    self._log_wrench(sample, acc_lat, acc_tilt)
                    last_wrench_log_time = now

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
                    recovery_contact = bool(
                        np.isfinite(force)
                        and self.config.visual_recovery_min_force_n
                        <= force
                        <= self.config.visual_recovery_max_force_n
                    )
                    recovery_depth = bool(
                        -self.config.stall_progress_m
                        <= depth
                        <= self.config.seat_mouth_zone_m
                    )
                    recovery_budget = bool(
                        recovery_path_m
                        < self.config.visual_recovery_max_total_m - 1e-12
                    )
                    activation_frame_stamps = _sc_camera_frame_stamps(
                        observation
                    )
                    activation_frame_timing = _sc_frame_timing_fields(
                        self.policy, activation_frame_stamps
                    )
                    recovery_timestamped = bool(
                        len(activation_frame_stamps)
                        >= self.config.visual_recovery_min_views
                    )
                    recovery_baselined = bool(
                        len(getattr(self, "_visual_recovery_baseline", {}))
                        >= self.config.visual_recovery_min_views
                    )
                    if (
                        self.config.visual_recovery_enable
                        and recovery_contact
                        and recovery_depth
                        and recovery_budget
                        and recovery_timestamped
                        and recovery_baselined
                    ):
                        last_frame_stamps = dict(
                            getattr(
                                self,
                                "_visual_recovery_last_frame_stamps",
                                {},
                            )
                        )
                        for camera, stamp_s in activation_frame_stamps.items():
                            last_frame_stamps[camera] = max(
                                stamp_s,
                                last_frame_stamps.get(
                                    camera, float("-inf")
                                ),
                            )
                        self._visual_recovery_last_frame_stamps = (
                            last_frame_stamps
                        )
                        recovery_active = True
                        recovery_start_depth = depth
                        recovery_next_step_wall_s = now
                        recovery_low_force_since = None
                        recovery_frame_wait_since = None
                        recovery_steps_this_episode = 0
                        acc_lat = np.zeros(2, dtype=np.float64)
                        acc_tilt = np.zeros(2, dtype=np.float64)
                        command_depth = depth
                        last_command_time = now
                        progress = WallProgressWatch.start(
                            depth, now, self.config
                        )
                        stall_grace_deadline = None
                        self.log.warn(
                            f"[sc] SC_VISUAL_RECOVERY_ACTIVATE "
                            f"depth_mm={depth * 1000.0:.2f} "
                            f"force_N={force:.2f} "
                            f"timestamped_views="
                            f"{len(activation_frame_stamps)} "
                            f"existing_offset_mm="
                            f"{np.round(recovery_offset_xy * 1000.0, 3).tolist()} "
                            f"{activation_frame_timing} "
                            "action_budget_remaining_s="
                            f"{_sc_action_budget_remaining_s(self.policy):.3f} "
                            "wrench_nudge_reset=true"
                        )
                        continue
                    self.log.warn(
                        f"[sc] SC_VISUAL_RECOVERY_SKIPPED "
                        f"enabled={self.config.visual_recovery_enable} "
                        f"light_contact={recovery_contact} "
                        f"mouth_depth={recovery_depth} "
                        f"budget={recovery_budget}"
                        f" timestamped_views={recovery_timestamped}"
                        f" baseline_views={recovery_baselined} "
                        f"{activation_frame_timing}"
                    )
                    dwell_status = self._stall_event_dwell(
                        tip_pos, tip_rotation, depth
                    )
                    if dwell_status is not None:
                        return dwell_status
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

            target_tip = (
                self.port_pos
                + self.Rp[:, 0] * recovery_offset_xy[0]
                + self.Rp[:, 1] * recovery_offset_xy[1]
                + self.Rp[:, 2] * command_depth
            )
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
                stiffness=seat_stiffness,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)

    def _retry_snapshot(self):
        observation = self.get_observation()
        depth, _lateral_xy, _rotation_error, tip_pos, tip_rotation = self._errors()
        force = self._force_magnitude(observation)
        return depth, force, tip_pos, tip_rotation

    def _seat_attempt_budget_required_s(
        self, *, full_retry: bool, retry_prep: bool = False
    ) -> float:
        required = (
            self.config.stall_timeout_wall_s
            + self.config.seat_stall_grace_s
            + self.config.stall_event_dwell_wall_s
            + self.config.command_dt_sim_s
        )
        if retry_prep:
            required += self.RETRY_UNLOAD_HOLD_WALL_S
        if full_retry:
            required += (
                min(self.config.align_timeout_wall_s, self.RETRY_ALIGN_BUDGET_S)
                + self.RETRY_REPERCEIVE_BUDGET_S
                + 2.0
            )
        return required

    def _seat_attempt_has_budget(
        self, *, full_retry: bool, retry_prep: bool = False
    ) -> bool:
        remaining = _sc_action_budget_remaining_s(self.policy)
        if not np.isfinite(remaining):
            return True
        return remaining >= self._seat_attempt_budget_required_s(
            full_retry=full_retry,
            retry_prep=retry_prep,
        )

    def _retry_command_depth(self, target_depth: float, *, hold_wall_s: float) -> None:
        depth, _force, tip_pos, _tip_rotation = self._retry_snapshot()
        target_depth = min(float(target_depth), depth)
        target_tip = tip_pos + self.Rp[:, 2] * (target_depth - depth)
        self.log.info(
            f"[sc] SC_RETRY_UNLOAD from_mm={depth * 1000.0:.2f} "
            f"to_mm={target_depth * 1000.0:.2f}"
        )
        deadline = time.monotonic() + max(0.0, float(hold_wall_s))
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            self.policy.set_pose_target(
                self.move_robot,
                self._tcp_target(target_tip, self.Rs),
                stiffness=self._seating_stiffness(),
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)

    def _retry_reprime_grasp(self) -> bool:
        before = sc_grasp_transform(self.policy)
        before_tip = None
        if before is not None:
            before_tip = np.asarray(before[0], dtype=np.float64).reshape(3)
        if not prime_sc_plug_pose(self.policy, self.get_observation, self.move_robot):
            self.log.error("[sc] SC_REPRIME_FAILED")
            return False
        after = sc_grasp_transform(self.policy)
        if before_tip is not None and after is not None:
            after_tip = np.asarray(after[0], dtype=np.float64).reshape(3)
            delta_mm = (after_tip - before_tip) * 1000.0
            self.log.info(
                f"[sc] SC_REPRIME_DELTA dx_mm={delta_mm[0]:.3f} "
                f"dy_mm={delta_mm[1]:.3f} dz_mm={delta_mm[2]:.3f}"
            )
        else:
            self.log.info("[sc] SC_REPRIME_DELTA dx_mm=nan dy_mm=nan dz_mm=nan")
        return True

    def _retry_reperceive_port(self) -> bool:
        """Refresh the cached port pose.  Never fatal -- falls back to cached.

        Runs *after* the grasp re-prime, because the candidate selection inside
        ``perceive_sc_port_pose_consensus`` is nearest-to-tip and the tip comes
        from the grasp transform: re-priming first makes the port choice better,
        not just fresher.

        Every failure path keeps the cached pose and returns False rather than
        aborting.  The cached pose already produced a converged alignment, so
        falling back to it is strictly better than conceding the run --
        docs/scoring.md:106-114, the same argument the perception gates use.
        """
        try:
            perceived = perceive_sc_port_pose_consensus(
                self.policy, self.task, self.get_observation, self.move_robot
            )
        except Exception as exc:  # noqa: BLE001 - never fail a retry on vision
            self.log.warn(
                f"[sc] SC_REPERCEIVE_FAILED reason={exc!r} action=keep_cached"
            )
            return False
        if perceived is None:
            self.log.warn(
                "[sc] SC_REPERCEIVE_FAILED reason=no_consensus action=keep_cached"
            )
            return False
        port_pos, port_quat, reproj_px = perceived
        port_pos = np.asarray(port_pos, dtype=np.float64).reshape(3)
        shift = port_pos - self.port_pos
        shift_norm = float(np.linalg.norm(shift))
        # A retry re-perceives with the plug already sitting in a mouth.  3 of
        # the 6 runs on 2026-07-28 selected the neighbouring port 40 mm away;
        # accepting that jump here would command the arm sideways with the plug
        # still engaged.  Port pose was repeatable to 30 um across those runs,
        # so a genuine refresh is sub-millimetre -- this gate only ever rejects
        # a port swap, never a real correction.
        if shift_norm > self.RETRY_REPERCEIVE_MAX_SHIFT_M:
            self.log.error(
                f"[sc] SC_REPERCEIVE_REJECTED shift_mm={shift_norm * 1000.0:.2f} "
                f"gate_mm={self.RETRY_REPERCEIVE_MAX_SHIFT_M * 1000.0:.1f} "
                "action=keep_cached -- a shift this large is a different port, "
                "not a refresh"
            )
            return False
        shift_mm = shift * 1000.0
        self.port_pos = port_pos
        self._visual_origin_port_pos = self.port_pos.copy()
        self.port_quat = np.asarray(port_quat, dtype=np.float64).reshape(4)
        self.Rp = port_frame(self.port_quat)
        _, R_tip = self._tip_pose()
        self.Rs = (
            seat_frame(self.Rp, R_tip) if SC_PRESERVE_HANDOFF_YAW else self.Rp.copy()
        )
        self.R_yaw = self.Rp.T @ self.Rs
        self.log.info(
            f"[sc] SC_REPERCEIVE_DELTA dx_mm={shift_mm[0]:.3f} "
            f"dy_mm={shift_mm[1]:.3f} dz_mm={shift_mm[2]:.3f} "
            f"shift_mm={shift_norm * 1000.0:.3f} reproj_px={reproj_px:.2f}"
        )
        return True

    def run(self) -> bool:
        if sc_grasp_transform(self.policy) is None:
            # Reachable only outside the field flow: run_sc_insertion aborts
            # before constructing this controller when priming fails.
            self.log.warn(
                "[sc] no measured grasp transform -- geometry is running on the "
                "fixed-grasp SC_TIP_IN_TCP default, which is only acceptable in "
                "camera-less tests; field runs must prime_sc_plug_pose first"
            )
        self.send_feedback("sc align to perceived opening")
        align_status = self._align()
        if align_status is False:
            return False
        if align_status is not True:
            # ALIGN_TIMED_OUT: _align() already logged SC_ALIGN_TIMEOUT and
            # SC_ALIGN_TIMEOUT_PROCEEDING.  Seat anyway -- see _align's
            # docstring for why this cannot score worse than the abort it
            # replaces.
            self.log.warn(
                f"[sc] seating despite an unconverged alignment (status={align_status})"
            )
        depth, _, _, _, _ = self._errors()
        standoff_depth = depth
        self.send_feedback("sc force-regulated seating")
        timing = getattr(self, "timing", None)
        attempt = 1
        while not self.MAX_SEAT_ATTEMPTS or attempt <= self.MAX_SEAT_ATTEMPTS:
            if not self._seat_attempt_has_budget(full_retry=False):
                depth, _force, _tip_pos, _tip_rotation = self._retry_snapshot()
                self.log.warn(
                    f"[sc] SC_RETRY_EXHAUSTED attempts={attempt - 1} "
                    f"final_depth_mm={depth * 1000.0:.2f}"
                )
                return False
            if timing is not None:
                timing.mark("seating_start", attempt=attempt)
            outcome = self._seat()
            if timing is not None:
                timing.mark("seating_end", attempt=attempt, outcome=outcome)
            if outcome == SEATED:
                self.log.info(
                    f"[sc] insertion event confirmed for {self.expected_event}"
                )
                return True
            if outcome != STALLED:
                self.log.warn(f"[sc] seating ended without confirmation: {outcome}")
                return False
            if self.MAX_SEAT_ATTEMPTS and attempt >= self.MAX_SEAT_ATTEMPTS:
                depth, _force, _tip_pos, _tip_rotation = self._retry_snapshot()
                self.log.warn(
                    f"[sc] SC_RETRY_EXHAUSTED attempts={attempt} "
                    f"final_depth_mm={depth * 1000.0:.2f}"
                )
                self.log.warn(f"[sc] seating ended without confirmation: {outcome}")
                return False

            next_attempt = attempt + 1
            # Alternate the two recovery rungs for as long as the budget lasts:
            # attempt 2 unloads in place, 3 is a full retry, 4 unloads again, 5
            # is another full retry, and so on.  Cheap-then-expensive is the
            # right order every cycle, not just the first: a full retry re-primes
            # the grasp and re-perceives the port, so the unload that follows it
            # is being tried against a *different* pose than the one before it.
            full_retry = next_attempt % 2 == 1
            if not self._seat_attempt_has_budget(
                full_retry=full_retry,
                retry_prep=True,
            ):
                depth, _force, _tip_pos, _tip_rotation = self._retry_snapshot()
                self.log.warn(
                    f"[sc] SC_RETRY_EXHAUSTED attempts={attempt} "
                    f"final_depth_mm={depth * 1000.0:.2f}"
                )
                return False

            depth, force, _tip_pos, _tip_rotation = self._retry_snapshot()
            budget = _sc_action_budget_remaining_s(self.policy)
            cap = self.MAX_SEAT_ATTEMPTS if self.MAX_SEAT_ATTEMPTS else "budget"
            self.log.warn(
                f"[sc] SC_RETRY_START attempt={next_attempt}/{cap} "
                f"rung={'full' if full_retry else 'unload'} "
                f"reason=stalled depth_mm={depth * 1000.0:.2f} "
                f"force_N={force:.2f} budget_s={budget:.3f}"
            )
            if not full_retry:
                self._retry_command_depth(
                    depth - self.RETRY_UNLOAD_M,
                    hold_wall_s=self.RETRY_UNLOAD_HOLD_WALL_S,
                )
            else:
                self._retry_command_depth(
                    standoff_depth,
                    hold_wall_s=self.RETRY_UNLOAD_HOLD_WALL_S,
                )
                if not self._retry_reprime_grasp():
                    return False
                # Grasp first, then port: candidate selection is nearest-to-tip,
                # so the fresh tip improves which port gets chosen.  Failure
                # here is non-fatal and keeps the cached pose.
                self._retry_reperceive_port()
                self.send_feedback("sc retry align to perceived opening")
                align_status = self._align()
                if align_status is False:
                    return False
                if align_status is not True:
                    self.log.warn(
                        "[sc] retry seating despite an unconverged alignment "
                        f"(status={align_status})"
                    )
                depth, _, _, _, _ = self._errors()
                standoff_depth = depth
            self.send_feedback("sc retry force-regulated seating")
            attempt = next_attempt


# --------------------------------------------------------------------------
# Perception: physical SC front-mouth pose (four corners plus centre).
# --------------------------------------------------------------------------
def _sc_detection_centroid(det):
    kps = np.asarray(det.get("kps"), dtype=np.float64)
    if kps.ndim != 2 or kps.shape[0] < SC_MOUTH_KEYPOINT_COUNT:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.asarray(kps[4, :2], dtype=np.float64)


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

    Deliberately the TCP and not the tip: the TCP comes straight from TF with
    no model in the loop, and it sits ~58 mm from the tip -- far inside a gate
    whose radius is hundreds of pixels, so the coarse proximity test gains
    nothing from the measured tip and stays valid even before priming.
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
    fit_X = candidate.get("fit_X")
    fit_shift_mm = (
        float(np.linalg.norm(np.asarray(fit_X) - candidate["X"]) * 1000.0)
        if fit_X is not None else float("nan")
    )
    fit_reproj = candidate.get("fit_reproj_px")
    fit_reproj_text = (
        f"{float(fit_reproj):.2f}" if fit_reproj is not None else "unavailable"
    )
    log.info(
        f"[sc] SC_PERCEPT_BEST score={candidate['score']:.2f} "
        f"centre_reproj="
        f"{candidate.get('centre_reproj_px', float('nan')):.2f}px "
        f"mean5_reproj={candidate['reproj_px']:.2f}px "
        f"fit_reproj={fit_reproj_text}px fit_shift={fit_shift_mm:.2f}mm "
        f"center_vs_corners="
        f"{candidate.get('center_disagreement_m', float('nan')) * 1000.0:.2f}mm "
        f"width={candidate['width']*1000:.2f}mm height={candidate['height']*1000:.2f}mm "
        f"fit_model={candidate.get('fit_width', candidate['width'])*1000:.2f}x"
        f"{candidate.get('fit_height', candidate['height'])*1000:.2f}mm "
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


def _rolled_kps(kps, roll: int) -> np.ndarray:
    """Cyclically relabel the four corners while preserving explicit KP4."""

    points = np.asarray(kps, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < SC_MOUTH_KEYPOINT_COUNT:
        raise ValueError("SC mouth pose requires four corners plus centre")
    reordered = points[:SC_MOUTH_KEYPOINT_COUNT].copy()
    reordered[:4] = np.roll(reordered[:4], -roll, axis=0)
    return reordered


def _triangulate_mouth_keypoints(policy, picks, rolls) -> np.ndarray:
    Ps = [pick["P"] for pick in picks]
    rolled = [_rolled_kps(pick["kps"], r) for pick, r in zip(picks, rolls)]
    return np.array(
        [
            policy._pc.triangulate([tuple(k[i]) for k in rolled], Ps)
            for i in range(SC_MOUTH_KEYPOINT_COUNT)
        ],
        dtype=np.float64,
    )


def _mean_reproj_px(policy, picks, rolls, kp_3d):
    errors = []
    for pick, roll in zip(picks, rolls):
        kps = _rolled_kps(pick["kps"], roll)
        for i in range(SC_MOUTH_KEYPOINT_COUNT):
            err = policy._reproject_error_px(kp_3d[i], pick["K"], pick["T"], kps[i])
            if err is not None:
                errors.append(err)
    return float(np.mean(errors)) if errors else None


def _centre_reproj_px(policy, picks, kp_3d):
    """Mean reprojection error of KP4 -- the explicit mouth centre -- alone.

    Deliberately no ``rolls`` argument.  ``_rolled_kps`` only permutes indices
    0-3, so ``pick["kps"][4]`` and therefore ``kp_3d[4]`` are IDENTICAL under
    every one of the 4^(n-1) roll assignments ``_best_keypoint_correspondence``
    searches: KP4 is roll-invariant.  That is precisely why this is a
    trustworthy ranking key and ``_mean_reproj_px`` is not -- the four corners
    are near-symmetric and ambiguous between viewpoints (that ambiguity is the
    whole reason the roll search exists), so their reprojection error is partly
    a measure of correspondence-search noise, not of centre quality.  KP4 is
    also the sole source of the commanded position (``sc_multiview_candidates``
    sets ``X = kp_3d[4]``), so this is a direct measurement of the number that
    actually drives the plug.
    """
    errors = []
    for pick in picks:
        kps = np.asarray(pick["kps"], dtype=np.float64)
        err = policy._reproject_error_px(kp_3d[4], pick["K"], pick["T"], kps[4])
        if err is not None:
            errors.append(err)
    return float(np.mean(errors)) if errors else None


def _best_keypoint_correspondence(policy, picks):
    """Resolve which camera's corner 0 is which physical corner.

    ``detect_sc_pose`` returns four corners plus an explicit centre.  A cyclic
    corner ambiguity can still occur between viewpoints, so the corner labels
    are resolved by reprojection while KP4 remains fixed.

    2026-07-25 18:46 is the worked example: the left camera saw the same physical
    port as the centre camera (centroids within 1.1 px) but labelled it rotated
    by two.  Matching as-is put the corners 18.6 px apart; rotating by two put
    them 2.4 px apart.  The result reprojected at 11.5 px against a 5.0 px gate,
    with residuals uniformly high on ALL FOUR corners -- the signature of a
    labelling mismatch rather than one bad corner -- and measured the opening
    3.35 x 7.14 mm instead of 7.09 x 4.06, i.e. transposed.

    So try each cyclic relabelling and keep whichever actually reprojects.  The
    first camera is held fixed as the reference (a roll applied to every camera
    at once is a pure relabelling: same 3D points, same reprojection error), so
    the search is 4^(n-1) -- 16 assignments for three cameras.

    Returns ``(kp_3d, rolls, mean_reproj_px)`` or None.
    """
    options = range(4) if SC_KEYPOINT_ROLL_SEARCH else (0,)
    best = None
    for rolls in itertools.product((0,), *([options] * (len(picks) - 1))):
        try:
            kp_3d = _triangulate_mouth_keypoints(policy, picks, rolls)
        except Exception:
            continue
        mean_px = _mean_reproj_px(policy, picks, rolls, kp_3d)
        if mean_px is None:
            continue
        if best is None or mean_px < best[2]:
            best = (kp_3d, rolls, mean_px)
    if best is None:
        return None

    # A roll applied to EVERY camera leaves the triangulated points and the
    # reprojection untouched and only renames the corners, so it is free to
    # normalise the convention: the mouth is wider than it is tall, so `width`
    # must come out as the long axis.  Without this, a reference camera
    # that happens to start a quarter-turn round transposes width and height,
    # which the geometry residual and candidate score both read.
    kp_3d, rolls, mean_px = best
    width = float(np.linalg.norm(((kp_3d[0] + kp_3d[3]) * 0.5) - ((kp_3d[1] + kp_3d[2]) * 0.5)))
    height = float(np.linalg.norm(((kp_3d[0] + kp_3d[1]) * 0.5) - ((kp_3d[2] + kp_3d[3]) * 0.5)))
    if height > width:
        rolls = tuple((r + 1) % 4 for r in rolls)
        kp_3d = kp_3d.copy()
        kp_3d[:4] = np.roll(kp_3d[:4], -1, axis=0)
    return kp_3d, rolls, mean_px


def _rigid_fit_3d(local_pts, world_pts):
    """Rigid initialization mapping a known rectangle onto raw triangulation."""

    local = np.asarray(local_pts, dtype=np.float64).reshape(-1, 3)
    world = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    if local.shape != world.shape or len(local) < 3:
        raise ValueError("SC rigid initialization needs at least three point pairs")
    local_center = local.mean(axis=0)
    world_center = world.mean(axis=0)
    covariance = (local - local_center).T @ (world - world_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    translation = world_center - rotation @ local_center
    return rotation, translation


def _joint_sc_rectangle_fit(picks, local_kps, raw_kp_3d):
    """Fit one rigid known-size rectangle to every camera/corner observation.

    Independent DLT gives the correspondence search a cheap, reliable
    initializer, but its four unconstrained 3D corners can shrink and shear in
    a way that moves their mean.  This bundle adjustment instead has one shared
    six-DoF rectangle pose and minimizes all camera/corner pixel residuals at
    once.  A soft-L1 loss limits the influence of one weakly localized corner.

    Returns ``(fitted_corners, center, mean_reprojection_px)`` or ``None``.
    """

    if least_squares is None:
        return None
    local = np.asarray(local_kps, dtype=np.float64).reshape(4, 3)
    raw = np.asarray(raw_kp_3d, dtype=np.float64).reshape(4, 3)
    try:
        initial_rotation, initial_center = _rigid_fit_3d(local, raw)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    def fitted_points(params):
        delta_rotation = rotation_from_axis_angle(params[:3])
        rotation = delta_rotation @ initial_rotation
        center = initial_center + params[3:]
        return (rotation @ local.T).T + center, center

    def residuals(params):
        points, _ = fitted_points(params)
        values = []
        for pick in picks:
            P = np.asarray(pick["P"], dtype=np.float64).reshape(3, 4)
            observed = np.asarray(pick["kps"], dtype=np.float64)[:4, :2]
            for point, uv in zip(points, observed):
                projected = P @ np.array([point[0], point[1], point[2], 1.0])
                if not np.all(np.isfinite(projected)) or projected[2] <= 1e-8:
                    values.extend((1e4, 1e4))
                else:
                    values.extend(
                        (projected[:2] / projected[2] - uv).tolist()
                    )
        return np.asarray(values, dtype=np.float64)

    # Keep optimization local to the already-gated DLT solution.  The bounds
    # prevent a planar mirror/scale ambiguity from jumping to another port.
    rotation_bound = np.deg2rad(20.0)
    translation_bound = 0.010
    try:
        result = least_squares(
            residuals,
            np.zeros(6, dtype=np.float64),
            bounds=(
                np.array([-rotation_bound] * 3 + [-translation_bound] * 3),
                np.array([+rotation_bound] * 3 + [+translation_bound] * 3),
            ),
            loss="soft_l1",
            f_scale=2.0,
            x_scale=np.array([0.05, 0.05, 0.05, 0.002, 0.002, 0.004]),
            max_nfev=100,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if not result.success or not np.all(np.isfinite(result.x)):
        return None

    fitted, center = fitted_points(result.x)
    residual = residuals(result.x).reshape(-1, 2)
    reproj = float(np.mean(np.linalg.norm(residual, axis=1)))
    if (
        not np.all(np.isfinite(fitted))
        or not np.all(np.isfinite(center))
        or not np.isfinite(reproj)
    ):
        return None
    return fitted, center, reproj


def sc_multiview_candidates(policy, per_cam):
    """Resolve associations and triangulate the physical mouth centre.

    KP4 is the model's explicit mouth centre and is the motion target.  The
    corners determine the plane orientation and provide a known-size consistency
    diagnostic.  The rigid fit remains diagnostic-only because held-out
    single-view PnP is not accurate enough to override the multiview centre.
    """
    log = policy.get_logger()
    per_cam = _select_sc_detections_for_triangulation(policy, per_cam, log=log)
    cams = [cam for cam, dets in per_cam.items() if dets]
    if len(cams) < 2:
        return []

    raw_candidates = []
    rejects = []
    rolled_count = 0
    for picks in itertools.product(*[per_cam[cam] for cam in cams]):
        try:
            resolved = _best_keypoint_correspondence(policy, picks)
        except Exception as exc:
            rejects.append(("triangulate_error", f"{type(exc).__name__}"))
            continue
        if resolved is None:
            rejects.append(("triangulate_error", "no correspondence reprojected"))
            continue
        kp_3d, rolls, raw_reproj = resolved
        if any(rolls):
            rolled_count += 1
        # Everything downstream -- reprojection, diagnostics, the kps recorded on
        # the candidate -- must see the SAME relabelling that was triangulated.
        picks = tuple(
            dict(pick, kps=_rolled_kps(pick["kps"], roll))
            for pick, roll in zip(picks, rolls)
        )
        raw_X = np.asarray(kp_3d[4], dtype=np.float64)
        # KP4 is roll-invariant (see _centre_reproj_px), so this is safe to
        # compute once here regardless of which relabelling `rolls` picked.
        centre_reproj = _centre_reproj_px(policy, picks, kp_3d)
        centre_reproj_px = centre_reproj if centre_reproj is not None else float("inf")
        corner_X = np.mean(kp_3d[:4], axis=0)
        center_disagreement = float(np.linalg.norm(raw_X - corner_X))
        if raw_X[2] < -0.05 or raw_X[2] > 0.25:
            rejects.append(("depth", f"z={raw_X[2] * 1000:.0f}mm"))
            continue

        width = float(np.linalg.norm(((kp_3d[0] + kp_3d[3]) * 0.5) - ((kp_3d[1] + kp_3d[2]) * 0.5)))
        height = float(np.linalg.norm(((kp_3d[0] + kp_3d[1]) * 0.5) - ((kp_3d[2] + kp_3d[3]) * 0.5)))
        if not (SC_MIN_OPENING_M <= width <= SC_MAX_OPENING_M) or not (
            SC_MIN_OPENING_M <= height <= SC_MAX_OPENING_M
        ):
            rejects.append(("size", f"{width * 1000:.1f}x{height * 1000:.1f}mm"))
            continue

        residual = abs(width - SC_POSE_WIDTH_M) + abs(height - SC_POSE_HEIGHT_M)
        raw_score = raw_reproj + residual * 250.0 - 0.02 * float(
            np.mean([pick.get("conf", 0.0) for pick in picks])
        )
        raw_candidates.append({
            "picks": picks, "kp_3d": kp_3d, "raw_X": raw_X,
            "corner_X": corner_X,
            "center_disagreement_m": center_disagreement,
            "width": width, "height": height,
            "opening": SC_POSE_CONVENTION,
            "opening_residual_m": residual,
            "raw_reproj_px": raw_reproj, "raw_score": raw_score,
            "centre_reproj_px": centre_reproj_px,
        })

    # Bundle adjustment is several orders of magnitude dearer than DLT. Run it
    # only on a bounded shortlist for diagnostics, while every valid raw
    # candidate remains eligible for selection.
    candidates = []
    raw_candidates.sort(key=lambda candidate: candidate["raw_score"])
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        picks = raw_candidate["picks"]
        kp_3d = raw_candidate["kp_3d"]
        raw_X = raw_candidate["raw_X"]
        width = raw_candidate["width"]
        height = raw_candidate["height"]
        residual = raw_candidate["opening_residual_m"]
        expected_width = SC_POSE_WIDTH_M
        expected_height = SC_POSE_HEIGHT_M
        q_wxyz, yaw = policy._estimate_sfp_port_orientation(kp_3d[:4])
        if q_wxyz is None:
            rejects.append(("degenerate_axis", "raw in-plane axis invalid"))
            continue

        fit_kp_3d = None
        fit_X = None
        fit_reproj = None
        if candidate_index < SC_JOINT_FIT_TOP_K:
            fitted = _joint_sc_rectangle_fit(
                picks, SC_POSE_LOCAL_KPS_M[:4], kp_3d[:4]
            )
            if fitted is not None:
                fit_kp_3d, fit_X, fit_reproj = fitted

        errors = []
        camera_diagnostics = []
        for cam, pick in zip(cams, picks):
            reproj_px = []
            for i in range(SC_MOUTH_KEYPOINT_COUNT):
                err = policy._reproject_error_px(
                    kp_3d[i], pick["K"], pick["T"], pick["kps"][i]
                )
                if err is not None:
                    errors.append(err)
                    reproj_px.append(float(err))
                else:
                    reproj_px.append(None)
            camera_diagnostics.append({
                "cam": cam,
                "conf": float(pick.get("conf", 0.0)),
                "centroid": _sc_detection_centroid(pick),
                "kps": np.asarray(
                    pick["kps"][:SC_MOUTH_KEYPOINT_COUNT], dtype=np.float64
                ),
                "reproj_px": reproj_px,
            })
        if not errors:
            rejects.append(("no_reproj", "every keypoint failed to reproject"))
            continue
        candidates.append({
            "X": raw_X, "kp_3d": kp_3d, "raw_kp_3d": kp_3d,
            "raw_X": raw_X, "q_wxyz": q_wxyz, "yaw": yaw,
            "corner_X": raw_candidate["corner_X"],
            "center_disagreement_m": raw_candidate["center_disagreement_m"],
            "score": float(raw_candidate["raw_score"]),
            "reproj_px": raw_candidate["raw_reproj_px"],
            "raw_reproj_px": raw_candidate["raw_reproj_px"],
            "centre_reproj_px": raw_candidate["centre_reproj_px"],
            "fit_kp_3d": fit_kp_3d,
            "fit_X": fit_X,
            "fit_reproj_px": fit_reproj,
            "width": width, "height": height,
            "fit_width": expected_width, "fit_height": expected_height,
            "opening": SC_POSE_CONVENTION,
            "opening_residual_m": residual,
            "camera_diagnostics": camera_diagnostics,
        })

    # Log whenever anything was discarded, not only on total failure: a run that
    # produces one candidate out of eight is also worth knowing about.
    if rejects:
        _log_sc_rejections(log, rejects)
    if rolled_count:
        # Not an error -- this is the fix working -- but it says the detector's
        # corner order is unstable between viewpoints, which is what would have
        # to be retrained away.  Track it: if it climbs towards every
        # combination, the roll search is carrying the whole pipeline.
        log.info(
            f"[sc] SC_KEYPOINT_ROLL {rolled_count} combination(s) needed a corner "
            "relabelling to reproject; index-to-index matching would have "
            "rejected them"
        )

    candidates.sort(key=lambda c: c["score"])
    return candidates


def _sc_candidate_lateral_distance(candidate, tip_pos):
    """Return a tip-to-mouth distance on the candidate port's lateral plane."""
    Rp = port_frame(candidate["q_wxyz"])
    delta_port = Rp.T @ (np.asarray(tip_pos, dtype=np.float64) - candidate["X"])
    return float(np.linalg.norm(delta_port[:2]))


def perceive_sc_port_pose(policy, task, obs):
    """One-frame SC opening pose: ``(pos, quat_wxyz, centre_reproj_px)`` or ``None``.

    Ranks and selects by ``centre_reproj_px`` -- KP4 alone, roll-invariant --
    not the 5-keypoint ``reproj_px`` mean the ambiguous corners dominate (see
    ``_centre_reproj_px`` and ``SC_STRICT_PERCEPTION`` above).  When nothing
    clears ``SC_MAX_SELECT_REPROJ_PX`` this DEGRADES to ranking every
    candidate instead of rejecting the frame, because refusing to attempt
    scores worse than attempting with a mediocre estimate
    (docs/scoring.md:106-114).  ``RL_INSERT_SC_STRICT_PERCEPTION=1`` restores
    the old reject-on-gate-miss behaviour.  The fatal
    ``SC_MAX_HANDOFF_LATERAL_M`` wrong-port gate below is unaffected either
    way -- it still returns ``None`` when nothing is within reach of the tip.
    """
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
            if (
                kps.ndim != 2
                or kps.shape[0] != SC_MOUTH_KEYPOINT_COUNT
                or kps.shape[1] < 2
                or not np.all(np.isfinite(kps[:, :2]))
            ):
                continue
            usable.append({
                "kps": kps[:, :2],
                "conf": det.get("conf", 0.0),
                "K": K,
                "T": T,
                "P": policy._pc.build_projection_matrix(K, T),
            })
        if dets and not usable:
            shapes = [np.asarray(det.get("kps")).shape for det in dets]
            log.error(
                f"[sc] {cam}: rejected SC pose detections with keypoint shapes "
                f"{shapes}; runtime requires exactly {SC_MOUTH_KEYPOINT_COUNT} "
                "keypoints from best_sc_mouth_pose.pt"
            )
        if usable:
            per_cam[cam] = usable

    candidates = sc_multiview_candidates(policy, per_cam)
    if not candidates:
        log.warn("[sc] multiview matching found no SC opening candidates")
        return None

    # Diagnose the same candidate the select gate reports on. candidates[0] is
    # best by *score* (reproj + shape residual - confidence bonus), which is not
    # necessarily best by centre reproj, and logging two different candidates
    # under the word "best" is how a log costs you a field run to interpret.
    def _centre_key(c):
        return c.get("centre_reproj_px", c["reproj_px"])

    best_by_centre = min(candidates, key=_centre_key)
    _log_sc_best_candidate(log, best_by_centre)

    clean = [c for c in candidates if _centre_key(c) <= SC_MAX_SELECT_REPROJ_PX]
    if not clean:
        best_centre_px = _centre_key(best_by_centre)
        if SC_STRICT_PERCEPTION:
            log.warn(
                f"[sc] no candidate under {SC_MAX_SELECT_REPROJ_PX:.1f}px centre "
                f"select gate (best centre {best_centre_px:.1f}px, "
                f"{len(candidates)} candidates) -- rejecting frame "
                "(RL_INSERT_SC_STRICT_PERCEPTION=1)"
            )
            return None
        # Degrade rather than reject: the corners' ambiguity says nothing
        # about KP4's own correctness (see docstring above), and refusing to
        # attempt scores strictly worse than attempting with a mediocre
        # estimate.  Every candidate becomes eligible for the lateral-distance
        # tie-break below; SC_MAX_HANDOFF_LATERAL_M remains fatal.
        log.warn(
            "[sc] SC_PERCEPT_DEGRADED reason=no_candidate_under_select_gate "
            f"best_centre_reproj_px={best_centre_px:.2f} "
            f"best_mean5_reproj_px={best_by_centre['reproj_px']:.2f} "
            f"candidates={len(candidates)} gate_px={SC_MAX_SELECT_REPROJ_PX:.1f} "
            "action=rank_all_candidates_by_centre_reproj"
        )
        clean = candidates

    try:
        tcp_pos, tcp_quat = policy._tcp()
        # Measured per-grasp tip once primed (the field flow primes before
        # perception); pre-prime the constant only degrades this RANKING gate,
        # it never fabricates a pose.
        tip_pos, _ = sc_tip_from_tcp(policy, tcp_pos, tcp_quat)
    except Exception:
        chosen = clean[0]
    else:
        in_range = [
            c for c in clean
            if _sc_candidate_lateral_distance(c, tip_pos) <= SC_MAX_HANDOFF_LATERAL_M
        ]
        if not in_range:
            nearest = min(clean, key=lambda c: _sc_candidate_lateral_distance(c, tip_pos))
            log.warn(
                f"[sc] all candidates beyond {SC_MAX_HANDOFF_LATERAL_M*1000:.0f}mm lateral "
                f"handoff gate (nearest {_sc_candidate_lateral_distance(nearest, tip_pos)*1000:.1f}mm)"
            )
            return None
        chosen = min(in_range, key=lambda c: _sc_candidate_lateral_distance(c, tip_pos))

    exp_w, exp_h = SC_POSE_WIDTH_M, SC_POSE_HEIGHT_M
    chosen_centre_px = _centre_key(chosen)
    log.info(
        f"[sc] SC_OPENING convention={chosen['opening']} "
        f"width={chosen['width']*1000:.2f}mm height={chosen['height']*1000:.2f}mm "
        f"expected={exp_w*1000:.2f}x{exp_h*1000:.2f}mm "
        f"residual={chosen['opening_residual_m']*1000:.2f}mm "
        f"center_vs_corners={chosen['center_disagreement_m']*1000:.2f}mm "
        f"centre_reproj={chosen_centre_px:.1f}px "
        f"mean5_reproj={chosen['reproj_px']:.1f}px"
    )
    if chosen["opening_residual_m"] > SC_OPENING_RESIDUAL_WARN_M:
        log.warn(
            f"[sc] triangulated opening {chosen['width']*1000:.2f}x"
            f"{chosen['height']*1000:.2f}mm is {chosen['opening_residual_m']*1000:.2f}mm "
            f"from the physical mouth contract "
            f"({exp_w*1000:.2f}x{exp_h*1000:.2f}mm). Check the "
            "SC_PERCEPT_BEST per-camera confidences before trusting the pose."
        )
    # centre_reproj_px -- KP4 alone -- is the returned/ranking number; the
    # 5-keypoint mean stays visible above (SC_OPENING/SC_PERCEPT_BEST) purely
    # for corner-health monitoring, never as a gate.
    return (
        np.asarray(chosen["X"], dtype=np.float64),
        np.asarray(chosen["q_wxyz"], dtype=np.float64),
        float(chosen_centre_px),
    )


def perceive_sc_port_pose_consensus(policy, task, get_observation, move_robot=None):
    """Median of an agreeing cluster of single-frame SC poses.

    The detector is expensive enough to consume a material part of the action
    budget. Check the policy deadline on both sides of each inference rather
    than discovering it only after all seven frames have been processed.

    Both failure modes below (too few usable frames, or too much disagreement
    between them) used to return ``None``.  They now DEGRADE to the single
    best-measured sample -- lowest ``centre_reproj_px`` -- instead, because
    refusing to attempt an insertion scores strictly worse than attempting
    with a mediocre estimate (docs/scoring.md:106-114; see the module
    docstring and ``SC_STRICT_PERCEPTION`` above).
    ``RL_INSERT_SC_STRICT_PERCEPTION=1`` restores the old ``None`` returns.
    There is one case with no fallback available at all: zero usable samples,
    where there is nothing to degrade to.

    ``perceive_sc_port_pose`` itself now degrades (see its docstring), so a
    frame it returns can legitimately carry a ``centre_reproj_px`` above
    ``SC_MAX_PORT_REPROJ_PX``.  Under ``SC_STRICT_PERCEPTION`` off, EVERY
    non-None, finite-reprojection frame is collected here; frames under the
    gate are preferred for the consensus computation below, and only when
    NONE of them clear the gate does this fall back to the full pool (logging
    ``SC_PERCEPT_DEGRADED reason=no_sample_under_port_reproj_gate``).
    Filtering every returned frame at collection time -- silently dropping
    the ones perceive_sc_port_pose already degraded to produce -- would
    re-introduce, one level up, exactly the reject-on-gate-miss veto the
    select-gate degrade above exists to remove.  Under ``SC_STRICT_PERCEPTION``
    the collection loop filters at the gate exactly as before.
    """

    log = policy.get_logger()
    start_wall_s = time.monotonic()
    start_sim_s = _sc_sim_time_seconds(policy)

    def log_timing(outcome: str, *, usable_samples: int) -> None:
        now_wall_s = time.monotonic()
        now_sim_s = _sc_sim_time_seconds(policy)
        sim_elapsed_s = (
            now_sim_s - start_sim_s
            if np.isfinite(now_sim_s) and np.isfinite(start_sim_s)
            else float("nan")
        )
        log.info(
            f"[sc] SC_PERCEPTION_TIMING outcome={outcome} "
            f"samples={usable_samples}/{SC_PERCEPT_SAMPLES} "
            f"wall_s={now_wall_s - start_wall_s:.3f} "
            f"sim_s={sim_elapsed_s:.3f} "
            f"action_budget_remaining_s="
            f"{_sc_action_budget_remaining_s(policy, now_wall_s=now_wall_s):.3f}"
        )

    def best_sample(pool):
        """Single lowest-centre-reprojection sample -- the degrade fallback."""
        return min(pool, key=lambda s: s[2])

    # Under SC_STRICT_PERCEPTION, filter at the gate right here at collection
    # time -- identical to the pre-degrade behaviour.  Otherwise collect every
    # non-None, finite-reprojection frame; the gate becomes a preference
    # applied below, not a veto applied here.
    all_samples = []
    for _ in range(SC_PERCEPT_SAMPLES):
        _sc_enforce_action_deadline(policy, move_robot)
        obs = get_observation()
        if obs is not None:
            res = perceive_sc_port_pose(policy, task, obs)
            if res is not None:
                X, q, reproj = res
                if np.isfinite(reproj) and (
                    not SC_STRICT_PERCEPTION or reproj <= SC_MAX_PORT_REPROJ_PX
                ):
                    all_samples.append(
                        (np.asarray(X, float), np.asarray(q, float), float(reproj))
                    )
        _sc_enforce_action_deadline(policy, move_robot)
        policy.sleep_for(SC_PERCEPT_SAMPLE_DT)
        _sc_enforce_action_deadline(policy, move_robot)

    if not all_samples:
        # Nothing to fall back to -- this is the one case SC_STRICT_PERCEPTION
        # cannot change.  Under SC_STRICT_PERCEPTION this means "zero frames
        # cleared the reproj gate" (as before); otherwise it means perception
        # produced nothing at all across every frame.
        log.error(
            f"[sc] perception consensus failed: 0/{SC_PERCEPT_SAMPLES} frames "
            "passed reproj -- no sample to fall back to"
        )
        log_timing("no_samples", usable_samples=0)
        return None

    if SC_STRICT_PERCEPTION:
        samples = all_samples
    else:
        under_gate = [s for s in all_samples if s[2] <= SC_MAX_PORT_REPROJ_PX]
        if under_gate:
            samples = under_gate
        else:
            worst_pool_best = best_sample(all_samples)
            log.warn(
                "[sc] SC_PERCEPT_DEGRADED reason=no_sample_under_port_reproj_gate "
                f"frames={len(all_samples)}/{SC_PERCEPT_SAMPLES} "
                f"best_centre_reproj_px={worst_pool_best[2]:.2f} "
                f"gate_px={SC_MAX_PORT_REPROJ_PX:.1f} action=use_full_pool"
            )
            samples = all_samples

    if len(samples) < SC_PERCEPT_MIN_AGREE:
        positions = np.array([s[0] for s in samples])
        spread_mm = float(
            np.max(np.linalg.norm(positions - np.median(positions, axis=0), axis=1))
        ) * 1000.0
        if SC_STRICT_PERCEPTION:
            log.error(
                f"[sc] perception consensus failed: only {len(samples)}/{SC_PERCEPT_SAMPLES} "
                f"frames passed reproj (need {SC_PERCEPT_MIN_AGREE})"
            )
            log_timing("insufficient_samples", usable_samples=len(samples))
            return None
        chosen = best_sample(samples)
        log.warn(
            "[sc] SC_PERCEPT_DEGRADED reason=insufficient_samples "
            f"samples={len(samples)}/{SC_PERCEPT_SAMPLES} "
            f"need={SC_PERCEPT_MIN_AGREE} spread_mm={spread_mm:.2f} "
            f"best_centre_reproj_px={chosen[2]:.2f} action=use_single_best_sample"
        )
        log_timing("degraded_insufficient_samples", usable_samples=len(samples))
        return chosen

    positions = np.array([s[0] for s in samples])
    med = np.median(positions, axis=0)
    keep = [s for s in samples
            if float(np.linalg.norm(s[0] - med)) <= SC_PERCEPT_AGREE_TOL_M]
    if len(keep) < SC_PERCEPT_MIN_AGREE:
        spread = float(np.max(np.linalg.norm(positions - med, axis=1))) * 1000
        if SC_STRICT_PERCEPTION:
            log.error(
                f"[sc] perception consensus failed: {len(keep)}/{len(samples)} agree within "
                f"{SC_PERCEPT_AGREE_TOL_M*1000:.1f}mm (spread={spread:.1f}mm)"
            )
            log_timing("disagreement", usable_samples=len(samples))
            return None
        chosen = best_sample(samples)
        log.warn(
            "[sc] SC_PERCEPT_DEGRADED reason=disagreement "
            f"agree={len(keep)}/{len(samples)} "
            f"tol_mm={SC_PERCEPT_AGREE_TOL_M*1000:.1f} spread_mm={spread:.2f} "
            f"best_centre_reproj_px={chosen[2]:.2f} action=use_single_best_sample"
        )
        log_timing("degraded_disagreement", usable_samples=len(samples))
        return chosen

    port_pos = np.median(np.array([s[0] for s in keep]), axis=0)
    reproj = float(np.median([s[2] for s in keep]))
    best = min(keep, key=lambda s: float(np.linalg.norm(s[0] - port_pos)))
    log.info(
        f"[sc] perception consensus: {len(keep)}/{len(samples)} agree, "
        f"port={np.round(port_pos, 5).tolist()} reproj={reproj:.2f}px"
    )
    log_timing("accepted", usable_samples=len(samples))
    return port_pos, best[1], reproj


# --------------------------------------------------------------------------
# SC plug-pose: measure the grasped tip instead of assuming it.
# --------------------------------------------------------------------------
def configure_sc_plug_pose(policy) -> bool:
    """Load the SC plug-pose model once per process; ``False`` if unavailable.

    Deliberately lazy (called from ``run_sc_insertion``) rather than hooked into
    lifecycle configure like ``configure_v50``: the mainline ``RLInsert`` has no
    such hook, and a missing SC checkpoint must fail the SC run closed without
    ever touching the SFP path.  Never raises for an absent model.
    """

    if getattr(policy, "_sc_plug_estimator", None) is not None:
        return True
    log = policy.get_logger()
    from .sc_plug_pose import ScPlugPoseEstimator, default_sc_plug_pose_weights

    weights = default_sc_plug_pose_weights()
    if weights is None:
        log.error(
            "[sc] SC plug-pose weights not found (set AIC_SC_PLUG_POSE_WEIGHTS); "
            "no fixed-grasp fallback is allowed -- refusing to insert"
        )
        return False
    try:
        estimator = ScPlugPoseEstimator(
            str(weights),
            # imgsz=960 is not optional: the model trains at 960 and the
            # cameras deliver 1152x1024; Ultralytics' 640 default is leak #1
            # in docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md.
            imgsz=_env_int("RL_INSERT_SC_PLUG_IMGSZ", 960),
            conf_threshold=_env_float("RL_INSERT_SC_PLUG_CONF", 0.25),
            # Crop-refine is what reaches the measured 0.27 mm median; without
            # it the estimator sits at ~0.46 mm against the 0.4 mm working
            # target (docs/SC_PLUG_POSE_RESULTS.md).  Per-instance, so the SFP
            # estimator's default-off behaviour is untouched.
            crop_refine=_env_bool("RL_INSERT_SC_PLUG_CROP_REFINE", True),
        )
        # Pay the YOLO first-inference cost now, not inside the run.  A
        # no-detection result on black pixels is expected; an exception is not.
        from .sfp_plug_pose import PlugPoseView

        estimator.detect_views([
            PlugPoseView(
                camera_name="sc_warmup",
                image_bgr=np.zeros((640, 640, 3), dtype=np.uint8),
                K=np.array(
                    [[500.0, 0.0, 320.0], [0.0, 500.0, 320.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                T_world_from_camera=np.eye(4),
                stamp_s=0.0,
                frame_id="sc-warmup",
            )
        ])
    except Exception as exc:
        log.error(
            f"[sc] SC plug-pose estimator failed to initialise: "
            f"{type(exc).__name__}: {exc}"
        )
        return False
    policy._sc_plug_estimator = estimator
    log.info(
        f"[sc] plug-pose estimator ready: weights={weights} "
        f"imgsz={estimator.imgsz} crop_refine={estimator.crop_refine} "
        f"crop_pad_scale={estimator.crop_pad_scale}"
    )
    return True


def prime_sc_plug_pose(policy, get_observation, move_robot) -> bool:
    """Measure ``sc_tip_link`` and cache the per-grasp TCP->tip transform.

    Resets and re-solves ``policy._sc_grasp_transform`` unconditionally: the
    grasp is new every run, so a transform left over from a previous run is
    exactly the stale assumption this exists to remove.  Fail-closed: any
    missing precondition returns ``False`` and the caller must not insert.
    """

    log = policy.get_logger()
    policy._sc_grasp_transform = None
    if not configure_sc_plug_pose(policy):
        return False
    from .sfp_plug_pose import stamp_to_seconds
    from .v50_controller import _observation_stamp_s, _plug_views_from_observation

    deadline = time.monotonic() + 2.0
    observation = None
    while time.monotonic() < deadline:
        policy._enforce_action_deadline(move_robot)
        candidate = get_observation()
        if candidate is not None and _observation_stamp_s(candidate) is not None:
            observation = candidate
            break
        time.sleep(0.02)
    if observation is None:
        log.error("[sc] no timestamped observation for plug-pose priming")
        return False
    views = _plug_views_from_observation(policy, observation)
    if len(views) < 2:
        log.error(f"[sc] only {len(views)} camera views usable for plug-pose priming")
        return False
    estimator = policy._sc_plug_estimator
    # Image age must be measured in the image headers' clock domain (ROS
    # simulation time), never against time.monotonic().
    now_s = stamp_to_seconds(policy._parent_node.get_clock().now())
    try:
        detections = estimator.detect_views(views)
    except Exception as exc:
        log.error(
            f"[sc] PLUG_POSE_REJECT reason=detector_error:{type(exc).__name__}:{exc}"
        )
        return False
    by_camera = {d.camera_name: d for d in detections}
    for view in views:
        detection = by_camera.get(view.camera_name)
        if detection is None:
            log.info(
                f"[sc] PLUG_POSE_INPUT camera={view.camera_name} "
                f"stamp={view.stamp_s:.3f} detection=none"
            )
            continue
        kp_conf = np.asarray(detection.keypoint_confidences, dtype=np.float64)
        usable = int(np.count_nonzero(kp_conf >= estimator.min_keypoint_confidence))
        log.info(
            f"[sc] PLUG_POSE_INPUT camera={view.camera_name} "
            f"stamp={view.stamp_s:.3f} box_conf={detection.box_confidence:.3f} "
            f"usable_kp={usable}/{len(kp_conf)}"
        )
    estimate = estimator.estimate_multiview(
        views,
        now_s=now_s,
        max_age_s=_env_float("RL_INSERT_SC_PLUG_MAX_AGE_S", 0.35),
        detections=detections,
    )
    if estimate is None:
        log.error(
            "[sc] PLUG_POSE_REJECT reason="
            f"{getattr(estimator, 'last_failure_reason', None) or 'unknown'} "
            "no_fixed_grasp_fallback=true"
        )
        return False
    tcp_pos, tcp_quat = policy._tcp()
    policy._sc_grasp_transform = solve_tip_in_tcp(
        tcp_pos,
        tcp_quat,
        estimate.position_world,
        estimate.rotation_world_from_plug,
    )
    # Log the measured-vs-constant disagreement: this is the number the calib
    # dump could never produce, and it is the direct measurement of the +7 mm
    # phantom depth the fixed-grasp default caused in the field.
    const_tip, _ = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
    delta_mm = (np.asarray(estimate.position_world) - const_tip) * 1000.0
    log.info(
        f"[sc] plug-pose primed: confidence={estimate.confidence:.3f} "
        f"views={estimate.view_count} reproj={estimate.reprojection_error_px:.2f}px "
        f"tip={np.round(estimate.position_world, 5).tolist()} "
        f"measured_minus_fixed_grasp_mm={np.round(delta_mm, 2).tolist()}"
    )
    return True


# --------------------------------------------------------------------------
# Skill entry point.
# --------------------------------------------------------------------------
def run_sc_insertion(policy, task, get_observation, move_robot, send_feedback) -> bool:
    """Perceive the SC opening, sanity-check the handoff, align, and seat."""
    from .rl_insert_contract import port_frame

    log = policy.get_logger()
    timing = _ScTiming(policy, log)
    timing.mark("sc_start")

    def finish(outcome: bool, reason: str) -> bool:
        timing.summary(outcome, reason=reason)
        return bool(outcome)

    # FIRST, before perception even runs.  The grasp calibration needs only the
    # TCP and the plug's ground-truth TF frame -- it does not depend on the port
    # pose, the cameras, or anything perception produces.  Placing it after
    # perception (as it was on 2026-07-25) means a perception failure costs the
    # calibration sample too, and the 10-grasp campaign that 6c requires cannot
    # make progress on any run where the port is not found.  Logs only.
    if _sc_calib_dump_enabled():
        dump_sc_grasp_calibration(policy, task)

    # Measure the grasped plug BEFORE port perception, mirroring v50: the port
    # tie-break ranks candidates by distance to the tip, so even candidate
    # selection should use measured plug geometry.  The robot is stationary
    # here, so the transform stays valid through the perception dwell.
    send_feedback("sc plug-pose priming")
    timing.mark("plug_pose_start")
    _sc_enforce_action_deadline(policy, move_robot)
    if not prime_sc_plug_pose(policy, get_observation, move_robot):
        log.error(
            "[sc] plug-pose priming failed -- refusing to insert without a "
            "measured tip (no fixed-grasp fallback)"
        )
        return finish(False, "plug_pose_failed")
    _sc_enforce_action_deadline(policy, move_robot)
    timing.mark("plug_pose_ready")

    send_feedback("sc opening perception")
    timing.mark("port_perception_start")

    perceived = perceive_sc_port_pose_consensus(
        policy, task, get_observation, move_robot
    )
    if perceived is None:
        log.error("[sc] perception failed to produce an SC port pose")
        return finish(False, "port_perception_failed")
    port_pos, port_quat, reproj_px = perceived
    port_quat = np.asarray(port_quat, dtype=np.float64)
    port_quat /= max(float(np.linalg.norm(port_quat)), 1e-12)
    log.info(
        f"[sc] perceived port p={np.round(port_pos, 5).tolist()} "
        f"q_wxyz={np.round(port_quat, 5).tolist()} reproj={reproj_px:.2f}px"
    )
    timing.mark("port_perceived", reproj_px=f"{reproj_px:.3f}")

    obs = get_observation()
    if obs is None:
        log.error("[sc] no observation available for the wrench baseline")
        return finish(False, "wrench_baseline_missing")
    policy._wrench_baseline = policy._wrench_vector(obs)

    Rp = port_frame(port_quat)
    tcp_pos, tcp_quat = policy._tcp()
    tip_pos, R_tip = sc_tip_from_tcp(policy, tcp_pos, tcp_quat)
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
        return finish(False, "handoff_distance")

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
        if sc_grasp_transform(policy) is None:
            cause = (
                f"SC_TIP_IN_TCP_POS is "
                f"{'CALIBRATED' if SC_TIP_CALIBRATED else 'the UNCALIBRATED SFP default'}"
                "; re-solve it with RL_INSERT_CALIB_DUMP=1 over ~10 grasps."
            )
        else:
            cause = (
                "the tip is the per-grasp MEASURED plug pose, so either the "
                "upstream macro parked inside the port envelope or the "
                "measurement itself is wrong -- read the PLUG_POSE_INPUT lines."
            )
        log.error(
            f"[sc] handoff depth is {handoff_delta[2]*1000:+.2f}mm -- the plug tip "
            "is computed to be INSIDE the port before any motion, which is "
            f"impossible (gate {SC_MAX_HANDOFF_DEPTH_M*1000:.1f}mm). "
            f"{cause} Refusing to seat against a tip position this wrong."
        )
        return finish(False, "handoff_depth")

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
    timing.mark(
        "handoff_validated",
        lateral_mm=f"{np.linalg.norm(handoff_delta[:2]) * 1000.0:.3f}",
        depth_mm=f"{handoff_delta[2] * 1000.0:.3f}",
    )
    outcome = ScInsertionController(
        policy, task, get_observation, move_robot, send_feedback,
        port_pos=port_pos, port_quat=port_quat, Rp=Rp, Rs=Rs, timing=timing,
    ).run()
    return finish(outcome, "seated" if outcome else "controller_failed")


__all__ = [
    "SCConfig",
    "SC_BORE_PITCH_M",
    "SC_INSERT_DEPTH_M",
    "SC_OPENING_HEIGHT_M",
    "SC_OPENING_WIDTH_M",
    "SC_POSE_CONVENTION",
    "SC_POSE_HEIGHT_M",
    "SC_POSE_LOCAL_KPS_M",
    "SC_POSE_WIDTH_M",
    "SC_PRESERVE_HANDOFF_YAW",
    "ScInsertionController",
    "configure_sc_plug_pose",
    "next_sc_depth",
    "perceive_sc_port_pose",
    "perceive_sc_port_pose_consensus",
    "prime_sc_plug_pose",
    "run_sc_insertion",
    "sc_grasp_transform",
    "sc_multiview_candidates",
    "sc_tcp_pose_for_tip",
    "sc_tip_from_tcp",
    "sc_tip_pose_from_tcp",
    "seat_frame",
    "tcp_pose_for_sc_tip",
]
