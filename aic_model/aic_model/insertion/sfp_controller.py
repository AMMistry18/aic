"""Current plug-relative deterministic SFP insertion controller.

This module is the canonical SFP state machine used by ``InsertionPolicy``.
Keeping the controller separate from the ROS policy entry point makes its
motion, perception, and recovery behavior testable without importing ROS.

The controller has one non-negotiable perception contract: a fresh, validated
plug pose is required before motion and after every lift recovery.  The old
fixed cable-angle bias is never used as a fallback.  Once a fresh plug pose has
been observed, the measured TCP-to-plug transform is used kinematically while
the cameras become occluded during seating.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import time
from typing import Optional

import numpy as np


INSERT_DEPTH_M = 0.0458
LOCAL_SFP_PORT_KPS = np.array(
    [
        [0.00685, 0.0043, 0.0],
        [-0.00685, 0.0043, 0.0],
        [-0.00685, -0.0043, 0.0],
        [0.00685, -0.0043, 0.0],
    ],
    dtype=np.float64,
)

SEATED = "seated"
STALLED = "stalled"
HARD_FAILURE = "hard_failure"
# WEDGED means the plug is stuck short of the seat -- either the gross
# lateral/rotation excursion check tripped, or it stopped advancing while still
# in the bore (diag-5's ~36 mm stalls carried 4.3-4.8 N of lateral bind, which is
# a jam by any other name). Only WEDGED earns a retract-and-retry: STALLED is
# reserved for a plug at seat depth whose event never arrived, where backing out
# would throw away an insertion that may already be physically complete.
WEDGED = "wedged"
# Correction gains below ship at 1e-4 / 0.01 -- deliberately short of the
# observe constants (0.00015 / 0.02).  Diag-5's ~36 mm stalls carried 4-5 N of
# lateral bind while the correction sat near 0.1 mm, which is the authority the
# raise buys; the 0.4 mm era exists because a drastic correction can worsen a
# bind, so seat_align_max_step_m bounds the per-sample slew.
# P3's MOUTH_SPEED_SCALE=0.5 and STALL_GRACE_S=3.0 are now the bench-tuned
# defaults below; re-check SEAT_SLOPE/SEAT_WRENCH logs if retuning further.
SEAT_ALIGN_OBSERVE_FORCE_GAIN = 0.00015
SEAT_ALIGN_OBSERVE_MOMENT_GAIN = 0.02
SEAT_WRENCH_LOG_PERIOD_S = 0.2
SEAT_SLOPE_LOG_PERIOD_S = 0.2
SEAT_SLOPE_WINDOW_S = 0.5
MAX_PORT_REPROJ_PX = float(os.environ.get("RL_INSERT_MAX_REPROJ_PX", "25.0"))
# The force/torque sensor is fixed at ``ati/tool_link``.  ``gripper/tcp`` is
# +172 mm along the same axes (the only participant TF the policy may request).
# The per-run visual TCP->tip transform supplies the remaining lever arm.
WRENCH_SENSOR_TO_TCP_POS_M = np.array([0.0, 0.0, 0.172], dtype=np.float64)

_DEEP_RECOVERY_ACCEPTED = "deep_recovery_accepted"
_DEEP_RECOVERY_FAILED = "deep_recovery_failed"
_DEEP_RECOVERY_SKIPPED = "deep_recovery_skipped"


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class SfpConfig:
    """SFP safety and timing limits, expressed in wall time where applicable."""

    command_dt_sim_s: float = 0.05
    align_timeout_wall_s: float = 15.0
    align_lateral_tol_m: float = 0.0010
    align_rotation_tol_rad: float = np.deg2rad(1.5)
    align_max_lateral_step_m: float = 0.0015
    align_max_rotation_step_rad: float = np.deg2rad(1.5)
    stall_timeout_wall_s: float = 2.5
    stall_progress_m: float = 0.0008
    free_speed_m_s: float = 0.015
    contact_speed_m_s: float = 0.006
    contact_force_n: float = 3.0
    target_axial_force_n: float = 10.0
    seat_force_cap_n: float = 12.0
    force_abort_n: float = 18.0
    force_abort_wall_s: float = 0.25
    axial_stiffness_n_m: float = 500.0
    max_axial_lead_m: float = 0.020
    lateral_safety_m: float = 0.006
    rotation_safety_rad: float = np.deg2rad(15.0)
    seat_align_enable: bool = True
    seat_align_force_gain: float = 0.0001
    seat_align_moment_gain: float = 0.01
    seat_align_max_lat_m: float = 0.0007
    seat_align_max_tilt_rad: float = 0.0122
    # Per-sample slew limit on the correction, independent of the low pass. The
    # 0.4 mm cap era exists because a drastic correction can worsen a bind; this
    # bounds how fast the target can be chased even when the wrench is extreme,
    # so raising the gain buys authority without buying a lurch.
    seat_align_max_step_m: float = 0.00015
    seat_align_max_tilt_step_rad: float = 0.0026
    # Low-pass coefficient on the seat alignment correction: 0 = raw proportional,
    # ->1 = heavily smoothed. Named "release" for the env var it already ships with.
    seat_align_release_decay: float = 0.7
    seat_mouth_zone_m: float = 0.006
    seat_mouth_speed_scale: float = 0.5
    seat_stall_grace_s: float = 3.0
    # Diag-5: three ~36 mm (believed) stalls all pressed exactly the setpoint
    # cap's spring budget (~7.4 N at 5 mm overtravel) and could not advance;
    # 8 mm is the validated ceiling and buys ~20% more press.
    seat_overtravel_m: float = 0.008
    seat_candidate_depth_m: float = 0.0445
    insertion_event_timeout_wall_s: float = 10.0
    plug_max_age_s: float = 0.35
    # Wedge retry: back the plug out to where it started and run the whole
    # attempt again. Retries are unbounded by count (max_wedge_retries=0); the
    # action deadline is the only terminator, so the budget is what actually
    # limits how many attempts fit in a trial.
    wedge_retry_enable: bool = True
    max_wedge_retries: int = 0
    wedge_retry_on_wall_stall: bool = True
    # A deep wall-stall is normally a side-wall/tilt bind, not a reason to
    # discard the whole insertion.  Before the legacy visual/full-retract
    # fallback, unload slightly and verify one force-guided micro-correction.
    # Every motion here is deliberately smaller than the port clearance and
    # bounded by a finite local budget.
    wedge_recovery_enable: bool = True
    wedge_recovery_min_depth_m: float = 0.030
    wedge_recovery_max_local_attempts: int = 2
    wedge_recovery_unload_m: float = 0.0010
    wedge_recovery_min_unload_m: float = 0.0005
    wedge_recovery_unload_timeout_wall_s: float = 1.5
    wedge_recovery_probe_depth_m: float = 0.00075
    wedge_recovery_probe_speed_m_s: float = 0.002
    wedge_recovery_probe_timeout_wall_s: float = 1.5
    wedge_recovery_min_progress_m: float = 0.00025
    wedge_recovery_force_gain_m_per_n: float = 0.00005
    wedge_recovery_moment_gain_rad_per_nm: float = 0.003
    wedge_recovery_probe_max_lat_m: float = 0.00020
    wedge_recovery_probe_max_tilt_rad: float = np.deg2rad(0.25)
    wedge_recovery_max_lat_m: float = 0.00070
    wedge_recovery_max_tilt_rad: float = 0.0122
    wedge_recovery_min_lat_force_n: float = 1.0
    wedge_recovery_min_moment_nm: float = 0.10
    wedge_recovery_score_improvement_fraction: float = 0.10
    retract_clear_depth_m: float = -0.003
    retract_step_m: float = 0.0015
    retract_free_step_m: float = 0.004
    retract_arrive_tol_m: float = 0.002
    retract_timeout_wall_s: float = 10.0
    # Cap on how far the retract setpoint may lead the plug OUTWARD, i.e. the
    # reverse of force_lead_m.  At the 500 N/m axial stiffness the default
    # bounds the nominal pull at ~12 N -- enough to break the 4-5 N lateral
    # binds diag-5 measured, well under the 18 N hard abort.
    retract_pull_lead_m: float = 0.024

    @classmethod
    def from_env(cls) -> "SfpConfig":
        return cls(
            command_dt_sim_s=_env_float("RL_INSERT_V50_COMMAND_DT_S", 0.05),
            align_timeout_wall_s=_env_float("RL_INSERT_V50_ALIGN_TIMEOUT_S", 15.0),
            stall_timeout_wall_s=_env_float("RL_INSERT_V50_STALL_TIMEOUT_S", 2.5),
            stall_progress_m=_env_float("RL_INSERT_V50_STALL_PROGRESS_M", 0.0008),
            free_speed_m_s=_env_float("RL_INSERT_V50_FREE_SPEED_M_S", 0.015),
            contact_speed_m_s=_env_float("RL_INSERT_V50_CONTACT_SPEED_M_S", 0.006),
            contact_force_n=_env_float("RL_INSERT_V50_CONTACT_FORCE_N", 3.0),
            target_axial_force_n=_env_float("RL_INSERT_V50_TARGET_FORCE_N", 10.0),
            seat_force_cap_n=_env_float("RL_INSERT_V50_SEAT_FORCE_CAP_N", 12.0),
            force_abort_n=_env_float("RL_INSERT_FORCE_ABORT_N", 18.0),
            force_abort_wall_s=_env_float("RL_INSERT_V50_FORCE_ABORT_DWELL_S", 0.25),
            axial_stiffness_n_m=_env_float("RL_INSERT_V50_AXIAL_STIFFNESS_N_M", 500.0),
            max_axial_lead_m=_env_float("RL_INSERT_V50_MAX_AXIAL_LEAD_M", 0.020),
            seat_align_enable=_env_bool("RL_INSERT_V50_SEAT_ALIGN_ENABLE", True),
            seat_align_force_gain=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_FORCE_GAIN", 0.0001
            ),
            seat_align_moment_gain=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MOMENT_GAIN", 0.01
            ),
            seat_align_max_lat_m=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MAX_LAT_M", 0.0007
            ),
            seat_align_max_tilt_rad=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MAX_TILT_RAD", 0.0122
            ),
            seat_align_max_step_m=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MAX_STEP_M", 0.00015
            ),
            seat_align_max_tilt_step_rad=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_MAX_TILT_STEP_RAD", 0.0026
            ),
            seat_align_release_decay=_env_float(
                "RL_INSERT_V50_SEAT_ALIGN_RELEASE_DECAY", 0.7
            ),
            seat_mouth_zone_m=_env_float("RL_INSERT_V50_SEAT_MOUTH_ZONE_M", 0.006),
            seat_mouth_speed_scale=_env_float(
                "RL_INSERT_V50_SEAT_MOUTH_SPEED_SCALE", 0.5
            ),
            seat_stall_grace_s=_env_float("RL_INSERT_V50_SEAT_STALL_GRACE_S", 3.0),
            seat_overtravel_m=_env_float("RL_INSERT_V50_SEAT_OVERTRAVEL_M", 0.008),
            insertion_event_timeout_wall_s=_env_float(
                "RL_INSERT_V50_EVENT_TIMEOUT_S", 10.0
            ),
            plug_max_age_s=_env_float("RL_INSERT_V50_PLUG_MAX_AGE_S", 0.35),
            wedge_retry_enable=_env_bool("RL_INSERT_V50_WEDGE_RETRY_ENABLE", True),
            max_wedge_retries=_env_int("RL_INSERT_V50_MAX_WEDGE_RETRIES", 0),
            wedge_retry_on_wall_stall=_env_bool(
                "RL_INSERT_V50_WEDGE_RETRY_ON_WALL_STALL", True
            ),
            wedge_recovery_enable=_env_bool(
                "RL_INSERT_V50_WEDGE_RECOVERY_ENABLE", True
            ),
            wedge_recovery_min_depth_m=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MIN_DEPTH_M", 0.030
            ),
            wedge_recovery_max_local_attempts=_env_int(
                "RL_INSERT_V50_WEDGE_RECOVERY_MAX_LOCAL_ATTEMPTS", 2
            ),
            wedge_recovery_unload_m=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_UNLOAD_M", 0.0010
            ),
            wedge_recovery_min_unload_m=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MIN_UNLOAD_M", 0.0005
            ),
            wedge_recovery_unload_timeout_wall_s=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_UNLOAD_TIMEOUT_S", 1.5
            ),
            wedge_recovery_probe_depth_m=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_PROBE_DEPTH_M", 0.00075
            ),
            wedge_recovery_probe_speed_m_s=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_PROBE_SPEED_M_S", 0.002
            ),
            wedge_recovery_probe_timeout_wall_s=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_PROBE_TIMEOUT_S", 1.5
            ),
            wedge_recovery_min_progress_m=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MIN_PROGRESS_M", 0.00025
            ),
            wedge_recovery_force_gain_m_per_n=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_FORCE_GAIN_M_PER_N", 0.00005
            ),
            wedge_recovery_moment_gain_rad_per_nm=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MOMENT_GAIN_RAD_PER_NM", 0.003
            ),
            wedge_recovery_probe_max_lat_m=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_PROBE_MAX_LAT_M", 0.00020
            ),
            wedge_recovery_probe_max_tilt_rad=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_PROBE_MAX_TILT_RAD",
                float(np.deg2rad(0.25)),
            ),
            wedge_recovery_max_lat_m=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MAX_LAT_M", 0.00070
            ),
            wedge_recovery_max_tilt_rad=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MAX_TILT_RAD",
                0.0122,
            ),
            wedge_recovery_min_lat_force_n=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MIN_LAT_FORCE_N", 1.0
            ),
            wedge_recovery_min_moment_nm=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_MIN_MOMENT_NM", 0.10
            ),
            wedge_recovery_score_improvement_fraction=_env_float(
                "RL_INSERT_V50_WEDGE_RECOVERY_SCORE_IMPROVEMENT", 0.10
            ),
            retract_clear_depth_m=_env_float(
                "RL_INSERT_V50_RETRACT_CLEAR_DEPTH_M", -0.003
            ),
            retract_step_m=_env_float("RL_INSERT_V50_RETRACT_STEP_M", 0.0015),
            retract_free_step_m=_env_float(
                "RL_INSERT_V50_RETRACT_FREE_STEP_M", 0.004
            ),
            retract_arrive_tol_m=_env_float(
                "RL_INSERT_V50_RETRACT_ARRIVE_TOL_M", 0.002
            ),
            retract_timeout_wall_s=_env_float(
                "RL_INSERT_V50_RETRACT_TIMEOUT_S", 10.0
            ),
            retract_pull_lead_m=_env_float(
                "RL_INSERT_V50_RETRACT_PULL_LEAD_M", 0.024
            ),
        ).validated()

    def validated(self) -> "SfpConfig":
        if not 0.0 < self.target_axial_force_n < self.seat_force_cap_n:
            raise ValueError("SFP target force must be below the seat force cap")
        if not self.seat_force_cap_n < self.force_abort_n <= 18.0:
            raise ValueError("SFP force cap must be below the <=18 N hard abort")
        if self.axial_stiffness_n_m <= 0.0:
            raise ValueError("SFP axial stiffness must be positive")
        if not 0.0 <= self.seat_overtravel_m <= 0.008:
            raise ValueError("SFP seat overtravel must stay within 0-8 mm")
        if self.seat_align_max_lat_m < 0.0 or self.seat_align_max_tilt_rad < 0.0:
            raise ValueError("SFP seat alignment correction caps must be non-negative")
        if self.seat_align_max_lat_m > 0.001 or self.seat_align_max_tilt_rad > 0.0175:
            raise ValueError(
                "SFP seat alignment caps must stay within the port clearance "
                "(<=1 mm lateral, <=1 deg tilt)"
            )
        if self.seat_align_max_step_m <= 0.0 or self.seat_align_max_tilt_step_rad <= 0.0:
            raise ValueError("SFP seat alignment slew limits must be positive")
        if self.max_wedge_retries < 0:
            raise ValueError("SFP max wedge retries must be >= 0 (0 means unlimited)")
        if not 0.0 <= self.wedge_recovery_min_depth_m < self.seat_candidate_depth_m:
            raise ValueError(
                "SFP deep wedge recovery depth must be inside the pre-seat bore"
            )
        if self.wedge_recovery_max_local_attempts < 0:
            raise ValueError("SFP deep wedge recovery attempt budget must be >= 0")
        if (
            self.wedge_recovery_unload_m <= 0.0
            or self.wedge_recovery_min_unload_m <= 0.0
            or self.wedge_recovery_min_unload_m > self.wedge_recovery_unload_m
            or self.wedge_recovery_unload_timeout_wall_s <= 0.0
            or self.wedge_recovery_probe_depth_m <= 0.0
            or self.wedge_recovery_probe_speed_m_s <= 0.0
            or self.wedge_recovery_probe_timeout_wall_s <= 0.0
            or self.wedge_recovery_min_progress_m <= 0.0
        ):
            raise ValueError("SFP deep wedge recovery distances and times must be positive")
        if self.wedge_recovery_min_progress_m > self.wedge_recovery_probe_depth_m:
            raise ValueError(
                "SFP deep wedge recovery progress must fit inside its probe"
            )
        if (
            self.wedge_recovery_force_gain_m_per_n < 0.0
            or self.wedge_recovery_moment_gain_rad_per_nm < 0.0
            or self.wedge_recovery_min_lat_force_n < 0.0
            or self.wedge_recovery_min_moment_nm < 0.0
        ):
            raise ValueError("SFP deep wedge recovery wrench settings must be non-negative")
        if (
            self.wedge_recovery_probe_max_lat_m <= 0.0
            or self.wedge_recovery_probe_max_tilt_rad <= 0.0
            or self.wedge_recovery_max_lat_m <= 0.0
            or self.wedge_recovery_max_tilt_rad <= 0.0
            or self.wedge_recovery_probe_max_lat_m > self.wedge_recovery_max_lat_m
            or self.wedge_recovery_probe_max_tilt_rad > self.wedge_recovery_max_tilt_rad
            or self.wedge_recovery_max_lat_m > self.seat_align_max_lat_m
            or self.wedge_recovery_max_tilt_rad > self.seat_align_max_tilt_rad
        ):
            raise ValueError(
                "SFP deep wedge recovery must stay inside the seat-alignment envelope"
            )
        if not 0.0 < self.wedge_recovery_score_improvement_fraction < 1.0:
            raise ValueError(
                "SFP deep wedge recovery score improvement must be within (0, 1)"
            )
        if self.retract_clear_depth_m > 0.0:
            raise ValueError(
                "SFP retract clear depth must be at or outside the port mouth"
            )
        if (
            self.retract_step_m <= 0.0
            or self.retract_free_step_m <= 0.0
            or self.retract_arrive_tol_m <= 0.0
            or self.retract_timeout_wall_s <= 0.0
            or self.retract_pull_lead_m <= 0.0
        ):
            raise ValueError("SFP retract parameters must be positive")
        if self.axial_stiffness_n_m * self.retract_pull_lead_m >= self.force_abort_n:
            raise ValueError(
                "SFP retract pull lead demands a nominal pull at or above the "
                "hard force abort; lower RL_INSERT_V50_RETRACT_PULL_LEAD_M"
            )
        if not 0.0 <= self.seat_align_release_decay <= 1.0:
            raise ValueError("SFP seat alignment release decay must be within 0-1")
        if (
            self.seat_mouth_zone_m < 0.0
            or self.seat_mouth_speed_scale < 0.0
            or self.seat_stall_grace_s < 0.0
        ):
            raise ValueError("SFP P3 observe-first parameters must be non-negative")
        return self

    @property
    def force_lead_m(self) -> float:
        return min(
            self.max_axial_lead_m,
            self.target_axial_force_n / self.axial_stiffness_n_m,
        )


def quaternion_to_matrix(quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s,
                          (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                          (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s,
                          (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0.0:
        q = -q
    return q


def axis_angle(rotation) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))
    if angle <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    denom = 2.0 * np.sin(angle)
    if abs(denom) <= 1e-9:
        values, vectors = np.linalg.eigh((R + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
    else:
        axis /= denom
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return axis * angle


def rotation_from_axis_angle(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle <= 1e-12:
        return np.eye(3)
    axis = v / angle
    K = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def clamp_vector_norm(vector, max_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm):
        return np.zeros_like(value)
    if max_norm <= 0.0:
        return np.zeros_like(value)
    if norm > max_norm:
        return value * (max_norm / norm)
    return value


def shift_wrench_moment_to_point(force, moment_at_source, source_to_point) -> np.ndarray:
    """Express a wrench moment at a point displaced from its source.

    ``source_to_point`` points from the origin at which the wrench was measured
    to the point at which the returned moment is wanted.  This is the standard
    spatial-wrench reference shift: ``M_point = M_source - r x F``.
    """

    force = np.asarray(force, dtype=np.float64).reshape(3)
    moment = np.asarray(moment_at_source, dtype=np.float64).reshape(3)
    lever = np.asarray(source_to_point, dtype=np.float64).reshape(3)
    if not (
        np.all(np.isfinite(force))
        and np.all(np.isfinite(moment))
        and np.all(np.isfinite(lever))
    ):
        return np.full(3, np.nan, dtype=np.float64)
    return moment - np.cross(lever, force)


def transverse_wrench_score(
    force_port,
    moment_port,
    config: SfpConfig,
) -> float:
    """Dimensionless side-contact score used to accept a recovery probe."""

    force = np.asarray(force_port, dtype=np.float64).reshape(3)
    moment = np.asarray(moment_port, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(force[:2])):
        return float("inf")
    score = float(np.linalg.norm(force[:2])) / max(
        config.wedge_recovery_min_lat_force_n, 1e-9
    )
    # A missing/untrusted moment must not disable the force-only fallback.
    if np.all(np.isfinite(moment[:2])):
        score += float(np.linalg.norm(moment[:2])) / max(
            config.wedge_recovery_min_moment_nm, 1e-9
        )
    return score


def deep_wedge_recovery_probes(
    force_port,
    moment_port,
    config: SfpConfig,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return bounded signed recovery probes in the port/action frame.

    The first probe follows the controller's established ``-F``/``-M`` sign
    convention.  A compliant cable or residual calibration error can invert
    that intuition, so callers measure the result and try the mirrored probe
    once rather than treating force direction as ground truth.
    """

    force = np.asarray(force_port, dtype=np.float64).reshape(3)
    moment = np.asarray(moment_port, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(force[:2])):
        return ()
    lateral_force = float(np.linalg.norm(force[:2]))
    moment_valid = bool(np.all(np.isfinite(moment[:2])))
    lateral_moment = float(np.linalg.norm(moment[:2])) if moment_valid else 0.0
    if (
        lateral_force < config.wedge_recovery_min_lat_force_n
        and lateral_moment < config.wedge_recovery_min_moment_nm
    ):
        # Axial-only loading can be a bottom stop or hard obstruction.  Do not
        # scrub sideways without evidence of a side-wall/tilt contact.
        return ()

    lateral = clamp_vector_norm(
        -config.wedge_recovery_force_gain_m_per_n * force[:2],
        config.wedge_recovery_probe_max_lat_m,
    )
    tilt = np.zeros(2, dtype=np.float64)
    if moment_valid:
        tilt = clamp_vector_norm(
            -config.wedge_recovery_moment_gain_rad_per_nm * moment[:2],
            config.wedge_recovery_probe_max_tilt_rad,
        )
    if not np.any(lateral) and not np.any(tilt):
        return ()
    return ((lateral, tilt), (-lateral, -tilt))


def solve_tip_in_tcp(tcp_pos, tcp_quat, tip_pos, tip_rotation):
    """Return the measured rigid plug-tip transform in the current TCP frame."""

    tcp_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
    tip_pos = np.asarray(tip_pos, dtype=np.float64).reshape(3)
    R_tcp = quaternion_to_matrix(tcp_quat)
    R_tip = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    return R_tcp.T @ (tip_pos - tcp_pos), R_tcp.T @ R_tip


def tip_from_tcp_transform(tcp_pos, tcp_quat, tip_in_tcp_pos, tip_in_tcp_rotation):
    R_tcp = quaternion_to_matrix(tcp_quat)
    tip_pos = (
        np.asarray(tcp_pos, dtype=np.float64).reshape(3)
        + R_tcp @ np.asarray(tip_in_tcp_pos, dtype=np.float64).reshape(3)
    )
    tip_rotation = R_tcp @ np.asarray(tip_in_tcp_rotation, dtype=np.float64).reshape(3, 3)
    return tip_pos, tip_rotation


def tcp_for_tip_transform(tip_pos, tip_rotation, tip_in_tcp_pos, tip_in_tcp_rotation):
    R_tip = np.asarray(tip_rotation, dtype=np.float64).reshape(3, 3)
    R_rel = np.asarray(tip_in_tcp_rotation, dtype=np.float64).reshape(3, 3)
    R_tcp = R_tip @ R_rel.T
    tcp_pos = (
        np.asarray(tip_pos, dtype=np.float64).reshape(3)
        - R_tcp @ np.asarray(tip_in_tcp_pos, dtype=np.float64).reshape(3)
    )
    return tcp_pos, matrix_to_quaternion(R_tcp)


def sfp_tip_from_tcp(policy, tcp_pos, tcp_quat):
    """Use the per-run visual grasp transform once it has been established.

    The legacy transform remains available only before SFP control starts so the
    old port detector can rank multiple port candidates.  ``run_sfp_insertion``
    refuses to move unless ``_sfp_grasp_transform`` has been populated from a
    fresh visual plug observation.
    """

    transform = getattr(policy, "_sfp_grasp_transform", None)
    if transform is None:
        from .contract import sfp_tip_pose_from_tcp

        return sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)
    return tip_from_tcp_transform(tcp_pos, tcp_quat, *transform)


def sfp_tcp_pose_for_tip(policy, tip_pos, tip_rotation):
    transform = getattr(policy, "_sfp_grasp_transform", None)
    if transform is None:
        from .contract import tcp_pose_for_sfp_tip

        return tcp_pose_for_sfp_tip(tip_pos, tip_rotation)
    return tcp_for_tip_transform(tip_pos, tip_rotation, *transform)


def next_persistent_depth(
    current_depth: float,
    commanded_depth: float,
    elapsed_wall_s: float,
    force_n: float,
    config: SfpConfig,
    axial_force_n: Optional[float] = None,
) -> float:
    """Advance an absolute seat setpoint while retaining bounded axial lead.

    The freeze and contact-slowdown decisions read the plug-frame AXIAL force
    when it is available, not the three-axis norm. Diag-5's ~36 mm stalls carried
    4.3-4.8 N of lateral bind against 6-7 N axial, so a norm-gated cap freezes
    the setpoint on scrape and cable drag that the plug could still push through.
    The >=18 N hard abort stays on the norm -- that one is a safety limit.
    """

    current_depth = float(current_depth)
    commanded_depth = max(float(commanded_depth), current_depth)
    gate_force = force_n
    if axial_force_n is not None and np.isfinite(axial_force_n):
        gate_force = abs(float(axial_force_n))
    if np.isfinite(gate_force) and gate_force >= config.seat_force_cap_n:
        candidate = commanded_depth
    else:
        speed = (
            config.contact_speed_m_s
            if np.isfinite(gate_force) and gate_force >= config.contact_force_n
            else config.free_speed_m_s
        )
        candidate = commanded_depth + speed * max(0.0, float(elapsed_wall_s))
    # This is a persistent absolute setpoint.  It grows while the plug is stuck,
    # unlike v49's ``current_tip + one_step`` command.  The lead cap bounds the
    # nominal impedance demand to approximately target_axial_force_n.
    return min(
        INSERT_DEPTH_M + config.seat_overtravel_m,
        candidate,
        current_depth + config.force_lead_m,
    )


def next_retract_depth(
    current_depth: float,
    commanded_depth: float,
    config: SfpConfig,
) -> float:
    """Recede an absolute retract setpoint while bounding the pull.

    ``next_persistent_depth`` run in reverse, for the same reason: commanding
    ``current_tip - one_step`` afresh every tick pins the offset at
    ``retract_step_m`` forever, so the pull plateaus at stiffness * step
    (~0.75 N) -- the v49 chase-the-tip mistake, reproduced backwards.  A wedge
    held by diag-5's 4-5 N of lateral bind never comes out under that; it
    grinds until the timeout and the retry dies.  Here the setpoint keeps
    RECEDING while the plug is stuck, so the pull builds until the lead cap
    bounds the nominal demand at axial_stiffness_n_m * retract_pull_lead_m
    (~12 N by default, under the 18 N hard abort; enforced by ``validated``).
    """

    current_depth = float(current_depth)
    # Never sit deeper than the plug actually is: after a sudden pop-free the
    # stale setpoint would otherwise command the plug back INTO the bore.
    commanded_depth = min(float(commanded_depth), current_depth)
    # Deliberately no retract_clear_depth_m floor: a plug stuck just inside the
    # mouth needs the full pull as much as a deep wedge does, and flooring at
    # the clear depth would cap it at stiffness * (depth - clear) instead.  The
    # caller's clearance check terminates the walk; the lead cap keeps the
    # command within retract_pull_lead_m of the measured tip everywhere.
    return max(
        commanded_depth - config.retract_step_m,
        current_depth - config.retract_pull_lead_m,
    )


@dataclass
class WallProgressWatch:
    progress_m: float
    timeout_s: float
    best_depth: float
    progress_time: float

    @classmethod
    def start(cls, depth: float, now: float, config: SfpConfig):
        return cls(config.stall_progress_m, config.stall_timeout_wall_s, depth, now)

    def stalled(self, depth: float, now: float) -> bool:
        if float(depth) >= self.best_depth + self.progress_m:
            self.best_depth = float(depth)
            self.progress_time = float(now)
        return float(now) - self.progress_time >= self.timeout_s


def configure_sfp_controller(policy) -> None:
    """Load the dedicated plug estimator and initialize per-action state."""

    if getattr(policy, "_sfp_configured", False):
        return
    from .sfp_plug_pose import SfpPlugPoseEstimator

    weights = Path(
        os.environ.get("AIC_SFP_PLUG_POSE_WEIGHTS", "/models/sfp_plug_pose_best.pt")
    )
    if not weights.is_file():
        raise FileNotFoundError(
            f"SFP plug-pose weights missing at {weights}; no fixed-bias fallback is allowed"
        )
    policy._sfp_config = SfpConfig.from_env()
    policy._sfp_plug_estimator = SfpPlugPoseEstimator(
        str(weights),
        imgsz=_env_int("RL_INSERT_V50_PLUG_IMGSZ", 960),
        conf_threshold=_env_float("RL_INSERT_V50_PLUG_CONF", 0.25),
    )
    policy._sfp_grasp_transform = None
    policy._sfp_pending_world_plug = None
    _warm_sfp_perception(policy)
    policy._sfp_configured = True
    policy.get_logger().info(
        f"[sfp] plug-relative controller ready; weights={weights} "
        f"force_target={policy._sfp_config.target_axial_force_n:.1f}N "
        f"force_abort={policy._sfp_config.force_abort_n:.1f}N"
    )


def _warm_sfp_perception(policy) -> None:
    """Pay both YOLO first-inference costs during lifecycle configuration."""

    from .sfp_plug_pose import PlugPoseView

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    # A no-detection result is expected for black pixels; an exception is not.
    policy._pc.detect_nic(image, conf_thresh=0.99)
    policy._sfp_plug_estimator.detect_views(
        [
            PlugPoseView(
                camera_name="sfp_warmup",
                image_bgr=image,
                K=np.array(
                    [[500.0, 0.0, 320.0], [0.0, 500.0, 320.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                T_world_from_camera=np.eye(4),
                stamp_s=0.0,
                frame_id="sfp-warmup",
            )
        ]
    )
    policy.get_logger().info(
        "[sfp] port and plug YOLO first-inference warmup completed during configure"
    )


def _message_stamp(message):
    header = getattr(message, "header", None)
    return getattr(header, "stamp", None)


def _observation_stamp_s(observation) -> Optional[float]:
    from .sfp_plug_pose import stamp_to_seconds

    values = []
    for name in ("left_image", "center_image", "right_image"):
        try:
            value = stamp_to_seconds(_message_stamp(getattr(observation, name, None)))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(float(value))
    return max(values) if values else None


# The scoring topic identifies the cable instance that completed the insertion,
# e.g. "cable_0#0#nic_card_mount_0/sfp_port_0", while Task names only the module
# and port. Without stripping that prefix the equality test in _event_status can
# never match, so a correct insertion still reports a wrong-port hard failure.
_CABLE_INSTANCE_PREFIX = re.compile(r"^cable_\d+#\d+#")


def _normalize_event(value: object) -> str:
    text = str(value or "").strip().strip("/")
    return _CABLE_INSTANCE_PREFIX.sub("", text, count=1).strip("/")


def _plug_views_from_observation(policy, observation):
    from .sfp_plug_pose import PlugPoseView, stamp_to_seconds

    messages = {
        "left_camera": getattr(observation, "left_image", None),
        "center_camera": getattr(observation, "center_image", None),
        "right_camera": getattr(observation, "right_image", None),
    }
    result = []
    for camera, (image_bgr, K, T_cam_from_base) in policy._build_views(
        observation
    ).items():
        try:
            stamp_s = stamp_to_seconds(_message_stamp(messages[camera]))
        except (TypeError, ValueError):
            continue
        result.append(
            PlugPoseView(
                camera_name=camera,
                image_bgr=image_bgr,
                K=K,
                T_world_from_camera=policy._pc.invert_transform(T_cam_from_base),
                stamp_s=float(stamp_s),
                frame_id=f"{camera}:{float(stamp_s):.9f}",
            )
        )
    return result


def prime_sfp_plug_pose(policy, get_observation, move_robot) -> bool:
    """Observe the plug before port selection and calibrate its TCP transform.

    Port consensus ranks multiple detected cages by distance to the plug.  SFP
    therefore primes the direct plug estimate first, so even candidate selection
    uses measured plug geometry rather than the legacy grasp assumption.  The
    robot remains stationary while the subsequent port consensus is collected,
    making this same fresh world plug pose valid for initial relative alignment.
    """

    policy._sfp_grasp_transform = None
    policy._sfp_pending_world_plug = None
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
        policy.get_logger().error("[sfp] no timestamped observation for plug priming")
        return False
    views = _plug_views_from_observation(policy, observation)
    if len(views) < 2:
        policy.get_logger().error("[sfp] fewer than two views for plug priming")
        return False
    from .sfp_plug_pose import stamp_to_seconds

    now_s = stamp_to_seconds(policy._parent_node.get_clock().now())
    try:
        detections = policy._sfp_plug_estimator.detect_views(views)
    except Exception as exc:
        policy.get_logger().error(
            f"[sfp] PLUG_POSE_REJECT reason=detector_error:{type(exc).__name__}:{exc}"
        )
        return False
    by_camera = {d.camera_name: d for d in detections}
    for view in views:
        detection = by_camera.get(view.camera_name)
        if detection is None:
            policy.get_logger().info(
                f"[sfp] PLUG_POSE_INPUT camera={view.camera_name} stamp={view.stamp_s:.3f} "
                "detection=none"
            )
            continue
        kp_conf = np.asarray(detection.keypoint_confidences, dtype=np.float64)
        kp_xy = np.asarray(detection.keypoints_px, dtype=np.float64)
        usable = int(np.count_nonzero(kp_conf >= policy._sfp_plug_estimator.min_keypoint_confidence))
        policy.get_logger().info(
            f"[sfp] PLUG_POSE_INPUT camera={view.camera_name} stamp={view.stamp_s:.3f} "
            f"box_conf={detection.box_confidence:.3f} usable_kp={usable}/{len(kp_conf)} "
            f"kp_conf={np.round(kp_conf, 3).tolist()} kp_xy={np.round(kp_xy, 1).tolist()}"
        )
    estimate = policy._sfp_plug_estimator.estimate_multiview(
        views, now_s=now_s, max_age_s=policy._sfp_config.plug_max_age_s,
        detections=detections,
    )
    if estimate is None:
        policy.get_logger().error(
            "[sfp] PLUG_POSE_REJECT reason="
            f"{getattr(policy._sfp_plug_estimator, 'last_failure_reason', None) or 'unknown'} "
            "no_fixed_grasp_fallback=true"
        )
        return False
    tcp_pos, tcp_quat = policy._tcp()
    policy._sfp_grasp_transform = solve_tip_in_tcp(
        tcp_pos,
        tcp_quat,
        estimate.position_world,
        estimate.rotation_world_from_plug,
    )
    policy._sfp_pending_world_plug = estimate
    policy.get_logger().info(
        f"[sfp] direct plug primed before port selection: confidence="
        f"{estimate.confidence:.3f} views={estimate.view_count} "
        f"reproj={estimate.reprojection_error_px:.2f}px "
        f"frames={list(estimate.source_frame_ids)}"
    )
    return True


class SfpInsertionController:
    """Bounded visual -> persistent-seat -> lift/re-perceive state machine."""

    STIFFNESS = [350.0, 350.0, 500.0, 60.0, 60.0, 60.0]
    DAMPING = [90.0, 90.0, 100.0, 25.0, 25.0, 25.0]
    HOLD_STIFFNESS = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    HOLD_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

    def __init__(
        self,
        policy,
        task,
        get_observation,
        move_robot,
        send_feedback,
        *,
        port_pos,
        port_quat,
        Rp,
    ):
        self.policy = policy
        self.task = task
        self.get_observation = get_observation
        self.move_robot = move_robot
        self.send_feedback = send_feedback
        self.config = getattr(policy, "_sfp_config", SfpConfig.from_env())
        self.port_pos = np.asarray(port_pos, dtype=np.float64).reshape(3)
        self.port_quat = np.asarray(port_quat, dtype=np.float64).reshape(4)
        self.Rp = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
        self.log = policy.get_logger()
        parent = policy._parent_node
        self.event_generation = int(getattr(parent, "_insertion_event_generation", 0))
        self.expected_event = _normalize_event(
            f"{getattr(task, 'target_module_name', '')}/{getattr(task, 'port_name', '')}"
        )
        self.last_observation_stamp = None
        self.last_accepted_plug_stamp = None
        self._port_pos_initial = self.port_pos.copy()

    def _event(self):
        parent = self.policy._parent_node
        generation = int(getattr(parent, "_insertion_event_generation", 0))
        value = _normalize_event(getattr(parent, "_insertion_event_value", ""))
        if generation <= self.event_generation:
            return None
        return value

    def _event_status(self):
        value = self._event()
        if value is None:
            return None
        # Deployment policy: any fresh scoring insertion event is success.
        # Flowstate owns task/port bookkeeping, while this controller only
        # needs to confirm that the held cable physically seated.  Rejecting a
        # valid event because its port name differs aborts the whole process
        # after a successful insertion.
        if value != self.expected_event:
            self.log.warn(
                f"[sfp] accepting insertion event for alternate port '{value}' "
                f"(requested '{self.expected_event}')"
            )
        return SEATED

    def _force_magnitude(self, observation) -> float:
        if observation is None:
            return float("nan")
        wrench = self.policy._wrench_vector(observation) - self.policy._wrench_baseline
        return float(np.linalg.norm(wrench[:3]))

    def _wrench_plug_frame(self, observation):
        """Tared wrench at the plug tip, expressed in the port/action frame.

        The historical method name is retained for callers, but its axes are
        the port frame because corrections are commanded along ``Rp``.  The
        source topic is measured at ``ati/tool_link``; torque is shifted to the
        current visual plug tip before rotating into those axes.  If the input
        frame is not the documented sensor frame, force remains available for
        bounded recovery but torque is deliberately marked invalid.
        """
        if observation is None:
            nan3 = np.full(3, np.nan, dtype=np.float64)
            return nan3, nan3.copy()
        wrench_sensor = (
            self.policy._wrench_vector(observation) - self.policy._wrench_baseline
        )
        tcp_pos, tcp_quat = self.policy._tcp()
        R_tcp = quaternion_to_matrix(tcp_quat)
        force_base = R_tcp @ wrench_sensor[:3]
        moment_base_sensor = R_tcp @ wrench_sensor[3:]

        header = getattr(getattr(observation, "wrist_wrench", None), "header", None)
        frame_id = str(getattr(header, "frame_id", "") or "").strip().lstrip("/")
        if frame_id == "ati/tool_link":
            tip_pos, _ = self.policy._tip_from_tcp(tcp_pos, tcp_quat)
            sensor_pos = tcp_pos - R_tcp @ WRENCH_SENSOR_TO_TCP_POS_M
            moment_base_tip = shift_wrench_moment_to_point(
                force_base,
                moment_base_sensor,
                np.asarray(tip_pos, dtype=np.float64).reshape(3) - sensor_pos,
            )
        else:
            # No silent use of a torque origin we cannot prove.  A force-only
            # probe remains safe and lets the recovery fall back cleanly.
            moment_base_tip = np.full(3, np.nan, dtype=np.float64)
        base_to_port = self.Rp.T
        return base_to_port @ force_base, base_to_port @ moment_base_tip

    def _seat_alignment_sample(self, observation, depth, force, acc_lat, acc_tilt):
        """Low-passed *proportional* wrench correction, not an accumulator.

        This used to integrate the per-sample correction and rely on the clamp to
        bound it, which meant any sustained contact force saturated it. Field log 3
        runs 2 and 5: the correction pinned at the clamp three samples after first
        chamfer touch, at only 1.7 N, and the plug stopped descending while axial
        force climbed to 7 N. Shrinking the clamp in f07d3a1 moved that jam from
        37 mm to 2 mm rather than removing it, because the leak only ran on the
        out-of-contact branch and so never fired under load.

        A low-pass of a clamped proportional target settles at the target (~0.12 mm
        at 4 N lateral) instead of the clamp, and decays to zero on its own once the
        contact goes away -- in or out of contact.
        """
        f_plug, m_plug = self._wrench_plug_frame(observation)
        contact = bool(np.isfinite(force) and force >= self.config.contact_force_n)
        finite_force = bool(np.all(np.isfinite(f_plug[:2])))
        finite_moment = bool(np.all(np.isfinite(m_plug[:2])))
        if contact and finite_force:
            log_force_gain = (
                self.config.seat_align_force_gain
                if self.config.seat_align_force_gain != 0.0
                else SEAT_ALIGN_OBSERVE_FORCE_GAIN
            )
            log_moment_gain = (
                self.config.seat_align_moment_gain
                if self.config.seat_align_moment_gain != 0.0
                else SEAT_ALIGN_OBSERVE_MOMENT_GAIN
            )
            d_lat_would = -log_force_gain * f_plug[:2]
            target_lat = clamp_vector_norm(
                -self.config.seat_align_force_gain * f_plug[:2],
                self.config.seat_align_max_lat_m,
            )
            if finite_moment:
                d_tilt_would = -log_moment_gain * m_plug[:2]
                target_tilt = clamp_vector_norm(
                    -self.config.seat_align_moment_gain * m_plug[:2],
                    self.config.seat_align_max_tilt_rad,
                )
            else:
                d_tilt_would = np.zeros(2, dtype=np.float64)
                target_tilt = np.zeros(2, dtype=np.float64)
        else:
            d_lat_would = np.zeros(2, dtype=np.float64)
            d_tilt_would = np.zeros(2, dtype=np.float64)
            target_lat = np.zeros(2, dtype=np.float64)
            target_tilt = np.zeros(2, dtype=np.float64)
        decay = self.config.seat_align_release_decay
        prev_lat = np.asarray(acc_lat, dtype=np.float64).reshape(2)
        prev_tilt = np.asarray(acc_tilt, dtype=np.float64).reshape(2)
        acc_lat = clamp_vector_norm(
            decay * prev_lat + (1.0 - decay) * target_lat,
            self.config.seat_align_max_lat_m,
        )
        acc_tilt = clamp_vector_norm(
            decay * prev_tilt + (1.0 - decay) * target_tilt,
            self.config.seat_align_max_tilt_rad,
        )
        # Slew limit on top of the low pass: the gains are large enough that an
        # extreme wrench would otherwise step a fifth of a millimetre in one
        # sample, which is the lurch the small-cap era was protecting against.
        acc_lat = prev_lat + clamp_vector_norm(
            acc_lat - prev_lat, self.config.seat_align_max_step_m
        )
        acc_tilt = prev_tilt + clamp_vector_norm(
            acc_tilt - prev_tilt, self.config.seat_align_max_tilt_step_rad
        )
        return acc_lat, acc_tilt, (depth, f_plug, m_plug, d_lat_would, d_tilt_would)

    def _log_seat_wrench(self, sample, acc_lat, acc_tilt, *, summary: Optional[str] = None):
        depth, f_plug, m_plug, d_lat_would, d_tilt_would = sample
        suffix = f" summary={summary}" if summary else ""
        self.log.info(
            f"[sfp] SEAT_WRENCH depth={depth*1000.0:.1f}mm "
            f"axial_N={f_plug[2]:.2f} "
            f"lat_N={np.round(f_plug[:2], 2).tolist()} "
            f"|lat|={np.linalg.norm(f_plug[:2]):.2f} "
            f"moment_Nm={np.round(m_plug[:2], 3).tolist()} "
            f"|M|={np.linalg.norm(m_plug[:2]):.3f} "
            f"nudge_would_mm={np.round(d_lat_would * 1000.0, 3).tolist()} "
            f"tilt_would_deg={np.degrees(np.linalg.norm(d_tilt_would)):.3f} "
            f"nudge_applied_mm={np.round(acc_lat * 1000.0, 3).tolist()} "
            f"tilt_applied_deg={np.degrees(np.linalg.norm(acc_tilt)):.3f}"
            f"{suffix}"
        )

    def _seat_target_pose(self, depth, acc_lat, acc_tilt):
        """Tip pose at ``depth`` along the port axis, plus the seat corrections."""
        target_tip = self.port_pos + self.Rp[:, 2] * depth
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
        return target_tip, target_rotation

    def _combined_seat_correction(
        self, acc_lat, acc_tilt, recovery_lat, recovery_tilt
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine ordinary wrench following with a verified local recovery.

        The force follower is intentionally transient; a recovery correction
        must persist through the subsequent slow seat and final event dwell.
        Clamping the sum, rather than each component independently, guarantees
        that a recovery never exceeds the same port-clearance envelope.
        """

        return (
            clamp_vector_norm(
                np.asarray(acc_lat, dtype=np.float64).reshape(2)
                + np.asarray(recovery_lat, dtype=np.float64).reshape(2),
                self.config.seat_align_max_lat_m,
            ),
            clamp_vector_norm(
                np.asarray(acc_tilt, dtype=np.float64).reshape(2)
                + np.asarray(recovery_tilt, dtype=np.float64).reshape(2),
                self.config.seat_align_max_tilt_rad,
            ),
        )

    def _hold_tip(self, tip_pos, tip_rotation) -> None:
        self.policy.set_pose_target(
            self.move_robot,
            self.policy._tcp_target_for_tip(tip_pos, tip_rotation),
            stiffness=self.HOLD_STIFFNESS,
            damping=self.HOLD_DAMPING,
        )

    def _wait_new_observation(self, *, after_stamp, timeout_wall_s):
        deadline = time.monotonic() + timeout_wall_s
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            observation = self.get_observation()
            if observation is not None:
                stamp = _observation_stamp_s(observation)
                if stamp is not None and (after_stamp is None or stamp > after_stamp + 1e-9):
                    self.last_observation_stamp = stamp
                    return observation
            time.sleep(0.02)
        return None

    def _plug_views(self, observation):
        return _plug_views_from_observation(self.policy, observation)

    def _activate_plug_pose(self, observation) -> bool:
        views = self._plug_views(observation)
        if len(views) < 2:
            self.log.error("[sfp] fewer than two fresh plug camera views; fail closed")
            return False
        from .sfp_plug_pose import stamp_to_seconds

        # Image age must be measured in the same ROS/simulation clock domain as
        # the image headers, never against time.monotonic().
        now_s = stamp_to_seconds(self.policy._parent_node.get_clock().now())
        estimate = self.policy._sfp_plug_estimator.estimate_relative_to_port(
            views,
            self.port_pos,
            self.Rp,
            now_s=now_s,
            max_age_s=self.config.plug_max_age_s,
            min_stamp_s=self.last_accepted_plug_stamp,
        )
        if estimate is None:
            self.log.error(
                "[sfp] fresh plug pose unavailable; refusing fixed-bias fallback"
            )
            return False
        tip_pos = self.port_pos + self.Rp @ np.asarray(
            estimate.translation_port, dtype=np.float64
        ).reshape(3)
        tip_rotation = self.Rp @ np.asarray(
            estimate.rotation_port_from_plug, dtype=np.float64
        ).reshape(3, 3)
        tcp_pos, tcp_quat = self.policy._tcp()
        self.policy._sfp_grasp_transform = solve_tip_in_tcp(
            tcp_pos, tcp_quat, tip_pos, tip_rotation
        )
        self.last_accepted_plug_stamp = float(estimate.stamp_s)
        self.log.info(
            "[sfp] fresh plug pose accepted: "
            f"delta_port_mm={np.round(estimate.translation_port * 1000.0, 2).tolist()} "
            f"confidence={estimate.confidence:.3f} views={estimate.view_count} "
            f"reproj={estimate.reprojection_error_px:.2f}px "
            f"frames={list(estimate.source_frame_ids)}"
        )
        return True

    def _activate_initial_plug_pose(self) -> bool:
        estimate = getattr(self.policy, "_sfp_pending_world_plug", None)
        if estimate is None or getattr(self.policy, "_sfp_grasp_transform", None) is None:
            self.log.error("[sfp] no fresh primed plug pose for initial control")
            return False
        relative = self.Rp.T @ (
            np.asarray(estimate.position_world, dtype=np.float64) - self.port_pos
        )
        self.last_accepted_plug_stamp = float(estimate.stamp_s)
        self.policy._sfp_pending_world_plug = None
        self.log.info(
            f"[sfp] initial plug-to-port delta_mm="
            f"{np.round(relative * 1000.0, 2).tolist()} from direct plug pose"
        )
        return True

    def _tip_pose(self):
        tcp_pos, tcp_quat = self.policy._tcp()
        return self.policy._tip_from_tcp(tcp_pos, tcp_quat)

    def _errors(self):
        tip_pos, tip_rotation = self._tip_pose()
        delta = self.Rp.T @ (tip_pos - self.port_pos)
        rotation_error = axis_angle(self.Rp.T @ tip_rotation)
        return (
            float(delta[2]),
            delta[:2],
            rotation_error,
            tip_pos,
            tip_rotation,
        )

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
                    f"[sfp] plug-relative alignment valid: lateral={lateral*1000:.2f}mm "
                    f"rotation={np.degrees(rotation):.2f}deg depth={depth*1000:.1f}mm"
                )
                return True
            lateral_step = -lateral_xy
            lateral_norm = float(np.linalg.norm(lateral_step))
            if lateral_norm > self.config.align_max_lateral_step_m:
                lateral_step *= self.config.align_max_lateral_step_m / lateral_norm
            target_tip = (
                tip_pos
                + self.Rp[:, 0] * lateral_step[0]
                + self.Rp[:, 1] * lateral_step[1]
                + self.Rp[:, 2] * float(np.clip(align_depth - depth, -0.002, 0.002))
            )
            if rotation > self.config.align_max_rotation_step_rad:
                remaining = 1.0 - self.config.align_max_rotation_step_rad / rotation
                target_rotation = self.Rp @ rotation_from_axis_angle(
                    remaining * rotation_error
                )
            else:
                target_rotation = self.Rp
            self.policy.set_pose_target(
                self.move_robot,
                self.policy._tcp_target_for_tip(target_tip, target_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.error("[sfp] plug-relative alignment exceeded wall-time budget")
        return False

    def _wait_for_insertion_event(self, fixed_tip, fixed_rotation=None) -> str:
        """Hold the plug against the back pad until the sim publishes the event.

        The detection pad's near face sits about 1 mm PAST INSERT_DEPTH_M, and
        the TouchPlugin needs a full second of unbroken contact. Holding the tip
        at exactly INSERT_DEPTH_M therefore commands a setpoint behind the plug
        once it reaches the pad, so the impedance loop pulls it off the very
        surface it has to press. The caller now hands in the overtravel depth
        and the seat corrections that got the plug in, so the press stays
        positive for the whole dwell.
        """
        rotation = self.Rp if fixed_rotation is None else fixed_rotation
        deadline = time.monotonic() + self.config.insertion_event_timeout_wall_s
        hard_force_since = None
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            event_status = self._event_status()
            if event_status is not None:
                return event_status
            observation = self.get_observation()
            force = self._force_magnitude(observation)
            tip_pos, _ = self._tip_pose()
            if np.isfinite(force) and force > self.config.force_abort_n:
                self._hold_tip(tip_pos, rotation)
                hard_force_since = hard_force_since or time.monotonic()
                if time.monotonic() - hard_force_since >= self.config.force_abort_wall_s:
                    self.log.error(
                        f"[sfp] >{self.config.force_abort_n:.1f}N during event dwell"
                    )
                    return HARD_FAILURE
            else:
                hard_force_since = None
                self.policy.set_pose_target(
                    self.move_robot,
                    self.policy._tcp_target_for_tip(fixed_tip, rotation),
                    stiffness=self.STIFFNESS,
                    damping=self.DAMPING,
                )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.warn("[sfp] seated geometry produced no matching insertion event")
        return STALLED

    def _unload_deep_wedge(self, start_depth: float):
        """Back out only enough to remove side load before a local probe."""

        target_depth = max(0.0, float(start_depth) - self.config.wedge_recovery_unload_m)
        deadline = time.monotonic() + self.config.wedge_recovery_unload_timeout_wall_s
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            event_status = self._event_status()
            if event_status is not None:
                return event_status, None
            observation = self.get_observation()
            depth, _, _, tip_pos, tip_rotation = self._errors()
            force = self._force_magnitude(observation)
            if np.isfinite(force) and force > self.config.force_abort_n:
                self._hold_tip(tip_pos, tip_rotation)
                self.log.error(
                    f"[sfp] >{self.config.force_abort_n:.1f}N while unloading a deep wedge"
                )
                return HARD_FAILURE, None
            if float(start_depth) - depth >= self.config.wedge_recovery_min_unload_m:
                return _DEEP_RECOVERY_ACCEPTED, (depth, tip_pos, tip_rotation)
            # Keep the observed lateral position and orientation while backing
            # out; correcting a cocked plug under side load can cam it harder
            # into the cage wall.
            target_tip = tip_pos + self.Rp[:, 2] * (target_depth - depth)
            self.policy.set_pose_target(
                self.move_robot,
                self.policy._tcp_target_for_tip(target_tip, tip_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)
        self.log.warn(
            f"[sfp] deep wedge could not unload {self.config.wedge_recovery_min_unload_m*1000.0:.1f}mm"
        )
        return _DEEP_RECOVERY_FAILED, None

    def _attempt_force_guided_deep_recovery(
        self,
        *,
        depth: float,
        force: float,
        force_port,
        moment_port,
        current_lat,
        current_tilt,
    ):
        """Test tiny signed wrench-guided probes before a full retrace.

        A force/torque sensor provides a resultant wrench rather than a unique
        contact point.  We therefore use it to seed a primary and mirrored
        correction, then accept only the correction that both gets past the
        prior stall and lowers the measured transverse wrench.
        """

        if (
            not self.config.wedge_recovery_enable
            or depth < self.config.wedge_recovery_min_depth_m
        ):
            return _DEEP_RECOVERY_SKIPPED, None, None, depth
        if np.isfinite(force) and force >= self.config.seat_force_cap_n:
            self.log.warn(
                f"[sfp] deep wedge force={force:.2f}N is at the seat cap; "
                "skipping local probe"
            )
            return _DEEP_RECOVERY_SKIPPED, None, None, depth

        probes = deep_wedge_recovery_probes(force_port, moment_port, self.config)
        baseline_score = transverse_wrench_score(force_port, moment_port, self.config)
        if not probes or not np.isfinite(baseline_score):
            return _DEEP_RECOVERY_SKIPPED, None, None, depth

        force_port = np.asarray(force_port, dtype=np.float64).reshape(3)
        moment_port = np.asarray(moment_port, dtype=np.float64).reshape(3)
        moment_text = (
            f"{np.round(moment_port[:2], 3).tolist()}"
            if np.all(np.isfinite(moment_port[:2]))
            else "unavailable"
        )
        self.log.warn(
            "[sfp] deep wedge force-guided recovery: "
            f"depth={depth*1000.0:.1f}mm lat_N={np.round(force_port[:2], 2).tolist()} "
            f"moment_Nm={moment_text} score={baseline_score:.2f}"
        )

        base_lat = clamp_vector_norm(
            np.asarray(current_lat, dtype=np.float64).reshape(2),
            self.config.wedge_recovery_max_lat_m,
        )
        base_tilt = clamp_vector_norm(
            np.asarray(current_tilt, dtype=np.float64).reshape(2),
            self.config.wedge_recovery_max_tilt_rad,
        )
        for probe_index, (delta_lat, delta_tilt) in enumerate(probes, start=1):
            unload_status, unloaded = self._unload_deep_wedge(depth)
            if unload_status in (SEATED, HARD_FAILURE):
                return unload_status, None, None, depth
            if unload_status != _DEEP_RECOVERY_ACCEPTED or unloaded is None:
                return _DEEP_RECOVERY_FAILED, None, None, depth
            unloaded_depth, _, _ = unloaded
            recovery_lat = clamp_vector_norm(
                base_lat + np.asarray(delta_lat, dtype=np.float64).reshape(2),
                self.config.wedge_recovery_max_lat_m,
            )
            recovery_tilt = clamp_vector_norm(
                base_tilt + np.asarray(delta_tilt, dtype=np.float64).reshape(2),
                self.config.wedge_recovery_max_tilt_rad,
            )
            total_lat, total_tilt = self._combined_seat_correction(
                np.zeros(2, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                recovery_lat,
                recovery_tilt,
            )

            # Let the tiny XY/tilt adjustment settle at the unloaded depth
            # before asking the plug to move forward again.  Returning to the
            # pre-probe correction first means a mirrored trial never jumps
            # directly across both sides of the bore.
            transition_corrections = (
                (base_lat, base_tilt),
                (recovery_lat, recovery_tilt),
                (recovery_lat, recovery_tilt),
            )
            for transition_lat, transition_tilt in transition_corrections:
                self.policy._enforce_action_deadline(self.move_robot)
                event_status = self._event_status()
                if event_status is not None:
                    return event_status, None, None, unloaded_depth
                current_observation = self.get_observation()
                current_depth, _, _, current_tip, _ = self._errors()
                current_force = self._force_magnitude(current_observation)
                if np.isfinite(current_force) and current_force > self.config.force_abort_n:
                    self._hold_tip(current_tip, self.Rp)
                    return HARD_FAILURE, None, None, current_depth
                phase_lat, phase_tilt = self._combined_seat_correction(
                    np.zeros(2, dtype=np.float64),
                    np.zeros(2, dtype=np.float64),
                    transition_lat,
                    transition_tilt,
                )
                target_tip, target_rotation = self._seat_target_pose(
                    unloaded_depth, phase_lat, phase_tilt
                )
                self.policy.set_pose_target(
                    self.move_robot,
                    self.policy._tcp_target_for_tip(target_tip, target_rotation),
                    stiffness=self.STIFFNESS,
                    damping=self.DAMPING,
                )
                self.policy.sleep_for(self.config.command_dt_sim_s)

            probe_goal = min(
                INSERT_DEPTH_M + self.config.seat_overtravel_m,
                float(depth) + self.config.wedge_recovery_probe_depth_m,
            )
            command_depth = float(unloaded_depth)
            last_command_time = time.monotonic()
            deadline = time.monotonic() + self.config.wedge_recovery_probe_timeout_wall_s
            improved_samples = 0
            while time.monotonic() < deadline:
                self.policy._enforce_action_deadline(self.move_robot)
                event_status = self._event_status()
                if event_status is not None:
                    return event_status, None, None, command_depth
                observation = self.get_observation()
                current_depth, lateral_xy, rotation_error, tip_pos, _ = self._errors()
                current_force = self._force_magnitude(observation)
                if np.isfinite(current_force) and current_force > self.config.force_abort_n:
                    self._hold_tip(tip_pos, self.Rp)
                    return HARD_FAILURE, None, None, current_depth
                if (
                    np.isfinite(current_force)
                    and current_force >= self.config.seat_force_cap_n
                ):
                    break
                if (
                    float(np.linalg.norm(lateral_xy)) > self.config.lateral_safety_m
                    or float(np.linalg.norm(rotation_error)) > self.config.rotation_safety_rad
                ):
                    break

                current_force_port, current_moment_port = self._wrench_plug_frame(
                    observation
                )
                current_score = transverse_wrench_score(
                    current_force_port, current_moment_port, self.config
                )
                if (
                    current_depth >= depth + self.config.wedge_recovery_min_progress_m
                    and np.isfinite(current_score)
                    and current_score
                    <= baseline_score
                    * (1.0 - self.config.wedge_recovery_score_improvement_fraction)
                ):
                    improved_samples += 1
                    if improved_samples >= 2:
                        self.log.info(
                            "[sfp] deep wedge probe accepted: "
                            f"probe={probe_index}/{len(probes)} "
                            f"depth_gain_mm={(current_depth-depth)*1000.0:.2f} "
                            f"score={current_score:.2f}/{baseline_score:.2f} "
                            f"lat_mm={np.round(recovery_lat*1000.0, 3).tolist()} "
                            f"tilt_deg={np.degrees(np.linalg.norm(recovery_tilt)):.3f}"
                        )
                        return (
                            _DEEP_RECOVERY_ACCEPTED,
                            recovery_lat,
                            recovery_tilt,
                            current_depth,
                        )
                else:
                    improved_samples = 0

                now = time.monotonic()
                elapsed = max(self.config.command_dt_sim_s, now - last_command_time)
                command_depth = min(
                    probe_goal,
                    command_depth + self.config.wedge_recovery_probe_speed_m_s * elapsed,
                )
                last_command_time = now
                target_tip, target_rotation = self._seat_target_pose(
                    command_depth, total_lat, total_tilt
                )
                self.policy.set_pose_target(
                    self.move_robot,
                    self.policy._tcp_target_for_tip(target_tip, target_rotation),
                    stiffness=self.STIFFNESS,
                    damping=self.DAMPING,
                )
                self.policy.sleep_for(self.config.command_dt_sim_s)

            self.log.warn(
                "[sfp] deep wedge probe did not improve; "
                f"probe={probe_index}/{len(probes)}"
            )
        return _DEEP_RECOVERY_FAILED, None, None, depth

    def _seat(self) -> str:
        depth, _, _, _, _ = self._errors()
        command_depth = depth
        now = time.monotonic()
        last_command_time = now
        progress = WallProgressWatch.start(depth, now, self.config)
        hard_force_since = None
        acc_lat = np.zeros(2, dtype=np.float64)
        acc_tilt = np.zeros(2, dtype=np.float64)
        recovery_lat = np.zeros(2, dtype=np.float64)
        recovery_tilt = np.zeros(2, dtype=np.float64)
        local_recovery_attempts = 0
        last_wrench_log_time = 0.0
        slope_samples = []
        last_slope_log_time = 0.0
        stall_grace_deadline = None

        def log_seat_slope(*, summary: Optional[str] = None):
            if not slope_samples:
                return
            _, first_depth, first_force = slope_samples[0]
            _, last_depth, last_force = slope_samples[-1]
            d_depth_mm = (last_depth - first_depth) * 1000.0
            d_force = last_force - first_force
            if np.isfinite(d_depth_mm) and abs(d_depth_mm) > 1e-6:
                d_force_per_mm = d_force / d_depth_mm
            elif np.isfinite(d_force) and abs(d_force) > 1e-9:
                d_force_per_mm = float(np.copysign(np.inf, d_force))
            else:
                d_force_per_mm = 0.0
            suffix = f" summary={summary}" if summary else ""
            self.log.info(
                f"[sfp] SEAT_SLOPE depth={last_depth*1000.0:.1f}mm "
                f"axial_N={last_force:.2f} dDepth_mm={d_depth_mm:.3f} "
                f"dForce_N={d_force:.2f} dForce_per_mm={d_force_per_mm:.2f}"
                f"{suffix}"
            )

        def try_deep_recovery(depth, force, force_port, moment_port):
            """Return True when seating can continue, else an outcome or False."""

            nonlocal acc_lat
            nonlocal acc_tilt
            nonlocal recovery_lat
            nonlocal recovery_tilt
            nonlocal local_recovery_attempts
            nonlocal progress
            nonlocal command_depth
            nonlocal last_command_time
            nonlocal stall_grace_deadline
            if (
                local_recovery_attempts >= self.config.wedge_recovery_max_local_attempts
                or depth < self.config.wedge_recovery_min_depth_m
                or not self.config.wedge_recovery_enable
            ):
                return False
            current_lat, current_tilt = self._combined_seat_correction(
                acc_lat, acc_tilt, recovery_lat, recovery_tilt
            )
            recovery_status, next_lat, next_tilt, recovered_depth = (
                self._attempt_force_guided_deep_recovery(
                    depth=depth,
                    force=force,
                    force_port=force_port,
                    moment_port=moment_port,
                    current_lat=current_lat,
                    current_tilt=current_tilt,
                )
            )
            if recovery_status == _DEEP_RECOVERY_SKIPPED:
                return False
            local_recovery_attempts += 1
            if recovery_status in (SEATED, HARD_FAILURE):
                return recovery_status
            if recovery_status != _DEEP_RECOVERY_ACCEPTED:
                return False
            # The verified correction becomes persistent; reset the ordinary
            # low-pass follower so the combined target cannot double-count it.
            recovery_lat = np.asarray(next_lat, dtype=np.float64).reshape(2)
            recovery_tilt = np.asarray(next_tilt, dtype=np.float64).reshape(2)
            acc_lat = np.zeros(2, dtype=np.float64)
            acc_tilt = np.zeros(2, dtype=np.float64)
            recovered_now = time.monotonic()
            progress = WallProgressWatch.start(recovered_depth, recovered_now, self.config)
            command_depth = recovered_depth
            last_command_time = recovered_now
            stall_grace_deadline = None
            slope_samples.clear()
            self.send_feedback("deep wedge micro-recovery accepted; resuming seat")
            return True

        while True:
            self.policy._enforce_action_deadline(self.move_robot)
            event_status = self._event_status()
            if event_status is not None:
                return event_status
            observation = self.get_observation()
            depth, lateral_xy, rotation_error, tip_pos, _ = self._errors()
            lateral = float(np.linalg.norm(lateral_xy))
            rotation = float(np.linalg.norm(rotation_error))
            force = self._force_magnitude(observation)
            now = time.monotonic()
            seat_wrench_sample = None
            if self.config.seat_align_enable:
                acc_lat, acc_tilt, seat_wrench_sample = self._seat_alignment_sample(
                    observation, depth, force, acc_lat, acc_tilt
                )
                if now - last_wrench_log_time >= SEAT_WRENCH_LOG_PERIOD_S:
                    self._log_seat_wrench(seat_wrench_sample, acc_lat, acc_tilt)
                    last_wrench_log_time = now
            if seat_wrench_sample is not None:
                force_port = np.asarray(seat_wrench_sample[1], dtype=np.float64)
                moment_port = np.asarray(seat_wrench_sample[2], dtype=np.float64)
                axial_force = float(force_port[2])
            else:
                force_port, moment_port = self._wrench_plug_frame(observation)
                axial_force = float(force_port[2])
            slope_samples.append((now, depth, axial_force))
            while slope_samples and now - slope_samples[0][0] > SEAT_SLOPE_WINDOW_S:
                slope_samples.pop(0)
            if now - last_slope_log_time >= SEAT_SLOPE_LOG_PERIOD_S:
                log_seat_slope()
                last_slope_log_time = now

            if np.isfinite(force) and force > self.config.force_abort_n:
                self._hold_tip(tip_pos, self.Rp)
                hard_force_since = hard_force_since or now
                if now - hard_force_since >= self.config.force_abort_wall_s:
                    self.log.error(
                        f"[sfp] sustained force {force:.1f}N exceeds "
                        f"{self.config.force_abort_n:.1f}N; held and aborted"
                    )
                    return HARD_FAILURE
                self.policy.sleep_for(self.config.command_dt_sim_s)
                continue
            hard_force_since = None

            if lateral > self.config.lateral_safety_m or rotation > self.config.rotation_safety_rad:
                self.log.warn(
                    f"[sfp] wedge geometry: lateral={lateral*1000:.1f}mm "
                    f"rotation={np.degrees(rotation):.1f}deg"
                )
                if seat_wrench_sample is not None:
                    self._log_seat_wrench(
                        seat_wrench_sample, acc_lat, acc_tilt, summary="stall"
                    )
                log_seat_slope(summary="stall")
                return WEDGED

            if depth >= self.config.seat_candidate_depth_m:
                total_lat, total_tilt = self._combined_seat_correction(
                    acc_lat, acc_tilt, recovery_lat, recovery_tilt
                )
                fixed_tip, fixed_rotation = self._seat_target_pose(
                    INSERT_DEPTH_M + self.config.seat_overtravel_m,
                    total_lat,
                    total_tilt,
                )
                return self._wait_for_insertion_event(fixed_tip, fixed_rotation)

            stalled_now = progress.stalled(depth, now)
            if stall_grace_deadline is not None and not stalled_now:
                self.log.info(
                    f"[sfp] stall grace recovered: depth={depth*1000:.1f}mm "
                    f"best={progress.best_depth*1000:.1f}mm"
                )
                stall_grace_deadline = None
            if stalled_now:
                if self.config.seat_stall_grace_s > 0.0:
                    if stall_grace_deadline is None:
                        stall_grace_deadline = now + self.config.seat_stall_grace_s
                        self.log.warn(
                            f"[sfp] wall-time stall grace: depth={depth*1000:.1f}mm "
                            f"best={progress.best_depth*1000:.1f}mm force={force:.2f}N "
                            f"grace_s={self.config.seat_stall_grace_s:.1f}"
                        )
                    if now >= stall_grace_deadline:
                        self.log.warn(
                            f"[sfp] wall-time stall: depth={depth*1000:.1f}mm "
                            f"best={progress.best_depth*1000:.1f}mm force={force:.2f}N"
                        )
                        if seat_wrench_sample is not None:
                            self._log_seat_wrench(
                                seat_wrench_sample, acc_lat, acc_tilt, summary="stall"
                            )
                        log_seat_slope(summary="stall")
                        recovery_result = try_deep_recovery(
                            depth, force, force_port, moment_port
                        )
                        if recovery_result is True:
                            continue
                        if recovery_result in (SEATED, HARD_FAILURE):
                            return recovery_result
                        return self._wall_stall_outcome()
                else:
                    self.log.warn(
                        f"[sfp] wall-time stall: depth={depth*1000:.1f}mm "
                        f"best={progress.best_depth*1000:.1f}mm force={force:.2f}N"
                    )
                    if seat_wrench_sample is not None:
                        self._log_seat_wrench(
                            seat_wrench_sample, acc_lat, acc_tilt, summary="stall"
                        )
                    log_seat_slope(summary="stall")
                    recovery_result = try_deep_recovery(
                        depth, force, force_port, moment_port
                    )
                    if recovery_result is True:
                        continue
                    if recovery_result in (SEATED, HARD_FAILURE):
                        return recovery_result
                    return self._wall_stall_outcome()

            depth_config = self.config
            if (
                self.config.seat_mouth_speed_scale != 1.0
                and depth < self.config.seat_mouth_zone_m
            ):
                depth_config = replace(
                    self.config,
                    free_speed_m_s=(
                        self.config.free_speed_m_s
                        * self.config.seat_mouth_speed_scale
                    ),
                    contact_speed_m_s=(
                        self.config.contact_speed_m_s
                        * self.config.seat_mouth_speed_scale
                    ),
                )
            command_depth = next_persistent_depth(
                depth,
                command_depth,
                now - last_command_time,
                force,
                depth_config,
                axial_force,
            )
            last_command_time = now
            total_lat, total_tilt = self._combined_seat_correction(
                acc_lat, acc_tilt, recovery_lat, recovery_tilt
            )
            target_tip, target_rotation = self._seat_target_pose(
                command_depth, total_lat, total_tilt
            )
            self.policy.set_pose_target(
                self.move_robot,
                self.policy._tcp_target_for_tip(target_tip, target_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)

    def _wall_stall_outcome(self) -> str:
        """A plug that stopped advancing inside the bore is a jam, not a mystery.

        Diag-5's three ~36 mm stalls all carried 4.3-4.8 N of lateral bind, so
        they are treated as wedges and earn a retry. Set
        RL_INSERT_V50_WEDGE_RETRY_ON_WALL_STALL=0 to restrict retries to the
        gross-excursion wedge check alone.
        """
        return WEDGED if self.config.wedge_retry_on_wall_stall else STALLED

    def _attempt_wedge_rescue(self) -> bool:
        """Let the visual-gap rescue re-aim at the physical opening, if present.

        The rescue lives on InsertionPolicy as a mixin. A missing or disabled
        rescue is not an error; it means retract-and-retry is the only option.
        Returns True when the port estimate was corrected and seating is worth
        re-running without backing out.
        """
        enabled = getattr(self.policy, "_visual_gap_wedge_enabled", None)
        rescue = getattr(self.policy, "_run_visual_gap_wedge_recovery", None)
        if rescue is None or enabled is None or not enabled():
            self.log.info("[sfp] no wedge rescue available; retract and retry")
            return False
        try:
            hole_point = rescue(
                self.get_observation,
                self.move_robot,
                # Measured against the ORIGINAL perception, so the rescue's own
                # excursion cap bounds total drift no matter how many retries
                # run; passing the already-corrected estimate would let the port
                # walk that cap again on every attempt.
                raw_port_pos=self._port_pos_initial,
                Rp=self.Rp,
                R_seat=self.Rp,
                local_port_kps=LOCAL_SFP_PORT_KPS,
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
                step_dt=self.config.command_dt_sim_s,
            )
        except Exception as exc:
            self.log.warn(
                f"[sfp] wedge rescue raised {type(exc).__name__}: {exc}; "
                "retract and retry"
            )
            return False
        if hole_point is None:
            return False
        shift = self.Rp.T @ (
            np.asarray(hole_point, dtype=np.float64).reshape(3) - self.port_pos
        )
        self.port_pos = np.asarray(hole_point, dtype=np.float64).reshape(3)
        self.log.warn(
            "[sfp] wedge rescue corrected the port estimate by "
            f"{np.round(shift[:2] * 1000.0, 2).tolist()}mm; re-seating without retract"
        )
        return True

    def _retract_to_start(self, start_tip_pos, start_tip_rotation) -> bool:
        """Back the plug out of the bore and return it near its starting pose.

        The seat setpoint cannot walk backwards: next_persistent_depth() clamps
        the command to at least the current depth, so asking for less depth is
        silently ignored. Retraction therefore runs next_retract_depth(), its
        mirror image: a persistent setpoint receding from the measured depth so
        the pull force BUILDS on a stuck plug instead of plateauing at
        stiffness * retract_step_m.  Lateral is re-anchored on the measured tip
        every tick, so the only sustained demand is the axial pull -- the plug
        finds its own way off the chamfer instead of being dragged against it.

        Withdrawal is two phases because orientation is the dangerous part. While
        the plug is still inside the cage its rotation is held at whatever was
        measured -- correcting a cocked plug in place cams it harder against the
        walls -- and only once it is clear of the mouth does it move back to the
        starting pose and orientation.
        """
        deadline = time.monotonic() + self.config.retract_timeout_wall_s
        cleared = False
        command_depth = None
        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            depth, _, _, tip_pos, tip_rotation = self._errors()
            if depth <= self.config.retract_clear_depth_m:
                cleared = True
                break
            if command_depth is None:
                command_depth = depth
            command_depth = next_retract_depth(depth, command_depth, self.config)
            self.policy.set_pose_target(
                self.move_robot,
                self.policy._tcp_target_for_tip(
                    tip_pos + self.Rp[:, 2] * (command_depth - depth), tip_rotation
                ),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)

        if not cleared:
            depth, _, _, _, _ = self._errors()
            self.log.error(
                f"[sfp] retract could not clear the port mouth: depth={depth*1000:.1f}mm "
                f"after {self.config.retract_timeout_wall_s:.1f}s"
            )
            return False

        while time.monotonic() < deadline:
            self.policy._enforce_action_deadline(self.move_robot)
            _, _, _, tip_pos, _ = self._errors()
            delta = np.asarray(start_tip_pos, dtype=np.float64).reshape(3) - tip_pos
            distance = float(np.linalg.norm(delta))
            if distance <= self.config.retract_arrive_tol_m:
                self.log.warn("[sfp] retracted to the starting pose; retrying insertion")
                return True
            step = delta * min(1.0, self.config.retract_free_step_m / max(distance, 1e-9))
            self.policy.set_pose_target(
                self.move_robot,
                self.policy._tcp_target_for_tip(tip_pos + step, start_tip_rotation),
                stiffness=self.STIFFNESS,
                damping=self.DAMPING,
            )
            self.policy.sleep_for(self.config.command_dt_sim_s)

        # Clear of the bore but short of the start pose is still a usable restart:
        # alignment re-converges from wherever the plug ended up.
        _, _, _, tip_pos, _ = self._errors()
        remaining = float(
            np.linalg.norm(np.asarray(start_tip_pos, dtype=np.float64).reshape(3) - tip_pos)
        )
        self.log.warn(
            f"[sfp] retract cleared the mouth but stopped {remaining*1000:.1f}mm "
            "from the starting pose; retrying from here"
        )
        return True

    def _refresh_plug_pose_after_retract(self) -> bool:
        """Re-solve the grasp transform now that the plug is clear of the port.

        This is the one viewpoint where re-perception is trustworthy: at the
        aligned standoff the cable drapes over the port, but a retracted plug is
        back in the pose priming already worked from.  Retry motion is forbidden
        from using the previous transform, so clear it before requesting a new
        observation and fail closed if that observation cannot establish one.
        """
        self.policy._sfp_grasp_transform = None
        observation = self._wait_new_observation(
            after_stamp=self.last_observation_stamp,
            timeout_wall_s=2.0,
        )
        if observation is None:
            self.log.error(
                "[sfp] no fresh plug observation after retract; retry aborted"
            )
            return False
        if not self._activate_plug_pose(observation):
            self.log.error(
                "[sfp] fresh plug pose after retract failed; retry aborted"
            )
            return False
        return True

    def _refresh_port_pose_after_retract(self) -> bool:
        """Replace the SFP port target using camera frames newer than the plug pose."""

        perceive = getattr(self.policy, "perceive_port_pose_consensus", None)
        if perceive is None:
            self.log.error("[sfp] SFP port refresh hook unavailable; retry aborted")
            return False

        def get_fresh_observation():
            return self._wait_new_observation(
                after_stamp=self.last_observation_stamp,
                timeout_wall_s=2.0,
            )

        perceived = perceive(self.task, get_fresh_observation)
        if perceived is None:
            self.log.error("[sfp] fresh SFP port pose after retract failed; retry aborted")
            return False
        try:
            new_pos, new_quat, reproj_px = perceived
            new_pos = np.asarray(new_pos, dtype=np.float64).reshape(3)
            new_quat = np.asarray(new_quat, dtype=np.float64).reshape(4)
            reproj_px = float(reproj_px)
            quat_norm = float(np.linalg.norm(new_quat))
            if (
                not np.all(np.isfinite(new_pos))
                or not np.all(np.isfinite(new_quat))
                or not np.isfinite(reproj_px)
                or quat_norm <= 1e-12
                or reproj_px > MAX_PORT_REPROJ_PX
            ):
                raise ValueError("non-finite or out-of-gate port pose")
            new_quat = new_quat / quat_norm
            from .contract import port_frame

            new_Rp = port_frame(new_quat)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            self.log.error(f"[sfp] invalid refreshed SFP port pose ({exc}); retry aborted")
            return False

        # Atomic replacement: no retry stage can observe a mixture of old and
        # new port position/orientation.  The visual rescue reference follows
        # this fresh target as well, so no later retry consults the old pose.
        self.port_pos = new_pos
        self.port_quat = new_quat
        self.Rp = new_Rp
        self._port_pos_initial = new_pos.copy()
        self.log.info(
            "[sfp] fresh port pose accepted: "
            f"p={np.round(new_pos, 5).tolist()} "
            f"q_wxyz={np.round(new_quat, 5).tolist()} reproj={reproj_px:.2f}px"
        )
        return True

    def _hold_legacy_safe_pose(self):
        """Hold the TCP itself when no visually calibrated plug transform exists."""

        tcp_pos, tcp_quat = self.policy._tcp()
        from .contract import sfp_tip_pose_from_tcp

        tip_pos, tip_rotation = sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)
        self._hold_tip(tip_pos, tip_rotation)

    def run(self) -> bool:
        self.send_feedback("SFP fresh plug-to-port perception")
        if not self._activate_initial_plug_pose():
            self._hold_legacy_safe_pose()
            return False

        # Where the plug started, so a wedge can be undone rather than accepted.
        start_tip_pos, start_tip_rotation = self._tip_pose()
        self._port_pos_initial = self.port_pos.copy()
        retries = 0
        rescued_since_retract = False

        while True:
            # Retries are bounded by the action deadline alone, so every cycle
            # must touch it even if a stage returns without entering its loop.
            self.policy._enforce_action_deadline(self.move_robot)
            if not self._align():
                return False
            self.send_feedback("SFP persistent force-regulated seating")
            outcome = self._seat()
            if outcome == SEATED:
                self.log.info(
                    f"[sfp] matching insertion event confirmed for {self.expected_event}"
                )
                self.send_feedback("correct-port insertion event confirmed")
                return True
            # A wrong-port event or a sustained over-force is not something another
            # attempt fixes, and a plug at seat depth is not worth backing out of.
            if outcome != WEDGED or not self.config.wedge_retry_enable:
                return False
            if (
                self.config.max_wedge_retries
                and retries >= self.config.max_wedge_retries
            ):
                self.log.error(
                    f"[sfp] wedged after {retries} retries; retry budget exhausted"
                )
                return False

            # Rescue first, retract only if there is no rescue to be had. One
            # rescue per retract keeps a drifting port estimate from being nudged
            # indefinitely without ever backing the plug out.
            if not rescued_since_retract and self._attempt_wedge_rescue():
                rescued_since_retract = True
                continue

            retries += 1
            self.send_feedback(f"wedged; retracting to retry (attempt {retries + 1})")
            self.log.warn(
                f"[sfp] wedged with no rescue; retract-and-retry {retries}"
                + (
                    ""
                    if not self.config.max_wedge_retries
                    else f"/{self.config.max_wedge_retries}"
                )
            )
            if not self._retract_to_start(start_tip_pos, start_tip_rotation):
                return False
            rescued_since_retract = False
            if not self._refresh_plug_pose_after_retract():
                return False
            if not self._refresh_port_pose_after_retract():
                return False


def run_sfp_insertion(
    policy,
    task,
    get_observation,
    move_robot,
    send_feedback,
    *,
    port_pos,
    port_quat,
    Rp,
) -> bool:
    return SfpInsertionController(
        policy,
        task,
        get_observation,
        move_robot,
        send_feedback,
        port_pos=port_pos,
        port_quat=port_quat,
        Rp=Rp,
    ).run()


__all__ = [
    "HARD_FAILURE",
    "INSERT_DEPTH_M",
    "SfpInsertionController",
    "SEATED",
    "STALLED",
    "WEDGED",
    "SfpConfig",
    "WallProgressWatch",
    "configure_sfp_controller",
    "next_persistent_depth",
    "next_retract_depth",
    "prime_sfp_plug_pose",
    "run_sfp_insertion",
    "solve_tip_in_tcp",
    "tcp_for_tip_transform",
    "tip_from_tcp_transform",
    "sfp_tcp_pose_for_tip",
    "sfp_tip_from_tcp",
]
