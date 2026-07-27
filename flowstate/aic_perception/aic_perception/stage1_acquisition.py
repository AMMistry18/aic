"""Deterministic Stage-1 insignia-acquisition policy.

The policy uses one offline-swept, fixed-orientation observation posture.  It
is a one-rung deterministic ladder: observe first, move only when the existing
Stage-2 detector cannot use the current view, then observe once more.  There is
no image-gradient servo, orientation phase machine, or unbounded recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np

from .arm_ik import JOINT_LIMITS


# Generated from the production UR5e/camera model with the same camera,
# gripper-mask, self-clearance, arm-in-camera, and legal-placement gates used
# by Stage 2.  The acquisition sweep proves that at least one calibrated camera
# fully frames INSIGNIA_RECT_CORNERS for every supported board yaw, tilt, and
# placement.
OBSERVATION_JOINTS_RAD = (
    -0.49178142333444225,
    -1.2239295912535437,
    -1.2705298822817923,
    -1.9160573594244148,
    1.626385063470916,
    0.34791393309254964,
)

# These caps accept all three historical live starts in the 144-case sweep,
# while rejecting the 501-degree reconfiguration that poisoned the hardware
# session.  They apply only to Stage 1; Stage 2 remains unchanged.
MAX_WORST_JOINT_TRAVEL_RAD = math.radians(185.0)
MAX_TOTAL_JOINT_TRAVEL_RAD = math.radians(250.0)
MIN_TCP_HEIGHT_M = 0.245
MAX_TCP_REACH_M = 1.20
# Historical home and home+J6 exits are already below the endpoint's 140 mm
# wrist/forearm keep-out.  Recovery may start there, but it may not lose more
# than 10 mm before reaching an endpoint that satisfies the full gate.
MIN_RECOVERY_SELF_CLEARANCE_M = 0.080
MAX_RECOVERY_CLEARANCE_LOSS_M = 0.010


@dataclass(frozen=True)
class JointTravel:
    deltas_rad: tuple[float, ...]
    worst_rad: float
    total_rad: float


@dataclass(frozen=True)
class PathValidation:
    safe: bool
    reason: str
    travel: JointTravel
    min_tcp_height_m: float = math.inf
    max_tcp_reach_m: float = 0.0
    min_board_clearance_m: float = math.inf


def joint_travel(
    start: Sequence[float], target: Sequence[float]
) -> JointTravel:
    """Measured physical travel without modulo-2pi wrapping."""

    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if (
        start_array.shape != (6,)
        or target_array.shape != (6,)
        or not np.all(np.isfinite(start_array))
        or not np.all(np.isfinite(target_array))
    ):
        raise ValueError("start and target must be six finite joints")
    deltas = np.abs(target_array - start_array)
    return JointTravel(
        tuple(float(value) for value in deltas),
        float(deltas.max()),
        float(deltas.sum()),
    )


def interpolated_joint_waypoints(
    start: Sequence[float],
    target: Sequence[float],
    *,
    max_segment_joint_rad: float,
) -> tuple[tuple[float, ...], ...]:
    """Split one fixed joint path into deterministic guarded transactions."""

    if not math.isfinite(max_segment_joint_rad) or max_segment_joint_rad <= 0.0:
        raise ValueError("max_segment_joint_rad must be positive and finite")
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    travel = joint_travel(start_array, target_array)
    segments = max(1, int(math.ceil(travel.worst_rad / max_segment_joint_rad)))
    return tuple(
        tuple(
            float(value)
            for value in start_array
            + (float(index) / float(segments)) * (target_array - start_array)
        )
        for index in range(1, segments + 1)
    )


def validate_observation_path(
    arm,
    start: Sequence[float],
    target: Sequence[float] = OBSERVATION_JOINTS_RAD,
    *,
    board_transforms: Iterable | None = None,
    min_board_clearance_m: float = 0.2393,
    endpoint_arm_clear: Callable[[object, np.ndarray], bool] | None = None,
    samples: int = 33,
    max_worst_joint_travel_rad: float = MAX_WORST_JOINT_TRAVEL_RAD,
    max_total_joint_travel_rad: float = MAX_TOTAL_JOINT_TRAVEL_RAD,
) -> PathValidation:
    """Gate the exact joint interpolation before Stage 1 commands it."""

    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    travel = joint_travel(start_array, target_array)
    if travel.worst_rad > max_worst_joint_travel_rad + 1e-9:
        return PathValidation(False, "worst-joint travel exceeds Stage-1 cap", travel)
    if travel.total_rad > max_total_joint_travel_rad + 1e-9:
        return PathValidation(False, "total joint travel exceeds Stage-1 cap", travel)
    if samples < 2:
        return PathValidation(False, "path sample count must be at least two", travel)
    if np.any(target_array < JOINT_LIMITS[:, 0] - 1e-9) or np.any(
        target_array > JOINT_LIMITS[:, 1] + 1e-9
    ):
        return PathValidation(False, "observation joints exceed physical limits", travel)

    boards = tuple(board_transforms or ())
    start_self_clearance = float(arm.self_clearance(start_array))
    recovery_floor = max(
        MIN_RECOVERY_SELF_CLEARANCE_M,
        min(start_self_clearance, arm.min_self_clearance_m)
        - MAX_RECOVERY_CLEARANCE_LOSS_M,
    )
    min_height = math.inf
    max_reach = 0.0
    min_board_clearance = math.inf
    endpoint_tcp = None
    endpoint_joints = None
    for fraction in np.linspace(0.0, 1.0, int(samples)):
        joints = start_array + float(fraction) * (target_array - start_array)
        if np.any(joints < JOINT_LIMITS[:, 0] - 1e-9) or np.any(
            joints > JOINT_LIMITS[:, 1] + 1e-9
        ):
            return PathValidation(False, "joint path leaves physical limits", travel)
        clearance = float(arm.self_clearance(joints))
        if clearance < recovery_floor:
            return PathValidation(
                False, "joint path worsens wrist/forearm recovery clearance", travel
            )
        tcp = arm.fk(joints)
        position = np.asarray(tcp.translation, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return PathValidation(False, "joint path FK is invalid", travel)
        height = float(position[2])
        reach = float(np.linalg.norm(position))
        min_height = min(min_height, height)
        max_reach = max(max_reach, reach)
        if height < MIN_TCP_HEIGHT_M:
            return PathValidation(
                False,
                "joint path drops below tallest-component clearance",
                travel,
                min_height,
                max_reach,
                min_board_clearance,
            )
        if reach > MAX_TCP_REACH_M:
            return PathValidation(
                False,
                "joint path exceeds Stage-1 reach envelope",
                travel,
                min_height,
                max_reach,
                min_board_clearance,
            )
        for board in boards:
            normal = np.asarray(board.rotation, dtype=float)[:, 2]
            clearance = float(
                np.dot(position - np.asarray(board.translation, dtype=float), normal)
            )
            min_board_clearance = min(min_board_clearance, clearance)
            if clearance < min_board_clearance_m:
                return PathValidation(
                    False,
                    "joint path violates board-normal component clearance",
                    travel,
                    min_height,
                    max_reach,
                    min_board_clearance,
                )
        endpoint_tcp = tcp
        endpoint_joints = joints

    if float(arm.self_clearance(target_array)) < arm.min_self_clearance_m:
        return PathValidation(
            False,
            "observation endpoint violates wrist/forearm self-clearance",
            travel,
            min_height,
            max_reach,
            min_board_clearance,
        )
    if (
        endpoint_arm_clear is not None
        and endpoint_tcp is not None
        and endpoint_joints is not None
        and not endpoint_arm_clear(endpoint_tcp, endpoint_joints)
    ):
        return PathValidation(
            False,
            "observation endpoint puts the arm in a wrist camera",
            travel,
            min_height,
            max_reach,
            min_board_clearance,
        )
    return PathValidation(
        True,
        "ok",
        travel,
        min_height,
        max_reach,
        min_board_clearance,
    )
