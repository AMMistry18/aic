"""Closed-chain UR5e kinematics and a numerical reachability test.

The survey search (``board_stage2.search_survey_pose``) historically judged a
candidate TCP pose "reachable" by a single base-origin sphere
(``norm(base_T_tcp.translation) <= max_reach``).  That is not what the arm can
actually do: reaching a point is about whether the six joints can fold into a
pose that puts the tool *there and pointing that way* within the joint limits,
not about straight-line distance.  The crude test both admits poses the real
IK cannot solve (Move Robot then reports "IK not computable") and rejects
genuinely reachable far-side poses, so the search settles for a near pose that
frames the parts from the wrong side.

This module provides an exact forward kinematics for the UR5e wrist chain taken
directly from the production MuJoCo model (``aic_utils/aic_mujoco/mjcf/
aic_robot.xml`` -- the same kinematics the workcell uses) and a damped
least-squares inverse-kinematics solver used purely as a *reachability gate*:
"does a joint-limit-valid solution exist near a sensible seed?".  The tool
offset (flange -> gripper/tcp) is **self-calibrated from one live
(joint-state, base_T_tcp) sample**, so no fragile frame convention is
hard-coded; the caller validates the model against live feedback before
trusting it and otherwise falls back to the sphere.

Only the robot's own kinematics are used here -- no task-board / port / scoring
transforms -- so this stays within the skill's permitted-TF policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .board_stage2 import Transform

# ---------------------------------------------------------------------------
# UR5e kinematic chain (base_link -> wrist_3), verbatim from the MuJoCo model.
# Each entry is the fixed parent->child transform (translation metres, quaternion
# w, x, y, z) that precedes joint i's rotation about the child-local +Z axis.
# ---------------------------------------------------------------------------
_LINKS: tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...] = (
    ((0.0, 0.0, 0.1625), (1.0, 0.0, 0.0, 0.0)),                    # base -> shoulder
    ((0.0, 0.0, 0.0), (0.70710678, 0.70710678, 0.0, 0.0)),        # shoulder -> upper_arm
    ((-0.425, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),                   # upper_arm -> forearm
    ((-0.3922, 0.0, 0.1333), (1.0, 0.0, 0.0, 0.0)),              # forearm -> wrist_1
    ((0.0, -0.0997, 0.0), (0.70710678, 0.70710678, 0.0, 0.0)),   # wrist_1 -> wrist_2
    ((0.0, 0.0996, 0.0), (0.70710678, -0.70710678, 0.0, 0.0)),   # wrist_2 -> wrist_3
)

# Joint limits (radians), from the MJCF <joint range> fields.  Elbow is +/-pi;
# the rest are +/-2pi.
_TWO_PI = 2.0 * math.pi
JOINT_LIMITS: np.ndarray = np.array(
    [
        [-_TWO_PI, _TWO_PI],
        [-_TWO_PI, _TWO_PI],
        [-math.pi, math.pi],
        [-_TWO_PI, _TWO_PI],
        [-_TWO_PI, _TWO_PI],
        [-_TWO_PI, _TWO_PI],
    ],
    dtype=float,
)

# Coarse reach envelope of the wrist-3 origin from the shoulder centre, used as a
# cheap pre-filter before the (more expensive) numerical solve.  Generous on
# purpose: the numerical IK is the authority; this only drops the obviously
# impossible.  Sum of the arm link extents with margin.
_REACH_MAX_M = 0.425 + 0.3922 + 0.1333 + 0.0997 + 0.0996 + 0.05
_REACH_MIN_M = 0.05


def _quat_wxyz_to_matrix_raw(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


# Precomputed fixed link rotations/translations (numpy) for a fast FK hot path.
_LINK_R: tuple[np.ndarray, ...] = tuple(
    _quat_wxyz_to_matrix_raw(*quat) for _pos, quat in _LINKS
)
_LINK_T: tuple[np.ndarray, ...] = tuple(
    np.asarray(pos, dtype=float) for pos, _quat in _LINKS
)


def _quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _log_so3(rotation: np.ndarray) -> np.ndarray:
    """Return the rotation vector (axis * angle) of a 3x3 rotation matrix."""
    cos_theta = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=float)
    if math.pi - theta < 1e-6:
        # Near pi: use the symmetric part to recover the axis robustly.
        a = 0.5 * (rotation + np.eye(3))
        axis = np.sqrt(np.clip(np.diag(a), 0.0, None))
        # Fix signs from off-diagonal terms.
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            axis[1] = math.copysign(axis[1], rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[2, 0])
        elif axis[1] >= axis[2]:
            axis[0] = math.copysign(axis[0], rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[2, 1])
        else:
            axis[0] = math.copysign(axis[0], rotation[2, 0])
            axis[1] = math.copysign(axis[1], rotation[2, 1])
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.zeros(3, dtype=float)
        return (axis / n) * theta
    w = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    return w * (theta / (2.0 * math.sin(theta)))


# Canonical seeds (radians) spanning a grid of shoulder-pan (which end of the
# workspace the arm faces -- the dominant DOF for reaching a far-side pose),
# shoulder-lift, and elbow up/down, so the local solver can find a valid branch
# even when the current pose is a poor starting point.  The current joint state
# is always tried first.
def _build_canonical_seeds() -> tuple[tuple[float, ...], ...]:
    seeds = []
    for pan in (-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi):
        for elbow in (1.6, -1.6):
            lift = -1.4
            wrist1 = -math.pi / 2.0 - lift - elbow  # keep the tool roughly level
            seeds.append((pan, lift, elbow, wrist1, -math.pi / 2.0, 0.0))
    return tuple(seeds)


_CANONICAL_SEEDS: tuple[tuple[float, ...], ...] = _build_canonical_seeds()


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


# Candidate ``base_link -> model-base`` corrections tried by ``autocalibrate``.
# The workcell TF frame the skill targets ("base_link") may differ from this
# module's kinematic base by a fixed frame convention (classically the UR
# base/base_link 180-deg-about-Z), so we auto-detect which one makes the
# recovered flange->TCP offset physically plausible.  Pure rotations only:
# base_link is the arm base, so no translation is expected.
_BASE_CANDIDATES: tuple[tuple[str, np.ndarray], ...] = (
    ("identity", np.eye(3)),
    ("Rz180", _rot_z(math.pi)),
    ("Rz90", _rot_z(math.pi / 2.0)),
    ("Rz-90", _rot_z(-math.pi / 2.0)),
    ("Rx180", _rot_x(math.pi)),
    ("Rx180 Rz180", _rot_x(math.pi) @ _rot_z(math.pi)),
)


@dataclass
class UR5eArm:
    """Forward kinematics + numerical reachability for the workcell UR5e.

    ``flange_T_tcp`` maps the wrist-3 flange to the controller's TCP frame.
    ``base`` is ``base_link -> model-base`` (the fixed frame convention between
    the workcell's ``base_link`` TF and this module's kinematic base); it and
    the tool offset are recovered together by :meth:`autocalibrate` from a single
    measured (joint-state, base_T_tcp) pair, so no convention is hard-coded.
    """

    flange_T_tcp: Transform = Transform(np.eye(3), np.zeros(3))
    base: Transform = Transform(np.eye(3), np.zeros(3))

    def _to_model(self, base_link_pose: Transform) -> Transform:
        """Express a base_link-frame pose in the kinematic (model) base frame."""
        return self.base.inverse().compose(base_link_pose)

    # -- forward kinematics --------------------------------------------------
    def _chain(self, joints: Sequence[float]):
        """Return (per-joint axes, per-joint points, flange Transform).

        Raw-numpy hot path (no per-link Transform objects): frame_{i} =
        frame_{i-1} @ fixed_i @ Rz(theta_i).  The joint axis/point are taken
        after the fixed transform, before the rotation.
        """
        rot = np.eye(3)
        pos = np.zeros(3)
        axes = []
        points = []
        for i in range(6):
            rp = rot @ _LINK_R[i]
            tp = rot @ _LINK_T[i] + pos
            axes.append(rp[:, 2])
            points.append(tp)
            theta = float(joints[i])
            c, s = math.cos(theta), math.sin(theta)
            rot = np.column_stack(
                (rp[:, 0] * c + rp[:, 1] * s, -rp[:, 0] * s + rp[:, 1] * c, rp[:, 2])
            )
            pos = tp
        return axes, points, Transform(rot, pos)

    def fk_flange(self, joints: Sequence[float]) -> Transform:
        _, _, frame = self._chain(joints)
        return frame

    def fk(self, joints: Sequence[float]) -> Transform:
        """base_link -> TCP for the six joint angles."""
        return self.base.compose(
            self.fk_flange(joints).compose(self.flange_T_tcp)
        )

    def jacobian(self, joints: Sequence[float]) -> tuple[np.ndarray, Transform]:
        axes, points, flange = self._chain(joints)
        base_T_tcp = flange.compose(self.flange_T_tcp)
        p_tcp = base_T_tcp.translation
        jac = np.zeros((6, 6), dtype=float)
        for i in range(6):
            jac[:3, i] = np.cross(axes[i], p_tcp - points[i])
            jac[3:, i] = axes[i]
        return jac, base_T_tcp

    # -- calibration ---------------------------------------------------------
    @classmethod
    def calibrated_from(
        cls,
        joints: Sequence[float],
        base_T_tcp: Transform,
        base: Transform | None = None,
    ) -> "UR5eArm":
        """Build an arm whose tool offset reproduces this measured sample under
        the given ``base`` (base_link -> model-base) convention."""
        if base is None:
            base = Transform(np.eye(3), np.zeros(3))
        arm = cls(base=base)
        flange = arm.fk_flange(joints)
        # base_T_tcp is base_link; move it into the model base before recovering
        # the (model-frame) flange->TCP offset.
        model_tcp = base.inverse().compose(base_T_tcp)
        flange_T_tcp = flange.inverse().compose(model_tcp)
        return cls(flange_T_tcp=flange_T_tcp, base=base)

    @classmethod
    def autocalibrate(
        cls,
        joints: Sequence[float],
        base_T_tcp: Transform,
        *,
        min_tool_m: float = 0.05,
        max_tool_m: float = 0.35,
    ) -> tuple["UR5eArm | None", str]:
        """Recover (base convention, tool offset) from one measured sample.

        Tries each candidate ``base_link -> model-base`` rotation and keeps the
        one whose recovered flange->TCP offset is physically plausible (a real
        UR5e tool is ~0.15-0.30 m off the flange, roughly along the flange axis).
        Returns ``(arm, description)`` or ``(None, diagnostics)``.
        """
        report = []
        for name, rot in _BASE_CANDIDATES:
            base = Transform(rot, np.zeros(3))
            arm = cls.calibrated_from(joints, base_T_tcp, base=base)
            off = arm.flange_T_tcp.translation
            mag = float(np.linalg.norm(off))
            # Fraction of the offset that lies along the flange +Z axis (a real
            # wrist tool points essentially straight out of the flange).
            axial = abs(float(off[2])) / mag if mag > 1e-9 else 0.0
            report.append(f"{name}:{mag * 1000:.0f}mm/ax{axial:.2f}")
            if min_tool_m <= mag <= max_tool_m and axial >= 0.6:
                return arm, f"base={name} tool={mag * 1000:.1f}mm axial={axial:.2f}"
        return None, "no plausible base; candidates " + " ".join(report)

    def fk_residual(
        self, joints: Sequence[float], base_T_tcp: Transform
    ) -> tuple[float, float]:
        """(position error metres, orientation error radians) of the model."""
        model = self.fk(joints)
        pos = float(np.linalg.norm(model.translation - base_T_tcp.translation))
        ori = float(
            np.linalg.norm(_log_so3(model.rotation @ base_T_tcp.rotation.T))
        )
        return pos, ori

    # -- inverse kinematics --------------------------------------------------
    def _wrist_center_reachable(self, model_T_tcp: Transform) -> bool:
        flange = model_T_tcp.compose(self.flange_T_tcp.inverse())
        shoulder = np.array([0.0, 0.0, float(_LINKS[0][0][2])], dtype=float)
        d = float(np.linalg.norm(flange.translation - shoulder))
        return _REACH_MIN_M <= d <= _REACH_MAX_M

    def solve(
        self,
        base_T_tcp: Transform,
        seed: Sequence[float] | None = None,
        *,
        pos_tol_m: float = 1e-3,
        ori_tol_rad: float = 5e-3,
        max_iters: int = 80,
        damping: float = 0.06,
    ) -> np.ndarray | None:
        """Return joint angles reaching ``base_T_tcp`` within limits, or None.

        ``base_T_tcp`` is in the workcell ``base_link`` frame; it is mapped into
        the kinematic model base via the calibrated ``base`` convention.
        """
        model_T_tcp = self._to_model(base_T_tcp)
        if not self._wrist_center_reachable(model_T_tcp):
            return None
        target_p = model_T_tcp.translation
        target_R = model_T_tcp.rotation
        lower = JOINT_LIMITS[:, 0]
        upper = JOINT_LIMITS[:, 1]
        eye6 = np.eye(6)
        seeds: list[np.ndarray] = []
        if seed is not None:
            seeds.append(np.asarray(seed, dtype=float))
        seeds.extend(np.asarray(s, dtype=float) for s in _CANONICAL_SEEDS)
        for q0 in seeds:
            q = np.clip(q0.copy(), lower, upper)
            prev_err = math.inf
            stalls = 0
            for _ in range(max_iters):
                jac, base_T_now = self.jacobian(q)
                e = np.empty(6, dtype=float)
                e[:3] = target_p - base_T_now.translation
                e[3:] = _log_so3(target_R @ base_T_now.rotation.T)
                err = float(np.linalg.norm(e))
                if (
                    float(np.linalg.norm(e[:3])) < pos_tol_m
                    and float(np.linalg.norm(e[3:])) < ori_tol_rad
                ):
                    if np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6):
                        return np.clip(q, lower, upper)
                    break  # converged but out of limits: try next seed
                # Abandon a seed that has stopped making progress (a local
                # minimum) so the fixed iteration budget is not wasted.
                if err > prev_err - 1e-4:
                    stalls += 1
                    if stalls >= 6:
                        break
                else:
                    stalls = 0
                prev_err = err
                jjt = jac @ jac.T + (damping * damping) * eye6
                dq = jac.T @ np.linalg.solve(jjt, e)
                # Bound the step so the linearisation stays valid.
                norm = float(np.linalg.norm(dq))
                if norm > 0.5:
                    dq *= 0.5 / norm
                q = np.clip(q + dq, lower, upper)
        return None

    def reachable(
        self, base_T_tcp: Transform, seed: Sequence[float] | None = None
    ) -> bool:
        return self.solve(base_T_tcp, seed) is not None
