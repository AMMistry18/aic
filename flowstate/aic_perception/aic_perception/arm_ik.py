"""Closed-chain UR5e kinematics and an exact reachability test.

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
aic_robot.xml`` -- the same kinematics the workcell uses) and the **closed-form
analytic** inverse kinematics for that chain, used as a *reachability gate*:
"does a joint-limit-valid solution exist?".  Analytic rather than iterative
because the gate's answer must be trustworthy in both directions: it enumerates
all eight UR branches exactly, so a rejection means the pose is genuinely
outside the workspace (no seed dependence, no local minima, no false
negatives), and it costs microseconds -- cheap enough to gate every framed
candidate rather than a truncated shortlist.  The tool offset (flange ->
gripper/tcp) is **self-calibrated from one live (joint-state, base_T_tcp)
sample**, so no fragile frame convention is hard-coded; the caller validates the
model against live feedback before trusting it and otherwise falls back to the
sphere.

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

# Absolute J1..J6 bounds configured on the downstream Flowstate Move Robot
# segment.  The skill still publishes only a Cartesian TCP target; mirroring
# the same bounds here makes its analytic IK gate reject windings that the real
# planner is forbidden to use.  These cover every endpoint in the 96-case SC
# sweep and both modeled Stage-1 exits while every interval remains well below
# one full revolution.
SC_MOVE_ROBOT_JOINT_LIMITS_DEG: np.ndarray = np.array(
    [
        [-53.6, 170.1],
        [-187.0, -28.9],
        [-122.4, 94.1],
        [-127.7, 43.8],
        [-116.1, 114.8],
        [-71.5, 180.8],
    ],
    dtype=float,
)
SC_MOVE_ROBOT_JOINT_LIMITS: np.ndarray = np.radians(
    SC_MOVE_ROBOT_JOINT_LIMITS_DEG
)

# Denavit-Hartenberg parameters of the same chain, in the classical UR form
#     T_i = Rz(theta_i) . Tz(d_i) . Tx(a_i) . Rx(alpha_i)
# The MJCF chain above *is* this chain: ``fk_flange`` and the DH product agree to
# ~1e-15 with no adapter transform on either end (asserted by
# ``test_arm_ik.test_mjcf_chain_is_the_classical_ur5e_dh_chain``).  That identity
# is what lets the textbook closed-form UR solution below run directly in this
# module's frame.
_DH_A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
_DH_D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
_DH_ALPHA = (math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0)


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


def _dh_matrix(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rigid_inverse(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    rot_t = matrix[:3, :3].T
    inverse[:3, :3] = rot_t
    inverse[:3, 3] = -rot_t @ matrix[:3, 3]
    return inverse


def _wrap_pi(angle: float) -> float:
    """Wrap to (-pi, pi].  Every joint limit is at least +/-pi, so the wrapped
    principal value is inside the limits whenever any co-terminal angle is."""
    return -((-angle + math.pi) % (2.0 * math.pi) - math.pi)


def _lift_near_reference(
    joints: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Choose the in-limit co-terminal representation nearest ``reference``.

    ``solve_all`` deliberately returns principal angles, but the real UR joints
    (except the elbow) span +/-2pi.  Treating ``-134 deg`` and ``+226 deg`` as
    interchangeable when measuring motion hides a complete revolution from the
    survey-pose selector.  Lift each principal angle by an integer multiple of
    2pi, staying inside the physical limits, so the returned target and its
    delta describe actual joint travel rather than circular pose equivalence.
    """
    principal = np.asarray(joints, dtype=float)
    reference = np.asarray(reference, dtype=float)
    lifted = principal.copy()
    for index, angle in enumerate(principal):
        lower, upper = JOINT_LIMITS[index]
        k_min = math.ceil((lower - angle - 1e-12) / _TWO_PI)
        k_max = math.floor((upper - angle + 1e-12) / _TWO_PI)
        choices = [angle + k * _TWO_PI for k in range(k_min, k_max + 1)]
        lifted[index] = min(
            choices,
            key=lambda value: (abs(value - reference[index]), abs(value)),
        )
    return lifted


def lift_into_joint_limits(
    joints: Sequence[float],
    joint_limits: np.ndarray,
    reference: Sequence[float] | None = None,
) -> np.ndarray | None:
    """Represent one IK solution inside an absolute joint-position window.

    A Cartesian TCP pose does not encode whether (for example) wrist joint 6 is
    ``-18 deg`` or the co-terminal ``+342 deg``.  Move Robot receives fixed
    position limits directly in Flowstate, so the analytic gate must express
    every solution inside that exact same window.  ``None`` means at least one
    joint has no co-terminal representation in the window.
    """

    principal = np.asarray(joints, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if principal.shape != (6,) or limits.shape != (6, 2):
        raise ValueError("expected six joints and a (6, 2) joint-limit array")
    if not np.all(np.isfinite(principal)) or not np.all(np.isfinite(limits)):
        raise ValueError("joint values and limits must be finite")
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("joint-limit lower bounds must not exceed upper bounds")
    if np.any(limits[:, 0] < JOINT_LIMITS[:, 0] - 1e-9) or np.any(
        limits[:, 1] > JOINT_LIMITS[:, 1] + 1e-9
    ):
        raise ValueError("joint window exceeds the modeled physical limits")

    preferred = (
        np.asarray(reference, dtype=float)
        if reference is not None
        else np.zeros(6, dtype=float)
    )
    if preferred.shape != (6,) or not np.all(np.isfinite(preferred)):
        raise ValueError("reference must contain six finite joint values")

    lifted = principal.copy()
    for index, angle in enumerate(principal):
        lower, upper = limits[index]
        k_min = math.ceil((lower - angle - 1e-12) / _TWO_PI)
        k_max = math.floor((upper - angle + 1e-12) / _TWO_PI)
        if k_min > k_max:
            return None
        choices = [angle + k * _TWO_PI for k in range(k_min, k_max + 1)]
        lifted[index] = min(
            choices,
            key=lambda value: (abs(value - preferred[index]), abs(value)),
        )
    return lifted


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
    # Points rigidly attached to the flange that must not be driven into the
    # forearm -- in practice the three wrist cameras, which stick ~108 mm off the
    # flange axis and swing into the forearm whenever the wrist folds back.  A
    # pose is only reachable if *some* branch keeps them clear: the planner
    # rejects the rest, and "IK could not find a collision free configuration"
    # fails the move just as hard as an unreachable pose.  Empty (the default)
    # disables the check, keeping pure kinematics for callers without the rig.
    flange_T_probes: tuple[Transform, ...] = ()
    # Calibrated against ground truth: for a survey pose the workcell planner
    # rejected outright (robot.forearm_link vs left_camera.camera_link, all
    # solutions), the best branch put a camera 111 mm from the forearm centreline.
    # 140 mm keeps a 26% margin over that while retaining ~40% of framed NIC
    # candidates, which is far more than the search needs.
    min_self_clearance_m: float = 0.140

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

    # -- inverse kinematics (closed form) ------------------------------------
    def solve_all(
        self,
        base_T_tcp: Transform,
        free_wrist_angle: float = 0.0,
    ) -> list[np.ndarray]:
        """Every joint-limit-valid configuration achieving ``base_T_tcp``.

        The classical UR closed form: the wrist centre fixes shoulder pan
        (2 branches), the tool axis fixes wrist 2 (2 branches), and the
        remaining planar 2R arm fixes the elbow (2 branches) -- at most eight
        exact solutions, enumerated with no seed and no iteration.  An empty
        list therefore means the pose is genuinely outside the arm's workspace.

        ``base_T_tcp`` is in the workcell ``base_link`` frame; it is mapped into
        the kinematic model base via the calibrated ``base`` convention.
        ``free_wrist_angle`` is the wrist-3 angle adopted at the wrist
        singularity, where it is indeterminate.
        """
        # Target flange pose in the DH/model frame.
        flange = self._to_model(base_T_tcp).compose(self.flange_T_tcp.inverse())
        target = np.eye(4)
        target[:3, :3] = flange.rotation
        target[:3, 3] = flange.translation
        d1, _, _, d4, d5, d6 = _DH_D
        _, a2, a3, _, _, _ = _DH_A

        # -- shoulder pan: the wrist centre lies on a cylinder of radius d4
        # about the (vertical) joint-1 axis, giving two symmetric branches.
        wrist_center = target[:3, 3] - d6 * target[:3, 2]
        radius = math.hypot(wrist_center[0], wrist_center[1])
        if radius < abs(d4):
            return []  # inside the shoulder's dead cylinder
        bearing = math.atan2(wrist_center[1], wrist_center[0])
        offset = math.asin(max(-1.0, min(1.0, d4 / radius)))
        solutions: list[np.ndarray] = []
        for theta1 in (bearing + offset, bearing + math.pi - offset):
            s1, c1 = math.sin(theta1), math.cos(theta1)
            # -- wrist 2: the tool's offset from the arm plane fixes |theta5|.
            cos5 = (target[0, 3] * s1 - target[1, 3] * c1 - d4) / d6
            if abs(cos5) > 1.0:
                continue
            for theta5 in (math.acos(cos5), -math.acos(cos5)):
                sin5 = math.sin(theta5)
                if abs(sin5) < 1e-7:
                    # Wrist singularity: theta6 is free (theta4 absorbs it).
                    theta6 = free_wrist_angle
                else:
                    theta6 = math.atan2(
                        (target[1, 1] * c1 - target[0, 1] * s1) / sin5,
                        (target[0, 0] * s1 - target[1, 0] * c1) / sin5,
                    )
                # -- the remaining chain is a planar 2R arm in the joint-1 frame.
                arm_1_4 = (
                    _rigid_inverse(_dh_matrix(theta1, _DH_A[0], d1, _DH_ALPHA[0]))
                    @ target
                    @ _rigid_inverse(_dh_matrix(theta6, _DH_A[5], d6, _DH_ALPHA[5]))
                    @ _rigid_inverse(_dh_matrix(theta5, _DH_A[4], d5, _DH_ALPHA[4]))
                )
                px, py = float(arm_1_4[0, 3]), float(arm_1_4[1, 3])
                cos3 = (px * px + py * py - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
                if abs(cos3) > 1.0:
                    continue  # elbow cannot span this distance
                sum_234 = math.atan2(arm_1_4[1, 0], arm_1_4[0, 0])
                for theta3 in (math.acos(cos3), -math.acos(cos3)):
                    theta2 = math.atan2(py, px) - math.atan2(
                        a3 * math.sin(theta3), a2 + a3 * math.cos(theta3)
                    )
                    joints = np.array(
                        [
                            _wrap_pi(theta1),
                            _wrap_pi(theta2),
                            _wrap_pi(theta3),
                            _wrap_pi(sum_234 - theta2 - theta3),
                            _wrap_pi(theta5),
                            _wrap_pi(theta6),
                        ]
                    )
                    if np.all(joints >= JOINT_LIMITS[:, 0] - 1e-9) and np.all(
                        joints <= JOINT_LIMITS[:, 1] + 1e-9
                    ):
                        solutions.append(joints)
        return solutions

    def link_segments(
        self, joints: Sequence[float]
    ) -> tuple[tuple[np.ndarray, np.ndarray, float], ...]:
        """Upper-arm and forearm as (start, end, radius) in the base_link frame.

        These are the links that can end up standing in the wrist cameras' own
        view.  The gripper is handled separately by a fixed image-space mask --
        valid because it is rigidly attached to wrist_3 -- but these two are
        upstream of the wrist, so where they land in the image depends entirely
        on the joint configuration and no static mask can represent them.
        Radii are the UR5e collision tubes (~60 mm and ~50 mm diameter).
        """
        _axes, points, _flange = self._chain(joints)
        return (
            (self.base.apply(points[1]), self.base.apply(points[2]), 0.060),
            (self.base.apply(points[2]), self.base.apply(points[3]), 0.050),
        )

    def self_clearance(self, joints: Sequence[float]) -> float:
        """Distance from the nearest flange probe to the forearm centreline.

        ``forearm_link`` runs from the elbow (joint-3 origin) to wrist_1
        (joint-4 origin); the wrist cameras ride on the flange.  Folding the
        wrist back swings them onto that segment, which is the self-collision
        the workcell planner reports.  ``inf`` when no probes are configured.
        """
        if not self.flange_T_probes:
            return math.inf
        _axes, points, flange = self._chain(joints)
        elbow, wrist_1 = points[2], points[3]
        segment = wrist_1 - elbow
        length_sq = float(segment @ segment)
        worst = math.inf
        for probe in self.flange_T_probes:
            point = flange.compose(probe).translation
            t = 0.0
            if length_sq > 1e-12:
                t = float((point - elbow) @ segment) / length_sq
                t = min(1.0, max(0.0, t))
            worst = min(worst, float(np.linalg.norm(point - (elbow + t * segment))))
        return worst

    def solve(
        self, base_T_tcp: Transform, seed: Sequence[float] | None = None
    ) -> np.ndarray | None:
        """Joint angles reaching ``base_T_tcp`` within limits and clear of the
        arm's own forearm, or None.

        With a ``seed`` (the current joint state) the branch requiring the least
        joint travel is returned, which is the configuration the controller
        would realistically adopt.
        """
        ranked = self.solve_ranked(base_T_tcp, seed)
        if not ranked:
            return None
        return ranked[0]

    def solve_ranked(
        self,
        base_T_tcp: Transform,
        seed: Sequence[float] | None = None,
        *,
        joint_limits: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """All forearm-clear IK branches in permitted finite coordinates.

        ``solve`` historically chose one branch before the caller could apply
        its image-space arm-occlusion test.  If that nearest branch put the
        upper arm in a wrist camera, the whole pose was rejected even when a
        different exact shoulder/elbow/wrist branch left every image clear.
        This method exposes the complete filtered set and orders it by travel.
        When ``joint_limits`` is supplied, each branch is represented inside
        that absolute window instead of merely choosing the physical-limit
        winding nearest the seed.  This mirrors a downstream Cartesian Move
        Robot segment that has the same fixed position limits.
        """

        solutions = [
            q
            for q in self.solve_all(base_T_tcp)
            if self.self_clearance(q) >= self.min_self_clearance_m
        ]
        if seed is None:
            reference = np.zeros(6, dtype=float)
        else:
            reference = np.asarray(seed, dtype=float)
        if joint_limits is None:
            lifted = [_lift_near_reference(q, reference) for q in solutions]
        else:
            lifted = [
                mapped
                for q in solutions
                if (
                    mapped := lift_into_joint_limits(
                        q, joint_limits, reference=reference
                    )
                )
                is not None
            ]
        lifted.sort(
            key=lambda q: (
                float(np.abs(q - reference).max()),
                float(np.abs(q - reference).sum()),
            )
        )
        return lifted

    def reachable(
        self, base_T_tcp: Transform, seed: Sequence[float] | None = None
    ) -> bool:
        return self.solve(base_T_tcp, seed) is not None
