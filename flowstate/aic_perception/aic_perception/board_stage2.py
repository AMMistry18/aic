"""In-place geometric Stage 2 for the Check Board Visibility skill.

Stage 1 (the adaptive viewpoint search) acquires a fully visible board view.
Stage 2 takes that view and, *without* any blind joint sweep, estimates the
board's full 6-DoF pose from CAD-derived planar landmarks and then computes a
single board-relative TCP survey pose that frames the loose SFP modules parked
on both SFP mount rails in all three wrist cameras at once.

This module is deliberately ROS-free so the geometry -- planar PnP with
orientation disambiguation, the CAD-derived loose-SFP envelope, the multi-camera
candidate scoring, and the post-move verification -- can be unit-tested with
synthetic intrinsics, extrinsics and detections.  The only third-party
dependencies are numpy and (for ``solvePnP``) OpenCV, both already required by
the surrounding perception package.

Coordinate conventions
----------------------
* ``base``  -- the robot ``base_link`` frame (all TFs are resolved here).
* ``board`` -- the task-board frame whose origin is the centre of the base
  plate; +Z is the outward board normal.  CAD landmark coordinates below are
  expressed in this frame (metres).
* ``cam``   -- an optical frame: +X image-right, +Y image-down, +Z into scene.

A transform ``a_T_b`` maps a point expressed in ``b`` into ``a`` via
``p_a = R @ p_b + t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from .board_visibility import rotation_matrix_from_quaternion


# ---------------------------------------------------------------------------
# CAD-derived landmark geometry (task-board frame, metres).
#
# Sourced from the repository CAD:
#   aic_assets/models/Task Board Base/model.sdf  -- base plate collision box
#     size (0.3, 0.425, 0.012) centred at (0, 0, 0.006): the board outline
#     rectangle whose top face is at z = 0.012.
#   aic_assets/models/Task Board Base/base_visual.glb -- the MAGENTA4 primitive
#     (position accessor 29, index accessor 33, transformed by node 6).  The
#     purple mark is an asymmetric three-sided outline; its *rendered material*
#     coordinates, rather than the unrelated collision-box centre, are used
#     below to disambiguate board orientation.
#   aic_description/urdf/task_board.urdf.xacro   -- SFP mount rails at board
#     X = 0.055, Y = +/-0.10625, Z = 0.01; mounts translate along the rail
#     (Y) by [-0.09625, 0.09625].
#   aic_assets/models/SFP Module/model.sdf       -- loose module bounding size.
# ---------------------------------------------------------------------------

BOARD_OUTLINE_HALF_X = 0.15
BOARD_OUTLINE_HALF_Y = 0.2125
BOARD_TOP_Z = 0.012

# Outline corners in a fixed winding (CCW looking down the +Z board normal),
# starting from the (+X, +Y) corner.  Correspondence to the detected image quad
# is resolved by trying every cyclic rotation and rejecting ambiguity.
BOARD_OUTLINE_CORNERS = np.array(
    [
        [+BOARD_OUTLINE_HALF_X, +BOARD_OUTLINE_HALF_Y, BOARD_TOP_Z],
        [-BOARD_OUTLINE_HALF_X, +BOARD_OUTLINE_HALF_Y, BOARD_TOP_Z],
        [-BOARD_OUTLINE_HALF_X, -BOARD_OUTLINE_HALF_Y, BOARD_TOP_Z],
        [+BOARD_OUTLINE_HALF_X, -BOARD_OUTLINE_HALF_Y, BOARD_TOP_Z],
    ],
    dtype=float,
)

# Unique vertices of the MAGENTA4 primitive after applying the GLB node
# transform, in task-board metres.  Keeping these CAD values in source makes
# accidental regressions back to the collision-box centre detectable in tests.
LOGO_MATERIAL_VERTICES = np.array(
    [
        [-0.11750002, 0.11750002, 0.01099998],
        [-0.12250002, 0.10250002, 0.01099998],
        [-0.10750002, 0.10250002, 0.01099998],
        [-0.10750001, 0.11750002, 0.01099998],
        [-0.12250002, 0.19750003, 0.01099996],
        [-0.11750002, 0.19250003, 0.01099996],
        [-0.03250000, 0.11750003, 0.01099999],
        [-0.09750002, 0.11750003, 0.01099998],
        [-0.09750002, 0.10250002, 0.01099998],
        [-0.02750000, 0.10250003, 0.01099999],
        [-0.02750000, 0.19750004, 0.01099997],
        [-0.03250000, 0.19250005, 0.01099998],
    ],
    dtype=float,
)

# Area centroid of the indexed MAGENTA4 triangles.  This is the physical point
# measured by a pixel-mask centroid in the ideal render.  It is deliberately
# not the GLB primitive's bounding-box centre: the mark is asymmetric.
LOGO_MATERIAL_CENTROID = np.array(
    [-0.07335001, 0.13965003, 0.01099998], dtype=float
)

# Compatibility name used by the initial Stage-2 integration.  Despite the old
# name, this now denotes the rendered magenta material centroid, not the SDF
# collision plate centre.
LOGO_PLATE_CENTER = LOGO_MATERIAL_CENTROID

# Axis-aligned bounding rectangle of the insignia in the board frame (metres),
# wound CCW looking down +Z from the (+X, +Y) corner to match
# ``BOARD_OUTLINE_CORNERS``.  Its four corners coincide with real bracket-stroke
# corners of ``LOGO_MATERIAL_VERTICES`` (max/min X at the outer verticals, max/min
# Y at the outer horizontals), so a four-point planar PnP against this rectangle
# is a genuine correspondence.  This is the primary, clip-proof pose target: the
# insignia stays in frame at survey standoffs where the full plate outline does not.
_INSIGNIA_Z = float(LOGO_MATERIAL_VERTICES[:, 2].mean())
_INSIGNIA_X_MIN = float(LOGO_MATERIAL_VERTICES[:, 0].min())
_INSIGNIA_X_MAX = float(LOGO_MATERIAL_VERTICES[:, 0].max())
_INSIGNIA_Y_MIN = float(LOGO_MATERIAL_VERTICES[:, 1].min())
_INSIGNIA_Y_MAX = float(LOGO_MATERIAL_VERTICES[:, 1].max())
INSIGNIA_RECT_CORNERS = np.array(
    [
        [_INSIGNIA_X_MAX, _INSIGNIA_Y_MAX, _INSIGNIA_Z],
        [_INSIGNIA_X_MIN, _INSIGNIA_Y_MAX, _INSIGNIA_Z],
        [_INSIGNIA_X_MIN, _INSIGNIA_Y_MIN, _INSIGNIA_Z],
        [_INSIGNIA_X_MAX, _INSIGNIA_Y_MIN, _INSIGNIA_Z],
    ],
    dtype=float,
)
# Asymmetric material centroid used to break the near-square rectangle's
# rotation/mirror ambiguity (offset ~1cm in -Y and slightly +X from the bbox
# centre): the same disambiguation role the logo plays for the outline PnP.
INSIGNIA_CENTROID = LOGO_MATERIAL_CENTROID

# SFP mount rails (board frame).
SFP_RAIL_X = 0.055
SFP_RAIL_Y_ABS = 0.10625
SFP_RAIL_Z = 0.01
SFP_RAIL_TRANSLATION = 0.09625  # xacro range; config range is +/-0.09425.
# CAD collision-corner radial reaches about each model origin:
#   SFP Mount  = 0.04882 m, SFP Module = 0.03347 m.
# The envelope uses the larger fixture reach and rounds it outward by over
# 6 mm.  Both CAD models are expressed about their attachment origins; taking
# the union (not summing unrelated radial extrema) covers either rendered body
# at every legal rail translation without inventing an impossible 17 cm-wide
# module.
SFP_MOUNT_RADIAL_REACH = 0.04882
SFP_MODULE_RADIAL_REACH = 0.03347
SFP_BODY_PAD_XY = 0.055
# Board-normal extent: from the plate top (~0.01) up over the mount (~0.017)
# and the protruding transceiver (~0.02).
SFP_ENVELOPE_Z_MIN = 0.0
SFP_ENVELOPE_Z_MAX = 0.06


def sfp_envelope_corners() -> np.ndarray:
    """Return the 8 board-frame corners bounding all legal loose SFP modules.

    A single conservative axis-aligned box (board frame) covering the staged
    physical SFP modules in task-board Zones 3/4: both ``sfp_mount_rail_*``
    fixtures over their complete legal translation range.  This intentionally
    does *not* target the NIC-card SFP ports in Zone 1; ports and physical
    modules are distinct task-board entities.
    """

    x_min = SFP_RAIL_X - SFP_BODY_PAD_XY
    x_max = SFP_RAIL_X + SFP_BODY_PAD_XY
    # Rail centre reaches +/-(SFP_RAIL_Y_ABS + SFP_RAIL_TRANSLATION); pad both
    # rails' outermost seats by the body half-extent.
    y_reach = SFP_RAIL_Y_ABS + SFP_RAIL_TRANSLATION + SFP_BODY_PAD_XY
    y_min = -y_reach
    y_max = +y_reach
    z_min = SFP_ENVELOPE_Z_MIN
    z_max = SFP_ENVELOPE_Z_MAX
    corners = []
    for x in (x_min, x_max):
        for y in (y_min, y_max):
            for z in (z_min, z_max):
                corners.append((x, y, z))
    return np.array(corners, dtype=float)


def sfp_envelope_center() -> np.ndarray:
    """Centroid of the conservative loose-SFP envelope (board frame)."""
    return sfp_envelope_corners().mean(axis=0)


def sfp_module_detail_boxes() -> tuple[np.ndarray, ...]:
    """Return conservative, individual-module detail probes.

    The terminal safety envelope above deliberately covers every legal rail
    position.  Its projected span is therefore *not* evidence that any one
    physical transceiver has enough image detail.  These six legal-seat boxes are
    conservative detail probes distributed over the legal rail extent.  They
    are used only for the minimum-detail gate; the full envelope remains the
    clipping/occlusion target.  Callers with measured module poses can replace
    them through ``module_envelopes_board``.
    """
    # The two staging groups have three 50 mm-pitch seats each.  Five modules
    # can occupy any five of these six legal seats, so all six are checked.  A
    # row that happens to leave one seat empty cannot make a distant/blurred
    # module pass on the strength of the union envelope's span.
    ys = (-0.15625, -0.10625, -0.05625, 0.05625, 0.10625, 0.15625)
    half_x, half_y = 0.025, 0.022
    boxes: list[np.ndarray] = []
    for y in ys:
        points = []
        for x in (SFP_RAIL_X - half_x, SFP_RAIL_X + half_x):
            for yy in (y - half_y, y + half_y):
                for z in (SFP_ENVELOPE_Z_MIN, SFP_ENVELOPE_Z_MAX):
                    points.append((x, yy, z))
        boxes.append(np.asarray(points, dtype=float))
    return tuple(boxes)


# LC / SFP / SC mount-rail board-X positions (task_board.urdf.xacro).  The module
# region spans all three rail families on both Y sides over their full travel.
LC_RAIL_X = 0.0275
SC_RAIL_X = 0.0985
MOUNT_RAIL_TRANSLATION = 0.09625


def module_coverage_corners() -> np.ndarray:
    """Return the 8 board-frame corners of the SFP/SC module region.

    The **minimum** coverage target: the whole pick + assembly strip where the
    modules and their cables sit -- every LC/SFP/SC mount rail on both Y sides
    over the full +/-``MOUNT_RAIL_TRANSLATION`` travel, padded by the SFP body
    half-extent.  Widens ``sfp_envelope_corners`` in board-X to include the SC
    rail; a survey pose framing this frames all module rows in all three cameras.
    """

    x_min = LC_RAIL_X - SFP_BODY_PAD_XY
    x_max = SC_RAIL_X + SFP_BODY_PAD_XY
    y_reach = SFP_RAIL_Y_ABS + MOUNT_RAIL_TRANSLATION + SFP_BODY_PAD_XY
    corners = []
    for x in (x_min, x_max):
        for y in (-y_reach, y_reach):
            for z in (SFP_ENVELOPE_Z_MIN, SFP_ENVELOPE_Z_MAX):
                corners.append((x, y, z))
    return np.array(corners, dtype=float)


def board_coverage_corners() -> np.ndarray:
    """Return the 8 board-frame corners of the whole board face.

    The **preferred** coverage target: the full 0.30 x 0.425 m plate outline
    extruded from the plate top up over the tallest module.  A survey pose
    framing this in all three cameras is a strict superset of the module region,
    so it also satisfies the minimum SFP/SC requirement.
    """

    corners = []
    for x in (-BOARD_OUTLINE_HALF_X, BOARD_OUTLINE_HALF_X):
        for y in (-BOARD_OUTLINE_HALF_Y, BOARD_OUTLINE_HALF_Y):
            for z in (BOARD_TOP_Z, SFP_ENVELOPE_Z_MAX):
                corners.append((x, y, z))
    return np.array(corners, dtype=float)


# ---------------------------------------------------------------------------
# Per-sector coverage targets (board frame, metres), derived from
# task_board.urdf.xacro component joint origins.  Framing the whole board in all
# three canted wrist cameras needs a standoff beyond the UR5e's reach; a single
# sector is small enough to frame in all three cameras from a reachable pose.
# ---------------------------------------------------------------------------


def _sector_box_corners(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    """Return the 8 board-frame corners of an axis-aligned sector box."""
    corners = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                corners.append((x, y, z))
    return np.array(corners, dtype=float)


def sfp_sector_corners() -> np.ndarray:
    """SFP pick modules on the +Y rail (SFP mount rail 1).

    Rail at board-X 0.055; rail-1 mounts sit at Y = 0.10625 +/- 0.09625 travel;
    the box adds the SFP body half-extent.  This is the ``STAGED_SFP_MODULE``
    survey target.
    """
    return _sector_box_corners((0.02, 0.09), (0.0, 0.225), (0.01, 0.06))


def sc_sector_corners() -> np.ndarray:
    """SC optical ports (Zone 2): sc_port_0/1 at board-X -0.075 +/- 0.055.

    The ``SC_DESTINATION_PORT`` survey target.
    """
    return _sector_box_corners((-0.14, -0.01), (-0.02, 0.10), (0.01, 0.05))


def nic_sector_corners() -> np.ndarray:
    """NIC card SFP-port destinations (Zone 1): five mounts at board-X -0.081.

    The ``NIC_SFP_DESTINATION`` survey target.
    """
    return _sector_box_corners((-0.14, -0.03), (-0.19, 0.01), (0.01, 0.05))


# ---------------------------------------------------------------------------
# Rigid-transform helpers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """A rigid transform ``a_T_b`` mapping points from frame b into frame a."""

    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(self.translation, dtype=float).reshape(3)
        if not (np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation))):
            raise ValueError("transform contains non-finite values")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def from_quaternion(
        cls,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        translation: Sequence[float],
    ) -> "Transform":
        return cls(
            rotation_matrix_from_quaternion(qx, qy, qz, qw),
            np.asarray(translation, dtype=float),
        )

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        out = pts @ self.rotation.T + self.translation
        return out[0] if single else out

    def inverse(self) -> "Transform":
        rt = self.rotation.T
        return Transform(rt, -rt @ self.translation)

    def compose(self, other: "Transform") -> "Transform":
        """Return ``self_T_c`` given ``other`` is ``b_T_c`` and self is ``a_T_b``."""
        return Transform(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
        )


def quaternion_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return an xyzw quaternion for a 3x3 rotation matrix."""
    m = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= float(np.linalg.norm(q))
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


# ---------------------------------------------------------------------------
# Pinhole camera model and projection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraModel:
    """A calibrated pinhole camera associated with a frame and image size."""

    name: str
    K: np.ndarray  # 3x3 intrinsics
    width: int
    height: int
    distortion: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=float),
        repr=False,
        compare=False,
    )
    distortion_model: str = "plumb_bob"

    def __post_init__(self) -> None:
        k = np.asarray(self.K, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(k)):
            raise ValueError("intrinsics contain non-finite values")
        if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
            raise ValueError("focal lengths must be positive")
        if self.width < 2 or self.height < 2:
            raise ValueError("image dimensions must be at least two pixels")
        distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if not np.all(np.isfinite(distortion)):
            raise ValueError("distortion coefficients contain non-finite values")
        model = self.distortion_model or "plumb_bob"
        supported_lengths = {
            "plumb_bob": {0, 4, 5},
            "rational_polynomial": {8},
        }
        if model not in supported_lengths:
            raise ValueError(f"unsupported camera distortion model {model!r}")
        if len(distortion) not in supported_lengths[model]:
            expected = sorted(supported_lengths[model])
            raise ValueError(
                f"{model} distortion expects coefficient count in {expected}, "
                f"got {len(distortion)}"
            )
        if len(distortion) == 0:
            distortion = np.zeros(5, dtype=float)
        object.__setattr__(self, "K", k)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "distortion_model", model)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])


def project_points(
    points_cam: np.ndarray, camera: CameraModel, min_depth: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Project optical-frame points to pixels.

    Returns ``(pixels, in_front)`` where ``pixels`` is ``Nx2`` and ``in_front``
    is a boolean mask that is true where the point is at least ``min_depth`` in
    front of the camera.  Behind-camera points get NaN pixels so callers cannot
    accidentally treat them as visible.
    """

    import cv2

    pts = np.atleast_2d(np.asarray(points_cam, dtype=float))
    depth = pts[:, 2]
    in_front = depth >= min_depth
    pixels = np.full((len(pts), 2), np.nan, dtype=float)
    if np.any(in_front):
        projected, _ = cv2.projectPoints(
            pts[in_front].astype(np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera.K,
            camera.distortion,
        )
        pixels[in_front] = projected.reshape(-1, 2)
    return pixels, in_front


# ---------------------------------------------------------------------------
# Board pose estimation: planar PnP with orientation disambiguation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardPoseEstimate:
    """A geometrically estimated board pose expressed in ``base_link``."""

    base_T_board: Transform
    reprojection_error_px: float
    ambiguity_ratio: float
    logo_error_px: float
    camera_name: str

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        return quaternion_from_matrix(self.base_T_board.rotation)


def board_pose_set_is_consistent(
    estimates: Mapping[str, BoardPoseEstimate],
    reference: BoardPoseEstimate,
    required_cameras: Sequence[str],
    *,
    max_translation_m: float = 0.04,
    max_angle_rad: float = math.radians(6.0),
) -> tuple[bool, str]:
    """Require every camera pose to agree with the plan and with every peer."""
    required = tuple(required_cameras)
    missing = sorted(set(required).difference(estimates))
    if missing:
        return False, f"missing board poses for {missing}"

    selected = {name: estimates[name] for name in required}
    comparisons: list[tuple[str, BoardPoseEstimate, BoardPoseEstimate]] = [
        (f"{name}->plan", estimate, reference)
        for name, estimate in selected.items()
    ]
    names = list(selected)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            comparisons.append(
                (
                    f"{first_name}<->{second_name}",
                    selected[first_name],
                    selected[second_name],
                )
            )

    failures = []
    for label, first, second in comparisons:
        translation = float(
            np.linalg.norm(
                first.base_T_board.translation
                - second.base_T_board.translation
            )
        )
        angle = _rotation_distance_rad(
            first.base_T_board.rotation,
            second.base_T_board.rotation,
        )
        if translation > max_translation_m or angle > max_angle_rad:
            failures.append(
                f"{label} drift={translation:.3f}m/{math.degrees(angle):.1f}deg"
            )
    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def _order_quad_ccw(quad: np.ndarray) -> np.ndarray:
    """Return the 4 pixel points ordered counter-clockwise in image space.

    Image +Y points down, so a mathematically clockwise sort of the angle about
    the centroid yields a CCW winding when viewed on screen -- matching the CCW
    board-frame corner order under a normal (non-mirrored) projection.
    """

    pts = np.asarray(quad, dtype=float).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(-angles)]


def estimate_board_pose(
    image_quad: np.ndarray,
    logo_centroid_px: Sequence[float] | None,
    camera: CameraModel,
    base_T_cam: Transform,
    *,
    max_reprojection_error_px: float = 6.0,
    min_ambiguity_ratio: float = 1.5,
    max_logo_error_px: float = 60.0,
    object_corners: np.ndarray | None = None,
    disambiguation_object_point: np.ndarray | None = None,
) -> tuple[BoardPoseEstimate | None, str]:
    """Estimate the board's 6-DoF pose from a planar quad and a disambig point.

    ``image_quad`` are the four detected corners (pixels, any order/winding) of a
    known board-frame rectangle -- by default the board outline
    (``BOARD_OUTLINE_CORNERS``); pass ``object_corners`` to PnP a different one
    (e.g. ``INSIGNIA_RECT_CORNERS`` for the clip-proof insignia).
    ``logo_centroid_px`` is a detected asymmetric point (the purple insignia
    centroid) used purely to break the rectangle's rotation/mirror ambiguity;
    ``disambiguation_object_point`` is its board-frame counterpart (default
    ``LOGO_PLATE_CENTER``).  The full ``base_T_cam`` extrinsic maps the recovered
    ``cam_T_board`` into ``base``.

    Returns ``(estimate, reason)``.  ``estimate`` is ``None`` when the board is
    ambiguous, the reprojection error is too high, or the logo contradicts the
    winner -- i.e. this fails closed rather than fabricating a pose from a
    clipped or degenerate detection.  ``reason`` is a human-readable diagnostic.
    """

    import cv2

    quad = np.asarray(image_quad, dtype=float).reshape(-1, 2)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return None, "board outline quad is not four finite points"

    ordered = _order_quad_ccw(quad).astype(np.float64)
    # A four-point PnP call can return a finite but meaningless pose for a
    # collapsed or self-overlapping detection.  Reject those before invoking
    # the solver.
    signed_area = 0.5 * float(
        np.sum(
            ordered[:, 0] * np.roll(ordered[:, 1], -1)
            - ordered[:, 1] * np.roll(ordered[:, 0], -1)
        )
    )
    if abs(signed_area) < 16.0:
        return None, "board outline quad is degenerate (area below 16 px^2)"

    object_points = (
        BOARD_OUTLINE_CORNERS if object_corners is None else np.asarray(object_corners)
    ).astype(np.float64)
    if object_points.shape != (4, 3) or not np.all(np.isfinite(object_points)):
        return None, "object corners must be four finite board-frame points"
    disambiguation_point = (
        LOGO_PLATE_CENTER
        if disambiguation_object_point is None
        else np.asarray(disambiguation_object_point, dtype=float)
    )
    dist = camera.distortion.astype(np.float64)

    # The detected quad's winding direction is unknown (a downward-looking
    # optical frame mirrors handedness in the image), so try both windings and
    # every cyclic shift: eight candidate correspondences to the CAD corners.
    windings = (ordered, ordered[::-1].copy())
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for winding in windings:
        for shift in range(4):
            image_points = np.roll(winding, shift, axis=0)
            # OpenCV's generic IPPE path has a several-pixel bias for this
            # non-square, offset-Z rectangle at near-fronto-parallel views.
            # ITERATIVE solves the exact four-point planar correspondence and
            # is stable at the reachable tilts.  Trying both windings and every
            # cyclic shift explicitly enumerates the planar orientation
            # hypotheses; the asymmetric magenta material selects the physical
            # one below.
            retval, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera.K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not retval:
                continue
            rmat, _ = cv2.Rodrigues(rvec)
            cam_T_board = Transform(rmat, tvec.reshape(3))
            # Only the board's rendered +Z face carries the purple material.
            # A planar solver can otherwise return a mathematically valid
            # backside/mirrored pose with the same outline reprojection.  The
            # source camera must lie strictly above the CAD top face when
            # expressed in the recovered board frame.
            camera_origin_board = cam_T_board.inverse().translation
            if camera_origin_board[2] <= BOARD_TOP_Z + 1e-3:
                continue
            proj_cam = cam_T_board.apply(object_points)
            pixels, in_front = project_points(proj_cam, camera)
            if not np.all(in_front):
                continue
            err = float(
                np.sqrt(np.mean(np.sum((pixels - image_points) ** 2, axis=1)))
            )
            candidates.append((err, rmat, tvec.reshape(3)))

    if not candidates:
        return None, "planar PnP produced no valid in-front solution"

    candidates.sort(key=lambda item: item[0])
    best_err = candidates[0][0]
    if best_err > max_reprojection_error_px:
        return (
            None,
            f"board reprojection error {best_err:.2f}px exceeds "
            f"{max_reprojection_error_px:.2f}px threshold",
        )

    # The board outline is a non-square rectangle, so a 90-degree corner
    # correspondence reprojects poorly but the 0- and 180-degree ones both fit
    # the outline almost equally.  Every correspondence whose outline error is
    # within the acceptance threshold is a genuine pose hypothesis; the logo is
    # what distinguishes them, so it must *select* among these low-error
    # hypotheses rather than merely validate the numerically smallest.
    hypotheses = [c for c in candidates if c[0] <= max_reprojection_error_px]

    logo_errors: list[float] = []
    if logo_centroid_px is not None:
        logo_target = np.asarray(logo_centroid_px, dtype=float).reshape(-1)
        if logo_target.shape != (2,) or not np.all(np.isfinite(logo_target)):
            return None, "logo centroid is not a finite pixel coordinate"
        for err, rmat, tvec in hypotheses:
            cam_T_board = Transform(rmat, tvec)
            logo_cam = cam_T_board.apply(disambiguation_point)
            logo_pixels, logo_in_front = project_points(logo_cam[None, :], camera)
            if not bool(logo_in_front[0]):
                logo_errors.append(math.inf)
            else:
                logo_errors.append(
                    float(np.linalg.norm(logo_pixels[0] - logo_target))
                )
        order = np.argsort(logo_errors)
        winner_idx = int(order[0])
        best_err, best_rmat, best_tvec = hypotheses[winner_idx]
        logo_error_px = logo_errors[winner_idx]
        if not math.isfinite(logo_error_px) or logo_error_px > max_logo_error_px:
            return (
                None,
                f"logo disambiguation error {logo_error_px:.1f}px exceeds "
                f"{max_logo_error_px:.1f}px; refusing to infer pose from a "
                "clipped or mismatched insignia",
            )
        # Ambiguity here is measured in logo space: two hypotheses whose logo
        # projections both land near the detection cannot be told apart.
        ambiguity_ratio = math.inf
        if len(order) > 1:
            runner_up = logo_errors[int(order[1])]
            ambiguity_ratio = runner_up / max(logo_error_px, 1e-6)
            if ambiguity_ratio < min_ambiguity_ratio:
                return (
                    None,
                    f"logo does not uniquely disambiguate board pose "
                    f"(ratio {ambiguity_ratio:.2f} < "
                    f"{min_ambiguity_ratio:.2f})",
                )
    else:
        # Without a logo, the outline alone must be unambiguous: the runner-up
        # correspondence has to reproject clearly worse than the winner.
        best_err, best_rmat, best_tvec = hypotheses[0]
        logo_error_px = 0.0
        ambiguity_ratio = math.inf
        if len(hypotheses) > 1:
            ambiguity_ratio = hypotheses[1][0] / max(best_err, 1e-6)
        if ambiguity_ratio < min_ambiguity_ratio:
            return (
                None,
                f"board pose is ambiguous (ratio {ambiguity_ratio:.2f} < "
                f"{min_ambiguity_ratio:.2f}) and no logo is available to break "
                "it",
            )

    cam_T_board = Transform(best_rmat, best_tvec)

    base_T_board = base_T_cam.compose(cam_T_board)
    estimate = BoardPoseEstimate(
        base_T_board=base_T_board,
        reprojection_error_px=best_err,
        ambiguity_ratio=ambiguity_ratio,
        logo_error_px=logo_error_px,
        camera_name=camera.name,
    )
    return estimate, "ok"


def estimate_board_pose_from_insignia(
    insignia_quad: np.ndarray,
    insignia_centroid_px: Sequence[float],
    camera: CameraModel,
    base_T_cam: Transform,
    *,
    max_reprojection_error_px: float = 8.0,
    min_ambiguity_ratio: float = 1.2,
    max_logo_error_px: float = 40.0,
) -> tuple[BoardPoseEstimate | None, str]:
    """Estimate the board pose from the insignia bracket alone (clip-proof).

    PnPs the detected insignia bounding-rectangle corners against
    ``INSIGNIA_RECT_CORNERS`` and uses the asymmetric mask centroid to resolve
    the rectangle ambiguity.  This is the primary Stage-2 pose source because the
    large insignia stays fully in frame at survey standoffs where the plate
    outline clips.  The tolerances are looser than the outline default because a
    ~9.5 cm marker reprojects a little less tightly than the full plate, and the
    centroid must select among four near-square rotations.
    """

    return estimate_board_pose(
        insignia_quad,
        insignia_centroid_px,
        camera,
        base_T_cam,
        max_reprojection_error_px=max_reprojection_error_px,
        min_ambiguity_ratio=min_ambiguity_ratio,
        max_logo_error_px=max_logo_error_px,
        object_corners=INSIGNIA_RECT_CORNERS,
        disambiguation_object_point=INSIGNIA_CENTROID,
    )


# ---------------------------------------------------------------------------
# Multi-camera survey-pose scoring and search.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperExclusion:
    """A calibrated image-space keep-out for a camera's own gripper.

    ``mask`` preserves the actual self-occlusion silhouette.  ``bbox`` remains
    available for callers/tests that only have a conservative rectangle.  If
    both are supplied, their union is enforced.  A projected target is treated
    as its filled convex hull, so robot pixels between projected 3-D vertices
    cannot slip through an edge-only test.
    """

    bbox: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1
    margin_px: float = 32.0
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.margin_px < 0.0 or not math.isfinite(self.margin_px):
            raise ValueError("gripper exclusion margin must be finite and non-negative")
        if self.bbox is not None:
            bbox = tuple(float(v) for v in self.bbox)
            if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
                raise ValueError("gripper exclusion bbox must contain four finite values")
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                raise ValueError("gripper exclusion bbox has inverted bounds")
            object.__setattr__(self, "bbox", bbox)
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.ndim != 2 or min(mask.shape) < 2:
                raise ValueError("gripper exclusion mask must be a 2-D image")
            # Own the pixels so a caller cannot mutate terminal safety after
            # construction.
            mask = mask.copy()
            mask.setflags(write=False)
            object.__setattr__(self, "mask", mask)
            # This is invariant for a calibrated gripper state.  Computing it
            # once avoids an O(image-area) distance transform for every pose
            # candidate in the deterministic multi-camera grid.
            import cv2

            distance = cv2.distanceTransform(
                (~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            distance.setflags(write=False)
            object.__setattr__(self, "_distance_to_mask", distance)

    def clearance_to(self, points_px: np.ndarray) -> float:
        """Signed clearance from a filled projected hull to robot keep-outs.

        Positive means every target pixel is more than ``margin_px`` from robot
        pixels.  Negative means overlap or inadequate safety clearance.
        """

        pts = np.atleast_2d(np.asarray(points_px, dtype=float))
        if pts.shape[1:] != (2,) or len(pts) < 1 or not np.all(np.isfinite(pts)):
            return -math.inf

        clearances: list[float] = []
        if self.bbox is not None:
            clearances.append(self._bbox_clearance(pts))
        if self.mask is not None:
            clearances.append(self._mask_clearance(pts))
        return min(clearances) if clearances else math.inf

    def _bbox_clearance(self, pts: np.ndarray) -> float:
        px0, py0 = pts[:, 0].min(), pts[:, 1].min()
        px1, py1 = pts[:, 0].max(), pts[:, 1].max()
        assert self.bbox is not None
        gx0, gy0, gx1, gy1 = self.bbox
        # Separation on each axis (positive => disjoint on that axis).
        sep_x = max(gx0 - px1, px0 - gx1)
        sep_y = max(gy0 - py1, py0 - gy1)
        if sep_x >= 0.0 or sep_y >= 0.0:
            return float(max(sep_x, sep_y) - self.margin_px)
        # Overlapping on both axes: penetration depth (negative).
        return float(max(sep_x, sep_y) - self.margin_px)

    def _mask_clearance(self, pts: np.ndarray) -> float:
        import cv2

        assert self.mask is not None
        if not self.mask.any():
            return math.inf

        height, width = self.mask.shape
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        if len(hull) < 3:
            return -math.inf
        # Rasterize only the projected hull's ROI.  Clipping here is safe
        # because the separate image-boundary gate rejects out-of-frame
        # geometry; it also avoids OpenCV integer overflow for a bad projection.
        hull[:, 0] = np.clip(hull[:, 0], 0, width - 1)
        hull[:, 1] = np.clip(hull[:, 1], 0, height - 1)
        x0 = max(0, int(math.floor(float(hull[:, 0].min()))))
        x1 = min(width - 1, int(math.ceil(float(hull[:, 0].max()))))
        y0 = max(0, int(math.floor(float(hull[:, 1].min()))))
        y1 = min(height - 1, int(math.ceil(float(hull[:, 1].max()))))
        target = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        local_hull = np.rint(hull - np.array([x0, y0])).astype(np.int32)
        cv2.fillConvexPoly(target, local_hull, 1)
        target_pixels = target.astype(bool)
        if not target_pixels.any():
            return -math.inf
        mask_roi = self.mask[y0 : y1 + 1, x0 : x1 + 1]
        if np.any(target_pixels & mask_roi):
            # A definite overlap is always unsafe, independently of the chosen
            # positive-clearance margin.
            return -max(1.0, self.margin_px)

        # The cached distance transform reports Euclidean distance to the
        # nearest robot pixel.  Sample only the target ROI.
        distances = getattr(self, "_distance_to_mask")
        return float(distances[y0 : y1 + 1, x0 : x1 + 1][target_pixels].min() - self.margin_px)


@dataclass(frozen=True)
class CameraCoverage:
    """Per-camera projection quality of the loose-SFP envelope."""

    camera_name: str
    feasible: bool
    boundary_margin_px: float
    gripper_clearance_px: float
    pixel_scale: float
    module_pixel_scale: float = 0.0
    reasons: tuple[str, ...] = ()

    @property
    def clearance(self) -> float:
        """The worst (minimum) of boundary and gripper clearance."""
        return min(self.boundary_margin_px, self.gripper_clearance_px)


def evaluate_camera_coverage(
    envelope_board: np.ndarray,
    board_T_cam: Transform | None,
    cam_from_board: Transform,
    camera: CameraModel,
    gripper: GripperExclusion,
    *,
    edge_margin_px: float = 12.0,
    min_pixel_scale: float = 0.05,
    module_envelopes_board: Sequence[np.ndarray] | None = None,
    min_module_pixel_scale: float = 0.012,
) -> CameraCoverage:
    """Project the board-frame envelope into one camera and grade it.

    ``cam_from_board`` maps board-frame points into this camera's optical frame.
    """

    reasons: list[str] = []
    envelope_cam = cam_from_board.apply(envelope_board)
    pixels, in_front = project_points(envelope_cam, camera)

    if not np.all(in_front):
        reasons.append("envelope_behind_camera")
        return CameraCoverage(camera.name, False, -math.inf, -math.inf, 0.0, 0.0, tuple(reasons))

    u = pixels[:, 0]
    v = pixels[:, 1]
    # Distance from every corner to the nearest image edge (positive = inside).
    boundary = min(
        float(u.min()),
        float(v.min()),
        float(camera.width - 1 - u.max()),
        float(camera.height - 1 - v.max()),
    )
    boundary_margin = boundary - edge_margin_px
    if boundary_margin < 0.0:
        reasons.append("envelope_outside_image")

    if gripper.mask is not None and gripper.mask.shape != (camera.height, camera.width):
        gripper_clearance = -math.inf
        reasons.append("gripper_mask_shape_mismatch")
    else:
        gripper_clearance = gripper.clearance_to(pixels)
        if gripper_clearance < 0.0:
            reasons.append("envelope_intersects_gripper")

    # Pixel scale: fraction of the image spanned by the projected envelope's
    # larger side.  Guards against a distant, unusably small view.
    span = max(float(u.max() - u.min()), float(v.max() - v.min()))
    pixel_scale = span / float(max(camera.width, camera.height))
    if pixel_scale < min_pixel_scale:
        reasons.append("envelope_too_small")

    detail_boxes = (
        tuple(module_envelopes_board)
        if module_envelopes_board is not None
        else sfp_module_detail_boxes()
    )
    module_scales: list[float] = []
    for box in detail_boxes:
        box_cam = cam_from_board.apply(np.asarray(box, dtype=float))
        box_px, box_front = project_points(box_cam, camera)
        if not np.all(box_front) or not np.all(np.isfinite(box_px)):
            module_scales.append(0.0)
            continue
        # The smaller projected side is what protects the visually thin SFP
        # body from being accepted merely because it is long in the image.
        span_u = float(box_px[:, 0].max() - box_px[:, 0].min())
        span_v = float(box_px[:, 1].max() - box_px[:, 1].min())
        module_scales.append(min(span_u, span_v) / float(max(camera.width, camera.height)))
    module_pixel_scale = min(module_scales) if module_scales else 0.0
    if module_pixel_scale < min_module_pixel_scale:
        reasons.append("module_too_small")

    feasible = not reasons
    return CameraCoverage(
        camera.name,
        feasible,
        boundary_margin,
        gripper_clearance,
        pixel_scale,
        module_pixel_scale,
        tuple(reasons),
    )


@dataclass(frozen=True)
class SurveyCandidate:
    """A scored board-relative TCP survey pose."""

    base_T_tcp: Transform
    min_clearance_px: float
    coverages: tuple[CameraCoverage, ...]
    standoff_m: float
    yaw_rad: float
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    min_module_pixel_scale: float = 0.0
    motion_m: float = math.inf
    angular_motion_rad: float = math.inf
    # The board-frame corner set this pose was found to frame in all cameras
    # (whole board or the module region); the post-move confirm re-checks it.
    coverage_target: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def feasible(self) -> bool:
        return bool(self.coverages) and all(c.feasible for c in self.coverages)


def _look_at_rotation(origin: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build base<-camera rotation with optical +Z aimed at ``target``.

    This is a true oblique view whenever the candidate origin has a board-X or
    board-Y offset.  ``yaw_rad`` only controls independent in-plane roll; it
    no longer masquerades as an oblique viewing search.
    """

    z_axis = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    if np.linalg.norm(z_axis) < 1e-6:
        raise ValueError("camera origin and target coincide")
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        up_hint = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest SO(3) angle between two rotation matrices."""
    delta = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    return math.acos(
        float(np.clip(0.5 * (np.trace(delta) - 1.0), -1.0, 1.0))
    )


def sampled_cartesian_path_is_safe(
    start: np.ndarray,
    end: np.ndarray,
    *,
    board_origin: np.ndarray,
    board_normal: np.ndarray,
    minimum_clearance: float,
    allow_outward_retreat: bool = False,
    samples: int = 13,
    minimum_height_m: float = 0.02,
    maximum_reach_m: float = 1.20,
) -> bool:
    """Validate a straight segment against conservative workspace guards.

    An outward retreat may start below ``minimum_clearance`` only when every
    sample moves monotonically away from the board and the endpoint reaches the
    requested clearance. This lets Stage 2 escape a close but valid Stage-1
    handoff without weakening the later lateral-transit guard.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    origin = np.asarray(board_origin, dtype=float)
    normal = np.asarray(board_normal, dtype=float)
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or origin.shape != (3,)
        or normal.shape != (3,)
        or not np.all(np.isfinite(np.concatenate((start, end, origin, normal))))
        or samples < 2
        or minimum_clearance < 0.0
    ):
        return False
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return False
    normal = normal / normal_norm
    start_clearance = float(np.dot(start - origin, normal))
    end_clearance = float(np.dot(end - origin, normal))
    if allow_outward_retreat:
        if end_clearance + 1e-9 < minimum_clearance:
            return False
        previous_clearance = start_clearance
    else:
        previous_clearance = math.nan

    for fraction in np.linspace(0.0, 1.0, samples):
        point = start + float(fraction) * (end - start)
        clearance = float(np.dot(point - origin, normal))
        if (
            not np.all(np.isfinite(point))
            or float(point[2]) < minimum_height_m
            or float(np.linalg.norm(point)) > maximum_reach_m
        ):
            return False
        if allow_outward_retreat:
            if clearance + 1e-9 < previous_clearance:
                return False
            previous_clearance = clearance
        elif clearance + 1e-9 < minimum_clearance:
            return False
    return True


def search_survey_pose(
    board_pose: BoardPoseEstimate,
    tcp_T_cam: Mapping[str, Transform],
    cameras: Mapping[str, CameraModel],
    grippers: Mapping[str, GripperExclusion],
    *,
    reference_camera: str = "center_camera",
    standoffs_m: Sequence[float] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25),
    yaws_rad: Sequence[float] | None = None,
    lateral_offsets_m: Sequence[float] | None = None,
    offsets_x_m: Sequence[float] = (-0.12, -0.06, 0.0, 0.06, 0.12),
    offsets_y_m: Sequence[float] = (-0.12, -0.06, 0.0, 0.06, 0.12),
    coverage_targets: Sequence[np.ndarray] | None = None,
    module_envelopes_board: Sequence[np.ndarray] | None = None,
    current_base_T_tcp: Transform | None = None,
    max_angular_motion_rad: float = math.radians(45.0),
    # The actual wrist-camera chain adds 223 mm above the TCP.  A 1.2 m
    # base-origin sphere incorrectly rejects safe wide-frame survey poses for
    # the 425 mm board; integration still applies its controller workspace and
    # conservative workspace and swept-orientation guards before executing
    # this geometric proposal.
    max_reach_m: float = 1.8,
    min_height_m: float = 0.02,
    edge_margin_px: float = 12.0,
) -> tuple[SurveyCandidate | None, str]:
    """Deterministically search for one board-relative TCP survey pose.

    ``coverage_targets`` are board-frame corner sets tried in order (most
    preferred first; default: the whole board face, then the SFP/SC module
    region).  For each target every candidate camera pose projects the target
    through all supplied cameras; a candidate is feasible only when every camera
    has the target in front, fully inside the image with a positive boundary
    margin, and clear of the gripper keep-out with a positive margin.  The first
    target that yields any feasible candidate wins; among those the pose that
    maximises the minimum clearance across the three cameras is returned, with
    the chosen board-frame target attached as ``candidate.coverage_target``.  So
    the returned pose frames the modules in all three cameras at minimum, and the
    whole board when reachable.  The distant-scale and per-module detail gates are
    intentionally not applied here.  Returns ``(None, reason)`` when no candidate
    is feasible or reachable for any target.
    """

    if reference_camera not in tcp_T_cam:
        return None, f"reference camera {reference_camera!r} has no extrinsic"
    if (
        not math.isfinite(max_angular_motion_rad)
        or max_angular_motion_rad < 0.0
    ):
        return None, "maximum angular motion must be finite and non-negative"
    required = set(cameras)
    missing_extrinsics = sorted(required.difference(tcp_T_cam))
    missing_grippers = sorted(required.difference(grippers))
    if missing_extrinsics:
        return None, f"missing camera extrinsics: {missing_extrinsics}"
    if missing_grippers:
        return None, f"missing gripper exclusions: {missing_grippers}"
    if yaws_rad is None:
        yaws_rad = tuple(np.deg2rad([-180.0, -90.0, -30.0, 0.0, 30.0, 90.0, 180.0]))
    # Historical single-axis callers retain their intent while new callers get
    # the complete board-plane search.
    if lateral_offsets_m is not None:
        offsets_x_m = lateral_offsets_m
    if coverage_targets is None:
        coverage_targets = (board_coverage_corners(), module_coverage_corners())
    coverage_targets = tuple(
        np.asarray(target, dtype=float) for target in coverage_targets
    )
    if not coverage_targets:
        return None, "no coverage targets supplied"

    base_T_board = board_pose.base_T_board
    board_normal_base = base_T_board.rotation[:, 2]
    board_normal_base = board_normal_base / np.linalg.norm(board_normal_base)
    # In-plane basis for lateral offsets and the up hint (board X/Y in base).
    board_x_base = base_T_board.rotation[:, 0]
    board_y_base = base_T_board.rotation[:, 1]
    ref_tcp_T_cam = tcp_T_cam[reference_camera]

    def _best_for_target(
        target_board: np.ndarray,
    ) -> tuple[SurveyCandidate | None, int]:
        # Aim the reference optical axis at the target centroid and search a
        # board-relative 3-D grid, giving real pitch/roll variation rather than a
        # top-down pose with image roll only.
        center_base = base_T_board.apply(target_board.mean(axis=0))
        best: SurveyCandidate | None = None
        evaluated = 0
        for standoff in standoffs_m:
            for offset_x in offsets_x_m:
                for offset_y in offsets_y_m:
                    for yaw in yaws_rad:
                        cam_origin = (
                            center_base
                            + board_normal_base * standoff
                            + board_x_base * offset_x
                            + board_y_base * offset_y
                        )
                        if cam_origin[2] < min_height_m:
                            continue
                        up_hint = (
                            math.cos(yaw) * board_y_base
                            + math.sin(yaw) * board_x_base
                        )
                        cam_rot = _look_at_rotation(
                            cam_origin, center_base, up_hint
                        )
                        base_T_refcam = Transform(cam_rot, cam_origin)
                        # Recover the TCP pose that realises this camera pose.
                        base_T_tcp = base_T_refcam.compose(ref_tcp_T_cam.inverse())
                        # Checking only the camera origin can admit a pose whose
                        # camera is above the board while its real TCP is below
                        # the allowed plane; guard the TCP itself.
                        if (
                            float(base_T_tcp.translation[2]) < min_height_m
                            or float(np.linalg.norm(base_T_tcp.translation))
                            > max_reach_m
                        ):
                            continue
                        if not np.all(np.isfinite(base_T_tcp.rotation)):
                            continue
                        angular_motion = (
                            _rotation_distance_rad(
                                current_base_T_tcp.rotation,
                                base_T_tcp.rotation,
                            )
                            if current_base_T_tcp is not None
                            else 0.0
                        )
                        if angular_motion > max_angular_motion_rad:
                            continue

                        coverages = []
                        for name, camera in cameras.items():
                            base_T_camera = base_T_tcp.compose(tcp_T_cam[name])
                            cam_from_board = (
                                base_T_camera.inverse().compose(base_T_board)
                            )
                            coverages.append(
                                evaluate_camera_coverage(
                                    target_board,
                                    None,
                                    cam_from_board,
                                    camera,
                                    grippers[name],
                                    edge_margin_px=edge_margin_px,
                                    # Only "framed and gripper-clear in every
                                    # camera" gates here; disable the distant-scale
                                    # and per-module detail checks.
                                    min_pixel_scale=0.0,
                                    module_envelopes_board=(),
                                    min_module_pixel_scale=0.0,
                                )
                            )
                        evaluated += 1
                        if not coverages or not all(
                            c.feasible for c in coverages
                        ):
                            continue
                        min_clear = min(c.clearance for c in coverages)
                        candidate = SurveyCandidate(
                            base_T_tcp=base_T_tcp,
                            min_clearance_px=min_clear,
                            coverages=tuple(coverages),
                            standoff_m=standoff,
                            yaw_rad=yaw,
                            offset_x_m=float(offset_x),
                            offset_y_m=float(offset_y),
                            motion_m=(
                                float(
                                    np.linalg.norm(
                                        base_T_tcp.translation
                                        - current_base_T_tcp.translation
                                    )
                                )
                                if current_base_T_tcp is not None
                                else float(
                                    np.linalg.norm(base_T_tcp.translation)
                                )
                            ),
                            angular_motion_rad=angular_motion,
                            coverage_target=target_board,
                        )
                        # Lexicographic, deterministic optimisation: prefer the
                        # *closest* feasible pose (least motion), so the survey
                        # frames the modules at the smallest necessary standoff
                        # -- bigger modules in frame and a shorter, safer move --
                        # then break ties by clearance and by angular motion.
                        candidate_key = (
                            -round(candidate.motion_m, 4),
                            round(candidate.min_clearance_px, 6),
                            -candidate.angular_motion_rad,
                        )
                        best_key = (
                            (
                                -round(best.motion_m, 4),
                                round(best.min_clearance_px, 6),
                                -best.angular_motion_rad,
                            )
                            if best is not None
                            else None
                        )
                        if best is None or candidate_key > best_key:
                            best = candidate
        return best, evaluated

    total_evaluated = 0
    for target_board in coverage_targets:
        best, evaluated = _best_for_target(target_board)
        total_evaluated += evaluated
        if best is not None:
            return best, "ok"
    return (
        None,
        "no feasible survey candidate satisfied all cameras for any coverage "
        f"target ({total_evaluated} candidates evaluated)",
    )


# ---------------------------------------------------------------------------
# Post-move verification.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying the survey view in every camera."""

    passed: bool
    coverages: tuple[CameraCoverage, ...]
    skew_ok: bool
    reason: str


def verify_survey_view(
    board_pose: BoardPoseEstimate,
    base_T_tcp: Transform,
    tcp_T_cam: Mapping[str, Transform],
    cameras: Mapping[str, CameraModel],
    grippers: Mapping[str, GripperExclusion],
    stamps_ns: Mapping[str, int],
    *,
    max_skew_ns: int = 50_000_000,
    coverage_target: np.ndarray | None = None,
    edge_margin_px: float = 12.0,
) -> VerificationResult:
    """Verify the chosen coverage target is framed in all cameras after settling.

    Requires the fresh frames to be within ``max_skew_ns`` of each other and the
    projected ``coverage_target`` (default: the module region) to be in front,
    inside the image with a positive boundary margin, and gripper-clear in every
    camera -- the same single acceptance ``search_survey_pose`` used to pick the
    pose.  The distant-scale and per-module detail gates are intentionally off.
    """

    if len(stamps_ns) < len(cameras):
        return VerificationResult(
            False, (), False, "missing fresh frame for at least one camera"
        )
    stamp_values = [int(stamps_ns[name]) for name in cameras if name in stamps_ns]
    skew_ok = (max(stamp_values) - min(stamp_values)) <= max_skew_ns if stamp_values else False
    if not skew_ok:
        return VerificationResult(
            False,
            (),
            False,
            f"three-camera timestamp skew exceeds {max_skew_ns} ns",
        )

    base_T_board = board_pose.base_T_board
    envelope_board = (
        module_coverage_corners()
        if coverage_target is None
        else np.asarray(coverage_target, dtype=float)
    )
    coverages = []
    for name, camera in cameras.items():
        if name not in tcp_T_cam:
            return VerificationResult(
                False, tuple(coverages), skew_ok, f"missing extrinsic for {name}"
            )
        base_T_camera = base_T_tcp.compose(tcp_T_cam[name])
        cam_from_board = base_T_camera.inverse().compose(base_T_board)
        gripper = grippers.get(name, GripperExclusion(None))
        coverages.append(
            evaluate_camera_coverage(
                envelope_board,
                None,
                cam_from_board,
                camera,
                gripper,
                edge_margin_px=edge_margin_px,
                min_pixel_scale=0.0,
                module_envelopes_board=(),
                min_module_pixel_scale=0.0,
            )
        )
    passed = bool(coverages) and all(c.feasible for c in coverages)
    reason = "ok" if passed else "one or more cameras failed envelope verification"
    return VerificationResult(passed, tuple(coverages), skew_ok, reason)


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return the ``(x0, y0, x1, y1)`` bounding box of a boolean mask, or None."""
    m = np.asarray(mask)
    if m.ndim != 2 or not m.any():
        return None
    ys, xs = np.nonzero(m)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
