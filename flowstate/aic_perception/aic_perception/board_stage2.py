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

from dataclasses import dataclass, field, replace
import math
from typing import Callable, Mapping, Sequence

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
    the box adds the SFP body half-extent.

    **Superseded as the ``STAGED_SFP_MODULE`` survey target.**  This box covers
    one rail only.  Staged modules occupy five of *six* legal seats spread over
    **both** rails (board Y -0.15625 .. +0.15625, see
    ``sfp_module_detail_boxes``), so framing this box guarantees nothing about
    the -Y rail: the search would push to its closest standoff with the +Y half
    just fitting and the outer -Y seat cropped.  That is exactly the observed
    hardware failure -- four modules returned at board Y -0.1066, -0.0565,
    +0.1066, +0.1566 with the fifth seat at -0.15625 outside the frame.  Kept
    for the regression sweep; use ``sfp_module_strip_corners``.
    """
    return _sector_box_corners((0.02, 0.09), (0.0, 0.225), (0.01, 0.06))


# Board-X strip that contains the staged modules.  Covers the CAD mount origins
# (SFP_RAIL_X 0.055 +/- the 0.025 body half-extent = 0.030 .. 0.080) *and* the
# detected module bodies, which sit at board X ~0.0862 because the transceiver
# protrudes from its mount origin -- only 3.8 mm inside the superseded sector's
# 0.09 edge, and 28 mm inside this one.
#
# Measured: board-X width is not what costs standoff.  85 mm, 65 mm and 50 mm
# boxes give bit-identical sweep results; only board Y moves the frontier.  So
# this is sized for margin, not for tightness.
SFP_SPAN_X = (0.030, 0.115)
SFP_SPAN_Z = (0.01, SFP_ENVELOPE_Z_MAX)
# Outermost legal seat centre on either rail, padded by the module body
# half-extent along Y (both from ``sfp_module_detail_boxes``).
SFP_SEAT_Y_ABS = 0.15625
SFP_SEAT_HALF_Y = 0.022


# Board-Y half-span of the staged-SFP coverage box.  Deliberately the *same*
# 0.225 m extent as the superseded one-rail sector -- only its placement moves.
#
# It does not itself contain the outermost seats (+/-0.15625).  It does not need
# to: the box sets what the survey aims at and how far it stands off, and the
# resulting view then holds every seat with 119-158 px of image margin, measured
# over the full 144-case board/placement/live-start sweep.  Growing it past this
# is a bad trade -- it buys ~30 px of seat margin nobody needs and pushes the
# selected standoff from 0.64 m out to 0.85-0.90 m, shrinking every module in
# the image.  See ``test/sfp_sweep_runner.py`` for the frontier:
#
#   y_half   found/144   all-5 framed   standoff
#   0.1125      92           92         0.64-0.85
#   0.1450      58           58         0.80-0.85
#   0.1600      35           35         0.85-0.90
#   0.1783       0            -         infeasible
SFP_COVERAGE_HALF_Y = 0.1125


def sfp_module_strip_corners() -> np.ndarray:
    """The staged-SFP coverage target: the module strip, centred (board frame).

    This is the ``STAGED_SFP_MODULE`` survey target.  It has the same size as
    the superseded ``sfp_sector_corners`` and sits in a different place, and
    that placement is the entire fix.

    The old box covered the +Y rail alone (Y 0.0 .. 0.225), so the point the
    survey aimed at sat 112.5 mm off the middle of the staged modules.  The
    modules run Y -0.15625 .. +0.15625 across both rails, so every bit of the
    search's framing slack was banked on the +Y side -- one sweep case puts the
    -Y end 57 px outside the image while the +Y end carries 318-387 px of
    margin.  That asymmetry is the 4-of-5 hardware failure.

    Straddling Y=0 spreads the same slack evenly over both ends of the strip.
    Measured over the 144-case sweep at identical search settings: the old box
    clips a module in **96 of its 96** found poses (35 of them showing only
    four of the six seats, worst seat 123.9 px outside frame); this one frames
    every module in **all 92** of its found poses, with 118.5 px to spare.
    """
    return _sector_box_corners(
        SFP_SPAN_X,
        (-SFP_COVERAGE_HALF_Y, SFP_COVERAGE_HALF_Y),
        SFP_SPAN_Z,
    )


def sc_sector_corners() -> np.ndarray:
    """SC optical ports (Zone 2): the five adapter bores themselves.

    The board carries **five** SC adapters in two rows (``task_board.urdf.xacro``):
    ``sc_port_0/1/2`` on SC_RAIL_0 at board Y +0.0295 and ``sc_port_3/4`` on
    SC_RAIL_1 at board Y +0.0705, each at board X ``-0.075 + t`` for a rail
    translation ``t`` in -0.060..+0.055, so the cluster spans X -0.135..-0.020.

    Like the NIC cages, each adapter is a recess that opens **straight up** --
    bore axis ``(0, 0, -1)`` in board frame, 0.00 deg off the board normal, with
    the entrance at board Z 0.0301 and the bore running 15.64 mm down.  The
    receptacle opening, taken from the adapter's own collision primitives (side
    walls at local |x| 12.05 mm x 1.69 mm thick, plates at local z +/-4.2 mm), is
    **7.6 x 22.4 mm**, so the limiting cone is ``atan(3.8/15.64) = 13.7 deg``
    across the narrow axis (35.6 deg across the wide one).  That is roughly twice
    as forgiving as the NIC cage band (7.5 deg), and the ports sit 149 mm lower
    on the board, which is why this sector is far easier to reach than NIC.

    The box is centred on the five **entrances** rather than the adapter bodies,
    for the same reason as ``nic_sector_corners``: the search aims the optical
    axis at this box's centroid.  The previous box (X -0.14..-0.01,
    Y -0.02..0.10, Z 0.01..0.05) predated the 3-port row and swept in ~47 mm of
    empty board on the -Y side, pulling the aim point 10 mm off the cluster.

    The ``SC_DESTINATION_PORT`` survey target.
    """
    return _sector_box_corners((-0.152, -0.003), (0.005, 0.095), (0.020, 0.035))


def sc_bore_sample_points() -> np.ndarray:
    """Board-frame SC mouth centres covering every allowed rail placement.

    Each adapter translates independently along board X in -0.135..-0.020 m,
    on one of the two fixed rows at board Y +0.0295 / +0.0705 m.  Sampling both
    ends and the midpoint of both rows is conservative for the view-direction
    constraint: if these six extrema can see down the bore, any five-port rail
    realization between them can too.
    """

    return np.array(
        [
            (x, y, 0.0301)
            for x in (-0.135, -0.0775, -0.020)
            for y in (0.0295, 0.0705)
        ],
        dtype=float,
    )


def rectangular_bore_visibility_margin(
    bore_points_board: np.ndarray,
    base_T_board: "Transform",
    base_T_tcp: "Transform",
    tcp_T_cam: Mapping[str, "Transform"],
    *,
    half_width_x_m: float,
    half_width_y_m: float,
    depth_m: float,
    camera_names: Sequence[str] | None = None,
    required_camera_count: int | None = None,
) -> float:
    """Worst normalized line-of-sight margin through rectangular bores.

    The bores open along board +Z and extend ``depth_m`` into the board.  For a
    camera ray from a mouth centre, the back-plane displacement is
    ``depth * (dx/dz, dy/dz)``. A non-negative result means that displacement
    fits inside both aperture half-widths for every sampled mouth in at least
    ``required_camera_count`` cameras. By default every named camera is
    required, preserving the original strict behavior.

    This is deliberately based on camera *origins*, not optical axes.  The
    three wrist cameras are spatially separated, so a sector can be framed in
    every image while a side camera still looks across the SC adapter's narrow
    7.6 mm mouth and cannot see its depth.
    """

    points = np.asarray(bore_points_board, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return -math.inf
    if (
        not math.isfinite(half_width_x_m)
        or not math.isfinite(half_width_y_m)
        or not math.isfinite(depth_m)
        or half_width_x_m <= 0.0
        or half_width_y_m <= 0.0
        or depth_m <= 0.0
    ):
        return -math.inf

    required = tuple(camera_names) if camera_names is not None else tuple(tcp_T_cam)
    camera_count = (
        len(required)
        if required_camera_count is None
        else int(required_camera_count)
    )
    if (
        not required
        or any(name not in tcp_T_cam for name in required)
        or camera_count < 1
        or camera_count > len(required)
    ):
        return -math.inf

    board_T_base = base_T_board.inverse()
    margins = []
    x_limit = half_width_x_m / depth_m
    y_limit = half_width_y_m / depth_m
    for name in required:
        camera_board = board_T_base.apply(
            base_T_tcp.compose(tcp_T_cam[name]).translation
        )
        rays = camera_board - points
        if np.any(rays[:, 2] <= 0.0) or not np.all(np.isfinite(rays)):
            return -math.inf
        x_ratio = np.abs(rays[:, 0] / rays[:, 2]) / x_limit
        y_ratio = np.abs(rays[:, 1] / rays[:, 2]) / y_limit
        margins.append(1.0 - np.maximum(x_ratio, y_ratio))
    ranked = np.sort(np.asarray(margins, dtype=float), axis=0)[::-1]
    return float(np.min(ranked[camera_count - 1]))


def rectangular_bore_depth_cue_px(
    bore_points_board: np.ndarray,
    base_T_board: "Transform",
    base_T_tcp: "Transform",
    tcp_T_cam: Mapping[str, "Transform"],
    cameras: Mapping[str, "CameraModel"],
    *,
    depth_m: float,
    camera_names: Sequence[str] | None = None,
    required_camera_count: int | None = None,
) -> float:
    """Worst projected mouth-to-back-center displacement, in pixels.

    A bore can be physically open yet look like a nearly symmetric flat rim to
    a pose model. Projecting the front mouth centre and corresponding bore
    back-centre measures the perspective/depth cue visible in the SC images:
    zero is head-on, while a few pixels exposes a displaced dark interior. The
    minimum spans every conservative mouth sample and the kth-best camera
    requested by ``required_camera_count``. By default every named camera is
    required. Fused perception can instead require two strong views while all
    three retain independent framing and gripper-clearance gates.
    """

    points = np.asarray(bore_points_board, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return -math.inf
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        return -math.inf
    required = tuple(camera_names) if camera_names is not None else tuple(cameras)
    camera_count = (
        len(required)
        if required_camera_count is None
        else int(required_camera_count)
    )
    if (
        not required
        or any(name not in tcp_T_cam or name not in cameras for name in required)
        or camera_count < 1
        or camera_count > len(required)
    ):
        return -math.inf

    back = points.copy()
    back[:, 2] -= depth_m
    base_front = base_T_board.apply(points)
    base_back = base_T_board.apply(back)
    cues = []
    for name in required:
        camera_T_base = base_T_tcp.compose(tcp_T_cam[name]).inverse()
        front_px, front_ok = project_points(
            camera_T_base.apply(base_front), cameras[name]
        )
        back_px, back_ok = project_points(
            camera_T_base.apply(base_back), cameras[name]
        )
        if (
            not np.all(front_ok)
            or not np.all(back_ok)
            or not np.all(np.isfinite(front_px))
            or not np.all(np.isfinite(back_px))
        ):
            return -math.inf
        cues.append(np.linalg.norm(back_px - front_px, axis=1))
    ranked = np.sort(np.asarray(cues, dtype=float), axis=0)[::-1]
    return float(np.min(ranked[camera_count - 1]))


def nic_sector_corners() -> np.ndarray:
    """NIC card SFP-port destinations (Zone 1): the ten port bores themselves.

    The box is centred on the **port entrances**, not on the card bodies, because
    the search aims the optical axis at this box's centroid and what has to be
    read is the bore.  Each port is a 16 x 12 mm aperture at the top of a 45.8 mm
    recess whose axis points straight up (0.7 deg off the board normal, from
    ``aic_world.xml``), so a port only shows the black depth the IVM keys on to a
    ray within ``atan(6/45.8) = 7.5 deg`` of that axis.  The ten entrances (five
    cards x two ports) sit at board Z 0.1793 spanning X -0.100..-0.077 and
    Y -0.186..-0.026; this box pads that laterally by 12 mm and drops to Z 0.125
    so the cage bodies stay framed with their ports.

    The previous box was centred on the card band (X -0.14..-0.03, Y -0.19..0.01,
    Z 0.07..0.17). Its centroid sat 16 mm off the port cluster, which aimed the
    camera off-centre and pushed the outermost port past the 7.5 deg cone -- worth
    two of the ten ports.  The ``NIC_SFP_DESTINATION`` survey target.
    """
    return _sector_box_corners((-0.1124, -0.0652), (-0.1978, -0.0138), (0.125, 0.185))


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
    required_clearance_px: float = 0.0,
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
    if (
        required_clearance_px > 0.0
        and boundary_margin < required_clearance_px
    ):
        if boundary_margin >= 0.0:
            reasons.append("insufficient_boundary_clearance")
        # The signed image-boundary clearance is already below the caller's
        # hard floor.  Gripper-mask clearance cannot rescue this camera, and is
        # the expensive part of the deterministic survey grid, so fail before
        # rasterizing the projected hull against the full-resolution mask.
        return CameraCoverage(
            camera.name,
            False,
            boundary_margin,
            math.inf,
            0.0,
            0.0,
            tuple(reasons),
        )

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
    # Optional target-specific image-formation margin.  Recessed ports use this
    # to distinguish "mouth is in frame" from "the camera can see down it".
    view_quality: float = math.inf
    # Reference-camera view direction resolved in the sector's rail basis.
    # These are diagnostics and regression-test hooks for directional views.
    cross_rail_tilt_rad: float = 0.0
    along_rail_tilt_rad: float = 0.0
    motion_m: float = math.inf
    angular_motion_rad: float = math.inf
    # Analytic-IK estimate from the live joint state.  These are physical,
    # unwrapped joint deltas, not modulo-2pi pose differences.
    max_joint_motion_rad: float = math.inf
    total_joint_motion_rad: float = math.inf
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
    # Finer steps at the near end: downstream pose estimation wants the closest
    # standoff that still frames the sector, not merely a feasible one.
    standoffs_m: Sequence[float] = (
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.58,
        0.60,
        0.62,
        0.64,
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
    yaws_rad: Sequence[float] | None = None,
    lateral_offsets_m: Sequence[float] | None = None,
    # Small board-plane offsets only.  Large offsets produce raking views that
    # foreshorten and self-occlude tall components (NIC cards, SC ports).
    offsets_x_m: Sequence[float] = (-0.06, -0.03, 0.0, 0.03, 0.06),
    offsets_y_m: Sequence[float] = (-0.06, -0.03, 0.0, 0.03, 0.06),
    # Board-frame shift of the point the reference camera looks at. Coverage is
    # still evaluated against the unchanged target envelope. SC can therefore
    # bias the view toward the extreme three-port-rail mouth without pretending
    # the other four ports no longer need to be framed.
    aim_offset_board_m: Sequence[float] = (0.0, 0.0, 0.0),
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
    # Keep the reference optical axis near the board normal.  A raking view
    # collapses the along-rail spacing of tall parts, which is what stops the
    # pose estimator separating adjacent NIC cards / SC ports.
    max_obliquity_rad: float = math.radians(20.0),
    # Do not accept a pose whose sector only just fits; leave real margin so a
    # small board-pose or execution error cannot clip a component out of a
    # camera.  At the survey standoffs this is roughly 30 mm of slack.
    min_required_clearance_px: float = 40.0,
    # Directional (rail-aware) obliquity for sectors of thin, repeated parts
    # (NIC cards, SC ports) whose model-based pose is ill-conditioned when viewed
    # straight down the part's protrusion axis.  When a (min, max) band is given
    # the search tilts the view *across* the sector's long (rail) axis by an
    # angle inside the band, holding the *along-rail* tilt within
    # ``max_along_rail_tilt_rad`` of zero, so each part's depth is revealed
    # without the neighbours occluding it.  ``None`` keeps the isotropic
    # ``max_obliquity_rad`` behaviour used for the SFP and full-board targets.
    cross_rail_tilt_band_rad: tuple[float, float] | None = None,
    max_along_rail_tilt_rad: float = math.radians(8.0),
    # Optional explicit board-frame in-plane axis along which the camera is
    # displaced to create the directional view.  When omitted, the legacy
    # sector-box heuristic uses the axis across the sector's longest extent.
    # SC supplies its mouth geometry explicitly because the cluster's long rail
    # is not the same thing as the side of one rectangular adapter that the
    # detector must see.
    directional_tilt_axis_board: Sequence[float] | None = None,
    # Restrict the cross-rail tilt to one side of the board: -1 keeps the camera
    # on the sector's -cross side, +1 the +cross side, 0 searches both.
    cross_rail_sign: float = 0.0,
    # When False only the reference camera must fully frame the sector; the
    # splayed side cameras are not required to.  The three wrist cameras cannot
    # hold all five NIC cards in frame together nearer than ~0.65 m, so a
    # detector that keys on a single dominant top-down view (looking straight
    # into the recessed SFP cages) needs this relaxation to get much closer.
    require_all_cameras_frame: bool = True,
    # Prefer the *farthest* feasible standoff instead of the closest.  Tall parts
    # that protrude toward the camera (NIC cards, 145 mm fins) suffer heavy
    # perspective distortion and worse tool occlusion up close; a higher, farther
    # view sees the whole part cleanly and undistorted -- which is what the
    # model-based estimator needs, and matches the working reference pose that
    # sits high and sees the cards small.
    prefer_far_standoff: bool = False,
    # Optional target-specific image-formation constraint.  Framing and gripper
    # clearance are necessary but not sufficient for recessed ports: a camera
    # can hold the mouth in-frame while a bore wall hides its depth.  The
    # callback returns a scalar margin and candidates below the floor are
    # rejected before IK ranking.  ``None`` leaves existing targets unchanged.
    view_quality: Callable[[Transform], float] | None = None,
    min_view_quality: float = -math.inf,
    # When live-seeded joint ranking is active, candidates at the chosen
    # standoff whose view score is within this much of the best score form one
    # perception-equivalent plateau.  Motion is minimized inside that plateau.
    # Zero preserves strict best-view-first behavior.
    view_quality_motion_tolerance: float = 0.0,
    # Real reachability.  When supplied, ``reachable(base_T_tcp) -> bool`` is the
    # authority on whether the arm can actually achieve a candidate TCP pose
    # (joint-limit-valid IK solution), replacing the crude base-origin
    # ``max_reach_m`` sphere -- which both admits kinematically-impossible poses
    # (Move Robot then reports "IK not computable") and rejects genuinely
    # reachable far-side poses, making the search settle for a near, wrong-side
    # view.  It is applied as a final gate over *every* framed candidate in rank
    # order, so the search commits to the best pose that is both correctly framed
    # *and* reachable.  Gating the whole ranked list matters: for the NIC sector
    # only the closest handful of framed poses are inside the arm's envelope,
    # while ``prefer_far_standoff`` ranks the (unreachable) far ones first, so a
    # truncated gate finds nothing reachable and the search fails with poses
    # available.  ``None`` keeps the legacy sphere for callers/tests without a
    # kinematic model.
    reachable: Callable[[Transform], bool] | None = None,
    # Optional live-seeded analytic IK result.  Return the physical joint delta
    # vector for this candidate, or None when no collision/view-clear
    # configuration exists.  Unlike ``reachable``, this lets the selector rank
    # equally useful camera poses by the arm motion they require and reject
    # contorted targets before handing a Cartesian pose to Move Robot.
    joint_motion: Callable[[Transform], Sequence[float] | None] | None = None,
    max_joint_motion_rad: float = math.radians(170.0),
    # Optional secondary joint-space constraint/ranking.  The callback receives
    # the physical unwrapped joint delta selected for a Cartesian candidate and
    # returns a non-negative error (lower is better).  SC uses it to keep wrist
    # joint 6 near the requested half-turn orientation, which puts the arm/tool
    # on the non-occluding side, while the existing absolute joint window and
    # max-motion gate remain authoritative.
    joint_motion_preference: Callable[[np.ndarray], float] | None = None,
    max_joint_preference_error: float = math.inf,
    # A preference may buy at most this much additional worst-joint travel over
    # the minimum-motion candidate in the same perception plateau.  This makes
    # the preference effective without resurrecting a violent arm route.
    joint_preference_motion_tolerance_rad: float = 0.0,
) -> tuple[SurveyCandidate | None, str]:
    """Deterministically search for one board-relative TCP survey pose.

    ``coverage_targets`` are board-frame corner sets tried in order (most
    preferred first; default: the whole board face, then the SFP/SC module
    region).  For each target every candidate camera pose projects the target
    through all supplied cameras; a candidate is feasible only when every camera
    has the target in front, fully inside the image with a positive boundary
    margin, clear of the gripper keep-out with a positive margin, and holding at
    least ``min_required_clearance_px`` of that margin so a small pose error
    cannot clip a component.  The first target that yields any feasible candidate
    wins; among those the pose with the *smallest standoff* is returned -- the
    closest view puts the most pixels on each component, which is what lets the
    downstream estimator separate adjacent NIC cards and SC ports -- and ties are
    broken first by any target-specific image-formation quality, then by
    live-seeded joint motion when supplied, overhead view, clearance, and
    Cartesian motion.  The chosen board-frame target is attached as
    ``candidate.coverage_target``.  The distant-scale and per-module detail gates
    are intentionally not applied here.  Returns ``(None, reason)`` when no
    candidate is feasible or reachable for any target.
    """

    if reference_camera not in tcp_T_cam:
        return None, f"reference camera {reference_camera!r} has no extrinsic"
    if (
        not math.isfinite(max_angular_motion_rad)
        or max_angular_motion_rad < 0.0
    ):
        return None, "maximum angular motion must be finite and non-negative"
    if math.isnan(min_view_quality):
        return None, "minimum view quality must not be NaN"
    if (
        not math.isfinite(view_quality_motion_tolerance)
        or view_quality_motion_tolerance < 0.0
    ):
        return None, "view-quality motion tolerance must be finite and non-negative"
    if (
        not math.isfinite(max_joint_motion_rad)
        or max_joint_motion_rad < 0.0
    ):
        return None, "maximum joint motion must be finite and non-negative"
    if math.isnan(max_joint_preference_error) or max_joint_preference_error < 0.0:
        return None, "maximum joint preference error must be non-negative"
    if (
        not math.isfinite(joint_preference_motion_tolerance_rad)
        or joint_preference_motion_tolerance_rad < 0.0
    ):
        return None, "joint preference motion tolerance must be finite and non-negative"
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
    aim_offset_board = np.asarray(aim_offset_board_m, dtype=float)
    if (
        aim_offset_board.shape != (3,)
        or not np.all(np.isfinite(aim_offset_board))
        or abs(float(aim_offset_board[2])) > 1e-9
    ):
        return None, "aim offset must be a finite board-plane xyz vector"

    base_T_board = board_pose.base_T_board
    board_normal_base = base_T_board.rotation[:, 2]
    board_normal_base = board_normal_base / np.linalg.norm(board_normal_base)
    # In-plane basis for lateral offsets and the up hint (board X/Y in base).
    board_x_base = base_T_board.rotation[:, 0]
    board_y_base = base_T_board.rotation[:, 1]
    ref_tcp_T_cam = tcp_T_cam[reference_camera]

    def _best_for_target(
        target_board: np.ndarray,
    ) -> tuple[SurveyCandidate | None, int, int]:
        # Aim the reference optical axis at the target centroid and search a
        # board-relative 3-D grid, giving real pitch/roll variation rather than a
        # top-down pose with image roll only.
        center_base = base_T_board.apply(
            target_board.mean(axis=0) + aim_offset_board
        )
        # Collect every framed, gripper-clear candidate with its ranking key so
        # the real reachability gate can be applied to the best ones in order
        # (rather than committing to a single geometric best that may be
        # unreachable).  A monotone counter keeps the sort deterministic on ties.
        feasible: list[tuple[tuple, int, SurveyCandidate]] = []
        order_counter = 0
        evaluated = 0
        # Directional obliquity is opt-in (a cross-rail tilt band was supplied).
        # The rail axis is the sector's own longer in-plane edge, taken in the
        # *estimated* board frame, so it follows the board wherever the insignia
        # places it -- never a fixed world axis or a hard-coded component point.
        directional = cross_rail_tilt_band_rad is not None
        rail_hat = cross_hat = board_x_base
        band_lo = band_hi = band_mid = 0.0
        if directional:
            if directional_tilt_axis_board is not None:
                axis_board = np.asarray(
                    directional_tilt_axis_board, dtype=float
                )
                if (
                    axis_board.shape != (3,)
                    or not np.all(np.isfinite(axis_board))
                    or abs(float(axis_board[2])) > 1e-6
                    or float(np.linalg.norm(axis_board[:2])) < 1e-9
                ):
                    return None, 0, 0
                axis_board = axis_board / np.linalg.norm(axis_board)
                cross_hat = (
                    board_x_base * float(axis_board[0])
                    + board_y_base * float(axis_board[1])
                )
                # Positive 90 degrees in the board plane.  Only absolute
                # along/cross tilt is measured, so the sign is immaterial here.
                rail_hat = (
                    -board_x_base * float(axis_board[1])
                    + board_y_base * float(axis_board[0])
                )
            else:
                extent = target_board.max(axis=0) - target_board.min(axis=0)
                if float(extent[0]) >= float(extent[1]):
                    rail_hat, cross_hat = board_x_base, board_y_base
                else:
                    rail_hat, cross_hat = board_y_base, board_x_base
            band_lo, band_hi = cross_rail_tilt_band_rad
            band_mid = 0.5 * (band_lo + band_hi)
            axis_a_hat, axis_b_hat = cross_hat, rail_hat

            signs = (
                (1.0, -1.0)
                if cross_rail_sign == 0.0
                else (math.copysign(1.0, cross_rail_sign),)
            )

            def _axis_a_samples(reach: float) -> Sequence[float]:
                # Cross-rail: tilt across the rail by each in-band angle.  Signs
                # restricted by ``cross_rail_sign`` so the camera stays on the
                # bore-facing side.  ``reach * tan(angle)`` fixes the *angle*, not
                # a distance, so the geometry holds as the standoff changes.
                return [
                    sign * reach * math.tan(float(alpha))
                    for alpha in np.linspace(band_lo, band_hi, 5)
                    for sign in signs
                ]

            def _axis_b_samples(reach: float) -> Sequence[float]:
                # Along-rail: held near zero so the parts do not occlude/
                # foreshorten one another.
                return [
                    reach * math.tan(beta)
                    for beta in (
                        -0.5 * max_along_rail_tilt_rad,
                        0.0,
                        0.5 * max_along_rail_tilt_rad,
                    )
                ]
        else:
            axis_a_hat, axis_b_hat = board_x_base, board_y_base

            def _axis_a_samples(reach: float) -> Sequence[float]:
                return offsets_x_m

            def _axis_b_samples(reach: float) -> Sequence[float]:
                return offsets_y_m

        for standoff in standoffs_m:
            for a_off in _axis_a_samples(standoff):
                for b_off in _axis_b_samples(standoff):
                    for yaw in yaws_rad:
                        offset_vec = axis_a_hat * a_off + axis_b_hat * b_off
                        cam_origin = (
                            center_base
                            + board_normal_base * standoff
                            + offset_vec
                        )
                        if cam_origin[2] < min_height_m:
                            continue
                        # View direction (sector centre -> camera), decomposed in
                        # the board frame.  ``obliquity`` is the total angle off
                        # the normal; directional sectors bound the along-rail and
                        # cross-rail tilt components separately.
                        to_camera = cam_origin - center_base
                        to_camera_norm = float(np.linalg.norm(to_camera))
                        normal_comp = float(np.dot(to_camera, board_normal_base))
                        if to_camera_norm < 1e-9 or normal_comp <= 0.0:
                            continue
                        obliquity = math.acos(
                            float(np.clip(normal_comp / to_camera_norm, -1.0, 1.0))
                        )
                        cross_tilt = 0.0
                        along_tilt = 0.0
                        if directional:
                            along_tilt = math.atan2(
                                abs(float(np.dot(to_camera, rail_hat))),
                                normal_comp,
                            )
                            cross_tilt = math.atan2(
                                abs(float(np.dot(to_camera, cross_hat))),
                                normal_comp,
                            )
                            # Along-rail tilt occludes/foreshortens the parts;
                            # cross-rail tilt reveals depth with the neighbours
                            # still separated -- require it in-band, along-rail ~0.
                            if along_tilt > max_along_rail_tilt_rad:
                                continue
                            if not (
                                band_lo - 1e-6 <= cross_tilt <= band_hi + 1e-6
                            ):
                                continue
                        elif obliquity > max_obliquity_rad:
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
                        if float(base_T_tcp.translation[2]) < min_height_m:
                            continue
                        # Reach: the base-origin sphere is a poor proxy (it
                        # rejects reachable far-side poses and admits
                        # kinematically-impossible ones).  Use it only when no
                        # real IK model is supplied; otherwise keep a generous
                        # absolute prune and let ``reachable`` be the authority.
                        reach = float(np.linalg.norm(base_T_tcp.translation))
                        reach_cap = max_reach_m if reachable is None else 1.15
                        if reach > reach_cap:
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

                        # For recessed SC mouths, reject rays that cannot reach
                        # the black back plane before doing three full-resolution
                        # gripper-mask clearances.  This is an equivalent
                        # reordering of hard gates, not a relaxed search.
                        target_view_quality = (
                            float(view_quality(base_T_tcp))
                            if view_quality is not None
                            else math.inf
                        )
                        if (
                            math.isnan(target_view_quality)
                            or target_view_quality < min_view_quality
                        ):
                            continue

                        # Check the reference camera first, then the remaining
                        # cameras in stable input order.  In the all-camera mode
                        # a single failure is terminal, so do not spend time
                        # evaluating masks for cameras that cannot change the
                        # outcome.  Accepted candidates still carry all camera
                        # coverages exactly as before.
                        camera_items = list(cameras.items())
                        camera_items.sort(
                            key=lambda item: item[0] != reference_camera
                        )
                        coverages = []
                        for name, camera in camera_items:
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
                                    required_clearance_px=(
                                        min_required_clearance_px
                                        if require_all_cameras_frame
                                        or name == reference_camera
                                        else 0.0
                                    ),
                                    # Only "framed and gripper-clear in every
                                    # camera" gates here; disable the distant-scale
                                    # and per-module detail checks.
                                    min_pixel_scale=0.0,
                                    module_envelopes_board=(),
                                    min_module_pixel_scale=0.0,
                                )
                            )
                            if require_all_cameras_frame and (
                                not coverages[-1].feasible
                                or coverages[-1].clearance
                                < min_required_clearance_px
                            ):
                                break
                        evaluated += 1
                        if not coverages:
                            continue
                        if require_all_cameras_frame:
                            # Every camera must fully frame and clear the sector.
                            if not all(c.feasible for c in coverages):
                                continue
                            min_clear = min(c.clearance for c in coverages)
                        else:
                            # Only the reference camera must frame the sector;
                            # the splayed side cameras fall away.  This is what
                            # lets a close top-down view exist at all -- the rig
                            # cannot hold five NIC cards in three cameras nearer
                            # than ~0.65 m, but the centre camera alone can look
                            # straight down from far closer.
                            ref_cov = next(
                                (
                                    c
                                    for c in coverages
                                    if c.camera_name == reference_camera
                                ),
                                None,
                            )
                            if ref_cov is None or not ref_cov.feasible:
                                continue
                            min_clear = ref_cov.clearance
                        # Reject poses that only just fit: downstream pose
                        # estimation needs real margin on the framing camera(s).
                        if min_clear < min_required_clearance_px:
                            continue
                        candidate = SurveyCandidate(
                            base_T_tcp=base_T_tcp,
                            min_clearance_px=min_clear,
                            coverages=tuple(coverages),
                            standoff_m=standoff,
                            yaw_rad=yaw,
                            offset_x_m=float(np.dot(offset_vec, board_x_base)),
                            offset_y_m=float(np.dot(offset_vec, board_y_base)),
                            view_quality=target_view_quality,
                            cross_rail_tilt_rad=cross_tilt,
                            along_rail_tilt_rad=along_tilt,
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
                        # Deterministic lexicographic objective.  Standoff
                        # dominates: normally the closest framing wins (most
                        # pixels), but ``prefer_far_standoff`` flips it to the
                        # farthest feasible pose for tall protruding parts, whose
                        # model match wants a distant, undistorted, less-occluded
                        # view.  Then: directional sectors prefer the cross-rail
                        # tilt nearest the band centre; isotropic sectors prefer
                        # the strongest target-specific image formation (when
                        # supplied), then the most overhead view.  Then
                        # clearance, then motion.  A finite view score must
                        # remain in the objective after passing its hard floor:
                        # SC hardware showed that a barely passing +0.054 bore
                        # margin loses the two diagonal end ports while larger
                        # margins preserve a dark rectangular interior.
                        #
                        standoff_key = (
                            round(standoff, 4)
                            if prefer_far_standoff
                            else -round(standoff, 4)
                        )
                        view_quality_key = (
                            round(target_view_quality, 4)
                            if math.isfinite(target_view_quality)
                            else 0.0
                        )
                        if directional:
                            candidate_key = (
                                standoff_key,
                                view_quality_key,
                                -round(abs(cross_tilt - band_mid), 4),
                                round(candidate.min_clearance_px, 6),
                                -round(candidate.motion_m, 4),
                                -candidate.angular_motion_rad,
                            )
                        else:
                            candidate_key = (
                                standoff_key,
                                view_quality_key,
                                -round(obliquity, 4),
                                round(candidate.min_clearance_px, 6),
                                -round(candidate.motion_m, 4),
                                -candidate.angular_motion_rad,
                            )
                        feasible.append((candidate_key, order_counter, candidate))
                        order_counter += 1
        framed = len(feasible)
        if not feasible:
            return None, evaluated, framed
        # Best-ranked first (descending key; the counter breaks ties in the
        # original insertion order, matching the old strict-> argmax).
        feasible.sort(key=lambda item: item[0], reverse=True)
        if reachable is None and joint_motion is None:
            return feasible[0][2], evaluated, framed
        if joint_motion is not None:
            # Preserve the resolution priority exactly: standoff first.  At one
            # standoff, retain the target-specific image-formation scores close
            # enough to the best to be perception-equivalent, then minimize
            # worst-joint and total travel.  This evaluates every roll in a
            # standoff group because roll changes both the separated cameras'
            # SC bore rays and the UR shoulder/wrist branch.
            group_start = 0
            while group_start < len(feasible):
                primary = feasible[group_start][0][:1]
                group_end = group_start + 1
                while (
                    group_end < len(feasible)
                    and feasible[group_end][0][:1] == primary
                ):
                    group_end += 1
                motion_valid = []
                for key, order, candidate in feasible[group_start:group_end]:
                    if reachable is not None and not reachable(candidate.base_T_tcp):
                        continue
                    raw_delta = joint_motion(candidate.base_T_tcp)
                    if raw_delta is None:
                        continue
                    delta = np.asarray(raw_delta, dtype=float)
                    if (
                        delta.ndim != 1
                        or delta.size == 0
                        or not np.all(np.isfinite(delta))
                    ):
                        continue
                    max_motion = float(np.abs(delta).max())
                    if max_motion > max_joint_motion_rad + 1e-9:
                        continue
                    total_motion = float(np.abs(delta).sum())
                    preference_error = (
                        float(joint_motion_preference(delta))
                        if joint_motion_preference is not None
                        else 0.0
                    )
                    if (
                        not math.isfinite(preference_error)
                        or preference_error < 0.0
                        or preference_error > max_joint_preference_error + 1e-9
                    ):
                        continue
                    scored = replace(
                        candidate,
                        max_joint_motion_rad=max_motion,
                        total_joint_motion_rad=total_motion,
                    )
                    motion_valid.append(
                        (
                            key,
                            order,
                            scored,
                            max_motion,
                            total_motion,
                            preference_error,
                        )
                    )
                if motion_valid:
                    # Define the visual plateau only over candidates Move Robot
                    # can actually reach inside its mirrored joint window.
                    # Otherwise an unreachable high-quality roll can suppress
                    # every reachable roll at this standoff.
                    best_view_key = max(item[0][1] for item in motion_valid)
                    accepted_view_key = (
                        best_view_key - view_quality_motion_tolerance
                    )
                    view_valid = []
                    for (
                        key,
                        order,
                        scored,
                        max_motion,
                        total_motion,
                        preference_error,
                    ) in motion_valid:
                        if key[1] < accepted_view_key - 1e-9:
                            continue
                        view_valid.append(
                            (
                                key,
                                order,
                                scored,
                                max_motion,
                                total_motion,
                                preference_error,
                            )
                        )
                    minimum_max_motion = min(
                        item[3] for item in view_valid
                    )
                    motion_ranked = []
                    for (
                        key,
                        order,
                        scored,
                        max_motion,
                        total_motion,
                        preference_error,
                    ) in view_valid:
                        if (
                            joint_motion_preference is not None
                            and max_motion
                            > minimum_max_motion
                            + joint_preference_motion_tolerance_rad
                            + 1e-9
                        ):
                            continue
                        if joint_motion_preference is not None:
                            motion_key = (
                                -round(preference_error, 6),
                                -round(max_motion, 6),
                                -round(total_motion, 6),
                                key[1],
                                *key[2:],
                            )
                        else:
                            motion_key = (
                                -round(max_motion, 6),
                                -round(total_motion, 6),
                                key[1],
                                *key[2:],
                            )
                        motion_ranked.append((motion_key, order, scored))
                    motion_ranked.sort(key=lambda item: item[0], reverse=True)
                    return motion_ranked[0][2], evaluated, framed
                group_start = group_end
            return None, evaluated, framed
        # Apply the real reachability gate to the candidates in rank order and
        # commit to the first that the arm can actually achieve.
        for _key, _order, candidate in feasible:
            if reachable(candidate.base_T_tcp):
                return candidate, evaluated, framed
        return None, evaluated, framed

    total_evaluated = 0
    total_framed = 0
    for target_board in coverage_targets:
        best, evaluated, framed = _best_for_target(target_board)
        total_evaluated += evaluated
        total_framed += framed
        if best is not None:
            return best, "ok"
    if (reachable is not None or joint_motion is not None) and total_framed > 0:
        return (
            None,
            f"{total_framed} pose(s) framed the target in all required cameras "
            "but none had a reachable, joint-motion-valid IK solution "
            f"({total_evaluated} candidates evaluated)",
        )
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
