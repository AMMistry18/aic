"""Camera-directed lateral alignment for the physical SC duplex opening.

The SC pose network is useful for selecting the correct adapter and estimating
its plane, but its four label corners have a repeatable common-centre bias.  This
module measures the physical opening in the same wrist-camera images instead:

* blue pixels confirm that the ROI belongs to an SC adapter;
* the two dark duplex bores provide the actual mouth centre;
* camera rays intersect the already-estimated port plane; and
* agreeing views produce a bounded lateral target.

Everything here is ROS-free and fail-soft.  Missing/occluded imagery returns
``None``; it is never itself a reason to abandon an insertion attempt.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .visual_gap import detect_dark_port_opening


SC_BLUE_LOWER = np.array([90, 80, 60], dtype=np.uint8)
SC_BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)


@dataclass(frozen=True)
class ScDuplexOpeningDetection:
    """Physical duplex-mouth centre detected in one image."""

    center_uv: np.ndarray
    bore_centers_uv: tuple[np.ndarray | None, np.ndarray | None]
    detected_bores: int
    blue_fraction: float
    score: float


@dataclass(frozen=True)
class ScOpeningEstimate:
    """Opening point fused on the port plane."""

    point_world: np.ndarray
    cameras: tuple[str, ...]
    disagreement_m: float
    single_view: bool


@dataclass(frozen=True)
class ScRecoveryEvidence:
    """Relative plug-offset evidence from one rectified camera view.

    ``direction_xy`` is expressed in the physical port plane: +X is the first
    lateral column of ``Rp`` and +Y is the second.  A low plug therefore asks
    for +Y regardless of how the camera image is rotated.  ``margins`` are
    retained for compatibility, but now hold current/pre-contact blue-side
    visibility ratios ordered ``[-X, +X, -Y, +Y]``.  Ratios below one mean
    that the plug is occluding that side of the blue housing.
    """

    direction_xy: np.ndarray
    confidence: float
    margins: np.ndarray
    blue_fraction: float
    valid_fraction: float
    balanced: bool


@dataclass(frozen=True)
class ScRecoveryEstimate:
    """Multi-camera recovery direction accepted for one bounded step."""

    direction_xy: np.ndarray
    confidence: float
    cameras: tuple[str, ...]
    resultant: float
    balanced: bool


@dataclass(frozen=True)
class ScBlueSideSignature:
    """Blue-housing visibility around a rectified SC mouth.

    ``blue_fractions`` is ordered ``[-X, +X, -Y, +Y]``.  It is captured while
    the plug is still clear of the port, then used as a per-camera reference at
    a shallow stall.  This makes the recovery cue insensitive to fixed exposure
    differences and to common pose bias in the original YOLO estimate.

    The optional masks make that reference safe when a calibrated gripper mask
    consistently covers part of the mouth.  They are canonical, rectified
    masks: a recovery frame is only compared over their shared support.  The
    arrays are made read-only when measured so a baseline cannot be modified
    after it has been captured.
    """

    blue_fractions: np.ndarray
    blue_fraction: float
    valid_fraction: float
    side_support_masks: tuple[np.ndarray, ...] | None = None
    side_blue_masks: tuple[np.ndarray, ...] | None = None
    corridor_support_mask: np.ndarray | None = None


_RECOVERY_OPENING_WIDTH_PX = 224
_RECOVERY_OPENING_HEIGHT_PX = 96
_RECOVERY_OPENING_WIDTH_M = 0.02241
_RECOVERY_OPENING_HEIGHT_M = 0.00785
_RECOVERY_PAD_X_PX = 48
_RECOVERY_PAD_Y_PX = 48
_RECOVERY_MIN_PROJECTED_EDGE_PX = 14.0
_RECOVERY_SIDE_BAND_HALF_WIDTHS = (10, 12)
_RECOVERY_MIN_BASELINE_SIDE_FRACTION = 0.50
_RECOVERY_MIN_SHARED_SIDE_SUPPORT_FRACTION = 0.50
_RECOVERY_MIN_SIGNAL_NORM = 0.05
# Kept only so the retired silhouette routine below remains callable while this
# branch rolls out.  The insertion controller never falls back to it: contact
# collapses its blue-gap measurement to zero on real images.
_RECOVERY_PLUG_SATURATION_MAX = 95
_RECOVERY_PLUG_VALUE_MIN = 45


def _finite_quad(value: np.ndarray) -> np.ndarray:
    quad = np.asarray(value, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        raise ValueError("each expected SC bore quad must be a finite 4x2 array")
    return quad


def _rectify_sc_port_view(
    image: np.ndarray,
    expected_bore_quads_uv: np.ndarray,
    ignored_pixels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return a fronto-parallel port patch, valid mask, and opening bounds.

    ``ScInsertionController._visual_bore_quads_world`` creates the two bore
    quads in a pinned order.  The four points selected below are therefore the
    physical duplex envelope ``[+x,+y],[-x,+y],[-x,-y],[+x,-y]``.  Mapping it
    to an inset rectangle preserves surrounding blue housing for clearance
    measurements while making screen up equal port-local +Y.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must be BGR or BGRA")
    bgr = np.ascontiguousarray(array[:, :, :3])
    bore_quads = np.asarray(expected_bore_quads_uv, dtype=np.float64)
    if bore_quads.shape != (2, 4, 2) or not np.all(np.isfinite(bore_quads)):
        raise ValueError("expected_bore_quads_uv must be a finite 2x4x2 array")

    source = np.asarray(
        [
            bore_quads[0, 0],  # +x, +y
            bore_quads[1, 1],  # -x, +y
            bore_quads[1, 2],  # -x, -y
            bore_quads[0, 3],  # +x, -y
        ],
        dtype=np.float32,
    )
    edge_lengths = np.linalg.norm(
        source - np.roll(source, -1, axis=0), axis=1
    )
    if (
        abs(float(cv2.contourArea(source))) < 160.0
        or float(np.min(edge_lengths)) < _RECOVERY_MIN_PROJECTED_EDGE_PX
    ):
        return None

    x0 = float(_RECOVERY_PAD_X_PX)
    y0 = float(_RECOVERY_PAD_Y_PX)
    x1 = x0 + float(_RECOVERY_OPENING_WIDTH_PX)
    y1 = y0 + float(_RECOVERY_OPENING_HEIGHT_PX)
    target = np.asarray(
        [[x1, y0], [x0, y0], [x0, y1], [x1, y1]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(source, target)
    if not np.all(np.isfinite(transform)):
        return None
    singular_values = np.linalg.svd(transform, compute_uv=False)
    if (
        singular_values[-1] <= 1e-12
        or singular_values[0] / singular_values[-1] > 1e8
    ):
        return None

    output_size = (
        _RECOVERY_OPENING_WIDTH_PX + 2 * _RECOVERY_PAD_X_PX,
        _RECOVERY_OPENING_HEIGHT_PX + 2 * _RECOVERY_PAD_Y_PX,
    )
    rectified = cv2.warpPerspective(
        bgr,
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    source_valid = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    if ignored_pixels is not None:
        ignored = np.asarray(ignored_pixels, dtype=bool)
        if ignored.shape != bgr.shape[:2]:
            raise ValueError("ignored_pixels shape must match the image")
        source_valid[ignored] = 0
    valid = cv2.warpPerspective(
        source_valid,
        transform,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    bounds = np.asarray([x0, y0, x1, y1], dtype=np.float64)
    return rectified, valid != 0, bounds


def _readonly_bool_mask(value: np.ndarray) -> np.ndarray:
    """Return an owned, immutable boolean mask for a frozen baseline."""

    mask = np.ascontiguousarray(np.asarray(value, dtype=bool)).copy()
    mask.setflags(write=False)
    return mask


def _readonly_float_array(value: np.ndarray) -> np.ndarray:
    """Return an owned, immutable float array for a frozen baseline."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).copy()
    array.setflags(write=False)
    return array


def _recovery_side_band_slices(
    bounds: np.ndarray, band_half_width_px: int
) -> tuple[tuple[slice, slice], ...]:
    """Return canonical side-band slices ordered ``[-X, +X, -Y, +Y]``."""

    band = int(band_half_width_px)
    x0, y0, x1, y1 = [int(round(value)) for value in bounds]
    return (
        (slice(y0, y1), slice(x0 - band, x0 + band)),
        (slice(y0, y1), slice(x1 - band, x1 + band)),
        (slice(y1 - band, y1 + band), slice(x0, x1)),
        (slice(y0 - band, y0 + band), slice(x0, x1)),
    )


def _signature_support_data(
    signature: ScBlueSideSignature,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray] | None:
    """Validate and expose the rich, canonical support stored in a signature."""

    supports = signature.side_support_masks
    blue_masks = signature.side_blue_masks
    corridor = signature.corridor_support_mask
    if supports is None or blue_masks is None or corridor is None:
        return None
    if len(supports) != 4 or len(blue_masks) != 4:
        return None
    normalized_supports = []
    normalized_blue = []
    for support, blue in zip(supports, blue_masks):
        support_array = np.asarray(support, dtype=bool)
        blue_array = np.asarray(blue, dtype=bool)
        if (
            support_array.ndim != 2
            or blue_array.shape != support_array.shape
            or support_array.size == 0
        ):
            return None
        normalized_supports.append(support_array)
        normalized_blue.append(blue_array & support_array)
    corridor_array = np.asarray(corridor, dtype=bool)
    if corridor_array.ndim != 2 or corridor_array.size == 0:
        return None
    return tuple(normalized_supports), tuple(normalized_blue), corridor_array


def aggregate_sc_blue_side_signatures(
    samples: list[ScBlueSideSignature] | tuple[ScBlueSideSignature, ...],
) -> ScBlueSideSignature | None:
    """Freeze a conservative rich baseline from repeated side signatures.

    A support pixel must be present in every baseline frame.  Blue is a strict
    majority observation (all frames for a two-frame baseline), preventing a
    one-frame mask or color glitch from entering the recovery reference.
    ``None`` means the samples do not describe the same rectified geometry.
    """

    signatures = tuple(samples)
    if not signatures:
        return None
    data = []
    for signature in signatures:
        if not isinstance(signature, ScBlueSideSignature):
            return None
        support_data = _signature_support_data(signature)
        if support_data is None:
            return None
        data.append(support_data)

    side_supports = []
    side_blue = []
    for index in range(4):
        support_values = [item[0][index] for item in data]
        blue_values = [item[1][index] for item in data]
        if len({array.shape for array in support_values}) != 1:
            return None
        support_stack = np.stack(
            support_values, axis=0
        )
        if len({array.shape for array in blue_values}) != 1:
            return None
        blue_stack = np.stack(blue_values, axis=0)
        support = np.all(support_stack, axis=0)
        # Strictly more than half avoids treating a one-frame blue artifact as
        # a valid reference when the minimum baseline has two samples.
        blue = np.sum(blue_stack, axis=0) * 2 > support_stack.shape[0]
        blue &= support
        side_supports.append(_readonly_bool_mask(support))
        side_blue.append(_readonly_bool_mask(blue))

    corridor_values = [item[2] for item in data]
    if len({array.shape for array in corridor_values}) != 1:
        return None
    corridor_stack = np.stack(corridor_values, axis=0)
    corridor = _readonly_bool_mask(np.all(corridor_stack, axis=0))
    fractions = np.asarray(
        [
            float(np.count_nonzero(blue) / max(1, np.count_nonzero(support)))
            for support, blue in zip(side_supports, side_blue)
        ],
        dtype=np.float64,
    )
    return ScBlueSideSignature(
        blue_fractions=_readonly_float_array(fractions),
        blue_fraction=float(
            np.median([signature.blue_fraction for signature in signatures])
        ),
        valid_fraction=float(
            np.median([signature.valid_fraction for signature in signatures])
        ),
        side_support_masks=tuple(side_supports),
        side_blue_masks=tuple(side_blue),
        corridor_support_mask=corridor,
    )


def _recovery_search_and_seed_masks(
    shape: tuple[int, int], bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x0, y0, x1, y1 = [float(value) for value in bounds]
    opening_w = x1 - x0
    opening_h = y1 - y0
    search = np.zeros((height, width), dtype=bool)
    sx0 = max(0, int(np.floor(x0 - 0.16 * opening_w)))
    sx1 = min(width, int(np.ceil(x1 + 0.16 * opening_w)))
    sy0 = max(0, int(np.floor(y0 - 0.55 * opening_h)))
    sy1 = min(height, int(np.ceil(y1 + 0.55 * opening_h)))
    search[sy0:sy1, sx0:sx1] = True

    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    seed = np.zeros_like(search)
    seed[
        max(0, int(np.floor(center_y - 0.27 * opening_h))):
        min(height, int(np.ceil(center_y + 0.27 * opening_h))),
        max(0, int(np.floor(center_x - 0.30 * opening_w))):
        min(width, int(np.ceil(center_x + 0.30 * opening_w))),
    ] = True
    return search, seed


def _select_center_plug_component(
    rectified: np.ndarray,
    blue: np.ndarray,
    valid: np.ndarray,
    bounds: np.ndarray,
    *,
    saturation_max: int,
) -> tuple[np.ndarray, dict] | None:
    """Select a gray/white component that demonstrably crosses the mouth.

    The SC plug texture is beige/gray/white while the adapter is saturated blue
    and the unoccluded bores are dark.  Requiring a component to intersect a
    large central seed prevents the surrounding gray task board from becoming
    the plug merely because it is also low saturation.
    """

    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    search, seed = _recovery_search_and_seed_masks(valid.shape, bounds)
    candidate = (
        (hsv[:, :, 1] <= int(saturation_max))
        & (hsv[:, :, 2] >= _RECOVERY_PLUG_VALUE_MIN)
        & (~blue)
        & valid
        & search
    )
    candidate_u8 = candidate.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel)
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidate_u8[~valid] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, 8)
    opening_area = float(
        _RECOVERY_OPENING_WIDTH_PX * _RECOVERY_OPENING_HEIGHT_PX
    )
    best = None
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / opening_area
        seed_overlap = int(np.count_nonzero(component & seed))
        if not 0.08 <= area_ratio <= 2.2 or seed_overlap < 12:
            continue
        # Seed overlap is the identity cue; area only breaks close ties.
        score = float(seed_overlap) + 0.02 * float(area)
        if best is None or score > best[0]:
            best = (score, component, area_ratio, seed_overlap)
    if best is None:
        return None

    _, component, area_ratio, seed_overlap = best
    ys, xs = np.nonzero(component)
    width_ratio = float(np.percentile(xs, 98) - np.percentile(xs, 2)) / float(
        _RECOVERY_OPENING_WIDTH_PX
    )
    height_ratio = float(np.percentile(ys, 98) - np.percentile(ys, 2)) / float(
        _RECOVERY_OPENING_HEIGHT_PX
    )
    if not 0.35 <= width_ratio <= 1.45 or not 0.30 <= height_ratio <= 1.85:
        return None
    return component, {
        "area_ratio": area_ratio,
        "seed_overlap": seed_overlap,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
    }


def _median_blue_gap(
    plug: np.ndarray,
    blue: np.ndarray,
    valid: np.ndarray,
    *,
    axis: int,
    positive: bool,
) -> tuple[float, int] | None:
    """Median non-blue clearance from the plug edge toward one port side."""

    values = []
    if axis == 0:
        # +/-X: scan image rows and search right/left from the plug.
        coordinates = np.flatnonzero(np.any(plug, axis=1))
        if coordinates.size:
            low, high = np.percentile(coordinates, [15.0, 85.0])
            coordinates = coordinates[
                (coordinates >= int(np.floor(low)))
                & (coordinates <= int(np.ceil(high)))
            ]
        for row in coordinates:
            plug_x = np.flatnonzero(plug[row])
            if plug_x.size == 0:
                continue
            edge = int(plug_x.max() if positive else plug_x.min())
            candidates = np.flatnonzero(blue[row] & valid[row])
            candidates = candidates[candidates > edge] if positive else candidates[
                candidates < edge
            ]
            if candidates.size == 0:
                continue
            nearest = int(candidates.min() if positive else candidates.max())
            values.append(max(0.0, float(abs(nearest - edge) - 1)))
    else:
        # +/-Y: canonical image top is +Y, bottom is -Y.
        coordinates = np.flatnonzero(np.any(plug, axis=0))
        if coordinates.size:
            low, high = np.percentile(coordinates, [15.0, 85.0])
            coordinates = coordinates[
                (coordinates >= int(np.floor(low)))
                & (coordinates <= int(np.ceil(high)))
            ]
        for column in coordinates:
            plug_y = np.flatnonzero(plug[:, column])
            if plug_y.size == 0:
                continue
            # positive Y searches upward (decreasing image row).
            edge = int(plug_y.min() if positive else plug_y.max())
            candidates = np.flatnonzero(blue[:, column] & valid[:, column])
            candidates = candidates[candidates < edge] if positive else candidates[
                candidates > edge
            ]
            if candidates.size == 0:
                continue
            nearest = int(candidates.max() if positive else candidates.min())
            values.append(max(0.0, float(abs(nearest - edge) - 1)))
    if len(values) < 8:
        return None
    return float(np.median(values)), len(values)


def _measure_recovery_margins(
    plug: np.ndarray,
    blue: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    minus_x = _median_blue_gap(plug, blue, valid, axis=0, positive=False)
    plus_x = _median_blue_gap(plug, blue, valid, axis=0, positive=True)
    minus_y = _median_blue_gap(plug, blue, valid, axis=1, positive=False)
    plus_y = _median_blue_gap(plug, blue, valid, axis=1, positive=True)
    if any(value is None for value in (minus_x, plus_x, minus_y, plus_y)):
        return None
    gaps_px = np.asarray(
        [minus_x[0], plus_x[0], minus_y[0], plus_y[0]], dtype=np.float64
    )
    margins = gaps_px / np.asarray(
        [
            _RECOVERY_OPENING_WIDTH_PX,
            _RECOVERY_OPENING_WIDTH_PX,
            _RECOVERY_OPENING_HEIGHT_PX,
            _RECOVERY_OPENING_HEIGHT_PX,
        ],
        dtype=np.float64,
    )
    # Directions are ultimately commanded in metric Rp coordinates.  Keeping
    # normalized image units here would overweight Y by the 22.41/7.85 port
    # aspect ratio whenever both axes are involved.
    raw_direction = np.asarray(
        [
            (margins[1] - margins[0]) * _RECOVERY_OPENING_WIDTH_M,
            (margins[3] - margins[2]) * _RECOVERY_OPENING_HEIGHT_M,
        ],
        dtype=np.float64,
    )
    scan_counts = np.asarray(
        [minus_x[1], plus_x[1], minus_y[1], plus_y[1]], dtype=np.float64
    )
    return margins, raw_direction, scan_counts


def _recovery_blue_measurement(
    image: np.ndarray,
    expected_bore_quads_uv: np.ndarray,
    ignored_pixels: np.ndarray | None,
    *,
    band_half_width_px: int,
    min_blue_fraction: float,
    diagnostics: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """Rectify one view and isolate its masked blue-housing measurement.

    Validity is retained as geometry rather than being interpolated away.  A
    later paired comparison decides whether the frame shares enough support
    with its frozen baseline; this lets a stable calibrated gripper mask be
    ignored without allowing a new occlusion to fabricate an escape direction.
    """

    rectified_result = _rectify_sc_port_view(
        image, expected_bore_quads_uv, ignored_pixels
    )
    if rectified_result is None:
        if diagnostics is not None:
            diagnostics["reason"] = "rectification_failed"
        return None
    rectified, valid, bounds = rectified_result
    valid_fraction = float(np.mean(valid))
    # Rich paired support below validates every relevant side band.  Keep only
    # an early empty-view guard here so a stable mask over the mouth itself
    # does not discard otherwise sufficient side-band evidence.
    if valid_fraction < 0.05:
        if diagnostics is not None:
            diagnostics.update(
                reason="insufficient_valid_roi", valid_fraction=valid_fraction
            )
        return None

    band = int(band_half_width_px)
    x0, y0, x1, y1 = [int(round(value)) for value in bounds]
    if (
        band <= 0
        or x0 - band < 0
        or y0 - band < 0
        or x1 + band > valid.shape[1]
        or y1 + band > valid.shape[0]
    ):
        if diagnostics is not None:
            diagnostics.update(reason="invalid_side_band", band_half_width_px=band)
        return None
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, SC_BLUE_LOWER, SC_BLUE_UPPER) != 0
    blue &= valid
    blue_u8 = blue.astype(np.uint8) * 255
    blue = cv2.morphologyEx(
        blue_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) != 0
    blue &= valid

    ring = np.ones_like(valid)
    ring[y0:y1, x0:x1] = False
    ring &= valid
    blue_fraction = float(
        np.count_nonzero(blue & ring) / max(1, np.count_nonzero(ring))
    )
    if blue_fraction < min_blue_fraction:
        if diagnostics is not None:
            diagnostics.update(
                reason="blue_association_failed",
                blue_fraction=blue_fraction,
                valid_fraction=valid_fraction,
            )
        return None
    return blue, valid, bounds, blue_fraction, valid_fraction


def measure_sc_blue_side_signature(
    image: np.ndarray,
    expected_bore_quads_uv: np.ndarray,
    ignored_pixels: np.ndarray | None = None,
    *,
    band_half_width_px: int,
    diagnostics: dict | None = None,
    min_blue_fraction: float = 0.08,
) -> ScBlueSideSignature | None:
    """Measure the four blue side bands of a visible SC adapter.

    This is used twice: once while the plug is clear to establish a stable
    per-camera reference, and once at a shallow stall.  It purposefully makes
    no assumption about the plug color or texture.
    """

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["reason"] = "initializing"
    measured = _recovery_blue_measurement(
        image,
        expected_bore_quads_uv,
        ignored_pixels,
        band_half_width_px=band_half_width_px,
        min_blue_fraction=min_blue_fraction,
        diagnostics=diagnostics,
    )
    if measured is None:
        return None
    blue, valid, bounds, blue_fraction, valid_fraction = measured
    band = int(band_half_width_px)
    x0, y0, x1, y1 = [int(round(value)) for value in bounds]
    # Canonical image top is physical +Y.  Preserve the command-frame order
    # [-X, +X, -Y, +Y], hence bottom precedes top in the final two entries.
    side_slices = _recovery_side_band_slices(bounds, band)
    side_supports = tuple(
        _readonly_bool_mask(valid[row_slice, column_slice])
        for row_slice, column_slice in side_slices
    )
    side_blue = tuple(
        _readonly_bool_mask(blue[row_slice, column_slice] & support)
        for (row_slice, column_slice), support in zip(side_slices, side_supports)
    )
    fractions = np.asarray(
        [
            float(np.count_nonzero(blue_side) / max(1, np.count_nonzero(support)))
            for support, blue_side in zip(side_supports, side_blue)
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(fractions)):
        if diagnostics is not None:
            diagnostics.update(reason="nonfinite_side_visibility")
        return None
    corridor_support = _readonly_bool_mask(
        valid[y0 - band:y1 + band, x0 - band:x1 + band]
    )
    side_support_fractions = np.asarray(
        [float(np.mean(support)) for support in side_supports], dtype=np.float64
    )
    signature = ScBlueSideSignature(
        blue_fractions=_readonly_float_array(fractions),
        blue_fraction=blue_fraction,
        valid_fraction=valid_fraction,
        side_support_masks=side_supports,
        side_blue_masks=side_blue,
        corridor_support_mask=corridor_support,
    )
    if diagnostics is not None:
        diagnostics.update(
            reason="accepted",
            blue_fractions=fractions.copy(),
            blue_fraction=blue_fraction,
            valid_fraction=valid_fraction,
            band_half_width_px=band,
            side_support_fractions=side_support_fractions,
            corridor_support_fraction=float(np.mean(corridor_support)),
        )
    return signature


def _signature_has_full_measurement_support(
    signature: ScBlueSideSignature,
) -> tuple[bool, dict] | None:
    """Return full-support status for a legacy fraction-only baseline path."""

    support_data = _signature_support_data(signature)
    if support_data is None:
        return None
    side_supports, _side_blue, corridor = support_data
    side_fractions = np.asarray(
        [float(np.mean(support)) for support in side_supports], dtype=np.float64
    )
    corridor_fraction = float(np.mean(corridor))
    return bool(
        np.all(side_fractions >= 1.0) and corridor_fraction >= 1.0
    ), {
        "current_support_fractions": side_fractions,
        "current_corridor_support_fraction": corridor_fraction,
    }


def _paired_side_blue_fractions(
    baseline: ScBlueSideSignature,
    current: ScBlueSideSignature,
    *,
    min_shared_support_fraction: float,
    min_baseline_side_fraction: float,
    min_blue_fraction: float,
    diagnostics: dict | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compare side visibility only over unchanged baseline/current support.

    The calibrated gripper mask is allowed to cover a stable part of a side
    band.  A changed mask is never treated as color evidence: even a very thin
    new masked strip could look exactly like the plug hiding the blue housing.
    """

    baseline_data = _signature_support_data(baseline)
    current_data = _signature_support_data(current)
    if baseline_data is None or current_data is None:
        if diagnostics is not None:
            diagnostics.update(reason="baseline_support_unavailable")
        return None
    baseline_supports, baseline_blue, baseline_corridor = baseline_data
    current_supports, current_blue, current_corridor = current_data
    if any(
        before.shape != after.shape
        for before, after in zip(baseline_supports, current_supports)
    ) or baseline_corridor.shape != current_corridor.shape:
        if diagnostics is not None:
            diagnostics.update(reason="baseline_support_geometry_changed")
        return None

    baseline_support_fractions = []
    current_support_fractions = []
    common_support_fractions = []
    new_occlusion_fractions = []
    new_visibility_fractions = []
    support_change_fractions = []
    common_supports = []
    for before, after in zip(baseline_supports, current_supports):
        common = before & after
        before_count = int(np.count_nonzero(before))
        after_count = int(np.count_nonzero(after))
        total_count = int(before.size)
        baseline_support_fractions.append(before_count / max(1, total_count))
        current_support_fractions.append(after_count / max(1, total_count))
        common_support_fractions.append(
            int(np.count_nonzero(common)) / max(1, total_count)
        )
        new_occlusion_fractions.append(
            int(np.count_nonzero(before & ~after)) / max(1, before_count)
        )
        new_visibility_fractions.append(
            int(np.count_nonzero(~before & after)) / max(1, after_count)
        )
        support_change_fractions.append(
            int(np.count_nonzero(before ^ after)) / max(1, total_count)
        )
        common_supports.append(common)

    corridor_common = baseline_corridor & current_corridor
    corridor_total = int(baseline_corridor.size)
    corridor_baseline_fraction = float(np.mean(baseline_corridor))
    corridor_current_fraction = float(np.mean(current_corridor))
    corridor_common_fraction = float(np.mean(corridor_common))
    corridor_new_occlusion_fraction = float(
        np.count_nonzero(baseline_corridor & ~current_corridor)
        / max(1, np.count_nonzero(baseline_corridor))
    )
    corridor_new_visibility_fraction = float(
        np.count_nonzero(~baseline_corridor & current_corridor)
        / max(1, np.count_nonzero(current_corridor))
    )
    corridor_change_fraction = float(
        np.count_nonzero(baseline_corridor ^ current_corridor)
        / max(1, corridor_total)
    )

    support_diagnostics = {
        "support_mode": "paired",
        "baseline_support_fractions": np.asarray(
            baseline_support_fractions, dtype=np.float64
        ),
        "current_support_fractions": np.asarray(
            current_support_fractions, dtype=np.float64
        ),
        "common_support_fractions": np.asarray(
            common_support_fractions, dtype=np.float64
        ),
        "new_occlusion_fractions": np.asarray(
            new_occlusion_fractions, dtype=np.float64
        ),
        "new_visibility_fractions": np.asarray(
            new_visibility_fractions, dtype=np.float64
        ),
        "support_change_fractions": np.asarray(
            support_change_fractions, dtype=np.float64
        ),
        "baseline_corridor_support_fraction": corridor_baseline_fraction,
        "current_corridor_support_fraction": corridor_current_fraction,
        "common_corridor_support_fraction": corridor_common_fraction,
        "new_corridor_occlusion_fraction": corridor_new_occlusion_fraction,
        "new_corridor_visibility_fraction": corridor_new_visibility_fraction,
        "corridor_support_change_fraction": corridor_change_fraction,
    }
    if diagnostics is not None:
        diagnostics.update(support_diagnostics)

    changed_support = (
        any(value > 0.0 for value in support_change_fractions)
        or corridor_change_fraction > 0.0
    )
    sufficient_common_support = (
        min(common_support_fractions) >= float(min_shared_support_fraction)
        and corridor_common_fraction >= float(min_shared_support_fraction)
    )
    if changed_support or not sufficient_common_support:
        if diagnostics is not None:
            diagnostics.update(
                reason="occluded_measurement_roi",
                support_change=(
                    "new_occlusion"
                    if any(value > 0.0 for value in new_occlusion_fractions)
                    or corridor_new_occlusion_fraction > 0.0
                    else "support_changed"
                ),
            )
        return None

    baseline_fractions = np.asarray(
        [
            float(np.count_nonzero(blue & common) / max(1, np.count_nonzero(common)))
            for blue, common in zip(baseline_blue, common_supports)
        ],
        dtype=np.float64,
    )
    current_fractions = np.asarray(
        [
            float(np.count_nonzero(blue & common) / max(1, np.count_nonzero(common)))
            for blue, common in zip(current_blue, common_supports)
        ],
        dtype=np.float64,
    )
    total_common = sum(int(np.count_nonzero(common)) for common in common_supports)
    current_blue_fraction = float(
        sum(
            int(np.count_nonzero(blue & common))
            for blue, common in zip(current_blue, common_supports)
        )
        / max(1, total_common)
    )
    if diagnostics is not None:
        diagnostics.update(
            paired_baseline_blue_fractions=baseline_fractions.copy(),
            paired_current_blue_fractions=current_fractions.copy(),
            paired_current_blue_fraction=current_blue_fraction,
        )
    if np.any(baseline_fractions < float(min_baseline_side_fraction)):
        if diagnostics is not None:
            diagnostics.update(
                reason="weak_baseline_side_visibility",
                baseline_blue_fractions=baseline_fractions.copy(),
            )
        return None
    if current_blue_fraction < float(min_blue_fraction):
        if diagnostics is not None:
            diagnostics.update(
                reason="blue_association_failed",
                blue_fraction=current_blue_fraction,
            )
        return None
    return baseline_fractions, current_fractions


def detect_sc_recovery_direction(
    image: np.ndarray,
    expected_bore_quads_uv: np.ndarray,
    ignored_pixels: np.ndarray | None = None,
    *,
    baseline_blue_fractions: dict[
        int, np.ndarray | ScBlueSideSignature
    ] | None = None,
    diagnostics: dict | None = None,
    min_blue_fraction: float = 0.08,
    min_baseline_side_fraction: float = _RECOVERY_MIN_BASELINE_SIDE_FRACTION,
    min_signal_norm: float = _RECOVERY_MIN_SIGNAL_NORM,
    min_variant_dot: float = 0.5,
) -> ScRecoveryEvidence | None:
    """Return a safe port-local escape direction from blue-side occlusion.

    At a shallow stall the plug often touches the blue adapter, making a
    geometric ``gap`` exactly zero on both sides.  Instead of trying to segment
    the plug, compare the current blue coverage in four canonical side bands
    with coverage captured from the *same camera* before approach.  A lower
    ratio on one side means that side is occluded, so the commanded correction
    points toward the opposing, more visible side.

    The two band widths are independent measurements.  Both must be directed
    and agree, or both must be balanced; mixed/contradictory evidence abstains.
    Rich ``ScBlueSideSignature`` baselines additionally compare immutable
    support/color masks, allowing a fixed partial gripper mask while rejecting
    a changed one.  Legacy fraction-only baselines retain the older fully-valid
    measurement gate.  This detector deliberately has no silhouette or
    blind-search fallback.
    """

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["reason"] = "initializing"
    if baseline_blue_fractions is None:
        if diagnostics is not None:
            diagnostics["reason"] = "baseline_unavailable"
        return None

    variants: list[tuple[int, ScBlueSideSignature, np.ndarray, np.ndarray]] = []
    support_by_band: dict[int, dict] = {}
    for band in _RECOVERY_SIDE_BAND_HALF_WIDTHS:
        try:
            baseline_record = baseline_blue_fractions[band]
            rich_baseline = (
                baseline_record
                if isinstance(baseline_record, ScBlueSideSignature)
                and _signature_support_data(baseline_record) is not None
                else None
            )
            baseline = np.asarray(
                (
                    baseline_record.blue_fractions
                    if isinstance(baseline_record, ScBlueSideSignature)
                    else baseline_record
                ),
                dtype=np.float64,
            ).reshape(4)
        except (KeyError, TypeError, ValueError):
            if diagnostics is not None:
                diagnostics.update(reason="baseline_unavailable", band_half_width_px=band)
            return None
        if (
            not np.all(np.isfinite(baseline))
            or np.any(baseline < float(min_baseline_side_fraction))
            or np.any(baseline > 1.0 + 1e-9)
        ):
            if diagnostics is not None:
                diagnostics.update(
                    reason="weak_baseline_side_visibility",
                    band_half_width_px=band,
                    baseline_blue_fractions=baseline.copy(),
                )
            return None
        local_diagnostics: dict = {}
        signature = measure_sc_blue_side_signature(
            image,
            expected_bore_quads_uv,
            ignored_pixels,
            band_half_width_px=band,
            diagnostics=local_diagnostics,
            min_blue_fraction=min_blue_fraction,
        )
        if signature is None:
            if diagnostics is not None:
                diagnostics.update(
                    reason=local_diagnostics.get("reason", "side_measurement_failed"),
                    band_half_width_px=band,
                    **{
                        key: value
                        for key, value in local_diagnostics.items()
                        if key not in {"reason", "band_half_width_px"}
                    },
                )
            return None
        if rich_baseline is not None:
            paired = _paired_side_blue_fractions(
                rich_baseline,
                signature,
                min_shared_support_fraction=(
                    _RECOVERY_MIN_SHARED_SIDE_SUPPORT_FRACTION
                ),
                min_baseline_side_fraction=min_baseline_side_fraction,
                min_blue_fraction=min_blue_fraction,
                diagnostics=local_diagnostics,
            )
            if paired is None:
                if diagnostics is not None:
                    diagnostics.update(
                        reason=local_diagnostics.get(
                            "reason", "side_measurement_failed"
                        ),
                        band_half_width_px=band,
                        **{
                            key: value
                            for key, value in local_diagnostics.items()
                            if key not in {"reason", "band_half_width_px"}
                        },
                    )
                return None
            baseline, current_fractions = paired
        else:
            full_support = _signature_has_full_measurement_support(signature)
            if full_support is None:
                if diagnostics is not None:
                    diagnostics.update(
                        reason="side_measurement_failed",
                        band_half_width_px=band,
                    )
                return None
            is_full_support, support_diagnostics = full_support
            local_diagnostics.update(
                support_mode="legacy_full_valid", **support_diagnostics
            )
            if not is_full_support:
                if diagnostics is not None:
                    diagnostics.update(
                        reason="occluded_measurement_roi",
                        band_half_width_px=band,
                        **{
                            key: value
                            for key, value in local_diagnostics.items()
                            if key not in {"reason", "band_half_width_px"}
                        },
                    )
                return None
            current_fractions = signature.blue_fractions
        if (
            not np.all(np.isfinite(baseline))
            or np.any(baseline < float(min_baseline_side_fraction))
            or np.any(baseline > 1.0 + 1e-9)
        ):
            if diagnostics is not None:
                diagnostics.update(
                    reason="weak_baseline_side_visibility",
                    band_half_width_px=band,
                    baseline_blue_fractions=baseline.copy(),
                )
            return None
        # The denominator floor prevents an accidental near-zero reference from
        # amplifying noise.  It cannot make a weak baseline acceptable because
        # the explicit >=0.50 gate above is much stricter.
        ratios = current_fractions / np.maximum(baseline, 0.05)
        visible_direction = np.asarray(
            [ratios[1] - ratios[0], ratios[3] - ratios[2]], dtype=np.float64
        )
        variants.append((band, signature, ratios, visible_direction))
        support_by_band[band] = {
            key: value for key, value in local_diagnostics.items() if key != "reason"
        }

    signal_norms = np.asarray(
        [np.linalg.norm(item[3]) for item in variants], dtype=np.float64
    )
    if not np.all(np.isfinite(signal_norms)):
        if diagnostics is not None:
            diagnostics["reason"] = "nonfinite_visibility_signal"
        return None
    directed = signal_norms >= float(min_signal_norm)
    if not np.any(directed):
        # A paired balanced observation is a strong instruction to stop lateral
        # recovery and resume the regular compliant seating controller.
        side_ratios = np.mean([item[2] for item in variants], axis=0)
        result = ScRecoveryEvidence(
            direction_xy=np.zeros(2, dtype=np.float64),
            confidence=float(
                np.clip(
                    1.0 - float(np.max(signal_norms)) / max(min_signal_norm, 1e-9),
                    0.0,
                    1.0,
                )
            ),
            margins=np.asarray(side_ratios, dtype=np.float64),
            blue_fraction=float(np.mean([item[1].blue_fraction for item in variants])),
            valid_fraction=float(np.mean([item[1].valid_fraction for item in variants])),
            balanced=True,
        )
        if diagnostics is not None:
            diagnostics.update(
                reason="balanced",
                visibility_ratios=result.margins.copy(),
                signal_norms=signal_norms.copy(),
                support_by_band=support_by_band,
            )
        return result
    if not np.all(directed):
        if diagnostics is not None:
            diagnostics.update(
                reason="direction_unstable",
                signal_norms=signal_norms.copy(),
                directed_variants=directed.copy(),
                support_by_band=support_by_band,
            )
        return None

    units = [item[3] / norm for item, norm in zip(variants, signal_norms)]
    agreement = float(np.dot(units[0], units[1]))
    if agreement < float(min_variant_dot):
        if diagnostics is not None:
            diagnostics.update(
                reason="direction_unstable",
                signal_norms=signal_norms.copy(),
                variant_dot=agreement,
                support_by_band=support_by_band,
            )
        return None

    # Ratios are dimensionless and their size is not a displacement estimate.
    # Convert only the *direction* to port metric axes before normalizing, so a
    # diagonal does not accidentally treat 1 canonical Y pixel as 1 X pixel.
    visible_mean = np.mean([item[3] for item in variants], axis=0)
    metric_direction = visible_mean * np.asarray(
        [
            _RECOVERY_OPENING_WIDTH_M / _RECOVERY_OPENING_WIDTH_PX,
            _RECOVERY_OPENING_HEIGHT_M / _RECOVERY_OPENING_HEIGHT_PX,
        ],
        dtype=np.float64,
    )
    metric_norm = float(np.linalg.norm(metric_direction))
    if not np.isfinite(metric_norm) or metric_norm <= 1e-12:
        if diagnostics is not None:
            diagnostics["reason"] = "nonfinite_visibility_direction"
        return None
    side_ratios = np.mean([item[2] for item in variants], axis=0)
    confidence = float(
        np.clip(
            0.30
            * min(
                1.0,
                float(np.mean([item[1].blue_fraction for item in variants]))
                / max(min_blue_fraction, 1e-9),
            )
            + 0.30 * np.clip(float(np.min(signal_norms)) / 0.15, 0.0, 1.0)
            + 0.40 * np.clip((agreement - min_variant_dot) / max(1.0 - min_variant_dot, 1e-9), 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    result = ScRecoveryEvidence(
        direction_xy=metric_direction / metric_norm,
        confidence=confidence,
        margins=np.asarray(side_ratios, dtype=np.float64),
        blue_fraction=float(np.mean([item[1].blue_fraction for item in variants])),
        valid_fraction=float(np.mean([item[1].valid_fraction for item in variants])),
        balanced=False,
    )
    if diagnostics is not None:
        diagnostics.update(
            reason="accepted",
            direction_xy=result.direction_xy.copy(),
            confidence=result.confidence,
            visibility_ratios=result.margins.copy(),
            signal_norms=signal_norms.copy(),
            variant_dot=agreement,
            band_half_widths=np.asarray(_RECOVERY_SIDE_BAND_HALF_WIDTHS),
            support_by_band=support_by_band,
        )
    return result


def fuse_sc_recovery_evidence(
    view_evidence: list[dict],
    *,
    min_views: int = 2,
    min_pairwise_dot: float = 0.5,
    min_resultant: float = 0.75,
) -> ScRecoveryEstimate | None:
    """Fuse relative camera directions; conflict always means no motion."""

    usable = []
    for item in view_evidence:
        evidence = item.get("evidence")
        if not isinstance(evidence, ScRecoveryEvidence):
            continue
        direction = np.asarray(evidence.direction_xy, dtype=np.float64).reshape(2)
        if (
            not np.all(np.isfinite(direction))
            or not np.isfinite(evidence.confidence)
            or evidence.confidence <= 0.0
        ):
            continue
        usable.append((str(item.get("camera", "unknown")), evidence, direction))
    if len(usable) < max(2, int(min_views)):
        return None

    balanced = [item for item in usable if item[1].balanced]
    directed = [item for item in usable if not item[1].balanced]
    if len(balanced) >= min_views and not directed:
        confidence = float(np.mean([item[1].confidence for item in balanced]))
        return ScRecoveryEstimate(
            direction_xy=np.zeros(2, dtype=np.float64),
            confidence=confidence,
            cameras=tuple(item[0] for item in balanced),
            resultant=1.0,
            balanced=True,
        )
    if len(directed) < min_views or balanced:
        return None

    units = []
    weights = []
    cameras = []
    for camera, evidence, direction in directed:
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return None
        units.append(direction / norm)
        weights.append(float(evidence.confidence))
        cameras.append(camera)
    for first, second in itertools.combinations(units, 2):
        if float(np.dot(first, second)) < min_pairwise_dot:
            return None

    units_array = np.asarray(units, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    vector = np.sum(weights_array[:, None] * units_array, axis=0)
    weight_sum = float(np.sum(weights_array))
    vector_norm = float(np.linalg.norm(vector))
    resultant = vector_norm / max(weight_sum, 1e-9)
    if resultant < min_resultant or vector_norm <= 1e-9:
        return None
    direction = vector / vector_norm
    confidence = float(np.clip(resultant * np.mean(weights_array), 0.0, 1.0))
    return ScRecoveryEstimate(
        direction_xy=direction,
        confidence=confidence,
        cameras=tuple(cameras),
        resultant=resultant,
        balanced=False,
    )


def bounded_recovery_offset_update(
    current_offset_xy: np.ndarray,
    direction_xy: np.ndarray,
    path_m: float,
    *,
    max_step_m: float,
    max_total_m: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Advance a recovery offset inside both path-length and radius budgets.

    Limiting radius alone would allow an endless oscillation.  Limiting path
    alone can cross the lateral safety circle on a diagonal.  The same small
    budget is therefore enforced independently as cumulative travel and as
    excursion from the frozen pre-alignment target.
    """

    current = np.asarray(current_offset_xy, dtype=np.float64).reshape(2)
    direction = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    if (
        not np.all(np.isfinite(current))
        or not np.all(np.isfinite(direction))
        or not np.isfinite(path_m)
        or max_step_m <= 0.0
        or max_total_m <= 0.0
    ):
        return None
    direction_norm = float(np.linalg.norm(direction))
    current_norm = float(np.linalg.norm(current))
    if direction_norm <= 1e-12 or current_norm > max_total_m + 1e-9:
        return None
    unit = direction / direction_norm
    path_remaining = float(max_total_m) - max(0.0, float(path_m))
    allowed = min(float(max_step_m), path_remaining)
    if allowed <= 1e-12:
        return None

    # Positive root of ||current + t*unit|| == max_total_m.
    projection = float(np.dot(current, unit))
    discriminant = projection * projection + (
        float(max_total_m) ** 2 - current_norm * current_norm
    )
    radial_remaining = -projection + np.sqrt(max(0.0, discriminant))
    allowed = min(allowed, float(radial_remaining))
    if allowed <= 1e-12:
        return None
    step = unit * allowed
    candidate = current + step
    return candidate, step, float(path_m) + allowed


def detect_sc_duplex_opening(
    image: np.ndarray,
    expected_bore_quads_uv: np.ndarray,
    ignored_pixels: np.ndarray | None = None,
    *,
    diagnostics: dict | None = None,
    search_scale: float = 1.65,
    min_blue_fraction: float = 0.01,
) -> ScDuplexOpeningDetection | None:
    """Detect the centre of the two physical SC bores.

    The housing is asymmetric, so its blue-blob centroid is deliberately never
    used as the insertion target.  Blue is only an association gate.  Each dark
    bore is localized independently near its CAD projection; when only one bore
    remains visible, the known projected bore-to-duplex offset recovers the
    centre with lower confidence.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must be BGR or BGRA")
    bgr = array[:, :, :3]
    bore_quads = np.asarray(expected_bore_quads_uv, dtype=np.float64)
    if bore_quads.shape != (2, 4, 2):
        raise ValueError("expected_bore_quads_uv must have shape 2x4x2")
    bore_quads = np.stack([_finite_quad(quad) for quad in bore_quads])
    if not 1.0 < search_scale <= 3.0:
        raise ValueError("search_scale must be in (1, 3]")

    height, width = bgr.shape[:2]
    if ignored_pixels is not None:
        ignored = np.asarray(ignored_pixels, dtype=bool)
        if ignored.shape != (height, width):
            raise ValueError("ignored_pixels shape must match the image")
    else:
        ignored = None

    expected_center = bore_quads.mean(axis=(0, 1))
    all_points = bore_quads.reshape(-1, 2)
    search_points = expected_center + search_scale * (all_points - expected_center)
    search_hull = cv2.convexHull(np.rint(search_points).astype(np.int32))
    search_hull[:, 0, 0] = np.clip(search_hull[:, 0, 0], 0, width - 1)
    search_hull[:, 0, 1] = np.clip(search_hull[:, 0, 1], 0, height - 1)
    valid = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(valid, search_hull, 255)
    if ignored is not None:
        valid[ignored] = 0
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 25:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(reason="too_few_valid_pixels", valid_pixels=valid_count)
        return None

    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, SC_BLUE_LOWER, SC_BLUE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    blue_fraction = float(np.count_nonzero((blue != 0) & (valid != 0)) / valid_count)
    if blue_fraction < min_blue_fraction:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(
                reason="blue_association_failed",
                blue_fraction=blue_fraction,
                valid_pixels=valid_count,
            )
        return None

    bore_detections = []
    bore_diagnostics = []
    for quad in bore_quads:
        diag = {}
        detection = detect_dark_port_opening(
            bgr,
            quad,
            ignored,
            diagnostics=diag,
            search_scale=search_scale,
            min_area_ratio=0.12,
            max_area_ratio=1.45,
            max_center_distance=0.85,
        )
        bore_detections.append(detection)
        bore_diagnostics.append(diag)

    visible = [index for index, detection in enumerate(bore_detections)
               if detection is not None]
    if not visible:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(
                reason="no_dark_bore",
                blue_fraction=blue_fraction,
                bores=bore_diagnostics,
            )
        return None

    centers: list[np.ndarray | None] = [
        None if detection is None else np.asarray(detection.center_uv, dtype=np.float64)
        for detection in bore_detections
    ]
    if len(visible) == 2:
        observed_delta = centers[1] - centers[0]
        expected_delta = bore_quads[1].mean(axis=0) - bore_quads[0].mean(axis=0)
        expected_separation = float(np.linalg.norm(expected_delta))
        observed_separation = float(np.linalg.norm(observed_delta))
        separation_ratio = observed_separation / max(expected_separation, 1e-6)
        if not 0.55 <= separation_ratio <= 1.55:
            if diagnostics is not None:
                diagnostics.clear()
                diagnostics.update(
                    reason="bore_pair_geometry",
                    blue_fraction=blue_fraction,
                    separation_ratio=separation_ratio,
                    bores=bore_diagnostics,
                )
            return None
        center_uv = 0.5 * (centers[0] + centers[1])
        score = float(np.mean([detection.score for detection in bore_detections]))
        reason = "accepted_pair"
    else:
        index = visible[0]
        # Infer the duplex centre from the observed bore and its projected CAD
        # offset.  This remains directed when one bore is hidden by the plug.
        projected_bore_center = bore_quads[index].mean(axis=0)
        center_uv = centers[index] + (expected_center - projected_bore_center)
        score = float(bore_detections[index].score + 0.5)
        separation_ratio = float("nan")
        reason = "accepted_single_bore"

    result = ScDuplexOpeningDetection(
        center_uv=np.asarray(center_uv, dtype=np.float64),
        bore_centers_uv=(centers[0], centers[1]),
        detected_bores=len(visible),
        blue_fraction=blue_fraction,
        score=score,
    )
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            reason=reason,
            center_uv=result.center_uv.copy(),
            blue_fraction=blue_fraction,
            detected_bores=len(visible),
            separation_ratio=separation_ratio,
            score=score,
            bores=bore_diagnostics,
        )
    return result


def project_point_px(projection: np.ndarray, point_world: np.ndarray) -> np.ndarray | None:
    """Project a world point, returning ``None`` behind/at the camera."""

    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    projected = np.asarray(projection, dtype=np.float64).reshape(3, 4) @ np.array(
        [point[0], point[1], point[2], 1.0], dtype=np.float64
    )
    if not np.all(np.isfinite(projected)) or projected[2] <= 1e-6:
        return None
    return projected[:2] / projected[2]


def ray_to_plane(
    uv: np.ndarray,
    K: np.ndarray,
    T_cam_from_world: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Intersect an image ray with a world-frame plane."""

    try:
        T_world_from_cam = np.linalg.inv(
            np.asarray(T_cam_from_world, dtype=np.float64).reshape(4, 4)
        )
        ray_cam = np.linalg.solve(
            np.asarray(K, dtype=np.float64).reshape(3, 3),
            np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64),
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    origin = T_world_from_cam[:3, 3]
    ray_world = T_world_from_cam[:3, :3] @ ray_cam
    ray_norm = float(np.linalg.norm(ray_world))
    if ray_norm <= 1e-9:
        return None
    ray_world /= ray_norm
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    denominator = float(np.dot(normal, ray_world))
    if abs(denominator) <= 1e-6:
        return None
    distance = float(
        np.dot(normal, np.asarray(plane_point, dtype=np.float64).reshape(3) - origin)
        / denominator
    )
    if not np.isfinite(distance) or distance <= 0.0:
        return None
    return origin + distance * ray_world


def fuse_sc_opening_hits(
    view_hits: list[dict],
    *,
    origin_port_pos: np.ndarray,
    Rp: np.ndarray,
    max_view_disagreement_m: float,
    max_total_offset_m: float,
    allow_single_view: bool = True,
) -> ScOpeningEstimate | None:
    """Fuse camera ray hits using the closest agreeing pair.

    One high-confidence view remains usable when the other wrists are occluded.
    A disagreement between multiple views does not silently choose a winner; it
    returns ``None`` so the controller retains its last valid target.
    """

    if not view_hits:
        return None
    origin = np.asarray(origin_port_pos, dtype=np.float64).reshape(3)
    frame = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
    points = [np.asarray(hit["plane_point"], dtype=np.float64).reshape(3)
              for hit in view_hits]
    cameras = [str(hit.get("camera", "unknown")) for hit in view_hits]

    if len(points) == 1:
        if not allow_single_view:
            return None
        point = points[0]
        disagreement = float("nan")
        used_cameras = (cameras[0],)
        single_view = True
    else:
        best_pair = None
        for i, j in itertools.combinations(range(len(points)), 2):
            delta = frame.T @ (points[i] - points[j])
            disagreement = float(np.linalg.norm(delta[:2]))
            if best_pair is None or disagreement < best_pair[0]:
                best_pair = (disagreement, i, j)
        if best_pair is None or best_pair[0] > max_view_disagreement_m:
            return None
        disagreement, i, j = best_pair
        kept_indices = [i, j]
        pair_mean = 0.5 * (points[i] + points[j])
        for index, point_candidate in enumerate(points):
            if index in kept_indices:
                continue
            delta = frame.T @ (point_candidate - pair_mean)
            if float(np.linalg.norm(delta[:2])) <= max_view_disagreement_m:
                kept_indices.append(index)
        point = np.mean([points[index] for index in kept_indices], axis=0)
        used_cameras = tuple(cameras[index] for index in kept_indices)
        single_view = False

    offset = frame.T @ (point - origin)
    if not np.all(np.isfinite(offset)) or float(np.linalg.norm(offset[:2])) > max_total_offset_m:
        return None
    point_on_origin_plane = point - frame[:, 2] * float(offset[2])
    return ScOpeningEstimate(
        point_world=point_on_origin_plane,
        cameras=used_cameras,
        disagreement_m=disagreement,
        single_view=single_view,
    )


def bounded_visual_port_update(
    current_port_pos: np.ndarray,
    origin_port_pos: np.ndarray,
    observed_port_pos: np.ndarray,
    Rp: np.ndarray,
    *,
    max_step_m: float,
    max_total_m: float,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a clipped lateral target and applied port-frame correction."""

    current = np.asarray(current_port_pos, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_port_pos, dtype=np.float64).reshape(3)
    observed = np.asarray(observed_port_pos, dtype=np.float64).reshape(3)
    frame = np.asarray(Rp, dtype=np.float64).reshape(3, 3)
    correction = (frame.T @ (observed - current))[:2]
    norm = float(np.linalg.norm(correction))
    if not np.isfinite(norm):
        return None
    step_limit = max(0.0, float(max_step_m) * float(step_scale))
    step = correction.copy()
    if norm > step_limit > 0.0:
        step *= step_limit / norm
    elif step_limit <= 0.0 and norm > 0.0:
        return None
    candidate = current + frame[:, 0] * step[0] + frame[:, 1] * step[1]
    total = (frame.T @ (candidate - origin))[:2]
    total_norm = float(np.linalg.norm(total))
    if total_norm > max_total_m:
        # Saturate at the immutable excursion boundary instead of crossing it.
        if total_norm <= 1e-12 or max_total_m <= 0.0:
            return None
        total *= float(max_total_m) / total_norm
        candidate = origin + frame[:, 0] * total[0] + frame[:, 1] * total[1]
        step = (frame.T @ (candidate - current))[:2]
    return candidate, step


__all__ = [
    "ScDuplexOpeningDetection",
    "ScBlueSideSignature",
    "ScOpeningEstimate",
    "ScRecoveryEvidence",
    "ScRecoveryEstimate",
    "aggregate_sc_blue_side_signatures",
    "bounded_recovery_offset_update",
    "bounded_visual_port_update",
    "detect_sc_duplex_opening",
    "detect_sc_recovery_direction",
    "fuse_sc_opening_hits",
    "fuse_sc_recovery_evidence",
    "measure_sc_blue_side_signature",
    "project_point_px",
    "ray_to_plane",
]
