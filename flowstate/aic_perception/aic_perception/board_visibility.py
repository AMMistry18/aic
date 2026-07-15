"""Detect and frame the dark AIC task board using permitted camera data.

This module is deliberately independent of ROS so the segmentation and steering
logic can be unit-tested with synthetic or saved camera images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MaskReport:
    """Board segmentation result for one camera image."""

    seen: bool
    full: bool
    edges: frozenset[str] = field(default_factory=frozenset)
    area_frac: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    rectangularity: float = 0.0
    centroid: tuple[float, float] | None = None
    # Clear space from the board component to each physical image edge.  The
    # separately reported artificial_bottom_contact records overlap with the
    # masked gripper band.
    clearance_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # The required clearance grows with projected board size.  This reserves
    # image context for components which protrude above the plate.
    context_pad_px: float = 0.0
    context_ok: bool = False
    detail_ok: bool = False
    shape_ok: bool = False
    artificial_bottom_contact: bool = False
    center_error: tuple[float, float] = (0.0, 0.0)
    clearance_score: float = 0.0
    detail_score: float = 0.0
    shape_score: float = 0.0
    center_score: float = 0.0
    quality_score: float = 0.0
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchAction:
    """Pure image-space search recommendation.

    ``mode`` is deliberately controller-agnostic.  The motion layer can map
    ``translate`` into camera-plane motion, ``backoff``/``approach`` into
    optical-axis motion, and ``reorient`` into a bounded wrist-pose change.
    """

    mode: str
    direction_image: tuple[float, float] = (0.0, 0.0)
    backoff: bool = False
    step_scale: float = 0.0
    reason: str = ""


def analyze_board(
    image: np.ndarray,
    margin_px: int = 15,
    min_area_frac: float = 0.005,
    ignore_bottom_frac: float = 0.15,
    morph_px: int = 5,
    min_contrast: float = 30.0,
    min_rectangularity: float = 0.60,
    min_detail_area_frac: float = 0.02,
    context_pad_frac: float = 0.10,
) -> MaskReport:
    """Return whether the whole dark task board is visible in ``image``.

    The mask uses a global Otsu inverse threshold. A contrast guard prevents
    Otsu from inventing a foreground split in an essentially uniform frame.
    Only the largest surviving connected component participates in edge tests.
    """
    import cv2

    img = np.asarray(image)
    if img.size == 0 or img.ndim not in (2, 3):
        raise ValueError("image must be a non-empty gray or color array")
    if img.ndim == 3 and img.shape[2] not in (3, 4):
        raise ValueError("color image must have three or four channels")
    if img.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if margin_px < 0 or morph_px <= 0:
        raise ValueError("margin_px must be non-negative and morph_px positive")
    if not 0.0 <= ignore_bottom_frac < 1.0:
        raise ValueError("ignore_bottom_frac must be in [0, 1)")
    if not 0.0 <= min_area_frac <= 1.0:
        raise ValueError("min_area_frac must be in [0, 1]")
    if not 0.0 <= min_detail_area_frac <= 1.0:
        raise ValueError("min_detail_area_frac must be in [0, 1]")
    if not 0.0 <= min_rectangularity <= 1.0:
        raise ValueError("min_rectangularity must be in [0, 1]")
    if not 0.0 <= context_pad_frac <= 1.0:
        raise ValueError("context_pad_frac must be in [0, 1]")

    if img.ndim == 2:
        gray = img.copy()
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    if height < 2 or width < 2:
        return MaskReport(seen=False, full=False)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    foreground = gray[mask > 0]
    background = gray[mask == 0]
    if (
        foreground.size == 0
        or background.size == 0
        or float(background.mean()) - float(foreground.mean()) < min_contrast
    ):
        return MaskReport(seen=False, full=False)

    band_y = height
    if ignore_bottom_frac > 0.0:
        band_y = max(1, int(height * (1.0 - ignore_bottom_frac)))
        mask[band_y:, :] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(morph_px), int(morph_px))
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Closing can grow foreground back into the deliberately excluded band.
    # Keep its boundary exact so contact with it remains a meaningful signal.
    if band_y < height:
        mask[band_y:, :] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return MaskReport(seen=False, full=False)

    component_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[component_index, cv2.CC_STAT_AREA])
    if area < min_area_frac * height * width:
        return MaskReport(seen=False, full=False)

    x0 = int(stats[component_index, cv2.CC_STAT_LEFT])
    y0 = int(stats[component_index, cv2.CC_STAT_TOP])
    x1 = x0 + int(stats[component_index, cv2.CC_STAT_WIDTH]) - 1
    y1 = y0 + int(stats[component_index, cv2.CC_STAT_HEIGHT]) - 1
    bbox_width = x1 - x0 + 1
    bbox_height = y1 - y0 + 1
    usable_bottom = min(band_y, height) - 1
    clearances = (
        float(x0),
        float(width - 1 - x1),
        float(y0),
        float(height - 1 - y1),
    )
    context_pad_px = max(
        float(margin_px),
        float(context_pad_frac) * float(max(bbox_width, bbox_height)),
    )

    edges: set[str] = set()
    for edge, clearance in zip(
        ("left", "right", "top", "bottom"), clearances
    ):
        if clearance <= context_pad_px:
            edges.add(edge)

    component = (labels == component_index).astype(np.uint8)

    # Do not let a narrow arm/finger bridge turn the whole board component into
    # a bottom-contacting blob.  The ignored band is deliberately opaque to
    # the detector, so the only useful evidence at that boundary is whether
    # the *broad body* of the largest component reaches it.  A horizontal
    # opening severs narrow vertical appendages while leaving an actually
    # bottom-clipped board intact.  Selecting the largest surviving core also
    # handles a thin bridge which widens into the gripper at the crop boundary.
    #
    # The opening width is deliberately modest (12% of a robust component-row
    # width, and at least two morphology kernels).  Thus ordinary perspective
    # taper and every substantial board contact still veto ``full``; only a
    # genuinely narrow connection can be ignored.
    artificial_bottom_contact = False
    raw_bottom_contact = bool(band_y < height and y1 >= usable_bottom)
    if raw_bottom_contact:
        component_roi = component[y0 : y1 + 1, x0 : x1 + 1]
        row_widths = np.count_nonzero(component_roi, axis=1)
        positive_row_widths = row_widths[row_widths > 0]
        if positive_row_widths.size == 0:
            # This should be unreachable for a selected connected component,
            # but failing closed is safer than declaring a complete view.
            artificial_bottom_contact = True
        else:
            reference_width = float(np.percentile(positive_row_widths, 75.0))
            bridge_width_px = max(
                2 * int(morph_px) + 1,
                int(np.ceil(0.12 * reference_width)),
            )
            bridge_width_px = min(
                max(1, int(round(reference_width))), bridge_width_px
            )
            broad_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (bridge_width_px, 1)
            )
            broad_component = cv2.morphologyEx(
                component, cv2.MORPH_OPEN, broad_kernel
            )
            broad_count, broad_labels, broad_stats, _ = (
                cv2.connectedComponentsWithStats(broad_component, 8)
            )
            if broad_count <= 1:
                artificial_bottom_contact = True
            else:
                broad_index = (
                    int(np.argmax(broad_stats[1:, cv2.CC_STAT_AREA])) + 1
                )
                broad_core = broad_labels == broad_index
                artificial_bottom_contact = bool(
                    np.any(broad_core[usable_bottom, :])
                )

    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rectangularity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        rect_width, rect_height = cv2.minAreaRect(contour)[1]
        rect_area = float(rect_width * rect_height)
        if rect_area > 0.0:
            rectangularity = min(1.0, area / rect_area)

    area_frac = area / float(height * width)
    context_ok = not edges
    detail_ok = area_frac >= min_detail_area_frac
    shape_ok = rectangularity >= min_rectangularity

    cx, cy = (float(value) for value in centroids[component_index])
    usable_width = max(1.0, float(width - 1))
    usable_height = max(1.0, float(usable_bottom))
    center_error = (
        (cx - 0.5 * usable_width) / (0.5 * usable_width),
        (cy - 0.5 * usable_height) / (0.5 * usable_height),
    )
    center_distance = float(np.linalg.norm(center_error))

    if context_pad_px > 0.0:
        clearance_score = float(
            np.clip(min(clearances) / context_pad_px, 0.0, 1.0)
        )
    else:
        clearance_score = 1.0
    if min_detail_area_frac > 0.0:
        detail_score = float(
            np.clip(area_frac / min_detail_area_frac, 0.0, 1.0)
        )
    else:
        detail_score = 1.0
    if min_rectangularity < 1.0:
        shape_score = float(
            np.clip(
                (rectangularity - min_rectangularity)
                / max(1e-9, 1.0 - min_rectangularity),
                0.0,
                1.0,
            )
        )
    else:
        shape_score = 1.0 if rectangularity >= 1.0 else 0.0
    center_score = float(np.clip(1.0 - center_distance, 0.0, 1.0))
    quality_score = float(
        np.clip(
            0.40 * clearance_score
            + 0.20 * detail_score
            + 0.25 * shape_score
            + 0.15 * center_score,
            0.0,
            1.0,
        )
    )

    failure_reasons: list[str] = []
    if not context_ok:
        failure_reasons.append("context_clipped")
    if artificial_bottom_contact:
        failure_reasons.append("artificial_bottom_contact")
    if not detail_ok:
        failure_reasons.append("insufficient_detail")
    if not shape_ok:
        failure_reasons.append("nonrectangular_board")

    full = bool(
        context_ok
        and detail_ok
        and shape_ok
        and not artificial_bottom_contact
    )
    return MaskReport(
        seen=True,
        full=full,
        edges=frozenset(edges),
        area_frac=area_frac,
        bbox=(x0, y0, x1, y1),
        rectangularity=float(rectangularity),
        centroid=(cx, cy),
        clearance_px=clearances,
        context_pad_px=float(context_pad_px),
        context_ok=context_ok,
        detail_ok=detail_ok,
        shape_ok=shape_ok,
        artificial_bottom_contact=artificial_bottom_contact,
        center_error=(float(center_error[0]), float(center_error[1])),
        clearance_score=clearance_score,
        detail_score=detail_score,
        shape_score=shape_score,
        center_score=center_score,
        quality_score=quality_score,
        failure_reasons=tuple(failure_reasons),
    )


def view_quality(report: MaskReport) -> float:
    """Return a bounded continuous view score for ranking and progress checks."""

    if not report.seen:
        return 0.0
    return float(np.clip(report.quality_score, 0.0, 1.0))


def search_progress(
    reports: Sequence[MaskReport],
    *,
    window: int = 4,
    min_gain: float = 0.01,
) -> str:
    """Classify recent visual progress without imposing an iteration limit."""

    if window < 2:
        raise ValueError("window must be at least 2")
    if min_gain < 0.0:
        raise ValueError("min_gain must be non-negative")
    if any(report.full for report in reports):
        return "complete"
    recent = list(reports[-window:])
    if len(recent) < 2 or any(not report.seen for report in recent):
        return "unknown"
    gain = view_quality(recent[-1]) - view_quality(recent[0])
    if gain >= min_gain:
        return "improving"
    if gain <= -min_gain:
        return "regressing"
    return "stalled"


def _continuous_direction(report: MaskReport) -> np.ndarray:
    """Return a normalized camera-plane correction from clearance deficits."""

    pad = float(report.context_pad_px)
    if pad > 0.0 and len(report.clearance_px) == 4:
        left, right, top, bottom = report.clearance_px
        deficits = np.maximum(
            0.0,
            pad - np.asarray([left, right, top, bottom], dtype=float),
        ) / pad
        direction = np.array(
            [deficits[1] - deficits[0], deficits[3] - deficits[2]],
            dtype=float,
        )
    else:
        direction = np.zeros(2, dtype=float)

    if float(np.linalg.norm(direction)) < 1e-9 and report.edges:
        direction, _ = decide_direction(report)
    if float(np.linalg.norm(direction)) < 1e-9:
        direction = np.asarray(report.center_error, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm > 0.0:
        direction /= norm
    return direction


def adaptive_action(
    report: MaskReport,
    history: Sequence[MaskReport] = (),
) -> SearchAction:
    """Recommend the next perception action from current evidence and history.

    Repeated edge labels do not force repeated translations forever.  When the
    same clipped edge persists while apparent area grows, the camera is likely
    moving closer or along an ineffective plane, so the recommendation changes
    to a retreat.  Conversely, any length of genuinely improving history remains
    eligible for another correction; there is no trial-count stop in this helper.
    """

    if not report.seen:
        return SearchAction(mode="hold", reason="board_not_seen")
    if report.full:
        return SearchAction(mode="done", reason="view_complete")

    direction = _continuous_direction(report)
    _, opposite_edge_backoff = decide_direction(report)
    recent = [*history[-3:], report]
    progress = search_progress(recent)
    same_edges = bool(report.edges) and all(
        item.edges == report.edges for item in recent
    )
    area_growth = bool(
        len(recent) >= 3
        and recent[0].area_frac > 0.0
        and report.area_frac > 1.08 * recent[0].area_frac
    )

    if opposite_edge_backoff or (
        len(recent) >= 3
        and same_edges
        and (progress in {"stalled", "regressing"} or area_growth)
    ):
        reason = (
            "opposite_context_edges"
            if opposite_edge_backoff
            else "persistent_edge_without_clearance"
        )
        return SearchAction(
            mode="backoff",
            backoff=True,
            step_scale=1.0,
            reason=reason,
        )

    if not report.detail_ok and report.context_ok:
        return SearchAction(
            mode="approach",
            direction_image=(float(direction[0]), float(direction[1])),
            step_scale=float(np.clip(1.0 - report.detail_score, 0.25, 1.0)),
            reason="insufficient_detail",
        )
    if not report.shape_ok and report.context_ok:
        return SearchAction(
            mode="reorient",
            direction_image=(float(direction[0]), float(direction[1])),
            step_scale=float(np.clip(1.0 - report.shape_score, 0.25, 1.0)),
            reason="poor_board_aspect",
        )

    pad = max(report.context_pad_px, 1.0)
    deficit = max(
        (max(0.0, pad - value) / pad for value in report.clearance_px),
        default=0.0,
    )
    return SearchAction(
        mode="translate",
        direction_image=(float(direction[0]), float(direction[1])),
        step_scale=float(np.clip(max(deficit, 0.25), 0.25, 1.0)),
        reason=f"context_correction_{progress}",
    )


def decide_direction(report: MaskReport) -> tuple[np.ndarray, bool]:
    """Return camera-plane translation direction and whether to back away.

    Image axes follow the optical-frame convention: +x is image-right and +y
    is image-down. A board clipped on the left therefore asks the camera to
    translate along -x. Opposite edges indicate insufficient standoff.
    """
    if not report.seen or report.full:
        return np.zeros(2, dtype=float), False

    edges = report.edges
    backoff = ("left" in edges and "right" in edges) or (
        "top" in edges and "bottom" in edges
    )
    direction = np.array(
        [
            (-1.0 if "left" in edges else 0.0)
            + (1.0 if "right" in edges else 0.0),
            (-1.0 if "top" in edges else 0.0)
            + (1.0 if "bottom" in edges else 0.0),
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(direction))
    if norm > 0.0:
        direction /= norm
    return direction, backoff


def world_delta(
    direction_image: np.ndarray,
    backoff: bool,
    step_m: float,
    base_image_right: np.ndarray,
    base_image_down: np.ndarray,
    base_backoff: np.ndarray,
    backoff_step_m: float | None = None,
) -> np.ndarray:
    """Convert a camera-plane direction into a base-frame translation."""

    if step_m <= 0.0:
        raise ValueError("step_m must be positive")
    direction = np.asarray(direction_image, dtype=float)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction_image must be a finite two-vector")

    def unit(vector: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite three-vector")
        norm = float(np.linalg.norm(value))
        if norm < 1e-9:
            raise ValueError(f"{name} must be nonzero")
        return value / norm

    right = unit(base_image_right, "base_image_right")
    down = unit(base_image_down, "base_image_down")
    away = unit(base_backoff, "base_backoff")
    if abs(float(np.dot(right, down))) > 0.2:
        raise ValueError("image right/down axes are not sufficiently orthogonal")

    if backoff:
        retreat = step_m if backoff_step_m is None else backoff_step_m
        if retreat <= 0.0:
            raise ValueError("backoff_step_m must be positive")
        # When the board spans opposite edges, distance is the dominant issue.
        # A pure retreat avoids combining two motions into a longer diagonal.
        return retreat * away
    return step_m * (direction[0] * right + direction[1] * down)


def combine_cameras(
    reports: Mapping[str, MaskReport],
) -> tuple[bool, str | None, MaskReport | None]:
    """Choose a complete view, or the easiest partial view to recover.

    A large visible area is not necessarily the best steering signal: a close
    camera can contain a large board blob while clipping it on three sides.
    Prefer views that touch the fewest edges, and prefer an ordinary
    image-plane correction over a retreat when another camera offers one.
    Area and rectangularity are only tie-breakers.
    """
    if not reports:
        return False, None, None

    complete = [(name, report) for name, report in reports.items() if report.full]
    if complete:
        name, report = max(complete, key=lambda item: view_quality(item[1]))
        return True, name, report

    detected = [(name, report) for name, report in reports.items() if report.seen]
    if not detected:
        return False, None, None

    def recovery_rank(item: tuple[str, MaskReport]) -> tuple[float, ...]:
        _, report = item
        _, backoff = decide_direction(report)
        return (
            1.0 if backoff else 0.0,
            float(len(report.edges)),
            -view_quality(report),
            -float(report.rectangularity),
            -float(report.area_frac),
        )

    name, report = min(detected, key=recovery_rank)
    return False, name, report


def rotation_matrix_from_quaternion(
    qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    """Return a 3x3 rotation matrix for an xyzw quaternion."""
    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("quaternion must be finite and nonzero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def optical_axes_in_base(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return image-right, image-down, and back-away axes in ``base_link``.

    The quaternion is the rotation from a permitted wrist-camera optical frame
    into ``base_link``. Optical +z looks into the scene, so back-away is -z.
    """
    rotation = rotation_matrix_from_quaternion(qx, qy, qz, qw)
    return rotation[:, 0], rotation[:, 1], -rotation[:, 2]
