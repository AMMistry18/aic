"""Detect and frame the dark AIC task board using permitted camera data.

This module is deliberately independent of ROS so the segmentation and steering
logic can be unit-tested with synthetic or saved camera images.

The board carries a bright magenta insignia on one side of the plate.  It is
the only purple object in the workcell, so it doubles as a fiducial: its
presence proves the board is in view, and the dark component touching it is
the board by construction (never the arm or a shadow).  The insignia hue was
measured at H=150+/-2 across every saved sim frame while the nearest blue
pixels (SC connectors, NIC PCBs) top out at H=110, so the [135, 165] band
below keeps a 25-unit margin and stays symmetric about 150 in case of an
RGB/BGR channel swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


PURPLE_HUE_RANGE = (135, 165)
PURPLE_SAT_MIN = 60
PURPLE_VAL_MIN = 60
# The insignia outline often fragments into an outline piece and a bar piece
# under occlusion or frame clipping, so blobs are unioned rather than taking
# only the largest one.
PURPLE_MIN_BLOB_PX = 50
PURPLE_MIN_TOTAL_PX = 100


def detect_purple_logo(image: np.ndarray):
    """Detect the board's magenta insignia in a BGR/BGRA image.

    Returns ``None`` when the image carries no hue (grayscale) or no credible
    purple region exists.  Otherwise returns ``(mask, centroid, area_px,
    bbox)`` where ``mask`` is the cleaned union of all purple blobs,
    ``centroid`` is the union centroid ``(x, y)`` in pixels, and ``bbox`` is
    ``(x0, y0, x1, y1)`` of the union.
    """
    import cv2

    if image.ndim != 3 or image.shape[0] < 8 or image.shape[1] < 8:
        return None
    bgr = image
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [PURPLE_HUE_RANGE[0], PURPLE_SAT_MIN, PURPLE_VAL_MIN], dtype=np.uint8
    )
    upper = np.array([PURPLE_HUE_RANGE[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    union = np.zeros_like(mask)
    total = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= PURPLE_MIN_BLOB_PX:
            union[labels == index] = 255
            total += area
    if total < PURPLE_MIN_TOTAL_PX:
        return None
    ys, xs = np.nonzero(union)
    centroid = (float(xs.mean()), float(ys.mean()))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return union, centroid, total, bbox


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
    # Quantitative separation between a protected board envelope and the
    # calibrated camera-fixed gripper silhouette.  The envelope is the board
    # component's convex hull expanded by the normal component-context pad, so
    # zero overlap protects protruding NIC/SC hardware rather than merely the
    # dark base plate. ``gripper_escape_direction`` is the desired *board image*
    # displacement away from the mask; camera-plane motion uses its negative.
    gripper_overlap_px: int = 0
    gripper_clearance_px: float = float("inf")
    gripper_escape_direction: tuple[float, float] = (0.0, -1.0)
    center_error: tuple[float, float] = (0.0, 0.0)
    clearance_score: float = 0.0
    detail_score: float = 0.0
    shape_score: float = 0.0
    center_score: float = 0.0
    quality_score: float = 0.0
    failure_reasons: tuple[str, ...] = ()
    # Signed angle error (degrees, in [-90, 90)) between the board plate's
    # long axis and the image's longer pixel axis.  Zero means the long side
    # of the physical board uses the camera's widest field of view.
    orientation_deg: float = 0.0
    long_axis_ratio: float = 1.0
    # Magenta-insignia fiducial evidence.  ``logo_seen`` can be true even when
    # ``seen`` is false (for example the plate fills the frame and the Otsu
    # contrast guard rejects it): the logo alone proves the board is in view.
    logo_seen: bool = False
    logo_center_error: tuple[float, float] = (0.0, 0.0)
    logo_area_frac: float = 0.0
    logo_bbox: tuple[int, int, int, int] | None = None


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
    ignore_mask: np.ndarray | None = None,
) -> MaskReport:
    """Return whether the whole dark task board is visible in ``image``.

    The mask uses a global Otsu inverse threshold. A contrast guard prevents
    Otsu from inventing a foreground split in an essentially uniform frame.
    ``ignore_mask`` identifies calibrated pixels occupied by the camera's own
    gripper.  Those pixels cannot become board foreground.  Contact with the
    calibrated mask is reported as a diagnostic only: the live wrist-camera
    masks intentionally extend beyond the rendered silhouette, so using a
    one-pixel contact as a completion veto rejects clearly separated, complete
    board views.
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

    ignored = np.zeros((height, width), dtype=bool)
    if ignore_mask is not None:
        supplied_ignore = np.asarray(ignore_mask)
        if supplied_ignore.shape != (height, width):
            raise ValueError("ignore_mask must match the image height and width")
        ignored = supplied_ignore.astype(bool, copy=True)

    # The insignia is detected before any grayscale decision so its evidence
    # survives every early return below.  Its gray value sits between the
    # plate and the floor, so without the explicit fill Otsu would flicker it
    # in and out of the board mask frame to frame.
    logo = detect_purple_logo(img)
    logo_fields: dict = {}
    logo_mask = None
    if logo is not None:
        logo_mask, logo_centroid, logo_area_px, logo_bbox = logo
        half_w = max(1.0, 0.5 * float(width - 1))
        half_h = max(1.0, 0.5 * float(height - 1))
        logo_fields = dict(
            logo_seen=True,
            logo_center_error=(
                (logo_centroid[0] - half_w) / half_w,
                (logo_centroid[1] - half_h) / half_h,
            ),
            logo_area_frac=logo_area_px / float(height * width),
            logo_bbox=logo_bbox,
        )

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_gray = gray.copy()
    threshold_gray[ignored] = 255

    _, mask = cv2.threshold(
        threshold_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    foreground = gray[(mask > 0) & ~ignored]
    background = gray[(mask == 0) & ~ignored]
    if (
        foreground.size == 0
        or background.size == 0
        or float(background.mean()) - float(foreground.mean()) < min_contrast
    ):
        return MaskReport(seen=False, full=False, **logo_fields)

    # The insignia is physically part of the plate: force it into the board
    # foreground so it can never punch a hole or shift the centroid.
    if logo_mask is not None:
        logo_mask = logo_mask.copy()
        logo_mask[ignored] = 0
        mask = cv2.bitwise_or(mask, logo_mask)

    band_y = height
    if ignore_bottom_frac > 0.0:
        band_y = max(1, int(height * (1.0 - ignore_bottom_frac)))
        ignored[band_y:, :] = True
    mask[ignored] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(morph_px), int(morph_px))
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Closing can grow foreground back into the deliberately excluded region.
    # Keep its boundary exact so contact with it remains a meaningful signal.
    mask[ignored] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return MaskReport(seen=False, full=False, **logo_fields)

    # Anchor component selection to the insignia when it is visible: the
    # component overlapping the (dilated) purple region is the board by
    # construction.  Largest-area selection is only the fallback; it can
    # alternate between unrelated dark fragments (arm, shadows, occluded
    # board halves), which was the root cause of the old identity churn.
    component_index = 0
    if logo_mask is not None:
        anchor = cv2.dilate(logo_mask, kernel, iterations=3)
        anchored_labels = labels[anchor > 0]
        anchored_labels = anchored_labels[anchored_labels > 0]
        if anchored_labels.size:
            counts = np.bincount(anchored_labels)
            candidate = int(np.argmax(counts))
            if (
                int(stats[candidate, cv2.CC_STAT_AREA])
                >= min_area_frac * height * width
            ):
                component_index = candidate
    if component_index == 0:
        component_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[component_index, cv2.CC_STAT_AREA])
    if area < min_area_frac * height * width:
        return MaskReport(seen=False, full=False, **logo_fields)

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
    # an ignore-mask-contacting blob.  The ignored region is deliberately
    # opaque to the detector, so the only useful evidence at that boundary is
    # whether the *broad body* of the largest component reaches it. A horizontal
    # opening severs narrow vertical appendages while leaving an actually
    # bottom-clipped board intact.  Selecting the largest surviving core also
    # handles a thin bridge which widens into the gripper at the crop boundary.
    #
    # The opening width is deliberately modest (12% of a robust component-row
    # width, and at least two morphology kernels).  The resulting contact bit
    # is useful for diagnostics and the broad core remains the reliable source
    # for J6 orientation, but contact does not veto an otherwise complete view.
    artificial_bottom_contact = False
    # Orientation must be measured from the broad plate body, never from a
    # thin gripper/arm bridge that happens to share the dark component.  The
    # latter can make minAreaRect's longer dimension follow the board's short
    # side, producing an exactly-orthogonal J6 target.
    orientation_component: np.ndarray | None = component
    ignored_u8 = ignored.astype(np.uint8)
    ignored_neighborhood = cv2.dilate(
        ignored_u8,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    ).astype(bool)
    raw_ignore_contact = bool(np.any(component.astype(bool) & ignored_neighborhood))
    if raw_ignore_contact:
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
                # No broad plate core survived.  The raw component is then
                # dominated by a narrow bridge/occluder, so any 90-degree
                # edge choice would be guesswork; suppress J6 alignment.
                orientation_component = None
            else:
                broad_index = (
                    int(np.argmax(broad_stats[1:, cv2.CC_STAT_AREA])) + 1
                )
                broad_core = broad_labels == broad_index
                orientation_component = broad_core.astype(np.uint8)
                artificial_bottom_contact = bool(
                    np.any(broad_core & ignored_neighborhood)
                )

    rectangularity = 0.0
    gripper_overlap_px = 0
    gripper_clearance_px = float("inf")
    gripper_escape_direction = (0.0, -1.0)
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        rect_width, rect_height = cv2.minAreaRect(contour)[1]
        rect_area = float(rect_width * rect_height)
        if rect_area > 0.0:
            rectangularity = min(1.0, area / rect_area)

        # Treat the entire projected board footprint plus its component
        # context as protected from the camera-fixed gripper.  This turns the
        # calibrated ignore mask into a useful geometric terminal constraint:
        # ignored pixels still cannot contaminate segmentation, and the policy
        # can now move until the task hardware is genuinely outside that mask.
        protected = np.zeros_like(component, dtype=np.uint8)
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(protected, hull, 1)
        envelope_pad_px = max(1, int(round(context_pad_px)))
        envelope_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * envelope_pad_px + 1, 2 * envelope_pad_px + 1),
        )
        protected = cv2.dilate(protected, envelope_kernel, iterations=1)
        protected_bool = protected.astype(bool)
        overlap = protected_bool & ignored
        gripper_overlap_px = int(np.count_nonzero(overlap))

        ignored_count = int(np.count_nonzero(ignored))
        if ignored_count > 0:
            free_space = (~ignored).astype(np.uint8)
            distance_to_gripper = cv2.distanceTransform(
                free_space, cv2.DIST_L2, 5
            )
            protected_distances = distance_to_gripper[protected_bool]
            if protected_distances.size > 0:
                gripper_clearance_px = (
                    0.0
                    if gripper_overlap_px > 0
                    else float(np.min(protected_distances))
                )

            protected_y, protected_x = np.nonzero(protected_bool)
            ignored_y, ignored_x = np.nonzero(ignored)
            if protected_x.size > 0 and ignored_x.size > 0:
                escape = np.array(
                    (
                        float(protected_x.mean() - ignored_x.mean()),
                        float(protected_y.mean() - ignored_y.mean()),
                    ),
                    dtype=float,
                )
                escape_norm = float(np.linalg.norm(escape))
                if escape_norm > 1e-9:
                    escape /= escape_norm
                    gripper_escape_direction = (
                        float(escape[0]),
                        float(escape[1]),
                    )

    orientation_deg = 0.0
    long_axis_ratio = 1.0
    orientation_contours = []
    if orientation_component is not None:
        orientation_contours, _ = cv2.findContours(
            orientation_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    if orientation_contours:
        contour = max(orientation_contours, key=cv2.contourArea)
        min_rect = cv2.minAreaRect(contour)
        rect_width, rect_height = min_rect[1]
        rect_area = float(rect_width * rect_height)
        if rect_area > 0.0:
            short_side = min(float(rect_width), float(rect_height))
            long_side = max(float(rect_width), float(rect_height))
            if short_side > 1e-6:
                long_axis_ratio = long_side / short_side

            # A component hard-clipped by two or more physical image edges
            # takes its minAreaRect geometry from the frame boundaries, not
            # the plate: the measured angle then reads confidently
            # axis-aligned regardless of the true board rotation (observed
            # on clipped center-camera close-ups).  Declare it ambiguous so
            # J6 waits until standoff clears the clipping.
            hard_edge_clips = sum(
                1 for value in clearances if value <= 2.0
            )
            if hard_edge_clips >= 2:
                long_axis_ratio = 1.0

            # cv2 reports the angle of the ``rect_width`` side.  Select the
            # geometrically longer rectangle side explicitly, then compare it
            # with the image's longer dimension.  This prevents a 90-degree
            # short-side target and also supports portrait camera streams.
            # A nearly square observation has no stable long/short identity;
            # a few pixels of segmentation noise can swap it by 90 degrees.
            # Refuse roll until perspective exposes a credible long side.
            if long_axis_ratio >= 1.10:
                board_long_axis_deg = float(min_rect[2])
                if rect_width < rect_height:
                    board_long_axis_deg += 90.0
                image_long_axis_deg = 0.0 if width >= height else 90.0
                orientation_deg = (
                    (board_long_axis_deg - image_long_axis_deg + 90.0) % 180.0
                ) - 90.0

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
    # Mask contact is diagnostic, not a visibility failure.  The calibrated
    # wrist-camera masks are conservative and, in the live right-camera trace,
    # their upper boundary reached the board's lower modules despite a visible
    # background gap to the rendered gripper.  The mask already prevents those
    # gripper pixels from entering the selected component; vetoing completion
    # on any boundary contact defeated the purpose of masking and cost four
    # unnecessary clearance moves.
    if not detail_ok:
        failure_reasons.append("insufficient_detail")
    if not shape_ok:
        failure_reasons.append("nonrectangular_board")

    full = bool(
        context_ok
        and detail_ok
        and shape_ok
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
        gripper_overlap_px=gripper_overlap_px,
        gripper_clearance_px=gripper_clearance_px,
        gripper_escape_direction=gripper_escape_direction,
        center_error=(float(center_error[0]), float(center_error[1])),
        clearance_score=clearance_score,
        detail_score=detail_score,
        shape_score=shape_score,
        center_score=center_score,
        quality_score=quality_score,
        failure_reasons=tuple(failure_reasons),
        orientation_deg=orientation_deg,
        long_axis_ratio=long_axis_ratio,
        **logo_fields,
    )


def view_quality(report: MaskReport) -> float:
    """Return a bounded continuous view score for ranking and progress checks."""

    if not report.seen:
        return 0.0
    return float(np.clip(report.quality_score, 0.0, 1.0))


def ivm_survey_rejection_reasons(
    report: MaskReport,
    *,
    max_area_frac: float = 0.36,
    min_rectangularity: float = 0.70,
    max_center_error_x: float = 0.15,
    max_center_error_y: float = 0.25,
    min_long_axis_ratio: float = 1.15,
    max_long_axis_error_deg: float = 2.0,
    clearance_scale: float = 1.25,
    min_gripper_clearance_px: float = 20.0,
) -> tuple[str, ...]:
    """Explain why a frame is not yet suitable for downstream IVM.

    ``MaskReport.full`` only answers whether the segmented plate fits in the
    image. Pose estimation needs a stronger terminal condition: the centered
    camera must have an identity anchor, enough context around protruding
    components, no overlap with the calibrated gripper silhouette, and the
    board's long side aligned with the camera's wide image dimension.

    Keeping this predicate separate from ``full`` is intentional. Partial and
    oblique side-camera reports remain useful steering evidence, but cannot
    prematurely release the downstream NIC/SFP/SC estimators.
    """

    reasons: list[str] = []
    if not report.full:
        reasons.append("plate_not_full")
    if not report.logo_seen:
        reasons.append("logo_not_seen")
    if report.area_frac > max_area_frac:
        reasons.append("board_too_large")
    if report.rectangularity < min_rectangularity:
        reasons.append("low_rectangularity")
    if abs(float(report.center_error[0])) > max_center_error_x:
        reasons.append("horizontal_off_center")
    if abs(float(report.center_error[1])) > max_center_error_y:
        reasons.append("vertical_off_center")
    if (
        report.artificial_bottom_contact
        or int(report.gripper_overlap_px) > 0
        or float(report.gripper_clearance_px) < min_gripper_clearance_px
    ):
        reasons.append("gripper_mask_contact")
    if report.long_axis_ratio < min_long_axis_ratio:
        reasons.append("long_axis_ambiguous")
    elif abs(float(report.orientation_deg)) > max_long_axis_error_deg:
        reasons.append("long_axis_misaligned")

    required_clearance = clearance_scale * float(report.context_pad_px)
    if min(report.clearance_px) < required_clearance:
        reasons.append("component_context_tight")
    return tuple(reasons)


def ivm_survey_ready(report: MaskReport, **kwargs) -> bool:
    """Return whether one center-camera frame is ready for all IVM models."""

    return not ivm_survey_rejection_reasons(report, **kwargs)


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
