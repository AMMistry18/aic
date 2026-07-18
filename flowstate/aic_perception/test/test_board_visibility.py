from __future__ import annotations

import cv2
import numpy as np
import pytest

from aic_perception.board_visibility import (
    MaskReport,
    adaptive_action,
    analyze_board,
    combine_cameras,
    decide_direction,
    detect_purple_logo,
    ivm_survey_ready,
    ivm_survey_rejection_reasons,
    optical_axes_in_base,
    rotation_matrix_from_quaternion,
    search_progress,
    view_quality,
    world_delta,
)


# The sim insignia measures H=150 exactly; BGR (180, 0, 180) reproduces it.
LOGO_BGR = (180, 0, 180)


def frame(rect=(70, 50, 250, 180), size=(320, 240), bg=210, board=45):
    width, height = size
    image = np.full((height, width, 3), bg, dtype=np.uint8)
    if rect is not None:
        x0, y0, x1, y1 = rect
        cv2.rectangle(image, (x0, y0), (x1, y1), (board,) * 3, -1)
    return image


def logo_frame(logo_rect=(210, 90, 240, 120), **kwargs):
    image = frame(**kwargs)
    x0, y0, x1, y1 = logo_rect
    cv2.rectangle(image, (x0, y0), (x1, y1), LOGO_BGR, -1)
    return image


def test_uniform_frame_is_not_a_board():
    report = analyze_board(frame(rect=None))
    assert not report.seen
    assert not report.full


def test_low_contrast_frame_is_rejected():
    report = analyze_board(frame(bg=150, board=135), min_contrast=30)
    assert not report.seen


def test_centered_board_is_full_and_rectangular():
    report = analyze_board(frame())
    assert report.seen and report.full
    assert report.edges == frozenset()
    assert 0.25 < report.area_frac < 0.40
    assert report.rectangularity > 0.95
    assert report.context_ok and report.detail_ok and report.shape_ok
    assert not report.artificial_bottom_contact
    assert view_quality(report) > 0.7


def survey_report(**changes):
    values = dict(
        seen=True,
        full=True,
        area_frac=0.20,
        rectangularity=0.85,
        clearance_px=(100.0, 100.0, 100.0, 100.0),
        context_pad_px=40.0,
        center_error=(0.0, 0.0),
        artificial_bottom_contact=False,
        orientation_deg=2.0,
        long_axis_ratio=1.25,
        logo_seen=True,
    )
    values.update(changes)
    return MaskReport(**values)


def test_ivm_survey_ready_requires_more_than_plate_visibility():
    assert ivm_survey_ready(survey_report())

    oblique_side_view = survey_report(
        rectangularity=0.63,
        center_error=(0.204, 0.219),
        artificial_bottom_contact=True,
        orientation_deg=65.4,
        long_axis_ratio=1.20,
        clearance_px=(298.0, 128.0, 306.0, 39.0),
        context_pad_px=36.0,
    )
    reasons = ivm_survey_rejection_reasons(oblique_side_view)
    assert not ivm_survey_ready(oblique_side_view)
    assert "low_rectangularity" in reasons
    assert "horizontal_off_center" in reasons
    assert "gripper_mask_contact" in reasons
    assert "long_axis_misaligned" in reasons
    assert "component_context_tight" in reasons


def test_ivm_survey_ready_rejects_clipped_or_ambiguous_center_view():
    report = survey_report(
        full=False,
        center_error=(-0.369, 0.195),
        artificial_bottom_contact=True,
        long_axis_ratio=1.09,
    )
    reasons = ivm_survey_rejection_reasons(report)
    assert "plate_not_full" in reasons
    assert "horizontal_off_center" in reasons
    assert "gripper_mask_contact" in reasons
    assert "long_axis_ambiguous" in reasons


@pytest.mark.parametrize(
    ("rect", "edge"),
    [
        ((0, 50, 180, 180), "left"),
        ((140, 50, 319, 180), "right"),
        ((70, 0, 250, 130), "top"),
        ((70, 80, 250, 239), "bottom"),
    ],
)
def test_reports_each_cut_edge(rect, edge):
    report = analyze_board(frame(rect=rect), ignore_bottom_frac=0.0)
    assert report.seen and not report.full
    assert edge in report.edges


def test_contact_with_ignored_finger_band_is_diagnostic_not_a_veto():
    report = analyze_board(
        frame(rect=(70, 80, 250, 230)), ignore_bottom_frac=0.15, margin_px=5
    )
    assert report.seen and report.full
    assert report.artificial_bottom_contact
    assert "artificial_bottom_contact" not in report.failure_reasons


def test_thin_gripper_bridge_into_ignored_band_does_not_veto_board():
    image = frame(rect=(70, 55, 250, 175))
    # Model a narrow arm/finger connection which widens into the gripper close
    # to the excluded band.  The raw largest component reaches the crop at
    # y=203, but its broad board body is wholly above it.
    cv2.rectangle(image, (156, 172), (164, 198), (45, 45, 45), -1)
    cv2.rectangle(image, (132, 195), (188, 239), (45, 45, 45), -1)

    report = analyze_board(
        image,
        ignore_bottom_frac=0.15,
        margin_px=15,
        context_pad_frac=0.05,
    )

    assert report.seen and report.full
    assert report.bbox is not None and report.bbox[3] == 203
    assert not report.artificial_bottom_contact
    assert "artificial_bottom_contact" not in report.failure_reasons
    assert abs(report.orientation_deg) < 3.0
    assert report.long_axis_ratio > 1.2


def test_board_above_finger_band_uses_physical_bottom_clearance():
    # The board clears the masked gripper band, but is less than margin_px from
    # that artificial boundary.  Since all physical corners remain in-frame,
    # the crop itself must not invent a bottom-clipped edge.
    report = analyze_board(
        frame(rect=(70, 55, 250, 190)),
        ignore_bottom_frac=0.15,
        margin_px=15,
        context_pad_frac=0.05,
    )
    assert report.seen and report.full
    assert "bottom" not in report.edges
    assert not report.artificial_bottom_contact


def test_dynamic_context_envelope_rejects_board_before_old_margin_contact():
    # The blob itself clears the old fixed 15px margin.  Its projected size asks
    # for more surrounding context so protruding components are not clipped.
    report = analyze_board(frame(rect=(17, 50, 198, 180)))
    assert report.seen and not report.full
    assert report.bbox is not None and report.bbox[0] > 15
    assert report.context_pad_px > 15
    assert "left" in report.edges
    assert not report.context_ok
    assert report.failure_reasons == ("context_clipped",)


def test_detected_but_too_small_board_lacks_usable_detail():
    report = analyze_board(
        frame(rect=(140, 100, 172, 126)),
        min_area_frac=0.001,
        min_detail_area_frac=0.02,
        ignore_bottom_frac=0.0,
    )
    assert report.seen and not report.full
    assert report.context_ok and report.shape_ok
    assert not report.detail_ok
    assert report.detail_score < 1.0
    assert "insufficient_detail" in report.failure_reasons


def test_moderate_perspective_passes_but_nonrectangular_blob_does_not():
    moderate = np.full((240, 320, 3), 210, dtype=np.uint8)
    cv2.fillConvexPoly(
        moderate,
        np.array([[70, 55], [245, 68], [232, 177], [82, 172]], np.int32),
        (45, 45, 45),
    )
    moderate_report = analyze_board(moderate, ignore_bottom_frac=0.0)
    assert moderate_report.seen and moderate_report.full
    assert moderate_report.shape_ok

    triangular = np.full((240, 320, 3), 210, dtype=np.uint8)
    cv2.fillConvexPoly(
        triangular,
        np.array([[160, 45], [250, 180], [70, 180]], np.int32),
        (45, 45, 45),
    )
    triangular_report = analyze_board(triangular, ignore_bottom_frac=0.0)
    assert triangular_report.seen and not triangular_report.full
    assert triangular_report.context_ok and triangular_report.detail_ok
    assert not triangular_report.shape_ok
    assert "nonrectangular_board" in triangular_report.failure_reasons


def test_thin_dark_cable_is_removed_before_largest_component_selection():
    image = frame()
    cv2.line(image, (0, 20), (319, 20), (20, 20, 20), 2)
    report = analyze_board(image, morph_px=5)
    assert report.seen and report.full
    assert report.bbox is not None and report.bbox[1] > 20


def test_tiny_blob_is_rejected():
    image = frame(rect=(100, 100, 110, 110))
    report = analyze_board(image, min_area_frac=0.01)
    assert not report.seen


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        analyze_board(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        analyze_board(np.zeros((10, 10), dtype=np.float32))


def test_logo_detected_and_filled_into_board_mask():
    plain = analyze_board(frame())
    with_logo = analyze_board(logo_frame())
    assert with_logo.seen and with_logo.full
    assert with_logo.logo_seen
    # The insignia sits right of image center in this frame.
    assert with_logo.logo_center_error[0] > 0.2
    assert with_logo.logo_bbox is not None
    # The fill keeps the bright logo inside the board component: without it
    # the logo would punch a hole and shrink the measured area.
    assert with_logo.area_frac == pytest.approx(plain.area_frac, abs=0.005)
    assert with_logo.rectangularity > 0.95


def test_logo_anchors_component_selection_over_larger_blob():
    # A larger dark blob (arm/shadow) competes with the board.  Largest-area
    # selection would pick the arm; the insignia anchor must pick the board.
    image = frame(rect=(200, 60, 300, 140))
    cv2.rectangle(image, (20, 40), (160, 200), (45, 45, 45), -1)
    cv2.rectangle(image, (255, 85), (285, 115), LOGO_BGR, -1)

    report = analyze_board(image, ignore_bottom_frac=0.0)
    assert report.seen and report.logo_seen
    assert report.bbox is not None
    assert report.bbox[0] >= 195, "anchor must select the board blob"


def test_largest_blob_fallback_without_logo():
    image = frame(rect=(200, 60, 300, 140))
    cv2.rectangle(image, (20, 40), (160, 200), (45, 45, 45), -1)
    report = analyze_board(image, ignore_bottom_frac=0.0)
    assert report.seen and not report.logo_seen
    assert report.bbox is not None
    assert report.bbox[0] <= 25, "without the logo the larger blob wins"


def test_logo_survives_contrast_guard_rejection():
    # Plate fills the frame: Otsu has no foreground/background split, so the
    # board mask is rejected, but the insignia still proves board presence.
    image = np.full((240, 320, 3), 45, dtype=np.uint8)
    cv2.rectangle(image, (40, 90), (70, 120), LOGO_BGR, -1)
    report = analyze_board(image)
    assert not report.seen
    assert report.logo_seen
    assert report.logo_center_error[0] < -0.5


def test_gray_input_never_reports_a_logo():
    gray = np.full((240, 320), 210, dtype=np.uint8)
    cv2.rectangle(gray, (70, 50), (250, 180), 45, -1)
    report = analyze_board(gray)
    assert report.seen
    assert not report.logo_seen
    assert detect_purple_logo(gray) is None


def test_blue_connectors_are_not_a_logo():
    # SC connectors measure around H=100; they must never satisfy the
    # tightened H in [135, 165] band.
    image = frame()
    cv2.rectangle(image, (100, 80), (140, 120), (200, 120, 20), -1)
    report = analyze_board(image)
    assert not report.logo_seen


def test_logo_blobs_are_unioned():
    # The insignia often fragments into an outline piece and a bar piece.
    image = frame()
    cv2.rectangle(image, (200, 80), (215, 130), LOGO_BGR, -1)
    cv2.rectangle(image, (228, 80), (243, 130), LOGO_BGR, -1)
    result = detect_purple_logo(image)
    assert result is not None
    _, centroid, area, bbox = result
    assert 215 < centroid[0] < 228, "centroid must be the union centroid"
    assert bbox[0] <= 200 and bbox[2] >= 243
    assert area >= 2 * 15 * 50 * 0.9


def test_orientation_measures_board_long_axis_angle():
    aligned = analyze_board(frame())
    assert abs(aligned.orientation_deg) < 3.0

    rotated = np.full((240, 320, 3), 210, dtype=np.uint8)
    box = cv2.boxPoints(((160.0, 110.0), (160.0, 80.0), 30.0)).astype(np.int32)
    cv2.fillConvexPoly(rotated, box, (45, 45, 45))
    result = analyze_board(rotated, ignore_bottom_frac=0.0)
    assert result.seen
    assert 25.0 <= abs(result.orientation_deg) <= 35.0


def test_orientation_maps_board_long_side_to_image_long_side():
    landscape = analyze_board(
        frame(rect=(40, 80, 280, 150), size=(320, 240)),
        ignore_bottom_frac=0.0,
    )
    portrait = analyze_board(
        frame(rect=(30, 100, 210, 160), size=(240, 320)),
        ignore_bottom_frac=0.0,
    )
    assert abs(landscape.orientation_deg) < 3.0
    assert abs(abs(portrait.orientation_deg) - 90.0) < 3.0


def test_frame_clipped_component_reports_ambiguous_orientation():
    # A board slice bounded by two image edges inherits its minAreaRect from
    # the frame, not the plate; the angle would read ~0 regardless of true
    # rotation, so the estimate must be declared ambiguous instead.
    clipped = analyze_board(
        frame(rect=(0, 0, 170, 130)), ignore_bottom_frac=0.0
    )
    assert clipped.seen
    assert clipped.long_axis_ratio == pytest.approx(1.0)
    assert clipped.orientation_deg == pytest.approx(0.0)

    unclipped = analyze_board(
        frame(rect=(40, 30, 210, 160)), ignore_bottom_frac=0.0
    )
    assert unclipped.seen
    assert unclipped.long_axis_ratio > 1.2


def test_direction_for_diagonal_cut_is_normalized():
    report = MaskReport(
        seen=True, full=False, edges=frozenset({"left", "top"})
    )
    direction, backoff = decide_direction(report)
    np.testing.assert_allclose(direction, [-np.sqrt(0.5), -np.sqrt(0.5)])
    assert not backoff


def test_adaptive_direction_changes_with_camera_evidence():
    left_report = analyze_board(
        frame(rect=(0, 50, 180, 180)), ignore_bottom_frac=0.0
    )
    right_report = analyze_board(
        frame(rect=(140, 50, 319, 180)), ignore_bottom_frac=0.0
    )
    left_action = adaptive_action(left_report)
    right_action = adaptive_action(right_report, history=(left_report,))
    assert left_action.mode == right_action.mode == "translate"
    assert left_action.direction_image[0] < 0.0
    assert right_action.direction_image[0] > 0.0


def test_persistent_same_edge_with_growing_area_switches_to_backoff():
    reports = [
        MaskReport(
            seen=True,
            full=False,
            edges=frozenset({"bottom"}),
            area_frac=area,
            rectangularity=0.8,
            clearance_px=(30.0, 30.0, 30.0, 0.0),
            context_pad_px=20.0,
            context_ok=False,
            detail_ok=True,
            shape_ok=True,
            quality_score=0.55,
        )
        for area in (0.10, 0.12, 0.14)
    ]
    action = adaptive_action(reports[-1], history=reports[:-1])
    assert action.mode == "backoff"
    assert action.backoff
    assert action.reason == "persistent_edge_without_clearance"


def test_more_than_six_improving_views_do_not_create_a_trial_limit():
    reports = [
        MaskReport(
            seen=True,
            full=False,
            edges=frozenset({"bottom"}),
            area_frac=0.12,
            rectangularity=0.8,
            clearance_px=(30.0, 30.0, 30.0, float(index)),
            context_pad_px=20.0,
            context_ok=False,
            detail_ok=True,
            shape_ok=True,
            quality_score=0.10 + 0.08 * index,
        )
        for index in range(8)
    ]
    assert search_progress(reports) == "improving"
    action = adaptive_action(reports[-1], history=reports[:-1])
    assert action.mode == "translate"
    assert not action.backoff


def test_opposite_edges_request_backoff():
    report = MaskReport(
        seen=True,
        full=False,
        edges=frozenset({"left", "right", "top"}),
    )
    direction, backoff = decide_direction(report)
    np.testing.assert_allclose(direction, [0.0, -1.0])
    assert backoff


def test_full_or_unseen_report_requests_no_motion():
    for report in (
        MaskReport(seen=False, full=False),
        MaskReport(seen=True, full=True),
    ):
        direction, backoff = decide_direction(report)
        np.testing.assert_array_equal(direction, [0.0, 0.0])
        assert not backoff


def test_world_delta_uses_camera_axes_and_backoff():
    delta = world_delta(
        np.array([-1.0, 0.0]),
        True,
        0.02,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        backoff_step_m=0.03,
    )
    np.testing.assert_allclose(delta, [0.0, 0.0, 0.03])


def test_world_delta_rejects_invalid_axes():
    with pytest.raises(ValueError):
        world_delta(
            np.array([1.0, 0.0]),
            False,
            0.02,
            np.zeros(3),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    with pytest.raises(ValueError):
        world_delta(
            np.array([1.0, 0.0]),
            False,
            0.02,
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )


def test_combine_cameras_prefers_complete_view_then_easiest_partial_view():
    partial = MaskReport(seen=True, full=False, area_frac=0.4)
    complete = MaskReport(seen=True, full=True, area_frac=0.2)
    done, camera, report = combine_cameras(
        {"left_camera": partial, "center_camera": complete}
    )
    assert done and camera == "center_camera" and report is complete

    easiest = MaskReport(
        seen=True,
        full=False,
        edges=frozenset({"bottom"}),
        area_frac=0.1,
        rectangularity=0.7,
    )
    close_and_clipped = MaskReport(
        seen=True,
        full=False,
        edges=frozenset({"top", "bottom", "right"}),
        area_frac=0.4,
        rectangularity=0.98,
    )
    done, camera, report = combine_cameras(
        {
            "left_camera": close_and_clipped,
            "center_camera": easiest,
        }
    )
    assert not done and camera == "center_camera" and report is easiest


def test_combine_cameras_uses_area_only_as_a_tie_breaker():
    smaller = MaskReport(
        seen=True,
        full=False,
        edges=frozenset({"bottom"}),
        area_frac=0.1,
        rectangularity=0.8,
    )
    larger = MaskReport(
        seen=True,
        full=False,
        edges=frozenset({"right"}),
        area_frac=0.2,
        rectangularity=0.8,
    )
    done, camera, report = combine_cameras(
        {"left_camera": smaller, "right_camera": larger}
    )
    assert not done and camera == "right_camera" and report is larger


def test_combine_cameras_reports_no_evidence():
    assert combine_cameras({}) == (False, None, None)
    assert combine_cameras(
        {"left_camera": MaskReport(seen=False, full=False)}
    ) == (False, None, None)


def test_optical_axes_follow_xyzw_quaternion():
    # +90 degrees around base Z: camera +X -> base +Y, camera +Y -> base -X.
    sine = np.sqrt(0.5)
    right, down, backoff = optical_axes_in_base(0.0, 0.0, sine, sine)
    np.testing.assert_allclose(right, [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(down, [-1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(backoff, [0.0, 0.0, -1.0], atol=1e-7)


def test_rotation_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        rotation_matrix_from_quaternion(0.0, 0.0, 0.0, 0.0)
