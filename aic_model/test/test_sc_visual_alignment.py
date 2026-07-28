from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))

from aic_model.sc_visual_alignment import (
    ScRecoveryEvidence,
    aggregate_sc_blue_side_signatures,
    bounded_recovery_offset_update,
    bounded_visual_port_update,
    detect_sc_duplex_opening,
    detect_sc_recovery_direction,
    fuse_sc_opening_hits,
    fuse_sc_recovery_evidence,
    measure_sc_blue_side_signature,
    project_point_px,
    ray_to_plane,
)


def _synthetic_duplex(*, show_left=True, show_right=True):
    image = np.full((140, 200, 3), 150, dtype=np.uint8)
    cv2.rectangle(image, (32, 35), (168, 105), (255, 80, 0), -1)
    left = np.array([[50, 52], [92, 52], [92, 88], [50, 88]], dtype=float)
    right = np.array([[108, 52], [150, 52], [150, 88], [108, 88]], dtype=float)
    if show_left:
        cv2.rectangle(image, (54, 56), (88, 84), (8, 8, 8), -1)
    if show_right:
        cv2.rectangle(image, (112, 56), (146, 84), (8, 8, 8), -1)
    return image, np.stack([left, right])


def _synthetic_recovery_view(*, shift_xy=(0, 0), blue=True, plug=True):
    """Fronto-parallel SC mouth; a plug only occludes blue when shifted."""

    image = np.full((192, 320, 3), 8, dtype=np.uint8)
    if blue:
        image[:] = (255, 80, 0)
    # The dark mouth leaves a mostly-blue reference in each 10/12 px side
    # band.  The plug reaches a band only when it is off centre, matching the
    # real contact cue rather than relying on a visible air gap.
    cv2.rectangle(image, (58, 58), (261, 133), (8, 8, 8), -1)
    if plug:
        shift_x, shift_y = shift_xy
        cv2.rectangle(
            image,
            (68 + shift_x, 61 + shift_y),
            (251 + shift_x, 130 + shift_y),
            (178, 178, 178),
            -1,
        )
    # Ordering is the controller's physical port ordering.  These selected
    # outer points make the canonical rectification an identity transform.
    right_bore = np.array(
        [[272, 48], [160, 48], [160, 144], [272, 144]], dtype=float
    )
    left_bore = np.array(
        [[160, 48], [48, 48], [48, 144], [160, 144]], dtype=float
    )
    return image, np.stack([right_bore, left_bore])


def _recovery_baseline(quads, *, warped_image=None):
    """Capture the detector's per-band clean reference for one test view."""

    if warped_image is None:
        image, _ = _synthetic_recovery_view(plug=False)
    else:
        image = warped_image
    baseline = {}
    for band in (10, 12):
        signature = measure_sc_blue_side_signature(
            image, quads, band_half_width_px=band
        )
        assert signature is not None
        baseline[band] = signature.blue_fractions
    return baseline


def _recovery_rich_baseline(quads, *, image=None, ignored_pixels=None):
    """Capture repeated immutable support/color records for one test view."""

    if image is None:
        image, _ = _synthetic_recovery_view(plug=False)
    baseline = {}
    for band in (10, 12):
        samples = []
        for _ in range(2):
            signature = measure_sc_blue_side_signature(
                image,
                quads,
                ignored_pixels,
                band_half_width_px=band,
            )
            assert signature is not None
            samples.append(signature)
        aggregate = aggregate_sc_blue_side_signatures(samples)
        assert aggregate is not None
        baseline[band] = aggregate
    return baseline


def _warp_recovery_view(image, quads):
    """Apply a strong view rotation/perspective while preserving quad labels."""

    source = np.array(
        [[0, 0], [319, 0], [319, 191], [0, 191]], dtype=np.float32
    )
    target = np.array(
        [[82, 34], [356, 71], [323, 272], [45, 238]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(image, transform, (400, 300))
    warped_quads = cv2.perspectiveTransform(
        quads.reshape(-1, 1, 2).astype(np.float32), transform
    ).reshape(2, 4, 2)
    return warped, warped_quads


def test_detects_duplex_midpoint_in_blue_adapter():
    image, quads = _synthetic_duplex()

    result = detect_sc_duplex_opening(image, quads)

    assert result is not None
    assert result.detected_bores == 2
    np.testing.assert_allclose(result.center_uv, [100.0, 70.0], atol=1.0)
    assert result.blue_fraction > 0.1


def test_single_visible_bore_uses_known_duplex_offset():
    image, quads = _synthetic_duplex(show_right=False)

    result = detect_sc_duplex_opening(image, quads)

    assert result is not None
    assert result.detected_bores == 1
    np.testing.assert_allclose(result.center_uv, [100.0, 70.0], atol=1.0)


def test_dark_shapes_without_blue_adapter_fail_association():
    image, quads = _synthetic_duplex()
    image[(image[:, :, 0] > 200)] = (150, 150, 150)

    diagnostics = {}
    result = detect_sc_duplex_opening(image, quads, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "blue_association_failed"


def test_projection_and_ray_plane_intersection_round_trip():
    K = np.array(
        [[200.0, 0.0, 100.0], [0.0, 200.0, 80.0], [0.0, 0.0, 1.0]]
    )
    T_cam_from_world = np.eye(4)
    point = np.array([0.2, -0.1, 2.0])
    uv = project_point_px(K @ T_cam_from_world[:3, :4], point)

    recovered = ray_to_plane(
        uv,
        K,
        T_cam_from_world,
        plane_point=np.array([0.0, 0.0, 2.0]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(uv, [120.0, 70.0], atol=1e-9)
    np.testing.assert_allclose(recovered, point, atol=1e-9)


def test_multiview_fusion_keeps_agreeing_pair_and_rejects_outlier():
    hits = [
        {"camera": "left", "plane_point": np.array([-0.0012, 0.0006, 2.0])},
        {"camera": "center", "plane_point": np.array([-0.0011, 0.00065, 2.0])},
        {"camera": "right", "plane_point": np.array([0.003, 0.003, 2.0])},
    ]

    result = fuse_sc_opening_hits(
        hits,
        origin_port_pos=np.array([0.0, 0.0, 2.0]),
        Rp=np.eye(3),
        max_view_disagreement_m=0.001,
        max_total_offset_m=0.003,
    )

    assert result is not None
    assert result.cameras == ("left", "center")
    assert result.single_view is False
    np.testing.assert_allclose(
        result.point_world, [-0.00115, 0.000625, 2.0], atol=1e-12
    )


def test_single_view_remains_directed_but_marked_lower_confidence():
    result = fuse_sc_opening_hits(
        [{"camera": "left", "plane_point": np.array([-0.0012, 0.0006, 2.0])}],
        origin_port_pos=np.array([0.0, 0.0, 2.0]),
        Rp=np.eye(3),
        max_view_disagreement_m=0.001,
        max_total_offset_m=0.003,
    )

    assert result is not None
    assert result.single_view is True
    assert result.cameras == ("left",)


def test_visual_update_clips_step_and_total_excursion():
    update = bounded_visual_port_update(
        current_port_pos=np.zeros(3),
        origin_port_pos=np.zeros(3),
        observed_port_pos=np.array([-0.002, 0.001, 0.0]),
        Rp=np.eye(3),
        max_step_m=0.0005,
        max_total_m=0.003,
    )

    assert update is not None
    target, step = update
    np.testing.assert_allclose(np.linalg.norm(step), 0.0005, rtol=1e-9)
    np.testing.assert_allclose(target[:2] / np.linalg.norm(target[:2]),
                               np.array([-2.0, 1.0]) / np.sqrt(5.0))

    capped = bounded_visual_port_update(
        current_port_pos=np.array([0.0028, 0.0, 0.0]),
        origin_port_pos=np.zeros(3),
        observed_port_pos=np.array([0.004, 0.0, 0.0]),
        Rp=np.eye(3),
        max_step_m=0.0005,
        max_total_m=0.003,
    )
    assert capped is not None
    np.testing.assert_allclose(capped[0], [0.003, 0.0, 0.0], atol=1e-12)


def test_recovery_direction_moves_toward_larger_clearance_on_each_axis():
    cases = [
        ((-15, 0), np.array([1.0, 0.0])),
        ((15, 0), np.array([-1.0, 0.0])),
        ((0, -7), np.array([0.0, -1.0])),
        ((0, 7), np.array([0.0, 1.0])),
    ]
    for shift, expected in cases:
        image, quads = _synthetic_recovery_view(shift_xy=shift)
        diagnostics = {}

        result = detect_sc_recovery_direction(
            image,
            quads,
            baseline_blue_fractions=_recovery_baseline(quads),
            diagnostics=diagnostics,
        )

        assert result is not None, diagnostics
        assert result.balanced is False
        assert float(np.dot(result.direction_xy, expected)) > 0.95


def test_recovery_direction_is_port_local_under_perspective():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 7))
    warped, warped_quads = _warp_recovery_view(image, quads)
    clean, _ = _synthetic_recovery_view(plug=False)
    warped_clean, _ = _warp_recovery_view(clean, quads)

    result = detect_sc_recovery_direction(
        warped,
        warped_quads,
        baseline_blue_fractions=_recovery_baseline(
            warped_quads, warped_image=warped_clean
        ),
    )

    assert result is not None
    assert result.balanced is False
    assert result.direction_xy[1] > 0.95


def test_recovery_diagonal_uses_metric_port_axes():
    # Four X and five Y band pixels are approximately equal in physical units
    # after canonical-pixel-to-metre scaling.
    image, quads = _synthetic_recovery_view(shift_xy=(15, 7))

    result = detect_sc_recovery_direction(
        image,
        quads,
        baseline_blue_fractions=_recovery_baseline(quads),
    )

    assert result is not None
    expected = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    assert float(np.dot(result.direction_xy, expected)) > 0.98


def test_recovery_treats_one_pixel_noise_as_balanced():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 1))

    result = detect_sc_recovery_direction(
        image,
        quads,
        baseline_blue_fractions=_recovery_baseline(quads),
    )

    assert result is not None
    assert result.balanced is True
    np.testing.assert_array_equal(result.direction_xy, [0.0, 0.0])


def test_recovery_requires_a_strong_clean_reference_and_agreeing_band_widths():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 7))
    diagnostics = {}

    assert (
        detect_sc_recovery_direction(image, quads, diagnostics=diagnostics)
        is None
    )
    assert diagnostics["reason"] == "baseline_unavailable"

    weak_reference = {10: np.full(4, 0.49), 12: np.full(4, 0.8)}
    assert (
        detect_sc_recovery_direction(
            image,
            quads,
            baseline_blue_fractions=weak_reference,
            diagnostics=diagnostics,
        )
        is None
    )
    assert diagnostics["reason"] == "weak_baseline_side_visibility"

    conflicting_reference = _recovery_baseline(quads)
    # For this low plug, reducing the -Y reference only for the 10px band
    # reverses that band's normalized evidence.  Two incompatible scales must
    # abstain instead of averaging a potentially wrong move.
    conflicting_reference[10] = conflicting_reference[10].copy()
    conflicting_reference[10][2] = 0.50
    assert (
        detect_sc_recovery_direction(
            image,
            quads,
            baseline_blue_fractions=conflicting_reference,
            diagnostics=diagnostics,
        )
        is None
    )
    assert diagnostics["reason"] == "direction_unstable"


def test_recovery_rejects_missing_blue_and_masked_roi():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 7), blue=False)
    diagnostics = {}
    assert (
        detect_sc_recovery_direction(
            image,
            quads,
            baseline_blue_fractions=_recovery_baseline(quads),
            diagnostics=diagnostics,
        )
        is None
    )
    assert diagnostics["reason"] == "blue_association_failed"

    image, quads = _synthetic_recovery_view(shift_xy=(0, 7))
    ignored = np.ones(image.shape[:2], dtype=bool)
    diagnostics = {}
    assert (
        detect_sc_recovery_direction(
            image,
            quads,
            ignored,
            baseline_blue_fractions=_recovery_baseline(quads),
            diagnostics=diagnostics,
        )
        is None
    )
    assert diagnostics["reason"] == "insufficient_valid_roi"


def test_recovery_rejects_an_undersampled_projected_port():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 7))
    center = quads.mean(axis=(0, 1))
    tiny_quads = center + 0.10 * (quads - center)
    diagnostics = {}

    result = detect_sc_recovery_direction(
        image,
        tiny_quads,
        baseline_blue_fractions=_recovery_baseline(quads),
        diagnostics=diagnostics,
    )

    assert result is None
    assert diagnostics["reason"] == "rectification_failed"


@pytest.mark.parametrize(
    "masked_slice",
    [
        (slice(48, 144), slice(48, 105)),
        (slice(48, 144), slice(215, 272)),
        (slice(48, 80), slice(48, 272)),
        (slice(112, 144), slice(48, 272)),
        (slice(48, 144), slice(68, 70)),
        (slice(48, 144), slice(250, 252)),
        (slice(57, 59), slice(48, 272)),
        (slice(133, 135), slice(48, 272)),
    ],
)
def test_recovery_rejects_partial_measurement_occlusion(masked_slice):
    image, quads = _synthetic_recovery_view()
    ignored = np.zeros(image.shape[:2], dtype=bool)
    ignored[masked_slice] = True
    diagnostics = {}

    result = detect_sc_recovery_direction(
        image,
        quads,
        ignored,
        baseline_blue_fractions=_recovery_baseline(quads),
        diagnostics=diagnostics,
    )

    assert result is None
    assert diagnostics["reason"] == "occluded_measurement_roi"


def test_recovery_accepts_a_stable_partial_mask_with_paired_support():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 7))
    clean, _ = _synthetic_recovery_view(plug=False)
    # This strips part of the left-side corridor, like a fixed calibrated
    # gripper silhouette.  It is present both before approach and at stall.
    ignored = np.zeros(image.shape[:2], dtype=bool)
    ignored[64:118, 48:54] = True
    baseline = _recovery_rich_baseline(
        quads, image=clean, ignored_pixels=ignored
    )
    diagnostics = {}

    result = detect_sc_recovery_direction(
        image,
        quads,
        ignored,
        baseline_blue_fractions=baseline,
        diagnostics=diagnostics,
    )

    assert result is not None, diagnostics
    assert result.direction_xy[1] > 0.95
    paired = diagnostics["support_by_band"][10]
    assert paired["support_mode"] == "paired"
    assert float(np.min(paired["common_support_fractions"])) < 1.0
    assert float(np.max(paired["support_change_fractions"])) == 0.0
    assert not baseline[10].side_support_masks[0].flags.writeable
    assert not baseline[10].side_blue_masks[0].flags.writeable


def test_recovery_rejects_a_new_mask_with_paired_support():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 7))
    clean, _ = _synthetic_recovery_view(plug=False)
    stable_ignored = np.zeros(image.shape[:2], dtype=bool)
    stable_ignored[64:118, 48:54] = True
    baseline = _recovery_rich_baseline(
        quads, image=clean, ignored_pixels=stable_ignored
    )
    current_ignored = stable_ignored.copy()
    # A newly masked top strip changes the canonical mouth support; it must not
    # be interpreted as the plug making one blue side less visible.
    current_ignored[57:59, 92:228] = True
    diagnostics = {}

    result = detect_sc_recovery_direction(
        image,
        quads,
        current_ignored,
        baseline_blue_fractions=baseline,
        diagnostics=diagnostics,
    )

    assert result is None
    assert diagnostics["reason"] == "occluded_measurement_roi"
    assert diagnostics["support_mode"] == "paired"
    assert diagnostics["support_change"] == "new_occlusion"
    assert diagnostics["new_corridor_occlusion_fraction"] > 0.0


def test_recovery_allows_an_ignored_region_outside_measurement_corridor():
    image, quads = _synthetic_recovery_view(shift_xy=(0, 7))
    ignored = np.zeros(image.shape[:2], dtype=bool)
    ignored[:24, :24] = True

    result = detect_sc_recovery_direction(
        image,
        quads,
        ignored,
        baseline_blue_fractions=_recovery_baseline(quads),
    )

    assert result is not None
    assert result.direction_xy[1] > 0.95


def _recovery_evidence(direction, confidence=0.9, *, balanced=False):
    return ScRecoveryEvidence(
        direction_xy=np.asarray(direction, dtype=float),
        confidence=confidence,
        margins=np.array([0.02, 0.04, 0.01, 0.06]),
        blue_fraction=0.5,
        valid_fraction=1.0,
        balanced=balanced,
    )


def test_recovery_fusion_requires_agreement_from_two_cameras():
    result = fuse_sc_recovery_evidence(
        [
            {"camera": "left", "evidence": _recovery_evidence([0.0, 1.0])},
            {"camera": "right", "evidence": _recovery_evidence([0.1, 0.99])},
        ]
    )

    assert result is not None
    assert result.cameras == ("left", "right")
    assert result.balanced is False
    assert result.direction_xy[1] > 0.99

    contradictory = fuse_sc_recovery_evidence(
        [
            {"camera": "left", "evidence": _recovery_evidence([0.0, 1.0])},
            {"camera": "right", "evidence": _recovery_evidence([0.0, -1.0])},
        ]
    )
    assert contradictory is None


def test_recovery_fusion_accepts_only_unanimous_balanced_stop():
    balanced = _recovery_evidence([0.0, 0.0], balanced=True)
    result = fuse_sc_recovery_evidence(
        [
            {"camera": "left", "evidence": balanced},
            {"camera": "right", "evidence": balanced},
        ]
    )
    assert result is not None
    assert result.balanced is True
    np.testing.assert_array_equal(result.direction_xy, [0.0, 0.0])

    mixed = fuse_sc_recovery_evidence(
        [
            {"camera": "left", "evidence": balanced},
            {"camera": "right", "evidence": _recovery_evidence([1.0, 0.0])},
        ]
    )
    assert mixed is None


def test_recovery_step_has_independent_path_and_radius_caps():
    first = bounded_recovery_offset_update(
        np.zeros(2),
        np.array([1.0, 0.0]),
        0.0,
        max_step_m=0.00025,
        max_total_m=0.002,
    )
    assert first is not None
    np.testing.assert_allclose(first[0], [0.00025, 0.0])
    assert first[2] == 0.00025

    # A tangent-ish move at the radial boundary is shortened to stay inside it.
    radial = bounded_recovery_offset_update(
        np.array([0.00199, 0.0]),
        np.array([0.0, 1.0]),
        0.001,
        max_step_m=0.00025,
        max_total_m=0.002,
    )
    assert radial is not None
    assert np.linalg.norm(radial[0]) <= 0.002 + 1e-12
    assert np.linalg.norm(radial[1]) < 0.00025

    assert (
        bounded_recovery_offset_update(
            np.zeros(2),
            np.array([1.0, 0.0]),
            0.002,
            max_step_m=0.00025,
            max_total_m=0.002,
        )
        is None
    )
