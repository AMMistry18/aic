from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))

from aic_model.sc_visual_alignment import (
    bounded_visual_port_update,
    detect_sc_duplex_opening,
    fuse_sc_opening_hits,
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
