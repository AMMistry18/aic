import base64

import cv2
import numpy as np

from aic_model.insertion.visual_gap import (
    detect_dark_port_opening,
    encode_port_opening_diagnostic_jpeg,
)
from aic_model.insertion.visual_gap_recovery import (
    VISUAL_GAP_DEBUG,
    VISUAL_GAP_WEDGE_ENABLE,
    VisualGapRecoveryMixin,
)


def test_detect_dark_opening_near_projected_quad():
    image = np.full((140, 180, 3), 190, dtype=np.uint8)
    cv2.rectangle(image, (63, 49), (119, 91), (120, 120, 120), -1)
    cv2.rectangle(image, (75, 59), (108, 82), (12, 12, 12), -1)
    quad = np.array([[68, 54], [113, 54], [113, 87], [68, 87]], dtype=float)

    result = detect_dark_port_opening(image, quad)

    assert result is not None
    np.testing.assert_allclose(result.center_uv, [91.5, 70.5], atol=1.0)


def test_gripper_mask_excludes_dark_distractor():
    image = np.full((140, 180, 3), 195, dtype=np.uint8)
    cv2.rectangle(image, (75, 58), (108, 82), (15, 15, 15), -1)
    cv2.rectangle(image, (45, 80), (90, 125), (0, 0, 0), -1)
    quad = np.array([[65, 50], [115, 50], [115, 90], [65, 90]], dtype=float)
    ignored = np.zeros(image.shape[:2], dtype=bool)
    ignored[80:, 45:91] = True

    result = detect_dark_port_opening(image, quad, ignored)

    assert result is not None
    np.testing.assert_allclose(result.center_uv, [91.5, 70.0], atol=1.5)


def test_low_contrast_crop_fails_closed():
    image = np.full((100, 120, 3), 145, dtype=np.uint8)
    cv2.rectangle(image, (48, 42), (72, 58), (137, 137, 137), -1)
    quad = np.array([[42, 36], [78, 36], [78, 64], [42, 64]], dtype=float)

    assert detect_dark_port_opening(image, quad) is None


def test_low_contrast_diagnostics_explain_rejection():
    image = np.full((100, 120, 3), 145, dtype=np.uint8)
    cv2.rectangle(image, (48, 42), (72, 58), (137, 137, 137), -1)
    quad = np.array([[42, 36], [78, 36], [78, 64], [42, 64]], dtype=float)
    diagnostics = {}

    result = detect_dark_port_opening(
        image, quad, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "low_contrast"
    assert diagnostics["valid_pixels"] > 0
    assert diagnostics["contrast"] < 24.0


def test_invalid_ignore_mask_shape_is_rejected():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    quad = np.array([[42, 36], [78, 36], [78, 64], [42, 64]], dtype=float)

    try:
        detect_dark_port_opening(image, quad, np.zeros((10, 10), dtype=bool))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("shape mismatch should raise ValueError")


def test_wedge_recovery_gate_defaults_off():
    assert VISUAL_GAP_WEDGE_ENABLE is False
    assert VISUAL_GAP_DEBUG is False
    assert VisualGapRecoveryMixin._visual_gap_wedge_enabled() is False


def test_diagnostic_jpeg_contains_four_panel_crop():
    image = np.full((140, 180, 3), 190, dtype=np.uint8)
    cv2.rectangle(image, (63, 49), (119, 91), (120, 120, 120), -1)
    cv2.rectangle(image, (75, 59), (108, 82), (12, 12, 12), -1)
    quad = np.array([[68, 54], [113, 54], [113, 87], [68, 87]], dtype=float)
    diagnostics = {}
    result = detect_dark_port_opening(
        image, quad, diagnostics=diagnostics)

    encoded = encode_port_opening_diagnostic_jpeg(
        image, quad, result, diagnostics)
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert result is not None
    assert diagnostics["reason"] == "accepted"
    assert decoded is not None
    assert decoded.shape[0] == 222
    assert decoded.shape[1] > decoded.shape[0]


def test_wedge_recovery_projection_and_plane_intersection_round_trip():
    class FakePerceptionCore:
        @staticmethod
        def invert_transform(transform):
            return np.linalg.inv(transform)

    recovery = VisualGapRecoveryMixin()
    recovery._pc = FakePerceptionCore()
    K = np.array([
        [200.0, 0.0, 100.0],
        [0.0, 200.0, 80.0],
        [0.0, 0.0, 1.0],
    ])
    T_cam_from_base = np.eye(4)
    point = np.array([0.2, -0.1, 2.0])
    P = K @ T_cam_from_base[:3, :4]

    uv = recovery._visual_gap_project_point(P, point)
    recovered = recovery._visual_gap_ray_to_port_plane(
        uv,
        K,
        T_cam_from_base,
        port_pos=np.array([0.0, 0.0, 2.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(uv, [120.0, 70.0], atol=1e-9)
    np.testing.assert_allclose(recovered, point, atol=1e-9)


def test_wedge_recovery_consensus_rejects_outlier_camera():
    hits = [
        {"plane_point": np.array([0.0010, -0.0020, 2.0])},
        {"plane_point": np.array([0.0012, -0.0019, 2.0])},
        {"plane_point": np.array([0.0200, 0.0200, 2.0])},
    ]

    consensus = VisualGapRecoveryMixin._visual_gap_consensus(
        hits, port_pos=np.array([0.0, 0.0, 2.0]), Rp=np.eye(3))

    np.testing.assert_allclose(consensus, [0.0011, -0.00195, 2.0], atol=1e-9)
