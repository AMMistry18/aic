"""Pure-geometry unit tests for the in-place geometric Stage 2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aic_perception.board_stage2 import (
    BOARD_OUTLINE_CORNERS,
    BoardPoseEstimate,
    CameraModel,
    GripperExclusion,
    INSIGNIA_CENTROID,
    INSIGNIA_RECT_CORNERS,
    LOGO_MATERIAL_CENTROID,
    LOGO_MATERIAL_VERTICES,
    LOGO_PLATE_CENTER,
    SFP_RAIL_TRANSLATION,
    SFP_RAIL_X,
    SFP_RAIL_Y_ABS,
    Transform,
    bbox_from_mask,
    board_coverage_corners,
    board_pose_set_is_consistent,
    estimate_board_pose,
    estimate_board_pose_from_insignia,
    evaluate_camera_coverage,
    module_coverage_corners,
    nic_sector_corners,
    project_points,
    quaternion_from_matrix,
    sampled_cartesian_path_is_safe,
    sc_sector_corners,
    search_survey_pose,
    sfp_envelope_center,
    sfp_sector_corners,
    sfp_envelope_corners,
    sfp_module_detail_boxes,
    verify_survey_view,
)
from aic_perception.board_visibility import (
    detect_insignia_polygon,
    rotation_matrix_from_quaternion,
)
from aic_perception.gripper_masks import GripperMaskBank


# A representative wrist-camera intrinsic (640x480, ~500 px focal length).
def make_camera(name="center_camera", width=640, height=480, f=500.0):
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraModel(name=name, K=K, width=width, height=height)


def axis_angle_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rpy_rotation(roll, pitch, yaw):
    """URDF fixed-joint rotation (Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    return (
        axis_angle_rotation([0, 0, 1], yaw)
        @ axis_angle_rotation([0, 1, 0], pitch)
        @ axis_angle_rotation([1, 0, 0], roll)
    )


# ---------------------------------------------------------------------------
# Transform / projection primitives.
# ---------------------------------------------------------------------------


def test_transform_inverse_roundtrip():
    R = axis_angle_rotation([0.3, -0.7, 0.5], 0.9)
    t = np.array([0.1, -0.2, 0.35])
    tf = Transform(R, t)
    pts = np.array([[0.0, 0.0, 0.0], [0.15, -0.2, 0.01], [-0.1, 0.05, 0.2]])
    back = tf.inverse().apply(tf.apply(pts))
    assert np.allclose(back, pts, atol=1e-9)


def test_transform_compose_matches_sequential_apply():
    a_T_b = Transform(axis_angle_rotation([0, 0, 1], 0.4), np.array([0.1, 0.0, 0.2]))
    b_T_c = Transform(axis_angle_rotation([1, 0, 0], -0.3), np.array([-0.05, 0.1, 0.0]))
    pts = np.array([[0.02, 0.03, 0.04], [0.1, -0.1, 0.2]])
    composed = a_T_b.compose(b_T_c).apply(pts)
    sequential = a_T_b.apply(b_T_c.apply(pts))
    assert np.allclose(composed, sequential, atol=1e-12)


def test_quaternion_matrix_roundtrip():
    R = axis_angle_rotation([0.2, 0.9, -0.3], 1.1)
    q = quaternion_from_matrix(R)
    R2 = rotation_matrix_from_quaternion(*q)
    assert np.allclose(R, R2, atol=1e-9)


def test_project_points_behind_camera_is_not_visible():
    cam = make_camera()
    pts = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    pixels, in_front = project_points(pts, cam)
    assert not in_front[0]
    assert in_front[1]
    assert np.isnan(pixels[0]).all()
    assert np.allclose(pixels[1], [cam.cx, cam.cy])


def test_camera_model_rejects_unsupported_distortion():
    with pytest.raises(ValueError, match="unsupported"):
        CameraModel(
            name="fisheye",
            K=make_camera().K,
            width=640,
            height=480,
            distortion=np.zeros(4),
            distortion_model="equidistant",
        )
    with pytest.raises(ValueError, match="coefficient count"):
        CameraModel(
            name="bad_rational",
            K=make_camera().K,
            width=640,
            height=480,
            distortion=np.zeros(5),
            distortion_model="rational_polynomial",
        )


def test_project_points_applies_plumb_bob_distortion():
    plain = make_camera()
    distorted = CameraModel(
        name="distorted",
        K=plain.K,
        width=plain.width,
        height=plain.height,
        distortion=np.array([0.18, -0.06, 0.003, -0.002, 0.01]),
        distortion_model="plumb_bob",
    )
    point = np.array([[0.12, 0.07, 0.45]])
    plain_px, _ = project_points(point, plain)
    distorted_px, _ = project_points(point, distorted)
    assert np.linalg.norm(distorted_px - plain_px) > 0.5


# ---------------------------------------------------------------------------
# Board pose estimation across arbitrary rotated poses.
# ---------------------------------------------------------------------------


def test_purple_landmark_matches_magenta4_glb_geometry():
    """The pose landmark comes from MAGENTA4, not the SDF collision centre."""
    assert np.allclose(
        LOGO_MATERIAL_VERTICES.min(axis=0),
        [-0.12250002, 0.10250002, 0.01099996],
        atol=2e-8,
    )
    assert np.allclose(
        LOGO_MATERIAL_VERTICES.max(axis=0),
        [-0.0275, 0.19750005, 0.01099999],
        atol=2e-8,
    )
    assert np.allclose(
        LOGO_MATERIAL_CENTROID,
        [-0.07335001, 0.13965003, 0.01099998],
        atol=2e-8,
    )
    # Regression guard: the old guessed collision-box centre had y=0.05 m.
    assert LOGO_MATERIAL_CENTROID[1] > 0.13
    assert LOGO_PLATE_CENTER is LOGO_MATERIAL_CENTROID


def _render_board(base_T_board, base_T_cam, camera, corners=BOARD_OUTLINE_CORNERS):
    """Project CAD board corners (+ logo) into image pixels for a known pose."""
    cam_from_base = base_T_cam.inverse()
    cam_from_board = cam_from_base.compose(base_T_board)
    quad_cam = cam_from_board.apply(corners)
    quad_px, in_front = project_points(quad_cam, camera)
    assert in_front.all()
    logo_cam = cam_from_board.apply(LOGO_PLATE_CENTER[None, :])
    logo_px, _ = project_points(logo_cam, camera)
    return quad_px, logo_px[0]


@pytest.mark.parametrize(
    "yaw_deg,tilt_deg,tilt_axis",
    [
        (0.0, 0.0, [1, 0, 0]),
        (37.0, 0.0, [1, 0, 0]),
        (-52.0, 12.0, [1, 0, 0]),
        (120.0, 18.0, [0, 1, 0]),
        (-160.0, 20.0, [1, 1, 0]),
    ],
)
def test_estimate_board_pose_recovers_arbitrary_rotation(yaw_deg, tilt_deg, tilt_axis):
    camera = make_camera()
    # Board sits ~0.5 m in front of a downward-looking camera, arbitrarily
    # rotated (yaw about its normal + a reachable tilt about an in-plane axis).
    R_board = axis_angle_rotation([0, 0, 1], math.radians(yaw_deg))
    R_board = axis_angle_rotation(tilt_axis, math.radians(tilt_deg)) @ R_board
    base_T_board = Transform(R_board, np.array([0.0, 0.0, 0.0]))
    # Camera looks down -Z of base onto the board placed below it.
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.02, -0.01, 0.55]),
    )
    quad_px, logo_px = _render_board(base_T_board, base_T_cam, camera)

    est, reason = estimate_board_pose(quad_px, logo_px, camera, base_T_cam)
    assert est is not None, reason
    assert est.reprojection_error_px < 1.0
    assert np.allclose(est.base_T_board.translation, base_T_board.translation, atol=2e-3)
    # Rotation recovered (allow sign-free comparison of the board normal).
    recovered_normal = est.base_T_board.rotation[:, 2]
    true_normal = base_T_board.rotation[:, 2]
    assert np.allclose(recovered_normal, true_normal, atol=1e-2)
    # Physical cheirality: the source camera is above the rendered +Z face.
    board_T_camera = est.base_T_board.inverse().compose(base_T_cam)
    assert board_T_camera.translation[2] > 0.012


def test_estimate_board_pose_disambiguates_with_logo():
    """The logo must pick the correct 180-degree yaw among symmetric quads."""
    camera = make_camera()
    R_board = axis_angle_rotation([0, 0, 1], math.radians(90.0))
    base_T_board = Transform(R_board, np.zeros(3))
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.0, 0.0, 0.55]),
    )
    quad_px, logo_px = _render_board(base_T_board, base_T_cam, camera)

    est, reason = estimate_board_pose(quad_px, logo_px, camera, base_T_cam)
    assert est is not None, reason
    # The recovered logo location must match the CAD quadrant, not its mirror.
    logo_base = est.base_T_board.apply(LOGO_PLATE_CENTER)
    true_logo_base = base_T_board.apply(LOGO_PLATE_CENTER)
    assert np.allclose(logo_base, true_logo_base, atol=5e-3)


def test_estimate_board_pose_handles_noisy_unordered_quad():
    camera = make_camera()
    R_board = axis_angle_rotation([0, 1, 0], math.radians(17.0))
    R_board = axis_angle_rotation([0, 0, 1], math.radians(143.0)) @ R_board
    base_T_board = Transform(R_board, np.array([0.03, -0.02, 0.01]))
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.01, -0.03, 0.62]),
    )
    quad_px, logo_px = _render_board(base_T_board, base_T_cam, camera)
    corner_noise = np.array(
        [[0.8, -0.5], [-0.6, 0.7], [0.4, 0.9], [-0.9, -0.3]]
    )
    # Deliberately scramble the detector's corner order.
    noisy_unordered = (quad_px + corner_noise)[[2, 0, 3, 1]]
    est, reason = estimate_board_pose(
        noisy_unordered, logo_px + np.array([0.5, -0.4]), camera, base_T_cam
    )
    assert est is not None, reason
    assert est.reprojection_error_px < 2.0
    assert np.allclose(
        est.base_T_board.translation, base_T_board.translation, atol=5e-3
    )
    assert np.dot(
        est.base_T_board.rotation[:, 2], base_T_board.rotation[:, 2]
    ) > 0.995


def test_estimate_board_pose_recovers_with_camera_distortion():
    plain = make_camera()
    camera = CameraModel(
        name=plain.name,
        K=plain.K,
        width=plain.width,
        height=plain.height,
        distortion=np.array([0.16, -0.04, 0.002, -0.003, 0.008]),
        distortion_model="plumb_bob",
    )
    R_board = axis_angle_rotation([0, 0, 1], math.radians(-73.0))
    R_board = axis_angle_rotation([1, 1, 0], math.radians(14.0)) @ R_board
    base_T_board = Transform(R_board, np.array([0.02, -0.01, 0.0]))
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.01, -0.02, 0.58]),
    )
    quad_px, logo_px = _render_board(base_T_board, base_T_cam, camera)
    est, reason = estimate_board_pose(quad_px, logo_px, camera, base_T_cam)
    assert est is not None, reason
    assert est.reprojection_error_px < 0.1
    assert np.allclose(
        est.base_T_board.translation, base_T_board.translation, atol=1e-3
    )
    assert np.dot(
        est.base_T_board.rotation[:, 2], base_T_board.rotation[:, 2]
    ) > 0.999


def test_estimate_board_pose_rejects_logo_ambiguity():
    camera = make_camera()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.0, 0.0, 0.55]),
    )
    quad_px, _ = _render_board(base_T_board, base_T_cam, camera)
    cam_from_board = base_T_cam.inverse().compose(base_T_board)
    center_px, _ = project_points(
        cam_from_board.apply(np.array([[0.0, 0.0, 0.012]])), camera
    )
    est, reason = estimate_board_pose(
        quad_px,
        center_px[0],
        camera,
        base_T_cam,
        max_logo_error_px=200.0,
    )
    assert est is None
    assert "disambiguate" in reason.lower()


def test_estimate_board_pose_rejects_clipped_logo():
    """A logo centroid far from any CAD-consistent projection fails closed."""
    camera = make_camera()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.0, 0.0, 0.55]),
    )
    quad_px, _ = _render_board(base_T_board, base_T_cam, camera)
    # A clipped logo reads at the image corner, nowhere near the true insignia.
    bogus_logo = np.array([2.0, 2.0])
    est, reason = estimate_board_pose(
        quad_px, bogus_logo, camera, base_T_cam, max_logo_error_px=40.0
    )
    assert est is None
    assert "logo" in reason.lower()


def test_estimate_board_pose_rejects_high_reprojection_error():
    camera = make_camera()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.0, 0.0, 0.55]),
    )
    quad_px, logo_px = _render_board(base_T_board, base_T_cam, camera)
    # Corrupt one corner heavily so no correspondence reprojects cleanly.
    quad_px = quad_px.copy()
    quad_px[0] += np.array([80.0, -60.0])
    est, reason = estimate_board_pose(
        quad_px, logo_px, camera, base_T_cam, max_reprojection_error_px=6.0
    )
    assert est is None
    assert "reprojection" in reason.lower()


def test_estimate_board_pose_rejects_non_quad():
    camera = make_camera()
    base_T_cam = Transform(np.eye(3), np.zeros(3))
    est, reason = estimate_board_pose(
        np.array([[1.0, 2.0], [3.0, 4.0]]), None, camera, base_T_cam
    )
    assert est is None
    assert "four" in reason.lower()


# ---------------------------------------------------------------------------
# Loose-SFP envelope + camera coverage.
# ---------------------------------------------------------------------------


def test_sfp_envelope_covers_both_rails_and_is_conservative():
    corners = sfp_envelope_corners()
    assert corners.shape == (8, 3)
    ys = corners[:, 1]
    # Must reach past both rails' outermost seats (+/-0.2025) with body pad.
    assert ys.min() <= -0.20
    assert ys.max() >= 0.20
    # Straddles the board centre so a single view frames both rails.
    center = sfp_envelope_center()
    assert abs(center[1]) < 1e-9
    xs = corners[:, 0]
    assert xs.min() < SFP_RAIL_X
    assert xs.max() > SFP_RAIL_X
    assert ys.max() >= SFP_RAIL_Y_ABS + SFP_RAIL_TRANSLATION


def test_evaluate_camera_coverage_feasible_and_infeasible():
    camera = make_camera()
    envelope = sfp_envelope_corners()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    # A well-placed downward camera 0.5 m above the envelope centre: enough to
    # frame the wide ~0.5 m two-rail envelope on a 65-degree-FOV camera.
    center = sfp_envelope_center()
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        center + np.array([0.0, 0.0, 0.65]),
    )
    cam_from_board = base_T_cam.inverse().compose(base_T_board)
    good = evaluate_camera_coverage(
        envelope, None, cam_from_board, camera, GripperExclusion(None)
    )
    assert good.feasible, good.reasons
    assert good.boundary_margin_px > 0

    # Too close: the envelope spills outside the frame.
    near_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        center + np.array([0.0, 0.0, 0.08]),
    )
    cam_from_board_near = near_cam.inverse().compose(base_T_board)
    bad = evaluate_camera_coverage(
        envelope, None, cam_from_board_near, camera, GripperExclusion(None)
    )
    assert not bad.feasible
    assert "envelope_outside_image" in bad.reasons


def test_evaluate_camera_coverage_rejects_gripper_intrusion():
    camera = make_camera()
    envelope = sfp_envelope_corners()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    center = sfp_envelope_center()
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        center + np.array([0.0, 0.0, 0.4]),
    )
    cam_from_board = base_T_cam.inverse().compose(base_T_board)
    # A gripper box covering the whole frame guarantees intrusion.
    gripper = GripperExclusion((0.0, 0.0, camera.width, camera.height), margin_px=8.0)
    cov = evaluate_camera_coverage(
        envelope, None, cam_from_board, camera, gripper
    )
    assert not cov.feasible
    assert "envelope_intersects_gripper" in cov.reasons


def test_gripper_exclusion_clearance_sign():
    excl = GripperExclusion((100.0, 100.0, 200.0, 200.0), margin_px=5.0)
    # Cloud well to the left: positive clearance minus margin.
    left = np.array([[10.0, 150.0], [50.0, 150.0]])
    assert excl.clearance_to(left) == pytest.approx(50.0 - 5.0)
    # Cloud overlapping the box: negative.
    overlap = np.array([[120.0, 120.0], [180.0, 180.0]])
    assert excl.clearance_to(overlap) < 0.0
    # No box -> infinite clearance.
    assert math.isinf(GripperExclusion(None).clearance_to(left))


def test_gripper_mask_preserves_clear_opening_that_bbox_would_reject():
    mask = np.zeros((240, 320), dtype=bool)
    mask[40:220, 20:60] = True
    mask[40:220, 260:300] = True
    target = np.array([[120, 100], [200, 100], [200, 160], [120, 160]], dtype=float)
    exact = GripperExclusion(mask=mask, margin_px=16.0)
    assert exact.clearance_to(target) > 0.0
    # Collapsing those disconnected fingers to one bbox would incorrectly
    # declare the clear space between them obstructed.
    coarse = GripperExclusion((20.0, 40.0, 299.0, 219.0), margin_px=16.0)
    assert coarse.clearance_to(target) < 0.0


def test_gripper_mask_rejects_pixels_inside_projected_hull():
    mask = np.zeros((240, 320), dtype=bool)
    # A thin finger crosses the interior, but touches none of the four target
    # vertices. Vertex-only overlap checks would miss it.
    mask[80:190, 158:162] = True
    target = np.array([[110, 100], [210, 100], [210, 170], [110, 170]], dtype=float)
    exact = GripperExclusion(mask=mask, margin_px=0.0)
    assert exact.clearance_to(target) < 0.0


def test_gripper_mask_enforces_positive_32px_clearance():
    mask = np.zeros((240, 320), dtype=bool)
    mask[80:180, 40:60] = True
    # Target begins about 20 pixels from the finger, so it is non-overlapping
    # but still fails the initial 32 px uncertainty clearance.
    target = np.array([[80, 100], [120, 100], [120, 160], [80, 160]], dtype=float)
    assert GripperExclusion(mask=mask).clearance_to(target) < 0.0


def test_evaluate_camera_coverage_rejects_wrong_mask_shape():
    camera = make_camera()
    envelope = sfp_envelope_corners()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    center = sfp_envelope_center()
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        center + np.array([0.0, 0.0, 0.7]),
    )
    cam_from_board = base_T_cam.inverse().compose(base_T_board)
    wrong_shape = GripperExclusion(mask=np.zeros((10, 10), dtype=bool))
    cov = evaluate_camera_coverage(
        envelope, None, cam_from_board, camera, wrong_shape
    )
    assert not cov.feasible
    assert "gripper_mask_shape_mismatch" in cov.reasons


# ---------------------------------------------------------------------------
# Candidate search across all three cameras.
# ---------------------------------------------------------------------------


def _three_camera_rig():
    """A plausible 3-wrist-camera rig: center down, sides toed-in."""
    cameras = {
        "left_camera": make_camera("left_camera"),
        "center_camera": make_camera("center_camera"),
        "right_camera": make_camera("right_camera"),
    }
    down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    tcp_T_cam = {
        "center_camera": Transform(down, np.array([0.0, 0.0, 0.05])),
        "left_camera": Transform(down, np.array([-0.04, 0.0, 0.05])),
        "right_camera": Transform(down, np.array([0.04, 0.0, 0.05])),
    }
    grippers = {name: GripperExclusion(None) for name in cameras}
    return cameras, tcp_T_cam, grippers


def _production_camera_rig():
    """Exact wrist-camera chain from the production URDF/Basler macro.

    tcp -> cam_mount: z=0.0265+0.0245+0.172 m (the fixed ATI and tool stack);
    cam_mount -> camera_link: ``ur_gz.urdf.xacro`` fixed-joint origins; and
    camera_link -> sensor -> optical: the Basler macro's fixed transforms.
    The masks are the shipped calibrated GripperMaskBank silhouettes, resized
    to the production 1152x1024 streams.
    """
    width, height, hfov = 1152, 1024, 0.8718
    focal = (width / 2.0) / math.tan(hfov / 2.0)
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    cameras = {
        name: CameraModel(name=name, K=K, width=width, height=height)
        for name in ("left_camera", "center_camera", "right_camera")
    }
    # The stack places the TCP 223 mm along cam_mount +Z, hence the inverse
    # tcp_T_mount translation used by the projection chain is negative.
    tcp_T_mount = Transform(np.eye(3), np.array([0.0, 0.0, -(0.0265 + 0.0245 + 0.172)]))
    mount_T_link = {
        "center_camera": Transform(
            rpy_rotation(0.0, -1.30899630, 1.57079623),
            np.array([0.0, -0.1077, -0.00719]),
        ),
        "left_camera": Transform(
            rpy_rotation(0.0, -1.30899630, 0.523599027),
            np.array([-0.09326, -0.053843, -0.007188]),
        ),
        "right_camera": Transform(
            rpy_rotation(0.0, -1.30899630, 2.61799343),
            np.array([0.09326, -0.053843, -0.007188]),
        ),
    }
    link_T_sensor = Transform(np.eye(3), np.array([0.02174, 0.0, 0.0145]))
    sensor_T_optical = Transform(
        rpy_rotation(-math.pi / 2.0, 0.0, -math.pi / 2.0), np.zeros(3)
    )
    tcp_T_cam = {
        name: tcp_T_mount.compose(mount_T_link[name]).compose(link_T_sensor).compose(sensor_T_optical)
        for name in cameras
    }
    bank = GripperMaskBank()
    grippers = {
        name: GripperExclusion(mask=bank.ignored_pixels(name, (height, width)), margin_px=32.0)
        for name in cameras
    }
    return cameras, tcp_T_cam, grippers


def _board_pose(yaw_deg=0.0, tilt_deg=0.0):
    R = axis_angle_rotation([0, 0, 1], math.radians(yaw_deg))
    R = axis_angle_rotation([1, 0, 0], math.radians(tilt_deg)) @ R
    # Board placed in the workspace in front of and below the base.
    from aic_perception.board_stage2 import BoardPoseEstimate

    base_T_board = Transform(R, np.array([0.4, 0.0, 0.2]))
    return BoardPoseEstimate(base_T_board, 0.3, math.inf, 0.0, "center_camera")


def test_board_pose_consistency_requires_every_camera_and_rejects_one_conflict():
    reference = _board_pose()
    estimates = {
        name: BoardPoseEstimate(
            reference.base_T_board,
            0.3,
            math.inf,
            0.0,
            name,
        )
        for name in ("left_camera", "center_camera", "right_camera")
    }
    passed, reason = board_pose_set_is_consistent(
        estimates, reference, tuple(estimates)
    )
    assert passed, reason

    conflicting = dict(estimates)
    conflicting["right_camera"] = BoardPoseEstimate(
        Transform(
            reference.base_T_board.rotation,
            reference.base_T_board.translation + np.array([0.08, 0.0, 0.0]),
        ),
        0.3,
        math.inf,
        0.0,
        "right_camera",
    )
    passed, reason = board_pose_set_is_consistent(
        conflicting, reference, tuple(conflicting)
    )
    assert not passed
    assert "right_camera" in reason

    del conflicting["left_camera"]
    passed, reason = board_pose_set_is_consistent(
        conflicting,
        reference,
        ("left_camera", "center_camera", "right_camera"),
    )
    assert not passed
    assert "missing" in reason


def test_search_survey_pose_finds_feasible_all_camera_pose():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=25.0, tilt_deg=8.0)
    candidate, reason = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert candidate is not None, reason
    assert candidate.feasible
    assert len(candidate.coverages) == 3
    assert candidate.min_clearance_px > 0.0
    # Every camera individually passes.
    assert all(c.feasible for c in candidate.coverages)


def test_search_survey_pose_maximises_min_clearance():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose()
    candidate, reason = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert candidate is not None, reason
    # The chosen candidate's min clearance is the true maximum over the grid:
    # re-scoring its own pose reproduces the reported min clearance.
    base_T_tcp = candidate.base_T_tcp
    recomputed = []
    for name, cam in cameras.items():
        base_T_camera = base_T_tcp.compose(tcp_T_cam[name])
        cam_from_board = base_T_camera.inverse().compose(board.base_T_board)
        # Re-score against the candidate's own chosen coverage target (whole
        # board or module region), disabling the same scale/detail gates the
        # search disabled, so the reported min clearance is reproduced.
        cov = evaluate_camera_coverage(
            candidate.coverage_target,
            None,
            cam_from_board,
            cam,
            grippers[name],
            min_pixel_scale=0.0,
            module_envelopes_board=(),
            min_module_pixel_scale=0.0,
        )
        recomputed.append(cov.clearance)
    assert min(recomputed) == pytest.approx(candidate.min_clearance_px, abs=1e-6)


def test_search_tie_prefers_orientation_nearest_current_pose():
    camera = make_camera()
    cameras = {"center_camera": camera}
    tcp_T_cam = {"center_camera": Transform(np.eye(3), np.zeros(3))}
    grippers = {"center_camera": GripperExclusion(None)}
    board = _board_pose()
    current, reason = search_survey_pose(
        board,
        tcp_T_cam,
        cameras,
        grippers,
        standoffs_m=(0.8,),
        offsets_x_m=(0.0,),
        offsets_y_m=(0.0,),
        yaws_rad=(0.0,),
    )
    assert current is not None, reason

    selected, reason = search_survey_pose(
        board,
        tcp_T_cam,
        cameras,
        grippers,
        standoffs_m=(0.8,),
        offsets_x_m=(0.0,),
        offsets_y_m=(0.0,),
        yaws_rad=(-math.pi, 0.0),
        current_base_T_tcp=current.base_T_tcp,
    )
    assert selected is not None, reason
    assert selected.yaw_rad == pytest.approx(0.0)
    assert selected.angular_motion_rad == pytest.approx(0.0)


def test_search_rejects_candidates_beyond_bounded_orientation_change():
    camera = make_camera()
    cameras = {"center_camera": camera}
    tcp_T_cam = {"center_camera": Transform(np.eye(3), np.zeros(3))}
    grippers = {"center_camera": GripperExclusion(None)}
    board = _board_pose()
    current, reason = search_survey_pose(
        board,
        tcp_T_cam,
        cameras,
        grippers,
        standoffs_m=(0.8,),
        offsets_x_m=(0.0,),
        offsets_y_m=(0.0,),
        yaws_rad=(0.0,),
    )
    assert current is not None, reason
    selected, reason = search_survey_pose(
        board,
        tcp_T_cam,
        cameras,
        grippers,
        standoffs_m=(0.8,),
        offsets_x_m=(0.0,),
        offsets_y_m=(0.0,),
        yaws_rad=(math.pi,),
        current_base_T_tcp=current.base_T_tcp,
        max_angular_motion_rad=math.radians(45.0),
    )
    assert selected is None
    assert "candidate" in reason


def test_search_survey_pose_fails_closed_when_gripper_blocks_all():
    cameras, tcp_T_cam, _ = _three_camera_rig()
    board = _board_pose()
    # A gripper mask spanning every frame makes all candidates infeasible.
    blocking = {
        name: GripperExclusion((0.0, 0.0, cam.width, cam.height))
        for name, cam in cameras.items()
    }
    candidate, reason = search_survey_pose(board, tcp_T_cam, cameras, blocking)
    assert candidate is None
    assert "candidate" in reason.lower()


def test_search_survey_pose_requires_reference_extrinsic():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose()
    candidate, reason = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, reference_camera="missing_camera"
    )
    assert candidate is None
    assert "extrinsic" in reason.lower()


def test_search_requires_every_camera_extrinsic_and_gripper_exclusion():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose()
    missing_extrinsic = dict(tcp_T_cam)
    del missing_extrinsic["right_camera"]
    candidate, reason = search_survey_pose(
        board, missing_extrinsic, cameras, grippers
    )
    assert candidate is None
    assert "right_camera" in reason

    missing_gripper = dict(grippers)
    del missing_gripper["left_camera"]
    candidate, reason = search_survey_pose(
        board, tcp_T_cam, cameras, missing_gripper
    )
    assert candidate is None
    assert "left_camera" in reason


@pytest.mark.parametrize(
    "yaw_deg,tilt_deg,translation",
    [
        (0.0, 0.0, (0.40, 0.0, 0.20)),
        (32.0, 8.0, (0.46, -0.08, 0.18)),
        (-41.0, 12.0, (0.36, 0.10, 0.24)),
    ],
)
def test_production_rig_default_search_finds_all_camera_safe_view(
    yaw_deg, tilt_deg, translation
):
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    R = axis_angle_rotation([1, 0, 0], math.radians(tilt_deg)) @ axis_angle_rotation(
        [0, 0, 1], math.radians(yaw_deg)
    )
    from aic_perception.board_stage2 import BoardPoseEstimate

    board = BoardPoseEstimate(
        Transform(R, np.asarray(translation, dtype=float)), 0.3, math.inf, 0.0, "center_camera"
    )
    candidate, reason = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert candidate is not None, reason
    assert candidate.feasible
    assert len(candidate.coverages) == 3
    assert all(c.gripper_clearance_px >= 0.0 for c in candidate.coverages)


def _sector_survey(sector, **overrides):
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    board = _board_pose()
    kwargs = dict(
        coverage_targets=(sector,),
        max_reach_m=0.85,
        min_height_m=0.02,
    )
    kwargs.update(overrides)
    candidate, reason = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, **kwargs
    )
    return candidate, reason, board, tcp_T_cam


def _reference_obliquity_rad(candidate, board, tcp_T_cam):
    """Angle between the board normal and the reference camera's viewpoint."""
    center = board.base_T_board.apply(candidate.coverage_target.mean(axis=0))
    to_camera = (
        candidate.base_T_tcp.compose(tcp_T_cam["center_camera"]).translation - center
    )
    to_camera = to_camera / np.linalg.norm(to_camera)
    normal = board.base_T_board.rotation[:, 2]
    return math.acos(float(np.clip(np.dot(to_camera, normal / np.linalg.norm(normal)), -1.0, 1.0)))


def test_isotropic_sfp_survey_is_close_and_near_overhead():
    """The SFP modules keep the flat, close, near-overhead framing (no band).

    NIC and SC take the directional cross-rail path instead -- covered by
    ``test_directional_sector_survey_tilts_across_the_rail_not_along_it``.
    """
    sector = sfp_sector_corners()
    candidate, reason, board, tcp_T_cam = _sector_survey(sector)
    assert candidate is not None, reason
    assert candidate.standoff_m <= 0.70
    assert math.degrees(_reference_obliquity_rad(candidate, board, tcp_T_cam)) <= 20.0
    assert candidate.min_clearance_px >= 40.0


def test_sector_survey_takes_the_closest_standoff_not_the_roomiest():
    """Standoff dominates the objective; clearance only breaks ties."""
    sector = sfp_sector_corners()
    close, reason, _, _ = _sector_survey(sector)
    assert close is not None, reason
    farther, reason, _, _ = _sector_survey(
        sector, standoffs_m=(close.standoff_m + 0.06,)
    )
    assert farther is not None, reason
    # The roomier pose exists and is rejected purely because it is farther.
    assert farther.min_clearance_px > close.min_clearance_px
    assert close.standoff_m < farther.standoff_m


def test_sector_survey_rejects_a_pose_that_only_just_fits():
    """The clearance floor is a hard reject, not a preference."""
    sector = nic_sector_corners()
    candidate, reason, _, _ = _sector_survey(sector)
    assert candidate is not None, reason
    assert candidate.min_clearance_px >= 40.0
    raised, reason, _, _ = _sector_survey(
        sector, min_required_clearance_px=candidate.min_clearance_px + 1.0
    )
    # Raising the floor cannot return the marginal pose again: whatever comes
    # back clears the raised bar.
    assert (
        raised is None
        or raised.min_clearance_px >= candidate.min_clearance_px + 1.0
    )
    # An unreachable floor fails closed instead of degrading to a tight fit.
    impossible, _, _, _ = _sector_survey(sector, min_required_clearance_px=10_000.0)
    assert impossible is None


def test_center_only_framing_reaches_a_closer_top_down_pose():
    """Dropping the all-camera requirement lets the top-down view get close.

    The three splayed cameras cannot hold the whole NIC sector nearer than
    ~0.65 m; the center camera alone can look straight down from far closer,
    which is what resolves the recessed SFP cages.
    """
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    board = _board_pose()
    tgt = nic_sector_corners()
    all_cams, r1 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers,
        coverage_targets=(tgt,), max_reach_m=0.85, min_height_m=0.02,
    )
    center_only, r2 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers,
        coverage_targets=(tgt,), max_reach_m=0.85, min_height_m=0.02,
        require_all_cameras_frame=False,
    )
    assert all_cams is not None, r1
    assert center_only is not None, r2
    # Meaningfully closer, and still essentially top-down (into the ports).
    assert center_only.standoff_m < all_cams.standoff_m - 0.1
    assert math.degrees(_reference_obliquity_rad(center_only, board, tcp_T_cam)) <= 20.0
    # The reference (center) camera genuinely frames the sector.
    ref = next(
        c for c in center_only.coverages if c.camera_name == "center_camera"
    )
    assert ref.feasible


def test_prefer_far_standoff_picks_the_high_undistorted_view():
    """Tall protruding parts want the farthest reachable standoff, not the closest.

    Up close the 145 mm cards foreshorten badly and the tool occludes an end
    card; a high, far view sees them small and undistorted, which is what the
    model-based estimator needs.
    """
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    board = _board_pose()
    tgt = nic_sector_corners()
    near, r1 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers,
        coverage_targets=(tgt,), max_reach_m=0.85, min_height_m=0.02,
        require_all_cameras_frame=False,
    )
    far, r2 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers,
        coverage_targets=(tgt,), max_reach_m=0.85, min_height_m=0.02,
        require_all_cameras_frame=False, prefer_far_standoff=True,
    )
    assert near is not None, r1
    assert far is not None, r2
    # The far preference genuinely backs the camera off, and stays reachable.
    assert far.standoff_m > near.standoff_m + 0.1
    assert np.linalg.norm(far.base_T_tcp.translation) <= 0.85
    # More standoff means more framing margin on the single framing camera.
    assert far.min_clearance_px > near.min_clearance_px


def test_nic_sector_box_frames_the_sfp_cage_band_not_the_mount_base():
    """The NIC box frames the cage band at the card tips.

    The SFP cages sit ~0.13 m out along the board normal and the tips reach
    ~0.17 m; the box spans that band (Z ~0.07..0.17) so the ports the estimator
    locks onto are framed, while staying short enough that the tilted bore-view
    pose can hold the whole band -- the full-height box clipped the tips.
    """
    z = nic_sector_corners()[:, 2]
    assert z.max() >= 0.16, z.max()          # reaches the tips / cages
    assert 0.13 >= z.min() >= 0.03           # covers the cage (~0.13), not the base


def _reference_rail_tilts(candidate, board, tcp_T_cam):
    """(cross-rail, along-rail) tilt of the reference camera, in degrees.

    Derives the rail axis exactly as the search does: the sector box's longer
    in-plane edge, taken in the estimated board frame.
    """
    target = candidate.coverage_target
    center = board.base_T_board.apply(target.mean(axis=0))
    to_camera = (
        candidate.base_T_tcp.compose(tcp_T_cam["center_camera"]).translation - center
    )
    normal = board.base_T_board.rotation[:, 2]
    board_x = board.base_T_board.rotation[:, 0]
    board_y = board.base_T_board.rotation[:, 1]
    extent = target.max(axis=0) - target.min(axis=0)
    rail, cross = (board_x, board_y) if extent[0] >= extent[1] else (board_y, board_x)
    n_comp = float(np.dot(to_camera, normal))
    cross_deg = math.degrees(math.atan2(abs(float(np.dot(to_camera, cross))), n_comp))
    along_deg = math.degrees(math.atan2(abs(float(np.dot(to_camera, rail))), n_comp))
    return cross_deg, along_deg


@pytest.mark.parametrize(
    "sector", [sc_sector_corners(), nic_sector_corners()], ids=["sc", "nic"]
)
def test_directional_sector_survey_tilts_across_the_rail_not_along_it(sector):
    """Thin protruding parts need a cross-rail tilt for a conditioned pose.

    Straight-down (near-normal) collapses each part along the viewing axis, and
    along-rail tilt makes the parts occlude one another; the survey must instead
    tilt across the rail by an in-band angle while keeping the along-rail
    component near zero.
    """
    band = (math.radians(28.0), math.radians(42.0))
    candidate, reason, board, tcp_T_cam = _sector_survey(
        sector, cross_rail_tilt_band_rad=band
    )
    assert candidate is not None, reason
    cross_deg, along_deg = _reference_rail_tilts(candidate, board, tcp_T_cam)
    assert 28.0 - 0.5 <= cross_deg <= 42.0 + 0.5, cross_deg
    assert along_deg <= 8.0 + 0.5, along_deg
    # It is decisively not the near-normal view the isotropic search would pick.
    assert cross_deg >= 20.0


def test_cross_rail_sign_keeps_the_camera_on_the_chosen_side():
    """A forced sign puts the camera on one side of the rail (the bore side).

    Recessed ports open toward one face, so the camera must sit on that side to
    look down the bore; the default (sign 0) may pick either side.
    """
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    board = _board_pose()
    tgt = nic_sector_corners()
    board_x = board.base_T_board.rotation[:, 0]
    center = board.base_T_board.apply(tgt.mean(axis=0))
    band = (math.radians(12.0), math.radians(22.0))

    def cross_offset(cand):
        cam = cand.base_T_tcp.compose(tcp_T_cam["center_camera"]).translation
        return float(np.dot(cam - center, board_x))

    neg, r1 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, coverage_targets=(tgt,),
        cross_rail_tilt_band_rad=band, cross_rail_sign=-1.0,
        require_all_cameras_frame=False, prefer_far_standoff=True,
        max_reach_m=0.80, min_height_m=0.02,
    )
    pos, r2 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, coverage_targets=(tgt,),
        cross_rail_tilt_band_rad=band, cross_rail_sign=+1.0,
        require_all_cameras_frame=False, prefer_far_standoff=True,
        max_reach_m=0.80, min_height_m=0.02,
    )
    assert neg is not None, r1
    assert pos is not None, r2
    # -1 puts the camera on the board -X side, +1 on the +X side.
    assert cross_offset(neg) < -0.1
    assert cross_offset(pos) > 0.1


def test_directional_band_leaves_the_isotropic_sfp_search_near_overhead():
    """A None band (the SFP default) keeps the flat, near-overhead behaviour."""
    candidate, reason, board, tcp_T_cam = _sector_survey(
        sfp_sector_corners(), cross_rail_tilt_band_rad=None
    )
    assert candidate is not None, reason
    cross_deg, along_deg = _reference_rail_tilts(candidate, board, tcp_T_cam)
    assert cross_deg <= 20.0 and along_deg <= 20.0


def test_production_rig_old_single_axis_grid_misses_case_new_grid_recovers():
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    # A yawed pose where the full 2-D board-plane search has a safe oblique
    # candidate, while the old centred single-X grid cannot clear all masks.
    board = _board_pose(yaw_deg=42.0, tilt_deg=10.0)
    new, reason = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert new is not None, reason
    old, _ = search_survey_pose(
        board,
        tcp_T_cam,
        cameras,
        grippers,
        # Reproduce the old implementation: short one-axis nudge, no Y
        # offset, and a top-down equivalent (only roll changes).
        standoffs_m=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
        lateral_offsets_m=(-0.03, 0.0, 0.03),
        offsets_y_m=(0.0,),
    )
    assert old is None


def test_search_uses_true_oblique_view_and_both_board_plane_axes():
    """A selected offset must change viewing direction, not only image roll."""
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=31.0, tilt_deg=9.0)
    candidate, reason = search_survey_pose(
        board,
        tcp_T_cam,
        cameras,
        grippers,
        # Force a nonzero Y probe; old search had no way to express it.
        offsets_x_m=(0.0,),
        offsets_y_m=(0.08,),
        yaws_rad=(0.0,),
    )
    assert candidate is not None, reason
    assert candidate.offset_x_m == pytest.approx(0.0)
    assert candidate.offset_y_m == pytest.approx(0.08)
    ref_cam = candidate.base_T_tcp.compose(tcp_T_cam["center_camera"])
    view_axis = ref_cam.rotation[:, 2]
    # The reference axis aims at the chosen coverage target's centroid.
    target = board.base_T_board.apply(candidate.coverage_target.mean(axis=0))
    expected = target - ref_cam.translation
    expected /= np.linalg.norm(expected)
    assert np.dot(view_axis, expected) > 0.999999
    # It is actually oblique to the board normal, so this cannot be a
    # top-down pose with only a cosmetic in-plane roll.
    assert abs(np.dot(view_axis, -board.base_T_board.rotation[:, 2])) < 0.999


def test_module_detail_gate_cannot_pass_on_large_union_span():
    camera = make_camera(width=640, height=480, f=500.0)
    envelope = sfp_envelope_corners()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    # Far enough that the broad rail envelope still spans a useful fraction of
    # the frame, while individual physical SFP boxes do not have usable detail.
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        sfp_envelope_center() + np.array([0.0, 0.0, 2.0]),
    )
    cov = evaluate_camera_coverage(
        envelope,
        None,
        base_T_cam.inverse().compose(base_T_board),
        camera,
        GripperExclusion(None),
        min_pixel_scale=0.05,
        min_module_pixel_scale=0.02,
    )
    assert cov.pixel_scale > 0.05
    assert cov.module_pixel_scale < 0.02
    assert not cov.feasible
    assert "module_too_small" in cov.reasons


def test_detail_probes_cover_all_six_staged_legal_seats():
    boxes = sfp_module_detail_boxes()
    assert len(boxes) == 6
    centers = [float(box[:, 1].mean()) for box in boxes]
    assert centers == pytest.approx([-0.15625, -0.10625, -0.05625, 0.05625, 0.10625, 0.15625])


def test_sampled_path_allows_only_monotonic_outward_close_retreat():
    origin = np.zeros(3)
    normal = np.array([0.0, 0.0, 1.0])
    close = np.array([0.10, 0.10, 0.08])
    safe = np.array([0.10, 0.10, 0.16])

    assert not sampled_cartesian_path_is_safe(
        close,
        safe,
        board_origin=origin,
        board_normal=normal,
        minimum_clearance=0.12,
    )
    assert sampled_cartesian_path_is_safe(
        close,
        safe,
        board_origin=origin,
        board_normal=normal,
        minimum_clearance=0.12,
        allow_outward_retreat=True,
    )
    assert not sampled_cartesian_path_is_safe(
        safe,
        close,
        board_origin=origin,
        board_normal=normal,
        minimum_clearance=0.12,
        allow_outward_retreat=True,
    )


# ---------------------------------------------------------------------------
# Verification across all cameras + timestamp skew.
# ---------------------------------------------------------------------------


def test_verify_survey_view_passes_for_feasible_pose():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=15.0)
    candidate, reason = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert candidate is not None, reason
    stamps = {name: 1_000_000_000 + i * 1_000_000 for i, name in enumerate(cameras)}
    # Confirm the same coverage target the pose was chosen for (as the runner
    # does); the min-motion pose just-fits that target, not a larger default.
    result = verify_survey_view(
        board,
        candidate.base_T_tcp,
        tcp_T_cam,
        cameras,
        grippers,
        stamps,
        coverage_target=candidate.coverage_target,
    )
    assert result.passed, result.reason
    assert result.skew_ok
    assert len(result.coverages) == 3


def test_verify_survey_view_rejects_timestamp_skew():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose()
    candidate, _ = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    stamps = {
        "left_camera": 1_000_000_000,
        "center_camera": 1_000_000_000,
        "right_camera": 2_000_000_000,  # 1 s late
    }
    result = verify_survey_view(
        board, candidate.base_T_tcp, tcp_T_cam, cameras, grippers, stamps
    )
    assert not result.passed
    assert not result.skew_ok
    assert "skew" in result.reason.lower()


def test_verify_survey_view_fails_when_a_camera_misses_envelope():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose()
    candidate, _ = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    # Nudge the TCP far sideways so at least one camera loses the envelope.
    bad_tcp = Transform(
        candidate.base_T_tcp.rotation,
        candidate.base_T_tcp.translation + np.array([2.0, 0.0, 0.0]),
    )
    stamps = {name: 1_000_000_000 for name in cameras}
    result = verify_survey_view(
        board, bad_tcp, tcp_T_cam, cameras, grippers, stamps
    )
    assert not result.passed


def test_verify_survey_view_requires_all_camera_stamps():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose()
    candidate, _ = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    stamps = {"center_camera": 1_000_000_000}  # only one camera
    result = verify_survey_view(
        board, candidate.base_T_tcp, tcp_T_cam, cameras, grippers, stamps
    )
    assert not result.passed


def test_bbox_from_mask():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 8:15] = True
    assert bbox_from_mask(mask) == (8.0, 5.0, 14.0, 9.0)
    assert bbox_from_mask(np.zeros((5, 5), dtype=bool)) is None


# ---------------------------------------------------------------------------
# Insignia-driven pose (clip-proof) and two-tier coverage.
# ---------------------------------------------------------------------------


def _render_insignia(base_T_board, base_T_cam, camera):
    """Project the insignia rectangle corners + centroid to pixels."""
    cam_from_board = base_T_cam.inverse().compose(base_T_board)
    quad_cam = cam_from_board.apply(INSIGNIA_RECT_CORNERS)
    quad_px, in_front = project_points(quad_cam, camera)
    assert in_front.all()
    centroid_cam = cam_from_board.apply(INSIGNIA_CENTROID[None, :])
    centroid_px, _ = project_points(centroid_cam, camera)
    return quad_px, centroid_px[0]


@pytest.mark.parametrize(
    "yaw_deg,tilt_deg,tilt_axis",
    [
        (0.0, 0.0, [1, 0, 0]),
        (40.0, 10.0, [1, 0, 0]),
        (-115.0, 15.0, [0, 1, 0]),
        (160.0, 18.0, [1, 1, 0]),
    ],
)
def test_estimate_board_pose_from_insignia_recovers_pose(yaw_deg, tilt_deg, tilt_axis):
    camera = make_camera()
    R_board = axis_angle_rotation([0, 0, 1], math.radians(yaw_deg))
    R_board = axis_angle_rotation(tilt_axis, math.radians(tilt_deg)) @ R_board
    base_T_board = Transform(R_board, np.array([0.0, 0.0, 0.0]))
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([0.02, -0.01, 0.45]),
    )
    quad_px, centroid_px = _render_insignia(base_T_board, base_T_cam, camera)

    est, reason = estimate_board_pose_from_insignia(
        quad_px, centroid_px, camera, base_T_cam
    )
    assert est is not None, reason
    assert est.reprojection_error_px < 1.0
    assert np.allclose(est.base_T_board.translation, base_T_board.translation, atol=3e-3)
    angle = _rotation_distance(est.base_T_board.rotation, base_T_board.rotation)
    assert angle < math.radians(2.0)


def _rotation_distance(a, b):
    delta = np.asarray(a).T @ np.asarray(b)
    return math.acos(float(np.clip(0.5 * (np.trace(delta) - 1.0), -1.0, 1.0)))


def test_insignia_pose_does_not_depend_on_a_visible_board_outline():
    """The insignia PnP works even when the full plate is cropped out of frame."""
    camera = make_camera()
    base_T_board = Transform(np.eye(3), np.zeros(3))
    # Close standoff so the 0.425 m plate outline would clip, but the ~9.5 cm
    # insignia stays fully framed.
    base_T_cam = Transform(
        np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        np.array([INSIGNIA_CENTROID[0], INSIGNIA_CENTROID[1], 0.30]),
    )
    # The full board outline is off-frame at this pose...
    board_cam = base_T_cam.inverse().compose(base_T_board).apply(BOARD_OUTLINE_CORNERS)
    board_px, _ = project_points(board_cam, camera)
    off_frame = (
        (board_px[:, 0] < 0).any()
        or (board_px[:, 0] > camera.width).any()
        or (board_px[:, 1] < 0).any()
        or (board_px[:, 1] > camera.height).any()
    )
    assert off_frame, "test setup should crop the plate outline"
    # ...yet the insignia PnP still recovers the pose.
    quad_px, centroid_px = _render_insignia(base_T_board, base_T_cam, camera)
    est, reason = estimate_board_pose_from_insignia(quad_px, centroid_px, camera, base_T_cam)
    assert est is not None, reason
    assert np.allclose(est.base_T_board.translation, base_T_board.translation, atol=3e-3)


def test_detect_insignia_polygon_on_a_synthetic_bracket():
    """A rendered purple bracket yields four ordered corners and a centroid."""
    W, H = 320, 240
    image = np.zeros((H, W, 3), dtype=np.uint8)
    # A magenta open bracket (three sides), asymmetric, well inside the frame.
    magenta = (211, 0, 211)  # BGR ~ H150 purple
    x0, y0, x1, y1 = 90, 70, 210, 180
    t = 12
    image[y0:y1, x0:x0 + t] = magenta          # left vertical
    image[y0:y0 + t, x0:x1] = magenta          # top horizontal
    image[y1 - t:y1, x0:x1] = magenta          # bottom horizontal (open right)
    detected = detect_insignia_polygon(image)
    assert detected is not None
    quad, centroid = detected
    assert quad.shape == (4, 2)
    # The recovered rectangle spans roughly the drawn bracket extent.
    assert quad[:, 0].min() < x0 + 6 and quad[:, 0].max() > x1 - 6
    assert quad[:, 1].min() < y0 + 6 and quad[:, 1].max() > y1 - 6
    assert 0 <= centroid[0] <= W and 0 <= centroid[1] <= H
    # A hueless (grayscale) image has no insignia.
    assert detect_insignia_polygon(np.full((H, W, 3), 40, np.uint8)) is None


def test_coverage_targets_module_is_subset_of_whole_board():
    board = board_coverage_corners()
    module = module_coverage_corners()
    assert board.shape == (8, 3) and module.shape == (8, 3)
    # Whole-board face spans the full 0.30 x 0.425 m plate.
    assert board[:, 0].min() == pytest.approx(-0.15)
    assert board[:, 0].max() == pytest.approx(0.15)
    assert board[:, 1].min() == pytest.approx(-0.2125)
    assert board[:, 1].max() == pytest.approx(0.2125)
    # The module region covers both Y-side rails including the SC rail (X~0.0985).
    assert module[:, 0].max() > SFP_RAIL_X
    assert module[:, 1].max() > SFP_RAIL_Y_ABS + SFP_RAIL_TRANSLATION


def test_search_prefers_whole_board_then_falls_back_to_module_region():
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=20.0, tilt_deg=7.0)
    # Default two-tier: whole board preferred.
    candidate, reason = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert candidate is not None, reason
    assert candidate.coverage_target is not None
    # If whole board is unreachable in all three cameras, the module region is
    # still framed (explicit single-target search proves feasibility).
    module_only, module_reason = search_survey_pose(
        board,
        tcp_T_cam,
        cameras,
        grippers,
        coverage_targets=(module_coverage_corners(),),
    )
    assert module_only is not None, module_reason
    assert all(c.feasible for c in module_only.coverages)


def test_reachability_gate_returns_the_ungated_best_when_all_reachable():
    """With a reachable() that accepts everything, the gate is a no-op: it
    returns the same pose the ungated search would pick."""
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=20.0, tilt_deg=7.0)
    ungated, r0 = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert ungated is not None, r0
    gated, r1 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, reachable=lambda pose: True
    )
    assert gated is not None, r1
    assert np.allclose(gated.base_T_tcp.translation, ungated.base_T_tcp.translation)


def test_reachability_gate_rejects_all_and_reports_framed_but_unreachable():
    """When nothing is reachable, the search returns no pose and says so -- it
    never publishes an unreachable pose."""
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=20.0, tilt_deg=7.0)
    candidate, reason = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, reachable=lambda pose: False
    )
    assert candidate is None
    assert "reachable" in reason and "framed" in reason


def test_reachability_gate_falls_through_to_the_next_reachable_candidate():
    """If the top-ranked pose is unreachable, the gate commits to the best
    remaining pose the arm can actually achieve, not None."""
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=20.0, tilt_deg=7.0)
    best, r0 = search_survey_pose(board, tcp_T_cam, cameras, grippers)
    assert best is not None, r0
    top = np.asarray(best.base_T_tcp.translation)
    # The gate receives each candidate's base_T_tcp Transform.  Reject exactly
    # the ungated best pose; accept everything else.
    def reject_top(base_T_tcp):
        return not np.allclose(base_T_tcp.translation, top)
    gated, r1 = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, reachable=reject_top
    )
    assert gated is not None, r1
    assert not np.allclose(gated.base_T_tcp.translation, top)


def test_reachability_gate_scans_every_framed_candidate_not_a_shortlist():
    """The one reachable pose may rank last, so the gate must scan the lot.

    This is the production failure this gate was written for: for the NIC sector
    only the *closest* framed poses are inside the UR5e envelope, while
    ``prefer_far_standoff`` ranks the far ones first -- so the single reachable
    candidate sat ~260 places down a 263-long ranking.  A gate that only checked
    a shortlist of the best-ranked poses reported "framed N but none reachable"
    and Stage 2 returned done=false with a perfectly good pose available.
    """
    cameras, tcp_T_cam, grippers = _three_camera_rig()
    board = _board_pose(yaw_deg=20.0, tilt_deg=7.0)
    seen = []

    def census(base_T_tcp):
        seen.append(np.asarray(base_T_tcp.translation))
        return False

    assert search_survey_pose(
        board, tcp_T_cam, cameras, grippers, reachable=census
    )[0] is None
    assert len(seen) > 30, "too few framed candidates to prove the point"
    last = seen[-1]

    # Only the very last-ranked pose is reachable: the gate must still find it.
    def only_the_last(base_T_tcp):
        return bool(np.allclose(base_T_tcp.translation, last))

    gated, reason = search_survey_pose(
        board, tcp_T_cam, cameras, grippers, reachable=only_the_last
    )
    assert gated is not None, reason
    assert np.allclose(gated.base_T_tcp.translation, last)
