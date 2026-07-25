from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))

import aic_model.sc_controller as sc_controller  # noqa: E402
from aic_model.rl_insert_contract import (  # noqa: E402
    matrix_to_quat,
    port_frame,
    quat_to_matrix,
)
from aic_model.v50_controller import rotation_from_axis_angle  # noqa: E402
from aic_model.sc_controller import (  # noqa: E402
    SC_TIP_IN_TCP_POS,
    SCConfig,
    SC_INSERT_DEPTH_M,
    SC_OPENING_HEIGHT_M,
    SC_OPENING_WIDTH_M,
    ScInsertionController,
    _normalize_event,
    _select_sc_detections_for_triangulation,
    classify_opening,
    next_sc_depth,
    sc_tip_pose_from_tcp,
    tcp_pose_for_sc_tip,
)


def test_axial_lead_cannot_exceed_the_sc_bore():
    config = SCConfig().validated()
    # The SFP ladder (8 N / 500 N/m = 16 mm) is longer than the whole 15.64 mm
    # SC insertion; the SC config must bound the lead well inside the bore.
    assert config.force_lead_m < SC_INSERT_DEPTH_M
    assert np.isclose(config.force_lead_m, 0.005)

    # Guard the failure mode explicitly: copying the SFP numbers must not
    # validate.
    with pytest.raises(ValueError, match="must stay inside"):
        SCConfig(target_axial_force_n=8.0, seat_force_cap_n=10.0,
                 force_abort_n=18.0, max_axial_lead_m=0.020).validated()


def test_config_bounds_are_scaled_to_the_bore():
    config = SCConfig().validated()
    assert config.target_axial_force_n < config.seat_force_cap_n < config.force_abort_n
    assert 0.0 < config.seat_candidate_depth_m <= SC_INSERT_DEPTH_M
    assert 0.0 <= config.seat_overtravel_m <= 0.003
    # Overtravel and mouth zone should stay proportionate to SFP's fractions of
    # its own bore (~11% and ~13%).
    assert 0.05 < config.seat_overtravel_m / SC_INSERT_DEPTH_M < 0.15
    assert 0.05 < config.seat_mouth_zone_m / SC_INSERT_DEPTH_M < 0.20

    with pytest.raises(ValueError, match="candidate depth"):
        SCConfig(seat_candidate_depth_m=0.030).validated()
    with pytest.raises(ValueError, match="overtravel"):
        SCConfig(seat_overtravel_m=0.004).validated()


def test_persistent_depth_stops_at_bore_plus_overtravel():
    config = SCConfig().validated()
    current_depth = 0.002
    command_depth = current_depth
    for _ in range(500):
        command_depth = next_sc_depth(
            current_depth, command_depth, 0.1, force_n=0.5, config=config
        )
    # Stuck at 2 mm, the setpoint must saturate at the bounded lead, never run
    # to the far end of the port.
    assert np.isclose(command_depth, current_depth + config.force_lead_m)
    assert command_depth < SC_INSERT_DEPTH_M

    deep = SC_INSERT_DEPTH_M - 0.0005
    command_depth = deep
    for _ in range(500):
        command_depth = next_sc_depth(deep, command_depth, 0.1, force_n=0.5, config=config)
    assert np.isclose(command_depth, SC_INSERT_DEPTH_M + config.seat_overtravel_m)


def test_persistent_depth_holds_at_the_force_cap():
    config = SCConfig().validated()
    held = next_sc_depth(0.004, 0.008, 1.0, force_n=config.seat_force_cap_n, config=config)
    assert np.isclose(held, 0.008)


def test_opening_classification_names_the_two_real_label_conventions():
    # The hypotheses must be what the COLLECTORS project, not SDF geometry: the
    # model outlines neither the duplex opening nor a bore.
    label, residual, _ = classify_opening(0.0088, 0.0060)
    assert label == "gt_label", "DataCollectorScPoseGT's 8.8x6.0 rectangle"
    assert residual < 1e-9

    label, residual, _ = classify_opening(0.02578, 0.00927)
    assert label == "outer_face", "DataCollectorPoseSC's 25.78x9.27 outer face"
    assert residual < 1e-9


def test_field_measured_opening_classifies_as_the_shipped_label_convention():
    # 2026-07-25 run: 7.09 x 4.06 mm triangulated. Undersized against the label
    # because the outer cameras detect the target weakly, but unambiguously the
    # gt_label convention rather than the 25.78 mm face.
    label, residual, _ = classify_opening(0.00709, 0.00406)
    assert label == "gt_label"
    assert residual < sc_controller.SC_OPENING_RESIDUAL_WARN_M, (
        "known triangulation shrinkage must not trip the 'unknown convention' warning"
    )


def test_no_bore_offset_is_ever_reported():
    # Both collectors project from sc_port_base_link_entrance -- the duplex
    # centre -- so a detection is never half a bore off. The old code returned
    # half a pitch here and logged a warning telling the operator to correct by
    # 6.35 mm, which would have pushed the plug off-centre by exactly that.
    for w, h in ((0.0088, 0.0060), (0.02578, 0.00927), (0.00709, 0.00406),
                 (0.00971, 0.00785)):
        assert classify_opening(w, h)[2] == 0.0


def test_pnp_rectangle_matches_the_convention_the_weights_emit():
    # PnP scales the pose by the ratio of this rectangle to the observed one, so
    # the old 22.41 mm duplex entry would have placed the port ~2.5x too far.
    spans = sc_controller.LOCAL_SC_PORT_KPS.max(axis=0) - sc_controller.LOCAL_SC_PORT_KPS.min(axis=0)
    assert np.isclose(spans[0], 0.0088)
    assert np.isclose(spans[1], 0.0060)
    assert np.allclose(sc_controller.LOCAL_SC_PORT_KPS.mean(axis=0), 0.0), (
        "must stay centred on the entrance, which is what both collectors project from"
    )


def test_clear_opening_height_is_the_binding_value_not_the_roomiest():
    # The channel height varies along its depth: +4.050 under the full-depth
    # _mid* rails, but +3.800 through the 10.8 mm-deep cube_collider_box.001
    # lip, which the plug must pass. Budgeting against the roomier 8.10 mm
    # overstates vertical clearance by 17%.
    assert np.isclose(SC_OPENING_HEIGHT_M, 0.00785)
    vertical_clearance = (SC_OPENING_HEIGHT_M - sc_controller.SC_PLUG_HEIGHT_M) / 2.0
    assert np.isclose(vertical_clearance, 0.000725)
    lateral_clearance = (SC_OPENING_WIDTH_M - sc_controller.SC_PLUG_WIDTH_M) / 2.0
    assert vertical_clearance < lateral_clearance, "vertical is the binding axis"


def test_tip_transform_round_trips():
    tcp_pos = np.array([-0.31, 0.39, 0.24])
    tcp_quat = np.array([0.965925826, 0.0, 0.258819045, 0.0])
    tip_pos, R_tip = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
    recovered_tcp_pos, R_tcp = tcp_pose_for_sc_tip(tip_pos, R_tip)
    np.testing.assert_allclose(recovered_tcp_pos, tcp_pos, atol=1e-12)
    round_trip_tip, _ = sc_tip_pose_from_tcp(
        recovered_tcp_pos, np.array([0.965925826, 0.0, 0.258819045, 0.0])
    )
    np.testing.assert_allclose(round_trip_tip, tip_pos, atol=1e-12)


def test_event_normalisation_strips_the_cable_prefix():
    # The scoring topic prefixes the cable instance; the task does not. v50's
    # normaliser leaves the prefix on, so its equality test cannot match.
    assert (
        _normalize_event("cable_0#0#sc_mount_rail_0/sc_port_0")
        == "sc_mount_rail_0/sc_port_0"
    )
    assert _normalize_event("sc_mount_rail_0/sc_port_0") == "sc_mount_rail_0/sc_port_0"
    assert _normalize_event(None) == ""


class _AlignHarness(ScInsertionController):
    def __init__(self, f_plug, m_plug, config=None):
        self.config = (config or SCConfig()).validated()
        self._f = np.asarray(f_plug, dtype=np.float64)
        self._m = np.asarray(m_plug, dtype=np.float64)

    def _wrench_plug_frame(self, observation):
        return self._f, self._m


class _FakeProjectionCore:
    def build_projection_matrix(self, K, T):
        return np.array(
            [
                [0.0, 0.0, 0.0, 100.0],
                [0.0, 0.0, 0.0, 100.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


class _FakeScPolicy:
    def __init__(self, tcp_available=True):
        self._pc = _FakeProjectionCore()
        self._tcp_available = tcp_available

    def _tcp(self):
        if not self._tcp_available:
            raise RuntimeError("tcp unavailable")
        return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _sc_det(cx, cy, conf):
    kps = np.array(
        [
            [cx - 1.0, cy - 1.0],
            [cx + 1.0, cy - 1.0],
            [cx + 1.0, cy + 1.0],
            [cx - 1.0, cy + 1.0],
        ],
        dtype=np.float64,
    )
    return {
        "kps": kps,
        "conf": conf,
        "K": np.eye(3, dtype=np.float64),
        "T": np.eye(4, dtype=np.float64),
        "P": np.eye(3, 4, dtype=np.float64),
    }


def test_sc_tip_prefilter_drops_detection_outside_radius():
    per_cam = {
        "cam_a": [_sc_det(100.0, 100.0, 0.7), _sc_det(500.0, 100.0, 0.9)],
        "cam_b": [_sc_det(100.0, 100.0, 0.8)],
    }

    selected = _select_sc_detections_for_triangulation(_FakeScPolicy(), per_cam)

    assert len(selected["cam_a"]) == 1
    np.testing.assert_allclose(np.mean(selected["cam_a"][0]["kps"], axis=0), [100.0, 100.0])


def test_sc_tip_prefilter_keeps_detection_inside_radius():
    per_cam = {
        "cam_a": [_sc_det(240.0, 100.0, 0.4)],
        "cam_b": [_sc_det(100.0, 100.0, 0.8)],
    }

    selected = _select_sc_detections_for_triangulation(_FakeScPolicy(), per_cam)

    assert selected["cam_a"] == per_cam["cam_a"]


def test_sc_tip_prefilter_falls_back_to_confidence_when_tip_unavailable():
    per_cam = {
        "cam_a": [_sc_det(900.0, 900.0, 0.1), _sc_det(800.0, 800.0, 0.9)],
        "cam_b": [_sc_det(700.0, 700.0, 0.8)],
    }

    selected = _select_sc_detections_for_triangulation(
        _FakeScPolicy(tcp_available=False), per_cam
    )

    assert len(selected["cam_a"]) == 2
    assert selected["cam_a"][0]["conf"] == 0.9
    assert selected["cam_a"][1]["conf"] == 0.1


def test_sc_tip_prefilter_keeps_each_camera_nonempty_when_detection_is_in_radius():
    per_cam = {
        "cam_a": [_sc_det(500.0, 100.0, 0.9), _sc_det(99.0, 101.0, 0.3)],
        "cam_b": [_sc_det(-300.0, 100.0, 0.9), _sc_det(101.0, 99.0, 0.4)],
    }

    selected = _select_sc_detections_for_triangulation(_FakeScPolicy(), per_cam)

    assert all(selected[cam] for cam in per_cam)
    np.testing.assert_allclose(np.mean(selected["cam_a"][0]["kps"], axis=0), [99.0, 101.0])
    np.testing.assert_allclose(np.mean(selected["cam_b"][0]["kps"], axis=0), [101.0, 99.0])


def test_alignment_is_proportional_and_never_saturates_on_light_contact():
    config = SCConfig().validated()
    harness = _AlignHarness(f_plug=[1.6, -0.6, -3.0], m_plug=[0.0, -0.18, 0.0], config=config)
    acc_lat = np.zeros(2, dtype=np.float64)
    acc_tilt = np.zeros(2, dtype=np.float64)

    # The SFP field failure: an accumulator pinned at its clamp a few samples
    # after first chamfer touch and jammed the plug. This law must not.
    for _ in range(60):
        acc_lat, acc_tilt, _sample = harness._alignment_sample(
            None, 0.0, 3.0, acc_lat, acc_tilt
        )
        assert np.linalg.norm(acc_lat) < 0.25 * config.seat_align_max_lat_m
        assert np.linalg.norm(acc_tilt) < 0.25 * config.seat_align_max_tilt_rad


def test_alignment_washes_out_when_contact_is_lost():
    config = SCConfig().validated()
    harness = _AlignHarness(f_plug=[3.2, -3.0, -6.0], m_plug=[0.0, -0.65, 0.0], config=config)
    acc_lat = np.zeros(2, dtype=np.float64)
    acc_tilt = np.zeros(2, dtype=np.float64)

    for _ in range(60):
        acc_lat, acc_tilt, _sample = harness._alignment_sample(
            None, 0.0, 6.0, acc_lat, acc_tilt
        )
    settled = np.linalg.norm(acc_lat)
    assert np.isclose(settled, config.seat_align_force_gain * np.linalg.norm([3.2, -3.0]))

    for _ in range(30):
        acc_lat, acc_tilt, _sample = harness._alignment_sample(
            None, 0.0, 0.5, acc_lat, acc_tilt
        )
    assert np.linalg.norm(acc_lat) < 0.02 * settled


# --- pre-filter anchoring and emptied-camera reporting ---------------------
_FX = 1000.0
_CX = 576.0
_CY = 512.0
_TCP_Z = 0.5


class _PinholeCore:
    """A real pinhole projection, so the filter's anchor point actually matters.

    _FakeProjectionCore maps every 3D point to the same pixel, which cannot tell
    a TCP-anchored filter from a tip-anchored one.
    """

    def build_projection_matrix(self, K, T):
        return np.array(
            [[_FX, 0.0, _CX, 0.0],
             [0.0, _FX, _CY, 0.0],
             [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )


class _PinholePolicy:
    def __init__(self):
        self._pc = _PinholeCore()

    def _tcp(self):
        # Identity orientation, so tip = tcp + SC_TIP_IN_TCP_POS exactly.
        return (
            np.array([0.0, 0.0, _TCP_Z], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )


class _RecordingLog:
    def __init__(self):
        self.info_lines = []
        self.warn_lines = []
        self.error_lines = []

    def info(self, message):
        self.info_lines.append(str(message))

    def warn(self, message):
        self.warn_lines.append(str(message))

    def error(self, message):
        self.error_lines.append(str(message))


def _project(point):
    x, y, z = point
    return np.array([_FX * x / z + _CX, _FX * y / z + _CY], dtype=np.float64)


def test_sc_prefilter_anchors_on_the_tcp_not_the_uncalibrated_tip(monkeypatch):
    # The SC tip transform is the uncalibrated SFP default, so centring a
    # perception gate on it would inherit that error.
    tcp_px = _project([0.0, 0.0, _TCP_Z])
    tip_px = _project(np.array([0.0, 0.0, _TCP_Z]) + SC_TIP_IN_TCP_POS)
    separation = float(np.linalg.norm(tip_px - tcp_px))
    assert separation > 10.0, "fixture must separate the two anchors"

    # Radius small enough that only the correct anchor's detection survives.
    monkeypatch.setattr(sc_controller, "SC_MAX_DETECT_PX_FROM_TIP", separation / 2.0)
    per_cam = {
        "cam_a": [_sc_det(*tcp_px, 0.5), _sc_det(*tip_px, 0.9)],
        "cam_b": [_sc_det(*tcp_px, 0.5)],
    }

    selected = _select_sc_detections_for_triangulation(_PinholePolicy(), per_cam)

    assert len(selected["cam_a"]) == 1
    np.testing.assert_allclose(
        np.mean(selected["cam_a"][0]["kps"], axis=0), tcp_px, atol=1e-6
    )


def test_sc_prefilter_names_the_camera_it_empties(monkeypatch):
    # Emptying a camera makes triangulation impossible, and the caller only says
    # "no candidates"; the pre-filter must say it was the cause.
    monkeypatch.setattr(sc_controller, "SC_MAX_DETECT_PX_FROM_TIP", 5.0)
    tcp_px = _project([0.0, 0.0, _TCP_Z])
    per_cam = {
        "cam_a": [_sc_det(tcp_px[0] + 400.0, tcp_px[1], 0.9)],
        "cam_b": [_sc_det(*tcp_px, 0.8)],
    }
    log = _RecordingLog()

    selected = _select_sc_detections_for_triangulation(_PinholePolicy(), per_cam, log=log)

    assert selected["cam_a"] == []
    assert selected["cam_b"]
    assert any("removed all" in line and "cam_a" in line for line in log.warn_lines)
    assert any("mode=tcp_filter" in line for line in log.info_lines)


def test_sc_prefilter_does_not_warn_when_a_camera_survives(monkeypatch):
    monkeypatch.setattr(sc_controller, "SC_MAX_DETECT_PX_FROM_TIP", 250.0)
    tcp_px = _project([0.0, 0.0, _TCP_Z])
    per_cam = {
        "cam_a": [_sc_det(*tcp_px, 0.9)],
        "cam_b": [_sc_det(*tcp_px, 0.8)],
    }
    log = _RecordingLog()

    _select_sc_detections_for_triangulation(_PinholePolicy(), per_cam, log=log)

    assert log.warn_lines == []


# --- the size gate, against the label convention the shipped weights use ------
#
# best_sc_pose.pt does not outline the port.  It labels a rectangle centred on
# the mouth measuring 8.8 x 6.0 mm -- about a quarter of the 25.78 mm adapter,
# confirmed by running the weights over testing/check_sc_previews.  The gate
# must be sized against THAT, not against the port's physical dimensions.

_LABEL_WIDTH_M = 0.0088
_LABEL_HEIGHT_M = 0.0060
_STEREO_BASELINE_M = 0.030
_PORT_DEPTH_M = 0.150


class _StereoCore:
    """Two real cameras and a real DLT triangulation.

    _FakeProjectionCore collapses every point onto one pixel and _PinholeCore is
    monocular; neither can exercise a gate that acts on triangulated *size*.
    """

    @staticmethod
    def build_projection_matrix(K, T_cam_from_world):
        return K @ T_cam_from_world[:3, :4]

    @staticmethod
    def triangulate(pts_2d, Ps):
        rows = []
        for (u, v), P in zip(pts_2d, Ps):
            rows.append(u * P[2] - P[0])
            rows.append(v * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        X = Vt[-1]
        return X[:3] / X[3]


def _stereo_policy(log=None):
    # The reprojection and orientation steps are taken from RLInsert itself
    # rather than reimplemented, so this exercises the shipped geometry.
    # Imported lazily: a failure here must not break collection of this file.
    from aic_model.RLInsert import RLInsert

    class _StereoPolicy:
        _reproject_error_px = RLInsert._reproject_error_px
        _estimate_sfp_port_orientation = RLInsert._estimate_sfp_port_orientation

        def __init__(self):
            self._pc = _StereoCore()
            self._log = log if log is not None else _RecordingLog()

        def get_logger(self):
            return self._log

        def _tcp(self):
            # On the port's optical axis, so the pre-filter keeps both cameras.
            return (
                np.array([0.0, 0.0, _PORT_DEPTH_M], dtype=np.float64),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )

    return _StereoPolicy()


def _stereo_views(width_m, height_m):
    """Per-camera detections of one centred rectangle, in LOCAL_SC_PORT_KPS order."""
    K = np.array([[_FX, 0.0, _CX], [0.0, _FX, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)
    corners = np.array(
        [
            [+width_m / 2.0, +height_m / 2.0, _PORT_DEPTH_M],
            [-width_m / 2.0, +height_m / 2.0, _PORT_DEPTH_M],
            [-width_m / 2.0, -height_m / 2.0, _PORT_DEPTH_M],
            [+width_m / 2.0, -height_m / 2.0, _PORT_DEPTH_M],
        ],
        dtype=np.float64,
    )

    per_cam = {}
    for name, shift in (("cam_a", 0.0), ("cam_b", -_STEREO_BASELINE_M)):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = shift
        P = _StereoCore.build_projection_matrix(K, T)
        kps = []
        for corner in corners:
            x = P @ np.array([*corner, 1.0], dtype=np.float64)
            kps.append([x[0] / x[2], x[1] / x[2]])
        per_cam[name] = [
            {"kps": np.array(kps, dtype=np.float64), "conf": 0.9, "K": K, "T": T, "P": P}
        ]
    return per_cam


def test_size_gate_admits_the_shipped_label_rectangle():
    candidates = sc_controller.sc_multiview_candidates(
        _stereo_policy(), _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M)
    )

    assert candidates, "the model's own 8.8x6.0mm label must not be rejected"
    assert candidates[0]["width"] == pytest.approx(_LABEL_WIDTH_M, abs=1e-5)
    assert candidates[0]["height"] == pytest.approx(_LABEL_HEIGHT_M, abs=1e-5)


def test_size_gate_survives_a_realistic_underestimate_of_the_short_axis():
    # The 2026-07-25 field run lost every frame here.  A 6.0 mm axis measured
    # 20% short is 4.8 mm, which the old 5 mm floor discarded outright; the
    # floor has to leave room for triangulation noise, not just the nominal.
    candidates = sc_controller.sc_multiview_candidates(
        _stereo_policy(), _stereo_views(_LABEL_WIDTH_M * 0.8, _LABEL_HEIGHT_M * 0.8)
    )

    assert candidates, "a 20% short measurement must still yield a candidate"


def test_size_gate_still_rejects_something_far_too_small():
    # Loosening the floor must not turn the gate off altogether.
    candidates = sc_controller.sc_multiview_candidates(
        _stereo_policy(), _stereo_views(0.0008, 0.0005)
    )

    assert candidates == []


def test_rejected_combinations_name_the_gate_and_the_measurement():
    # "no candidates" alone cost a field run to diagnose; the log must say which
    # gate fired and what it measured.
    log = _RecordingLog()

    candidates = sc_controller.sc_multiview_candidates(
        _stereo_policy(log), _stereo_views(0.0008, 0.0005)
    )

    assert candidates == []
    line = "\n".join(log.warn_lines)
    assert "SC_PERCEPT_REJECT" in line
    assert "'size'" in line
    assert "0.8x0.5mm" in line, "the measured rectangle must appear, not just the reason"


# ---------------------------------------------------------------------------
# Seat frame: insertion axis from perception, in-plane twist from the plug.
# ---------------------------------------------------------------------------
# Field run 2026-07-25 logged rot_err_deg=[3.19, -4.37, -89.55] on every one of
# the 7 consensus frames.  The controller chased that ~90 deg and timed out.  It
# is a frame-convention offset (SFP grasp transform standing in for the
# uncalibrated SC one), not a real misalignment -- see SC_PRESERVE_HANDOFF_YAW.
FIELD_PORT_QUAT = np.array([0.0, 0.68978, 0.72402, 0.0], dtype=np.float64)
FIELD_ROT_ERR_DEG = np.array([3.19, -4.37, -89.55], dtype=np.float64)


def _field_frames():
    """The perceived port frame and plug orientation from the 2026-07-25 run."""
    Rp = port_frame(FIELD_PORT_QUAT)
    R_tip = Rp @ rotation_from_axis_angle(np.radians(FIELD_ROT_ERR_DEG))
    return Rp, R_tip


class _FakeNode:
    _insertion_event_generation = 0


class _StubPolicy:
    """Enough policy for __init__ and _errors(); no ROS, no motion."""

    def __init__(self, tcp_pos, tcp_quat):
        self._parent_node = _FakeNode()
        self._tcp_pos = np.asarray(tcp_pos, dtype=np.float64)
        self._tcp_quat = np.asarray(tcp_quat, dtype=np.float64)

    def get_logger(self):
        return _RecordingLog()

    def _tcp(self):
        return self._tcp_pos, self._tcp_quat


class _StubTask:
    target_module_name = "sc_port_0"
    port_name = "sc_port_base"
    cable_name = "cable_0"
    plug_name = "sc_tip"


def _controller_for(R_tip, port_pos, Rp, Rs):
    """Build a controller whose plug currently sits at ``R_tip``."""
    R_tcp = R_tip @ quat_to_matrix(sc_controller.SC_TIP_IN_TCP_QUAT).T
    tcp_quat = matrix_to_quat(R_tcp)
    # place the TCP so the derived tip lands exactly on the port mouth
    tcp_pos = np.zeros(3, dtype=np.float64)
    tip_pos, _ = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
    tcp_pos = port_pos - (tip_pos - tcp_pos)
    return ScInsertionController(
        _StubPolicy(tcp_pos, tcp_quat), _StubTask(),
        lambda: None, None, lambda *_: None,
        port_pos=port_pos, port_quat=FIELD_PORT_QUAT, Rp=Rp, Rs=Rs,
    )


def test_seat_frame_keeps_the_insertion_axis_and_takes_the_twist_from_the_plug():
    Rp, R_tip = _field_frames()

    Rs = sc_controller.seat_frame(Rp, R_tip)

    assert np.allclose(Rs[:, 2], Rp[:, 2]), "insertion axis must still come from perception"
    assert np.allclose(Rs.T @ Rs, np.eye(3), atol=1e-9), "must stay orthonormal"
    assert float(np.linalg.det(Rs)) == pytest.approx(1.0), "must stay right-handed"
    # Rs[:,0] is the plug's own x-axis flattened into the opening plane.
    flattened = R_tip[:, 0] - float(np.dot(R_tip[:, 0], Rp[:, 2])) * Rp[:, 2]
    assert np.allclose(Rs[:, 0], flattened / np.linalg.norm(flattened))


def test_field_2026_07_25_yaw_offset_no_longer_costs_60_alignment_steps():
    """The regression this change exists for: reproduce the run that timed out.

    _align slews at align_max_rotation_step_rad per iteration, so the rotation
    error divided by that cap is the number of iterations alignment needs.  The
    old behaviour needed ~60 against a 15 s budget and never converged.
    """
    Rp, R_tip = _field_frames()
    port_pos = np.array([-0.32466, 0.12952, 0.03032], dtype=np.float64)
    step = SCConfig().validated().align_max_rotation_step_rad

    chasing = _controller_for(R_tip, port_pos, Rp, Rs=Rp)          # old behaviour
    preserving = _controller_for(R_tip, port_pos, Rp,
                                 Rs=sc_controller.seat_frame(Rp, R_tip))

    _, _, old_err, _, _ = chasing._errors()
    _, _, new_err, _, _ = preserving._errors()
    old_deg = float(np.degrees(np.linalg.norm(old_err)))
    new_deg = float(np.degrees(np.linalg.norm(new_err)))

    assert old_deg == pytest.approx(89.71, abs=0.1), "fixture must reproduce the field error"
    assert np.ceil(np.radians(old_deg) / step) >= 55, "old behaviour needed ~60 steps"
    # What is left is the genuine handoff tilt, inside the 6.9 deg budget the
    # 1.2 mm lateral clearance allows on a 20 mm plug.
    assert new_deg < 6.9
    assert np.ceil(np.radians(new_deg) / step) <= 5, "must now converge in a few steps"


def test_seat_frame_does_not_disturb_position_servoing():
    # Rp and Rs share column 2, so the lateral plane is identical; only rotation
    # targets may move.  If a position term ever migrates onto Rs this fails.
    Rp, R_tip = _field_frames()
    port_pos = np.array([-0.32466, 0.12952, 0.03032], dtype=np.float64)
    offset = np.array([0.0012, -0.0008, 0.004], dtype=np.float64)

    chasing = _controller_for(R_tip, port_pos - offset, Rp, Rs=Rp)
    preserving = _controller_for(R_tip, port_pos - offset, Rp,
                                 Rs=sc_controller.seat_frame(Rp, R_tip))

    old_depth, old_lat, _, _, _ = chasing._errors()
    new_depth, new_lat, _, _, _ = preserving._errors()

    assert old_depth == pytest.approx(new_depth)
    assert np.allclose(old_lat, new_lat)


def test_seat_tilt_correction_is_applied_in_the_frame_it_was_measured_in():
    """_wrench_plug_frame resolves the wrench through Rp, so acc_tilt is about
    Rp's lateral axes.  _seat must compose it as ``Rp @ tilt @ R_yaw``; writing
    the obvious ``Rs @ tilt`` instead would rotate the correction by the very
    offset the seat frame exists to absorb.
    """
    Rp, R_tip = _field_frames()
    Rs = sc_controller.seat_frame(Rp, R_tip)
    ctrl = _controller_for(R_tip, np.zeros(3), Rp, Rs=Rs)
    tilt = rotation_from_axis_angle(np.array([0.010, -0.020, 0.0]))

    correct = ctrl.Rp @ tilt @ ctrl.R_yaw
    naive = ctrl.Rs @ tilt

    # the tilt lands on the world axis it was actually measured about
    assert np.allclose(correct @ ctrl.Rs.T, Rp @ tilt @ Rp.T)
    assert not np.allclose(correct, naive), "the 90 deg offset really does matter here"


def test_seat_frame_falls_back_when_the_plug_has_no_in_plane_direction():
    Rp, _ = _field_frames()
    # plug x-axis parallel to the insertion axis: nothing to preserve
    degenerate = np.column_stack([Rp[:, 2], Rp[:, 0], Rp[:, 1]])

    assert np.allclose(sc_controller.seat_frame(Rp, degenerate), Rp)


def test_controller_defaults_to_the_perception_frame():
    # Omitting Rs must reproduce the pre-2026-07-25 behaviour exactly.
    Rp, R_tip = _field_frames()
    ctrl = ScInsertionController(
        _StubPolicy(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])), _StubTask(),
        lambda: None, None, lambda *_: None,
        port_pos=np.zeros(3), port_quat=FIELD_PORT_QUAT, Rp=Rp,
    )

    assert np.allclose(ctrl.Rs, ctrl.Rp)
    assert np.allclose(ctrl.R_yaw, np.eye(3))


def test_run_sc_insertion_hands_the_controller_a_preserved_twist_frame(monkeypatch):
    """Guard the wiring, not just the geometry.

    The unit tests above construct the controller directly, so they would still
    pass if run_sc_insertion quietly stopped building a seat frame -- exactly the
    kind of thing a rebase drops. Assert the frame actually reaches the
    controller.
    """
    Rp, R_tip = _field_frames()
    port_pos = np.array([-0.32466, 0.12952, 0.03032], dtype=np.float64)
    R_tcp = R_tip @ quat_to_matrix(sc_controller.SC_TIP_IN_TCP_QUAT).T
    tcp_quat = matrix_to_quat(R_tcp)
    tip_pos, _ = sc_tip_pose_from_tcp(np.zeros(3), tcp_quat)
    tcp_pos = port_pos - tip_pos

    seen = {}

    class _CapturingController:
        def __init__(self, *_args, **kwargs):
            seen.update(kwargs)

        def run(self):
            return True

    class _RunPolicy(_StubPolicy):
        _wrench_baseline = None

        def _wrench_vector(self, _obs):
            return np.zeros(6, dtype=np.float64)

    monkeypatch.setattr(sc_controller, "perceive_sc_port_pose_consensus",
                        lambda *_: (port_pos, FIELD_PORT_QUAT, 4.35))
    monkeypatch.setattr(sc_controller, "ScInsertionController", _CapturingController)

    assert sc_controller.run_sc_insertion(
        _RunPolicy(tcp_pos, tcp_quat), _StubTask(),
        lambda: object(), None, lambda *_: None,
    )

    assert np.allclose(seen["Rp"], Rp), "perception frame must still be passed through"
    assert not np.allclose(seen["Rs"], seen["Rp"]), "a seat frame must actually be built"
    assert np.allclose(seen["Rs"], sc_controller.seat_frame(Rp, R_tip))


# ---------------------------------------------------------------------------
# Handoff depth sign guard.
# ---------------------------------------------------------------------------
def _run_sc_with_handoff_depth(monkeypatch, depth_m, log):
    """Drive run_sc_insertion with the tip placed at a chosen depth."""
    Rp, R_tip = _field_frames()
    port_pos = np.array([-0.32466, 0.12952, 0.03032], dtype=np.float64)
    R_tcp = R_tip @ quat_to_matrix(sc_controller.SC_TIP_IN_TCP_QUAT).T
    tcp_quat = matrix_to_quat(R_tcp)
    # want (Rp.T @ (tip - port))[2] == depth_m, with a little lateral offset
    want_tip = port_pos + Rp[:, 2] * depth_m + Rp[:, 0] * 0.0035
    tip_at_origin, _ = sc_tip_pose_from_tcp(np.zeros(3), tcp_quat)
    tcp_pos = want_tip - tip_at_origin

    class _RunPolicy(_StubPolicy):
        _wrench_baseline = None

        def get_logger(self):
            return log

        def _wrench_vector(self, _obs):
            return np.zeros(6, dtype=np.float64)

    ran = {"seated": False}

    class _NeverReached:
        def __init__(self, *_a, **_k):
            pass

        def run(self):
            ran["seated"] = True
            return True

    monkeypatch.setattr(sc_controller, "perceive_sc_port_pose_consensus",
                        lambda *_: (port_pos, FIELD_PORT_QUAT, 4.35))
    monkeypatch.setattr(sc_controller, "ScInsertionController", _NeverReached)
    result = sc_controller.run_sc_insertion(
        _RunPolicy(tcp_pos, tcp_quat), _StubTask(),
        lambda: object(), None, lambda *_: None,
    )
    return result, ran["seated"]


def test_handoff_refuses_a_tip_computed_inside_the_port(monkeypatch):
    """The 2026-07-25 failure: +6.99 mm before any motion, then a fake seat.

    A positive handoff depth is physically impossible -- the plug is outside the
    port until it is pushed in.  Left unchecked it exceeds seat_candidate_depth_m,
    so _seat skips the whole approach and waits for an event that cannot arrive,
    which RL_INSERT_REPORT_MISS_AS_SUCCESS reports as success.
    """
    log = _RecordingLog()

    result, seated = _run_sc_with_handoff_depth(monkeypatch, +0.00699, log)

    assert result is False
    assert not seated, "must refuse before constructing the controller"
    line = "\n".join(log.error_lines)
    assert "INSIDE the port" in line
    assert "+6.99mm" in line, "the impossible measurement must be in the log"
    assert "SC_TIP_IN_TCP_POS" in line, "must name the cause, not just the symptom"


def test_handoff_accepts_a_plug_sitting_outside_the_mouth(monkeypatch):
    # Where the plug actually is at handoff: outside, approaching along -insert.
    result, seated = _run_sc_with_handoff_depth(monkeypatch, -0.00699, _RecordingLog())

    assert result is True
    assert seated


def test_handoff_tolerates_a_plug_right_at_the_mouth(monkeypatch):
    # Zero is the physical truth and perception noise straddles it; the gate is
    # tolerance for that, not a licence to start partly inserted.
    result, seated = _run_sc_with_handoff_depth(monkeypatch, +0.0005, _RecordingLog())

    assert result is True
    assert seated


def test_handoff_depth_gate_is_tighter_than_the_seat_trigger(monkeypatch):
    # The gate is only useful if it fires well before a depth that would let
    # _seat skip its approach entirely.
    assert sc_controller.SC_MAX_HANDOFF_DEPTH_M < SCConfig().validated().seat_candidate_depth_m


# ---------------------------------------------------------------------------
# SC grasp-calibration dump (6c).
# ---------------------------------------------------------------------------
class _FakeTransform:
    def __init__(self, pos, quat_wxyz):
        class _V:
            pass

        class _Q:
            pass

        self.transform = _V()
        self.transform.translation = _V()
        self.transform.rotation = _Q()
        (self.transform.translation.x, self.transform.translation.y,
         self.transform.translation.z) = [float(v) for v in pos]
        (self.transform.rotation.w, self.transform.rotation.x,
         self.transform.rotation.y, self.transform.rotation.z) = [
            float(v) for v in quat_wxyz]


def test_calibration_dump_solves_the_sc_transform_from_a_ground_truth_frame():
    log = _RecordingLog()
    tcp_pos = np.array([-0.31, 0.39, 0.24])
    tcp_quat = np.array([0.965925826, 0.0, 0.258819045, 0.0])
    # A "true" tip 40 mm out along the TCP's own z, rotated 90 deg about it --
    # i.e. exactly the convention offset the field logs show.
    R_tcp = quat_to_matrix(tcp_quat)
    true_pos = tcp_pos + R_tcp @ np.array([0.0, 0.0, 0.040])
    R_true = R_tcp @ rotation_from_axis_angle(np.array([0.0, 0.0, np.radians(-90.0)]))

    class _CalibPolicy(_StubPolicy):
        def get_logger(self):
            return log

        def _lookup_transform(self, target, frame, timeout_sec=0.3):
            if frame != "cable_0/sc_tip_link":
                raise RuntimeError("no such frame")
            return _FakeTransform(true_pos, matrix_to_quat(R_true))

    found = sc_controller.dump_sc_grasp_calibration(
        _CalibPolicy(tcp_pos, tcp_quat), _StubTask())

    assert found
    text = "\n".join(log.info_lines)
    assert "SOLVED RL_INSERT_SC_TIP_IN_TCP_POS" in text
    assert "SOLVED RL_INSERT_SC_TIP_IN_TCP_QUAT" in text
    # the solved position must be the true offset expressed in the TCP frame
    solved = [ln for ln in log.info_lines if "SOLVED RL_INSERT_SC_TIP_IN_TCP_POS" in ln][0]
    assert "0.04" in solved


def test_calibration_probes_the_frame_naming_the_sim_actually_publishes():
    """The names must carry the cable_N/ prefix AND the _link suffix.

    A bare "sc_tip" resolves nothing -- the SC Plug model.sdf declares
    sc_tip_link, merged into the cable model, so the published frame is
    cable_0/sc_tip_link. This mirrors DataCollectorScPlugPoseGT, which is the
    naming known to work.
    """
    class _Task:
        cable_name = "cable_0"
        plug_name = "sc_tip"

    frames = sc_controller.sc_calib_frame_candidates(_Task())

    assert frames[0] == "cable_0/sc_tip_link", "task-derived name must come first"
    assert "sc_tip_link" in frames, "bare-link fallback"
    assert not any(f.startswith("sfp") for f in frames), "must not inherit SFP names"
    assert all(f.endswith("_link") for f in frames), (
        "every candidate needs the _link suffix the sim publishes"
    )


def test_calibration_frame_candidates_survive_a_missing_task_field():
    class _Bare:
        pass

    frames = sc_controller.sc_calib_frame_candidates(_Bare())

    assert frames, "must still fall back to the static list"
    assert "sc_tip_link" in frames


def test_calibration_dump_probes_the_task_frame_first():
    seen = []

    class _ProbePolicy(_StubPolicy):
        def get_logger(self):
            return _RecordingLog()

        def _lookup_transform(self, target, frame, timeout_sec=0.3):
            seen.append(frame)
            raise RuntimeError("nothing resolves")

    class _Task:
        cable_name = "cable_3"
        plug_name = "sc_tip"
        target_module_name = "sc_port_0"
        port_name = "sc_port_base"

    assert not sc_controller.dump_sc_grasp_calibration(
        _ProbePolicy(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])), _Task())
    assert seen[0] == "cable_3/sc_tip_link", "the task's own cable must be tried first"


def test_calibration_dump_runs_even_when_the_depth_gate_refuses(monkeypatch):
    """The chicken-and-egg guard.

    The depth gate refuses every run until SC_TIP_IN_TCP is calibrated, and this
    dump is how it gets calibrated. If the dump ever moves below the gate there
    is no way in at all.
    """
    monkeypatch.setenv("RL_INSERT_CALIB_DUMP", "1")
    log = _RecordingLog()
    dumped = {"n": 0}
    monkeypatch.setattr(sc_controller, "dump_sc_grasp_calibration",
                        lambda *_a, **_k: dumped.__setitem__("n", dumped["n"] + 1) or True)

    result, seated = _run_sc_with_handoff_depth(monkeypatch, +0.00699, log)

    assert result is False, "the impossible depth must still be refused"
    assert not seated
    assert dumped["n"] == 1, "but the calibration sample must have been taken first"


def test_calibration_dump_runs_even_when_perception_fails(monkeypatch):
    """2026-07-25 18:46: perception found nothing and the run produced no
    [sc-calib] block at all, so a whole grasp was wasted.

    The dump depends on the TCP and a TF frame -- not on the port pose, the
    cameras, or anything perception returns. It must not sit behind perception.
    """
    monkeypatch.setenv("RL_INSERT_CALIB_DUMP", "1")
    dumped = {"n": 0}
    monkeypatch.setattr(sc_controller, "dump_sc_grasp_calibration",
                        lambda *_a, **_k: dumped.__setitem__("n", dumped["n"] + 1) or True)
    monkeypatch.setattr(sc_controller, "perceive_sc_port_pose_consensus",
                        lambda *_: None)

    class _P(_StubPolicy):
        def get_logger(self):
            return _RecordingLog()

    result = sc_controller.run_sc_insertion(
        _P(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])), _StubTask(),
        lambda: object(), None, lambda *_: None,
    )

    assert result is False, "perception failure must still fail the run"
    assert dumped["n"] == 1, "but the grasp sample must have been taken anyway"
