from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))

import aic_model.sc_controller as sc_controller  # noqa: E402
from aic_model.sc_controller import (  # noqa: E402
    SC_TIP_IN_TCP_POS,
    SCConfig,
    SC_BORE_PITCH_M,
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


def test_opening_classification_distinguishes_duplex_from_single_bore():
    label, residual, offset = classify_opening(SC_OPENING_WIDTH_M, SC_OPENING_HEIGHT_M)
    assert label == "duplex"
    assert residual < 1e-9
    assert offset == 0.0

    label, _, offset = classify_opening(0.0097, SC_OPENING_HEIGHT_M)
    assert label == "single_bore"
    # A single-bore detection is half a duplex pitch off the point the duplex
    # plug actually enters.
    assert np.isclose(offset, SC_BORE_PITCH_M / 2.0)


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

    def info(self, message):
        self.info_lines.append(str(message))

    def warn(self, message):
        self.warn_lines.append(str(message))


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
