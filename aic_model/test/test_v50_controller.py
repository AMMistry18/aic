from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))
sys.path.insert(0, str(REPO_ROOT / "docker" / "aic_model"))

from aic_model.v50_controller import (  # noqa: E402
    HARD_FAILURE,
    INSERT_DEPTH_M,
    PlugRelativeV50Controller,
    SEATED,
    STALLED,
    V50Config,
    WallProgressWatch,
    _normalize_event,
    next_persistent_depth,
    prime_v50_plug_pose,
    solve_tip_in_tcp,
    tcp_for_tip_transform,
    tip_from_tcp_transform,
)
import patch_v49_plug_relative_v50 as overlay  # noqa: E402


def test_v50_config_bounds_force_and_seating():
    config = V50Config().validated()
    assert np.isclose(config.force_lead_m, 0.016)
    assert config.target_axial_force_n < config.seat_force_cap_n < 18.0
    assert 0.0 <= config.seat_overtravel_m <= 0.008
    assert np.isclose(config.seat_align_force_gain, 0.00003)
    assert np.isclose(config.seat_align_moment_gain, 0.004)
    assert np.isclose(config.seat_align_max_lat_m, 0.0004)
    assert np.isclose(config.seat_align_max_tilt_rad, 0.0087)
    assert np.isclose(config.seat_mouth_speed_scale, 0.25)
    assert np.isclose(config.seat_align_release_decay, 0.7)
    assert np.isclose(config.seat_stall_grace_s, 1.5)

    with pytest.raises(ValueError, match="hard abort"):
        V50Config(force_abort_n=19.0).validated()
    with pytest.raises(ValueError, match="overtravel"):
        V50Config(seat_overtravel_m=0.009).validated()
    with pytest.raises(ValueError, match="release decay"):
        V50Config(seat_align_release_decay=1.5).validated()


def test_event_normalization_strips_the_cable_instance_prefix():
    # Field logs: the scoring topic names the cable instance, the Task does not,
    # so the raw strings never compared equal even on a correct insertion.
    assert (
        _normalize_event("cable_0#0#nic_card_mount_0/sfp_port_0")
        == "nic_card_mount_0/sfp_port_0"
    )
    # The prefix numbers are not fixed -- later runs published cable_1#0#.
    assert (
        _normalize_event("cable_1#0#nic_card_mount_0/sfp_port_1")
        == "nic_card_mount_0/sfp_port_1"
    )
    assert (
        _normalize_event("cable_12#34#nic_card_mount_2/sfp_port_1")
        == "nic_card_mount_2/sfp_port_1"
    )


def test_event_normalization_leaves_ordinary_values_alone():
    # Already normalized.
    assert (
        _normalize_event("nic_card_mount_0/sfp_port_0") == "nic_card_mount_0/sfp_port_0"
    )
    # Whitespace and slash handling must survive the change.
    assert (
        _normalize_event("  /nic_card_mount_0/sfp_port_0/  ")
        == "nic_card_mount_0/sfp_port_0"
    )
    assert (
        _normalize_event("\t/cable_0#0#nic_card_mount_0/sfp_port_0/\n")
        == "nic_card_mount_0/sfp_port_0"
    )
    assert _normalize_event("") == ""
    assert _normalize_event(None) == ""


@pytest.mark.parametrize(
    "value",
    [
        # Only a leading, fully-numeric cable_<n>#<n># prefix is removed.
        "cable_a#0#nic_card_mount_0/sfp_port_0",
        "cable_0#nic_card_mount_0/sfp_port_0",
        "cable#0#nic_card_mount_0/sfp_port_0",
        "cable_0#0nic_card_mount_0/sfp_port_0",
        # Not at the start.
        "sfp_cable_0#0#nic_card_mount_0/sfp_port_0",
        # A hash elsewhere in the name must not be treated as a prefix.
        "nic_card_mount_0/sfp_port_0#0",
    ],
)
def test_event_normalization_does_not_touch_nonmatching_values(value):
    assert _normalize_event(value) == value


def test_persistent_depth_accumulates_force_lead_while_stalled():
    config = V50Config().validated()
    current_depth = 0.006
    command_depth = current_depth
    for _ in range(20):
        command_depth = next_persistent_depth(
            current_depth, command_depth, 0.1, force_n=1.0, config=config
        )

    # v49 repeatedly requested only current+~1mm.  v50 retains an absolute
    # setpoint until it reaches the bounded 8N / 500N/m = 16mm lead.
    assert np.isclose(command_depth - current_depth, 0.016)
    held = next_persistent_depth(
        current_depth, command_depth, 1.0, force_n=10.0, config=config
    )
    assert np.isclose(held, command_depth)


def test_persistent_depth_allows_bounded_overtravel_near_insert_depth():
    config = V50Config(seat_overtravel_m=0.003).validated()
    current_depth = INSERT_DEPTH_M - 0.001
    command_depth = current_depth

    for _ in range(50):
        command_depth = next_persistent_depth(
            current_depth, command_depth, 0.1, force_n=1.0, config=config
        )

    assert command_depth > INSERT_DEPTH_M
    assert np.isclose(command_depth, INSERT_DEPTH_M + config.seat_overtravel_m)


def test_wall_progress_watch_uses_elapsed_time_not_loop_count():
    watch = WallProgressWatch.start(0.010, now=100.0, config=V50Config())
    for _ in range(500):
        assert not watch.stalled(0.0102, 101.0)
    assert watch.stalled(0.0102, 102.51)
    assert not watch.stalled(0.0110, 102.52)


def test_visual_grasp_transform_round_trip():
    tcp_pos = np.array([-0.42, 0.18, 0.23])
    tcp_quat = np.array([0.965925826, 0.0, 0.258819045, 0.0])
    tip_pos = np.array([-0.40, 0.175, 0.18])
    angle = np.deg2rad(7.0)
    tip_rotation = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(angle), -np.sin(angle)],
         [0.0, np.sin(angle), np.cos(angle)]]
    )
    relative = solve_tip_in_tcp(tcp_pos, tcp_quat, tip_pos, tip_rotation)
    recovered_tip = tip_from_tcp_transform(tcp_pos, tcp_quat, *relative)
    np.testing.assert_allclose(recovered_tip[0], tip_pos, atol=1e-12)
    np.testing.assert_allclose(recovered_tip[1], tip_rotation, atol=1e-12)
    recovered_tcp = tcp_for_tip_transform(tip_pos, tip_rotation, *relative)
    np.testing.assert_allclose(recovered_tcp[0], tcp_pos, atol=1e-12)
    # Quaternion sign is irrelevant, so compare its reconstructed rotation.
    round_trip_tip = tip_from_tcp_transform(
        recovered_tcp[0], recovered_tcp[1], *relative
    )
    np.testing.assert_allclose(round_trip_tip[1], tip_rotation, atol=1e-12)


def test_priming_uses_direct_fresh_plug_pose_before_port_selection():
    stamp = SimpleNamespace(sec=1, nanosec=0)
    image_message = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
    observation = SimpleNamespace(
        left_image=image_message,
        center_image=image_message,
        right_image=image_message,
    )
    world_estimate = SimpleNamespace(
        position_world=np.array([0.1, -0.2, 0.3]),
        rotation_world_from_plug=np.eye(3),
        confidence=0.9,
        view_count=2,
        reprojection_error_px=0.5,
        source_frame_ids=("left:1", "center:1"),
        stamp_s=1.0,
    )

    class Estimator:
        def estimate_multiview(self, views, *, now_s, max_age_s):
            assert len(views) == 2
            assert np.isclose(now_s, 1.05)
            assert np.isclose(max_age_s, 0.35)
            return world_estimate

    parent = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_050_000_000)
        )
    )
    policy = SimpleNamespace(
        _v50_grasp_transform=(np.ones(3), np.eye(3)),
        _v50_pending_world_plug=None,
        _v50_plug_estimator=Estimator(),
        _v50_config=V50Config(),
        _parent_node=parent,
        _pc=SimpleNamespace(invert_transform=lambda transform: np.linalg.inv(transform)),
        _build_views=lambda _obs: {
            "left_camera": (np.zeros((8, 8, 3), dtype=np.uint8), np.eye(3), np.eye(4)),
            "center_camera": (np.zeros((8, 8, 3), dtype=np.uint8), np.eye(3), np.eye(4)),
        },
        _tcp=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        _enforce_action_deadline=lambda _move: None,
        get_logger=lambda: _Log(),
    )
    assert prime_v50_plug_pose(policy, lambda: observation, lambda *_args: None)
    assert policy._v50_pending_world_plug is world_estimate
    np.testing.assert_allclose(policy._v50_grasp_transform[0], world_estimate.position_world)


def test_priming_failure_clears_stale_grasp_and_fails_closed():
    stamp = SimpleNamespace(sec=1, nanosec=0)
    image_message = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
    observation = SimpleNamespace(
        left_image=image_message,
        center_image=image_message,
        right_image=image_message,
    )

    class Estimator:
        def estimate_multiview(self, *_args, **_kwargs):
            return None

    policy = SimpleNamespace(
        _v50_grasp_transform=(np.ones(3), np.eye(3)),
        _v50_pending_world_plug=object(),
        _v50_plug_estimator=Estimator(),
        _v50_config=V50Config(),
        _parent_node=SimpleNamespace(
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=1_050_000_000)
            )
        ),
        _pc=SimpleNamespace(invert_transform=lambda transform: np.linalg.inv(transform)),
        _build_views=lambda _obs: {
            "left_camera": (np.zeros((8, 8, 3), dtype=np.uint8), np.eye(3), np.eye(4)),
            "center_camera": (np.zeros((8, 8, 3), dtype=np.uint8), np.eye(3), np.eye(4)),
        },
        _enforce_action_deadline=lambda _move: None,
        get_logger=lambda: _Log(),
    )
    assert not prime_v50_plug_pose(policy, lambda: observation, lambda *_args: None)
    assert policy._v50_grasp_transform is None
    assert policy._v50_pending_world_plug is None


class _Log:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


class _SequenceHarness(PlugRelativeV50Controller):
    def __init__(self, outcomes, *, initial_pose=True, visual=True, refresh=True):
        self.config = V50Config().validated()
        self.outcomes = list(outcomes)
        self.initial_pose = initial_pose
        self.visual_result = visual
        self.refresh_result = refresh
        self.trace = []
        self.log = _Log()
        self.expected_event = "nic_card_mount_0/sfp_port_0"
        self.send_feedback = lambda message: self.trace.append(("feedback", message))

    def _activate_initial_plug_pose(self):
        self.trace.append("fresh-plug")
        return self.initial_pose

    def _hold_legacy_safe_pose(self):
        self.trace.append("safe-hold")

    def _align(self):
        self.trace.append("align")
        return True

    def _seat(self):
        self.trace.append("seat")
        return self.outcomes.pop(0)

    def _visual_rescue(self):
        self.trace.append("visual")
        return self.visual_result

    def _lift_and_refresh(self):
        self.trace.append("lift-fresh")
        return self.refresh_result


class _AlignHarness(PlugRelativeV50Controller):
    def __init__(self, f_plug, m_plug, config=None):
        self.config = (config or V50Config()).validated()
        self._f = np.asarray(f_plug, dtype=np.float64)
        self._m = np.asarray(m_plug, dtype=np.float64)

    def _wrench_plug_frame(self, observation):
        return self._f, self._m


def test_seat_alignment_settles_at_the_proportional_target_not_the_clamp():
    config = V50Config().validated()
    f_plug = [3.2, -3.0, -7.0]
    m_plug = [0.0, -0.65, 0.0]
    harness = _AlignHarness(f_plug=f_plug, m_plug=m_plug, config=config)
    acc_lat = np.zeros(2, dtype=np.float64)
    acc_tilt = np.zeros(2, dtype=np.float64)

    for _ in range(60):
        acc_lat, acc_tilt, _sample = harness._seat_alignment_sample(
            None, 0.0, 7.0, acc_lat, acc_tilt
        )

    expected_lat = config.seat_align_force_gain * np.linalg.norm(f_plug[:2])
    expected_tilt = config.seat_align_moment_gain * np.linalg.norm(m_plug[:2])
    assert np.isclose(np.linalg.norm(acc_lat), expected_lat)
    assert np.isclose(np.linalg.norm(acc_tilt), expected_tilt)
    # The whole point: sustained contact must not pin the correction at the clamp.
    assert np.linalg.norm(acc_lat) < 0.5 * config.seat_align_max_lat_m
    assert np.linalg.norm(acc_tilt) < 0.5 * config.seat_align_max_tilt_rad


def test_light_chamfer_contact_never_saturates_the_correction():
    # Field log 3 run 2: 1.7 N of chamfer contact pinned the old accumulator at the
    # clamp three samples in, and the plug stopped descending at 2.1 mm.
    config = V50Config().validated()
    harness = _AlignHarness(
        f_plug=[1.6, -0.6, -3.6],
        m_plug=[0.0, -0.18, 0.0],
        config=config,
    )
    acc_lat = np.zeros(2, dtype=np.float64)
    acc_tilt = np.zeros(2, dtype=np.float64)

    for _ in range(60):
        acc_lat, acc_tilt, _sample = harness._seat_alignment_sample(
            None, 0.0, 3.6, acc_lat, acc_tilt
        )
        assert np.linalg.norm(acc_lat) < 0.25 * config.seat_align_max_lat_m
        assert np.linalg.norm(acc_tilt) < 0.25 * config.seat_align_max_tilt_rad


def test_seat_alignment_still_clamps_under_extreme_wrench():
    config = V50Config().validated()
    harness = _AlignHarness(
        f_plug=[60.0, -40.0, -12.0],
        m_plug=[0.0, -8.0, 0.0],
        config=config,
    )
    acc_lat = np.zeros(2, dtype=np.float64)
    acc_tilt = np.zeros(2, dtype=np.float64)

    for _ in range(60):
        acc_lat, acc_tilt, _sample = harness._seat_alignment_sample(
            None, 0.0, 12.0, acc_lat, acc_tilt
        )
        assert np.linalg.norm(acc_lat) <= config.seat_align_max_lat_m + 1e-12
        assert np.linalg.norm(acc_tilt) <= config.seat_align_max_tilt_rad + 1e-12

    assert np.isclose(np.linalg.norm(acc_lat), config.seat_align_max_lat_m)
    assert np.isclose(np.linalg.norm(acc_tilt), config.seat_align_max_tilt_rad)


def test_seat_alignment_bias_washes_out_when_contact_is_lost():
    config = V50Config().validated()
    harness = _AlignHarness(
        f_plug=[3.2, -3.0, -7.0],
        m_plug=[0.0, -0.65, 0.0],
        config=config,
    )
    acc_lat = np.zeros(2, dtype=np.float64)
    acc_tilt = np.zeros(2, dtype=np.float64)

    for _ in range(30):
        acc_lat, acc_tilt, _sample = harness._seat_alignment_sample(
            None, 0.0, 7.0, acc_lat, acc_tilt
        )
    # Run-8 field regression: bias survived 30 mm of zero-force travel and
    # jammed the plug at 37.8 mm.
    for _ in range(20):
        acc_lat, acc_tilt, _sample = harness._seat_alignment_sample(
            None, 0.0, 0.5, acc_lat, acc_tilt
        )

    assert np.linalg.norm(acc_lat) < 0.02 * config.seat_align_max_lat_m
    assert np.linalg.norm(acc_tilt) < 0.02 * config.seat_align_max_tilt_rad


def test_stalled_seat_fails_after_one_align_and_seat_without_recovery():
    harness = _SequenceHarness([STALLED])
    assert harness.run() is False
    actions = [entry for entry in harness.trace if isinstance(entry, str)]
    assert actions == [
        "fresh-plug",
        "align",
        "seat",
    ]


def test_missing_plug_pose_fails_closed_before_any_motion():
    harness = _SequenceHarness([], initial_pose=False)
    assert harness.run() is False
    assert harness.trace == [
        ("feedback", "v50 fresh plug-to-port perception"),
        "fresh-plug",
        "safe-hold",
    ]


def test_hard_failure_never_invokes_visual_or_lift_recovery():
    harness = _SequenceHarness([HARD_FAILURE])
    assert harness.run() is False
    assert "visual" not in harness.trace
    assert "lift-fresh" not in harness.trace


def test_overlay_rewrites_v49_dispatch_and_truthful_result():
    rl_source = "\n".join(
        [
            overlay.RLIMPORT_OLD,
            overlay.DEADLINE_OLD,
            overlay.PERCEPTION_INIT_OLD,
            overlay.TIP_OLD,
            overlay.TCP_TARGET_OLD,
            overlay.PORT_PERCEPTION_OLD,
            overlay.SCRIPT_DISPATCH_OLD,
        ]
    )
    patched_rl = overlay.patch_rlinsert_source(rl_source)
    assert "return run_v50_script(" in patched_rl
    assert '"45.0"' in patched_rl
    assert "configure_v50(self)" in patched_rl
    assert "prime_v50_plug_pose(self" in patched_rl
    assert "return self._run_script(" not in patched_rl

    model_source = "\n".join(
        [
            overlay.AIC_IMPORT_OLD,
            overlay.EVENT_INIT_OLD,
            overlay.EVENT_SHUTDOWN_OLD,
            overlay.EVENT_CALLBACK_OLD,
            overlay.TRUTHFUL_RESULT_OLD,
        ]
    )
    patched_model = overlay.patch_aic_model_source(model_source)
    assert 'String, "/scoring/insertion_event"' in patched_model
    # An unconfirmed insertion must NOT abort the goal: Flowstate would terminate
    # the enclosing 5-insertion process on the first recoverable miss.
    assert "import os" in patched_model
    assert "RL_INSERT_REPORT_MISS_AS_SUCCESS" in patched_model
    assert "result.success = True" in patched_model
    assert "Cable insertion ended safely without confirmation" in patched_model
    # Strict reporting stays reachable via the env override.
    assert "goal_handle.abort()" in patched_model
    assert "Cable insertion failed: no correct-port event" in patched_model


def test_overlay_path_rejects_any_non_v49_input(tmp_path):
    candidate = tmp_path / "RLInsert.py"
    candidate.write_text("not v49", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing non-v49"):
        overlay.patch_path(
            candidate,
            overlay.EXPECTED_V49_RLINSERT_SHA256,
            overlay.patch_rlinsert_source,
            "RLInsert",
        )


def test_release_dockerfile_disables_bias_and_pins_safety():
    dockerfile = (
        REPO_ROOT / "docker" / "aic_model" / "Dockerfile.plug_relative_v50"
    ).read_text(encoding="utf-8")
    assert "RL_INSERT_SCRIPT_BIAS_X_M=0" in dockerfile
    assert "RL_INSERT_SCRIPT_BIAS_Y_M=0" in dockerfile
    assert "RL_INSERT_SCRIPT_BIAS_RX_RAD=0" in dockerfile
    assert "RL_INSERT_FORCE_ABORT_N=18.0" in dockerfile
    assert "RL_INSERT_ACTION_TIME_BUDGET_S=45" in dockerfile
    assert "best_sfp_plug_pose.pt" in dockerfile
