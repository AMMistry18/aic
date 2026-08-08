from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))

from aic_model.v50_controller import (  # noqa: E402
    HARD_FAILURE,
    INSERT_DEPTH_M,
    PlugRelativeV50Controller,
    SEATED,
    STALLED,
    V50Config,
    WEDGED,
    WallProgressWatch,
    _normalize_event,
    deep_wedge_recovery_probes,
    next_persistent_depth,
    next_retract_depth,
    prime_v50_plug_pose,
    shift_wrench_moment_to_point,
    solve_tip_in_tcp,
    tcp_for_tip_transform,
    tip_from_tcp_transform,
    transverse_wrench_score,
)
import aic_model.v50_controller as v50_controller_module  # noqa: E402
from aic_model.rl_insert_contract import port_frame  # noqa: E402


def test_v50_config_bounds_force_and_seating():
    config = V50Config().validated()
    assert np.isclose(config.force_lead_m, 0.020)
    assert config.target_axial_force_n < config.seat_force_cap_n < 18.0
    assert 0.0 <= config.seat_overtravel_m <= 0.008
    assert np.isclose(config.seat_align_force_gain, 0.0001)
    assert np.isclose(config.seat_align_moment_gain, 0.01)
    assert np.isclose(config.seat_align_max_lat_m, 0.0007)
    assert np.isclose(config.seat_align_max_tilt_rad, 0.0122)
    assert np.isclose(config.seat_mouth_speed_scale, 0.5)
    assert np.isclose(config.seat_align_release_decay, 0.7)
    assert np.isclose(config.seat_stall_grace_s, 3.0)
    assert config.wedge_recovery_enable is True
    assert config.wedge_recovery_max_local_attempts == 2
    assert np.isclose(config.wedge_recovery_unload_m, 0.001)
    assert np.isclose(config.wedge_recovery_probe_max_lat_m, 0.0002)

    with pytest.raises(ValueError, match="hard abort"):
        V50Config(force_abort_n=19.0).validated()
    with pytest.raises(ValueError, match="overtravel"):
        V50Config(seat_overtravel_m=0.009).validated()
    with pytest.raises(ValueError, match="release decay"):
        V50Config(seat_align_release_decay=1.5).validated()
    # The correction may not exceed the port clearance it is steering inside of.
    with pytest.raises(ValueError, match="port clearance"):
        V50Config(seat_align_max_lat_m=0.0015).validated()
    with pytest.raises(ValueError, match="slew"):
        V50Config(seat_align_max_step_m=0.0).validated()
    with pytest.raises(ValueError, match="progress"):
        V50Config(
            wedge_recovery_probe_depth_m=0.0002,
            wedge_recovery_min_progress_m=0.0003,
        ).validated()
    with pytest.raises(ValueError, match="envelope"):
        V50Config(wedge_recovery_max_lat_m=0.0011).validated()


def test_tip_wrench_reference_shift_removes_a_pure_lever_arm_moment():
    # A lateral force applied through the plug tip produces a torque at the
    # sensor, but no residual torque when referenced back to that contact tip.
    sensor_to_tip = np.array([0.0, 0.0, 0.22])
    force = np.array([5.0, -2.0, 0.0])
    sensor_moment = np.cross(sensor_to_tip, force)

    tip_moment = shift_wrench_moment_to_point(
        force, sensor_moment, sensor_to_tip
    )

    np.testing.assert_allclose(tip_moment, np.zeros(3), atol=1e-12)


def test_controller_wrench_uses_the_documented_sensor_frame_and_tip_lever_arm():
    sensor_to_tip = np.array([0.0, 0.0, 0.22])
    tcp_to_tip = sensor_to_tip - np.array([0.0, 0.0, 0.172])
    force = np.array([5.0, -2.0, 0.0])
    controller = object.__new__(PlugRelativeV50Controller)
    controller.Rp = np.eye(3)
    controller.policy = SimpleNamespace(
        _wrench_vector=lambda _obs: np.concatenate(
            [force, np.cross(sensor_to_tip, force)]
        ),
        _wrench_baseline=np.zeros(6),
        _tcp=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        _tip_from_tcp=lambda _tcp_pos, _tcp_quat: (tcp_to_tip, np.eye(3)),
    )
    observation = SimpleNamespace(
        wrist_wrench=SimpleNamespace(
            header=SimpleNamespace(frame_id="ati/tool_link")
        )
    )

    force_port, moment_port = controller._wrench_plug_frame(observation)

    np.testing.assert_allclose(force_port, force)
    np.testing.assert_allclose(moment_port, np.zeros(3), atol=1e-12)


def test_deep_wedge_probes_are_bounded_and_include_a_measured_fallback_sign():
    config = V50Config().validated()
    probes = deep_wedge_recovery_probes(
        np.array([4.0, -3.0, -7.0]),
        np.array([0.4, -0.25, 0.0]),
        config,
    )

    assert len(probes) == 2
    primary_lat, primary_tilt = probes[0]
    mirrored_lat, mirrored_tilt = probes[1]
    assert np.linalg.norm(primary_lat) <= config.wedge_recovery_probe_max_lat_m
    assert np.linalg.norm(primary_tilt) <= config.wedge_recovery_probe_max_tilt_rad
    np.testing.assert_allclose(mirrored_lat, -primary_lat)
    np.testing.assert_allclose(mirrored_tilt, -primary_tilt)
    # First probe keeps the established force/moment correction convention.
    assert np.dot(primary_lat, np.array([4.0, -3.0])) < 0.0
    assert np.dot(primary_tilt, np.array([0.4, -0.25])) < 0.0


def test_deep_wedge_force_only_probe_and_score_fall_back_safely():
    config = V50Config().validated()
    force = np.array([3.0, 0.0, -7.0])
    probes = deep_wedge_recovery_probes(force, np.full(3, np.nan), config)

    assert len(probes) == 2
    assert np.allclose(probes[0][1], np.zeros(2))
    assert np.isfinite(transverse_wrench_score(force, np.full(3, np.nan), config))
    # Axial-only force is not evidence that side-wall correction is safe.
    assert deep_wedge_recovery_probes(
        np.array([0.0, 0.0, -9.0]), np.full(3, np.nan), config
    ) == ()


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
    # setpoint until it reaches the bounded 10N / 500N/m = 20mm lead.
    assert np.isclose(command_depth - current_depth, 0.020)
    held = next_persistent_depth(
        current_depth, command_depth, 1.0, force_n=12.0, config=config
    )
    assert np.isclose(held, command_depth)


def test_persistent_depth_freezes_on_axial_force_not_lateral_bind():
    # Diag-5's 36 mm stalls: ~6.3 N axial with 4.7 N of lateral bind, an 7.9 N
    # norm.  Gating the freeze on the norm stops the advance on scrape the plug
    # could still push through, so the axial component decides.
    config = V50Config().validated()
    current_depth = 0.036
    binding_norm = float(np.linalg.norm([4.7, 0.0, 6.3]))
    assert binding_norm > config.contact_force_n

    advanced = next_persistent_depth(
        current_depth, current_depth, 0.1, binding_norm, config, axial_force_n=-6.3
    )
    assert advanced > current_depth

    # Axial force at the cap still freezes it, sign-independently.
    frozen = next_persistent_depth(
        current_depth, current_depth, 0.1, binding_norm, config, axial_force_n=-12.5
    )
    assert np.isclose(frozen, current_depth)

    # With no axial reading supplied the norm keeps its old meaning.
    legacy = next_persistent_depth(
        current_depth, current_depth, 0.1, 12.5, config
    )
    assert np.isclose(legacy, current_depth)


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


def test_retract_depth_accumulates_pull_lead_on_a_stuck_plug():
    """The retry-killer: commanding current-1.5mm afresh each tick caps the
    pull at stiffness * 1.5mm = 0.75 N forever, which never unseats a wedge
    held by diag-5's 4-5 N lateral bind.  The persistent setpoint must keep
    receding until the lead cap, and no further."""
    config = V50Config().validated()
    current_depth = 0.036  # diag-5's wedge depth; the plug is not moving
    command_depth = current_depth

    offsets = []
    for _ in range(40):
        command_depth = next_retract_depth(current_depth, command_depth, config)
        offsets.append(current_depth - command_depth)

    # Monotone build-up, no oscillation, saturating exactly at the lead.
    assert all(b >= a - 1e-12 for a, b in zip(offsets, offsets[1:]))
    assert np.isclose(offsets[-1], config.retract_pull_lead_m)
    # At the documented 500 N/m that is ~12 N of pull, under the 18 N abort.
    nominal_pull = config.axial_stiffness_n_m * offsets[-1]
    assert 10.0 <= nominal_pull < config.force_abort_n


def test_retract_depth_walks_at_step_rate_when_the_plug_follows():
    # Free withdrawal must be unchanged from the old per-tick behaviour: when
    # the plug tracks the setpoint the offset never exceeds one step.
    config = V50Config().validated()
    current_depth = 0.020
    command_depth = current_depth

    for _ in range(10):
        command_depth = next_retract_depth(current_depth, command_depth, config)
        assert np.isclose(current_depth - command_depth, config.retract_step_m)
        current_depth = command_depth  # impedance catches up before next tick


def test_retract_depth_never_commands_back_into_the_bore_after_pop_free():
    # A wedge that lets go all at once leaves the plug shallower than the
    # setpoint; reusing the stale command would push it back IN.
    config = V50Config().validated()
    command_depth = next_retract_depth(0.036, 0.036, config)
    for _ in range(15):
        command_depth = next_retract_depth(0.036, command_depth, config)
    popped_depth = 0.002  # sprang most of the way out of the bore

    command_depth = next_retract_depth(popped_depth, command_depth, config)

    assert command_depth <= popped_depth


def test_retract_depth_builds_full_pull_even_when_stuck_near_the_mouth():
    # A chamfer catch just inside the mouth is still a stick; flooring the
    # setpoint at retract_clear_depth_m would cap the pull at
    # stiffness * (depth - clear) ~ 3.5 N here, resurrecting the plateau bug
    # for exactly the shallow catches the retry most often faces.
    config = V50Config().validated()
    stuck_depth = 0.004
    command_depth = stuck_depth
    for _ in range(50):
        command_depth = next_retract_depth(stuck_depth, command_depth, config)

    assert np.isclose(stuck_depth - command_depth, config.retract_pull_lead_m)
    assert command_depth < config.retract_clear_depth_m, \
        "the setpoint may recede past the clear point; actual clearance ends the loop"


def test_retract_pull_lead_must_stay_under_the_hard_abort():
    with pytest.raises(ValueError, match="hard force abort"):
        V50Config(retract_pull_lead_m=0.040).validated()  # 500 N/m * 40mm = 20 N
    with pytest.raises(ValueError, match="positive"):
        V50Config(retract_pull_lead_m=0.0).validated()


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
        min_keypoint_confidence = 0.5

        def detect_views(self, views):
            return [
                SimpleNamespace(
                    camera_name=view.camera_name,
                    box_confidence=0.9,
                    keypoint_confidences=np.array([0.9, 0.9, 0.9, 0.9]),
                    keypoints_px=np.zeros((4, 2)),
                )
                for view in views
            ]

        def estimate_multiview(self, views, *, now_s, max_age_s, detections):
            assert len(views) == 2
            assert np.isclose(now_s, 1.05)
            assert np.isclose(max_age_s, 0.35)
            assert len(detections) == 2
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
    def __init__(self):
        self.infos = []
        self.warns = []
        self.errors = []

    def info(self, _message):
        self.infos.append(str(_message))

    def warn(self, _message):
        self.warns.append(str(_message))

    def error(self, _message):
        self.errors.append(str(_message))


def test_any_fresh_insertion_event_is_success_even_for_an_alternate_port():
    controller = object.__new__(PlugRelativeV50Controller)
    node = SimpleNamespace(
        _insertion_event_generation=1,
        _insertion_event_value="cable_2#0#nic_card_mount_0/sfp_port_1",
    )
    controller.policy = SimpleNamespace(_parent_node=node)
    controller.event_generation = 0
    controller.expected_event = "nic_card_mount_0/sfp_port_0"
    controller.log = _Log()

    assert controller._event_status() == SEATED
    assert controller.log.errors == []
    assert any(
        "accepting insertion event for alternate port" in message
        for message in controller.log.warns
    )


class _SequenceHarness(PlugRelativeV50Controller):
    def __init__(
        self,
        outcomes,
        *,
        initial_pose=True,
        visual=True,
        plug_refresh=True,
        port_refresh=True,
        rescue=False,
        retract=True,
        config=None,
    ):
        self.config = (config or V50Config()).validated()
        self.outcomes = list(outcomes)
        self.initial_pose = initial_pose
        self.visual_result = visual
        self.plug_refresh_result = plug_refresh
        self.port_refresh_result = port_refresh
        self.rescue_result = rescue
        self.retract_result = retract
        self.trace = []
        self.log = _Log()
        self.expected_event = "nic_card_mount_0/sfp_port_0"
        self.send_feedback = lambda message: self.trace.append(("feedback", message))
        self.port_pos = np.zeros(3, dtype=np.float64)
        self.Rp = np.eye(3, dtype=np.float64)
        self.move_robot = None
        # The retry loop is bounded by the action deadline, so the double has to
        # offer one; counting the calls proves no cycle can skip it.
        self.deadline_checks = 0

        def _deadline(_move_robot):
            self.deadline_checks += 1

        self.policy = SimpleNamespace(_enforce_action_deadline=_deadline)

    def _activate_initial_plug_pose(self):
        self.trace.append("fresh-plug")
        return self.initial_pose

    def _hold_legacy_safe_pose(self):
        self.trace.append("safe-hold")

    def _tip_pose(self):
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    def _align(self):
        self.trace.append("align")
        return True

    def _seat(self):
        self.trace.append("seat")
        return self.outcomes.pop(0)

    def _attempt_wedge_rescue(self):
        self.trace.append("rescue")
        return self.rescue_result

    def _retract_to_start(self, _start_tip_pos, _start_tip_rotation):
        self.trace.append("retract")
        return self.retract_result

    def _refresh_plug_pose_after_retract(self):
        self.trace.append("re-perceive-plug")
        return self.plug_refresh_result

    def _refresh_port_pose_after_retract(self):
        self.trace.append("re-perceive-port")
        return self.port_refresh_result

    def _visual_rescue(self):
        self.trace.append("visual")
        return self.visual_result

    def _lift_and_refresh(self):
        self.trace.append("lift-fresh")
        return self.plug_refresh_result


def test_port_refresh_atomically_replaces_pose_using_newer_observations():
    controller = object.__new__(PlugRelativeV50Controller)
    controller.task = object()
    controller.log = _Log()
    controller.last_observation_stamp = 12.0
    controller.port_pos = np.array([9.0, 9.0, 9.0])
    controller.port_quat = np.array([1.0, 0.0, 0.0, 0.0])
    controller.Rp = np.eye(3)
    controller._port_pos_initial = controller.port_pos.copy()
    fresh_observation = object()
    waited_after = []

    def wait_new(*, after_stamp, timeout_wall_s):
        waited_after.append((after_stamp, timeout_wall_s))
        controller.last_observation_stamp = 13.0
        return fresh_observation

    def perceive(_task, get_observation):
        assert get_observation() is fresh_observation
        return np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, 0.0, 0.0]), 1.25

    controller._wait_new_observation = wait_new
    controller.policy = SimpleNamespace(perceive_port_pose_consensus=perceive)

    assert controller._refresh_port_pose_after_retract() is True
    np.testing.assert_allclose(controller.port_pos, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(controller.port_quat, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(controller.Rp, port_frame(controller.port_quat))
    np.testing.assert_allclose(controller._port_pos_initial, controller.port_pos)
    assert waited_after == [(12.0, 2.0)]


@pytest.mark.parametrize(
    "result",
    [
        None,
        (np.array([np.nan, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]), 1.0),
        (np.array([1.0, 2.0, 3.0]), np.zeros(4), 1.0),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]), 30.0),
    ],
)
def test_invalid_port_refresh_never_reuses_or_partially_updates_pose(result):
    controller = object.__new__(PlugRelativeV50Controller)
    controller.task = object()
    controller.log = _Log()
    controller.last_observation_stamp = 12.0
    old_pos = np.array([9.0, 8.0, 7.0])
    old_quat = np.array([1.0, 0.0, 0.0, 0.0])
    old_Rp = np.diag([1.0, -1.0, -1.0])
    controller.port_pos = old_pos.copy()
    controller.port_quat = old_quat.copy()
    controller.Rp = old_Rp.copy()
    controller._port_pos_initial = old_pos.copy()
    controller._wait_new_observation = lambda **_kwargs: object()
    controller.policy = SimpleNamespace(
        perceive_port_pose_consensus=lambda _task, _get_observation: result
    )

    assert controller._refresh_port_pose_after_retract() is False
    np.testing.assert_allclose(controller.port_pos, old_pos)
    np.testing.assert_allclose(controller.port_quat, old_quat)
    np.testing.assert_allclose(controller.Rp, old_Rp)
    np.testing.assert_allclose(controller._port_pos_initial, old_pos)


def test_failed_plug_refresh_clears_stale_grasp_transform():
    controller = object.__new__(PlugRelativeV50Controller)
    controller.log = _Log()
    controller.last_observation_stamp = 12.0
    controller.policy = SimpleNamespace(_v50_grasp_transform=("stale", "pose"))
    controller._wait_new_observation = lambda **_kwargs: None

    assert controller._refresh_plug_pose_after_retract() is False
    assert controller.policy._v50_grasp_transform is None


class _AlignHarness(PlugRelativeV50Controller):
    def __init__(self, f_plug, m_plug, config=None):
        self.config = (config or V50Config()).validated()
        self._f = np.asarray(f_plug, dtype=np.float64)
        self._m = np.asarray(m_plug, dtype=np.float64)

    def _wrench_plug_frame(self, observation):
        return self._f, self._m


def test_verified_recovery_and_live_wrench_correction_share_one_clearance_cap():
    harness = object.__new__(PlugRelativeV50Controller)
    harness.config = V50Config().validated()
    lat, tilt = harness._combined_seat_correction(
        np.array([0.0007, 0.0]),
        np.array([0.0122, 0.0]),
        np.array([0.0002, 0.0]),
        np.array([0.003, 0.0]),
    )

    assert np.linalg.norm(lat) <= harness.config.seat_align_max_lat_m + 1e-12
    assert np.linalg.norm(tilt) <= harness.config.seat_align_max_tilt_rad + 1e-12


def test_force_guided_deep_recovery_accepts_only_a_measured_improvement(monkeypatch):
    config = V50Config(
        wedge_recovery_probe_speed_m_s=0.01,
        wedge_recovery_probe_timeout_wall_s=1.0,
    ).validated()
    clock = [0.0]
    state = SimpleNamespace(depth=0.036)
    controller = object.__new__(PlugRelativeV50Controller)
    controller.config = config
    controller.Rp = np.eye(3)
    controller.port_pos = np.zeros(3)
    controller.log = _Log()
    controller.move_robot = None
    controller.get_observation = lambda: object()
    controller._event_status = lambda: None
    controller._force_magnitude = lambda _observation: 6.0
    controller._wrench_plug_frame = lambda _observation: (
        np.array([1.0, 0.0, -6.0]),
        np.zeros(3),
    )
    controller._errors = lambda: (
        state.depth,
        np.zeros(2),
        np.zeros(3),
        np.array([0.0, 0.0, state.depth]),
        np.eye(3),
    )
    controller._unload_deep_wedge = lambda _depth: (
        "deep_recovery_accepted",
        (state.depth, np.array([0.0, 0.0, state.depth]), np.eye(3)),
    )
    controller.policy = SimpleNamespace(
        _enforce_action_deadline=lambda _move: None,
        _tcp_target_for_tip=lambda tip, rotation: (tip, rotation),
        set_pose_target=lambda _move, target, **_kwargs: setattr(
            state, "depth", float(target[0][2])
        ),
        sleep_for=lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(v50_controller_module.time, "monotonic", lambda: clock[0])

    outcome, recovery_lat, recovery_tilt, recovered_depth = (
        controller._attempt_force_guided_deep_recovery(
            depth=0.037,
            force=7.0,
            force_port=np.array([4.0, 0.0, -6.0]),
            moment_port=np.zeros(3),
            current_lat=np.zeros(2),
            current_tilt=np.zeros(2),
        )
    )

    assert outcome == "deep_recovery_accepted"
    assert recovered_depth >= 0.037 + config.wedge_recovery_min_progress_m
    assert recovery_lat[0] < 0.0
    np.testing.assert_allclose(recovery_tilt, np.zeros(2))


def test_force_guided_deep_recovery_measures_the_mirrored_probe_too(monkeypatch):
    config = V50Config(
        wedge_recovery_probe_speed_m_s=0.01,
        wedge_recovery_probe_timeout_wall_s=0.35,
    ).validated()
    clock = [0.0]
    state = SimpleNamespace(depth=0.036, lateral_x=0.0)
    controller = object.__new__(PlugRelativeV50Controller)
    controller.config = config
    controller.Rp = np.eye(3)
    controller.port_pos = np.zeros(3)
    controller.log = _Log()
    controller.move_robot = None
    controller.get_observation = lambda: object()
    controller._event_status = lambda: None
    controller._force_magnitude = lambda _observation: 6.0
    controller._wrench_plug_frame = lambda _observation: (
        np.array(
            [1.0 if state.lateral_x > 0.0 else 4.0, 0.0, -6.0]
        ),
        np.zeros(3),
    )
    controller._errors = lambda: (
        state.depth,
        np.zeros(2),
        np.zeros(3),
        np.array([state.lateral_x, 0.0, state.depth]),
        np.eye(3),
    )

    def unload(_depth):
        state.depth = 0.036
        return (
            "deep_recovery_accepted",
            (state.depth, np.array([state.lateral_x, 0.0, state.depth]), np.eye(3)),
        )

    controller._unload_deep_wedge = unload

    def set_pose(_move, target, **_kwargs):
        state.lateral_x = float(target[0][0])
        state.depth = float(target[0][2])

    controller.policy = SimpleNamespace(
        _enforce_action_deadline=lambda _move: None,
        _tcp_target_for_tip=lambda tip, rotation: (tip, rotation),
        set_pose_target=set_pose,
        sleep_for=lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(v50_controller_module.time, "monotonic", lambda: clock[0])

    outcome, recovery_lat, _recovery_tilt, _recovered_depth = (
        controller._attempt_force_guided_deep_recovery(
            depth=0.037,
            force=7.0,
            force_port=np.array([4.0, 0.0, -6.0]),
            moment_port=np.zeros(3),
            current_lat=np.zeros(2),
            current_tilt=np.zeros(2),
        )
    )

    assert outcome == "deep_recovery_accepted"
    # The primary -F trial did not improve; the mirrored +F trial did.
    assert recovery_lat[0] > 0.0


def test_seat_target_pose_presses_past_the_seat_frame_during_event_dwell():
    # The detection pad's near face is about 1 mm past INSERT_DEPTH_M and the
    # TouchPlugin needs 1 s of unbroken contact, so holding the tip at exactly
    # INSERT_DEPTH_M commands a setpoint behind a plug that has reached the pad.
    config = V50Config().validated()
    harness = _AlignHarness(f_plug=[0.0, 0.0, -7.0], m_plug=[0.0, 0.0, 0.0], config=config)
    harness.port_pos = np.zeros(3, dtype=np.float64)
    harness.Rp = np.eye(3, dtype=np.float64)

    acc_lat = np.array([0.0003, -0.0002], dtype=np.float64)
    acc_tilt = np.array([0.001, 0.0], dtype=np.float64)
    dwell_depth = INSERT_DEPTH_M + config.seat_overtravel_m
    tip, rotation = harness._seat_target_pose(dwell_depth, acc_lat, acc_tilt)

    assert tip[2] > INSERT_DEPTH_M
    assert np.isclose(tip[2], dwell_depth)
    # The corrections that got the plug in are carried, not dropped at the door.
    assert np.isclose(tip[0], acc_lat[0])
    assert np.isclose(tip[1], acc_lat[1])
    assert not np.allclose(rotation, harness.Rp)


def test_seat_alignment_respects_the_per_sample_slew_limit():
    config = V50Config().validated()
    harness = _AlignHarness(
        f_plug=[60.0, -40.0, -12.0], m_plug=[0.0, -8.0, 0.0], config=config
    )
    acc_lat = np.zeros(2, dtype=np.float64)
    acc_tilt = np.zeros(2, dtype=np.float64)

    for _ in range(60):
        prev_lat, prev_tilt = acc_lat, acc_tilt
        acc_lat, acc_tilt, _sample = harness._seat_alignment_sample(
            None, 0.0, 12.0, acc_lat, acc_tilt
        )
        assert (
            np.linalg.norm(acc_lat - prev_lat)
            <= config.seat_align_max_step_m + 1e-12
        )
        assert (
            np.linalg.norm(acc_tilt - prev_tilt)
            <= config.seat_align_max_tilt_step_rad + 1e-12
        )


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
    # The margin is thinner than it was at the 3e-5 gain -- 4.4 N of bind now
    # earns ~0.44 mm of the 0.7 mm cap, which is the authority diag-5's 36 mm
    # stalls needed -- but settling below the cap is what must stay true.
    assert np.linalg.norm(acc_lat) < 0.8 * config.seat_align_max_lat_m
    assert np.linalg.norm(acc_tilt) < 0.8 * config.seat_align_max_tilt_rad


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
        assert np.linalg.norm(acc_lat) < 0.4 * config.seat_align_max_lat_m
        assert np.linalg.norm(acc_tilt) < 0.4 * config.seat_align_max_tilt_rad


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


def test_hard_failure_and_no_event_stall_never_retry():
    # A wrong-port event or sustained over-force is not fixed by another attempt,
    # and a plug already at seat depth must not be backed out of the port.
    for outcome in (HARD_FAILURE, STALLED):
        harness = _SequenceHarness([outcome])
        assert harness.run() is False
        assert "retract" not in harness.trace
        assert "rescue" not in harness.trace
        assert harness.trace.count("seat") == 1


def test_wedge_retries_until_it_seats_with_no_retry_ceiling():
    # Ten wedges then a seat: retries are unbounded by count, so the run must
    # keep backing out and trying rather than giving up at some fixed number.
    harness = _SequenceHarness([WEDGED] * 10 + [SEATED])
    assert harness.run() is True
    assert harness.trace.count("retract") == 10
    assert harness.trace.count("align") == 11
    assert harness.trace.count("seat") == 11
    # Every retract requires both fresh poses before any retry motion.
    assert harness.trace.count("re-perceive-plug") == 10
    assert harness.trace.count("re-perceive-port") == 10
    first_retract = harness.trace.index("retract")
    assert harness.trace[first_retract:first_retract + 4] == [
        "retract", "re-perceive-plug", "re-perceive-port", "align"
    ]
    # Unbounded by count means the deadline is the only thing that can stop it,
    # so every cycle must consult it.
    assert harness.deadline_checks == 11


def test_wedge_prefers_rescue_and_retracts_only_when_there_is_none():
    # Rescue first; the retract is the fallback for when no rescue is available.
    harness = _SequenceHarness([WEDGED, SEATED], rescue=True)
    assert harness.run() is True
    assert harness.trace.count("rescue") == 1
    assert "retract" not in harness.trace


def test_second_wedge_after_a_rescue_falls_through_to_retract():
    # One rescue per retract: a port estimate must not be nudged indefinitely
    # without ever backing the plug out.
    harness = _SequenceHarness([WEDGED, WEDGED, SEATED], rescue=True)
    assert harness.run() is True
    assert harness.trace.count("rescue") == 1
    assert harness.trace.count("retract") == 1


def test_failed_retract_ends_the_run_instead_of_looping():
    harness = _SequenceHarness([WEDGED] * 3, retract=False)
    assert harness.run() is False
    assert harness.trace.count("retract") == 1


@pytest.mark.parametrize(
    ("plug_refresh", "port_refresh", "expected_trace"),
    [
        (False, True, ["retract", "re-perceive-plug"]),
        (True, False, ["retract", "re-perceive-plug", "re-perceive-port"]),
    ],
)
def test_retry_aborts_instead_of_reusing_a_stale_pose(
    plug_refresh, port_refresh, expected_trace
):
    harness = _SequenceHarness(
        [WEDGED, SEATED],
        plug_refresh=plug_refresh,
        port_refresh=port_refresh,
    )

    assert harness.run() is False
    assert harness.trace.count("align") == 1
    assert harness.trace.count("seat") == 1
    start = harness.trace.index("retract")
    assert harness.trace[start:] == expected_trace


def test_wedge_retry_can_be_disabled_and_capped():
    disabled = _SequenceHarness(
        [WEDGED, SEATED], config=V50Config(wedge_retry_enable=False)
    )
    assert disabled.run() is False
    assert "retract" not in disabled.trace

    capped = _SequenceHarness(
        [WEDGED] * 5, config=V50Config(max_wedge_retries=2)
    )
    assert capped.run() is False
    assert capped.trace.count("retract") == 2


def test_v50_controller_overlay_matches_the_source_controller():
    source = REPO_ROOT / "aic_model" / "aic_model" / "v50_controller.py"
    overlay_source = (
        REPO_ROOT
        / "docker"
        / "aic_model"
        / "v50_overlay"
        / "aic_model"
        / "v50_controller.py"
    )
    assert overlay_source.read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_release_dockerfile_disables_bias_and_pins_safety():
    dockerfile = (REPO_ROOT / "docker" / "aic_model" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "RL_INSERT_SCRIPT_BIAS_X_M=0" in dockerfile
    assert "RL_INSERT_SCRIPT_BIAS_Y_M=0" in dockerfile
    assert "RL_INSERT_SCRIPT_BIAS_RX_RAD=0" in dockerfile
    assert "RL_INSERT_FORCE_ABORT_N=18.0" in dockerfile
    assert "best_sfp_plug_pose.pt" in dockerfile
    # The budget must leave room for retries but stay inside the engine's
    # per-task time_limit; a run that never returns scores nothing at all.
    # Asserted against the shipped time_limit rather than a literal, because the
    # two have to move together -- raising the budget alone just means the engine
    # cuts the action off mid-ladder.
    budget = re.search(r"RL_INSERT_ACTION_TIME_BUDGET_S=(\d+)", dockerfile)
    assert budget is not None
    time_limits = {
        int(match)
        for match in re.findall(
            r'"time_limit":\s*(\d+)',
            (REPO_ROOT / "generate_config.py").read_text(encoding="utf-8"),
        )
    }
    assert time_limits, "generate_config.py must pin a per-task time_limit"
    assert 60 <= int(budget.group(1)) <= min(time_limits) - 10

    # Baked ENV wins over source defaults, and Flowstate takes no runtime knobs,
    # so a seat-force retune in V50Config is a no-op in the image unless these
    # pins move with it. The release Dockerfile must agree with the source.
    config = V50Config().validated()
    assert f"RL_INSERT_V50_TARGET_FORCE_N={config.target_axial_force_n}" in dockerfile
    assert f"RL_INSERT_V50_SEAT_FORCE_CAP_N={config.seat_force_cap_n}" in dockerfile
