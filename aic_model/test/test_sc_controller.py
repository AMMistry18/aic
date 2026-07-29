from pathlib import Path
import importlib
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "aic_model"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
for _name in [n for n in sys.modules if n == "aic_model" or n.startswith("aic_model.")]:
    _module = sys.modules[_name]
    _file = getattr(_module, "__file__", None) or ""
    if not _file.startswith(str(SOURCE_ROOT)):
        del sys.modules[_name]

import aic_model.sc_controller as sc_controller  # noqa: E402
from aic_model.rl_insert_contract import (  # noqa: E402
    matrix_to_quat,
    port_frame,
    quat_to_matrix,
)
from aic_model.v50_controller import rotation_from_axis_angle  # noqa: E402
from aic_model.sc_visual_alignment import (  # noqa: E402
    ScBlueSideSignature,
    ScRecoveryEstimate,
)
from aic_model.v50_controller import HARD_FAILURE, SEATED, STALLED  # noqa: E402
from aic_model.sc_controller import (  # noqa: E402
    SC_TIP_IN_TCP_POS,
    SCConfig,
    SC_INSERT_DEPTH_M,
    SC_OPENING_HEIGHT_M,
    SC_OPENING_WIDTH_M,
    SC_POSE_CONVENTION,
    SC_POSE_HEIGHT_M,
    SC_POSE_LOCAL_KPS_M,
    SC_POSE_WIDTH_M,
    ScInsertionController,
    _normalize_event,
    _select_sc_detections_for_triangulation,
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
    assert config.align_lateral_tol_m == pytest.approx(0.0003)
    assert config.align_max_lateral_step_m == pytest.approx(0.005)
    assert config.align_mouth_max_lateral_step_m == pytest.approx(0.0015)
    assert config.align_lateral_tol_m < (
        SC_OPENING_HEIGHT_M - sc_controller.SC_PLUG_HEIGHT_M
    ) * 0.5

    with pytest.raises(ValueError, match="candidate depth"):
        SCConfig(seat_candidate_depth_m=0.030).validated()
    with pytest.raises(ValueError, match="overtravel"):
        SCConfig(seat_overtravel_m=0.004).validated()
    with pytest.raises(ValueError, match="align tolerance"):
        SCConfig(align_lateral_tol_m=0.001).validated()
    with pytest.raises(ValueError, match="baseline min samples"):
        SCConfig(
            visual_recovery_baseline_samples=1,
            visual_recovery_baseline_min_samples=2,
        ).validated()


def test_alignment_tolerance_has_a_deployment_override(monkeypatch):
    monkeypatch.setenv("RL_INSERT_SC_ALIGN_LATERAL_TOL_M", "0.0004")

    assert SCConfig.from_env().align_lateral_tol_m == pytest.approx(0.0004)


def test_stall_event_dwell_defaults_and_has_a_deployment_override(monkeypatch):
    assert SCConfig().validated().stall_event_dwell_wall_s == pytest.approx(3.0)

    monkeypatch.setenv("RL_INSERT_SC_STALL_EVENT_DWELL_WALL_S", "1.5")

    assert SCConfig.from_env().stall_event_dwell_wall_s == pytest.approx(1.5)


def test_stall_timers_default_to_the_extended_window(monkeypatch):
    monkeypatch.delenv("RL_INSERT_SC_STALL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("RL_INSERT_SC_SEAT_STALL_GRACE_S", raising=False)

    direct = SCConfig().validated()
    from_env = SCConfig.from_env()

    assert direct.stall_timeout_wall_s == pytest.approx(5.0)
    assert direct.seat_stall_grace_s == pytest.approx(3.0)
    assert from_env.stall_timeout_wall_s == pytest.approx(5.0)
    assert from_env.seat_stall_grace_s == pytest.approx(3.0)


def test_precontact_alignment_tuning_env_vars_are_read(monkeypatch):
    monkeypatch.setenv("RL_INSERT_SC_ALIGN_MAX_LATERAL_STEP_M", "0.006")
    monkeypatch.setenv("RL_INSERT_SC_PRECONTACT_STIFFNESS_N_M", "900")
    monkeypatch.setenv("RL_INSERT_SC_PRECONTACT_DAMPING_N_S_M", "180")

    reloaded = importlib.reload(sc_controller)
    try:
        assert reloaded.SCConfig.from_env().align_max_lateral_step_m == pytest.approx(
            0.006
        )
        np.testing.assert_allclose(
            reloaded.ScInsertionController.PRECONTACT_ALIGN_STIFFNESS,
            [900.0, 900.0, 900.0, 80.0, 80.0, 80.0],
        )
        np.testing.assert_allclose(
            reloaded.ScInsertionController.PRECONTACT_ALIGN_DAMPING,
            [180.0, 180.0, 180.0, 30.0, 30.0, 30.0],
        )
    finally:
        monkeypatch.delenv("RL_INSERT_SC_ALIGN_MAX_LATERAL_STEP_M", raising=False)
        monkeypatch.delenv("RL_INSERT_SC_PRECONTACT_STIFFNESS_N_M", raising=False)
        monkeypatch.delenv("RL_INSERT_SC_PRECONTACT_DAMPING_N_S_M", raising=False)
        importlib.reload(sc_controller)


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


def test_runtime_pose_contract_is_the_physical_front_mouth():
    spans = (
        SC_POSE_LOCAL_KPS_M[:4].max(axis=0)
        - SC_POSE_LOCAL_KPS_M[:4].min(axis=0)
    )
    assert SC_POSE_CONVENTION == "physical_front_mouth"
    assert SC_POSE_LOCAL_KPS_M.shape == (5, 3)
    assert spans[0] == pytest.approx(SC_POSE_WIDTH_M)
    assert spans[1] == pytest.approx(SC_POSE_HEIGHT_M)
    assert SC_POSE_WIDTH_M == pytest.approx(0.022407)
    assert SC_POSE_HEIGHT_M == pytest.approx(0.00810)
    np.testing.assert_allclose(SC_POSE_LOCAL_KPS_M[4], np.zeros(3))


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
    assert (
        _normalize_event("cable_0#1#/sc_port_0/sc_port_base")
        == "sc_port_0/sc_port_base"
    )
    assert _normalize_event("sc_mount_rail_0/sc_port_0") == "sc_mount_rail_0/sc_port_0"
    assert _normalize_event(None) == ""


def _calib_policy(frames_to_pose, tcp_pos):
    """Policy whose TF resolves exactly the frames given."""
    def _tf(pos):
        return type("_TF", (), {
            "transform": type("_T", (), {
                "translation": type("_V", (), dict(zip("xyz", pos)))(),
                "rotation": type("_Q", (), {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})(),
            })(),
            "header": type("_H", (), {
                "stamp": type("_S", (), {"sec": 1, "nanosec": 0})()})(),
        })()

    log = _EventLog()

    class _Policy:
        def _tcp(self):
            return np.asarray(tcp_pos, float), np.array([1.0, 0.0, 0.0, 0.0])

        def get_logger(self):
            return log

        def _lookup_transform(self, target, source, timeout_sec=0.2):
            if source not in frames_to_pose:
                raise RuntimeError(f"no such frame {source}")
            return _tf(frames_to_pose[source])

    return _Policy(), log


def test_calibration_refuses_a_frame_too_far_away_to_be_the_held_plug(monkeypatch):
    # 'selected_sc/sc_tip_link' is a real SC plug and resolves cleanly, but it is
    # static and ~30 cm off -- it is a second plug, not the grasped one.  Printing
    # it as SOLVED is how a wrong transform gets baked into the image.
    monkeypatch.setattr(sc_controller, "SC_CALIB_PLUG_FRAMES",
                        ["selected_sc/sc_tip_link"])
    tcp = [0.0, 0.0, 0.0]
    policy, log = _calib_policy({"selected_sc/sc_tip_link": [0.20, 0.20, 0.10]}, tcp)
    task = type("_Task", (), {"cable_name": "cable_0", "plug_name": "sc_tip"})()

    assert sc_controller.dump_sc_grasp_calibration(policy, task) is False
    assert any("REJECTED" in e for e in log.errors + getattr(log, "warns", []))


def test_calibration_accepts_a_frame_at_a_plausible_grasp_distance(monkeypatch):
    monkeypatch.setattr(sc_controller, "SC_CALIB_PLUG_FRAMES", ["cable_0/sc_tip_link"])
    tcp = [0.0, 0.0, 0.0]
    policy, _ = _calib_policy({"cable_0/sc_tip_link": [0.0, 0.0, 0.058]}, tcp)
    task = type("_Task", (), {"cable_name": "cable_0", "plug_name": "sc_tip"})()

    assert sc_controller.dump_sc_grasp_calibration(policy, task) is True


def test_tf_frame_names_parse_from_either_tf2_dump_format():
    # Which of the two dumps a tf2 build offers is not something the calibration
    # run should depend on -- a parse miss here costs a whole grasp sample.
    yaml_dump = (
        "base_link: \n  parent: 'world'\n"
        "cable_0/sc_tip_link: \n  parent: 'cable_0/sc_plug_link'\n"
    )
    string_dump = (
        "Frame cable_0/sc_tip_link exists with parent cable_0/sc_plug_link.\n"
        "Frame base_link exists with parent world.\n"
    )
    for dump in (yaml_dump, string_dump):
        assert sc_controller.parse_tf_frame_names(dump) == [
            "base_link", "cable_0/sc_tip_link",
        ]
    assert sc_controller.parse_tf_frame_names("") == []


class _EventLog:
    def __init__(self):
        self.errors = []
        self.warns = []
        self.infos = []

    def error(self, message):
        self.errors.append(str(message))

    def info(self, message):
        self.infos.append(str(message))

    def warn(self, message):
        self.warns.append(str(message))


def _event_harness(published, expected, generation=1):
    """Bare controller exercising only _event_status."""
    controller = object.__new__(ScInsertionController)
    node = type("_Node", (), {
        "_insertion_event_value": published,
        "_insertion_event_generation": generation,
    })()
    controller.policy = type("_Policy", (), {"_parent_node": node})()
    controller.log = _EventLog()
    controller.config = SCConfig().validated()
    controller.event_generation = 0
    controller.expected_event = expected
    return controller


def test_seat_event_matching_the_requested_port_is_seated():
    controller = _event_harness("sc_mount_rail_0/sc_port_0", "sc_mount_rail_0/sc_port_0")
    assert controller._event_status(SC_INSERT_DEPTH_M) == SEATED


def test_seat_event_at_the_real_field_shallow_depth_is_seated_immediately():
    # 2026-07-27 19:31 field run: the plug stalled at 1.39 mm and the scoring
    # event still fired 0.66 s later.  /scoring/insertion_event is a ~1 mm
    # tip-proximity trigger (aic#593), not full-bore contact, so a matching
    # event must be trusted at any depth -- no more waiting for depth to
    # catch up to seat_candidate_depth_m (15.2 mm).
    controller = _event_harness(
        "sc_mount_rail_0/sc_port_0", "sc_mount_rail_0/sc_port_0"
    )

    assert controller._event_status(0.00139) == SEATED
    assert controller.log.warns == []
    assert any(
        "SC_EVENT_ACCEPTED" in line and "depth_mm=1.39" in line
        for line in controller.log.infos
    )


def test_seat_event_at_negative_depth_is_seated_unconditionally():
    # depth_m is retained only for the acceptance log, not as a gate: the
    # policy must accept a fresh matching event regardless of its sign.
    controller = _event_harness(
        "sc_mount_rail_0/sc_port_0", "sc_mount_rail_0/sc_port_0"
    )

    assert controller._event_status(-0.0005) == SEATED


def test_non_strict_debug_mode_can_accept_a_different_port(monkeypatch):
    # Port selection is nearest-to-tip by design -- steering to the requested
    # port is the macro's job.  Failing here would make a physically correct
    # insertion indistinguishable from never seating at all.
    monkeypatch.setattr(sc_controller, "SC_STRICT_PORT_EVENT", False)
    controller = _event_harness("sc_mount_rail_0/sc_port_3", "sc_mount_rail_0/sc_port_0")
    assert controller._event_status(SC_INSERT_DEPTH_M) == SEATED
    # ...but it must name both ports, because scoring credits only the request.
    assert any("sc_port_3" in e and "sc_port_0" in e for e in controller.log.errors)


def test_strict_mode_restores_the_wrong_port_hard_failure(monkeypatch):
    monkeypatch.setattr(sc_controller, "SC_STRICT_PORT_EVENT", True)
    controller = _event_harness("sc_mount_rail_0/sc_port_3", "sc_mount_rail_0/sc_port_0")
    # Wrong-port safety remains immediate even if the contact plugin fires at
    # the mouth; only success-producing events are depth-gated.
    assert controller._event_status(0.0003) == HARD_FAILURE


def test_no_seat_verdict_until_a_new_event_generation_arrives():
    controller = _event_harness("sc_mount_rail_0/sc_port_0", "sc_mount_rail_0/sc_port_0",
                                generation=0)
    assert controller._event_status(SC_INSERT_DEPTH_M) is None


class _AlignHarness(ScInsertionController):
    def __init__(self, f_plug, m_plug, config=None):
        self.config = (config or SCConfig()).validated()
        self._f = np.asarray(f_plug, dtype=np.float64)
        self._m = np.asarray(m_plug, dtype=np.float64)

    def _wrench_plug_frame(self, observation):
        return self._f, self._m


def test_align_keeps_correcting_below_the_old_one_mm_stop():
    """A 0.9 mm residual used to start seating; it must now earn another move."""

    class _Policy:
        def __init__(self):
            self.targets = []

        def _enforce_action_deadline(self, _move_robot):
            pass

        def set_pose_target(self, _move_robot, target, **_kwargs):
            self.targets.append(target)

        def sleep_for(self, _duration):
            pass

    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig().validated()
    controller.policy = _Policy()
    controller.move_robot = object()
    controller.log = _RecordingLog()
    controller.Rp = np.eye(3)
    controller.Rs = np.eye(3)
    errors = iter(
        [
            (0.0, np.array([0.0009, 0.0]), np.zeros(3), np.zeros(3), np.eye(3)),
            (0.0, np.array([0.0009, 0.0]), np.zeros(3), np.zeros(3), np.eye(3)),
            (0.0, np.array([0.0003, 0.0]), np.zeros(3), np.zeros(3), np.eye(3)),
        ]
    )
    controller._errors = lambda: next(errors)
    controller._tcp_target = lambda tip, rotation: (tip, rotation)

    assert controller._align() is True
    assert len(controller.policy.targets) == 1


def test_precontact_alignment_keeps_one_stronger_lateral_segment():
    """A 7 mm handoff error uses the 5 mm precontact cap as one segment."""

    class _Policy:
        def __init__(self):
            self.commands = []

        def _enforce_action_deadline(self, _move_robot):
            pass

        def set_pose_target(self, _move_robot, target, **kwargs):
            self.commands.append((np.asarray(target, dtype=np.float64), kwargs))

        def sleep_for(self, _duration):
            pass

    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig().validated()
    controller.policy = _Policy()
    controller.move_robot = object()
    controller.log = _RecordingLog()
    controller.port_pos = np.zeros(3)
    controller.Rp = np.eye(3)
    controller.Rs = np.eye(3)
    controller._align_segment_lateral_xy = None
    controller._prime_visual_target = lambda: None

    def error(depth, lateral_x):
        return (
            depth,
            np.array([lateral_x, 0.0]),
            np.zeros(3),
            np.array([lateral_x, 0.0, depth]),
            np.eye(3),
        )

    # The second sample has moved slightly, but must retain the first segment
    # goal (2.0 mm), not compute a new 1.8 mm goal from its current 6.8 mm tip.
    errors = iter(
        [
            error(-0.005, 0.0070),
            error(-0.005, 0.0070),
            error(-0.005, 0.0068),
            error(-0.005, 0.0001),
        ]
    )
    controller._errors = lambda: next(errors)
    controller._tcp_target = lambda tip, _rotation: np.asarray(tip)

    assert controller._align() is True
    assert len(controller.policy.commands) == 2
    np.testing.assert_allclose(
        [command[0][0] for command in controller.policy.commands],
        [0.002, 0.002],
    )
    for _target, kwargs in controller.policy.commands:
        np.testing.assert_allclose(
            kwargs["stiffness"], controller.PRECONTACT_ALIGN_STIFFNESS
        )
    assert any("SC_ALIGN_SEGMENT mode=precontact" in line for line in controller.log.info_lines)


def test_at_mouth_alignment_keeps_the_conservative_lateral_step():
    """Near the mouth, a 7 mm error is still capped to the old 1.5 mm step."""

    class _Policy:
        def __init__(self):
            self.commands = []

        def _enforce_action_deadline(self, _move_robot):
            pass

        def set_pose_target(self, _move_robot, target, **kwargs):
            self.commands.append((np.asarray(target, dtype=np.float64), kwargs))

        def sleep_for(self, _duration):
            pass

    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig().validated()
    controller.policy = _Policy()
    controller.move_robot = object()
    controller.log = _RecordingLog()
    controller.port_pos = np.zeros(3)
    controller.Rp = np.eye(3)
    controller.Rs = np.eye(3)
    controller._align_segment_lateral_xy = None
    controller._prime_visual_target = lambda: None

    def error(depth, lateral_x):
        return (
            depth,
            np.array([lateral_x, 0.0]),
            np.zeros(3),
            np.array([lateral_x, 0.0, depth]),
            np.eye(3),
        )

    errors = iter(
        [
            error(0.0, 0.0070),
            error(0.0, 0.0070),
            error(0.0, 0.0001),
        ]
    )
    controller._errors = lambda: next(errors)
    controller._tcp_target = lambda tip, _rotation: np.asarray(tip)

    assert controller._align() is True
    assert len(controller.policy.commands) == 1
    target, kwargs = controller.policy.commands[0]
    assert target[0] == pytest.approx(0.0055)
    np.testing.assert_allclose(kwargs["stiffness"], controller.STIFFNESS)


def _timeout_align_harness(monkeypatch):
    """Shared setup for the two timeout tests below: a residual that never
    converges within the 0.11s budget."""
    class _Clock:
        now = 0.0

        def monotonic(self):
            return self.now

    class _Policy:
        def __init__(self, clock):
            self.clock = clock
            self.targets = []

        def _enforce_action_deadline(self, _move_robot):
            pass

        def set_pose_target(self, _move_robot, target, **_kwargs):
            self.targets.append(target)

        def sleep_for(self, duration):
            self.clock.now += float(duration)

    clock = _Clock()
    monkeypatch.setattr(sc_controller, "time", clock)
    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig(align_timeout_wall_s=0.11).validated()
    controller.policy = _Policy(clock)
    controller.move_robot = object()
    controller.log = _RecordingLog()
    controller.port_pos = np.zeros(3)
    controller.Rp = np.eye(3)
    controller.Rs = np.eye(3)
    controller._align_segment_lateral_xy = None
    controller._prime_visual_target = lambda: None
    controller._errors = lambda: (
        0.0,
        np.array([0.0010, 0.0]),
        np.zeros(3),
        np.zeros(3),
        np.eye(3),
    )
    controller._tcp_target = lambda tip, _rotation: tip
    return controller


def test_alignment_timeout_reports_final_residual_and_command_count(monkeypatch):
    # 2026-07-27: aborting on an alignment timeout throws away a possibly
    # nearly-converged pose and scores zero for it (docs/scoring.md:106-114).
    # By default (SC_STRICT_PERCEPTION off) a timeout no longer aborts -- it
    # returns the distinct ALIGN_TIMED_OUT sentinel (neither True nor False)
    # so run() can proceed to seating while still being able to tell a
    # timeout apart from real convergence.  The SC_ALIGN_TIMEOUT diagnostics
    # themselves are unchanged.
    controller = _timeout_align_harness(monkeypatch)

    assert controller._align() == sc_controller.ALIGN_TIMED_OUT
    assert any(
        "SC_ALIGN_TIMEOUT final_depth_mm=0.000 final_lateral_mm=1.000"
        in line
        and "commands=3" in line
        for line in controller.log.error_lines
    )
    assert any(
        "SC_ALIGN_TIMEOUT_PROCEEDING" in line for line in controller.log.warn_lines
    )


def test_strict_perception_restores_the_old_align_timeout_abort(monkeypatch):
    monkeypatch.setattr(sc_controller, "SC_STRICT_PERCEPTION", True)
    controller = _timeout_align_harness(monkeypatch)

    assert controller._align() is False
    assert not any(
        "SC_ALIGN_TIMEOUT_PROCEEDING" in line for line in controller.log.warn_lines
    )


class _RunRetryClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _RunRetryPolicy:
    def __init__(self, controller, clock, budget_s):
        self.controller = controller
        self.clock = clock
        self.targets = []
        self.events = []
        self._sc_grasp_transform = (np.zeros(3), np.eye(3))
        self._action_deadline_wall = clock.now + float(budget_s)

    def _enforce_action_deadline(self, _move_robot):
        self.events.append(("enforce", self.clock.now))

    def set_pose_target(self, _move_robot, target, **_kwargs):
        target = np.asarray(target, dtype=np.float64).copy()
        self.targets.append(target)
        self.events.append(("command", float(target[2]), self.clock.now))

    def sleep_for(self, duration):
        self.clock.now += float(duration)

    def get_logger(self):
        return self.controller.log


def _run_retry_harness(
    monkeypatch,
    outcomes,
    *,
    budget_s=300.0,
    seat_elapsed_s=0.0,
    standoff_depth=-0.010,
    stalled_depth=0.0012,
):
    clock = _RunRetryClock()
    monkeypatch.setattr(sc_controller.time, "monotonic", clock.monotonic)
    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig(
        command_dt_sim_s=0.05,
        align_timeout_wall_s=0.20,
        stall_event_dwell_wall_s=0.10,
    ).validated()
    controller.RETRY_UNLOAD_HOLD_WALL_S = 0.10
    controller.policy = _RunRetryPolicy(controller, clock, budget_s)
    controller.log = _RecordingLog()
    controller.send_feedback = lambda message: controller.policy.events.append(
        ("feedback", message)
    )
    controller.expected_event = "sc_mount_rail_0/sc_port_0"
    controller.get_observation = lambda: object()
    controller.move_robot = object()
    controller.port_pos = np.zeros(3)
    controller.Rp = np.eye(3)
    controller.Rs = np.eye(3)
    controller.R_yaw = np.eye(3)
    controller.depth = float(standoff_depth)
    controller.force = 3.0
    controller.seat_calls = 0
    controller.align_calls = 0
    controller._tcp_target = lambda tip, _rotation: np.asarray(tip)
    controller._force_magnitude = lambda _observation: controller.force
    controller._seating_stiffness = lambda: np.eye(6)

    def _errors():
        tip = np.array([0.0, 0.0, controller.depth], dtype=np.float64)
        return controller.depth, np.zeros(2), np.zeros(3), tip, np.eye(3)

    def _align():
        controller.align_calls += 1
        controller.depth = float(standoff_depth)
        return True

    outcomes = list(outcomes)

    def _seat():
        controller.seat_calls += 1
        controller.policy.events.append(("seat", controller.seat_calls))
        if seat_elapsed_s:
            clock.now += float(seat_elapsed_s)
        outcome = outcomes.pop(0)
        if outcome == STALLED:
            controller.policy.events.append(("dwell_timeout", controller.seat_calls))
            controller.depth = float(stalled_depth)
        elif outcome == SEATED:
            controller.depth = SC_INSERT_DEPTH_M
        return outcome

    controller._errors = _errors
    controller._align = _align
    controller._seat = _seat
    return controller


def test_run_proceeds_to_seating_after_an_alignment_timeout():
    """run() must not throw away an unconverged-but-attemptable pose.

    _seat() remains the safety net for a genuinely bad wedge (it trips
    lateral_safety_m / rotation_safety_rad and returns STALLED), so letting
    a timed-out alignment reach it cannot make the outcome worse than the
    abort it replaces.
    """
    controller = object.__new__(ScInsertionController)
    controller.policy = type("_P", (), {"_sc_grasp_transform": object()})()
    controller.log = _RecordingLog()
    controller.send_feedback = lambda *_a, **_k: None
    controller.expected_event = "sc_mount_rail_0/sc_port_0"
    controller.config = SCConfig().validated()
    controller.get_observation = lambda: object()
    controller._errors = lambda: (
        -0.010,
        np.zeros(2),
        np.zeros(3),
        np.array([0.0, 0.0, -0.010]),
        np.eye(3),
    )
    controller._align = lambda: sc_controller.ALIGN_TIMED_OUT
    controller._seat_attempt_has_budget = lambda full_retry=False: True
    seat_calls = []

    def _seat():
        seat_calls.append(True)
        return SEATED

    controller._seat = _seat

    assert controller.run() is True
    assert seat_calls == [True]
    assert any(
        "unconverged alignment" in line for line in controller.log.warn_lines
    )


def test_retry_starts_after_stall_dwell_times_out(monkeypatch):
    controller = _run_retry_harness(monkeypatch, [STALLED, SEATED])

    assert controller.run() is True

    events = [event[0] for event in controller.policy.events]
    assert events.index("dwell_timeout") < events.index("command")
    assert controller.seat_calls == 2
    assert any(
        "SC_RETRY_START attempt=2/budget rung=unload" in line
        for line in controller.log.warn_lines
    )


def test_late_event_during_stall_dwell_does_not_retry(monkeypatch):
    controller = _run_retry_harness(monkeypatch, [SEATED])

    assert controller.run() is True

    assert controller.seat_calls == 1
    assert not controller.policy.targets
    assert not any("SC_RETRY_START" in line for line in controller.log.warn_lines)


def test_attempt_a_unloads_before_redescending(monkeypatch):
    controller = _run_retry_harness(monkeypatch, [STALLED, SEATED])

    assert controller.run() is True

    command_events = [
        event for event in controller.policy.events if event[0] == "command"
    ]
    assert command_events
    assert min(event[1] for event in command_events) < 0.0012
    events = [event[0] for event in controller.policy.events]
    assert events.index("command") < events.index("seat", events.index("command"))


def _patch_retry_perception(monkeypatch, controller, *, port_pos, prime_calls,
                            port_calls):
    """Stub both re-prime and re-perception for the full-retry branch."""

    def re_prime(policy, _get_observation, _move_robot):
        prime_calls.append(True)
        policy._sc_grasp_transform = (
            np.array([0.001, -0.002, 0.003]),
            np.eye(3),
        )
        return True

    def re_perceive(*_args, **_kwargs):
        port_calls.append(True)
        if port_pos is None:
            return None
        return np.asarray(port_pos, dtype=np.float64), np.array(
            [1.0, 0.0, 0.0, 0.0]
        ), 1.6

    monkeypatch.setattr(sc_controller, "prime_sc_plug_pose", re_prime)
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose_consensus", re_perceive
    )
    controller._tip_pose = lambda: (np.zeros(3), np.eye(3))
    # Real controllers always carry this (set in __init__); the retry harness
    # never needed it until re-perception, which passes it through.
    controller.task = object()


def test_attempt_b_reprimes_and_reperceives_the_port(monkeypatch):
    """The full retry refreshes BOTH poses, grasp first.

    Grasp before port is load-bearing: candidate selection inside
    perceive_sc_port_pose_consensus is nearest-to-tip, and the tip comes from
    the grasp transform.
    """
    controller = _run_retry_harness(monkeypatch, [STALLED, STALLED, SEATED])
    prime_calls = []
    port_calls = []
    _patch_retry_perception(
        monkeypatch,
        controller,
        port_pos=[0.0004, -0.0003, 0.0],
        prime_calls=prime_calls,
        port_calls=port_calls,
    )

    assert controller.run() is True

    assert prime_calls == [True]
    assert port_calls == [True]
    assert controller.align_calls == 2
    # The refreshed pose is actually adopted, not just logged.
    np.testing.assert_allclose(controller.port_pos, [0.0004, -0.0003, 0.0])
    assert any("SC_REPRIME_DELTA" in line for line in controller.log.info_lines)
    assert any("SC_REPERCEIVE_DELTA" in line for line in controller.log.info_lines)
    info = "\n".join(controller.log.info_lines)
    assert info.index("SC_REPRIME_DELTA") < info.index("SC_REPERCEIVE_DELTA")


def test_reperceive_rejects_a_jump_to_the_neighbouring_port(monkeypatch):
    """A 40 mm shift is a different port, not a refresh -- keep the cached pose.

    3 of 6 runs on 2026-07-28 selected the neighbouring port. Adopting that
    mid-retry would command the arm sideways with the plug still in a mouth.
    """
    controller = _run_retry_harness(monkeypatch, [STALLED, STALLED, SEATED])
    _patch_retry_perception(
        monkeypatch,
        controller,
        port_pos=[0.040, 0.0, 0.0],
        prime_calls=[],
        port_calls=[],
    )

    assert controller.run() is True

    np.testing.assert_allclose(controller.port_pos, np.zeros(3))
    assert any(
        "SC_REPERCEIVE_REJECTED" in line for line in controller.log.error_lines
    )


def test_reperceive_failure_is_not_fatal_and_keeps_the_cached_pose(monkeypatch):
    controller = _run_retry_harness(monkeypatch, [STALLED, STALLED, SEATED])
    _patch_retry_perception(
        monkeypatch,
        controller,
        port_pos=None,
        prime_calls=[],
        port_calls=[],
    )

    assert controller.run() is True

    np.testing.assert_allclose(controller.port_pos, np.zeros(3))
    assert controller.align_calls == 2
    assert any(
        "SC_REPERCEIVE_FAILED" in line for line in controller.log.warn_lines
    )


def test_full_retry_budget_reserves_reperception_but_not_the_align_timeout():
    """The reservation must cover re-perception without over-reserving align.

    align_timeout_wall_s is 60 s deployed while alignment converges in 2-6 s;
    reserving the timeout would refuse retries that comfortably fit.
    """
    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig(align_timeout_wall_s=60.0).validated()

    cheap = controller._seat_attempt_budget_required_s(full_retry=False)
    full = controller._seat_attempt_budget_required_s(full_retry=True)

    assert full - cheap == pytest.approx(
        ScInsertionController.RETRY_ALIGN_BUDGET_S
        + ScInsertionController.RETRY_REPERCEIVE_BUDGET_S
        + 2.0
    )
    assert full < 60.0


def test_retry_ladder_is_not_capped_by_attempt_count(monkeypatch):
    """A fourth attempt must run; only the deadline ends the ladder.

    This used to stop at 3 and concede. With MAX_SEAT_ATTEMPTS=0 the same
    outcome sequence has to reach the seat on attempt 4.
    """
    controller = _run_retry_harness(monkeypatch, [STALLED, STALLED, STALLED, SEATED])
    _patch_retry_perception(
        monkeypatch,
        controller,
        port_pos=[0.0004, -0.0003, 0.0],
        prime_calls=[],
        port_calls=[],
    )

    assert controller.run() is True

    assert controller.seat_calls == 4
    assert not any(
        "SC_RETRY_EXHAUSTED" in line for line in controller.log.warn_lines
    )


def test_retry_ladder_alternates_unload_and_full_retry(monkeypatch):
    """unload, full, unload, full -- cheap rung first on every cycle.

    A full retry re-primes the grasp and re-perceives the port, so the unload
    that follows it is being tried against a different pose than the one before
    it, which is why the cheap rung is worth repeating rather than escalating
    to full-only after the first cycle.
    """
    prime_calls = []
    port_calls = []
    controller = _run_retry_harness(
        monkeypatch, [STALLED, STALLED, STALLED, STALLED, STALLED, SEATED]
    )
    _patch_retry_perception(
        monkeypatch,
        controller,
        port_pos=[0.0004, -0.0003, 0.0],
        prime_calls=prime_calls,
        port_calls=port_calls,
    )

    assert controller.run() is True

    rungs = [
        line.split("rung=")[1].split()[0]
        for line in controller.log.warn_lines
        if "SC_RETRY_START" in line
    ]
    assert rungs == ["unload", "full", "unload", "full", "unload"]
    # Two full rungs, each refreshing both poses; the initial align plus two.
    assert prime_calls == [True, True]
    assert port_calls == [True, True]
    assert controller.align_calls == 3
    assert controller.seat_calls == 6


def test_retry_ladder_ends_on_the_deadline_not_a_count(monkeypatch):
    """With a budget that affords one cheap rung, the ladder stops there."""
    config = SCConfig(
        command_dt_sim_s=0.05,
        align_timeout_wall_s=0.20,
        stall_event_dwell_wall_s=0.10,
    ).validated()
    per_attempt = (
        config.stall_timeout_wall_s
        + config.seat_stall_grace_s
        + config.stall_event_dwell_wall_s
        + config.command_dt_sim_s
    )
    controller = _run_retry_harness(
        monkeypatch,
        [STALLED] * 8,
        # Two seats plus the unload hold fit; a third does not.
        budget_s=2.0 * per_attempt + 0.10 + 0.05,
        seat_elapsed_s=per_attempt,
    )

    assert controller.run() is False

    assert controller.seat_calls == 2
    rungs = [
        line.split("rung=")[1].split()[0]
        for line in controller.log.warn_lines
        if "SC_RETRY_START" in line
    ]
    assert rungs == ["unload"]
    assert any(
        "SC_RETRY_EXHAUSTED attempts=2" in line
        for line in controller.log.warn_lines
    )


def test_retry_budget_guard_skips_unaffordable_retry(monkeypatch):
    config = SCConfig(
        command_dt_sim_s=0.05,
        align_timeout_wall_s=0.20,
        stall_event_dwell_wall_s=0.10,
    ).validated()
    required = (
        config.stall_timeout_wall_s
        + config.seat_stall_grace_s
        + config.stall_event_dwell_wall_s
        + config.command_dt_sim_s
    )
    controller = _run_retry_harness(
        monkeypatch,
        [STALLED, SEATED],
        budget_s=required + 0.05,
        seat_elapsed_s=0.10,
    )

    assert controller.run() is False

    assert controller.seat_calls == 1
    assert not any("SC_RETRY_START" in line for line in controller.log.warn_lines)
    assert any(
        "SC_RETRY_EXHAUSTED attempts=1" in line
        for line in controller.log.warn_lines
    )


def test_run_still_aborts_when_align_hard_fails():
    controller = object.__new__(ScInsertionController)
    controller.policy = type("_P", (), {"_sc_grasp_transform": object()})()
    controller.log = _RecordingLog()
    controller.send_feedback = lambda *_a, **_k: None
    controller._align = lambda: False
    seat_calls = []
    controller._seat = lambda: seat_calls.append(True)

    assert controller.run() is False
    assert seat_calls == []


def test_missing_visual_observation_is_fail_soft_and_retains_target():
    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig(visual_align_enable=True).validated()
    controller.log = _RecordingLog()
    controller.get_observation = lambda: None
    controller.port_pos = np.array([0.1, -0.2, 0.3])
    controller._visual_origin_port_pos = controller.port_pos.copy()
    controller._visual_last_attempt_wall_s = float("-inf")
    controller._visual_last_miss_log_wall_s = float("-inf")

    before = controller.port_pos.copy()
    corrected = controller._visual_refine_port(None, phase="align")

    assert corrected is False
    np.testing.assert_array_equal(controller.port_pos, before)
    assert any(
        "retain_last_target_and_continue" in message
        for message in controller.log.warn_lines
    )


def test_visual_target_uses_temporal_median_then_freezes():
    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig(
        visual_align_consensus_samples=5,
        visual_align_consensus_min_agree=4,
        visual_align_consensus_spread_m=0.00025,
    ).validated()
    controller.log = _RecordingLog()
    controller.Rp = np.eye(3)
    controller.port_pos = np.zeros(3)
    controller._visual_origin_port_pos = np.zeros(3)
    controller._visual_target_locked = False
    controller._visual_last_correction_xy = np.zeros(2)
    controller._visual_samples_xy = [
        np.array([-0.00050, 0.00010]),
        np.array([-0.00052, 0.00012]),
        np.array([-0.00048, 0.00009]),
        np.array([-0.00051, 0.00011]),
        np.array([+0.00080, -0.00060]),  # temporal outlier
    ]

    assert controller._finalize_visual_target(phase="prealign") is True

    assert controller._visual_target_locked is True
    np.testing.assert_allclose(
        controller.port_pos[:2],
        [-0.000505, 0.000105],
        atol=0.000011,
    )
    assert any(
        "SC_VISUAL_LOCK" in message and "target_frozen=true" in message
        for message in controller.log.info_lines
    )


def test_visual_target_insufficient_batch_freezes_raw_pose_and_continues():
    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig(
        visual_align_consensus_samples=7,
        visual_align_consensus_min_agree=4,
    ).validated()
    controller.log = _RecordingLog()
    controller.Rp = np.eye(3)
    controller.port_pos = np.array([0.1, -0.2, 0.3])
    controller._visual_origin_port_pos = controller.port_pos.copy()
    controller._visual_target_locked = False
    controller._visual_samples_xy = [
        np.array([-0.0005, 0.0001]),
        np.array([-0.0006, 0.0002]),
    ]

    before = controller.port_pos.copy()
    assert controller._finalize_visual_target(phase="prealign") is False

    assert controller._visual_target_locked is True
    np.testing.assert_array_equal(controller.port_pos, before)
    assert any(
        "freeze_raw_pose_and_continue" in message
        for message in controller.log.warn_lines
    )


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
            [cx, cy],
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


class _RecoveryClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class _RecoveryPolicy:
    def __init__(self, controller, clock):
        self.controller = controller
        self.clock = clock
        self.targets = []
        self.command_kwargs = []
        self.enforce_calls = 0

    def _enforce_action_deadline(self, _move_robot):
        self.enforce_calls += 1
        assert self.enforce_calls < 100, "recovery test failed to terminate"

    def set_pose_target(self, _move_robot, target, **_kwargs):
        self.targets.append(np.asarray(target, dtype=np.float64).copy())
        self.command_kwargs.append(dict(_kwargs))

    def sleep_for(self, duration):
        self.clock.now += float(duration)
        if self.controller.on_sleep is not None:
            self.controller.on_sleep(self.targets[-1])


class _RecoverySeatHarness(ScInsertionController):
    """Deterministic no-ROS exercise of the complete seat/recovery state."""

    def __init__(
        self,
        *,
        depth=0.0003,
        force=1.0,
        estimates=(),
        estimate_failures=(),
        config=None,
    ):
        self.config = (
            config
            or SCConfig(
                seat_align_enable=False,
                stall_timeout_wall_s=0.10,
                seat_stall_grace_s=0.0,
                command_dt_sim_s=0.05,
                visual_recovery_settle_wall_s=0.0,
            )
        ).validated()
        self.depth = float(depth)
        self.force = float(force)
        self.estimates = list(estimates)
        self.estimate_failures = list(estimate_failures)
        self.estimate_calls = 0
        self.log = _RecordingLog()
        self.clock = _RecoveryClock()
        self.policy = _RecoveryPolicy(self, self.clock)
        self.move_robot = object()
        self.port_pos = np.zeros(3, dtype=np.float64)
        self.Rp = np.eye(3)
        self.Rs = np.eye(3)
        self.R_yaw = np.eye(3)
        self.on_sleep = None
        self.finish = False
        self.holds = []
        self.dwell_tip = None
        self.get_observation = lambda: _stamped_recovery_observation(1.0)
        # Seat tests override the image estimator, but recovery activation now
        # correctly requires that a clean pre-contact reference existed.
        self._visual_recovery_baseline = {
            "left_camera": {10: np.ones(4), 12: np.ones(4)},
            "right_camera": {10: np.ones(4), 12: np.ones(4)},
        }

    def _errors(self):
        tip = np.array([0.0, 0.0, self.depth], dtype=np.float64)
        return self.depth, np.zeros(2), np.zeros(3), tip, np.eye(3)

    def _force_magnitude(self, _observation):
        return self.force

    def _event_status(self, _depth_m):
        return STALLED if self.finish else None

    def _tcp_target(self, tip_pos, _tip_rotation):
        return np.asarray(tip_pos, dtype=np.float64)

    def _visual_recovery_estimate(self, _observation):
        self.estimate_calls += 1
        result = self.estimates.pop(0) if self.estimates else None
        failure = (
            self.estimate_failures.pop(0)
            if self.estimate_failures
            else None
        )
        self._visual_recovery_last_failure_reason = (
            failure if result is None else None
        )
        return result

    def _hold_tip(self, tip_pos, _tip_rotation):
        self.holds.append(np.asarray(tip_pos, dtype=np.float64).copy())

    def _wait_for_insertion_event(self, fixed_tip):
        self.dwell_tip = np.asarray(fixed_tip, dtype=np.float64).copy()
        return STALLED


def _recovery_estimate(direction=(1.0, 0.0), *, balanced=False):
    return ScRecoveryEstimate(
        direction_xy=np.asarray(direction, dtype=np.float64),
        confidence=0.9,
        cameras=("left", "right"),
        resultant=0.98,
        balanced=balanced,
    )


def _stamped_recovery_observation(stamp_s):
    header = type("_Header", (), {"stamp": float(stamp_s)})()
    message = type("_Image", (), {"header": header})()
    return type(
        "_Observation",
        (),
        {
            "left_image": message,
            "center_image": message,
            "right_image": message,
        },
    )()


def test_controller_recovery_projection_preserves_port_local_direction():
    image = np.full((192, 320, 3), (255, 80, 0), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(image, (58, 58), (261, 133), (8, 8, 8), -1)
    # Low plug: there is more physical blue clearance in port-local +Y.
    cv2.rectangle(image, (68, 64), (251, 141), (178, 178, 178), -1)

    class _ProjectionCore:
        def build_projection_matrix(self, _K, _T):
            x_scale = 224.0 / SC_OPENING_WIDTH_M
            y_scale = -96.0 / SC_OPENING_HEIGHT_M
            return np.array(
                [
                    [x_scale, 0.0, 0.0, 160.0],
                    [0.0, y_scale, 0.0, 96.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

    class _Policy:
        _pc = _ProjectionCore()

        def _build_views(self, _observation):
            view = (image, np.eye(3), np.eye(4))
            return {"left_camera": view, "right_camera": view}

    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig().validated()
    controller.policy = _Policy()
    controller.port_pos = np.zeros(3)
    controller.Rp = np.eye(3)
    controller.log = _RecordingLog()
    controller._visual_ignored_pixels = (
        lambda _camera, shape: np.zeros(shape[:2], dtype=bool)
    )
    controller._visual_recovery_baseline = {
        "left_camera": {10: np.ones(4), 12: np.ones(4)},
        "right_camera": {10: np.ones(4), 12: np.ones(4)},
    }

    result = controller._visual_recovery_estimate(
        _stamped_recovery_observation(1.0)
    )

    assert result is not None
    assert result.direction_xy[1] > 0.95
    assert result.cameras == ("left_camera", "right_camera")
    assert any(
        "accepted=['left_camera', 'right_camera'] rejected={}" in line
        for line in controller.log.info_lines
    )
    assert any(
        "SC_VISUAL_RECOVERY_SUPPORT" in line
        for line in controller.log.info_lines
    )
    assert any("frame_age_s=" in line for line in controller.log.info_lines)
    assert (
        controller._visual_recovery_estimate(
            _stamped_recovery_observation(1.0)
        )
        is None
    )
    assert controller._visual_recovery_last_failure_reason == "stale_frames"
    assert (
        controller._visual_recovery_estimate(
            _stamped_recovery_observation(1.1)
        )
        is not None
    )


def test_controller_recovery_requires_gripper_masks():
    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig().validated()
    controller.log = _RecordingLog()
    controller.port_pos = np.zeros(3)
    controller.Rp = np.eye(3)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    class _ProjectionCore:
        def build_projection_matrix(self, _K, _T):
            return np.array(
                [
                    [1000.0, 0.0, 0.0, 50.0],
                    [0.0, -1000.0, 0.0, 50.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

    class _Policy:
        _pc = _ProjectionCore()

        def _build_views(self, _observation):
            view = (image, np.eye(3), np.eye(4))
            return {"left_camera": view, "right_camera": view}

    controller.policy = _Policy()
    controller._visual_ignored_pixels = lambda _camera, _shape: None

    result = controller._visual_recovery_estimate(
        _stamped_recovery_observation(1.0)
    )

    assert result is None
    assert any(
        "gripper_mask_unavailable" in line
        for line in controller.log.warn_lines + controller.log.info_lines
    )


def test_controller_captures_fresh_post_target_blue_side_baselines(monkeypatch):
    """A stale pre-contact frame cannot be counted twice as independent proof."""

    class _ProjectionCore:
        def build_projection_matrix(self, _K, _T):
            return np.array(
                [[100.0, 0.0, 0.0, 120.0], [0.0, 100.0, 0.0, 120.0],
                 [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

    class _Policy:
        def __init__(self):
            self._pc = _ProjectionCore()
            self.sleeps = []

        def _enforce_action_deadline(self, _move_robot):
            pass

        def sleep_for(self, duration):
            self.sleeps.append(float(duration))

        def _build_views(self, _observation):
            image = np.full((300, 300, 3), (255, 80, 0), dtype=np.uint8)
            view = (image, np.eye(3), np.eye(4))
            return {"left_camera": view, "right_camera": view}

    stamps = iter((1.0, 1.0, 1.1))

    def observation():
        stamp = next(stamps)
        message = type(
            "_Image", (), {"header": type("_Header", (), {"stamp": stamp})()}
        )()
        return type(
            "_Observation", (),
            {"left_image": message, "right_image": message, "center_image": None},
        )()

    controller = object.__new__(ScInsertionController)
    controller.config = SCConfig(
        visual_recovery_baseline_samples=3,
        visual_recovery_baseline_min_samples=2,
    ).validated()
    controller.policy = _Policy()
    controller.move_robot = object()
    controller.log = _RecordingLog()
    controller.get_observation = observation
    controller.port_pos = np.zeros(3)
    controller.Rp = np.eye(3)
    controller._visual_ignored_pixels = (
        lambda _camera, shape: np.zeros(shape[:2], dtype=bool)
    )

    def signature(_image, _quads, _ignored, *, band_half_width_px, **_kwargs):
        fraction = 0.80 + 0.01 * band_half_width_px
        support = tuple(np.ones((10, 10), dtype=bool) for _ in range(4))
        blue = []
        for _ in range(4):
            side = np.zeros((10, 10), dtype=bool)
            side.flat[: int(round(fraction * side.size))] = True
            blue.append(side)
        return ScBlueSideSignature(
            blue_fractions=np.full(4, fraction),
            blue_fraction=0.8,
            valid_fraction=1.0,
            side_support_masks=support,
            side_blue_masks=tuple(blue),
            corridor_support_mask=np.ones((20, 20), dtype=bool),
        )

    monkeypatch.setattr(
        sc_controller, "measure_sc_blue_side_signature", signature
    )

    assert controller._prime_visual_recovery_baseline() is True
    assert set(controller._visual_recovery_baseline) == {
        "left_camera", "right_camera"
    }
    for camera in controller._visual_recovery_baseline:
        np.testing.assert_allclose(
            controller._visual_recovery_baseline[camera][10].blue_fractions, 0.90
        )
        np.testing.assert_allclose(
            controller._visual_recovery_baseline[camera][12].blue_fractions, 0.92
        )
    assert controller._visual_recovery_baseline_last_frame_stamps == {
        "left_camera": 1.1,
        "right_camera": 1.1,
    }
    assert any("stale_frame" in line for line in controller.log.info_lines)


def test_port_perception_checks_deadline_around_each_sample(monkeypatch):
    class _Policy:
        def __init__(self):
            self.log = _RecordingLog()
            self.enforce_calls = 0
            self.sleeps = 0

        def get_logger(self):
            return self.log

        def _enforce_action_deadline(self, _move_robot):
            self.enforce_calls += 1

        def sleep_for(self, _duration):
            self.sleeps += 1

    policy = _Policy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 2)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 1)
    monkeypatch.setattr(
        sc_controller,
        "perceive_sc_port_pose",
        lambda *_args: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 1.0),
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is not None
    assert policy.enforce_calls == 6  # before inference, after it, after sleep
    assert policy.sleeps == 2
    assert any("SC_PERCEPTION_TIMING outcome=accepted" in line for line in policy.log.info_lines)


class _ConsensusPolicy:
    """Minimal policy exercising only perceive_sc_port_pose_consensus's loop."""

    def __init__(self):
        self.log = _RecordingLog()

    def get_logger(self):
        return self.log

    def _enforce_action_deadline(self, _move_robot):
        pass

    def sleep_for(self, _duration):
        pass


def test_consensus_below_min_agree_falls_back_to_the_best_sample(monkeypatch):
    """2026-07-27: refusing to attempt scores worse than attempting with a
    mediocre estimate (docs/scoring.md).  Too few usable frames must degrade
    to the single best (lowest centre_reproj_px) sample instead of None."""
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    best_pos = np.array([0.0, 0.0, 0.10])
    best_quat = np.array([1.0, 0.0, 0.0, 0.0])
    results = iter(
        [
            (best_pos, best_quat, 1.8),  # lowest centre reproj -- the fallback
            (best_pos + 0.0005, best_quat, 3.2),
            None,
        ]
    )
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose", lambda *_args: next(results)
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is not None
    port_pos, quat, reproj = result
    np.testing.assert_allclose(port_pos, best_pos)
    np.testing.assert_allclose(quat, best_quat)
    assert reproj == pytest.approx(1.8)
    assert any("SC_PERCEPT_DEGRADED" in line for line in policy.log.warn_lines)
    assert any(
        "reason=insufficient_samples" in line for line in policy.log.warn_lines
    )
    assert any(
        "outcome=degraded_insufficient_samples" in line
        for line in policy.log.info_lines
    )


def test_consensus_disagreement_falls_back_to_the_best_sample(monkeypatch):
    """Enough samples arrive, but they disagree beyond SC_PERCEPT_AGREE_TOL_M
    -- this must also degrade to the single best sample rather than None."""
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_AGREE_TOL_M", 0.001)
    best_pos = np.array([0.0, 0.0, 0.10])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    # Three samples, none of which cluster within 1mm of each other's median.
    results = iter(
        [
            (best_pos, quat, 2.0),
            (best_pos + np.array([0.010, 0.0, 0.0]), quat, 3.5),
            (best_pos - np.array([0.010, 0.0, 0.0]), quat, 4.0),
        ]
    )
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose", lambda *_args: next(results)
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is not None
    port_pos, _quat, reproj = result
    np.testing.assert_allclose(port_pos, best_pos)
    assert reproj == pytest.approx(2.0)
    assert any(
        "SC_PERCEPT_DEGRADED" in line and "reason=disagreement" in line
        for line in policy.log.warn_lines
    )
    assert any(
        "outcome=degraded_disagreement" in line for line in policy.log.info_lines
    )


def test_consensus_zero_usable_samples_still_returns_none(monkeypatch):
    """There is nothing to fall back to with zero usable samples -- this is
    the one case SC_STRICT_PERCEPTION cannot change."""
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    monkeypatch.setattr(sc_controller, "perceive_sc_port_pose", lambda *_args: None)

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is None
    assert any(
        "outcome=no_samples" in line for line in policy.log.info_lines
    )
    assert not any("SC_PERCEPT_DEGRADED" in line for line in policy.log.warn_lines)


def test_consensus_keeps_frames_that_degraded_above_the_port_reproj_gate(monkeypatch):
    """perceive_sc_port_pose can now legitimately return a frame whose
    centre_reproj_px is above SC_MAX_PORT_REPROJ_PX (its own select gate
    already degraded once).  The consensus sample loop must not silently
    drop those frames -- that would reintroduce, one level up, exactly the
    reject-on-gate-miss veto the select-gate degrade removed.  All 7 frames
    above the 6.0px gate must still produce a pose, not None."""
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    best_pos = np.array([0.0, 0.0, 0.10])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    # Every frame is above SC_MAX_PORT_REPROJ_PX (6.0px, default) but they
    # agree tightly in position.
    results = iter(
        [
            (best_pos, quat, 6.4),
            (best_pos + np.array([0.0001, 0.0, 0.0]), quat, 7.1),
            (best_pos - np.array([0.0001, 0.0, 0.0]), quat, 8.9),
        ]
    )
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose", lambda *_args: next(results)
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is not None
    port_pos, _quat, reproj = result
    np.testing.assert_allclose(port_pos, best_pos, atol=1e-3)
    assert any(
        "SC_PERCEPT_DEGRADED" in line
        and "reason=no_sample_under_port_reproj_gate" in line
        and "best_centre_reproj_px=6.40" in line
        and "gate_px=6.0" in line
        for line in policy.log.warn_lines
    )
    assert any("outcome=accepted" in line for line in policy.log.info_lines)


def test_strict_perception_restores_none_when_nothing_clears_the_port_reproj_gate(
    monkeypatch,
):
    """Under SC_STRICT_PERCEPTION the sample loop must filter at the 6.0px
    gate exactly as before -- frames that degraded past it must not survive
    into the consensus at all, and the run must abort."""
    monkeypatch.setattr(sc_controller, "SC_STRICT_PERCEPTION", True)
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    best_pos = np.array([0.0, 0.0, 0.10])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    results = iter(
        [
            (best_pos, quat, 6.4),
            (best_pos + np.array([0.0001, 0.0, 0.0]), quat, 7.1),
            (best_pos - np.array([0.0001, 0.0, 0.0]), quat, 8.9),
        ]
    )
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose", lambda *_args: next(results)
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is None
    assert not any("SC_PERCEPT_DEGRADED" in line for line in policy.log.warn_lines)


def test_consensus_prefers_under_gate_samples_over_degraded_ones(monkeypatch):
    """When both kinds of frames are present, the under-gate ones must win
    the consensus outright -- the full pool (including the degraded, above-
    gate frame) must never even be considered while good samples exist."""
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 4)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    good_pos = np.array([0.0, 0.0, 0.10])
    bad_pos = good_pos + np.array([0.050, 0.0, 0.0])  # far outlier, if ever used
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    results = iter(
        [
            (good_pos, quat, 1.5),                     # under gate
            (good_pos + np.array([0.0001, 0.0, 0.0]), quat, 2.0),  # under gate
            (good_pos - np.array([0.0001, 0.0, 0.0]), quat, 3.0),  # under gate
            (bad_pos, quat, 9.0),                       # degraded, above gate
        ]
    )
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose", lambda *_args: next(results)
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is not None
    port_pos, _quat, _reproj = result
    np.testing.assert_allclose(port_pos, good_pos, atol=1e-3)
    # The full-pool degrade path must never have fired -- three good samples
    # were enough on their own.
    assert not any(
        "reason=no_sample_under_port_reproj_gate" in line
        for line in policy.log.warn_lines
    )
    assert any("outcome=accepted" in line for line in policy.log.info_lines)


def test_strict_perception_restores_consensus_none_on_insufficient_samples(
    monkeypatch,
):
    monkeypatch.setattr(sc_controller, "SC_STRICT_PERCEPTION", True)
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    best_pos = np.array([0.0, 0.0, 0.10])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    results = iter([(best_pos, quat, 1.8), None, None])
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose", lambda *_args: next(results)
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is None
    assert not any("SC_PERCEPT_DEGRADED" in line for line in policy.log.warn_lines)


def test_strict_perception_restores_consensus_none_on_disagreement(monkeypatch):
    monkeypatch.setattr(sc_controller, "SC_STRICT_PERCEPTION", True)
    policy = _ConsensusPolicy()
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_SAMPLES", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_MIN_AGREE", 3)
    monkeypatch.setattr(sc_controller, "SC_PERCEPT_AGREE_TOL_M", 0.001)
    best_pos = np.array([0.0, 0.0, 0.10])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    results = iter(
        [
            (best_pos, quat, 2.0),
            (best_pos + np.array([0.010, 0.0, 0.0]), quat, 3.5),
            (best_pos - np.array([0.010, 0.0, 0.0]), quat, 4.0),
        ]
    )
    monkeypatch.setattr(
        sc_controller, "perceive_sc_port_pose", lambda *_args: next(results)
    )

    result = sc_controller.perceive_sc_port_pose_consensus(
        policy, object(), lambda: object(), object()
    )

    assert result is None
    assert not any("SC_PERCEPT_DEGRADED" in line for line in policy.log.warn_lines)


def test_seating_stiffness_is_port_frame_axial_force_contract():
    controller = _RecoverySeatHarness()
    angle = np.deg2rad(35.0)
    controller.Rp = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )

    stiffness = controller._seating_stiffness()
    port_translation = controller.Rp.T @ stiffness[:3, :3] @ controller.Rp
    np.testing.assert_allclose(
        port_translation,
        np.diag([90.0, 90.0, controller.config.axial_stiffness_n_m]),
        atol=1e-10,
    )
    axis = controller.Rp[:, 2]
    assert axis @ stiffness[:3, :3] @ axis == pytest.approx(
        controller.config.axial_stiffness_n_m
    )
    np.testing.assert_allclose(stiffness[3:, 3:], np.diag(controller.STIFFNESS[3:]))


def test_contact_phase_stiffness_constants_stay_unchanged():
    np.testing.assert_allclose(
        ScInsertionController.STIFFNESS,
        [90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
    )
    np.testing.assert_allclose(
        ScInsertionController.DAMPING,
        [50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
    )
    assert ScInsertionController.SEAT_LATERAL_STIFFNESS_N_M == pytest.approx(90.0)
    assert SCConfig().validated().axial_stiffness_n_m == pytest.approx(500.0)
    np.testing.assert_allclose(
        ScInsertionController.HOLD_STIFFNESS,
        [200.0, 200.0, 200.0, 80.0, 80.0, 80.0],
    )
    np.testing.assert_allclose(
        ScInsertionController.HOLD_DAMPING,
        [80.0, 80.0, 80.0, 30.0, 30.0, 30.0],
    )


def test_recovery_sends_impedance_matching_its_one_newton_force_lead(monkeypatch):
    controller = _RecoverySeatHarness(estimates=[_recovery_estimate(), None])

    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    recovery_commands = [
        (target, kwargs)
        for target, kwargs in zip(controller.policy.targets, controller.policy.command_kwargs)
        if target[0] > 0.0
    ]
    assert recovery_commands
    target, kwargs = recovery_commands[0]
    stiffness = np.asarray(kwargs["stiffness"], dtype=np.float64)
    assert stiffness.shape == (6, 6)
    axis = controller.Rp[:, 2]
    axial_stiffness = float(axis @ stiffness[:3, :3] @ axis)
    recovery_lead_m = target[2] - controller.depth
    assert axial_stiffness == pytest.approx(controller.config.axial_stiffness_n_m)
    assert axial_stiffness * recovery_lead_m == pytest.approx(
        controller.config.visual_recovery_force_n
    )


def _run_recovery_harness(monkeypatch, controller):
    fake_time = type(
        "_FakeTime", (), {"monotonic": controller.clock.monotonic}
    )()
    monkeypatch.setattr(sc_controller, "time", fake_time)
    return controller._seat()


@pytest.mark.parametrize(
    ("depth", "force"),
    [
        (0.003, 1.0),  # already beyond the visible mouth
        (0.0003, 2.0),  # too much load for image-guided lateral motion
    ],
)
def test_visual_recovery_only_activates_at_shallow_light_contact(
    monkeypatch, depth, force
):
    controller = _RecoverySeatHarness(
        depth=depth,
        force=force,
        estimates=[_recovery_estimate()],
    )

    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 0
    assert any(
        "SC_VISUAL_RECOVERY_SKIPPED" in line
        for line in controller.log.warn_lines
    )


def test_weak_recovery_vision_commands_no_lateral_motion(monkeypatch):
    controller = _RecoverySeatHarness(estimates=[None])

    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 1
    assert all(abs(target[0]) < 1e-12 for target in controller.policy.targets)
    assert controller.holds


def test_visual_recovery_does_not_activate_without_clean_baselines(monkeypatch):
    controller = _RecoverySeatHarness(estimates=[_recovery_estimate()])
    controller._visual_recovery_baseline = {}

    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 0
    assert any(
        "baseline_views=False" in line for line in controller.log.warn_lines
    )


def test_visual_recovery_reacquires_each_step_and_stops_at_cap(monkeypatch):
    controller = _RecoverySeatHarness(
        estimates=[
            _recovery_estimate(),
            _recovery_estimate(),
            _recovery_estimate(),
        ],
        config=SCConfig(
            seat_align_enable=False,
            stall_timeout_wall_s=0.10,
            seat_stall_grace_s=0.0,
            command_dt_sim_s=0.05,
            visual_recovery_settle_wall_s=0.0,
            visual_recovery_max_total_m=0.0005,
            visual_recovery_step_m=0.00025,
        ),
    )

    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 3
    lateral_targets = [
        target[0] for target in controller.policy.targets if target[0] > 0.0
    ]
    np.testing.assert_allclose(lateral_targets, [0.00025, 0.0005])
    assert max(lateral_targets) <= 0.0005
    assert any(
        "travel_cap_reached" in line for line in controller.log.warn_lines
    )


def test_visual_recovery_stops_if_force_rises_after_a_step(monkeypatch):
    controller = _RecoverySeatHarness(
        estimates=[_recovery_estimate(), _recovery_estimate()]
    )

    def raise_force_after_lateral_target(target):
        if target[0] > 0.0:
            controller.force = 1.6

    controller.on_sleep = raise_force_after_lateral_target
    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 1
    assert controller.holds
    assert any("reason=force_gate" in line for line in controller.log.warn_lines)


def test_visual_recovery_waits_for_fresh_frames_without_another_step(
    monkeypatch,
):
    controller = _RecoverySeatHarness(
        estimates=[
            _recovery_estimate(),
            None,
            None,
            None,
        ],
        estimate_failures=[
            None,
            "stale_frames",
            "stale_frames",
            "stale_frames",
        ],
        config=SCConfig(
            seat_align_enable=False,
            stall_timeout_wall_s=0.10,
            seat_stall_grace_s=0.0,
            command_dt_sim_s=0.05,
            visual_recovery_settle_wall_s=0.0,
            visual_recovery_frame_timeout_wall_s=0.09,
        ),
    )

    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 4
    lateral_targets = [
        target[0] for target in controller.policy.targets if target[0] > 0.0
    ]
    assert lateral_targets
    np.testing.assert_allclose(lateral_targets, 0.00025)
    assert any(
        "reason=fresh_frame_timeout" in line
        for line in controller.log.warn_lines
    )


def test_visual_recovery_requires_a_post_stall_frame_for_first_step(
    monkeypatch,
):
    controller = _RecoverySeatHarness(
        estimates=[None, None, None],
        estimate_failures=[
            "stale_frames",
            "stale_frames",
            "stale_frames",
        ],
        config=SCConfig(
            seat_align_enable=False,
            stall_timeout_wall_s=0.10,
            seat_stall_grace_s=0.0,
            command_dt_sim_s=0.05,
            visual_recovery_settle_wall_s=0.0,
            visual_recovery_frame_timeout_wall_s=0.09,
        ),
    )

    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 3
    assert all(abs(target[0]) < 1e-12 for target in controller.policy.targets)
    assert controller._visual_recovery_last_frame_stamps == {
        "left_camera": 1.0,
        "center_camera": 1.0,
        "right_camera": 1.0,
    }
    assert any(
        "reason=fresh_frame_timeout" in line
        for line in controller.log.warn_lines
    )


def test_depth_gain_exits_recovery_and_preserves_offset(monkeypatch):
    controller = _RecoverySeatHarness(estimates=[_recovery_estimate()])
    lateral_commands = 0

    def advance_then_finish(target):
        nonlocal lateral_commands
        if target[0] <= 0.0:
            return
        lateral_commands += 1
        if lateral_commands == 1:
            controller.depth += 0.0006
        else:
            controller.finish = True

    controller.on_sleep = advance_then_finish
    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 1
    lateral_targets = [
        target[0] for target in controller.policy.targets if target[0] > 0.0
    ]
    np.testing.assert_allclose(lateral_targets[-2:], [0.00025, 0.00025])
    assert any(
        "reason=depth_advanced" in line for line in controller.log.info_lines
    )


def test_balanced_after_step_resumes_seating_at_corrected_offset(monkeypatch):
    controller = _RecoverySeatHarness(
        estimates=[
            _recovery_estimate(),
            _recovery_estimate((0.0, 0.0), balanced=True),
        ]
    )
    lateral_commands = 0

    def finish_after_normal_resume(target):
        nonlocal lateral_commands
        if target[0] > 0.0:
            lateral_commands += 1
            if lateral_commands == 2:
                controller.finish = True

    controller.on_sleep = finish_after_normal_resume
    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.estimate_calls == 2
    assert controller.holds == []
    assert any(
        "clearances_balanced_after_step" in line
        for line in controller.log.info_lines
    )


def test_recovery_offset_is_carried_into_event_dwell(monkeypatch):
    controller = _RecoverySeatHarness(estimates=[_recovery_estimate()])

    def seat_after_first_step(target):
        if target[0] > 0.0:
            controller.depth = controller.config.seat_candidate_depth_m

    controller.on_sleep = seat_after_first_step
    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == STALLED
    assert controller.dwell_tip is not None
    assert controller.dwell_tip[0] == pytest.approx(0.00025)
    assert controller.dwell_tip[2] == pytest.approx(SC_INSERT_DEPTH_M)


def test_stall_with_a_late_event_during_the_dwell_is_seated(monkeypatch):
    # 2026-07-27 19:31 field run: the scoring event fired 0.66 s AFTER _seat()
    # had already returned STALLED.  A stall that is about to become a real
    # event must not be reported as STALLED just because the event hadn't
    # landed yet on the tick the stall was declared.
    #
    # depth=0.003 is past seat_mouth_zone_m, so recovery gating never
    # activates and _hold_tip is untouched until _stall_event_dwell calls it
    # -- that makes "holds became non-empty" an unambiguous signal that the
    # dwell (and only the dwell) is now polling for the event.
    controller = _RecoverySeatHarness(depth=0.003, force=1.0)

    def event_status_once_dwelling(_depth_m):
        return SEATED if controller.holds else None

    monkeypatch.setattr(controller, "_event_status", event_status_once_dwelling)
    outcome = _run_recovery_harness(monkeypatch, controller)

    assert outcome == SEATED
    assert controller.holds
    assert any(
        "SC_STALL_EVENT_DWELL_START" in line for line in controller.log.info_lines
    )
    assert not any(
        "SC_STALL_EVENT_DWELL_TIMEOUT" in line for line in controller.log.warn_lines
    )


def test_stall_with_no_event_during_the_dwell_is_stalled(monkeypatch):
    controller = _RecoverySeatHarness(
        depth=0.003,
        force=1.0,
        config=SCConfig(
            seat_align_enable=False,
            stall_timeout_wall_s=0.10,
            seat_stall_grace_s=0.0,
            command_dt_sim_s=0.05,
            visual_recovery_settle_wall_s=0.0,
            stall_event_dwell_wall_s=0.15,
        ),
    )

    outcome = _run_recovery_harness(monkeypatch, controller)

    # _RecoverySeatHarness._event_status only ever returns non-None once
    # controller.finish is set, which this test never does, so the dwell
    # genuinely runs out its full window with no event.
    assert outcome == STALLED
    assert any(
        "SC_STALL_EVENT_DWELL_START" in line for line in controller.log.info_lines
    )
    assert any(
        "SC_STALL_EVENT_DWELL_TIMEOUT" in line for line in controller.log.warn_lines
    )


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


def test_port_selection_gates_lateral_distance_in_the_candidate_port_frame(monkeypatch):
    """A high handoff stays selectable when it is laterally over the mouth."""
    log = _RecordingLog()
    tip_pos = np.array([0.3, -0.2, 0.4], dtype=np.float64)
    # Runtime SC candidates use world -Z as inward.  The 40 mm axial separation
    # would fail the retired 30 mm 3D gate; its 9 mm port-frame lateral
    # separation must pass the 10 mm gate.
    Rp = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )
    q_wxyz = matrix_to_quat(Rp)
    target = {
        "X": tip_pos - Rp @ np.array([0.009, 0.0, 0.040]),
        "q_wxyz": q_wxyz,
        "score": 1.0,
        "reproj_px": 1.0,
        "width": _LABEL_WIDTH_M,
        "height": _LABEL_HEIGHT_M,
        "opening": SC_POSE_CONVENTION,
        "opening_residual_m": 0.0,
        "center_disagreement_m": 0.0,
    }
    assert np.linalg.norm(target["X"] - tip_pos) > 0.030

    class _SelectionCore:
        def detect_sc_pose(self, *_args, **_kwargs):
            return []

        def build_projection_matrix(self, *_args):
            return np.eye(3, 4)

    class _SelectionPolicy:
        _pc = _SelectionCore()

        def get_logger(self):
            return log

        def _build_views(self, _obs):
            return {"left": (None, np.eye(3), np.eye(4)), "right": (None, np.eye(3), np.eye(4))}

        def _tcp(self):
            return tip_pos - SC_TIP_IN_TCP_POS, np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(sc_controller, "sc_multiview_candidates", lambda *_: [target])

    perceived = sc_controller.perceive_sc_port_pose(_SelectionPolicy(), None, object())

    assert perceived is not None
    np.testing.assert_allclose(perceived[0], target["X"])
    assert sc_controller._sc_candidate_lateral_distance(target, tip_pos) == pytest.approx(0.009)


def test_port_selection_ranks_lateral_distance_not_full_3d_distance(monkeypatch):
    """The mouth most laterally aligned wins even when it is axially farther."""
    log = _RecordingLog()
    tip_pos = np.array([0.3, -0.2, 0.4], dtype=np.float64)
    Rp = np.diag([1.0, -1.0, -1.0])
    q_wxyz = matrix_to_quat(Rp)

    def candidate(lateral_m, axial_m):
        return {
            "X": tip_pos - Rp @ np.array([lateral_m, 0.0, axial_m]),
            "q_wxyz": q_wxyz,
            "score": 1.0,
            "reproj_px": 1.0,
            "width": _LABEL_WIDTH_M,
            "height": _LABEL_HEIGHT_M,
            "opening": SC_POSE_CONVENTION,
            "opening_residual_m": 0.0,
            "center_disagreement_m": 0.0,
        }

    laterally_aligned = candidate(0.002, 0.040)
    nearer_in_3d = candidate(0.009, 0.001)
    assert np.linalg.norm(nearer_in_3d["X"] - tip_pos) < np.linalg.norm(
        laterally_aligned["X"] - tip_pos
    )

    class _SelectionCore:
        def detect_sc_pose(self, *_args, **_kwargs):
            return []

        def build_projection_matrix(self, *_args):
            return np.eye(3, 4)

    class _SelectionPolicy:
        _pc = _SelectionCore()

        def get_logger(self):
            return log

        def _build_views(self, _obs):
            return {"left": (None, np.eye(3), np.eye(4)), "right": (None, np.eye(3), np.eye(4))}

        def _tcp(self):
            return tip_pos - SC_TIP_IN_TCP_POS, np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(
        sc_controller,
        "sc_multiview_candidates",
        lambda *_: [nearer_in_3d, laterally_aligned],
    )

    perceived = sc_controller.perceive_sc_port_pose(_SelectionPolicy(), None, object())

    assert perceived is not None
    np.testing.assert_allclose(perceived[0], laterally_aligned["X"])


def _select_gate_policy(log, tip_pos, candidates):
    """Minimal policy exercising only perceive_sc_port_pose's selection logic."""

    class _SelectionCore:
        def detect_sc_pose(self, *_args, **_kwargs):
            return []

        def build_projection_matrix(self, *_args):
            return np.eye(3, 4)

    class _SelectionPolicy:
        _pc = _SelectionCore()

        def get_logger(self):
            return log

        def _build_views(self, _obs):
            return {"left": (None, np.eye(3), np.eye(4)), "right": (None, np.eye(3), np.eye(4))}

        def _tcp(self):
            return tip_pos - SC_TIP_IN_TCP_POS, np.array([1.0, 0.0, 0.0, 0.0])

    return _SelectionPolicy()


def _real_field_candidate(tip_pos, *, centre_reproj_px=2.02, reproj_px=5.34):
    """2026-07-27: all 7 frames' 5-keypoint mean landed 5.30-5.38px against
    the 5.0px select gate while KP4 alone triangulated at ~2.0px."""
    return {
        "X": tip_pos.copy(),
        "q_wxyz": matrix_to_quat(np.eye(3)),
        "score": 1.0,
        "reproj_px": reproj_px,             # 5-keypoint mean
        "centre_reproj_px": centre_reproj_px,  # KP4 alone
        "width": SC_POSE_WIDTH_M,
        "height": SC_POSE_HEIGHT_M,
        "opening": SC_POSE_CONVENTION,
        "opening_residual_m": 0.0,
        "center_disagreement_m": 0.0,
    }


def test_select_gate_passes_on_a_good_centre_despite_a_noisy_corner_mean(monkeypatch):
    """The select gate must gate on centre_reproj_px (KP4 alone), not the
    5-keypoint mean the near-symmetric, viewpoint-ambiguous corners dominate.
    The 2026-07-27 field frame -- mean 5.30-5.38px, centre ~2.0px -- must
    return a pose, not None, and it clears the (now centre-based) gate
    cleanly, so no degrade is needed for this one."""
    log = _RecordingLog()
    tip_pos = np.array([0.0, 0.0, 0.10], dtype=np.float64)
    candidate = _real_field_candidate(tip_pos)

    monkeypatch.setattr(sc_controller, "sc_multiview_candidates", lambda *_: [candidate])

    perceived = sc_controller.perceive_sc_port_pose(
        _select_gate_policy(log, tip_pos, [candidate]), None, object()
    )

    assert perceived is not None
    _, _, reproj = perceived
    assert reproj == pytest.approx(2.02)
    assert not any("SC_PERCEPT_DEGRADED" in line for line in log.warn_lines)


def test_select_gate_degrades_when_nothing_clears_even_the_centre_gate(monkeypatch):
    """When even the centre reprojection misses the gate, the frame must
    still return the best-available pose (ranked by centre) rather than
    None, with SC_PERCEPT_DEGRADED naming the reason, the best centre and
    5-kp mean, and the candidate count."""
    log = _RecordingLog()
    tip_pos = np.array([0.0, 0.0, 0.10], dtype=np.float64)
    candidate = _real_field_candidate(tip_pos, centre_reproj_px=6.4, reproj_px=8.1)

    monkeypatch.setattr(sc_controller, "sc_multiview_candidates", lambda *_: [candidate])

    perceived = sc_controller.perceive_sc_port_pose(
        _select_gate_policy(log, tip_pos, [candidate]), None, object()
    )

    assert perceived is not None
    _, _, reproj = perceived
    assert reproj == pytest.approx(6.4)
    assert any(
        "SC_PERCEPT_DEGRADED" in line
        and "best_centre_reproj_px=6.40" in line
        and "best_mean5_reproj_px=8.10" in line
        and "candidates=1" in line
        for line in log.warn_lines
    )


def test_strict_perception_restores_the_select_gate_abort(monkeypatch):
    monkeypatch.setattr(sc_controller, "SC_STRICT_PERCEPTION", True)
    log = _RecordingLog()
    tip_pos = np.array([0.0, 0.0, 0.10], dtype=np.float64)
    candidate = _real_field_candidate(tip_pos, centre_reproj_px=9.0, reproj_px=12.0)

    monkeypatch.setattr(sc_controller, "sc_multiview_candidates", lambda *_: [candidate])

    perceived = sc_controller.perceive_sc_port_pose(
        _select_gate_policy(log, tip_pos, [candidate]), None, object()
    )

    assert perceived is None
    assert not any("SC_PERCEPT_DEGRADED" in line for line in log.warn_lines)
    assert not any("SC_PERCEPT_DEGRADED" in line for line in log.warn_lines)


def test_lateral_handoff_gate_is_not_softened_by_degraded_perception(monkeypatch):
    """SC_MAX_HANDOFF_LATERAL_M is a wrong-port defence (a wrong-port
    insertion scores -12, far worse than aborting) and must stay fatal even
    though the select/consensus gates above it now degrade instead of
    rejecting.  A clean-reprojecting candidate that is simply too far away
    laterally must still make perceive_sc_port_pose return None."""
    log = _RecordingLog()
    tip_pos = np.array([0.3, -0.2, 0.4], dtype=np.float64)
    far_candidate = {
        "X": tip_pos + np.array([0.020, 0.0, 0.0]),  # 20mm, beyond the 10mm gate
        "q_wxyz": matrix_to_quat(np.eye(3)),
        "score": 1.0,
        "reproj_px": 1.0,
        "centre_reproj_px": 1.0,
        "width": SC_POSE_WIDTH_M,
        "height": SC_POSE_HEIGHT_M,
        "opening": SC_POSE_CONVENTION,
        "opening_residual_m": 0.0,
        "center_disagreement_m": 0.0,
    }

    monkeypatch.setattr(
        sc_controller, "sc_multiview_candidates", lambda *_: [far_candidate]
    )

    perceived = sc_controller.perceive_sc_port_pose(
        _select_gate_policy(log, tip_pos, [far_candidate]), None, object()
    )

    assert perceived is None
    assert any("lateral" in line and "handoff gate" in line for line in log.warn_lines)


# --- physical-mouth five-keypoint multiview geometry --------------------------

_LABEL_WIDTH_M = SC_POSE_WIDTH_M
_LABEL_HEIGHT_M = SC_POSE_HEIGHT_M
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


def _stereo_views(width_m, height_m, *, center_xyz=None):
    """Per-camera detections of four corners plus the explicit mouth centre."""
    K = np.array([[_FX, 0.0, _CX], [0.0, _FX, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)
    points = np.array(
        [
            [+width_m / 2.0, +height_m / 2.0, _PORT_DEPTH_M],
            [-width_m / 2.0, +height_m / 2.0, _PORT_DEPTH_M],
            [-width_m / 2.0, -height_m / 2.0, _PORT_DEPTH_M],
            [+width_m / 2.0, -height_m / 2.0, _PORT_DEPTH_M],
            (
                [0.0, 0.0, _PORT_DEPTH_M]
                if center_xyz is None
                else np.asarray(center_xyz, dtype=np.float64)
            ),
        ],
        dtype=np.float64,
    )

    per_cam = {}
    for name, shift in (("cam_a", 0.0), ("cam_b", -_STEREO_BASELINE_M)):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = shift
        P = _StereoCore.build_projection_matrix(K, T)
        kps = []
        for point in points:
            x = P @ np.array([*point, 1.0], dtype=np.float64)
            kps.append([x[0] / x[2], x[1] / x[2]])
        per_cam[name] = [
            {"kps": np.array(kps, dtype=np.float64), "conf": 0.9, "K": K, "T": T, "P": P}
        ]
    return per_cam


def test_legacy_four_keypoint_checkpoint_fails_closed():
    log = _RecordingLog()

    class _LegacyCore:
        def detect_sc_pose(self, *_args, **_kwargs):
            return [{"kps": np.zeros((4, 2)), "conf": 0.99}]

        @staticmethod
        def build_projection_matrix(K, T):
            return np.asarray(K) @ np.asarray(T)[:3, :4]

    class _LegacyPolicy:
        _pc = _LegacyCore()

        def get_logger(self):
            return log

        def _build_views(self, _obs):
            return {
                "left": (np.zeros((8, 8, 3)), np.eye(3), np.eye(4)),
                "right": (np.zeros((8, 8, 3)), np.eye(3), np.eye(4)),
            }

    assert (
        sc_controller.perceive_sc_port_pose(_LegacyPolicy(), None, object()) is None
    )
    assert sum(
        "requires exactly 5 keypoints" in line for line in log.error_lines
    ) == 2


def test_size_gate_admits_the_physical_mouth_rectangle():
    candidates = sc_controller.sc_multiview_candidates(
        _stereo_policy(), _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M)
    )

    assert candidates, "the model's physical mouth geometry must not be rejected"
    assert candidates[0]["width"] == pytest.approx(_LABEL_WIDTH_M, abs=1e-5)
    assert candidates[0]["height"] == pytest.approx(_LABEL_HEIGHT_M, abs=1e-5)
    fitted = candidates[0]["kp_3d"]
    fitted_width = np.linalg.norm(
        (fitted[0] + fitted[3]) * 0.5 - (fitted[1] + fitted[2]) * 0.5
    )
    fitted_height = np.linalg.norm(
        (fitted[0] + fitted[1]) * 0.5 - (fitted[2] + fitted[3]) * 0.5
    )
    assert fitted_width == pytest.approx(_LABEL_WIDTH_M, abs=1e-9)
    assert fitted_height == pytest.approx(_LABEL_HEIGHT_M, abs=1e-9)
    np.testing.assert_allclose(candidates[0]["X"], [0.0, 0.0, _PORT_DEPTH_M], atol=1e-8)


def test_multiview_candidates_carry_centre_reproj_px():
    """Every candidate must expose centre_reproj_px alongside reproj_px --
    perceive_sc_port_pose's select gate depends on it being present."""
    candidates = sc_controller.sc_multiview_candidates(
        _stereo_policy(), _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M)
    )

    assert candidates
    assert "centre_reproj_px" in candidates[0]
    # A noiseless detection reprojects KP4 essentially exactly.
    assert candidates[0]["centre_reproj_px"] == pytest.approx(0.0, abs=1e-6)


def test_centre_reproj_px_is_correct_and_roll_invariant():
    """_centre_reproj_px measures KP4 alone and must be identical under every
    one of the 4 cyclic corner relabellings _rolled_kps can apply -- that
    roll-invariance is precisely why it is a trustworthy ranking key (see its
    docstring), unlike the 5-keypoint _mean_reproj_px the ambiguous corners
    dominate."""
    views = _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M)
    policy = _stereo_policy()
    picks = [views["cam_a"][0], views["cam_b"][0]]
    kp_3d = np.array(
        [
            [+_LABEL_WIDTH_M / 2.0, +_LABEL_HEIGHT_M / 2.0, _PORT_DEPTH_M],
            [-_LABEL_WIDTH_M / 2.0, +_LABEL_HEIGHT_M / 2.0, _PORT_DEPTH_M],
            [-_LABEL_WIDTH_M / 2.0, -_LABEL_HEIGHT_M / 2.0, _PORT_DEPTH_M],
            [+_LABEL_WIDTH_M / 2.0, -_LABEL_HEIGHT_M / 2.0, _PORT_DEPTH_M],
            [0.0, 0.0, _PORT_DEPTH_M],
        ],
        dtype=np.float64,
    )

    baseline = sc_controller._centre_reproj_px(policy, picks, kp_3d)
    assert baseline == pytest.approx(0.0, abs=1e-9)

    for roll in range(4):
        rolled_picks = [
            dict(pick, kps=sc_controller._rolled_kps(pick["kps"], roll))
            for pick in picks
        ]
        rolled = sc_controller._centre_reproj_px(policy, rolled_picks, kp_3d)
        assert rolled == pytest.approx(baseline, abs=1e-9)


def test_explicit_center_keypoint_is_the_motion_target():
    expected_center = np.array([0.0013, -0.0007, _PORT_DEPTH_M])
    candidates = sc_controller.sc_multiview_candidates(
        _stereo_policy(),
        _stereo_views(
            _LABEL_WIDTH_M,
            _LABEL_HEIGHT_M,
            center_xyz=expected_center,
        ),
    )

    assert candidates
    np.testing.assert_allclose(candidates[0]["X"], expected_center, atol=1e-8)
    np.testing.assert_allclose(
        candidates[0]["corner_X"], [0.0, 0.0, _PORT_DEPTH_M], atol=1e-8
    )
    assert candidates[0]["center_disagreement_m"] == pytest.approx(
        np.linalg.norm(expected_center[:2]), abs=1e-8
    )


def test_joint_fit_is_retained_as_diagnostic_under_corner_noise():
    views = _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M)
    noise = {
        "cam_a": np.array(
            [[1.0, -0.5], [-0.5, 0.5], [-1.0, -0.5], [0.5, 0.5]]
        ),
        "cam_b": np.array(
            [[-0.5, 0.25], [0.25, -0.25], [0.5, 0.25], [-0.25, -0.25]]
        ),
    }
    for camera, delta in noise.items():
        views[camera][0]["kps"][:4] += delta

    candidate = sc_controller.sc_multiview_candidates(
        _stereo_policy(), views
    )[0]
    # The explicit centre drives motion. The rigid corner rectangle remains a
    # diagnostic and cannot override that centre.
    fitted = candidate["fit_kp_3d"]
    assert fitted is not None
    fitted_width = np.linalg.norm(
        (fitted[0] + fitted[3]) * 0.5 - (fitted[1] + fitted[2]) * 0.5
    )
    fitted_height = np.linalg.norm(
        (fitted[0] + fitted[1]) * 0.5 - (fitted[2] + fitted[3]) * 0.5
    )

    assert fitted_width == pytest.approx(_LABEL_WIDTH_M, abs=1e-9)
    assert fitted_height == pytest.approx(_LABEL_HEIGHT_M, abs=1e-9)
    assert candidate["reproj_px"] == candidate["raw_reproj_px"]
    assert candidate["fit_reproj_px"] is not None
    assert np.isfinite(candidate["reproj_px"])
    assert candidate["fit_reproj_px"] != pytest.approx(candidate["reproj_px"])


def test_size_gate_survives_a_realistic_underestimate_of_the_short_axis():
    # The broad corruption floor leaves room for triangulation shrinkage.
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
    # Stub the vision priming: without a transform installed the geometry runs
    # on the fixed-grasp constant, which is exactly what this fixture encodes.
    monkeypatch.setattr(sc_controller, "prime_sc_plug_pose", lambda *_: True)

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
def _run_sc_with_handoff_depth(monkeypatch, depth_m, log, prime=None):
    """Drive run_sc_insertion with the tip placed at a chosen depth.

    ``prime`` replaces the vision priming; the default stubs it to succeed
    WITHOUT installing a measured transform, so the handoff geometry runs on
    the fixed-grasp constant exactly as these fixtures were built for.
    """
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
    monkeypatch.setattr(sc_controller, "prime_sc_plug_pose",
                        prime if prime is not None else (lambda *_: True))
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
# Measured plug pose (prime_sc_plug_pose) wiring.
# ---------------------------------------------------------------------------
def test_run_sc_insertion_refuses_when_plug_pose_priming_fails(monkeypatch):
    """No measured tip, no insertion: the run must stop before perception."""
    log = _RecordingLog()

    class _RunPolicy(_StubPolicy):
        def get_logger(self):
            return log

    def _never(*_a, **_k):
        raise AssertionError("perception must not run after a priming failure")

    monkeypatch.setattr(sc_controller, "perceive_sc_port_pose_consensus", _never)
    monkeypatch.setattr(sc_controller, "prime_sc_plug_pose", lambda *_: False)

    result = sc_controller.run_sc_insertion(
        _RunPolicy(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])), _StubTask(),
        lambda: object(), None, lambda *_: None,
    )

    assert result is False
    assert any("no fixed-grasp fallback" in line for line in log.error_lines)


def _priming_that_moves_the_tip_to(port_pos, Rp, depth_m):
    """A prime stub that installs a MEASURED transform placing the tip at
    ``depth_m`` along the insertion axis, with the constant's orientation."""

    def _prime(policy, *_a):
        tcp_pos, tcp_quat = policy._tcp()
        want_tip = port_pos + Rp[:, 2] * depth_m + Rp[:, 0] * 0.0035
        _, R_tip = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
        policy._sc_grasp_transform = sc_controller.solve_tip_in_tcp(
            tcp_pos, tcp_quat, want_tip, R_tip
        )
        return True

    return _prime


def test_handoff_blames_the_measurement_not_the_constant_when_primed(monkeypatch):
    """Primed runs must attribute an impossible depth to macro/measurement --
    telling the operator to re-solve SC_TIP_IN_TCP would send them down a road
    that no longer exists."""
    Rp, _ = _field_frames()
    port_pos = np.array([-0.32466, 0.12952, 0.03032], dtype=np.float64)
    log = _RecordingLog()

    # Constant says 6.99 mm OUTSIDE (would pass); measurement says INSIDE.
    result, seated = _run_sc_with_handoff_depth(
        monkeypatch, -0.00699, log,
        prime=_priming_that_moves_the_tip_to(port_pos, Rp, +0.00699),
    )

    assert result is False
    assert not seated
    line = "\n".join(log.error_lines)
    assert "INSIDE the port" in line
    assert "MEASURED" in line
    assert "SC_TIP_IN_TCP_POS" not in line, "must not blame the retired constant"


def test_handoff_trusts_the_measured_tip_over_a_lying_constant(monkeypatch):
    """The 2026-07-25 phantom: the CONSTANT computed the tip +6.99 mm inside the
    port.  With the measured transform saying the plug is really outside the
    mouth, the run must proceed -- this is the exact failure the model fixes."""
    Rp, _ = _field_frames()
    port_pos = np.array([-0.32466, 0.12952, 0.03032], dtype=np.float64)

    result, seated = _run_sc_with_handoff_depth(
        monkeypatch, +0.00699, _RecordingLog(),
        prime=_priming_that_moves_the_tip_to(port_pos, Rp, -0.00699),
    )

    assert result is True
    assert seated


def test_sc_tip_helpers_round_trip_through_the_measured_transform():
    tcp_pos = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    Rp, R_tip = _field_frames()
    tcp_quat = matrix_to_quat(R_tip @ quat_to_matrix(sc_controller.SC_TIP_IN_TCP_QUAT).T)
    policy = _StubPolicy(tcp_pos, tcp_quat)

    measured_tip = tcp_pos + np.array([0.010, 0.020, -0.058], dtype=np.float64)
    policy._sc_grasp_transform = sc_controller.solve_tip_in_tcp(
        tcp_pos, tcp_quat, measured_tip, R_tip
    )

    tip_pos, tip_rot = sc_controller.sc_tip_from_tcp(policy, tcp_pos, tcp_quat)
    assert np.allclose(tip_pos, measured_tip, atol=1e-12)
    assert np.allclose(tip_rot, R_tip, atol=1e-12)
    const_tip, _ = sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
    assert not np.allclose(tip_pos, const_tip, atol=1e-4), \
        "the measured transform must actually differ from the constant here"

    # The inverse must consume the SAME transform, or every commanded pose
    # would be offset by the constant-vs-measured disagreement.
    back_tcp_pos, back_R_tcp = sc_controller.sc_tcp_pose_for_tip(policy, tip_pos, tip_rot)
    assert np.allclose(back_tcp_pos, tcp_pos, atol=1e-12)
    assert np.allclose(back_R_tcp, quat_to_matrix(tcp_quat), atol=1e-12)

    # Without a transform both helpers must reproduce the constant behaviour.
    bare = _StubPolicy(tcp_pos, tcp_quat)
    tip_bare, _ = sc_controller.sc_tip_from_tcp(bare, tcp_pos, tcp_quat)
    assert np.allclose(tip_bare, const_tip, atol=1e-12)


def test_prime_sc_plug_pose_fails_closed_when_no_estimate(monkeypatch):
    import aic_model.v50_controller as v50_controller_mod

    log = _RecordingLog()

    class _Clock:
        def now(self):
            class _Now:
                nanoseconds = int(2.0e9)

            return _Now()

    class _PrimeNode(_FakeNode):
        def get_clock(self):
            return _Clock()

    class _PrimePolicy(_StubPolicy):
        def __init__(self):
            super().__init__(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
            self._parent_node = _PrimeNode()
            # a run that primed before must not inherit last run's grasp
            self._sc_grasp_transform = ("stale", "stale")

        def get_logger(self):
            return log

        def _enforce_action_deadline(self, _move_robot):
            pass

    class _View:
        def __init__(self, name):
            self.camera_name = name
            self.stamp_s = 1.9

    class _RefusingEstimator:
        min_keypoint_confidence = 0.35
        last_failure_reason = "insufficient_detected_views:0<2"

        def detect_views(self, _views):
            return []

        def estimate_multiview(self, *_a, **_k):
            return None

    policy = _PrimePolicy()
    policy._sc_plug_estimator = _RefusingEstimator()
    monkeypatch.setattr(v50_controller_mod, "_observation_stamp_s", lambda _o: 1.9)
    monkeypatch.setattr(v50_controller_mod, "_plug_views_from_observation",
                        lambda _p, _o: [_View("left_camera"), _View("right_camera")])

    result = sc_controller.prime_sc_plug_pose(policy, lambda: object(), None)

    assert result is False
    assert policy._sc_grasp_transform is None, \
        "a failed prime must clear any stale transform, never keep it"
    line = "\n".join(log.error_lines)
    assert "PLUG_POSE_REJECT" in line
    assert "insufficient_detected_views" in line
    assert "no_fixed_grasp_fallback=true" in line


def test_configure_sc_plug_pose_refuses_without_weights(monkeypatch):
    import aic_model.sc_plug_pose as sc_plug_pose_mod

    log = _RecordingLog()

    class _Policy(_StubPolicy):
        def get_logger(self):
            return log

    monkeypatch.setattr(sc_plug_pose_mod, "default_sc_plug_pose_weights", lambda: None)
    policy = _Policy(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    assert sc_controller.configure_sc_plug_pose(policy) is False
    assert getattr(policy, "_sc_plug_estimator", None) is None
    assert any("AIC_SC_PLUG_POSE_WEIGHTS" in line for line in log.error_lines)


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
    monkeypatch.setattr(sc_controller, "prime_sc_plug_pose", lambda *_: True)
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


# ---------------------------------------------------------------------------
# Keypoint corner-order (roll) resolution.
# ---------------------------------------------------------------------------
def _roll_one_camera(per_cam, cam, roll):
    """Relabel one camera's corners, as a weak detection does in the field."""
    out = dict(per_cam)
    det = dict(out[cam][0])
    det["kps"] = np.asarray(det["kps"], dtype=np.float64).copy()
    det["kps"][:4] = np.roll(det["kps"][:4], -roll, axis=0)
    out[cam] = [det]
    return out


@pytest.mark.parametrize("roll", [1, 2, 3])
def test_a_relabelled_camera_still_triangulates(roll):
    """The 2026-07-25 18:46 failure, reproduced for every cyclic relabelling.

    Index-to-index matching pairs corner 0 with corner 2 and reprojects at
    ~11.5 px against a 5.0 px gate. Searching the relabellings recovers the
    true rectangle.
    """
    views = _roll_one_camera(
        _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M), "cam_b", roll)

    candidates = sc_controller.sc_multiview_candidates(_stereo_policy(), views)

    assert candidates, f"a camera labelled from corner {roll} must still resolve"
    best = candidates[0]
    assert best["reproj_px"] < 1.0, "the recovered correspondence must reproject cleanly"
    assert best["width"] == pytest.approx(_LABEL_WIDTH_M, abs=1e-5)
    assert best["height"] == pytest.approx(_LABEL_HEIGHT_M, abs=1e-5)


def test_index_to_index_matching_really_does_fail_without_the_search(monkeypatch):
    # Pin that the parametrised test above is testing something real: with the
    # search off, a relabelled camera is exactly the field failure.
    monkeypatch.setattr(sc_controller, "SC_KEYPOINT_ROLL_SEARCH", False)
    views = _roll_one_camera(
        _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M), "cam_b", 2)

    candidates = sc_controller.sc_multiview_candidates(_stereo_policy(), views)

    assert not candidates or candidates[0]["reproj_px"] > 5.0, (
        "without the search a 180-degree relabelling must blow the select gate"
    )


def test_correct_correspondence_is_left_alone():
    # No relabelling needed: the search must not disturb a clean detection.
    clean = _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M)

    candidates = sc_controller.sc_multiview_candidates(_stereo_policy(), clean)

    assert candidates
    assert candidates[0]["reproj_px"] < 1e-6
    assert candidates[0]["width"] == pytest.approx(_LABEL_WIDTH_M, abs=1e-5)


def test_width_is_normalised_to_the_long_axis():
    # A reference camera that starts a quarter turn round would otherwise
    # transpose width and height in the candidate score.
    views = _roll_one_camera(
        _roll_one_camera(_stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M), "cam_a", 1),
        "cam_b", 1)

    candidates = sc_controller.sc_multiview_candidates(_stereo_policy(), views)

    assert candidates
    assert candidates[0]["width"] > candidates[0]["height"]
    assert candidates[0]["width"] == pytest.approx(_LABEL_WIDTH_M, abs=1e-5)


def test_roll_usage_is_reported():
    log = _RecordingLog()
    views = _roll_one_camera(
        _stereo_views(_LABEL_WIDTH_M, _LABEL_HEIGHT_M), "cam_b", 2)

    sc_controller.sc_multiview_candidates(_stereo_policy(log), views)

    assert any("SC_KEYPOINT_ROLL" in line for line in log.info_lines), (
        "an unstable detector corner order must be visible in the logs"
    )
