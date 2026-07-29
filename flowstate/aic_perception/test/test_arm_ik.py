"""Tests for the UR5e forward/inverse kinematics reachability gate."""

import math

import numpy as np
import pytest

from aic_perception.arm_ik import (
    JOINT_LIMITS,
    UR5eArm,
    _DH_A,
    _DH_ALPHA,
    _DH_D,
    _dh_matrix,
    capsule_intersects_camera_view,
    lift_into_joint_limits,
)
from aic_perception.board_stage2 import Transform


_TEST_JOINT_WINDOW_DEG = np.array(
    [
        [-180.0, 180.0],
        [-190.0, -20.0],
        [-150.0, 150.0],
        [-170.0, 100.0],
        [-130.0, 130.0],
        [-90.0, 190.0],
    ],
    dtype=float,
)
_TEST_JOINT_WINDOW = np.radians(_TEST_JOINT_WINDOW_DEG)


class _Camera:
    K = np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ]
    )
    width = 100
    height = 100


def test_capsule_view_gate_rejects_camera_origin_inside_capsule():
    assert capsule_intersects_camera_view(
        [-0.10, 0.0, -0.02],
        [0.10, 0.0, -0.02],
        0.03,
        _Camera(),
    )


def test_capsule_view_gate_clips_crossing_segment_at_near_plane():
    assert capsule_intersects_camera_view(
        [0.02, 0.0, -0.20],
        [0.02, 0.0, 0.20],
        0.01,
        _Camera(),
    )
    assert not capsule_intersects_camera_view(
        [0.20, 0.0, -0.20],
        [0.20, 0.0, 0.02],
        0.01,
        _Camera(),
    )


def test_capsule_view_gate_can_limit_the_test_to_a_sector():
    segment = ([0.03, 0.0, 0.20], [0.03, 0.0, 0.30])
    assert capsule_intersects_camera_view(*segment, 0.005, _Camera())
    assert not capsule_intersects_camera_view(
        *segment,
        0.005,
        _Camera(),
        bounds_px=(45.0, 45.0, 55.0, 55.0),
    )


def _random_joints(rng, n):
    # A realistic band (avoid full +/-2pi wrap that trivially aliases).
    band = np.array(
        [
            [-3.14, 3.14],
            [-3.14, 0.2],
            [-2.9, 2.9],
            [-3.14, 3.14],
            [-3.14, 3.14],
            [-3.14, 3.14],
        ]
    )
    return [np.array([rng.uniform(a, b) for a, b in band]) for _ in range(n)]


def test_zero_config_matches_ur5e_geometry():
    arm = UR5eArm()
    flange = arm.fk_flange([0, 0, 0, 0, 0, 0])
    # UR5e is near full horizontal extension at the zero configuration.
    assert np.linalg.norm(flange.translation) == pytest.approx(0.852, abs=0.01)
    # Rotation stays a proper orthonormal matrix.
    assert np.linalg.norm(flange.rotation @ flange.rotation.T - np.eye(3)) < 1e-9


def test_mjcf_chain_is_the_classical_ur5e_dh_chain():
    """The MJCF link chain and the DH parameters describe the same arm.

    The closed-form solver is derived on the DH chain and applied directly to
    poses expressed in the MJCF model frame, with no adapter transform at either
    end.  That is only valid because the two products are identical -- assert it
    rather than assume it.
    """
    arm = UR5eArm()
    rng = np.random.default_rng(11)
    for q in _random_joints(rng, 25) + [np.zeros(6)]:
        dh = np.eye(4)
        for i in range(6):
            dh = dh @ _dh_matrix(q[i], _DH_A[i], _DH_D[i], _DH_ALPHA[i])
        mjcf = arm.fk_flange(q)
        assert np.allclose(mjcf.rotation, dh[:3, :3], atol=1e-12)
        assert np.allclose(mjcf.translation, dh[:3, 3], atol=1e-12)


def test_fk_ik_round_trip_recovers_every_reachable_pose_exactly():
    """A closed-form solver has no seed dependence and no local minima: every
    pose the arm can strike is recovered, to machine precision."""
    arm = UR5eArm()
    rng = np.random.default_rng(1)
    for q_true in _random_joints(rng, 120):
        target = arm.fk(q_true)
        q = arm.solve(target, seed=q_true + rng.normal(0, 0.1, 6))
        assert q is not None, "closed-form IK missed a self-generated pose"
        pos, ori = arm.fk_residual(q, target)
        assert pos < 1e-9 and ori < 1e-9
        assert np.all(q >= JOINT_LIMITS[:, 0] - 1e-9)
        assert np.all(q <= JOINT_LIMITS[:, 1] + 1e-9)


def test_solve_without_a_seed_is_just_as_complete():
    """No seed, no penalty -- this is what makes the gate's "unreachable"
    verdict trustworthy rather than a search that gave up."""
    arm = UR5eArm()
    rng = np.random.default_rng(2)
    for q_true in _random_joints(rng, 120):
        assert arm.reachable(arm.fk(q_true))


def test_solve_all_enumerates_the_branches_and_each_one_is_exact():
    """The eight UR configurations (shoulder x wrist x elbow) are enumerated, so
    a pose is only rejected when *no* branch reaches it."""
    arm = UR5eArm()
    rng = np.random.default_rng(3)
    counts = []
    for q_true in _random_joints(rng, 40):
        target = arm.fk(q_true)
        solutions = arm.solve_all(target)
        counts.append(len(solutions))
        for q in solutions:
            pos, ori = arm.fk_residual(q, target)
            assert pos < 1e-9 and ori < 1e-9
    # Interior poses have several distinct arm configurations, not just one.
    assert max(counts) == 8
    assert float(np.mean(counts)) > 4.0


def test_solve_picks_the_branch_nearest_the_seed():
    """The skill passes the live joint state; the returned configuration should
    be the one the controller would actually adopt."""
    arm = UR5eArm()
    q_true = np.array([0.4, -1.3, 1.5, -1.7, -1.5, 0.2])
    target = arm.fk(q_true)
    assert len(arm.solve_all(target)) > 1
    assert np.allclose(arm.solve(target, seed=q_true), q_true, atol=1e-9)


def test_solve_preserves_the_nearest_in_limit_coterminal_branch():
    """A +226 deg live joint must not be reported as a -134 deg target.

    Those angles produce the same Cartesian pose, but commanding one from the
    other is a physical full revolution.  This is the wraparound that produced
    the observed violent Move Robot transit.
    """
    arm = UR5eArm()
    q_true = np.array([0.4, -1.3, 1.5, -1.7, -1.5, math.radians(226.0)])
    target = arm.fk(q_true)
    solved = arm.solve(target, seed=q_true)
    assert solved is not None
    assert np.allclose(solved, q_true, atol=1e-9)
    assert abs(solved[5] - q_true[5]) < 1e-9


def test_solve_ranked_exposes_every_forearm_clear_finite_branch():
    arm = UR5eArm()
    seed = np.array([0.4, -1.3, 1.5, -1.7, -1.5, math.radians(226.0)])
    target = arm.fk(seed)

    ranked = arm.solve_ranked(target, seed)

    assert len(ranked) > 1
    assert np.allclose(ranked[0], arm.solve(target, seed))
    assert all(np.all(q >= JOINT_LIMITS[:, 0] - 1e-9) for q in ranked)
    assert all(np.all(q <= JOINT_LIMITS[:, 1] + 1e-9) for q in ranked)
    # At least the matching wrist branch must retain the +226-degree finite
    # representation rather than collapse to its -134-degree principal angle.
    assert ranked[0][5] == pytest.approx(seed[5])


def test_solve_ranked_can_reuse_an_already_enumerated_branch_set():
    arm = UR5eArm()
    seed = np.array([0.4, -1.3, 1.5, -1.7, -1.5, math.radians(226.0)])
    target = arm.fk(seed)
    branches = arm.solve_all(target)

    expected = arm.solve_ranked(target, seed)
    reused = arm.solve_ranked(target, seed, solutions=branches)

    assert len(reused) == len(expected)
    for actual, wanted in zip(reused, expected):
        assert actual == pytest.approx(wanted)


def test_default_self_clearance_tracks_the_hardware_accepted_branch():
    # Move Robot executed a branch measured at 122 mm.  The analytic gate keeps
    # a small margin below it instead of hiding that branch at the old 140 mm.
    assert UR5eArm().min_self_clearance_m == pytest.approx(0.140)


def test_optional_joint_window_maps_coterminal_wrist_values():
    # The generic IK API can still represent a solution inside a caller-owned
    # window, but production SC target selection no longer installs one.
    raw = np.radians([20.0, -80.0, 70.0, -100.0, 257.9, 341.8])
    mapped = lift_into_joint_limits(
        raw,
        _TEST_JOINT_WINDOW,
        reference=np.radians([-9.15, -77.59, -95.39, -97.02, 90.01, 170.84]),
    )
    assert mapped is not None
    assert np.degrees(mapped[4]) == pytest.approx(-102.1)
    assert np.degrees(mapped[5]) == pytest.approx(-18.2)
    assert np.all(mapped >= _TEST_JOINT_WINDOW[:, 0])
    assert np.all(mapped <= _TEST_JOINT_WINDOW[:, 1])


def test_solve_ranked_can_honor_an_optional_caller_joint_window():
    arm = UR5eArm()
    seed = np.radians([-9.15, -77.59, -95.39, -97.02, 90.01, 170.84])
    target_joints = np.radians([20.0, -80.0, 70.0, -100.0, -102.1, -18.2])
    target = arm.fk(target_joints)

    ranked = arm.solve_ranked(
        target,
        seed,
        joint_limits=_TEST_JOINT_WINDOW,
    )

    assert ranked
    assert all(np.all(q >= _TEST_JOINT_WINDOW[:, 0]) for q in ranked)
    assert all(np.all(q <= _TEST_JOINT_WINDOW[:, 1]) for q in ranked)
    assert any(np.allclose(q, target_joints, atol=1e-9) for q in ranked)


def test_calibration_recovers_tool_offset_and_is_exact_at_the_sample():
    true_tool = Transform(np.eye(3), np.array([0.0, 0.0, 0.16]))
    arm = UR5eArm(flange_T_tcp=true_tool)
    q = np.array([0.3, -1.2, 1.4, -1.6, -1.57, 0.2])
    measured = arm.fk(q)
    calibrated = UR5eArm.calibrated_from(q, measured)
    assert np.linalg.norm(
        calibrated.flange_T_tcp.translation - true_tool.translation
    ) < 1e-9
    pos, ori = calibrated.fk_residual(q, measured)
    assert pos < 1e-9 and ori < 1e-9


def test_autocalibrate_recovers_a_flipped_base_convention():
    """The workcell base_link classically differs from the kinematic base by a
    180-deg-about-Z flip; autocalibrate must detect it from one sample and then
    judge reachability correctly in the base_link frame."""
    import math
    from aic_perception.arm_ik import _rot_z

    tool_true = Transform(np.eye(3), np.array([0.0, 0.0, 0.20]))
    truth = UR5eArm(flange_T_tcp=tool_true)  # model frame, base=identity
    base_link_T_model = Transform(_rot_z(math.pi), np.zeros(3))

    q = [0.5, -1.2, 1.5, -1.8, -1.57, 0.3]
    base_link_tcp = base_link_T_model.compose(truth.fk_flange(q).compose(tool_true))

    # Identity assumption yields a wildly implausible tool offset (the bug).
    naive = UR5eArm.calibrated_from(q, base_link_tcp)
    assert np.linalg.norm(naive.flange_T_tcp.translation) > 0.4

    arm, desc = UR5eArm.autocalibrate(q, base_link_tcp)
    assert arm is not None, desc
    assert np.allclose(arm.base.rotation, _rot_z(math.pi))
    assert np.allclose(arm.flange_T_tcp.translation, [0.0, 0.0, 0.20], atol=1e-6)

    # Reachability now correct in base_link: a real pose is reachable, far isn't.
    q2 = [-0.3, -1.5, 1.7, -1.6, -1.4, 0.0]
    reachable_pose = base_link_T_model.compose(
        truth.fk_flange(q2).compose(tool_true)
    )
    assert arm.reachable(reachable_pose)
    assert not arm.reachable(Transform(np.eye(3), np.array([2.0, 0.0, 0.5])))


def test_high_survey_poses_are_out_of_reach_but_the_close_ones_are_not():
    """The NIC search's far/high candidates are genuinely unreachable.

    Framing all five 145 mm cards in the centre camera wants a 0.7-1.25 m
    standoff, and the board sits at the height of the arm's own base -- so the
    far half of that sweep asks the TCP to hover a metre above the base while
    looking *down*, which no UR5e configuration does.  The gate must say so
    (that is why the search now gates every framed candidate, taking the
    farthest one that is genuinely reachable) and must still accept the close
    end of the sweep.
    """
    arm = UR5eArm(flange_T_tcp=Transform(np.eye(3), np.array([0.0, 0.0, 0.197])))
    looking_down = np.diag([1.0, -1.0, -1.0])
    over_the_board = np.array([-0.40, 0.22, 0.0])
    heights = [
        z
        for z in np.arange(0.30, 1.10, 0.05)
        if arm.reachable(Transform(looking_down, over_the_board + [0, 0, z]))
    ]
    assert heights, "a downward survey view over the board must exist somewhere"
    # One contiguous band that ends well below the far standoffs the NIC search
    # generates -- there is no second, higher window to reach for.
    assert min(heights) < 0.35 and 0.55 < max(heights) < 0.70


def test_wrist_camera_keep_out_rejects_a_pose_the_workcell_planner_refused():
    """Ground truth from a real run, not a synthetic case.

    For the published survey pose (-0.1001, 0.4162, 0.6859) the workcell planner
    reported *every* IK solution colliding -- ``robot.forearm_link`` against
    ``left_camera.camera_link`` -- and the move failed outright.  Its four
    reported configurations are this module's two branches, wrapped +/-360 deg;
    that they reproduce the published TCP to 0 mm also pins the whole model (DH
    chain, base=Rz180, 197.1 mm tool) against the real robot.

    A purely kinematic gate calls the pose reachable, which is exactly how it got
    published.  With the wrist cameras registered as flange probes it must not.
    """
    from dataclasses import replace
    from aic_perception.arm_ik import _rot_z

    tool = Transform(np.eye(3), np.array([0.0, 0.0, 0.1971]))
    arm = UR5eArm(flange_T_tcp=tool, base=Transform(_rot_z(math.pi), np.zeros(3)))
    planner_configs_deg = [
        [-77.4232, -98.0038, -24.1589, -133.821, 118.235, -75.7524],
        [-77.4232, -98.0038, -24.1589, 226.179, 118.235, -75.7524],
        [-77.4232, -98.0038, -24.1589, -133.821, -241.765, 284.248],
        [282.577, -121.178, 24.1589, -158.964, -241.765, 284.248],
    ]
    published = np.array([-0.1001, 0.4162, 0.6859])
    for config in planner_configs_deg:
        assert np.allclose(
            arm.fk(np.radians(config)).translation, published, atol=1e-4
        ), "model disagrees with the robot the planner was driving"

    # The wrist cameras, placed off the flange exactly as the production rig has
    # them (the extrinsics the skill already recovers from the permitted TF).
    from test_board_stage2 import _production_camera_rig

    _cameras, tcp_T_cam, _grippers = _production_camera_rig()
    gated = replace(
        arm,
        flange_T_probes=tuple(tool.compose(e) for e in tcp_T_cam.values()),
    )

    pose = arm.fk(np.radians(planner_configs_deg[0]))
    assert arm.reachable(pose), "kinematically it does solve -- that was the trap"
    assert not gated.reachable(pose), "gate must refuse what the planner refuses"
    # Every branch is inside the keep-out, which is why the planner had nothing.
    assert all(
        gated.self_clearance(q) < gated.min_self_clearance_m
        for q in gated.solve_all(pose)
    )

    # An ordinary extended-wrist pose is unaffected.
    open_pose = arm.fk([0.2, -1.4, 1.5, -1.6, -1.57, 0.0])
    assert gated.reachable(open_pose)


def test_link_segments_track_the_configuration_not_the_wrist():
    """The upper arm and forearm move independently of wrist_3.

    This is why the gripper's fixed image-space mask cannot represent them, and
    why a survey pose can be top-down, collision-free and fully framed while the
    robot's own arm lies across the picture -- observed on a field run at
    obliquity 0.0 deg.  Rotating only the wrist must leave these segments put;
    rotating the elbow must move them.
    """
    arm = UR5eArm(base=Transform(np.eye(3), np.zeros(3)))
    base_config = [0.3, -1.2, 1.4, -1.6, -1.57, 0.2]

    segments = arm.link_segments(base_config)
    assert len(segments) == 2
    for start, end, radius in segments:
        assert 0.03 < radius < 0.08  # UR5e collision tubes
        assert np.linalg.norm(end - start) > 0.3  # real link lengths

    # Wrist-only change: the arm links must not move at all.
    wrist_only = list(base_config)
    wrist_only[5] += 1.0
    for (a0, b0, _r), (a1, b1, _r2) in zip(segments, arm.link_segments(wrist_only)):
        assert np.allclose(a0, a1, atol=1e-9)
        assert np.allclose(b0, b1, atol=1e-9)

    # Elbow change: they must move, which is exactly what no static mask can see.
    elbow = list(base_config)
    elbow[2] += 0.6
    moved = max(
        float(np.linalg.norm(b1 - b0))
        for (_a0, b0, _r), (_a1, b1, _r2) in zip(segments, arm.link_segments(elbow))
    )
    assert moved > 0.05


def test_unreachable_pose_is_rejected():
    arm = UR5eArm()
    far = Transform(np.eye(3), np.array([2.0, 0.0, 0.5]))
    assert not arm.reachable(far)
    assert arm.solve(far) is None


def test_reachable_returns_bool_for_a_self_generated_pose():
    arm = UR5eArm()
    target = arm.fk([0.2, -1.4, 1.5, -1.6, -1.57, 0.0])
    assert arm.reachable(target) is True
