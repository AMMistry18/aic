"""Tests for the UR5e forward/inverse kinematics reachability gate."""

import math

import numpy as np
import pytest

from aic_perception.arm_ik import UR5eArm, JOINT_LIMITS
from aic_perception.board_stage2 import Transform


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


def test_fk_ik_round_trip_recovers_reachable_poses():
    """With a near seed (as the skill supplies the current joints) almost every
    reachable pose is recovered exactly; the rare miss is a near-singular pose,
    which a reachability gate is entitled to reject."""
    arm = UR5eArm()
    rng = np.random.default_rng(1)
    solved = 0
    trials = _random_joints(rng, 80)
    for q_true in trials:
        target = arm.fk(q_true)
        q = arm.solve(target, seed=q_true + rng.normal(0, 0.1, 6))
        if q is None:
            continue
        pos, ori = arm.fk_residual(q, target)
        assert pos < 2e-3 and ori < 1e-2
        # Solutions respect the joint limits.
        assert np.all(q >= JOINT_LIMITS[:, 0] - 1e-6)
        assert np.all(q <= JOINT_LIMITS[:, 1] + 1e-6)
        solved += 1
    assert solved >= int(0.97 * len(trials))


def test_cold_solve_without_seed_finds_a_branch():
    """Canonical seeds alone recover most reachable poses (no truth hint)."""
    arm = UR5eArm()
    rng = np.random.default_rng(2)
    found = 0
    trials = _random_joints(rng, 60)
    for q_true in trials:
        target = arm.fk(q_true)
        if arm.reachable(target):
            found += 1
    assert found >= int(0.9 * len(trials))


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


def test_unreachable_pose_is_rejected():
    arm = UR5eArm()
    far = Transform(np.eye(3), np.array([2.0, 0.0, 0.5]))
    assert not arm.reachable(far)
    assert arm.solve(far) is None


def test_reachable_returns_bool_for_a_self_generated_pose():
    arm = UR5eArm()
    target = arm.fk([0.2, -1.4, 1.5, -1.6, -1.57, 0.0])
    assert arm.reachable(target) is True
