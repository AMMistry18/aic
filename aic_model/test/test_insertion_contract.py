import numpy as np

from aic_model.insertion.contract import (
    accumulate_seat_residual,
    build_observation69,
    deploy_action_delta,
    guided_tip_target,
    quat_to_matrix,
    seat_residual_target_pose,
    seat_training_action_delta_port,
    sfp_tip_pose_from_tcp,
    tcp_pose_for_sfp_tip,
)


def _observation(port_quat):
    tcp_pos = np.array([-0.4, 0.15, 0.25])
    tcp_quat = np.array([0.98, 0.17, 0.01, -0.02])
    tip_pos, tip_rotation = sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)
    return build_observation69(
        joint_pos=np.zeros(6),
        joint_vel=np.zeros(6),
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        tcp_linear_velocity_world=np.zeros(3),
        tcp_angular_velocity_world=np.zeros(3),
        port_pos=np.array([-0.42, 0.15, 0.18]),
        port_quat=port_quat,
        tip_pos=tip_pos,
        tip_rotation=tip_rotation,
        wrench=np.ones(6),
        last_action=np.zeros(6),
        wrench_mode="zero",
    )


def test_quaternion_sign_does_not_change_observation():
    q = np.array([0.0, 0.998, 0.063, 0.0])
    assert np.array_equal(_observation(q), _observation(-q))


def test_positive_axial_action_is_inward_without_lateral_motion():
    q = np.array([0.0, 0.998, 0.063, 0.0])
    frame = quat_to_matrix(q)
    translation, _ = deploy_action_delta([0, 0, 1, 0, 0, 0], q)
    assert np.isclose(np.dot(translation, frame[:, 2]), 0.0035)
    assert np.linalg.norm(frame[:, :2].T @ translation) < 1e-12


def test_seat_action_delta_matches_training_conversion_and_yaw_clip():
    action_after_gain = np.full(6, 0.20)
    dpos_port, drot_port = seat_training_action_delta_port(action_after_gain)
    assert np.allclose(dpos_port, [0.3e-3, 0.3e-3, 0.7e-3])
    assert np.allclose(
        drot_port,
        np.radians([0.9167324722, 0.9167324722, 1.0]),
        atol=1e-10,
    )


def test_seat_residual_sequence_is_handoff_anchored_and_clamped():
    residual_pos = np.zeros(3)
    residual_rot = np.zeros(3)
    clipped = False
    action_after_gain = np.array([0.2, -0.2, 0.2, 0.2, -0.2, 0.2])
    for _ in range(1_000):
        residual_pos, residual_rot, step_clipped = accumulate_seat_residual(
            residual_pos, residual_rot, action_after_gain
        )
        clipped = clipped or step_clipped
    assert clipped
    assert np.allclose(residual_pos, [0.20, -0.20, 0.20])
    assert np.allclose(residual_rot, [0.35, -0.35, 0.35])

    base_pos = np.array([-0.3, 0.4, 0.2])
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    port_quat = np.array([0.0, 0.0, 1.0, 0.0])
    target_pos, target_rotation = seat_residual_target_pose(
        base_pos, base_quat, port_quat, residual_pos, residual_rot
    )
    frame = quat_to_matrix(port_quat)
    assert np.allclose(target_pos, base_pos + frame @ residual_pos)
    assert np.allclose(target_rotation.T @ target_rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(target_rotation), 1.0, atol=1e-12)


def test_tcp_tip_transform_round_trip():
    tcp_pos = np.array([-0.4, 0.15, 0.25])
    tcp_quat = np.array([0.98, 0.17, 0.01, -0.02])
    tip_pos, tip_rotation = sfp_tip_pose_from_tcp(tcp_pos, tcp_quat)
    recovered_pos, recovered_quat = tcp_pose_for_sfp_tip(tip_pos, tip_rotation)
    assert np.allclose(recovered_pos, tcp_pos, atol=1e-12)
    assert np.allclose(
        quat_to_matrix(recovered_quat), quat_to_matrix(tcp_quat), atol=1e-12
    )


def test_guided_target_stays_centered_and_aligned():
    port_pos = np.array([-0.32, 0.35, 0.18])
    port_quat = np.array([0.0, 0.004, 0.999992, 0.0])
    target_pos, target_rotation, next_plan, target_depth = guided_tip_target(
        port_pos=port_pos,
        port_quat=port_quat,
        current_depth=-0.022,
        planned_depth=-0.022,
    )
    frame = quat_to_matrix(port_quat)
    target_port = frame.T @ (target_pos - port_pos)
    assert np.allclose(target_port[:2], 0.0, atol=1e-12)
    assert np.isclose(target_port[2], -0.0205)
    assert np.isclose(next_plan, -0.0205)
    assert np.isclose(target_depth, -0.0205)
    assert np.allclose(target_rotation, frame, atol=1e-12)


def test_guided_target_caps_lead_and_slows_near_contact():
    kwargs = {
        "port_pos": np.zeros(3),
        "port_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    _pos, _rotation, next_plan, target_depth = guided_tip_target(
        **kwargs,
        current_depth=-0.010,
        planned_depth=0.020,
        max_lead=0.004,
    )
    assert np.isclose(next_plan, 0.0215)
    assert np.isclose(target_depth, -0.006)

    _pos, _rotation, _next_plan, target_depth = guided_tip_target(
        **kwargs,
        current_depth=-0.005,
        planned_depth=0.020,
        max_lead=0.020,
    )
    assert np.isclose(target_depth, 0.015)

    _pos, _rotation, next_plan, target_depth = guided_tip_target(
        **kwargs,
        current_depth=0.010,
        planned_depth=0.010,
    )
    assert np.isclose(next_plan, 0.01075)
    assert np.isclose(target_depth, 0.01075)
