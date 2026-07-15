from __future__ import annotations

import pytest

from aic_perception.board_visibility import MaskReport
from aic_perception.viewpoint_search import ActionKind, AdaptiveViewpointPlanner


def report(*, edges=(), area=0.13, seen=True, full=False, center=(0.0, 0.0), bottom=False):
    return MaskReport(
        seen=seen, full=full, edges=frozenset(edges), area_frac=area,
        rectangularity=0.8, center_error=center, artificial_bottom_contact=bottom,
    )


def views(center: MaskReport, left: MaskReport | None = None, right: MaskReport | None = None):
    result = {"center_camera": center}
    if left is not None:
        result["left_camera"] = left
    if right is not None:
        result["right_camera"] = right
    return result


def confirm_j1(planner: AdaptiveViewpointPlanner, reports):
    """Supply the two fresh center-camera frames required to leave J1."""

    confirmation = planner.next_action(reports)
    assert confirmation.kind is ActionKind.OBSERVE
    return planner.next_action(reports)


def test_one_complete_camera_finishes_without_all_three():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(views(report(full=True), report(area=0.01), report(area=0.01)))
    assert action.kind is ActionKind.DONE
    assert action.camera == "center_camera"


def test_exact_workflow_is_j1_then_j2_then_j3_then_j4():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    # J1: board is horizontally off-center.
    j1 = planner.next_action(views(report(center=(0.5, 0.4), bottom=True, edges=("bottom", "top"))))
    assert j1.kind is ActionKind.BASE_YAW
    # J1 aligned in two fresh frames: J2 is next and probes clearance.
    aligned = views(report(center=(0.0, 0.4), bottom=True, edges=("bottom", "top")))
    j2 = confirm_j1(planner, aligned)
    assert j2.kind is ActionKind.BACKOFF
    # Lower edge visible: J3 becomes next.
    j3 = planner.next_action(views(report(center=(0.0, 0.4), edges=("top",))))
    assert j3.kind is ActionKind.UP_CLEARANCE
    # Upper edge visible: J4 roll is next.
    j4 = planner.next_action(views(report(center=(0.0, 0.0))))
    assert j4.kind is ActionKind.CAMERA_ROLL
    assert j4.aim_direction[1] == 0.0


def test_bad_relative_probe_reverses_direction_from_image_feedback():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    aligned = views(report(center=(0.0, 0.4), bottom=True, edges=("bottom",)))
    first = confirm_j1(planner, aligned)
    assert first.kind is ActionKind.BACKOFF
    # Still clipped after a regressive probe: next J2 action changes sign.
    second = planner.next_action(views(report(center=(0.0, 0.4), bottom=True, edges=("bottom",), area=0.7)))
    assert second.kind is ActionKind.BACKOFF
    assert second.axial_direction == -first.axial_direction


def test_yaw_uses_center_camera_not_three_camera_area_gate():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(views(
        report(center=(-0.4, 0.0)), report(area=0.01), report(area=0.01)
    ))
    assert action.kind is ActionKind.BASE_YAW


def test_live_false_gripper_blob_does_not_end_j1_or_reverse_once():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    board_corner = report(
        edges=("left", "top"),
        area=0.027,
        center=(-0.896, -0.697),
    )
    first = planner.next_action(views(board_corner))
    assert first.kind is ActionKind.BASE_YAW

    # Exact center-camera report from the failed run after the first yaw: the
    # centered 2.1% bottom-contacting component was the gripper, not the board.
    false_gripper = report(
        area=0.021,
        center=(0.011, 0.928),
        bottom=True,
    )
    second = planner.next_action(views(false_gripper))

    assert second.kind is ActionKind.BASE_YAW
    assert second.aim_direction[0] == first.aim_direction[0]


def test_losing_center_alignment_after_j2_restarts_at_j1_before_roll():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    j2 = confirm_j1(
        planner,
        views(report(area=0.12, center=(0.0, 0.4), bottom=True, edges=("bottom", "top"))),
    )
    assert j2.kind is ActionKind.BACKOFF

    shifted = report(area=0.10, center=(-0.7, 0.2), edges=("left",))
    next_action = planner.next_action(views(shifted))

    assert next_action.kind is ActionKind.BASE_YAW
    assert "J1 horizontal yaw" in next_action.reason


def test_losing_center_mask_twice_reverses_without_frame_by_frame_chatter():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    first = planner.next_action(
        views(report(area=0.03, center=(-0.8, -0.5), edges=("left",)))
    )
    second = planner.next_action(
        {},
    )
    third = planner.next_action(
        {},
    )

    assert first.kind is second.kind is third.kind is ActionKind.BASE_YAW
    assert second.aim_direction[0] == first.aim_direction[0]
    assert third.aim_direction[0] == -first.aim_direction[0]


def test_confirmed_improving_yaw_is_monotonic_and_uses_coarse_steps():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    corner = report(area=0.027, center=(-0.896, -0.697), edges=("left", "top"))
    wrong = planner.next_action(views(corner))
    false_gripper = report(area=0.021, center=(0.0, 0.92), bottom=True)
    noisy_regression = planner.next_action(views(false_gripper))
    reverse = planner.next_action(views(false_gripper))
    restore = planner.next_action(views(corner))
    improved = planner.next_action(
        views(report(area=0.064, center=(-0.811, -0.631), edges=("left", "top")))
    )

    assert noisy_regression.aim_direction[0] == wrong.aim_direction[0]
    assert wrong.aim_direction[0] == -reverse.aim_direction[0]
    assert restore.aim_direction[0] == reverse.aim_direction[0]
    assert improved.aim_direction[0] == reverse.aim_direction[0]
    assert restore.angular_scale == pytest.approx(1.5)
    assert improved.angular_scale == pytest.approx(1.5)


def test_yaw_preflight_boundary_is_terminal_not_an_oscillation_command():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(
        views(report(area=0.11, center=(-0.7, -0.5), edges=("left", "top")))
    )
    planner.mark_yaw_unavailable(
        action,
        reason="start-relative joint-1 envelope reached",
    )
    terminal = planner.next_action(
        views(report(area=0.11, center=(-0.7, -0.5), edges=("left", "top")))
    )

    assert terminal.kind is ActionKind.STAGNATED
    assert terminal.terminal
    assert "cannot continue" in terminal.reason


def test_centered_oversized_board_stops_yaw_and_requests_standoff():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    failed_run_view = report(
        area=0.636,
        center=(0.049, -0.039),
        edges=("left", "right", "top"),
    )

    action = confirm_j1(planner, views(failed_run_view))

    assert action.kind is ActionKind.BACKOFF
    assert "J2 relative clearance" in action.reason


def test_j2_does_not_use_a_fixed_move_count_while_board_is_oversized():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    oversized = views(
        report(
            area=0.62,
            center=(0.03, -0.05),
            edges=("left", "right", "top"),
        )
    )

    first = confirm_j1(planner, oversized)
    actions = [first] + [planner.next_action(oversized) for _ in range(6)]

    assert all(action.kind is ActionKind.BACKOFF for action in actions)


def test_later_clearance_uses_alignment_hysteresis_instead_of_restarting_yaw():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    aligned = views(report(area=0.30, center=(0.05, -0.3), edges=("top",)))
    first_up = confirm_j1(
        planner,
        aligned,
    )
    slightly_shifted = planner.next_action(
        views(report(area=0.28, center=(0.12, -0.28), edges=("top",)))
    )

    assert first_up.kind is ActionKind.UP_CLEARANCE
    assert slightly_shifted.kind is ActionKind.UP_CLEARANCE


def test_side_camera_cannot_end_or_take_over_center_camera_yaw():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    center = report(
        area=0.14,
        center=(0.48, 0.68),
        edges=("right",),
        bottom=True,
    )
    left = report(
        area=0.26,
        center=(-0.119, 0.2),
        edges=(),
        bottom=True,
    )

    first = planner.next_action(views(center, left=left))
    assert first.kind is ActionKind.BASE_YAW
    assert first.camera == "center_camera"

    # Exact shape of the failed transition: the left camera satisfies the old
    # scalar gate while center is still +0.340 and clipped on the right.
    second = planner.next_action(
        views(
            report(area=0.329, center=(0.340, 0.4), edges=("right",), bottom=True),
            left=report(area=0.262, center=(-0.119, 0.2), bottom=True),
        )
    )
    assert second.kind is ActionKind.BASE_YAW
    assert second.camera == "center_camera"


def test_single_horizontal_edge_cannot_pass_j1_even_near_center():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(
        views(report(area=0.30, center=(0.05, 0.0), edges=("right",)))
    )

    assert action.kind is ActionKind.BASE_YAW


def test_normal_view_uses_tight_point_one_center_gate():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)

    outside = planner.next_action(views(report(area=0.30, center=(0.11, 0.0))))

    assert outside.kind is ActionKind.BASE_YAW


def test_j2_center_drift_returns_to_j1_before_another_clearance_move():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    aligned = views(
        report(area=0.60, center=(0.05, 0.0), edges=("left", "right", "top"))
    )
    j2 = confirm_j1(planner, aligned)
    assert j2.kind is ActionKind.BACKOFF

    drifted = planner.next_action(
        views(report(area=0.52, center=(0.14, 0.0), edges=("left", "right", "top")))
    )

    assert drifted.kind is ActionKind.BASE_YAW


def test_one_regressive_frame_does_not_reverse_yaw():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    first = planner.next_action(
        views(report(area=0.20, center=(0.50, 0.0), edges=("right",)))
    )
    noisy = planner.next_action(
        views(report(area=0.19, center=(0.52, 0.0), edges=("right",)))
    )

    assert noisy.kind is ActionKind.BASE_YAW
    assert noisy.aim_direction[0] == first.aim_direction[0]


def test_failed_alignment_confirmation_uses_new_frame_direction():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    candidate = planner.next_action(
        views(report(area=0.30, center=(0.08, 0.0)))
    )
    correction = planner.next_action(
        views(report(area=0.30, center=(-0.13, 0.0)))
    )

    assert candidate.kind is ActionKind.OBSERVE
    assert correction.kind is ActionKind.BASE_YAW
    assert correction.aim_direction[0] < 0.0


def test_deadline_is_terminal_and_sticky():
    planner = AdaptiveViewpointPlanner()
    terminal = planner.next_action({}, deadline_reached=True)
    assert terminal.kind is ActionKind.DEADLINE
    assert planner.next_action(views(report())) is terminal


def test_parameter_validation():
    with pytest.raises(ValueError):
        AdaptiveViewpointPlanner(expected_cameras=())
    with pytest.raises(ValueError):
        AdaptiveViewpointPlanner(min_goal_area_frac=0.5, max_goal_area_frac=0.4)
