from __future__ import annotations

import pytest

from aic_perception.board_visibility import MaskReport
from aic_perception.viewpoint_search import ActionKind, AdaptiveViewpointPlanner


def report(
    *,
    edges=(),
    area=0.13,
    seen=True,
    full=False,
    center=(0.0, 0.0),
    bottom=False,
    rectangularity=0.8,
):
    return MaskReport(
        seen=seen, full=full, edges=frozenset(edges), area_frac=area,
        rectangularity=rectangularity, center_error=center,
        artificial_bottom_contact=bottom,
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
    # Lower edge must hold in a second fresh frame before J3 moves.
    lower_ready = views(report(center=(0.0, 0.4), edges=("top",)))
    assert planner.next_action(lower_ready).kind is ActionKind.OBSERVE
    j3 = planner.next_action(lower_ready)
    assert j3.kind is ActionKind.UP_CLEARANCE
    # Upper edge independently holds twice before J4 roll is allowed.
    all_edges_visible = views(report(center=(0.0, 0.0)))
    assert planner.next_action(all_edges_visible).kind is ActionKind.OBSERVE
    j4 = planner.next_action(all_edges_visible)
    assert j4.kind is ActionKind.CAMERA_ROLL
    assert j4.aim_direction[1] == 0.0


def test_j2_never_reverses_direction_from_noisy_image_feedback():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    aligned = views(report(center=(0.0, 0.4), bottom=True, edges=("bottom",)))
    first = confirm_j1(planner, aligned)
    assert first.kind is ActionKind.BACKOFF
    # Still clipped after a regressive frame: clearance remains away from the
    # board. Mask noise must never make J2 drive back toward it.
    second = planner.next_action(views(report(center=(0.0, 0.4), bottom=True, edges=("bottom",), area=0.7)))
    assert second.kind is ActionKind.BACKOFF
    assert second.axial_direction == first.axial_direction == 1.0


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


@pytest.mark.parametrize(
    ("center_x", "edge", "expected_sign"),
    ((-0.166, "left", 1.0), (0.566, "right", -1.0)),
)
def test_j1_polarity_reduces_live_center_camera_error(
    center_x, edge, expected_sign
):
    """The live plant moves image x with the commanded J1 sign."""

    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(
        views(report(area=0.124, center=(center_x, 0.20), edges=(edge,)))
    )

    assert action.kind is ActionKind.BASE_YAW
    assert action.aim_direction[0] == expected_sign
    moved_x = center_x + expected_sign * 0.05
    assert abs(moved_x) < abs(center_x)


def test_live_j1_closed_loop_converges_without_a_direction_reversal():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    center_x = -0.166
    signs = []

    for _ in range(8):
        action = planner.next_action(
            views(report(area=0.124, center=(center_x, 0.20)))
        )
        if action.kind is ActionKind.OBSERVE:
            break
        assert action.kind is ActionKind.BASE_YAW
        signs.append(action.aim_direction[0])
        next_x = center_x + action.aim_direction[0] * 0.05
        assert abs(next_x) < abs(center_x)
        center_x = next_x

    assert signs
    assert set(signs) == {1.0}
    assert action.kind is ActionKind.OBSERVE


def test_start_relative_yaw_boundary_changes_geometry_instead_of_oscillating():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(
        views(report(area=0.11, center=(-0.7, -0.5), edges=("left", "top")))
    )
    planner.mark_yaw_unavailable(
        action,
        reason="start-relative joint-1 envelope reached",
    )
    recovery = planner.next_action(
        views(report(area=0.11, center=(-0.7, -0.5), edges=("left", "top")))
    )

    assert recovery.kind is ActionKind.BACKOFF
    assert not recovery.terminal
    assert planner.phase == "j2_bottom_clearance"


def test_global_yaw_travel_boundary_is_terminal():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(
        views(report(area=0.11, center=(-0.7, -0.5), edges=("left", "top")))
    )
    planner.mark_yaw_unavailable(
        action,
        reason="cumulative travel envelope reached",
        global_unavailable=True,
    )

    terminal = planner.next_action(
        views(report(area=0.11, center=(-0.7, -0.5), edges=("left", "top")))
    )
    assert terminal.kind is ActionKind.STAGNATED
    assert terminal.terminal
    assert "cannot continue" in terminal.reason


def test_repeated_start_relative_yaw_rejections_use_bounded_recovery_ladder():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.06,
        alignment_confirmation_frames=1,
        phase_confirmation_frames=1,
    )
    clipped = views(
        report(area=0.11, center=(-0.7, -0.5), edges=("left", "top"))
    )
    top_clipped = views(report(area=0.11, center=(0.0, 0.0), edges=("top",)))
    clear = views(report(area=0.11, center=(0.0, 0.0)))
    shifted_high_quality = views(
        report(area=0.20, center=(-0.3, 0.0), rectangularity=0.95)
    )

    first_yaw = planner.next_action(clipped)
    planner.mark_yaw_unavailable(first_yaw, reason="J1 envelope reached")
    first_recovery = planner.next_action(clipped)
    assert first_recovery.kind is ActionKind.BACKOFF
    assert planner.next_action(top_clipped).kind is ActionKind.UP_CLEARANCE
    assert planner.next_action(clear).kind is ActionKind.CAMERA_ROLL

    second_yaw = planner.next_action(shifted_high_quality)
    assert second_yaw.kind is ActionKind.BASE_YAW
    planner.mark_yaw_unavailable(second_yaw, reason="J1 envelope reached")
    second_recovery = planner.next_action(shifted_high_quality)
    assert second_recovery.kind is ActionKind.UP_CLEARANCE
    assert planner.next_action(clear).kind is ActionKind.CAMERA_ROLL

    third_yaw = planner.next_action(shifted_high_quality)
    assert third_yaw.kind is ActionKind.BASE_YAW
    planner.mark_yaw_unavailable(third_yaw, reason="J1 envelope reached")
    terminal = planner.next_action(clipped)

    assert [first_recovery.kind, second_recovery.kind] == [
        ActionKind.BACKOFF,
        ActionKind.UP_CLEARANCE,
    ]
    assert terminal.kind is ActionKind.STAGNATED


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
    # First fresh lower-edge frame is confirmation only; the second can move.
    assert first_up.kind is ActionKind.OBSERVE
    first_up = planner.next_action(aligned)
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
        views(report(area=0.52, center=(0.24, 0.0), edges=("left", "right", "top")))
    )

    assert drifted.kind is ActionKind.BASE_YAW


def test_repeatable_j2_drift_is_precompensated_instead_of_cycling():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    oversized = lambda x: views(
        report(
            area=0.60,
            center=(x, 0.0),
            edges=("left", "right", "top"),
        )
    )

    assert planner.next_action(oversized(0.05)).kind is ActionKind.OBSERVE
    assert planner.next_action(oversized(0.05)).kind is ActionKind.BACKOFF
    # J2 produced a repeatable +0.19 normalized image shift.
    assert planner.next_action(oversized(0.24)).kind is ActionKind.BASE_YAW

    # Returning merely to zero is intentionally no longer sufficient: J1
    # learns a small opposite offset so the next J2 step lands inside the
    # exit hysteresis instead of starting the same cycle again.
    assert planner.next_action(oversized(0.05)).kind is ActionKind.BASE_YAW
    assert planner.next_action(oversized(-0.08)).kind is ActionKind.OBSERVE
    assert planner.next_action(oversized(-0.08)).kind is ActionKind.BACKOFF

    after_compensated_j2 = planner.next_action(oversized(0.11))
    assert after_compensated_j2.kind is ActionKind.BACKOFF
    assert planner.phase == "j2_bottom_clearance"


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
    assert correction.aim_direction[0] > 0.0


def test_unstable_centered_masks_cannot_prematurely_leave_j1():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    first = planner.next_action(
        views(report(area=0.12, center=(-0.08, 0.65)))
    )
    swapped = planner.next_action(
        views(report(area=0.13, center=(0.07, -0.62)))
    )

    assert first.kind is ActionKind.OBSERVE
    assert swapped.kind is ActionKind.OBSERVE
    assert planner.phase == "j1_yaw_alignment"

    # Two stable frames for the new component may now advance, but the first
    # discontinuous pair never did.
    advanced = planner.next_action(
        views(report(area=0.13, center=(0.06, -0.61)))
    )
    assert advanced.kind is ActionKind.OBSERVE
    assert planner.phase == "j2_bottom_clearance"


def test_j2_and_j3_require_independent_stable_confirmation_windows():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.06,
        alignment_confirmation_frames=1,
        phase_confirmation_frames=2,
    )
    ready = views(report(area=0.20, center=(0.0, 0.0)))

    assert planner.next_action(ready).kind is ActionKind.OBSERVE  # J2 frame 1
    assert planner.next_action(ready).kind is ActionKind.OBSERVE  # J3 frame 1
    assert planner.phase == "j3_top_clearance"
    assert planner.next_action(ready).kind is ActionKind.CAMERA_ROLL


def test_unstable_clearance_masks_restart_confirmation_not_phase():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.06,
        alignment_confirmation_frames=1,
        phase_confirmation_frames=2,
    )
    first = planner.next_action(
        views(report(area=0.20, center=(0.0, 0.55)))
    )
    jumped = planner.next_action(
        views(report(area=0.34, center=(0.0, -0.55)))
    )

    assert first.kind is ActionKind.OBSERVE
    assert jumped.kind is ActionKind.OBSERVE
    assert planner.phase == "j2_bottom_clearance"


def test_non_global_boundary_forces_clearance_even_for_small_mask():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    small = views(
        report(area=0.02, center=(-0.7, -0.5), edges=("left", "top"))
    )
    action = planner.next_action(small)
    assert action.kind is ActionKind.BASE_YAW

    planner.mark_yaw_unavailable(
        action,
        reason="start-relative joint-1 envelope reached",
    )
    recovery = planner.next_action(small)

    assert recovery.kind is ActionKind.BACKOFF
    assert recovery.axial_direction == 1.0


def test_stable_deep_gripper_component_cannot_prematurely_leave_j1():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    gripper = views(
        report(
            area=0.07,
            center=(0.0, 0.92),
            edges=("bottom",),
            bottom=True,
            rectangularity=0.15,
        )
    )

    first = planner.next_action(gripper)
    second = planner.next_action(gripper)

    assert first.kind is ActionKind.BASE_YAW
    assert second.kind is ActionKind.BASE_YAW
    assert planner.phase == "j1_yaw_alignment"


def test_no_view_yaw_cycles_use_j2_then_j3_and_terminate():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    actions = []
    for _ in range(20):
        action = planner.next_action({})
        actions.append(action)
        if action.terminal:
            break

    recovery_kinds = [
        item.kind
        for item in actions
        if item.kind in {ActionKind.BACKOFF, ActionKind.UP_CLEARANCE}
    ]
    assert recovery_kinds == [ActionKind.BACKOFF, ActionKind.UP_CLEARANCE]
    assert actions[-1].kind is ActionKind.STAGNATED
    assert len(actions) < 20


def test_j3_falls_back_to_monotonic_j2_when_bottom_is_lost():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.06,
        alignment_confirmation_frames=1,
        phase_confirmation_frames=1,
    )
    top_clipped = views(report(area=0.20, center=(0.0, 0.0), edges=("top",)))
    assert planner.next_action(top_clipped).kind is ActionKind.UP_CLEARANCE

    bottom_lost = planner.next_action(
        views(
            report(
                area=0.20,
                center=(0.0, 0.0),
                edges=("bottom", "top"),
                bottom=True,
            )
        )
    )
    assert bottom_lost.kind is ActionKind.BACKOFF
    assert bottom_lost.axial_direction == 1.0
    assert planner.phase == "j2_bottom_clearance"


def test_j4_roll_is_bounded_to_one_reversal_then_geometry_recovery():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.06,
        alignment_confirmation_frames=1,
        phase_confirmation_frames=1,
        max_roll_moves=6,
    )
    flat = views(report(area=0.20, center=(0.0, 0.0)))
    actions = [planner.next_action(flat) for _ in range(5)]

    assert actions[0].kind is ActionKind.CAMERA_ROLL
    assert actions[1].kind is ActionKind.CAMERA_ROLL
    assert actions[2].kind is ActionKind.CAMERA_ROLL
    assert actions[2].aim_direction[0] == -actions[1].aim_direction[0]
    assert actions[3].kind is ActionKind.CAMERA_ROLL
    # Equal current quality is kept; an unnecessary rollback would add motion
    # without restoring anything better.
    assert actions[4].kind is ActionKind.BACKOFF
    assert not actions[4].request_rollback
    assert "keep the best current roll view" in actions[4].reason

    tail = [planner.next_action(flat) for _ in range(3)]
    assert tail[-1].kind is ActionKind.STAGNATED
    assert tail[-1].terminal


def test_j4_monotonic_improvement_still_has_a_finite_excursion():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.06,
        alignment_confirmation_frames=1,
        phase_confirmation_frames=1,
        max_roll_moves=3,
    )
    actions = []
    for area in (0.15, 0.18, 0.21, 0.24):
        actions.append(
            planner.next_action(
                views(report(area=area, center=(0.0, 0.0)))
            )
        )

    assert [item.kind for item in actions[:3]] == [
        ActionKind.CAMERA_ROLL,
        ActionKind.CAMERA_ROLL,
        ActionKind.CAMERA_ROLL,
    ]
    assert actions[3].kind is ActionKind.BACKOFF
    assert "keep the best current roll view" in actions[3].reason
    terminal = planner.next_action(
        views(report(area=0.25, center=(0.0, 0.0)))
    )
    assert terminal.kind is ActionKind.STAGNATED


def test_j3_j4_edge_flicker_cannot_reset_global_roll_limit():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.06,
        alignment_confirmation_frames=1,
        phase_confirmation_frames=1,
        max_roll_moves=2,
    )
    clear = views(report(area=0.20, center=(0.0, 0.0)))
    top_lost = views(
        report(area=0.20, center=(0.0, 0.0), edges=("top",))
    )

    actions = [
        planner.next_action(clear),
        planner.next_action(top_lost),
        planner.next_action(clear),
        planner.next_action(top_lost),
        planner.next_action(clear),
    ]

    assert [item.kind for item in actions].count(ActionKind.CAMERA_ROLL) == 2
    assert actions[-1].kind is ActionKind.BACKOFF
    # A J3/J4 re-entry did not reset the two-command global excursion.
    assert planner.next_action(clear).kind is ActionKind.STAGNATED


def test_unexpected_camera_cannot_complete_or_steer_policy():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action({"rogue_camera": report(full=True, area=0.2)})

    assert action.kind is ActionKind.BASE_YAW
    assert not planner.coverage_achieved


def test_stale_yaw_rejection_cannot_overwrite_terminal_state():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    action = planner.next_action(
        views(report(area=0.2, center=(0.5, 0.0), edges=("right",)))
    )
    terminal = planner.next_action({}, deadline_reached=True)

    planner.mark_yaw_unavailable(action, reason="late controller response")
    assert planner.next_action({}) is terminal
    assert terminal.kind is ActionKind.DEADLINE


def test_mismatched_nonterminal_yaw_rejection_is_refused():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    old = planner.next_action(
        views(report(area=0.2, center=(0.5, 0.0), edges=("right",)))
    )
    current = planner.next_action(
        views(report(area=0.21, center=(0.4, 0.0), edges=("right",)))
    )
    assert current.kind is ActionKind.BASE_YAW

    with pytest.raises(ValueError, match="stale"):
        planner.mark_yaw_unavailable(old, reason="late rejection")


def _drive_live_fragment_boundary(planner: AdaptiveViewpointPlanner):
    """Replay center-camera reports immediately before the live oscillation."""

    trace = (
        report(area=0.124, center=(-0.166, 0.706), bottom=True),
        report(area=0.141, center=(-0.209, 0.676), bottom=True),
        report(area=0.158, center=(-0.255, 0.646), bottom=True),
        report(area=0.142, center=(-0.211, 0.673), bottom=True),
        report(area=0.131, center=(-0.183, 0.693), bottom=True),
        report(area=0.120, center=(-0.155, 0.712), bottom=True),
    )
    actions = [planner.next_action(views(item)) for item in trace]
    assert all(action.kind is ActionKind.BASE_YAW for action in actions)
    return trace[-1], actions[-1]


def test_live_center_component_swap_locks_out_yaw_and_uses_j2():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    continuous_view, boundary_move = _drive_live_fragment_boundary(planner)
    swapped_fragment = report(
        area=0.121,
        center=(0.588, -0.691),
        edges=("right", "top"),
    )

    recovery = planner.next_action(views(swapped_fragment))

    assert recovery.kind is ActionKind.BACKOFF
    assert not recovery.request_rollback
    assert "identity discontinuity" in recovery.reason
    assert planner.phase == "j2_bottom_clearance"

    # Even though this new component has a large horizontal error, J1 stays
    # locked out until the prescribed J2/J3 geometry ladder is complete.
    same_fragment = views(swapped_fragment)
    assert planner.next_action(same_fragment).kind is ActionKind.OBSERVE
    assert planner.next_action(same_fragment).kind is ActionKind.UP_CLEARANCE


def test_smooth_center_crossing_is_not_mistaken_for_component_swap():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    _drive_live_fragment_boundary(planner)
    smooth_crossing = report(
        area=0.125,
        center=(0.080, 0.690),
        bottom=True,
    )

    action = planner.next_action(views(smooth_crossing))

    assert action.kind is ActionKind.OBSERVE


def test_slow_cumulative_j1_regression_reverses_once_then_uses_clearance():
    planner = AdaptiveViewpointPlanner(min_goal_area_frac=0.06)
    first_leg = (0.56, 0.57, 0.58, 0.595, 0.61)
    first_actions = [
        planner.next_action(
            views(report(area=0.20, center=(x, 0.10), edges=("right",)))
        )
        for x in first_leg
    ]

    # Every individual regression is small, but the cumulative regression is
    # unambiguous. It must reverse within four feedback frames.
    assert [item.aim_direction[0] for item in first_actions] == [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
    ]

    recovery = None
    for x in (0.625, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.56, 0.58):
        recovery = planner.next_action(
            views(report(area=0.20, center=(x, 0.10), edges=("right",)))
        )
        if recovery.kind is ActionKind.ROLLBACK:
            break

    assert recovery.kind is ActionKind.ROLLBACK
    assert recovery.request_rollback
    assert "second yaw reversal" in recovery.reason

    # A rollback is not trusted blindly: if the camera returns a different
    # component, yaw stays locked out and recovery changes geometry.
    after_bad_restore = planner.next_action(
        views(report(area=0.20, center=(-0.60, -0.60), edges=("left", "top")))
    )
    assert after_bad_restore.kind is ActionKind.BACKOFF
    assert "rollback camera validation failed" in after_bad_restore.reason


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
    with pytest.raises(ValueError):
        AdaptiveViewpointPlanner(yaw_probe_scale=0.0)
    with pytest.raises(ValueError):
        AdaptiveViewpointPlanner(horizontal_coverage_area_frac=0.0)
    with pytest.raises(ValueError):
        AdaptiveViewpointPlanner(horizontal_center_threshold=0.3, horizontal_exit_threshold=0.2)
    with pytest.raises(ValueError):
        AdaptiveViewpointPlanner(max_roll_moves=1)
