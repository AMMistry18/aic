from __future__ import annotations

import math

import pytest

from aic_perception.board_visibility import MaskReport
from aic_perception.viewpoint_search import (
    ActionKind,
    AdaptiveViewpointPlanner,
    ViewpointAction,
)


def report(
    *,
    seen: bool = True,
    full: bool = False,
    center: tuple[float, float] = (0.0, 0.0),
    area: float = 0.20,
    edges: tuple[str, ...] = (),
    rectangularity: float = 0.9,
    bottom_contact: bool = False,
    orientation: float = 0.0,
    long_axis_ratio: float = 1.4,
    logo: bool = False,
    failure_reasons: tuple[str, ...] = (),
    clearance: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    context_pad: float = 0.0,
    gripper_overlap: int = 0,
    gripper_clearance: float = float("inf"),
    gripper_escape: tuple[float, float] = (0.0, -1.0),
) -> MaskReport:
    return MaskReport(
        seen=seen,
        full=full,
        edges=frozenset(edges),
        area_frac=area,
        rectangularity=rectangularity,
        artificial_bottom_contact=bottom_contact,
        gripper_overlap_px=gripper_overlap,
        gripper_clearance_px=gripper_clearance,
        gripper_escape_direction=gripper_escape,
        center_error=center,
        quality_score=0.5,
        clearance_px=clearance,
        context_pad_px=context_pad,
        failure_reasons=failure_reasons,
        orientation_deg=orientation,
        long_axis_ratio=long_axis_ratio,
        logo_seen=logo,
        logo_center_error=(0.4, 0.0),
        logo_area_frac=0.01 if logo else 0.0,
    )


def strict_report(**changes) -> MaskReport:
    values = dict(
        full=True,
        area=0.30,
        clearance=(100.0, 100.0, 100.0, 100.0),
        context_pad=36.0,
        gripper_overlap=0,
        gripper_clearance=30.0,
        orientation=0.5,
        long_axis_ratio=1.4,
        logo=True,
    )
    values.update(changes)
    return report(**values)


def views(
    center: MaskReport,
    left: MaskReport | None = None,
    right: MaskReport | None = None,
) -> dict[str, MaskReport]:
    reports = {"center_camera": center}
    if left is not None:
        reports["left_camera"] = left
    if right is not None:
        reports["right_camera"] = right
    return reports


def enter_level(planner: AdaptiveViewpointPlanner) -> None:
    aligned = views(report(center=(0.02, 0.0), orientation=2.0))
    assert planner.next_action(aligned).kind is ActionKind.OBSERVE
    assert planner.next_action(aligned).kind is ActionKind.OBSERVE
    assert planner.phase == "j6_align"
    assert planner.next_action(aligned).kind is ActionKind.OBSERVE
    assert planner.phase == "j2_4_level"


def enter_ascend(planner: AdaptiveViewpointPlanner) -> None:
    enter_level(planner)
    planner.mark_level_complete()
    assert planner.phase == "ascend_clearance"


def emit_first_roll(planner: AdaptiveViewpointPlanner) -> ViewpointAction:
    rotated = views(report(center=(0.02, 0.0), orientation=45.0))
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    action = planner.next_action(rotated)
    assert action.kind is ActionKind.CAMERA_ROLL
    return action


# ---------------------------------------------------------------------------
# Strict phase ordering


def test_acquire_uses_j1_and_side_cameras_are_only_direction_hints():
    planner = AdaptiveViewpointPlanner()
    center = report(seen=False)
    side = report(full=True, center=(0.7, 0.0), area=0.3)
    action = planner.next_action(views(center, left=side))
    assert action.kind is ActionKind.BASE_YAW
    assert planner.phase == "acquire_sweep"
    assert not planner.coverage_achieved


def test_j1_proportional_centering_precedes_j6():
    planner = AdaptiveViewpointPlanner()
    action = planner.next_action(
        views(report(center=(0.5, 0.0), orientation=45.0))
    )
    assert action.kind is ActionKind.BASE_YAW
    assert action.aim_direction[0] == -1.0
    assert planner.phase == "j1_center"


def test_j1_centering_learns_live_image_response_for_faster_correction():
    planner = AdaptiveViewpointPlanner()
    first = planner.next_action(views(report(center=(0.60, 0.0))))
    assert first.kind is ActionKind.BASE_YAW

    # The fresh frame moved 0.20 normalized image units after one signed
    # command scale.  The next correction should use that measured response,
    # rather than another minimum-size blind increment.
    second = planner.next_action(views(report(center=(0.40, 0.0))))
    assert second.kind is ActionKind.BASE_YAW
    assert "learned J1 image response" in second.reason
    assert second.angular_scale > first.angular_scale


def test_center_confirmation_enters_align_before_issuing_j6():
    planner = AdaptiveViewpointPlanner()
    rotated = views(report(center=(0.02, 0.0), orientation=-45.0))
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    assert planner.phase == "j6_align"
    roll = planner.next_action(rotated)
    assert roll.kind is ActionKind.CAMERA_ROLL
    assert roll.aim_direction[0] == -1.0


def test_each_j6_step_requires_two_fresh_same_sign_frames():
    planner = AdaptiveViewpointPlanner()
    rotated = views(report(center=(0.0, 0.0), orientation=55.0))
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    assert planner.next_action(rotated).kind is ActionKind.CAMERA_ROLL
    # The motion clears the old confirmation streak.
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    assert planner.next_action(rotated).kind is ActionKind.CAMERA_ROLL


def test_only_reliable_aligned_center_estimate_enters_level():
    planner = AdaptiveViewpointPlanner()
    enter_level(planner)
    assert planner.phase == "j2_4_level"
    wait = planner.next_action(views(report(full=True, orientation=0.0)))
    assert wait.kind is ActionKind.OBSERVE
    assert not planner.coverage_achieved


def test_one_aligned_j6_frame_cannot_enter_level():
    planner = AdaptiveViewpointPlanner()
    aligned = views(report(center=(0.0, 0.0), orientation=2.0))

    assert planner.next_action(aligned).kind is ActionKind.OBSERVE
    assert planner.next_action(aligned).kind is ActionKind.OBSERVE
    assert planner.phase == "j6_align"


def test_misaligned_exhausted_j6_never_enters_level():
    planner = AdaptiveViewpointPlanner(
        max_roll_moves=0,
        max_zoom_out_backoffs=0,
        max_recenter_entries=0,
    )
    rotated = views(report(center=(0.0, 0.0), orientation=40.0))
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    action = planner.next_action(rotated)
    assert action.kind is ActionKind.STAGNATED
    assert planner.phase == "j6_align"


def test_ambiguous_j6_uses_bounded_zoom_then_stagnates():
    planner = AdaptiveViewpointPlanner(
        max_zoom_out_backoffs=2,
        max_recenter_entries=0,
    )
    ambiguous = views(
        report(center=(0.0, 0.0), orientation=45.0, long_axis_ratio=1.05)
    )
    assert planner.next_action(ambiguous).kind is ActionKind.OBSERVE
    assert planner.next_action(ambiguous).kind is ActionKind.BACKOFF
    assert planner.next_action(ambiguous).kind is ActionKind.BACKOFF
    terminal = planner.next_action(ambiguous)
    assert terminal.kind is ActionKind.STAGNATED
    assert planner.phase == "j6_align"


def test_clipped_center_uses_bounded_zoom_then_stagnates():
    planner = AdaptiveViewpointPlanner(max_zoom_out_backoffs=2)
    clipped = views(
        report(center=(-0.7, -0.5), area=0.5, edges=("left", "right"))
    )

    assert planner.next_action(clipped).kind is ActionKind.BACKOFF
    assert planner.next_action(clipped).kind is ActionKind.BACKOFF
    terminal = planner.next_action(clipped)
    assert terminal.kind is ActionKind.STAGNATED
    assert "centroid" in terminal.reason


def test_flapping_j6_sign_uses_bounded_zoom_fallback():
    planner = AdaptiveViewpointPlanner(
        max_zoom_out_backoffs=1,
        max_recenter_entries=0,
    )
    positive = views(report(center=(0.0, 0.0), orientation=40.0))
    negative = views(report(center=(0.0, 0.0), orientation=-40.0))
    assert planner.next_action(positive).kind is ActionKind.OBSERVE
    assert planner.next_action(positive).kind is ActionKind.OBSERVE
    for frame in (negative, positive, negative, positive, negative):
        assert planner.next_action(frame).kind is ActionKind.OBSERVE
    assert planner.next_action(positive).kind is ActionKind.BACKOFF


def test_rejected_j6_can_use_bounded_j1_fallback_before_stagnating():
    planner = AdaptiveViewpointPlanner(max_zoom_out_backoffs=0)
    roll = emit_first_roll(planner)
    planner.mark_roll_unavailable(roll, reason="J6 envelope", allow_j1_fallback=True)
    fallback = planner.next_action(
        views(report(center=(0.4, 0.0), orientation=40.0))
    )
    assert fallback.kind is ActionKind.BASE_YAW
    assert "J6 alignment fallback" in fallback.reason
    assert planner.phase == "j1_center"


def test_j6_projection_drift_returns_through_center_before_level():
    planner = AdaptiveViewpointPlanner()
    emit_first_roll(planner)
    drifted_but_aligned = views(
        report(center=(0.7, 0.0), orientation=2.0, edges=("top",))
    )
    action = planner.next_action(drifted_but_aligned)
    assert action.kind is ActionKind.BASE_YAW
    assert planner.phase == "j1_center"


def test_no_j1_or_j6_after_level():
    planner = AdaptiveViewpointPlanner()
    enter_level(planner)
    planner.mark_level_complete()
    action = planner.next_action(
        views(report(center=(0.7, 0.0), orientation=50.0, edges=("top",)))
    )
    assert action.kind is ActionKind.TRANSLATE
    assert action.kind not in (ActionKind.BASE_YAW, ActionKind.CAMERA_ROLL)


# ---------------------------------------------------------------------------
# Level acknowledgement and completion


def test_level_acknowledgement_is_explicit_and_phase_checked():
    planner = AdaptiveViewpointPlanner()
    with pytest.raises(ValueError, match="j2_4_level"):
        planner.mark_level_complete()
    enter_level(planner)
    planner.mark_level_complete()
    assert planner.phase == "ascend_clearance"


def test_fresh_tilt_loss_can_request_immediate_relevel():
    planner = AdaptiveViewpointPlanner()
    enter_ascend(planner)

    planner.request_relevel()
    assert planner.phase == "j2_4_level"
    assert planner.next_action(views(report(full=True))).kind is ActionKind.OBSERVE
    planner.mark_level_complete()
    assert planner.phase == "ascend_clearance"


def test_prelevel_full_center_frame_cannot_finish():
    planner = AdaptiveViewpointPlanner()
    full = views(report(full=True, area=0.2, orientation=0.0))
    assert planner.next_action(full).kind is ActionKind.OBSERVE
    assert planner.next_action(full).kind is ActionKind.OBSERVE
    assert planner.next_action(full).kind is ActionKind.OBSERVE
    assert planner.phase == "j2_4_level"
    assert not planner.coverage_achieved


def test_postlevel_survey_requires_two_fresh_strict_frames():
    planner = AdaptiveViewpointPlanner()
    enter_level(planner)
    planner.mark_level_complete()
    candidate = planner.next_action(views(strict_report()))
    assert candidate.kind is ActionKind.OBSERVE
    done = planner.next_action(views(strict_report()))
    assert done.kind is ActionKind.DONE
    assert done.terminal
    assert planner.phase == "done"
    assert planner.coverage_achieved
    assert planner.selected_camera == "center_camera"
    assert planner.next_action({}) is done


def test_qualifying_fresh_frame_wins_at_deadline_boundary():
    planner = AdaptiveViewpointPlanner()
    enter_ascend(planner)

    assert planner.next_action(views(strict_report())).kind is ActionKind.OBSERVE
    done = planner.next_action(views(strict_report()), deadline_reached=True)
    assert done.kind is ActionKind.DONE


def test_full_side_camera_never_completes_after_level():
    planner = AdaptiveViewpointPlanner()
    enter_ascend(planner)
    action = planner.next_action(
        views(
            report(full=False, edges=("top",), area=0.3),
            left=report(full=True, area=0.2),
        )
    )
    assert action.kind is ActionKind.TRANSLATE
    assert not planner.coverage_achieved


def test_strict_survey_requires_expanded_component_context_margin():
    planner = AdaptiveViewpointPlanner()
    enter_ascend(planner)
    action = planner.next_action(
        views(
            report(
                full=True,
                area=0.2,
                clearance=(37.0, 37.0, 37.0, 37.0),
                context_pad=36.0,
            )
        )
    )
    assert action.kind is ActionKind.UP_CLEARANCE


def test_full_center_frame_that_is_too_large_gets_more_clearance():
    planner = AdaptiveViewpointPlanner()
    enter_ascend(planner)
    action = planner.next_action(views(report(full=True, area=0.60)))
    assert action.kind is ActionKind.UP_CLEARANCE
    assert not planner.coverage_achieved


def test_strict_survey_rejects_gripper_overlap_then_escapes_from_mask():
    planner = AdaptiveViewpointPlanner(max_occlusion_translates=2)
    enter_ascend(planner)
    blocked = strict_report(
        gripper_overlap=12000,
        gripper_clearance=0.0,
        gripper_escape=(0.2, -0.98),
    )

    action = planner.next_action(views(blocked))

    assert action.kind is ActionKind.TRANSLATE
    assert action.image_direction[0] < 0.0
    assert action.image_direction[1] > 0.0
    assert "protected task-board envelope" in action.reason


def test_mask_escape_reverses_when_fresh_frame_gets_worse():
    planner = AdaptiveViewpointPlanner(max_occlusion_translates=2)
    enter_ascend(planner)
    first = planner.next_action(
        views(
            strict_report(
                gripper_overlap=1000,
                gripper_clearance=0.0,
                gripper_escape=(0.0, -1.0),
            )
        )
    )
    second = planner.next_action(
        views(
            strict_report(
                gripper_overlap=3000,
                gripper_clearance=0.0,
                gripper_escape=(0.0, -1.0),
            )
        )
    )

    assert first.kind is second.kind is ActionKind.TRANSLATE
    assert first.image_direction[1] == -second.image_direction[1]
    assert "polarity reversed" in second.reason


def test_fully_framed_board_below_detail_scale_uses_bounded_approach():
    planner = AdaptiveViewpointPlanner(
        min_goal_area_frac=0.26,
        max_goal_area_frac=0.36,
    )
    enter_ascend(planner)

    action = planner.next_action(
        views(strict_report(full=False, area=0.20))
    )

    assert action.kind is ActionKind.APPROACH
    assert action.axial_direction < 0.0


@pytest.mark.parametrize(
    ("orientation", "ratio"),
    [(35.0, 1.4), (0.0, 1.05)],
)
def test_terminal_survey_rechecks_long_axis_gate(
    orientation: float, ratio: float
):
    planner = AdaptiveViewpointPlanner()
    enter_ascend(planner)
    action = planner.next_action(
        views(
            report(
                full=True,
                area=0.2,
                orientation=orientation,
                long_axis_ratio=ratio,
            )
        )
    )
    assert action.kind is not ActionKind.DONE
    assert not planner.coverage_achieved


def test_j6_alignment_tolerance_is_two_degrees_with_fine_correction():
    planner = AdaptiveViewpointPlanner()
    assert planner.roll_align_threshold_deg == pytest.approx(2.0)

    slightly_misaligned = views(
        report(center=(0.0, 0.0), orientation=2.1, long_axis_ratio=1.4)
    )
    assert planner.next_action(slightly_misaligned).kind is ActionKind.OBSERVE
    assert planner.next_action(slightly_misaligned).kind is ActionKind.OBSERVE
    correction = planner.next_action(slightly_misaligned)

    assert correction.kind is ActionKind.CAMERA_ROLL
    assert correction.angular_scale == pytest.approx(math.radians(2.1) / 0.10)


# ---------------------------------------------------------------------------
# Post-level clearance and stopped-short recovery


def test_postlevel_clipping_uses_only_joints_2_4_clearance():
    planner = AdaptiveViewpointPlanner(max_zoom_out_backoffs=0)
    enter_ascend(planner)
    action = planner.next_action(
        views(report(full=False, area=0.5, edges=("left", "right")))
    )
    assert action.kind is ActionKind.UP_CLEARANCE
    assert planner.phase == "ascend_clearance"


def test_leveling_projection_shift_uses_bounded_joints_2_4_translation():
    planner = AdaptiveViewpointPlanner(max_postlevel_translates=2)
    enter_ascend(planner)
    shifted = views(
        report(
            full=False,
            area=0.18,
            center=(-0.67, -0.47),
            edges=("left", "top"),
        )
    )

    action = planner.next_action(shifted)
    assert action.kind is ActionKind.TRANSLATE
    assert action.image_direction[0] < 0.0
    assert action.image_direction[1] < 0.0
    assert "re-center" in action.reason


def test_bottom_only_obstruction_is_centered_before_optical_backoff():
    planner = AdaptiveViewpointPlanner(
        max_zoom_out_backoffs=1,
        max_postlevel_translates=1,
        max_occlusion_translates=1,
    )
    enter_ascend(planner)
    blocked = views(
        report(full=False, area=0.3, edges=("bottom",), bottom_contact=True)
    )
    assert planner.next_action(blocked).kind is ActionKind.TRANSLATE
    assert planner.next_action(blocked).kind is ActionKind.TRANSLATE
    assert planner.next_action(blocked).kind is ActionKind.BACKOFF


def test_vertical_visual_servo_moves_both_directions_and_reverses_bad_sign():
    low_planner = AdaptiveViewpointPlanner(max_postlevel_translates=2)
    enter_ascend(low_planner)
    low = views(
        report(full=False, area=0.2, center=(0.0, 0.35), edges=("bottom",))
    )
    first = low_planner.next_action(low)
    assert first.kind is ActionKind.TRANSLATE
    assert first.image_direction[1] > 0.0

    # A fresh frame proves the assumed image-Y polarity was wrong.  The next
    # correction must reverse instead of repeating the same wrong-way roll.
    worse = views(
        report(full=False, area=0.2, center=(0.0, 0.45), edges=("bottom",))
    )
    reversed_action = low_planner.next_action(worse)
    assert reversed_action.kind is ActionKind.TRANSLATE
    assert reversed_action.image_direction[1] < 0.0
    assert "polarity reversed" in reversed_action.reason

    high_planner = AdaptiveViewpointPlanner(max_postlevel_translates=1)
    enter_ascend(high_planner)
    high = views(
        report(full=False, area=0.2, center=(0.0, -0.35), edges=("top",))
    )
    assert high_planner.next_action(high).image_direction[1] < 0.0


def test_alignment_zoom_does_not_consume_clearance_zoom_budget():
    planner = AdaptiveViewpointPlanner(
        max_zoom_out_backoffs=1,
        max_postlevel_translates=0,
        max_occlusion_translates=0,
        max_recenter_entries=0,
    )
    ambiguous = views(
        report(center=(0.0, 0.0), orientation=45.0, long_axis_ratio=1.05)
    )
    aligned = views(report(center=(0.0, 0.0), orientation=0.0))

    assert planner.next_action(ambiguous).kind is ActionKind.OBSERVE
    assert planner.next_action(ambiguous).kind is ActionKind.BACKOFF
    assert planner.next_action(aligned).kind is ActionKind.OBSERVE
    assert planner.next_action(aligned).kind is ActionKind.OBSERVE
    assert planner.phase == "j2_4_level"
    planner.mark_level_complete()
    blocked = views(
        report(full=False, area=0.3, edges=("bottom",), bottom_contact=True)
    )
    assert planner.next_action(blocked).kind is ActionKind.BACKOFF


def test_partial_clearance_replans_from_the_next_fresh_report():
    planner = AdaptiveViewpointPlanner(max_zoom_out_backoffs=0)
    enter_ascend(planner)
    opposite_edges = views(
        report(full=False, area=0.5, edges=("top", "bottom"))
    )
    up = planner.next_action(opposite_edges)
    assert up.kind is ActionKind.UP_CLEARANCE
    planner.mark_clearance_partial(up, reason="moved 8 mm of requested 60 mm")
    fresh_one_sided = views(
        report(full=False, area=0.3, edges=("left",), center=(-0.5, 0.0))
    )
    replanned = planner.next_action(fresh_one_sided)
    assert replanned.kind is ActionKind.TRANSLATE
    assert "stopped short" not in replanned.reason


def test_partial_clearance_rejects_wrong_or_stale_action():
    planner = AdaptiveViewpointPlanner(max_zoom_out_backoffs=0)
    enter_ascend(planner)
    clipped = views(report(full=False, area=0.5, edges=("top", "bottom")))
    up = planner.next_action(clipped)
    with pytest.raises(ValueError, match="UP_CLEARANCE"):
        planner.mark_clearance_partial(
            ViewpointAction(999, ActionKind.BACKOFF), reason="wrong kind"
        )
    with pytest.raises(ValueError, match="stale"):
        planner.mark_clearance_partial(
            ViewpointAction(999, ActionKind.UP_CLEARANCE), reason="stale"
        )
    planner.mark_clearance_partial(up, reason="short")


def test_framed_but_nonfull_view_observes_then_stagnates():
    planner = AdaptiveViewpointPlanner(max_stall_frames=2)
    enter_ascend(planner)
    framed = views(
        report(full=False, area=0.2, failure_reasons=("nonrectangular_board",))
    )
    assert planner.next_action(framed).kind is ActionKind.OBSERVE
    assert planner.next_action(framed).kind is ActionKind.STAGNATED


def test_deadline_is_terminal_and_sticky():
    planner = AdaptiveViewpointPlanner()
    assert planner.next_action(views(report(seen=False))).kind is ActionKind.BASE_YAW
    terminal = planner.next_action({}, deadline_reached=True)
    assert terminal.kind is ActionKind.DEADLINE
    assert terminal.terminal
    assert planner.next_action(views(report(full=True))) is terminal
