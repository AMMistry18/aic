from __future__ import annotations

import math

import pytest

from aic_perception.board_visibility import MaskReport
from aic_perception.viewpoint_search import (
    ActionKind,
    AdaptiveViewpointPlanner,
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
    logo_center: tuple[float, float] = (0.4, 0.0),
    failure_reasons: tuple[str, ...] = (),
    clearance: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    context_pad: float = 0.0,
) -> MaskReport:
    return MaskReport(
        seen=seen,
        full=full,
        edges=frozenset(edges),
        area_frac=area,
        rectangularity=rectangularity,
        artificial_bottom_contact=bottom_contact,
        center_error=center,
        quality_score=0.5,
        clearance_px=clearance,
        context_pad_px=context_pad,
        failure_reasons=failure_reasons,
        orientation_deg=orientation,
        long_axis_ratio=long_axis_ratio,
        logo_seen=logo,
        logo_center_error=logo_center,
        logo_area_frac=0.01 if logo else 0.0,
    )


def views(center: MaskReport, left: MaskReport | None = None, right: MaskReport | None = None):
    reports = {"center_camera": center}
    if left is not None:
        reports["left_camera"] = left
    if right is not None:
        reports["right_camera"] = right
    return reports


def make_planner(**overrides) -> AdaptiveViewpointPlanner:
    return AdaptiveViewpointPlanner(**overrides)


def reach_ascend(planner: AdaptiveViewpointPlanner) -> None:
    """Drive the planner through CENTER with two confirmed centered frames."""

    centered = views(report(center=(0.02, 0.1), area=0.30, edges=("top",)))
    assert planner.next_action(centered).kind is ActionKind.OBSERVE
    action = planner.next_action(centered)
    assert action.moves_robot
    assert planner.phase == "ascend_clearance"


# ----------------------------------------------------------------------
# Terminal conditions


def test_complete_view_terminates_done():
    planner = make_planner()
    action = planner.next_action(views(report(full=True, area=0.20)))
    assert action.kind is ActionKind.DONE
    assert action.terminal
    assert planner.terminal and planner.coverage_achieved
    assert planner.selected_camera == "center_camera"


def test_full_view_outside_area_bounds_is_not_done():
    planner = make_planner()
    action = planner.next_action(views(report(full=True, area=0.60)))
    assert action.kind is not ActionKind.DONE


def test_unexpected_camera_name_cannot_finish():
    planner = make_planner()
    action = planner.next_action({"rogue_camera": report(full=True, area=0.2)})
    assert action.kind is ActionKind.BASE_YAW
    assert not planner.coverage_achieved


def test_deadline_is_terminal_and_sticky():
    planner = make_planner()
    action = planner.next_action(views(report(center=(0.6, 0.0))))
    assert action.kind is ActionKind.BASE_YAW
    terminal = planner.next_action({}, deadline_reached=True)
    assert terminal.kind is ActionKind.DEADLINE and terminal.terminal
    assert planner.next_action(views(report())) is terminal


# ----------------------------------------------------------------------
# ACQUIRE sweep


def test_sweep_without_evidence_uses_stable_direction():
    planner = make_planner()
    first = planner.next_action(views(report(seen=False)))
    second = planner.next_action(views(report(seen=False)))
    assert first.kind is second.kind is ActionKind.BASE_YAW
    assert first.aim_direction == second.aim_direction
    assert first.angular_scale == 1.0
    assert planner.phase == "acquire_sweep"


def test_sweep_direction_seeded_by_side_camera():
    planner = make_planner()
    right_view = report(edges=("right",), area=0.10)
    action = planner.next_action(views(report(seen=False), right=right_view))
    assert action.kind is ActionKind.BASE_YAW
    assert action.aim_direction[0] == -1.0


def test_gripper_blob_is_not_board_evidence():
    planner = make_planner()
    gripper = report(center=(0.02, 0.93), bottom_contact=True, area=0.05)
    action = planner.next_action(views(gripper))
    assert action.kind is ActionKind.BASE_YAW
    assert planner.phase == "acquire_sweep"


def test_gripper_blob_with_logo_is_board_evidence():
    planner = make_planner()
    anchored = report(center=(0.6, 0.93), bottom_contact=True, logo=True)
    action = planner.next_action(views(anchored))
    assert action.kind is ActionKind.BASE_YAW
    assert planner.phase == "j1_center"


# ----------------------------------------------------------------------
# CENTER: proportional yaw


def test_yaw_direction_opposes_error_sign():
    planner = make_planner()
    action = planner.next_action(views(report(center=(0.6, 0.0))))
    assert action.kind is ActionKind.BASE_YAW
    assert action.aim_direction[0] == -1.0

    planner = make_planner()
    action = planner.next_action(views(report(center=(-0.6, 0.0))))
    assert action.aim_direction[0] == 1.0


def test_yaw_scale_is_proportional_and_clamped():
    planner = make_planner()
    large = planner.next_action(views(report(center=(0.9, 0.0))))
    assert large.angular_scale == pytest.approx(1.35)

    planner = make_planner()
    moderate = planner.next_action(views(report(center=(-0.3, 0.0))))
    assert moderate.angular_scale == pytest.approx(0.45)

    planner = make_planner()
    capped = planner.next_action(views(report(center=(1.4, 0.0))))
    assert capped.angular_scale == pytest.approx(1.5)

    planner = make_planner()
    floored = planner.next_action(views(report(center=(0.16, 0.0))))
    assert floored.angular_scale == pytest.approx(0.24)


def test_centering_requires_two_fresh_frames():
    planner = make_planner()
    centered = views(report(center=(0.05, 0.0), area=0.30, edges=("top",)))
    first = planner.next_action(centered)
    assert first.kind is ActionKind.OBSERVE
    second = planner.next_action(centered)
    assert second.kind is ActionKind.UP_CLEARANCE
    assert planner.phase == "ascend_clearance"


def test_regressed_frame_resets_center_confirmation():
    planner = make_planner()
    assert planner.next_action(
        views(report(center=(0.05, 0.0)))
    ).kind is ActionKind.OBSERVE
    drifted = planner.next_action(views(report(center=(0.4, 0.0))))
    assert drifted.kind is ActionKind.BASE_YAW
    assert planner.next_action(
        views(report(center=(0.05, 0.0), area=0.3))
    ).kind is ActionKind.OBSERVE


def test_logo_only_view_increases_standoff_instead_of_chasing_logo():
    planner = make_planner()
    logo_only = views(report(seen=False, logo=True, logo_center=(0.1, 0.2)))
    action = planner.next_action(logo_only)
    assert action.kind is ActionKind.UP_CLEARANCE
    assert action.axial_direction == 1.0
    assert action.translation_scale == pytest.approx(2.0)
    assert planner.phase == "ascend_clearance"


def test_center_phase_only_moves_j1_even_when_rotated():
    # Strict ordering: no J6 during centering, no matter how credible the
    # long-side estimate is.
    planner = make_planner()
    rotated_offcenter = views(
        report(center=(0.5, 0.0), area=0.30, orientation=40.0)
    )
    action = planner.next_action(rotated_offcenter)
    assert action.kind is ActionKind.BASE_YAW
    assert planner.phase == "j1_center"


def test_align_phase_follows_confirmed_centering():
    planner = make_planner()
    centered_rotated = views(
        report(center=(0.02, 0.0), area=0.30, orientation=40.0)
    )
    # First frame confirms centering and counts as the first long-side
    # observation; the second consistent frame may command J6.
    assert planner.next_action(centered_rotated).kind is ActionKind.OBSERVE
    action = planner.next_action(centered_rotated)
    assert action.kind is ActionKind.CAMERA_ROLL
    assert action.aim_direction[0] == 1.0
    assert planner.phase == "j6_align"


def test_live_low_aspect_trace_commands_j6_before_any_clearance():
    planner = make_planner()
    # Iterations 11-12 from the live trace: the long edge is only moderately
    # elongated, but its -46-degree sign is stable in consecutive fresh frames.
    first = views(
        report(
            center=(-0.130, 0.114),
            area=0.286,
            edges=("left",),
            bottom_contact=True,
            orientation=-45.9,
            long_axis_ratio=1.23,
            logo=True,
        )
    )
    second = views(
        report(
            center=(-0.131, 0.115),
            area=0.286,
            edges=("left",),
            bottom_contact=True,
            orientation=-46.0,
            long_axis_ratio=1.23,
            logo=True,
        )
    )

    assert planner.next_action(first).kind is ActionKind.OBSERVE
    action = planner.next_action(second)
    assert action.kind is ActionKind.CAMERA_ROLL
    assert action.aim_direction[0] == -1.0
    assert "J6 long-side alignment" in action.reason


def test_align_phase_never_acts_on_a_single_frame_estimate():
    planner = make_planner()
    centered_aligned = views(report(center=(0.02, 0.0), area=0.30))
    assert planner.next_action(centered_aligned).kind is ActionKind.OBSERVE
    # Rotation appears only on the confirmation frame: centering completes,
    # but J6 must wait for a second consistent estimate.
    centered_rotated = views(
        report(center=(0.02, 0.0), area=0.30, orientation=40.0)
    )
    waiting = planner.next_action(centered_rotated)
    assert waiting.kind is ActionKind.OBSERVE
    assert planner.phase == "j6_align"
    confirmed = planner.next_action(centered_rotated)
    assert confirmed.kind is ActionKind.CAMERA_ROLL


def test_align_phase_sign_flip_restarts_confirmation():
    planner = make_planner()
    positive = views(report(center=(0.02, 0.0), area=0.30, orientation=40.0))
    negative = views(report(center=(0.02, 0.0), area=0.30, orientation=-40.0))
    assert planner.next_action(positive).kind is ActionKind.OBSERVE
    assert planner.next_action(negative).kind is ActionKind.OBSERVE
    action = planner.next_action(negative)
    assert action.kind is ActionKind.CAMERA_ROLL
    assert action.aim_direction[0] == -1.0


def test_align_phase_skips_ambiguous_or_aligned_boards():
    planner = make_planner()
    centered_ambiguous = views(
        report(
            center=(0.02, 0.0),
            area=0.30,
            orientation=45.0,
            long_axis_ratio=1.05,
            edges=("top",),
        )
    )
    assert planner.next_action(centered_ambiguous).kind is ActionKind.OBSERVE
    action = planner.next_action(centered_ambiguous)
    assert action.kind is ActionKind.UP_CLEARANCE
    assert planner.phase == "ascend_clearance"


def test_ratio_just_below_live_reliability_gate_remains_ambiguous():
    planner = make_planner()
    centered = views(
        report(
            center=(0.02, 0.0),
            area=0.30,
            orientation=-46.0,
            long_axis_ratio=1.14,
            edges=("top",),
        )
    )
    assert planner.next_action(centered).kind is ActionKind.OBSERVE
    action = planner.next_action(centered)
    assert action.kind is ActionKind.UP_CLEARANCE


# ----------------------------------------------------------------------
# CENTER: standoff-first guards (yaw provably cannot help)


def test_left_and_right_clipped_moves_away_instead_of_yawing():
    planner = make_planner()
    action = planner.next_action(
        views(report(center=(0.5, 0.0), area=0.30, edges=("left", "right")))
    )
    assert action.kind is ActionKind.UP_CLEARANCE
    assert planner.phase == "ascend_clearance"


def test_top_pinned_mass_moves_away_instead_of_yawing():
    # The live run-1 signature: camera in front of the board, only its near
    # strip visible at the top of the frame; yaw burned the whole workspace
    # envelope chasing an uncenterable sliver.
    planner = make_planner()
    action = planner.next_action(
        views(report(center=(-0.725, -0.633), area=0.094, edges=("left", "top")))
    )
    assert action.kind is ActionKind.UP_CLEARANCE
    assert planner.phase == "ascend_clearance"


def test_oversized_mask_moves_away_instead_of_yawing():
    planner = make_planner()
    action = planner.next_action(views(report(center=(0.5, 0.0), area=0.60)))
    assert action.kind is ActionKind.UP_CLEARANCE
    assert planner.phase == "ascend_clearance"


# ----------------------------------------------------------------------
# ASCEND: monotonic clearance


def test_oversized_board_ascends_with_scaled_step():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(views(report(center=(0.0, 0.1), area=0.90)))
    assert action.kind is ActionKind.UP_CLEARANCE
    assert action.translation_scale == pytest.approx(2.0)


def test_ascend_step_scale_is_capped():
    planner = make_planner(max_ascend_scale=2.5)
    reach_ascend(planner)
    action = planner.next_action(
        views(report(area=0.45 * 4.0, edges=("left", "right")))
    )
    assert action.kind is ActionKind.UP_CLEARANCE
    assert action.translation_scale == pytest.approx(2.5)


def test_bottom_contact_with_clear_top_retreats_along_optical_axis():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(
        views(report(center=(0.0, 0.4), area=0.30, edges=("bottom",)))
    )
    assert action.kind is ActionKind.BACKOFF
    assert action.axial_direction == 1.0


def test_gripper_band_contact_with_clear_top_retreats():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(
        views(report(center=(0.0, 0.4), area=0.30, bottom_contact=True))
    )
    assert action.kind is ActionKind.BACKOFF


def test_opposite_edges_ascend_rather_than_retreat():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(
        views(report(area=0.30, edges=("top", "bottom")))
    )
    assert action.kind is ActionKind.UP_CLEARANCE


def test_side_clipping_alone_still_ascends():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(views(report(area=0.30, edges=("left",))))
    assert action.kind is ActionKind.UP_CLEARANCE
    assert action.translation_scale == pytest.approx(1.5)


def test_logo_only_view_during_ascend_keeps_ascending():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(views(report(seen=False, logo=True)))
    assert action.kind is ActionKind.UP_CLEARANCE


def test_drift_during_ascend_recenters_within_budget():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(
        views(report(center=(0.5, 0.0), area=0.30, edges=("top",)))
    )
    assert action.kind is ActionKind.BASE_YAW
    assert planner.phase == "j1_center"


def test_recenter_budget_is_bounded():
    planner = make_planner(max_recenter_entries=1)
    reach_ascend(planner)
    drifted = views(report(center=(0.5, 0.0), area=0.30, edges=("top",)))
    assert planner.next_action(drifted).kind is ActionKind.BASE_YAW
    centered = views(report(center=(0.02, 0.0), area=0.30, edges=("top",)))
    assert planner.next_action(centered).kind is ActionKind.OBSERVE
    assert planner.next_action(centered).kind is ActionKind.UP_CLEARANCE
    # Budget exhausted: a second drift no longer interrupts the ascent.
    assert planner.next_action(drifted).kind is ActionKind.UP_CLEARANCE


def test_framed_but_incomplete_board_stalls_honestly():
    planner = make_planner(max_stall_frames=3)
    reach_ascend(planner)
    framed = views(
        report(
            area=0.20,
            rectangularity=0.3,
            failure_reasons=("nonrectangular_board",),
        )
    )
    assert planner.next_action(framed).kind is ActionKind.OBSERVE
    assert planner.next_action(framed).kind is ActionKind.OBSERVE
    terminal = planner.next_action(framed)
    assert terminal.kind is ActionKind.STAGNATED
    assert "nonrectangular_board" in terminal.reason


def test_undersized_framed_board_terminates_without_approach():
    planner = make_planner()
    reach_ascend(planner)
    terminal = planner.next_action(views(report(area=0.02)))
    assert terminal.kind is ActionKind.STAGNATED
    assert "never approaches" in terminal.reason


# ----------------------------------------------------------------------
# ASCEND: camera-roll alignment assist


def test_roll_assist_directly_corrects_long_axis_error():
    planner = make_planner()
    reach_ascend(planner)
    rotated = views(
        report(center=(0.0, 0.3), area=0.30, bottom_contact=True, orientation=30.0)
    )
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    probe = planner.next_action(rotated)
    assert probe.kind is ActionKind.CAMERA_ROLL
    assert probe.aim_direction[0] == 1.0
    assert probe.angular_scale == pytest.approx(3.0)

    improved = views(
        report(center=(0.0, 0.3), area=0.30, bottom_contact=True, orientation=15.0)
    )
    second = planner.next_action(improved)
    assert second.kind is ActionKind.CAMERA_ROLL
    assert second.aim_direction[0] == 1.0
    assert second.angular_scale == pytest.approx(math.radians(15.0) / 0.10)


def test_roll_assist_sign_flip_requires_reconfirmation():
    planner = make_planner()
    reach_ascend(planner)
    rotated = views(
        report(center=(0.0, 0.3), area=0.30, bottom_contact=True, orientation=30.0)
    )
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    assert planner.next_action(rotated).kind is ActionKind.CAMERA_ROLL
    crossed = views(
        report(center=(0.0, 0.3), area=0.30, bottom_contact=True, orientation=-20.0)
    )
    # A sign change is re-confirmed on a fresh frame before commanding J6.
    assert planner.next_action(crossed).kind is ActionKind.OBSERVE
    corrected = planner.next_action(crossed)
    assert corrected.kind is ActionKind.CAMERA_ROLL
    assert corrected.aim_direction[0] == -1.0
    assert corrected.angular_scale == pytest.approx(3.0)


def test_default_roll_budget_can_cover_full_quarter_turn_in_safe_steps():
    planner = make_planner()
    reach_ascend(planner)

    confirm = views(
        report(center=(0.0, 0.3), area=0.30, bottom_contact=True, orientation=89.0)
    )
    assert planner.next_action(confirm).kind is ActionKind.OBSERVE
    # Each fresh frame still has a reliable long edge.  A near-90-degree
    # initial error needs more than the old three-move budget when every direct
    # J6 command is deliberately capped at 0.30 rad.
    for orientation_deg in (89.0, 72.0, 55.0, 38.0, 21.0, 13.0):
        action = planner.next_action(
            views(
                report(
                    center=(0.0, 0.3),
                    area=0.30,
                    bottom_contact=True,
                    orientation=orientation_deg,
                )
            )
        )
        assert action.kind is ActionKind.CAMERA_ROLL
        assert 0.0 < action.angular_scale <= 3.0


def test_roll_assist_when_framed_but_rotated():
    planner = make_planner()
    reach_ascend(planner)
    framed = views(report(area=0.20, rectangularity=0.4, orientation=40.0))
    assert planner.next_action(framed).kind is ActionKind.OBSERVE
    assert planner.next_action(framed).kind is ActionKind.CAMERA_ROLL


def test_negative_orientation_rolls_negative_first():
    planner = make_planner()
    reach_ascend(planner)
    rotated = views(
        report(area=0.30, bottom_contact=True, orientation=-30.0)
    )
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    action = planner.next_action(rotated)
    assert action.kind is ActionKind.CAMERA_ROLL
    assert action.aim_direction[0] == -1.0


def test_aligned_board_never_rolls():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(
        views(report(area=0.30, bottom_contact=True, orientation=5.0))
    )
    assert action.kind is ActionKind.BACKOFF


def test_ambiguous_near_square_mask_never_chooses_an_edge_to_roll():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(
        views(
            report(
                area=0.30,
                bottom_contact=True,
                orientation=45.0,
                long_axis_ratio=1.05,
            )
        )
    )
    assert action.kind is ActionKind.BACKOFF
    assert "long/short edge ambiguous" in action.reason


def test_ambiguous_j6_uses_j1_when_horizontal_error_has_a_direction():
    planner = make_planner()
    reach_ascend(planner)
    action = planner.next_action(
        views(
            report(
                center=(0.25, 0.3),
                area=0.30,
                bottom_contact=True,
                orientation=45.0,
                long_axis_ratio=1.05,
            )
        )
    )
    assert action.kind is ActionKind.BASE_YAW
    assert action.aim_direction[0] == -1.0
    assert "J1 fallback" in action.reason


def test_ambiguous_centered_j6_zoom_is_bounded_then_changes_clearance_axis():
    planner = make_planner(max_zoom_out_backoffs=2)
    reach_ascend(planner)
    ambiguous = views(
        report(
            center=(0.0, 0.3),
            area=0.30,
            bottom_contact=True,
            orientation=45.0,
            long_axis_ratio=1.05,
        )
    )
    assert planner.next_action(ambiguous).kind is ActionKind.BACKOFF
    assert planner.next_action(ambiguous).kind is ActionKind.BACKOFF
    changed = planner.next_action(ambiguous)
    assert changed.kind is ActionKind.UP_CLEARANCE
    assert "instead of repeating" in changed.reason


def test_reliable_long_edge_after_zoom_returns_to_j6_alignment():
    planner = make_planner()
    reach_ascend(planner)
    ambiguous = views(
        report(
            area=0.30,
            bottom_contact=True,
            orientation=45.0,
            long_axis_ratio=1.05,
        )
    )
    assert planner.next_action(ambiguous).kind is ActionKind.BACKOFF
    reliable = views(
        report(
            area=0.24,
            bottom_contact=True,
            orientation=-25.0,
            long_axis_ratio=1.35,
        )
    )
    assert planner.next_action(reliable).kind is ActionKind.OBSERVE
    action = planner.next_action(reliable)
    assert action.kind is ActionKind.CAMERA_ROLL
    assert action.aim_direction[0] == -1.0


def test_aligned_lower_edge_backoff_is_bounded_instead_of_looping():
    planner = make_planner(max_zoom_out_backoffs=2)
    reach_ascend(planner)
    aligned = views(
        report(
            area=0.30,
            bottom_contact=True,
            orientation=2.0,
            long_axis_ratio=1.4,
        )
    )
    assert planner.next_action(aligned).kind is ActionKind.BACKOFF
    assert planner.next_action(aligned).kind is ActionKind.BACKOFF
    clearance = planner.next_action(aligned)
    assert clearance.kind is ActionKind.UP_CLEARANCE
    assert clearance.translation_scale == pytest.approx(2.5)


def test_tight_ivm_component_context_continues_clearance_ladder():
    planner = make_planner(max_zoom_out_backoffs=2)
    reach_ascend(planner)
    tight = views(
        report(
            area=0.20,
            orientation=2.0,
            long_axis_ratio=1.4,
            clearance=(100.0, 100.0, 100.0, 39.0),
            context_pad=36.0,
        )
    )
    assert planner.next_action(tight).kind is ActionKind.BACKOFF
    assert planner.next_action(tight).kind is ActionKind.BACKOFF
    clearance = planner.next_action(tight)
    assert clearance.kind is ActionKind.UP_CLEARANCE
    assert "IVM survey context" in clearance.reason


def test_unavailable_j6_falls_back_to_j1_for_useful_horizontal_error():
    planner = make_planner()
    reach_ascend(planner)
    rotated = views(
        report(
            center=(0.25, 0.3),
            area=0.30,
            bottom_contact=True,
            orientation=45.0,
        )
    )
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    roll = planner.next_action(rotated)
    assert roll.kind is ActionKind.CAMERA_ROLL
    planner.mark_roll_unavailable(roll, reason="joint mode rejected")

    fallback = planner.next_action(rotated)
    assert fallback.kind is ActionKind.BASE_YAW
    assert "J1 fallback" in fallback.reason


def test_unavailable_centered_j6_falls_back_to_joints_2_4_zoom():
    planner = make_planner()
    reach_ascend(planner)
    rotated = views(
        report(
            center=(0.0, 0.3),
            area=0.30,
            bottom_contact=True,
            orientation=-45.0,
        )
    )
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    roll = planner.next_action(rotated)
    assert roll.kind is ActionKind.CAMERA_ROLL
    planner.mark_roll_unavailable(roll, reason="joint mode rejected")

    fallback = planner.next_action(rotated)
    assert fallback.kind is ActionKind.BACKOFF
    assert "joint mode rejected" in fallback.reason


def test_global_j6_envelope_skips_j1_and_uses_zoom():
    planner = make_planner()
    reach_ascend(planner)
    rotated = views(
        report(
            center=(0.25, 0.3),
            area=0.30,
            bottom_contact=True,
            orientation=45.0,
        )
    )
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    roll = planner.next_action(rotated)
    assert roll.kind is ActionKind.CAMERA_ROLL
    planner.mark_roll_unavailable(
        roll,
        reason="global angular envelope exhausted",
        allow_j1_fallback=False,
    )

    fallback = planner.next_action(rotated)
    assert fallback.kind is ActionKind.BACKOFF
    assert "global angular envelope exhausted" in fallback.reason


def test_stale_or_non_roll_rejection_is_refused():
    planner = make_planner()
    reach_ascend(planner)
    rotated = views(
        report(
            center=(0.0, 0.3),
            area=0.30,
            bottom_contact=True,
            orientation=45.0,
        )
    )
    assert planner.next_action(rotated).kind is ActionKind.OBSERVE
    old = planner.next_action(rotated)
    current = planner.next_action(rotated)
    assert old.kind is current.kind is ActionKind.CAMERA_ROLL
    assert old.action_id != current.action_id
    with pytest.raises(ValueError, match="stale"):
        planner.mark_roll_unavailable(old, reason="late rejection")

    planner = make_planner()
    yaw = planner.next_action(views(report(center=(0.6, 0.0))))
    with pytest.raises(ValueError, match="camera-roll"):
        planner.mark_roll_unavailable(yaw, reason="wrong action")


# ----------------------------------------------------------------------
# Envelope rejections


def test_sweep_reverses_once_then_terminates():
    planner = make_planner()
    nothing = views(report(seen=False))
    first = planner.next_action(nothing)
    planner.mark_yaw_unavailable(first, reason="J1 envelope reached")
    assert not planner.terminal
    second = planner.next_action(nothing)
    assert second.kind is ActionKind.BASE_YAW
    assert second.aim_direction[0] == -first.aim_direction[0]
    planner.mark_yaw_unavailable(second, reason="J1 envelope reached")
    assert planner.terminal
    assert planner.next_action(nothing).kind is ActionKind.STAGNATED


def test_global_travel_exhaustion_terminates_immediately():
    planner = make_planner()
    action = planner.next_action(views(report(seen=False)))
    planner.mark_yaw_unavailable(
        action, reason="cumulative travel", global_unavailable=True
    )
    assert planner.terminal
    assert planner.next_action({}).kind is ActionKind.STAGNATED


def test_centering_envelope_rejection_falls_back_to_ascend_once():
    planner = make_planner()
    clipped = views(report(center=(0.7, 0.0), area=0.30, edges=("left",)))
    action = planner.next_action(clipped)
    assert action.kind is ActionKind.BASE_YAW
    planner.mark_yaw_unavailable(action, reason="J1 envelope reached")
    assert not planner.terminal
    fallback = planner.next_action(clipped)
    assert fallback.kind is ActionKind.UP_CLEARANCE
    # After one clearance evaluation the drift path may yaw again; a second
    # envelope rejection is terminal.
    again = planner.next_action(clipped)
    assert again.kind is ActionKind.BASE_YAW
    planner.mark_yaw_unavailable(again, reason="J1 envelope reached")
    assert planner.terminal
    assert planner.next_action(clipped).kind is ActionKind.STAGNATED


def test_evidence_resets_sweep_reversal_budget():
    planner = make_planner()
    nothing = views(report(seen=False))
    first = planner.next_action(nothing)
    planner.mark_yaw_unavailable(first, reason="J1 envelope reached")
    planner.next_action(views(report(center=(0.6, 0.0))))
    # Board evidence appeared, so a later acquisition loss gets a fresh budget.
    lost = planner.next_action(nothing)
    planner.mark_yaw_unavailable(lost, reason="J1 envelope reached")
    assert not planner.terminal


def test_stale_yaw_rejection_raises():
    planner = make_planner()
    old = planner.next_action(views(report(seen=False)))
    current = planner.next_action(views(report(seen=False)))
    assert current.action_id != old.action_id
    with pytest.raises(ValueError):
        planner.mark_yaw_unavailable(old, reason="late rejection")


def test_non_yaw_rejection_raises():
    planner = make_planner()
    action = planner.next_action(views(report(seen=False, logo=True)))
    assert action.kind is ActionKind.UP_CLEARANCE
    with pytest.raises(ValueError):
        planner.mark_yaw_unavailable(action, reason="not a yaw")


# ----------------------------------------------------------------------
# Constructor validation


def test_constructor_rejects_invalid_limits():
    with pytest.raises(ValueError):
        make_planner(min_goal_area_frac=0.5, max_goal_area_frac=0.4)
    with pytest.raises(ValueError):
        make_planner(expected_cameras=())
    with pytest.raises(ValueError):
        make_planner(center_threshold=0.5, recenter_threshold=0.4)
    with pytest.raises(ValueError):
        make_planner(confirmation_frames=0)
    with pytest.raises(ValueError):
        make_planner(yaw_gain=0.0)
    with pytest.raises(ValueError):
        make_planner(min_yaw_scale=0.5, max_yaw_scale=0.4)
    with pytest.raises(ValueError):
        make_planner(max_ascend_scale=0.5)
    with pytest.raises(ValueError):
        make_planner(max_stall_frames=0)
    with pytest.raises(ValueError):
        make_planner(roll_align_threshold_deg=0.0)
    with pytest.raises(ValueError):
        make_planner(max_roll_moves=-1)
    with pytest.raises(ValueError):
        make_planner(roll_probe_scale=8.0, max_roll_scale=6.0)
    with pytest.raises(ValueError):
        make_planner(min_long_axis_ratio=1.0)
    with pytest.raises(ValueError):
        make_planner(roll_confirmation_frames=0)
    with pytest.raises(ValueError):
        make_planner(max_zoom_out_backoffs=-1)
