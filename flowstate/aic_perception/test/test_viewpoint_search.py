from __future__ import annotations

import pytest

from aic_perception.board_visibility import MaskReport
from aic_perception.viewpoint_search import (
    ActionKind,
    AdaptiveViewpointPlanner,
)


def report(
    *,
    edges: tuple[str, ...] = ("bottom",),
    area: float = 0.12,
    rectangularity: float = 0.65,
    seen: bool = True,
    full: bool = False,
) -> MaskReport:
    return MaskReport(
        seen=seen,
        full=full,
        edges=frozenset(edges),
        area_frac=area,
        rectangularity=rectangularity,
    )


def test_same_improving_bottom_edge_changes_action_modes():
    planner = AdaptiveViewpointPlanner()

    first = planner.next_action({"center_camera": report(area=0.12)})
    second = planner.next_action({"center_camera": report(area=0.14)})
    third = planner.next_action({"center_camera": report(area=0.16)})

    assert first.kind == ActionKind.TRANSLATE
    assert first.image_direction == pytest.approx((0.0, 1.0))
    assert second.kind == ActionKind.COMBINED
    assert second.image_direction == pytest.approx((0.0, 1.0))
    assert second.aim_direction == pytest.approx((0.0, 1.0))
    assert second.axial_direction > 0.0
    assert third.kind == ActionKind.AIM
    assert len({first.kind, second.kind, third.kind}) == 3


def test_edge_change_changes_image_direction():
    planner = AdaptiveViewpointPlanner()

    down = planner.next_action(
        {"center_camera": report(edges=("bottom",), area=0.12)}
    )
    right = planner.next_action(
        {"center_camera": report(edges=("right",), area=0.12)}
    )

    assert down.image_direction == pytest.approx((0.0, 1.0))
    assert right.kind == ActionKind.TRANSLATE
    assert right.image_direction == pytest.approx((1.0, 0.0))


def test_regression_requests_rollback_then_chooses_an_alternative():
    planner = AdaptiveViewpointPlanner()

    first = planner.next_action({"center_camera": report(area=0.18)})
    rollback = planner.next_action(
        {
            "center_camera": report(
                area=0.08,
                rectangularity=0.40,
            )
        }
    )

    assert first.kind == ActionKind.TRANSLATE
    assert rollback.kind == ActionKind.ROLLBACK
    assert rollback.request_rollback
    assert rollback.rollback_of == first.action_id
    assert not rollback.terminal

    alternative = planner.next_action({"center_camera": report(area=0.18)})
    assert alternative.moves_robot
    assert alternative.kind not in {ActionKind.ROLLBACK, first.kind}
    assert planner.blacklisted_action_count == 1


def test_more_than_six_improving_actions_do_not_terminate():
    planner = AdaptiveViewpointPlanner()
    area = 0.10
    rectangularity = 0.45
    actions = []

    for _ in range(10):
        action = planner.next_action(
            {
                "center_camera": report(
                    area=area,
                    rectangularity=rectangularity,
                )
            }
        )
        actions.append(action)
        assert not action.terminal
        # Supply action-consistent progress: a retreat reduces scale, while
        # centering/aiming reveals more of the clipped board.
        if action.kind == ActionKind.BACKOFF:
            area -= 0.01
        else:
            area += 0.01
        rectangularity = min(0.98, rectangularity + 0.03)

    assert len(actions) > 6
    assert not planner.terminal
    assert len({action.kind for action in actions}) >= 4


def test_stagnation_exhausts_safe_alternatives():
    planner = AdaptiveViewpointPlanner()
    unchanged = {"center_camera": report(area=0.12, rectangularity=0.65)}
    attempted = []

    for _ in range(10):
        action = planner.next_action(unchanged)
        if action.terminal:
            break
        attempted.append(action.kind)
    else:  # pragma: no cover - makes an accidental infinite planner obvious
        pytest.fail("planner did not terminate after exhausting alternatives")

    assert action.kind == ActionKind.STAGNATED
    assert set(attempted) == {
        ActionKind.TRANSLATE,
        ActionKind.COMBINED,
        ActionKind.AIM,
        ActionKind.BACKOFF,
    }
    assert planner.blacklisted_action_count == 4


def test_camera_selection_uses_hysteresis_but_switches_for_clear_gain():
    planner = AdaptiveViewpointPlanner(camera_switch_margin=0.5)
    first = planner.next_action(
        {
            "center_camera": report(area=0.14, rectangularity=0.70),
            "right_camera": report(area=0.13, rectangularity=0.69),
        }
    )
    assert first.camera == "center_camera"

    held = planner.next_action(
        {
            "center_camera": report(area=0.16, rectangularity=0.72),
            "right_camera": report(area=0.17, rectangularity=0.74),
        }
    )
    assert held.camera == "center_camera"

    switched = planner.next_action(
        {
            "center_camera": report(area=0.18, rectangularity=0.74),
            "right_camera": report(area=0.32, rectangularity=0.96),
        }
    )
    assert switched.camera == "right_camera"


def test_goal_scale_requests_approach_or_done_and_deadline_is_explicit():
    too_small = AdaptiveViewpointPlanner(min_goal_area_frac=0.08)
    approach = too_small.next_action(
        {"center_camera": report(edges=(), area=0.03, full=True)}
    )
    assert approach.kind == ActionKind.APPROACH
    assert approach.axial_direction < 0.0

    ready = AdaptiveViewpointPlanner()
    done = ready.next_action(
        {"center_camera": report(edges=(), area=0.15, full=True)}
    )
    assert done.kind == ActionKind.DONE
    assert done.terminal

    timed = AdaptiveViewpointPlanner()
    deadline = timed.next_action({}, deadline_reached=True)
    assert deadline.kind == ActionKind.DEADLINE
    assert deadline.terminal
