"""Stage-1 image-plane seek policy.

Pure functions over synthetic reports, so the steering logic can be verified
without a robot.  The frames in the docstrings are the real ones from the
2026-07-27 hardware logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aic_perception.board_seek import (
    board_view_ready,
    pick_best_camera,
    select_work_target,
    CENTER_ERROR_TRIGGER,
    image_plane_direction,
    seek_progress_score,
    SEEK_HARD_MOVE_CEILING,
    SEEK_STALL_MOVES,
)


@dataclass(frozen=True)
class FakeReport:
    """Minimal stand-in for a board MaskReport / a PurpleReport.

    ``quality_score`` is present because ``pick_best_camera`` ranks on
    ``board_visibility.view_quality``, which reads it.
    """

    seen: bool = True
    full: bool = False
    edges: frozenset = field(default_factory=frozenset)
    area_frac: float = 0.30
    center_error: tuple = (0.0, 0.0)
    quality_score: float = 0.5


def board(edges=(), center=(0.0, 0.0), area=0.30, seen=True):
    return FakeReport(
        seen=seen, edges=frozenset(edges), center_error=center, area_frac=area
    )


# ---------------------------------------------------------------------------
# Steering direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edge, expected",
    (
        ("left", (-1.0, 0.0)),
        ("right", (1.0, 0.0)),
        ("top", (0.0, -1.0)),
        ("bottom", (0.0, 1.0)),
    ),
)
def test_a_single_clipped_edge_steers_toward_it(edge, expected):
    """One clipped edge is the strongest cue: move the camera that way.

    The direction is where the *camera* travels in image axes, so clipping on
    the left means moving image-left to bring the overflowing content in.
    """
    assert image_plane_direction(board(edges=(edge,))) == pytest.approx(expected)


def test_opposite_clipped_edges_fall_back_to_centre_error():
    """Clipped both sides means the target overflows that axis entirely.

    There is no "toward the edge" answer left, so the centre error decides.
    """
    report = board(edges=("left", "right"), center=(0.4, 0.0))
    assert image_plane_direction(report) == pytest.approx((1.0, 0.0))


def test_a_centred_unclipped_target_has_no_signal():
    assert image_plane_direction(board()) is None


def test_centre_error_below_the_trigger_is_not_a_signal():
    """Small residual error must not keep the loop nudging forever."""
    small = CENTER_ERROR_TRIGGER / 2.0
    assert image_plane_direction(board(center=(small, small))) is None


def test_the_hardware_stuck_frame_still_yields_a_direction():
    """Centre camera at 16:39:30: edges left+right+top, centre (0.034, -0.596).

    Left and right are both clipped so the X axis falls back to a centre error
    of 0.034, which is under the trigger and contributes nothing.  Top alone is
    clipped, so the whole signal is vertical.  This is the frame where the
    board overflows three sides -- the policy still moves, but note it
    translates rather than backing off.
    """
    report = board(edges=("left", "right", "top"), center=(0.034, -0.596))
    assert image_plane_direction(report) == pytest.approx((0.0, -1.0))


def test_board_view_ready_requires_no_clipped_edges():
    assert board_view_ready(board())
    assert not board_view_ready(board(edges=("top",)))
    assert not board_view_ready(board(seen=False))


# ---------------------------------------------------------------------------
# Progress score / stall detection
# ---------------------------------------------------------------------------


def test_any_purple_outranks_every_purple_free_view():
    """The insignia is the goal; the board is only a proxy for finding it."""
    perfect_board = {"center_camera": board(edges=(), center=(0.0, 0.0))}
    no_purple = {"center_camera": FakeReport(seen=False)}
    barely_purple = {"center_camera": FakeReport(seen=True, area_frac=0.001)}

    assert seek_progress_score(perfect_board, barely_purple) > seek_progress_score(
        perfect_board, no_purple
    )


def test_unclipping_the_board_is_progress():
    worse = {"center_camera": board(edges=("left", "right", "top"))}
    better = {"center_camera": board(edges=("top",))}
    none = {"center_camera": FakeReport(seen=False)}

    assert seek_progress_score(better, none) > seek_progress_score(worse, none)


def test_centring_the_board_is_progress_at_equal_clipping():
    off = {"center_camera": board(edges=("top",), center=(0.6, 0.3))}
    on = {"center_camera": board(edges=("top",), center=(0.1, 0.0))}
    none = {"center_camera": FakeReport(seen=False)}

    assert seek_progress_score(on, none) > seek_progress_score(off, none)


def test_more_cameras_seeing_purple_is_progress():
    none = {"center_camera": FakeReport(seen=False)}
    one = {"center_camera": FakeReport(seen=True, area_frac=0.01)}
    two = {
        "center_camera": FakeReport(seen=True, area_frac=0.01),
        "left_camera": FakeReport(seen=True, area_frac=0.01),
    }
    assert seek_progress_score(none, two) > seek_progress_score(none, one)


def test_losing_the_board_entirely_scores_worst():
    lost = {"center_camera": FakeReport(seen=False)}
    kept = {"center_camera": board(edges=("left", "right", "top", "bottom"))}
    none = {"center_camera": FakeReport(seen=False)}

    assert seek_progress_score(lost, none) < seek_progress_score(kept, none)


def test_termination_backstop_is_generous_but_finite():
    """The stall detector should always end the search first."""
    assert SEEK_STALL_MOVES >= 2
    assert SEEK_HARD_MOVE_CEILING > 10 * SEEK_STALL_MOVES


# ---------------------------------------------------------------------------
# Target selection
#
# These execute select_work_target rather than inspecting source.  An earlier
# revision imported `any_purple_seen`/`pick_purple_camera` from a trimmed
# `purple_insignia` that never defined them; every source-level test still
# passed and the skill died with an ImportError on the robot.  Calling the
# function is the only thing that catches that.
# ---------------------------------------------------------------------------


def purple(seen=True, full=False, center=(0.0, 0.0), area=0.01):
    return FakeReport(
        seen=seen, full=full, edges=frozenset(), center_error=center,
        area_frac=area,
    )


def test_select_work_target_steers_on_the_board_before_any_purple():
    boards = {
        "center_camera": board(edges=("top",)),
        "left_camera": board(edges=("top", "right")),
    }
    purples = {name: purple(seen=False) for name in boards}

    mode, camera, report = select_work_target(boards, purples)

    assert mode == "board"
    # Fewer clipped edges wins.
    assert camera == "center_camera"
    assert report is boards["center_camera"]


def test_select_work_target_switches_to_purple_as_soon_as_one_camera_sees_it():
    boards = {name: board(edges=("top",)) for name in ("center_camera", "left_camera")}
    purples = {
        "center_camera": purple(seen=False),
        "left_camera": purple(seen=True, area=0.004),
    }

    mode, camera, report = select_work_target(boards, purples)

    assert mode == "purple"
    assert camera == "left_camera"
    assert report is purples["left_camera"]


def test_purple_selection_prefers_the_camera_that_still_needs_work():
    boards = {name: board() for name in ("center_camera", "left_camera")}
    purples = {
        # Already framed and centred -- nothing to do here.
        "center_camera": purple(full=True, center=(0.0, 0.0)),
        # Clipped: this is the one worth steering on.
        "left_camera": FakeReport(
            seen=True, full=False, edges=frozenset({"right"}),
            center_error=(0.5, 0.0), area_frac=0.01,
        ),
    }

    mode, camera, _ = select_work_target(boards, purples)

    assert mode == "purple"
    assert camera == "left_camera"


def test_select_work_target_reports_nothing_when_no_camera_sees_anything():
    blind = {"center_camera": FakeReport(seen=False)}
    mode, camera, report = select_work_target(blind, blind)

    assert mode == "board"
    assert camera is None and report is None


def test_pick_best_camera_prefers_an_unclipped_view():
    reports = {
        "center_camera": board(edges=("top",), area=0.9),
        "left_camera": board(edges=(), area=0.1),
    }
    name, _ = pick_best_camera(reports)
    assert name == "left_camera"
