"""Image-plane board/insignia seeking policy for Stage 1.

Ported from the ``move_to_board_skill`` v3 experiment on branch
``navigate-to-purple`` (tip ``4a20097``), which is the only Stage-1 search that
has actually driven this arm toward the insignia.  ROS-free on purpose so the
whole policy is unit-testable against synthetic reports.

Why this shape, and not the two designs it replaces:

* The **phase machine** (``viewpoint_search.AdaptiveViewpointPlanner``,
  ACQUIRE -> CENTER -> ALIGN -> LEVEL -> ASCEND) consumed a board orientation
  that is degenerate exactly when it is needed: a mask clipped on two or more
  image edges yields a frame-aligned ``minAreaRect``, logged as
  ``long_ratio=1.00 long_axis_error=+0.0deg``.  It also split J1/J6 authority
  across phases, which coupled badly at the levelled pose.
* The **deterministic joint plan** commanded large reconfigurations to a fixed
  observation pose.  On hardware those failed to execute at all: first the
  profile outran its deadline, then the controller dropped out of joint target
  mode 0.43 s in (``controller left joint target mode; joint target reversed``).

This policy issues **small Cartesian image-plane translations at fixed
orientation** instead.  It never asks for joint target mode, never uses the
degenerate orientation cue, and has no phases to sequence: each step picks the
camera that most needs work and nudges the view toward centring it.
"""

from __future__ import annotations

import math

# Board must be inside the frame with a small fixed margin; this is the seek
# criterion, not the stricter Stage-2 survey framing.
BOARD_MARGIN_PX = 10
BOARD_CONTEXT_PAD_FRAC = 0.0
# One image-plane hop.  Small enough that its Cartesian profile always fits the
# move deadline, which is what the joint-plan approach could not guarantee.
CENTER_STEP_M = 0.03
CENTER_ERROR_TRIGGER = 0.10
# The seek runs until it stops making progress, not for a fixed number of
# hops: a corner start legitimately needs many more moves than a near-framed
# one, and any constant is wrong for one of them.  ``SEEK_STALL_MOVES``
# consecutive non-improving moves ends it.
SEEK_STALL_MOVES = 4
# Absolute backstop so the loop provably terminates.  Stage 1 has no aggregate
# wall clock, so an unbounded loop would hang the skill and with it the whole
# Flowstate process; the stall detector is what actually stops the search, and
# this should never bind.
SEEK_HARD_MOVE_CEILING = 200
MAX_SPEED_MPS = 0.055
SETTLE_TOLERANCE_M = 0.006
MOVE_TIMEOUT_SEC = 8.0


def board_view_ready(report) -> bool:
    """True when the board is in frame with the seek margin."""
    return bool(report.seen and not report.edges)


def pick_best_camera(reports, *, preferred=None):
    """Pick the camera to centre the board on, before any purple is visible.

    Prefer fewer clipped edges over raw area: an almost-framed view is worth
    more than a larger but messier silhouette.  Ties stick to the current work
    camera so the loop does not oscillate between cameras.
    """
    from aic_perception.board_visibility import view_quality

    seen = [(name, report) for name, report in reports.items() if report.seen]
    if not seen:
        return None, None

    def rank(item):
        name, report = item
        return (
            1 if board_view_ready(report) else 0,
            -len(report.edges),
            1 if preferred is not None and name == preferred else 0,
            1 if name == "center_camera" else 0,
            float(report.area_frac),
            float(view_quality(report)),
        )

    return max(seen, key=rank)


def image_plane_direction(report):
    """Unit (image-right, image-down) direction that centres a target.

    Works for a board ``MaskReport`` or a ``PurpleReport`` -- both expose
    ``edges`` and ``center_error``.  A single clipped edge is the strongest cue
    and is preferred; when opposite edges are both clipped the target overflows
    that axis entirely and the centre error is used instead.  ``None`` means
    there is no signal left on either axis.
    """

    def axis_sign(neg_edge, pos_edge, center_value):
        has_neg = neg_edge in report.edges
        has_pos = pos_edge in report.edges
        if has_neg and not has_pos:
            return -1.0
        if has_pos and not has_neg:
            return 1.0
        if abs(center_value) >= CENTER_ERROR_TRIGGER:
            return 1.0 if center_value > 0.0 else -1.0
        return 0.0

    dx = axis_sign("left", "right", float(report.center_error[0]))
    dy = axis_sign("top", "bottom", float(report.center_error[1]))
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    return (dx / norm, dy / norm)


def select_work_target(board_reports, purple_reports, *, preferred=None):
    """Choose the next thing to centre: ``("purple"|"board", camera, report)``.

    Purple wins the moment any camera sees it, because the insignia is the
    actual goal and the board is only a proxy for finding it.
    """
    from aic_perception.purple_insignia import any_purple_seen, pick_purple_camera

    if any_purple_seen(purple_reports):
        camera, report = pick_purple_camera(purple_reports, preferred=preferred)
        return "purple", camera, report
    camera, report = pick_best_camera(board_reports, preferred=preferred)
    return "board", camera, report


def seek_progress_score(board_reports, purple_reports) -> float:
    """How close this observation is to a Stage-2-usable view. Higher is better.

    Drives stall detection, so it only has to be *monotone in the right
    direction* -- it is not an objective anything optimises directly.

    Purple visible dominates everything: the insignia is the goal and the board
    is only a proxy for finding it.  Within that, more cameras seeing it and
    more of it unclipped both count.  Before any purple appears, progress means
    un-clipping the board, because a board that fits the frame is what reveals
    where the insignia is -- and it is also what makes the mask's orientation
    stop being degenerate.
    """
    visible_purple = [report for report in purple_reports.values() if report.seen]
    if visible_purple:
        unclipped = sum(1 for report in visible_purple if report.full)
        largest = max(report.area_frac for report in visible_purple)
        return (
            1000.0
            + 100.0 * len(visible_purple)
            + 50.0 * unclipped
            + 10.0 * largest
        )

    visible_board = [report for report in board_reports.values() if report.seen]
    if not visible_board:
        return -1000.0
    fewest_edges = min(len(report.edges) for report in visible_board)
    best_centering = min(
        abs(float(report.center_error[0])) + abs(float(report.center_error[1]))
        for report in visible_board
    )
    return -10.0 * fewest_edges - best_centering
