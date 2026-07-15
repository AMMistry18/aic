"""Camera-feedback policy for the four-stage task-board search.

The sequence is deliberately fixed by the workcell geometry: align with J1,
increase J2 clearance until the lower board is visible, use J3 clearance for
the upper edge, then use J4 camera-roll for fine framing.  Each stage is
*feedback directed*: its first relative direction is a probe and the opposite
direction is selected when the camera evidence regresses.  A single camera
with a complete padded board view is sufficient to finish; the other cameras
are steering evidence, not a mandatory pixel-stitching requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

from .board_visibility import MaskReport


class ActionKind(str, Enum):
    OBSERVE = "observe"                # fresh-frame confirmation; no motion
    BASE_YAW = "base_yaw"              # J1-equivalent base yaw arc
    BACKOFF = "backoff"                # J2-equivalent optical-axis clearance
    UP_CLEARANCE = "up_clearance"      # J3-equivalent vertical clearance
    CAMERA_ROLL = "camera_roll"        # J4-equivalent optical-axis roll
    # Retained for compatibility with older serialized diagnostics.
    TRANSLATE = "translate"
    APPROACH = "approach"
    AIM = "aim"
    COMBINED = "combined"
    HORIZONTAL_SCAN = "horizontal_scan"
    ROLLBACK = "rollback"
    DONE = "done"
    STAGNATED = "stagnated"
    NO_VIEW = "no_view"
    DEADLINE = "deadline"


_MOVEMENT_KINDS = frozenset({
    ActionKind.BASE_YAW, ActionKind.BACKOFF, ActionKind.UP_CLEARANCE,
    ActionKind.CAMERA_ROLL, ActionKind.TRANSLATE, ActionKind.APPROACH,
    ActionKind.AIM, ActionKind.COMBINED, ActionKind.HORIZONTAL_SCAN,
})


class _Phase(str, Enum):
    J1_YAW = "j1_yaw_alignment"
    J2_CLEARANCE = "j2_bottom_clearance"
    J3_CLEARANCE = "j3_top_clearance"
    J4_ROLL = "j4_camera_roll"


@dataclass(frozen=True)
class ViewpointAction:
    action_id: int
    kind: ActionKind
    camera: str | None = None
    image_direction: tuple[float, float] = (0.0, 0.0)
    axial_direction: float = 0.0
    aim_direction: tuple[float, float] = (0.0, 0.0)
    translation_scale: float = 1.0
    angular_scale: float = 0.0
    request_rollback: bool = False
    rollback_of: int | None = None
    terminal: bool = False
    reason: str = ""

    @property
    def moves_robot(self) -> bool:
        return self.kind in _MOVEMENT_KINDS or self.request_rollback


@dataclass(frozen=True)
class _Pending:
    action: ViewpointAction
    phase: _Phase
    score: float


def image_direction_for_edges(edges: frozenset[str]) -> tuple[float, float]:
    horizontal = (-1.0 if "left" in edges else 0.0) + (1.0 if "right" in edges else 0.0)
    vertical = (-1.0 if "top" in edges else 0.0) + (1.0 if "bottom" in edges else 0.0)
    magnitude = math.hypot(horizontal, vertical)
    return (horizontal / magnitude, vertical / magnitude) if magnitude else (0.0, 0.0)


def has_opposite_edges(edges: frozenset[str]) -> bool:
    return ("left" in edges and "right" in edges) or ("top" in edges and "bottom" in edges)


class AdaptiveViewpointPlanner:
    """Execute J1 -> J2 -> J3 -> J4 using the best stable camera view."""

    def __init__(
        self,
        *,
        min_goal_area_frac: float = 0.04,
        max_goal_area_frac: float = 0.45,
        expected_cameras: Sequence[str] = ("left_camera", "center_camera", "right_camera"),
        horizontal_coverage_area_frac: float | None = None,
        yaw_probe_scale: float = 0.75,
        coverage_improvement_tolerance: float = 0.02,
        camera_switch_margin: float = 0.35,
        improvement_tolerance: float = 0.025,
        regression_tolerance: float = 0.04,
        area_quantum: float = 0.025,
        rectangularity_quantum: float = 0.10,
        horizontal_center_threshold: float = 0.10,
        horizontal_exit_threshold: float = 0.20,
        alignment_confirmation_frames: int = 2,
        clearance_drift_tolerance: float = 0.08,
    ) -> None:
        if not 0.0 <= min_goal_area_frac < max_goal_area_frac <= 1.0:
            raise ValueError("goal area limits must satisfy 0 <= min < max <= 1")
        cameras = tuple(str(item) for item in expected_cameras)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("expected_cameras must be non-empty and unique")
        self.min_goal_area_frac = float(min_goal_area_frac)
        self.max_goal_area_frac = float(max_goal_area_frac)
        self.expected_cameras = cameras
        self.horizontal_coverage_area_frac = float(horizontal_coverage_area_frac or min_goal_area_frac)
        self.yaw_probe_scale = float(yaw_probe_scale)
        self.improvement_tolerance = float(coverage_improvement_tolerance)
        self.horizontal_center_threshold = float(horizontal_center_threshold)
        self.horizontal_exit_threshold = float(horizontal_exit_threshold)
        self.alignment_confirmation_frames = int(alignment_confirmation_frames)
        self.clearance_drift_tolerance = float(clearance_drift_tolerance)
        if self.alignment_confirmation_frames < 1:
            raise ValueError("alignment_confirmation_frames must be positive")
        if self.clearance_drift_tolerance < 0.0:
            raise ValueError("clearance_drift_tolerance must be non-negative")
        self.reset()

    def reset(self) -> None:
        self._phase = _Phase.J1_YAW
        self._pending: _Pending | None = None
        self._next_action_id = 1
        self._terminal_action: ViewpointAction | None = None
        self._selected_camera: str | None = None
        self._direction = 0.0
        self._phase_moves = 0
        self._yaw_globally_unavailable = False
        self._yaw_confirmed_moves = 0
        self._yaw_reversal_candidate = 0.0
        self._yaw_reversal_streak = 0
        self._j1_alignment_streak = 0
        self._j1_latched_error: float | None = None
        self._current_score = 0.0

    @property
    def selected_camera(self) -> str | None:
        return self._selected_camera

    @property
    def terminal(self) -> bool:
        return self._terminal_action is not None

    @property
    def phase(self) -> str:
        return self._phase.value

    @property
    def coverage_achieved(self) -> bool:
        return False  # compatibility: completion is intentionally per-camera.

    @property
    def blacklisted_action_count(self) -> int:
        return 0

    def report_quality(self, report: MaskReport) -> float:
        if not report.seen:
            return -100.0
        return (10.0 + 12.0 * float(report.full) + 2.0 * min(1.0, report.area_frac / self.max_goal_area_frac)
                + min(1.0, report.rectangularity) - 1.5 * len(report.edges)
                - 0.6 * abs(report.center_error[0]) - 0.3 * abs(report.center_error[1])
                - 2.0 * float(report.artificial_bottom_contact))

    def next_action(self, reports: Mapping[str, MaskReport], *, deadline_reached: bool = False) -> ViewpointAction:
        if self._terminal_action is not None:
            return self._terminal_action
        if deadline_reached:
            return self._terminate(ActionKind.DEADLINE, "viewpoint-search deadline reached before a complete view")
        goal = self._goal_camera(reports)
        if goal is not None:
            self._selected_camera = goal
            return self._terminate(ActionKind.DONE, f"complete usable board view in {goal}", goal)

        center = reports.get("center_camera")
        if center is None or not center.seen:
            pending = self._pending
            self._pending = None
            self._j1_alignment_streak = 0
            if pending is not None and pending.action.kind is ActionKind.BASE_YAW:
                previous = math.copysign(
                    1.0, pending.action.aim_direction[0] or 1.0
                )
                candidate = -previous
                if candidate == self._yaw_reversal_candidate:
                    self._yaw_reversal_streak += 1
                else:
                    self._yaw_reversal_candidate = candidate
                    self._yaw_reversal_streak = 1
                if self._yaw_reversal_streak >= 2:
                    self._direction = candidate
                    self._yaw_reversal_candidate = 0.0
                    self._yaw_reversal_streak = 0
                else:
                    self._direction = previous
                self._phase_moves += 1
            if self._phase is not _Phase.J1_YAW:
                self._advance(_Phase.J1_YAW)
            direction = self._side_camera_yaw_hint(reports)
            return self._emit(ActionKind.BASE_YAW, "center_camera", aim=(direction, 0.0), angular=0.75,
                              reason="J1 acquisition sweep: center camera lost the board; reverse/continue yaw")
        primary_name = "center_camera"
        primary = center
        self._selected_camera = primary_name
        self._consume_pending(primary)
        self._current_score = self._phase_score(primary, self._phase)

        # A small/partial center-camera mask can steer acquisition, but it is
        # never sufficient to finish J1. Scoring it here lets two repeated
        # regressions reverse the sweep without alternating every frame.
        if not self._credible_alignment_report(primary):
            self._j1_alignment_streak = 0
            if self._phase is not _Phase.J1_YAW:
                self._advance(_Phase.J1_YAW)
                self._current_score = self._phase_score(primary, self._phase)
            sign = self._choose_yaw_sign(primary)
            return self._emit(
                ActionKind.BASE_YAW,
                primary_name,
                aim=(sign, 0.0),
                angular=self._yaw_scale(primary),
                reason=(
                    "J1 acquisition sweep: center-camera board mask is too "
                    "small for an alignment decision"
                ),
            )

        # J2/J3/J4 can change the projected horizontal location.  Never keep
        # rolling or changing clearance after the center camera has lost the
        # actual board.  Restart the prescribed workflow at J1 immediately.
        if (
            self._phase is not _Phase.J1_YAW
            and (
                not self._yaw_aligned(
                    primary,
                    threshold=self.horizontal_exit_threshold,
                )
                or self._clearance_horizontal_drifted(primary)
            )
        ):
            self._advance(_Phase.J1_YAW)
            self._current_score = self._phase_score(primary, self._phase)

        # The exact requested ordering.  A phase can advance early once its
        # visual condition is satisfied; it never waits for all three cameras.
        if self._phase is _Phase.J1_YAW:
            if self._yaw_aligned(primary):
                self._j1_alignment_streak += 1
                if self._j1_alignment_streak < self.alignment_confirmation_frames:
                    # No motion occurs during confirmation, so discard the
                    # previous move direction. If the next fresh frame falls
                    # outside the gate, its center/edge evidence chooses the
                    # correction instead of blindly continuing stale yaw.
                    self._direction = 0.0
                    self._yaw_reversal_candidate = 0.0
                    self._yaw_reversal_streak = 0
                    return self._emit(
                        ActionKind.OBSERVE,
                        primary_name,
                        reason=(
                            "J1 alignment candidate: capture a second fresh "
                            "center-camera frame before changing joints"
                        ),
                    )
                self._j1_latched_error = abs(float(primary.center_error[0]))
                self._advance(_Phase.J2_CLEARANCE)
                self._current_score = self._phase_score(primary, self._phase)
            else:
                self._j1_alignment_streak = 0
                sign = self._choose_yaw_sign(primary)
                return self._emit(ActionKind.BASE_YAW, primary_name, aim=(sign, 0.0), angular=self._yaw_scale(primary),
                                  reason="J1 horizontal yaw: center the board in the center camera")
        if self._phase is _Phase.J2_CLEARANCE:
            if self._clearance_ready(primary):
                self._advance(_Phase.J3_CLEARANCE)
                self._current_score = self._phase_score(primary, self._phase)
            else:
                sign = self._choose_relative_sign(primary, default=1.0)
                return self._emit(ActionKind.BACKOFF, primary_name, axial=sign, scale=1.0,
                                  reason="J2 relative clearance: expose the lower board edge")
        if self._phase is _Phase.J3_CLEARANCE:
            if self._top_visible(primary):
                self._advance(_Phase.J4_ROLL)
                self._current_score = self._phase_score(primary, self._phase)
            else:
                sign = self._choose_relative_sign(primary, default=1.0)
                return self._emit(ActionKind.UP_CLEARANCE, primary_name, axial=sign, scale=1.0,
                                  reason="J3 relative clearance: expose the upper board edge")
        # J4 is a camera roll, not a pitch.  The wrapper maps this to a roll
        # about the selected optical axis; its sign is checked after each view.
        if self._phase is _Phase.J4_ROLL:
            if self._phase_moves >= 4:
                self._advance(_Phase.J1_YAW)
                self._current_score = self._phase_score(primary, self._phase)
            else:
                sign = self._choose_relative_sign(primary, default=1.0)
                return self._emit(ActionKind.CAMERA_ROLL, primary_name, aim=(sign, 0.0), angular=0.55,
                                  reason="J4 camera roll: fine-frame top and bottom board edges")
        return self.next_action(reports, deadline_reached=deadline_reached)

    def mark_yaw_unavailable(self, action: ViewpointAction, *, reason: str, global_unavailable: bool = False) -> None:
        if action.kind != ActionKind.BASE_YAW:
            raise ValueError("yaw rejection does not match the pending action")
        self._pending = None
        self._yaw_globally_unavailable = bool(global_unavailable)
        # A preflight boundary is a hard constraint, not negative image
        # feedback. Reversing here caused the observed two-pose oscillation:
        # image feedback immediately requested the improving direction again,
        # which the same boundary rejected. Stop once with a useful diagnostic
        # instead of commanding motion away from the best measured view.
        self._terminate(
            ActionKind.STAGNATED,
            f"J1 yaw cannot continue in its improving direction: {reason}",
            camera=action.camera,
        )

    def _goal_camera(self, reports: Mapping[str, MaskReport]) -> str | None:
        candidates = [(name, report) for name, report in reports.items()
                      if report.seen and report.full and self.min_goal_area_frac <= report.area_frac <= self.max_goal_area_frac]
        return max(candidates, key=lambda item: self.report_quality(item[1]))[0] if candidates else None

    def _credible_alignment_report(
        self,
        report: MaskReport | None,
    ) -> bool:
        return bool(
            report is not None
            and report.seen
            and report.area_frac >= self.min_goal_area_frac
        )

    def _consume_pending(self, current: MaskReport) -> None:
        pending = self._pending
        self._pending = None
        if pending is None or pending.phase is not self._phase:
            return
        self._phase_moves += 1
        score = self._phase_score(current, self._phase)
        if score < pending.score - self.improvement_tolerance:
            if pending.action.kind == ActionKind.BASE_YAW:
                previous = math.copysign(
                    1.0, pending.action.aim_direction[0] or 1.0
                )
                candidate = -previous
                left_only = "left" in current.edges and "right" not in current.edges
                right_only = "right" in current.edges and "left" not in current.edges
                explicit_opposite_edge = (
                    (candidate < 0.0 and left_only)
                    or (candidate > 0.0 and right_only)
                )
                if explicit_opposite_edge:
                    self._direction = candidate
                    self._yaw_reversal_candidate = 0.0
                    self._yaw_reversal_streak = 0
                    self._yaw_confirmed_moves = 0
                else:
                    if candidate == self._yaw_reversal_candidate:
                        self._yaw_reversal_streak += 1
                    else:
                        self._yaw_reversal_candidate = candidate
                        self._yaw_reversal_streak = 1
                    # One noisy mask cannot reverse J1. Keep the current
                    # improving direction until regression repeats.
                    if self._yaw_reversal_streak >= 2:
                        self._direction = candidate
                        self._yaw_reversal_candidate = 0.0
                        self._yaw_reversal_streak = 0
                        self._yaw_confirmed_moves = 0
                    else:
                        self._direction = previous
            elif pending.action.kind in {ActionKind.BACKOFF, ActionKind.UP_CLEARANCE}:
                self._direction = -float(pending.action.axial_direction or 1.0)
            else:
                self._direction = -float(pending.action.aim_direction[0] or 1.0)
        else:
            self._direction = (pending.action.axial_direction or pending.action.aim_direction[0] or self._direction)
            if pending.action.kind is ActionKind.BASE_YAW:
                if score > pending.score + self.improvement_tolerance:
                    self._yaw_confirmed_moves += 1
                    self._yaw_reversal_candidate = 0.0
                    self._yaw_reversal_streak = 0
                elif self._yaw_reversal_candidate:
                    previous = math.copysign(
                        1.0, pending.action.aim_direction[0] or 1.0
                    )
                    if self._yaw_reversal_candidate == -previous:
                        self._yaw_reversal_streak += 1
                        if self._yaw_reversal_streak >= 2:
                            self._direction = self._yaw_reversal_candidate
                            self._yaw_reversal_candidate = 0.0
                            self._yaw_reversal_streak = 0
                            self._yaw_confirmed_moves = 0

    def _yaw_aligned(
        self,
        report: MaskReport,
        *,
        threshold: float | None = None,
    ) -> bool:
        # A small centered blob touching the ignored gripper band is normally
        # the wrist/gripper, not the task board.  The previous policy advanced
        # on exactly that false positive (area 0.021, y=0.928).
        center_limit = (
            self.horizontal_center_threshold
            if threshold is None
            else float(threshold)
        )
        if report.area_frac < self.min_goal_area_frac:
            return False
        if abs(report.center_error[0]) > center_limit:
            return False
        left = "left" in report.edges
        right = "right" in report.edges
        oversized = (
            report.area_frac > self.max_goal_area_frac
            or (left and right)
        )
        # Edge-free is ideal at useful scale.  When an oversized board is
        # already centered, yaw cannot remove its edge contacts; J2 standoff
        # is the correct next degree of freedom.
        return oversized or not (left or right)

    def _bottom_visible(self, report: MaskReport) -> bool:
        return "bottom" not in report.edges and not report.artificial_bottom_contact

    def _clearance_ready(self, report: MaskReport) -> bool:
        """True when J2 standoff has produced a usable board scale."""

        return (
            self._bottom_visible(report)
            and report.area_frac <= self.max_goal_area_frac
            and not has_opposite_edges(report.edges)
        )

    def _clearance_horizontal_drifted(self, report: MaskReport) -> bool:
        """Return to J1 when a clearance move materially loses centering."""

        if self._phase is _Phase.J1_YAW:
            return False
        current_error = abs(float(report.center_error[0]))
        if current_error > self.horizontal_exit_threshold:
            return True
        if self._j1_latched_error is None:
            return False
        return current_error > self._j1_latched_error + self.clearance_drift_tolerance

    def _top_visible(self, report: MaskReport) -> bool:
        return "top" not in report.edges

    def _choose_yaw_sign(self, report: MaskReport) -> float:
        if self._direction:
            return math.copysign(1.0, self._direction)
        if "left" in report.edges and "right" not in report.edges:
            return -1.0
        if "right" in report.edges and "left" not in report.edges:
            return 1.0
        return math.copysign(1.0, report.center_error[0] or 1.0)

    def _yaw_scale(self, report: MaskReport) -> float:
        """Use a probe first, then monotonic coarse-to-fine yaw steps."""

        if self._yaw_confirmed_moves <= 0:
            return self.yaw_probe_scale
        error = abs(float(report.center_error[0]))
        if error > 0.50:
            return 1.50
        if error > 0.25:
            return 1.00
        return 0.50

    def _side_camera_yaw_hint(self, reports: Mapping[str, MaskReport]) -> float:
        """Use side masks only to seed a reversible center-camera sweep."""

        if self._direction:
            return math.copysign(1.0, self._direction)
        weighted = 0.0
        for name in ("left_camera", "right_camera"):
            report = reports.get(name)
            if report is None or not report.seen:
                continue
            if "left" in report.edges and "right" not in report.edges:
                weighted -= max(report.area_frac, 0.01)
            elif "right" in report.edges and "left" not in report.edges:
                weighted += max(report.area_frac, 0.01)
            else:
                weighted += report.center_error[0] * max(report.area_frac, 0.01)
        return math.copysign(1.0, weighted or 1.0)

    def _choose_relative_sign(self, report: MaskReport, *, default: float) -> float:
        return math.copysign(1.0, self._direction) if self._direction else default

    def _phase_score(self, report: MaskReport, phase: _Phase) -> float:
        if phase is _Phase.J1_YAW:
            detail_deficit = max(
                0.0,
                self.min_goal_area_frac - float(report.area_frac),
            ) / max(self.min_goal_area_frac, 1e-9)
            return (
                -abs(report.center_error[0])
                - 2.0 * float("left" in report.edges or "right" in report.edges)
                - 6.0 * float(report.artificial_bottom_contact)
                - 4.0 * detail_deficit
                + float(report.area_frac)
            )
        if phase is _Phase.J2_CLEARANCE:
            scale_excess = max(
                0.0,
                report.area_frac - self.max_goal_area_frac,
            ) / max(self.max_goal_area_frac, 1e-9)
            return (
                6.0 * float(self._clearance_ready(report))
                + 2.0 * float(self._bottom_visible(report))
                - 2.0 * float(report.artificial_bottom_contact)
                - float("bottom" in report.edges)
                - 3.0 * float(has_opposite_edges(report.edges))
                - 2.0 * scale_excess
            )
        if phase is _Phase.J3_CLEARANCE:
            return 4.0 * float(self._top_visible(report)) - float("top" in report.edges) - abs(report.center_error[1]) * 0.2
        return self.report_quality(report)

    def _advance(self, phase: _Phase) -> None:
        self._phase = phase
        self._phase_moves = 0
        self._direction = 0.0
        if phase is _Phase.J1_YAW:
            self._yaw_confirmed_moves = 0
            self._yaw_reversal_candidate = 0.0
            self._yaw_reversal_streak = 0
            self._j1_alignment_streak = 0

    def _emit(self, kind: ActionKind, camera: str, *, axial: float = 0.0, aim: tuple[float, float] = (0.0, 0.0), scale: float = 1.0, angular: float = 0.0, reason: str) -> ViewpointAction:
        action = ViewpointAction(self._next_action_id, kind, camera, axial_direction=axial, aim_direction=aim, translation_scale=scale, angular_scale=angular, reason=reason)
        self._next_action_id += 1
        # Only robot motion needs before/after feedback scoring. OBSERVE is a
        # deliberate fresh-frame confirmation with no pending movement.
        if action.moves_robot:
            self._pending = _Pending(action, self._phase, self._current_score)
        else:
            self._pending = None
        return action

    def _terminate(self, kind: ActionKind, reason: str, camera: str | None = None) -> ViewpointAction:
        self._terminal_action = ViewpointAction(self._next_action_id, kind, camera=camera, terminal=True, reason=reason)
        self._next_action_id += 1
        return self._terminal_action
