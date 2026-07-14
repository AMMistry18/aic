"""Stateful, image-feedback viewpoint search planning.

This module is intentionally independent of ROS and robot-control messages.  It
turns successive per-camera :class:`MaskReport` observations into dimensionless
motion requests.  The skill wrapper is responsible for mapping those requests
through the selected camera frame and enforcing time, workspace, force, and
controller limits.

The planner never stops because a particular number of actions was proposed.
It stops only for an explicit deadline, a completed view, loss of all visual
evidence, or exhaustion of the safe alternatives for a stagnant observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

from .board_visibility import MaskReport


class ActionKind(str, Enum):
    """Kinds of actions understood by the board-visibility skill wrapper."""

    TRANSLATE = "translate"
    BACKOFF = "backoff"
    APPROACH = "approach"
    AIM = "aim"
    COMBINED = "combined"
    ROLLBACK = "rollback"
    DONE = "done"
    STAGNATED = "stagnated"
    NO_VIEW = "no_view"
    DEADLINE = "deadline"


_MOVEMENT_KINDS = frozenset(
    {
        ActionKind.TRANSLATE,
        ActionKind.BACKOFF,
        ActionKind.APPROACH,
        ActionKind.AIM,
        ActionKind.COMBINED,
    }
)


@dataclass(frozen=True)
class ViewpointAction:
    """A dimensionless motion request produced from the latest camera images.

    ``image_direction`` is expressed as image-right/image-down.  The wrapper
    multiplies it by its configured Cartesian step and transforms it through
    the selected camera optical frame.  ``axial_direction`` is positive away
    from the board and negative toward it.  ``aim_direction`` is an image-plane
    pitch/yaw request which the wrapper scales by its angular step.

    A combined action may contain all three components.  A rollback action asks
    the wrapper to return to the pose saved immediately before ``rollback_of``.
    Terminal actions never request robot motion.
    """

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
class _ActionSpec:
    kind: ActionKind
    image_direction: tuple[float, float] = (0.0, 0.0)
    axial_direction: float = 0.0
    aim_direction: tuple[float, float] = (0.0, 0.0)
    translation_scale: float = 1.0
    angular_scale: float = 0.0


@dataclass(frozen=True)
class _PendingAction:
    action: ViewpointAction
    report: MaskReport
    state_key: tuple[object, ...]
    action_key: tuple[object, ...]


def image_direction_for_edges(edges: frozenset[str]) -> tuple[float, float]:
    """Return a normalized image-plane camera correction for clipped edges."""

    horizontal = (-1.0 if "left" in edges else 0.0) + (
        1.0 if "right" in edges else 0.0
    )
    vertical = (-1.0 if "top" in edges else 0.0) + (
        1.0 if "bottom" in edges else 0.0
    )
    norm = math.hypot(horizontal, vertical)
    if norm <= 1e-12:
        return (0.0, 0.0)
    return (horizontal / norm, vertical / norm)


def has_opposite_edges(edges: frozenset[str]) -> bool:
    return ("left" in edges and "right" in edges) or (
        "top" in edges and "bottom" in edges
    )


class AdaptiveViewpointPlanner:
    """Choose feedback-adaptive viewpoint actions without a move-count limit.

    Call :meth:`next_action` once for each post-settle camera snapshot.  The
    planner evaluates the previously requested action from the new report,
    remembers persistent edge patterns per camera, and avoids repeating an
    ineffective action at the same visual state.

    The caller owns the overall monotonic deadline and passes
    ``deadline_reached=True`` when it expires.  This keeps time deterministic in
    tests and avoids embedding sleeps or clocks in the pure planner.
    """

    def __init__(
        self,
        *,
        min_goal_area_frac: float = 0.04,
        max_goal_area_frac: float = 0.38,
        camera_switch_margin: float = 0.35,
        improvement_tolerance: float = 0.025,
        regression_tolerance: float = 0.04,
        area_quantum: float = 0.025,
        rectangularity_quantum: float = 0.10,
    ) -> None:
        if not 0.0 <= min_goal_area_frac < max_goal_area_frac <= 1.0:
            raise ValueError("goal area limits must satisfy 0 <= min < max <= 1")
        if camera_switch_margin < 0.0:
            raise ValueError("camera_switch_margin must be non-negative")
        if improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance must be non-negative")
        if regression_tolerance <= improvement_tolerance:
            raise ValueError(
                "regression_tolerance must exceed improvement_tolerance"
            )
        if area_quantum <= 0.0 or rectangularity_quantum <= 0.0:
            raise ValueError("state quantization values must be positive")

        self.min_goal_area_frac = float(min_goal_area_frac)
        self.max_goal_area_frac = float(max_goal_area_frac)
        self.camera_switch_margin = float(camera_switch_margin)
        self.improvement_tolerance = float(improvement_tolerance)
        self.regression_tolerance = float(regression_tolerance)
        self.area_quantum = float(area_quantum)
        self.rectangularity_quantum = float(rectangularity_quantum)
        self.reset()

    def reset(self) -> None:
        self._selected_camera: str | None = None
        self._last_edges: dict[str, frozenset[str]] = {}
        self._edge_runs: dict[str, int] = {}
        self._blacklisted: set[tuple[object, ...]] = set()
        self._pending: _PendingAction | None = None
        self._awaiting_rollback = False
        self._next_action_id = 1
        self._terminal_action: ViewpointAction | None = None

    @property
    def selected_camera(self) -> str | None:
        return self._selected_camera

    @property
    def terminal(self) -> bool:
        return self._terminal_action is not None

    @property
    def blacklisted_action_count(self) -> int:
        return len(self._blacklisted)

    def report_quality(self, report: MaskReport) -> float:
        """Return a scalar used for camera selection, not goal acceptance."""

        if not report.seen:
            return -1000.0
        area = min(1.0, max(0.0, float(report.area_frac)))
        rectangularity = min(1.0, max(0.0, float(report.rectangularity)))
        score = 10.0
        score -= 3.0 * len(report.edges)
        if has_opposite_edges(report.edges):
            score -= 1.0
        score += 1.5 * rectangularity
        # More visible board pixels are normally better for a partial view, but
        # cap this term so a close, heavily clipped view does not dominate.
        score += 2.0 * min(1.0, area / max(self.max_goal_area_frac, 1e-9))
        if area > self.max_goal_area_frac:
            score -= 4.0 * (area - self.max_goal_area_frac)
        if report.full:
            score += 20.0
        return score

    def next_action(
        self,
        reports: Mapping[str, MaskReport],
        *,
        deadline_reached: bool = False,
    ) -> ViewpointAction:
        """Consume a fresh multi-camera observation and return the next action."""

        if self._terminal_action is not None:
            return self._terminal_action
        if deadline_reached:
            return self._terminate(
                ActionKind.DEADLINE,
                "viewpoint-search deadline reached before a complete view",
            )

        current_reports = dict(reports)
        self._update_edge_runs(current_reports)
        if self._awaiting_rollback:
            # The caller has re-observed after returning to the saved pose.  Do
            # not judge the rollback as though it were a new search action.
            self._awaiting_rollback = False

        goal_camera = self._goal_camera(current_reports)
        if goal_camera is not None:
            self._selected_camera = goal_camera
            return self._terminate(
                ActionKind.DONE,
                f"complete usable board view in {goal_camera}",
                camera=goal_camera,
            )

        if self._pending is not None:
            feedback = self._evaluate_pending(current_reports)
            if feedback == "regressed":
                failed = self._pending
                self._blacklisted.add(failed.action_key)
                self._pending = None
                self._awaiting_rollback = True
                return self._new_action(
                    _ActionSpec(ActionKind.ROLLBACK),
                    camera=failed.action.camera,
                    request_rollback=True,
                    rollback_of=failed.action.action_id,
                    reason=(
                        f"{failed.action.kind.value} regressed the "
                        f"{failed.action.camera} view; restore its saved pose"
                    ),
                )
            if feedback == "stagnant":
                self._blacklisted.add(self._pending.action_key)
            self._pending = None

        selected = self._choose_camera(current_reports)
        if selected is None:
            return self._terminate(
                ActionKind.NO_VIEW,
                "no plausible board is visible in any supported camera",
            )

        candidate = self._candidate_for_camera(selected, current_reports[selected])
        if candidate is None:
            # Once a camera's safe actions are exhausted, deliberately try a
            # different visible camera even when ordinary hysteresis would keep
            # the current one.  Stagnation is global, not camera-local.
            alternatives = sorted(
                (
                    (name, report)
                    for name, report in current_reports.items()
                    if name != selected and report.seen
                ),
                key=lambda item: self.report_quality(item[1]),
                reverse=True,
            )
            for name, report in alternatives:
                candidate = self._candidate_for_camera(name, report)
                if candidate is not None:
                    selected = name
                    break

        if candidate is None:
            return self._terminate(
                ActionKind.STAGNATED,
                "all safe viewpoint actions were ineffective at the current view",
                camera=selected,
            )

        self._selected_camera = selected
        spec, state_key, action_key = candidate
        action = self._new_action(
            spec,
            camera=selected,
            reason=self._reason_for(spec, current_reports[selected]),
        )
        self._pending = _PendingAction(
            action=action,
            report=current_reports[selected],
            state_key=state_key,
            action_key=action_key,
        )
        return action

    def _update_edge_runs(self, reports: Mapping[str, MaskReport]) -> None:
        for camera, report in reports.items():
            edges = report.edges if report.seen else frozenset()
            if self._last_edges.get(camera) == edges:
                self._edge_runs[camera] = self._edge_runs.get(camera, 0) + 1
            else:
                self._last_edges[camera] = edges
                self._edge_runs[camera] = 1

    def _goal_camera(self, reports: Mapping[str, MaskReport]) -> str | None:
        candidates = [
            (camera, report)
            for camera, report in reports.items()
            if report.seen
            and report.full
            and self.min_goal_area_frac
            <= float(report.area_frac)
            <= self.max_goal_area_frac
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: self.report_quality(item[1]))[0]

    def _choose_camera(self, reports: Mapping[str, MaskReport]) -> str | None:
        detected = [
            (camera, report)
            for camera, report in reports.items()
            if report.seen
        ]
        if not detected:
            return None
        best_camera, best_report = max(
            detected, key=lambda item: self.report_quality(item[1])
        )
        if self._selected_camera is None:
            return best_camera
        current = reports.get(self._selected_camera)
        if current is None or not current.seen:
            return best_camera
        current_score = self.report_quality(current)
        best_score = self.report_quality(best_report)
        if (
            best_camera != self._selected_camera
            and best_score <= current_score + self.camera_switch_margin
        ):
            return self._selected_camera
        return best_camera

    def _evaluate_pending(self, reports: Mapping[str, MaskReport]) -> str:
        assert self._pending is not None
        previous = self._pending.report
        current = reports.get(self._pending.action.camera or "")
        if current is None or not current.seen:
            return "regressed"
        delta = self._progress_delta(previous, current, self._pending.action.kind)
        if delta < -self.regression_tolerance:
            return "regressed"
        if delta <= self.improvement_tolerance:
            return "stagnant"
        return "improved"

    @staticmethod
    def _progress_delta(
        previous: MaskReport,
        current: MaskReport,
        action_kind: ActionKind,
    ) -> float:
        """Measure action progress with the intended axial direction in mind."""

        if current.full and not previous.full:
            return 20.0
        if previous.full and not current.full:
            return -20.0
        delta = 3.0 * (len(previous.edges) - len(current.edges))
        delta += 1.5 * (
            float(current.rectangularity) - float(previous.rectangularity)
        )
        area_change = float(current.area_frac) - float(previous.area_frac)
        if action_kind == ActionKind.BACKOFF:
            area_change = -area_change
        # An approach should increase scale; translations, aiming, and combined
        # corrections normally reveal more of a clipped board.
        delta += 4.0 * area_change
        return delta

    def _candidate_for_camera(
        self, camera: str, report: MaskReport
    ) -> tuple[_ActionSpec, tuple[object, ...], tuple[object, ...]] | None:
        state_key = self._state_key(camera, report)
        for spec in self._action_specs(camera, report):
            action_key = self._action_key(state_key, spec)
            if action_key not in self._blacklisted:
                return spec, state_key, action_key
        return None

    def _action_specs(self, camera: str, report: MaskReport) -> list[_ActionSpec]:
        if report.full and report.area_frac < self.min_goal_area_frac:
            return [
                _ActionSpec(
                    ActionKind.APPROACH,
                    axial_direction=-1.0,
                    translation_scale=0.75,
                )
            ]
        if report.full and report.area_frac > self.max_goal_area_frac:
            return [
                _ActionSpec(
                    ActionKind.BACKOFF,
                    axial_direction=1.0,
                    translation_scale=0.75,
                )
            ]

        direction = image_direction_for_edges(report.edges)
        translate = _ActionSpec(
            ActionKind.TRANSLATE,
            image_direction=direction,
            translation_scale=1.0,
        )
        combined = _ActionSpec(
            ActionKind.COMBINED,
            image_direction=direction,
            axial_direction=0.35,
            aim_direction=direction,
            translation_scale=0.70,
            angular_scale=0.50,
        )
        aim = _ActionSpec(
            ActionKind.AIM,
            aim_direction=direction,
            translation_scale=0.0,
            angular_scale=0.75,
        )
        backoff = _ActionSpec(
            ActionKind.BACKOFF,
            axial_direction=1.0,
            translation_scale=0.75,
        )

        if has_opposite_edges(report.edges):
            base = [backoff, combined, aim, translate]
        elif direction == (0.0, 0.0):
            # A future stricter detector can report not-full because of
            # perspective/shape quality even without a clipped edge.
            base = [aim, backoff]
        else:
            base = [translate, combined, aim, backoff]

        run = max(1, self._edge_runs.get(camera, 1))
        offset = (run - 1) % len(base)
        return base[offset:] + base[:offset]

    def _state_key(self, camera: str, report: MaskReport) -> tuple[object, ...]:
        return (
            camera,
            tuple(sorted(report.edges)),
            bool(report.full),
            int(round(float(report.area_frac) / self.area_quantum)),
            int(
                round(
                    float(report.rectangularity)
                    / self.rectangularity_quantum
                )
            ),
        )

    @staticmethod
    def _action_key(
        state_key: tuple[object, ...], spec: _ActionSpec
    ) -> tuple[object, ...]:
        def rounded(values: tuple[float, float]) -> tuple[float, float]:
            return (round(values[0], 3), round(values[1], 3))

        return (
            *state_key,
            spec.kind.value,
            rounded(spec.image_direction),
            round(spec.axial_direction, 3),
            rounded(spec.aim_direction),
        )

    @staticmethod
    def _reason_for(spec: _ActionSpec, report: MaskReport) -> str:
        edges = ",".join(sorted(report.edges)) or "none"
        if spec.kind == ActionKind.TRANSLATE:
            return f"translate from image evidence at clipped edges {edges}"
        if spec.kind == ActionKind.COMBINED:
            return f"persistent edges {edges}; combine centering, aim, and standoff"
        if spec.kind == ActionKind.AIM:
            return f"persistent edges {edges}; change camera viewing angle"
        if spec.kind == ActionKind.BACKOFF:
            return f"increase standoff for clipped/oversized view at {edges}"
        if spec.kind == ActionKind.APPROACH:
            return "complete board is too small; approach for component resolution"
        return spec.kind.value

    def _new_action(
        self,
        spec: _ActionSpec,
        *,
        camera: str | None,
        reason: str,
        request_rollback: bool = False,
        rollback_of: int | None = None,
        terminal: bool = False,
    ) -> ViewpointAction:
        action = ViewpointAction(
            action_id=self._next_action_id,
            kind=spec.kind,
            camera=camera,
            image_direction=spec.image_direction,
            axial_direction=spec.axial_direction,
            aim_direction=spec.aim_direction,
            translation_scale=spec.translation_scale,
            angular_scale=spec.angular_scale,
            request_rollback=request_rollback,
            rollback_of=rollback_of,
            terminal=terminal,
            reason=reason,
        )
        self._next_action_id += 1
        return action

    def _terminate(
        self,
        kind: ActionKind,
        reason: str,
        *,
        camera: str | None = None,
    ) -> ViewpointAction:
        self._pending = None
        self._terminal_action = self._new_action(
            _ActionSpec(kind),
            camera=camera,
            reason=reason,
            terminal=True,
        )
        return self._terminal_action


__all__ = [
    "ActionKind",
    "AdaptiveViewpointPlanner",
    "ViewpointAction",
    "has_opposite_edges",
    "image_direction_for_edges",
]
