"""Camera-feedback policy for the four-stage task-board search.

The sequence is deliberately fixed by the workcell geometry: align with J1,
increase J2 clearance until the lower board is visible, use J3 clearance for
the upper edge, then use J4 camera-roll for fine framing.  J1 and J4 use
bounded image-feedback probes; J2 always moves away and J3 always moves up so
mask noise cannot reverse either clearance motion.  A single camera with a
complete padded board view is sufficient to finish; the other cameras are
steering evidence, not a mandatory pixel-stitching requirement.
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
    center_report: MaskReport | None = None


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
        phase_confirmation_frames: int = 2,
        clearance_drift_tolerance: float = 0.08,
        max_roll_moves: int = 6,
    ) -> None:
        if not 0.0 <= min_goal_area_frac < max_goal_area_frac <= 1.0:
            raise ValueError("goal area limits must satisfy 0 <= min < max <= 1")
        cameras = tuple(str(item) for item in expected_cameras)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("expected_cameras must be non-empty and unique")
        self.min_goal_area_frac = float(min_goal_area_frac)
        self.max_goal_area_frac = float(max_goal_area_frac)
        self.expected_cameras = cameras
        self.horizontal_coverage_area_frac = float(
            min_goal_area_frac
            if horizontal_coverage_area_frac is None
            else horizontal_coverage_area_frac
        )
        self.yaw_probe_scale = float(yaw_probe_scale)
        self.improvement_tolerance = float(improvement_tolerance)
        self.horizontal_center_threshold = float(horizontal_center_threshold)
        self.horizontal_exit_threshold = float(horizontal_exit_threshold)
        self.alignment_confirmation_frames = int(alignment_confirmation_frames)
        self.phase_confirmation_frames = int(phase_confirmation_frames)
        self.clearance_drift_tolerance = float(clearance_drift_tolerance)
        self.max_roll_moves = int(max_roll_moves)
        if self.alignment_confirmation_frames < 1:
            raise ValueError("alignment_confirmation_frames must be positive")
        if self.phase_confirmation_frames < 1:
            raise ValueError("phase_confirmation_frames must be positive")
        if self.clearance_drift_tolerance < 0.0:
            raise ValueError("clearance_drift_tolerance must be non-negative")
        if self.max_roll_moves < 2:
            raise ValueError("max_roll_moves must be at least 2")
        if self.yaw_probe_scale <= 0.0:
            raise ValueError("yaw_probe_scale must be positive")
        if self.improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance must be non-negative")
        if not 0.0 < self.horizontal_coverage_area_frac <= 1.0:
            raise ValueError("horizontal_coverage_area_frac must be in (0, 1]")
        if not (
            0.0 < self.horizontal_center_threshold
            <= self.horizontal_exit_threshold
            <= 1.0
        ):
            raise ValueError(
                "horizontal thresholds must satisfy 0 < center <= exit <= 1"
            )
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
        self._yaw_reversal_count = 0
        self._last_yaw_command_direction = 0.0
        self._yaw_direction_best_error = math.inf
        self._yaw_cycle_recovery_requested = False
        self._yaw_cycle_recovery_count = 0
        self._best_yaw_action_id: int | None = None
        self._best_yaw_report: MaskReport | None = None
        self._best_yaw_score = -math.inf
        self._force_one_clearance_move = False
        self._force_one_up_clearance_move = False
        self._geometry_recovery_lock = False
        self._rollback_expected_report: MaskReport | None = None
        self._rollback_camera_validation_failed = False
        self._j1_alignment_streak = 0
        self._j1_confirmation_report: MaskReport | None = None
        self._phase_ready_streak = 0
        self._phase_confirmation_report: MaskReport | None = None
        self._roll_reversal_count = 0
        self._roll_nonimproving_streak = 0
        self._roll_cycle_recovery_requested = False
        self._roll_moves_total = 0
        self._roll_recovery_count = 0
        self._best_roll_action_id: int | None = None
        self._best_roll_report: MaskReport | None = None
        self._best_roll_score = -math.inf
        self._j1_latched_error: float | None = None
        self._yaw_target_x = 0.0
        self._clearance_drift_samples = 0
        self._current_center_report: MaskReport | None = None
        self._identity_boundary_recoveries = 0
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
        return bool(
            self._terminal_action is not None
            and self._terminal_action.kind is ActionKind.DONE
        )

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
            self._current_center_report = None
            if self._rollback_expected_report is not None:
                self._rollback_expected_report = None
                self._rollback_camera_validation_failed = True
                self._geometry_recovery_lock = True
            pending = self._pending
            self._pending = None
            self._j1_alignment_streak = 0
            self._j1_confirmation_report = None
            if self._geometry_recovery_lock:
                if self._phase is _Phase.J2_CLEARANCE:
                    if (
                        pending is not None
                        and pending.action.kind is ActionKind.BACKOFF
                    ):
                        self._force_one_clearance_move = False
                        self._force_one_up_clearance_move = False
                        self._advance(_Phase.J3_CLEARANCE)
                        return self._emit(
                            ActionKind.UP_CLEARANCE,
                            "center_camera",
                            axial=1.0,
                            scale=1.0,
                            reason=(
                                "J3 locked recovery: J2 did not reacquire "
                                "the center camera; change vertical geometry"
                            ),
                        )
                    self._force_one_clearance_move = False
                    self._rollback_camera_validation_failed = False
                    return self._emit(
                        ActionKind.BACKOFF,
                        "center_camera",
                        axial=1.0,
                        scale=1.0,
                        reason=(
                            "J2 locked recovery: center camera remains lost; "
                            "continue monotonic standoff before any yaw retry"
                        ),
                    )
                if self._phase is _Phase.J3_CLEARANCE:
                    if (
                        pending is not None
                        and pending.action.kind is ActionKind.UP_CLEARANCE
                    ):
                        # Both monotonic recovery degrees of freedom were
                        # tried without reacquiring the board. Resume the
                        # bounded J1 sweep once; its next exhausted cycle is
                        # terminal instead of another clearance loop.
                        self._yaw_cycle_recovery_count = max(
                            self._yaw_cycle_recovery_count,
                            2,
                        )
                        self._advance(_Phase.J1_YAW)
                    else:
                        self._force_one_up_clearance_move = False
                        self._rollback_camera_validation_failed = False
                        return self._emit(
                            ActionKind.UP_CLEARANCE,
                            "center_camera",
                            axial=1.0,
                            scale=1.0,
                            reason=(
                                "J3 locked recovery: center camera remains "
                                "lost; continue vertical clearance before "
                                "one final bounded yaw retry"
                            ),
                        )
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
                    self._commit_yaw_reversal(candidate)
                    self._yaw_reversal_candidate = 0.0
                    self._yaw_reversal_streak = 0
                else:
                    self._direction = previous
                self._phase_moves += 1
            cycle_recovery = self._recover_from_yaw_cycle("center_camera")
            if cycle_recovery is not None:
                return cycle_recovery
            if self._phase is not _Phase.J1_YAW:
                self._advance(_Phase.J1_YAW)
            direction = self._side_camera_yaw_hint(reports)
            return self._emit_base_yaw(
                "center_camera",
                direction,
                angular=0.75,
                reason=(
                    "J1 acquisition sweep: center camera lost the board; "
                    "reverse/continue yaw"
                ),
            )
        primary_name = "center_camera"
        primary = center
        self._selected_camera = primary_name
        pending = self._pending
        if (
            pending is not None
            and pending.phase is _Phase.J1_YAW
            and pending.action.kind is ActionKind.BASE_YAW
            and pending.center_report is not None
            and self._center_component_identity_switched(
                pending.center_report,
                primary,
                pending.action,
            )
        ):
            # The largest connected component changed from the lower board
            # fragment to the upper/right fragment (or vice versa). Chasing
            # their unrelated centroids produces an unbounded J1 limit cycle.
            # The current mask no longer describes the same physical blob.
            # Do not chase it with J1 and do not rely on a Cartesian rollback
            # to make largest-component selection deterministic.  Keep the
            # measured pose and change the observable geometry through the
            # prescribed J2/J3 ladder.
            before = pending.center_report
            self._pending = None
            self._j1_latched_error = abs(self._yaw_error(before))
            recovery_index = self._identity_boundary_recoveries
            self._identity_boundary_recoveries += 1
            if recovery_index == 0:
                recovery_phase = _Phase.J2_CLEARANCE
                recovery_kind = ActionKind.BACKOFF
                recovery_reason = "use locked J2 clearance"
            elif recovery_index == 1:
                recovery_phase = _Phase.J3_CLEARANCE
                recovery_kind = ActionKind.UP_CLEARANCE
                recovery_reason = "use locked J3 vertical clearance"
            else:
                return self._terminate(
                    ActionKind.STAGNATED,
                    "center-mask identity boundary persisted after J2 and J3 geometry changes; refusing another yaw cycle",
                    primary_name,
                )
            self._geometry_recovery_lock = True
            self._advance(recovery_phase)
            self._current_center_report = primary
            self._current_score = self._phase_score(primary, self._phase)
            return self._emit(
                recovery_kind,
                primary_name,
                axial=1.0,
                scale=1.0,
                reason=(
                    "J1 center-mask identity discontinuity: stop yaw and "
                    f"{recovery_reason} before another horizontal search"
                ),
            )

        self._current_center_report = primary
        if self._rollback_expected_report is not None:
            self._rollback_camera_validation_failed = not self._reports_stable(
                self._rollback_expected_report,
                primary,
            )
            self._rollback_expected_report = None
            if self._rollback_camera_validation_failed:
                self._geometry_recovery_lock = True
        self._consume_pending(primary)
        self._current_score = self._phase_score(primary, self._phase)
        cycle_recovery = self._recover_from_yaw_cycle(primary_name)
        if cycle_recovery is not None:
            return cycle_recovery
        roll_recovery = self._recover_from_roll_cycle(primary_name)
        if roll_recovery is not None:
            return roll_recovery

        # A small/partial center-camera mask can steer acquisition, but it is
        # never sufficient to finish J1. Scoring it here lets two repeated
        # regressions reverse the sweep without alternating every frame.
        if (
            not self._credible_alignment_report(primary)
            and not (
                self._force_one_clearance_move
                or self._force_one_up_clearance_move
                or self._geometry_recovery_lock
            )
        ):
            self._j1_alignment_streak = 0
            self._j1_confirmation_report = None
            if self._phase is not _Phase.J1_YAW:
                self._advance(_Phase.J1_YAW)
                self._current_score = self._phase_score(primary, self._phase)
            sign = self._choose_yaw_sign(primary)
            return self._emit_base_yaw(
                primary_name,
                sign,
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
            and not (
                self._force_one_clearance_move
                or self._force_one_up_clearance_move
                or self._geometry_recovery_lock
            )
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
                if (
                    self._j1_confirmation_report is None
                    or self._reports_stable(
                        self._j1_confirmation_report,
                        primary,
                    )
                ):
                    self._j1_alignment_streak += 1
                else:
                    # There was no robot motion between these frames. A large
                    # mask jump is segmentation identity churn, not improved
                    # alignment. Restart confirmation from the new component
                    # and never change joints on that discontinuity.
                    self._j1_alignment_streak = 1
                self._j1_confirmation_report = primary
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
                self._j1_latched_error = abs(self._yaw_error(primary))
                self._advance(_Phase.J2_CLEARANCE)
                self._current_score = self._phase_score(primary, self._phase)
            else:
                self._j1_alignment_streak = 0
                self._j1_confirmation_report = None
                sign = self._choose_yaw_sign(primary)
                return self._emit_base_yaw(
                    primary_name,
                    sign,
                    angular=self._yaw_scale(primary),
                    reason=(
                        "J1 horizontal yaw: center the board in the center camera"
                    ),
                )
        if self._phase is _Phase.J2_CLEARANCE:
            if self._force_one_clearance_move:
                self._force_one_clearance_move = False
                self._phase_ready_streak = 0
                validation_note = (
                    " after rollback camera validation failed"
                    if self._rollback_camera_validation_failed
                    else ""
                )
                self._rollback_camera_validation_failed = False
                return self._emit(
                    ActionKind.BACKOFF,
                    primary_name,
                    axial=1.0,
                    scale=1.0,
                    reason=(
                        "J2 anti-cycle clearance: change camera geometry"
                        f"{validation_note} "
                        "before allowing another J1 search"
                    ),
                )
            if self._clearance_ready(primary):
                self._update_phase_confirmation(primary)
                if self._phase_ready_streak < self.phase_confirmation_frames:
                    return self._emit(
                        ActionKind.OBSERVE,
                        primary_name,
                        reason=(
                            "J2 clearance candidate: confirm the lower edge "
                            "and usable scale in a fresh center-camera frame"
                        ),
                    )
                self._advance(_Phase.J3_CLEARANCE)
                self._current_score = self._phase_score(primary, self._phase)
            else:
                self._phase_ready_streak = 0
                self._phase_confirmation_report = None
                sign = self._choose_relative_sign(primary, default=1.0)
                return self._emit(ActionKind.BACKOFF, primary_name, axial=sign, scale=1.0,
                                  reason="J2 relative clearance: expose the lower board edge")
        if self._phase is _Phase.J3_CLEARANCE:
            if self._force_one_up_clearance_move:
                self._force_one_up_clearance_move = False
                self._phase_ready_streak = 0
                validation_note = (
                    " after rollback camera validation failed"
                    if self._rollback_camera_validation_failed
                    else ""
                )
                self._rollback_camera_validation_failed = False
                return self._emit(
                    ActionKind.UP_CLEARANCE,
                    primary_name,
                    axial=1.0,
                    scale=1.0,
                    reason=(
                        "J3 anti-cycle clearance: change vertical camera"
                        f"{validation_note} "
                        "geometry before another J1 search"
                    ),
                )
            if not self._clearance_ready(primary):
                self._advance(_Phase.J2_CLEARANCE)
                self._current_score = self._phase_score(primary, self._phase)
                return self._emit(
                    ActionKind.BACKOFF,
                    primary_name,
                    axial=1.0,
                    scale=1.0,
                    reason=(
                        "J3 fallback to J2: lower edge or usable scale was lost"
                    ),
                )
            if self._top_visible(primary):
                self._update_phase_confirmation(primary)
                if self._phase_ready_streak < self.phase_confirmation_frames:
                    return self._emit(
                        ActionKind.OBSERVE,
                        primary_name,
                        reason=(
                            "J3 clearance candidate: confirm the upper edge "
                            "in a fresh center-camera frame"
                        ),
                    )
                self._geometry_recovery_lock = False
                self._advance(_Phase.J4_ROLL)
                self._current_score = self._phase_score(primary, self._phase)
            else:
                self._phase_ready_streak = 0
                self._phase_confirmation_report = None
                sign = self._choose_relative_sign(primary, default=1.0)
                return self._emit(ActionKind.UP_CLEARANCE, primary_name, axial=sign, scale=1.0,
                                  reason="J3 relative clearance: expose the upper board edge")
        # J4 is a camera roll, not a pitch.  The wrapper maps this to a roll
        # about the selected optical axis; its sign is checked after each view.
        if self._phase is _Phase.J4_ROLL:
            if not self._clearance_ready(primary):
                self._advance(_Phase.J2_CLEARANCE)
                self._current_score = self._phase_score(primary, self._phase)
                return self._emit(
                    ActionKind.BACKOFF,
                    primary_name,
                    axial=1.0,
                    scale=1.0,
                    reason="J4 fallback to J2: lower clearance was lost",
                )
            if not self._top_visible(primary):
                self._advance(_Phase.J3_CLEARANCE)
                self._current_score = self._phase_score(primary, self._phase)
                return self._emit(
                    ActionKind.UP_CLEARANCE,
                    primary_name,
                    axial=1.0,
                    scale=1.0,
                    reason="J4 fallback to J3: upper clearance was lost",
                )
            if self._roll_moves_total >= self.max_roll_moves:
                self._roll_cycle_recovery_requested = True
                recovery = self._recover_from_roll_cycle(primary_name)
                if recovery is not None:
                    return recovery
                return self._terminate(
                    ActionKind.STAGNATED,
                    "J4 roll reached its local excursion limit without a recoverable best pose",
                    primary_name,
                )
            sign = self._choose_relative_sign(primary, default=1.0)
            return self._emit(ActionKind.CAMERA_ROLL, primary_name, aim=(sign, 0.0), angular=0.55,
                              reason="J4 camera roll: fine-frame top and bottom board edges")
        return self.next_action(reports, deadline_reached=deadline_reached)

    def mark_yaw_unavailable(self, action: ViewpointAction, *, reason: str, global_unavailable: bool = False) -> None:
        if action.kind != ActionKind.BASE_YAW:
            raise ValueError("yaw rejection does not match the pending action")
        if self._terminal_action is not None:
            return
        if (
            self._pending is None
            or self._pending.action.action_id != action.action_id
        ):
            raise ValueError("yaw rejection is stale or does not match the pending action")
        self._pending = None
        self._yaw_globally_unavailable = bool(global_unavailable)
        # A preflight boundary is a hard constraint, not negative image
        # feedback. Reversing here caused the observed two-pose oscillation:
        # image feedback immediately requested the improving direction again,
        # which the same boundary rejected. Stop once with a useful diagnostic
        # instead of commanding motion away from the best measured view.
        if global_unavailable:
            self._terminate(
                ActionKind.STAGNATED,
                f"J1 yaw cannot continue in its improving direction: {reason}",
                camera=action.camera,
            )
            return
        recovery_index = self._yaw_cycle_recovery_count
        if recovery_index >= 2:
            self._terminate(
                ActionKind.STAGNATED,
                "J1 yaw remained constrained after bounded J2 and J3 geometry recoveries",
                camera=action.camera,
            )
            return
        self._yaw_cycle_recovery_count += 1
        self._geometry_recovery_lock = True
        if recovery_index == 0:
            self._force_one_clearance_move = True
            self._advance(_Phase.J2_CLEARANCE)
        else:
            self._force_one_up_clearance_move = True
            self._advance(_Phase.J3_CLEARANCE)

    def _goal_camera(self, reports: Mapping[str, MaskReport]) -> str | None:
        candidates = [(name, report) for name, report in reports.items()
                      if name in self.expected_cameras
                      if report.seen and report.full and self.min_goal_area_frac <= report.area_frac <= self.max_goal_area_frac]
        return max(candidates, key=lambda item: self.report_quality(item[1]))[0] if candidates else None

    def _credible_alignment_report(
        self,
        report: MaskReport | None,
    ) -> bool:
        return bool(
            report is not None
            and report.seen
            and report.area_frac >= self.horizontal_coverage_area_frac
            and not (
                report.artificial_bottom_contact
                and float(report.center_error[1]) > 0.80
            )
        )

    @staticmethod
    def _reports_stable(before: MaskReport, current: MaskReport) -> bool:
        """Reject no-motion component swaps during phase confirmation.

        Fresh simulated frames may move by a few pixels, but without a robot
        command the normalized centroid, area, and rectangularity cannot jump
        materially.  Requiring continuity here prevents two unrelated board
        fragments from satisfying a two-frame gate.
        """

        before_x, before_y = (float(value) for value in before.center_error)
        current_x, current_y = (float(value) for value in current.center_error)
        area_delta_limit = max(
            0.04,
            0.35 * max(float(before.area_frac), float(current.area_frac)),
        )
        return bool(
            before.seen
            and current.seen
            and abs(current_x - before_x) <= 0.10
            and abs(current_y - before_y) <= 0.20
            and abs(float(current.area_frac) - float(before.area_frac))
            <= area_delta_limit
            and abs(
                float(current.rectangularity) - float(before.rectangularity)
            ) <= 0.20
        )

    def _update_phase_confirmation(self, report: MaskReport) -> None:
        if (
            self._phase_confirmation_report is None
            or self._reports_stable(self._phase_confirmation_report, report)
        ):
            self._phase_ready_streak += 1
        else:
            self._phase_ready_streak = 1
        self._phase_confirmation_report = report

    def _center_component_identity_switched(
        self,
        before: MaskReport,
        current: MaskReport,
        action: ViewpointAction,
    ) -> bool:
        """Detect a disconnected board-fragment swap after a fine J1 move.

        A real center crossing changes horizontal error continuously and keeps
        approximately the same vertical centroid. The observed failure jumps
        between opposite image quadrants because largest-component selection
        alternates between two occluded board fragments. This predicate is
        deliberately strong so ordinary perspective motion cannot trigger it.
        """

        if not (
            self._credible_alignment_report(before)
            and self._credible_alignment_report(current)
        ):
            return False
        if action.angular_scale > 0.75:
            return False
        before_x, before_y = (float(value) for value in before.center_error)
        current_x, current_y = (float(value) for value in current.center_error)
        if before_x * current_x >= 0.0:
            return False
        horizontal_jump = abs(current_x - before_x)
        vertical_jump = abs(current_y - before_y)
        return bool(
            min(abs(before_x), abs(current_x))
            <= max(self.horizontal_exit_threshold, 0.25)
            and max(abs(before_x), abs(current_x)) >= 0.40
            and (
                horizontal_jump >= 0.80
                or (horizontal_jump >= 0.45 and vertical_jump >= 0.50)
            )
        )

    def _consume_pending(self, current: MaskReport) -> None:
        pending = self._pending
        self._pending = None
        if pending is None or pending.phase is not self._phase:
            return
        self._phase_moves += 1
        score = self._phase_score(current, self._phase)
        if pending.action.kind in {ActionKind.BACKOFF, ActionKind.UP_CLEARANCE}:
            # J2 is always away from the board and J3 is always base +Z.
            # Reversing either deterministic direction on a noisy mask caused
            # a second class of physical oscillation and violates the
            # prescribed joint workflow.
            if pending.action.kind is ActionKind.BACKOFF:
                self._learn_clearance_yaw_compensation(
                    pending.center_report,
                    current,
                )
            self._direction = math.copysign(
                1.0, pending.action.axial_direction or 1.0
            )
            return
        if pending.action.kind is ActionKind.CAMERA_ROLL:
            self._consume_roll_feedback(pending, score)
            return
        if pending.action.kind is ActionKind.BASE_YAW:
            self._consume_yaw_feedback(pending, current)
            return
        if score < pending.score - self.improvement_tolerance:
            self._direction = -float(
                pending.action.aim_direction[0] or 1.0
            )
        else:
            self._direction = (
                pending.action.axial_direction
                or pending.action.aim_direction[0]
                or self._direction
            )

    def _consume_yaw_feedback(
        self,
        pending: _Pending,
        current: MaskReport,
    ) -> None:
        """Use cumulative geometric progress, not a one-frame mixed score.

        J1 changes the center-camera horizontal centroid with the same sign as
        the commanded base-yaw arc in this workcell.  Progress is therefore a
        reduction in absolute horizontal error.  Tracking the best error for
        the whole direction catches the live failure where each wrong move
        regressed by less than the old per-frame score deadband.
        """

        previous = math.copysign(
            1.0, pending.action.aim_direction[0] or 1.0
        )
        before = pending.center_report
        before_error = (
            abs(self._yaw_error(before))
            if before is not None
            else math.inf
        )
        current_error = abs(self._yaw_error(current))
        self._yaw_direction_best_error = min(
            self._yaw_direction_best_error,
            before_error,
            current_error,
        )

        improved = current_error < before_error - 0.008
        local_regression = current_error > before_error + 0.015
        cumulative_regression = (
            current_error
            > self._yaw_direction_best_error + 0.030
        )
        if improved:
            self._yaw_confirmed_moves += 1
            self._yaw_reversal_candidate = 0.0
            self._yaw_reversal_streak = 0
            self._direction = previous
            return

        if local_regression or cumulative_regression:
            candidate = -previous
            if candidate == self._yaw_reversal_candidate:
                self._yaw_reversal_streak += 1
            else:
                self._yaw_reversal_candidate = candidate
                self._yaw_reversal_streak = 1
            if self._yaw_reversal_streak >= 2:
                self._commit_yaw_reversal(candidate)
                self._yaw_reversal_candidate = 0.0
                self._yaw_reversal_streak = 0
                self._yaw_confirmed_moves = 0
            else:
                self._direction = previous
            return

        # A single flat/noisy report neither reverses nor erases an already
        # accumulated regression. Once the cumulative threshold is crossed,
        # the next non-improving frame supplies the second confirmation.
        if self._yaw_reversal_candidate:
            self._yaw_reversal_streak += 1
            if self._yaw_reversal_streak >= 2:
                self._commit_yaw_reversal(
                    self._yaw_reversal_candidate
                )
                self._yaw_reversal_candidate = 0.0
                self._yaw_reversal_streak = 0
                self._yaw_confirmed_moves = 0
                return
        self._direction = previous

    def _commit_yaw_reversal(self, candidate: float) -> None:
        """Record the direction requested by confirmed image feedback.

        The actual one-reversal invariant is enforced centrally when a yaw
        command is emitted.  That also covers sign changes after an OBSERVE
        confirmation frame, which previously bypassed this method.
        """

        direction = math.copysign(1.0, candidate or 1.0)
        self._direction = direction

    def _consume_roll_feedback(self, pending: _Pending, score: float) -> None:
        """Bound J4 to one reversal and detect a flat local optimum."""

        previous = math.copysign(
            1.0, pending.action.aim_direction[0] or 1.0
        )
        if score > pending.score + self.improvement_tolerance:
            self._direction = previous
            self._roll_nonimproving_streak = 0
            return

        self._direction = previous
        self._roll_nonimproving_streak += 1
        regressed = score < pending.score - self.improvement_tolerance
        if not regressed and self._roll_nonimproving_streak < 2:
            return

        self._roll_nonimproving_streak = 0
        candidate = -previous
        if self._roll_reversal_count >= 1:
            self._roll_cycle_recovery_requested = True
            return
        self._roll_reversal_count += 1
        self._direction = candidate

    def _recover_from_yaw_cycle(
        self,
        camera: str,
    ) -> ViewpointAction | None:
        """Bound repeated J1 cycles with a J2-then-J3 recovery ladder."""

        if not self._yaw_cycle_recovery_requested:
            return None
        self._yaw_cycle_recovery_requested = False
        recovery_index = self._yaw_cycle_recovery_count
        if recovery_index >= 2:
            return self._terminate(
                ActionKind.STAGNATED,
                "J1 yaw remained ambiguous after bounded J2 and J3 geometry recoveries",
                camera,
            )
        self._yaw_cycle_recovery_count += 1
        self._geometry_recovery_lock = True
        if recovery_index == 0:
            recovery_phase = _Phase.J2_CLEARANCE
            recovery_kind = ActionKind.BACKOFF
            recovery_reason = "J2 clearance"
        else:
            recovery_phase = _Phase.J3_CLEARANCE
            recovery_kind = ActionKind.UP_CLEARANCE
            recovery_reason = "J3 vertical clearance"

        if self._best_yaw_action_id is None or self._best_yaw_report is None:
            self._pending = None
            self._advance(recovery_phase)
            return self._emit(
                recovery_kind,
                camera,
                axial=1.0,
                scale=1.0,
                reason=(
                    "J1 anti-cycle guard: no credible rollback pose exists; "
                    f"change camera geometry with {recovery_reason}"
                ),
            )
        best = self._best_yaw_report
        self._pending = None
        self._j1_latched_error = abs(self._yaw_error(best))
        if recovery_phase is _Phase.J2_CLEARANCE:
            self._force_one_clearance_move = True
        else:
            self._force_one_up_clearance_move = True
        self._advance(recovery_phase)
        self._current_score = self._phase_score(best, self._phase)
        self._rollback_expected_report = best
        return self._emit_rollback(
            self._best_yaw_action_id,
            camera,
            reason=(
                "J1 anti-cycle guard: a second yaw reversal was requested; "
                f"restore the best measured view and use {recovery_reason}"
            ),
        )

    def _recover_from_roll_cycle(
        self,
        camera: str,
    ) -> ViewpointAction | None:
        """Use at most one roll recovery in the complete search call."""

        if not self._roll_cycle_recovery_requested:
            return None
        self._roll_cycle_recovery_requested = False
        if self._roll_recovery_count >= 1:
            return self._terminate(
                ActionKind.STAGNATED,
                "J4 roll remained unresolved after one bounded geometry recovery",
                camera,
            )
        self._roll_recovery_count += 1
        self._geometry_recovery_lock = True

        current = self._current_center_report
        current_is_best = bool(
            current is not None
            and (
                self._best_roll_report is None
                or self.report_quality(current)
                >= self._best_roll_score - self.improvement_tolerance
            )
        )
        if (
            current_is_best
            or self._best_roll_action_id is None
            or self._best_roll_report is None
        ):
            # The latest report is a post-move view and therefore has no
            # pre-move rollback id. Keep it when it is best; rolling back here
            # would deliberately undo monotonic improvement.
            self._pending = None
            self._j1_latched_error = abs(self._yaw_error(current))
            self._advance(_Phase.J2_CLEARANCE)
            self._current_score = self._phase_score(current, self._phase)
            return self._emit(
                ActionKind.BACKOFF,
                camera,
                axial=1.0,
                scale=1.0,
                reason=(
                    "J4 anti-cycle guard: keep the best current roll view "
                    "and change standoff with J2"
                ),
            )
        best_action_id = self._best_roll_action_id
        best = self._best_roll_report
        self._pending = None
        self._j1_latched_error = abs(self._yaw_error(best))
        self._force_one_clearance_move = True
        self._advance(_Phase.J2_CLEARANCE)
        self._current_score = self._phase_score(best, self._phase)
        self._rollback_expected_report = best
        return self._emit_rollback(
            best_action_id,
            camera,
            reason=(
                "J4 anti-cycle guard: roll exhausted both directions; "
                "restore the best view and change standoff with J2"
            ),
        )

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
        if report.area_frac < self.horizontal_coverage_area_frac:
            return False
        if abs(self._yaw_error(report)) > center_limit:
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
        current_error = abs(self._yaw_error(report))
        if current_error > self.horizontal_exit_threshold:
            return True
        if self._j1_latched_error is None:
            return False
        drift_limit = max(
            self.horizontal_exit_threshold,
            self._j1_latched_error + self.clearance_drift_tolerance,
        )
        return current_error > drift_limit

    def _top_visible(self, report: MaskReport) -> bool:
        return "top" not in report.edges

    def _choose_yaw_sign(self, report: MaskReport) -> float:
        if self._direction:
            return math.copysign(1.0, self._direction)
        if "left" in report.edges and "right" not in report.edges:
            return 1.0
        if "right" in report.edges and "left" not in report.edges:
            return -1.0
        error = self._yaw_error(report)
        return -math.copysign(1.0, error or -1.0)

    def _yaw_scale(self, report: MaskReport) -> float:
        """Use a probe first, then monotonic coarse-to-fine yaw steps."""

        if self._yaw_confirmed_moves <= 0:
            return self.yaw_probe_scale
        error = abs(self._yaw_error(report))
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
        if abs(weighted) <= 1e-12:
            return 1.0
        return -math.copysign(1.0, weighted)

    def _choose_relative_sign(self, report: MaskReport, *, default: float) -> float:
        return math.copysign(1.0, self._direction) if self._direction else default

    def _yaw_error(self, report: MaskReport) -> float:
        return float(report.center_error[0]) - self._yaw_target_x

    def _learn_clearance_yaw_compensation(
        self,
        before: MaskReport | None,
        current: MaskReport,
    ) -> None:
        """Pre-compensate repeatable horizontal drift caused by J2.

        A standoff move can shift the projected board center even though J1
        did not move. Without this learned offset the workflow can repeatedly
        re-center to zero, take the same J2 step, and lose centering again.
        Only a continuous, moderate mask displacement is trusted.
        """

        if not (
            self._credible_alignment_report(before)
            and self._credible_alignment_report(current)
        ):
            return
        before_x, before_y = (float(value) for value in before.center_error)
        current_x, current_y = (float(value) for value in current.center_error)
        drift = current_x - before_x
        area_delta = abs(float(current.area_frac) - float(before.area_frac))
        if not (
            0.03 <= abs(drift) <= 0.35
            and abs(current_y - before_y) <= 0.35
            and area_delta <= max(0.12, 0.60 * float(before.area_frac))
        ):
            return
        measured_target = max(-0.08, min(0.08, -drift))
        if self._clearance_drift_samples:
            self._yaw_target_x = (
                0.65 * self._yaw_target_x + 0.35 * measured_target
            )
        else:
            self._yaw_target_x = measured_target
        self._clearance_drift_samples += 1

    def _phase_score(self, report: MaskReport, phase: _Phase) -> float:
        if phase is _Phase.J1_YAW:
            detail_deficit = max(
                0.0,
                self.min_goal_area_frac - float(report.area_frac),
            ) / max(self.min_goal_area_frac, 1e-9)
            return (
                -abs(self._yaw_error(report))
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
        self._phase_ready_streak = 0
        self._phase_confirmation_report = None
        self._direction = 0.0
        if phase is _Phase.J1_YAW:
            self._force_one_clearance_move = False
            self._force_one_up_clearance_move = False
            self._geometry_recovery_lock = False
            self._yaw_confirmed_moves = 0
            self._yaw_reversal_candidate = 0.0
            self._yaw_reversal_streak = 0
            self._yaw_reversal_count = 0
            self._last_yaw_command_direction = 0.0
            self._yaw_direction_best_error = math.inf
            self._yaw_cycle_recovery_requested = False
            self._best_yaw_action_id = None
            self._best_yaw_report = None
            self._best_yaw_score = -math.inf
            self._j1_alignment_streak = 0
            self._j1_confirmation_report = None
        # Roll counters intentionally survive J3/J4 re-entry. Otherwise a
        # top-edge fluctuation could reset the local limit and command J4
        # forever without ever reaching its anti-cycle recovery.

    def _emit(self, kind: ActionKind, camera: str, *, axial: float = 0.0, aim: tuple[float, float] = (0.0, 0.0), scale: float = 1.0, angular: float = 0.0, reason: str) -> ViewpointAction:
        action = ViewpointAction(self._next_action_id, kind, camera, axial_direction=axial, aim_direction=aim, translation_scale=scale, angular_scale=angular, reason=reason)
        self._next_action_id += 1
        if kind is ActionKind.CAMERA_ROLL:
            self._roll_moves_total += 1
        elif (
            kind in {ActionKind.BACKOFF, ActionKind.UP_CLEARANCE}
            and self._roll_moves_total
        ):
            # A clearance move changes the camera geometry, so an older J4
            # pose is not a meaningful rollback candidate. The global roll
            # count and one-recovery budget remain intact.
            self._best_roll_action_id = None
            self._best_roll_report = None
            self._best_roll_score = -math.inf
        if (
            kind is ActionKind.BASE_YAW
            and self._current_center_report is not None
            and self._credible_alignment_report(self._current_center_report)
        ):
            report = self._current_center_report
            horizontal_edges = float(
                "left" in report.edges or "right" in report.edges
            )
            score = (
                -abs(self._yaw_error(report))
                - 2.0 * horizontal_edges
                - 0.1 * abs(float(report.center_error[1]))
            )
            if score >= self._best_yaw_score:
                # This action's pre-move pose is saved by the wrapper, so its
                # id is also a rollback handle for the current measured view.
                self._best_yaw_score = score
                self._best_yaw_action_id = action.action_id
                self._best_yaw_report = report
        elif (
            kind is ActionKind.CAMERA_ROLL
            and self._current_center_report is not None
            and self._credible_alignment_report(self._current_center_report)
        ):
            report = self._current_center_report
            score = self.report_quality(report)
            if score >= self._best_roll_score:
                self._best_roll_score = score
                self._best_roll_action_id = action.action_id
                self._best_roll_report = report
        # Only robot motion needs before/after feedback scoring. OBSERVE is a
        # deliberate fresh-frame confirmation with no pending movement.
        if action.moves_robot:
            self._pending = _Pending(
                action,
                self._phase,
                self._current_score,
                self._current_center_report,
            )
        else:
            self._pending = None
        return action

    def _emit_base_yaw(
        self,
        camera: str,
        direction: float,
        *,
        angular: float,
        reason: str,
    ) -> ViewpointAction:
        """Emit at most one yaw-direction reversal per J1 entry."""

        requested = math.copysign(1.0, direction or 1.0)
        previous = self._last_yaw_command_direction
        current_error = (
            abs(self._yaw_error(self._current_center_report))
            if self._current_center_report is not None
            else math.inf
        )
        if previous and requested != math.copysign(1.0, previous):
            if self._yaw_reversal_count >= 1:
                self._yaw_cycle_recovery_requested = True
                recovery = self._recover_from_yaw_cycle(camera)
                if recovery is not None:
                    return recovery
            self._yaw_reversal_count += 1
            self._yaw_direction_best_error = current_error
        elif not previous:
            self._yaw_direction_best_error = current_error
        else:
            self._yaw_direction_best_error = min(
                self._yaw_direction_best_error,
                current_error,
            )
        self._last_yaw_command_direction = requested
        self._direction = requested
        return self._emit(
            ActionKind.BASE_YAW,
            camera,
            aim=(requested, 0.0),
            angular=angular,
            reason=reason,
        )

    def _emit_rollback(
        self,
        rollback_of: int,
        camera: str,
        *,
        reason: str,
    ) -> ViewpointAction:
        action = ViewpointAction(
            self._next_action_id,
            ActionKind.ROLLBACK,
            camera=camera,
            request_rollback=True,
            rollback_of=rollback_of,
            reason=reason,
        )
        self._next_action_id += 1
        self._pending = None
        return action

    def _terminate(self, kind: ActionKind, reason: str, camera: str | None = None) -> ViewpointAction:
        self._terminal_action = ViewpointAction(self._next_action_id, kind, camera=camera, terminal=True, reason=reason)
        self._next_action_id += 1
        return self._terminal_action
