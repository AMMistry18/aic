"""Deterministic insignia-anchored task-board viewpoint policy.

The policy is intentionally simple and ordered:

1. ``ACQUIRE``  - no board evidence: sweep J1 in one direction (one bounded
   reversal when an envelope rejects the sweep).
2. ``CENTER``   - board evidence: proportional J1 yaw (one joint, one move)
   on the board-component horizontal centroid until it is roughly
   image-centered.  There is no area or edge-fit requirement here; the board
   does not have to fit on screen.  When yaw provably cannot help (board
   clipped on both sides, oversized, or its visible mass pinned at a clipped
   top edge), the policy moves away first and re-centers later instead of
   burning the joint envelope.
3. ``ALIGN``    - J6 long-side alignment on its own, strictly after J1
   centering is confirmed and before any clearance motion.  A correction is
   only commanded after two consecutive fresh frames agree on its sign, so a
   single flaky estimate can never move the wrist.
4. ``ASCEND``   - move away from the board (base +Z, or an optical-axis
   retreat when only the bottom edge blocks) until one camera reports a
   complete view.  Both motions are monotonically away from the board, so
   image noise cannot command an approach.  A bounded, equally-confirmed J6
   assist remains available here for when misalignment only becomes
   measurable after standoff clears the frame clipping.

Board evidence is anchored to the magenta insignia whenever it is visible:
the insignia is the only purple object in the workcell, so a purple detection
is proof the board is in view even when the plate mask itself is rejected
(for example when the plate fills the frame).  In that logo-only state the
correct action is always to increase standoff, never to chase the logo with
yaw - the insignia sits off-center on the plate by design.

Before any search motion the wrapper levels the physical TCP/tool axis
straight down in bounded stages (the J5-pitch fix). J6 alignment then yaws
about that vertical tool axis, mapping the board's long plate edge to the
image's longer pixel dimension without coupling yaw back into module pitch.
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
    BACKOFF = "backoff"                # optical-axis retreat
    UP_CLEARANCE = "up_clearance"      # base +Z clearance
    CAMERA_ROLL = "camera_roll"        # J6 yaw about TCP/tool Z
    DONE = "done"
    STAGNATED = "stagnated"
    NO_VIEW = "no_view"
    DEADLINE = "deadline"
    # Retained for compatibility with the wrapper and older diagnostics.
    TRANSLATE = "translate"
    APPROACH = "approach"
    AIM = "aim"
    COMBINED = "combined"
    HORIZONTAL_SCAN = "horizontal_scan"
    ROLLBACK = "rollback"


_MOVEMENT_KINDS = frozenset({
    ActionKind.BASE_YAW, ActionKind.BACKOFF, ActionKind.UP_CLEARANCE,
    ActionKind.CAMERA_ROLL, ActionKind.TRANSLATE, ActionKind.APPROACH,
    ActionKind.AIM, ActionKind.COMBINED, ActionKind.HORIZONTAL_SCAN,
})

# The wrapper converts angular_scale into radians using its angular_step_rad
# parameter; this matches its deployed default.
_WRAPPER_ANGULAR_STEP_RAD = 0.10


class _Phase(str, Enum):
    ACQUIRE = "acquire_sweep"
    CENTER = "j1_center"
    ALIGN = "j6_align"
    ASCEND = "ascend_clearance"


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


def has_opposite_edges(edges: frozenset[str]) -> bool:
    return ("left" in edges and "right" in edges) or ("top" in edges and "bottom" in edges)


class AdaptiveViewpointPlanner:
    """ACQUIRE -> CENTER -> ASCEND with insignia-anchored evidence."""

    def __init__(
        self,
        *,
        min_goal_area_frac: float = 0.04,
        max_goal_area_frac: float = 0.45,
        expected_cameras: Sequence[str] = ("left_camera", "center_camera", "right_camera"),
        center_threshold: float = 0.15,
        recenter_threshold: float = 0.35,
        confirmation_frames: int = 2,
        max_recenter_entries: int = 4,
        yaw_gain: float = 1.5,
        min_yaw_scale: float = 0.15,
        max_yaw_scale: float = 1.5,
        max_ascend_scale: float = 3.0,
        max_stall_frames: int = 3,
        roll_align_threshold_deg: float = 12.0,
        # Six 0.30-rad bounded corrections cover the full 90-degree
        # long-edge ambiguity without ever issuing the previous 0.60-rad
        # one-shot wrist command.
        max_roll_moves: int = 6,
        roll_probe_scale: float = 1.5,
        max_roll_scale: float = 3.0,
        max_zoom_out_backoffs: int = 2,
        # Live trace: the plate's true stable estimate sits at ratio
        # 1.16-1.24 at working standoff, while square-noise flicker stays
        # below ~1.10.  Trust above 1.15 and let the two-frame sign
        # confirmation carry the rest.
        min_long_axis_ratio: float = 1.15,
        roll_confirmation_frames: int = 2,
    ) -> None:
        if not 0.0 <= min_goal_area_frac < max_goal_area_frac <= 1.0:
            raise ValueError("goal area limits must satisfy 0 <= min < max <= 1")
        cameras = tuple(str(item) for item in expected_cameras)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("expected_cameras must be non-empty and unique")
        if not 0.0 < center_threshold <= recenter_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < center <= recenter <= 1"
            )
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if max_recenter_entries < 0:
            raise ValueError("max_recenter_entries must be non-negative")
        if yaw_gain <= 0.0:
            raise ValueError("yaw_gain must be positive")
        if not 0.0 < min_yaw_scale <= max_yaw_scale:
            raise ValueError("yaw scales must satisfy 0 < min <= max")
        if max_ascend_scale < 1.0:
            raise ValueError("max_ascend_scale must be at least 1")
        if max_stall_frames < 1:
            raise ValueError("max_stall_frames must be positive")
        if not 0.0 < roll_align_threshold_deg < 90.0:
            raise ValueError("roll_align_threshold_deg must be in (0, 90)")
        if max_roll_moves < 0:
            raise ValueError("max_roll_moves must be non-negative")
        if not 0.0 < roll_probe_scale <= max_roll_scale:
            raise ValueError("roll scales must satisfy 0 < probe <= max")
        if max_zoom_out_backoffs < 0:
            raise ValueError("max_zoom_out_backoffs must be non-negative")
        if min_long_axis_ratio <= 1.0:
            raise ValueError("min_long_axis_ratio must exceed 1.0")
        if roll_confirmation_frames < 1:
            raise ValueError("roll_confirmation_frames must be positive")
        self.min_goal_area_frac = float(min_goal_area_frac)
        self.max_goal_area_frac = float(max_goal_area_frac)
        self.expected_cameras = cameras
        self.center_threshold = float(center_threshold)
        self.recenter_threshold = float(recenter_threshold)
        self.confirmation_frames = int(confirmation_frames)
        self.max_recenter_entries = int(max_recenter_entries)
        self.yaw_gain = float(yaw_gain)
        self.min_yaw_scale = float(min_yaw_scale)
        self.max_yaw_scale = float(max_yaw_scale)
        self.max_ascend_scale = float(max_ascend_scale)
        self.max_stall_frames = int(max_stall_frames)
        self.roll_align_threshold_deg = float(roll_align_threshold_deg)
        self.max_roll_moves = int(max_roll_moves)
        # Retain the parameter name for compatibility with the deployed
        # wrapper, but use it as a minimum deterministic correction rather
        # than as a polarity probe.
        self.roll_probe_scale = float(roll_probe_scale)
        self.max_roll_scale = float(max_roll_scale)
        self.max_zoom_out_backoffs = int(max_zoom_out_backoffs)
        self.min_long_axis_ratio = float(min_long_axis_ratio)
        self.roll_confirmation_frames = int(roll_confirmation_frames)
        self.reset()

    def reset(self) -> None:
        self._phase = _Phase.ACQUIRE
        self._next_action_id = 1
        self._terminal_action: ViewpointAction | None = None
        self._selected_camera: str | None = None
        self._sweep_direction = 0.0
        self._sweep_reversals = 0
        self._pending_yaw_id: int | None = None
        self._center_streak = 0
        self._recenter_entries = 0
        self._center_reject_fallbacks = 0
        self._force_ascend_once = False
        self._stall_streak = 0
        self._roll_moves = 0
        self._pending_roll_id: int | None = None
        self._roll_unavailable_reason: str | None = None
        self._roll_j1_fallback_allowed = True
        self._zoom_out_backoffs = 0
        self._roll_confirm_sign = 0.0
        self._roll_confirm_streak = 0
        self._roll_confirm_observes = 0

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

    def next_action(
        self,
        reports: Mapping[str, MaskReport],
        *,
        deadline_reached: bool = False,
    ) -> ViewpointAction:
        if self._terminal_action is not None:
            return self._terminal_action
        if deadline_reached:
            return self._terminate(
                ActionKind.DEADLINE,
                "viewpoint-search deadline reached before a complete view",
            )
        goal = self._goal_camera(reports)
        if goal is not None:
            self._selected_camera = goal
            return self._terminate(
                ActionKind.DONE, f"complete usable board view in {goal}", goal
            )

        center = reports.get("center_camera")
        self._selected_camera = "center_camera"
        if not self._board_evidence(center):
            return self._sweep(reports)
        self._sweep_reversals = 0

        self._update_roll_confirmation(center)
        error_x = self._steering_error(center)
        if self._phase is _Phase.ACQUIRE:
            self._phase = _Phase.CENTER

        if (
            self._phase in (_Phase.ALIGN, _Phase.ASCEND)
            and error_x is not None
            and abs(error_x) > self.recenter_threshold
            and not self._yaw_cannot_help(center)
            and not self._force_ascend_once
        ):
            if self._recenter_entries < self.max_recenter_entries:
                self._recenter_entries += 1
                self._phase = _Phase.CENTER
                self._center_streak = 0
            # Past the re-center budget, keep going: a complete view is
            # judged by the mask predicate, not by perfect centering.

        if self._phase is _Phase.CENTER:
            if error_x is None:
                # Logo-only evidence: the plate mask is unusable, which means
                # the camera is far too close.  Yawing toward the off-center
                # insignia would mis-center the board; standoff comes first.
                self._phase = _Phase.ASCEND
                self._center_streak = 0
                return self._emit(
                    ActionKind.UP_CLEARANCE,
                    "center_camera",
                    axial=1.0,
                    scale=2.0,
                    reason=(
                        "insignia visible but plate mask unusable; "
                        "increase standoff before centering"
                    ),
                )
            if self._yaw_cannot_help(center):
                # The visible mask is a clipped slice whose centroid yaw
                # cannot drive to center (both sides clipped, oversized, or
                # mass pinned at a clipped top edge).  Standoff first; the
                # drift path re-enters centering after the view opens up.
                self._phase = _Phase.ASCEND
                self._center_streak = 0
            elif abs(error_x) <= self.center_threshold:
                self._center_streak += 1
                if self._center_streak < self.confirmation_frames:
                    return self._emit(
                        ActionKind.OBSERVE,
                        "center_camera",
                        reason=(
                            "J1 center candidate: confirm in a fresh "
                            "center-camera frame"
                        ),
                    )
                self._phase = _Phase.ALIGN
            else:
                self._center_streak = 0
                direction = -math.copysign(1.0, error_x)
                scale = min(
                    self.max_yaw_scale,
                    max(self.min_yaw_scale, self.yaw_gain * abs(error_x)),
                )
                return self._emit_yaw(
                    direction,
                    scale,
                    reason=(
                        f"J1 proportional centering: horizontal error "
                        f"{error_x:+.3f}"
                    ),
                )

        if self._phase is _Phase.ALIGN:
            # Strict ordering: J6 long-side alignment runs on its own, after
            # J1 centering is confirmed and before any clearance motion.  A
            # single-frame estimate is never acted on; two consecutive fresh
            # frames must agree on the correction sign first.
            if not self._roll_available(center):
                self._phase = _Phase.ASCEND
            elif self._roll_confirmed():
                return self._emit_roll(center)
            else:
                return self._emit_roll_confirm_observe(
                    "J6 long-side candidate: confirm the signed estimate "
                    "in a second fresh frame"
                )

        return self._ascend(center)

    def mark_yaw_unavailable(
        self,
        action: ViewpointAction,
        *,
        reason: str,
        global_unavailable: bool = False,
    ) -> None:
        if action.kind is not ActionKind.BASE_YAW:
            raise ValueError("yaw rejection does not match a yaw action")
        if self._terminal_action is not None:
            return
        if (
            self._pending_yaw_id is None
            or action.action_id != self._pending_yaw_id
        ):
            raise ValueError(
                "yaw rejection is stale or does not match the pending action"
            )
        self._pending_yaw_id = None
        if global_unavailable:
            self._terminate(
                ActionKind.STAGNATED,
                f"J1 travel budget exhausted: {reason}",
                camera=action.camera,
            )
            return
        if self._phase is _Phase.ACQUIRE:
            if self._sweep_reversals >= 1:
                self._terminate(
                    ActionKind.STAGNATED,
                    "acquisition sweep exhausted both J1 directions "
                    f"without board evidence: {reason}",
                    camera=action.camera,
                )
                return
            self._sweep_reversals += 1
            self._sweep_direction = -(
                self._sweep_direction
                or float(action.aim_direction[0])
                or 1.0
            )
            return
        if self._center_reject_fallbacks < 1:
            # Increasing standoff shrinks the projection, so less yaw is
            # needed afterwards; try that once before giving up.
            self._center_reject_fallbacks += 1
            self._phase = _Phase.ASCEND
            self._center_streak = 0
            self._force_ascend_once = True
            return
        self._terminate(
            ActionKind.STAGNATED,
            f"J1 centering rejected at a joint envelope twice: {reason}",
            camera=action.camera,
        )

    def mark_roll_unavailable(
        self,
        action: ViewpointAction,
        *,
        reason: str,
        allow_j1_fallback: bool = True,
    ) -> None:
        """Disable J6 for this search and route the next frame to fallback.

        The wrapper calls this only after a direct J6 preflight rejection or a
        safely reversed J6 command. A useful horizontal error may then use J1;
        otherwise ASCEND enlarges the view through the joints 2-4 Cartesian
        standoff path.
        """

        if action.kind is not ActionKind.CAMERA_ROLL:
            raise ValueError("roll rejection does not match a camera-roll action")
        if self._terminal_action is not None:
            return
        if (
            self._pending_roll_id is None
            or action.action_id != self._pending_roll_id
        ):
            raise ValueError(
                "roll rejection is stale or does not match the pending action"
            )
        self._pending_roll_id = None
        self._roll_moves = self.max_roll_moves
        self._roll_unavailable_reason = str(reason)
        self._roll_j1_fallback_allowed = bool(allow_j1_fallback)
        self._phase = _Phase.ASCEND
        self._stall_streak = 0

    # ------------------------------------------------------------------
    # Phase bodies

    def _sweep(self, reports: Mapping[str, MaskReport]) -> ViewpointAction:
        self._phase = _Phase.ACQUIRE
        self._center_streak = 0
        direction = self._sweep_direction or self._side_camera_yaw_hint(reports)
        self._sweep_direction = direction
        return self._emit_yaw(
            direction,
            1.0,
            reason=(
                "acquisition sweep: no credible board or insignia evidence "
                "in the center camera"
            ),
        )

    def _ascend(self, report: MaskReport) -> ViewpointAction:
        self._force_ascend_once = False
        if not report.seen:
            return self._emit(
                ActionKind.UP_CLEARANCE,
                "center_camera",
                axial=1.0,
                scale=2.0,
                reason=(
                    "insignia-only view during ascend: plate mask unusable; "
                    "continue increasing standoff"
                ),
            )
        edges = report.edges
        bottom_blocked = (
            "bottom" in edges or report.artificial_bottom_contact
        )
        top_blocked = "top" in edges
        oversized = (
            report.area_frac > self.max_goal_area_frac
            or has_opposite_edges(edges)
        )
        clipped = bool(edges) or report.artificial_bottom_contact
        # ``MaskReport.full`` uses the base context pad. Downstream IVM needs
        # additional room for NIC cards and SC hardware which protrude beyond
        # the dark plate component. If the plate only barely clears that pad,
        # continue the same bounded retreat/clearance ladder instead of
        # observing until stagnation.
        survey_context_tight = (
            min(report.clearance_px)
            < 1.50 * float(report.context_pad_px)
        )
        roll_available = self._roll_available(report)
        orientation_ambiguous = (
            report.long_axis_ratio < self.min_long_axis_ratio
        )
        roll_exhausted = (
            self._roll_moves >= self.max_roll_moves
            and abs(report.orientation_deg) > self.roll_align_threshold_deg
        )
        j6_unavailable = orientation_ambiguous or roll_exhausted

        # If J6 cannot select a trustworthy long edge, use a measured
        # horizontal error to improve the view with J1 instead.  This lower
        # threshold is intentional: the fallback should not leave a moderately
        # off-center board parked against the gripper mask merely because the
        # normal ascend re-center threshold has not yet been crossed.
        fallback_error_x = self._steering_error(report)
        if (
            j6_unavailable
            and self._roll_j1_fallback_allowed
            and fallback_error_x is not None
            and abs(fallback_error_x) > self.center_threshold
            and not self._yaw_cannot_help(report)
            and self._recenter_entries < self.max_recenter_entries
        ):
            self._recenter_entries += 1
            self._phase = _Phase.CENTER
            self._center_streak = 0
            direction = -math.copysign(1.0, fallback_error_x)
            scale = min(
                self.max_yaw_scale,
                max(
                    self.min_yaw_scale,
                    self.yaw_gain * abs(fallback_error_x),
                ),
            )
            return self._emit_yaw(
                direction,
                scale,
                reason=(
                    "J6 long-edge alignment unavailable; J1 fallback for "
                    f"horizontal error {fallback_error_x:+.3f}"
                ),
            )

        # With no useful J1 correction, enlarge the view through an
        # optical-axis retreat (the joints 2-4 zoom-out path) and re-estimate
        # the plate aspect before allowing another J6 decision.
        if (
            j6_unavailable
            and self._zoom_out_backoffs < self.max_zoom_out_backoffs
            and report.area_frac >= self.min_goal_area_frac
            and (bottom_blocked or not clipped)
        ):
            self._zoom_out_backoffs += 1
            detail = (
                f"long/short edge ambiguous at ratio {report.long_axis_ratio:.2f}"
                if orientation_ambiguous
                else (
                    self._roll_unavailable_reason
                    or "bounded J6 correction budget exhausted"
                )
            )
            return self._emit(
                ActionKind.BACKOFF,
                "center_camera",
                axial=1.0,
                scale=1.5,
                reason=(
                    f"J6 alignment fallback: {detail}; zoom out with the "
                    "joints 2-4 Cartesian path and re-estimate"
                ),
            )
        if (
            not clipped
            and survey_context_tight
            and report.area_frac >= self.min_goal_area_frac
        ):
            zoom_exhausted = (
                self._zoom_out_backoffs >= self.max_zoom_out_backoffs
            )
            if not zoom_exhausted:
                self._zoom_out_backoffs += 1
            return self._emit(
                (
                    ActionKind.UP_CLEARANCE
                    if zoom_exhausted
                    else ActionKind.BACKOFF
                ),
                "center_camera",
                axial=1.0,
                scale=2.0 if zoom_exhausted else 1.5,
                reason=(
                    "IVM survey context remains tight after bounded zoom; "
                    "increase base +Z clearance"
                    if zoom_exhausted
                    else (
                        "plate fits but NIC/SC component context is tight; "
                        "zoom out with joints 2-4"
                    )
                ),
            )
        if not clipped and not oversized:
            if report.area_frac < self.min_goal_area_frac:
                return self._terminate(
                    ActionKind.STAGNATED,
                    "board is fully inside the frame but below the detail "
                    f"threshold (area {report.area_frac:.3f} < "
                    f"{self.min_goal_area_frac:.3f}); this policy never "
                    "approaches the board - lower the survey standoff",
                    "center_camera",
                )
            if roll_available:
                if self._roll_confirmed():
                    return self._emit_roll(report)
                return self._emit_roll_confirm_observe(
                    "J6 long-side candidate during clearance: confirm "
                    "the signed estimate in a second fresh frame"
                )
            # Fully framed at usable scale yet not complete: the remaining
            # blockers (shape, transient occlusion) are not clearance
            # problems.  Confirm on fresh frames, then stop honestly.
            self._stall_streak += 1
            if self._stall_streak >= self.max_stall_frames:
                return self._terminate(
                    ActionKind.STAGNATED,
                    "board framed at usable scale but the completeness "
                    f"predicate still fails ({', '.join(report.failure_reasons) or 'unknown'})",
                    "center_camera",
                )
            return self._emit(
                ActionKind.OBSERVE,
                "center_camera",
                reason=(
                    "board framed but not yet complete; re-checking on a "
                    "fresh frame"
                ),
            )
        self._stall_streak = 0
        if bottom_blocked and roll_available:
            # Gripper/bottom-band overlap while the board is rotated in the
            # image: aligning the board's long axis with the frame moves its
            # body out of the gripper region far more cheaply than travel.
            if self._roll_confirmed():
                return self._emit_roll(report)
            return self._emit_roll_confirm_observe(
                "J6 long-side candidate under gripper overlap: confirm "
                "the signed estimate in a second fresh frame"
            )
        if bottom_blocked and not top_blocked and not oversized:
            # The board is drifting into the lower frame/gripper band while
            # the top has clearance.  A pure optical-axis retreat shrinks the
            # projection about its current position instead of pushing it
            # further down-image the way a +Z step can.
            zoom_exhausted = (
                self._zoom_out_backoffs >= self.max_zoom_out_backoffs
            )
            if not zoom_exhausted:
                self._zoom_out_backoffs += 1
            return self._emit(
                (
                    ActionKind.UP_CLEARANCE
                    if zoom_exhausted
                    else ActionKind.BACKOFF
                ),
                "center_camera",
                axial=1.0,
                # Settling dominates per-move wall time, so continuing +Z
                # ascent takes larger steps than the bounded retreats.
                scale=2.5 if zoom_exhausted else 1.5,
                reason=(
                    "bounded optical-axis zoom-out exhausted; use joints 2-4 "
                    "base +Z clearance instead of repeating the same retreat"
                    if zoom_exhausted
                    else (
                        "ascend: lower edge or gripper-band contact with a "
                        "clear top edge; retreat along the optical axis"
                    )
                ),
            )
        scale = 1.5
        if oversized and self.max_goal_area_frac > 0.0:
            scale = min(
                self.max_ascend_scale,
                max(1.5, report.area_frac / self.max_goal_area_frac),
            )
        return self._emit(
            ActionKind.UP_CLEARANCE,
            "center_camera",
            axial=1.0,
            scale=scale,
            reason=(
                "ascend: increase base +Z clearance until the complete "
                "board fits"
            ),
        )

    # ------------------------------------------------------------------
    # Roll alignment assist

    def _roll_available(self, report: MaskReport) -> bool:
        """True when a credible signed long-side error exists and J6 may act."""

        return (
            report.seen
            and self._roll_moves < self.max_roll_moves
            # A sign estimate that keeps flapping across confirmation frames
            # is untrustworthy; after a bounded number of confirmation waits
            # J6 stands down and the clearance ladder proceeds without it.
            and self._roll_confirm_observes < 6
            and report.long_axis_ratio >= self.min_long_axis_ratio
            and abs(report.orientation_deg) > self.roll_align_threshold_deg
        )

    def _emit_roll_confirm_observe(self, reason: str) -> ViewpointAction:
        self._roll_confirm_observes += 1
        return self._emit(
            ActionKind.OBSERVE, "center_camera", reason=reason
        )

    def _update_roll_confirmation(self, report: MaskReport | None) -> None:
        """Track consecutive fresh frames agreeing on the J6 correction sign."""

        if (
            report is not None
            and report.seen
            and report.long_axis_ratio >= self.min_long_axis_ratio
            and abs(report.orientation_deg) > self.roll_align_threshold_deg
        ):
            sign = 1.0 if report.orientation_deg > 0.0 else -1.0
            if sign == self._roll_confirm_sign:
                self._roll_confirm_streak += 1
            else:
                self._roll_confirm_sign = sign
                self._roll_confirm_streak = 1
        else:
            self._roll_confirm_sign = 0.0
            self._roll_confirm_streak = 0

    def _roll_confirmed(self) -> bool:
        return self._roll_confirm_streak >= self.roll_confirmation_frames

    def _emit_roll(self, report: MaskReport) -> ViewpointAction:
        angle = float(report.orientation_deg)
        self._roll_moves += 1
        # A positive camera-module yaw rotates the fixed scene clockwise in
        # the image, so command the signed measured error directly.  Do not
        # retain or flip a direction from an earlier mask: the long-axis
        # estimator is modulo 180 degrees and every fresh frame is already a
        # complete signed correction to the image's longer dimension.
        direction = 1.0 if angle > 0.0 else -1.0
        scale = min(
            self.max_roll_scale,
            max(
                self.roll_probe_scale,
                math.radians(abs(angle)) / _WRAPPER_ANGULAR_STEP_RAD,
            ),
        )
        action = self._emit(
            ActionKind.CAMERA_ROLL,
            "center_camera",
            aim=(direction, 0.0),
            angular=scale,
            reason=(
                "J6 long-side alignment: board long axis to camera long "
                f"axis error {angle:+.1f} deg"
            ),
        )
        self._pending_roll_id = action.action_id
        return action

    # ------------------------------------------------------------------
    # Evidence helpers

    def _goal_camera(self, reports: Mapping[str, MaskReport]) -> str | None:
        candidates = [
            (name, report)
            for name, report in reports.items()
            if name in self.expected_cameras
            if report.seen
            and report.full
            and self.min_goal_area_frac
            <= report.area_frac
            <= self.max_goal_area_frac
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1].quality_score)[0]

    def _board_evidence(self, report: MaskReport | None) -> bool:
        if report is None:
            return False
        if report.logo_seen:
            return True
        if not report.seen:
            return False
        # A dark blob hugging the ignored gripper band with a very low
        # centroid is the arm/gripper, not the board.  With the insignia
        # absent there is nothing to anchor it to the plate.
        if (
            report.artificial_bottom_contact
            and float(report.center_error[1]) > 0.80
        ):
            return False
        return True

    def _yaw_cannot_help(self, report: MaskReport) -> bool:
        """True when the centroid is a clipped-slice artifact yaw cannot fix."""

        if not report.seen:
            return False
        edges = report.edges
        if "left" in edges and "right" in edges:
            return True
        if report.area_frac > self.max_goal_area_frac:
            return True
        # Board mass pinned high against a clipped top edge: the camera is in
        # front of the board looking at its near strip; standoff must come
        # first or yaw chases an uncenterable sliver across the whole
        # workspace envelope.
        if "top" in edges and float(report.center_error[1]) < -0.40:
            return True
        return False

    @staticmethod
    def _steering_error(report: MaskReport) -> float | None:
        """Horizontal centering error of the board component, if usable."""

        if report.seen:
            return float(report.center_error[0])
        return None

    def _side_camera_yaw_hint(self, reports: Mapping[str, MaskReport]) -> float:
        """Seed the sweep direction from side-camera evidence when present."""

        weighted = 0.0
        for name in ("left_camera", "right_camera"):
            report = reports.get(name)
            if report is None or not (report.seen or report.logo_seen):
                continue
            if report.seen:
                if "left" in report.edges and "right" not in report.edges:
                    weighted -= max(report.area_frac, 0.01)
                elif "right" in report.edges and "left" not in report.edges:
                    weighted += max(report.area_frac, 0.01)
                else:
                    weighted += report.center_error[0] * max(
                        report.area_frac, 0.01
                    )
            else:
                weighted += report.logo_center_error[0] * 0.01
        if abs(weighted) <= 1e-12:
            return 1.0
        return -math.copysign(1.0, weighted)

    # ------------------------------------------------------------------
    # Emission

    def _emit_yaw(
        self, direction: float, scale: float, *, reason: str
    ) -> ViewpointAction:
        requested = math.copysign(1.0, direction or 1.0)
        action = self._emit(
            ActionKind.BASE_YAW,
            "center_camera",
            aim=(requested, 0.0),
            angular=scale,
            reason=reason,
        )
        self._pending_yaw_id = action.action_id
        return action

    def _emit(
        self,
        kind: ActionKind,
        camera: str,
        *,
        axial: float = 0.0,
        aim: tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        angular: float = 0.0,
        reason: str,
    ) -> ViewpointAction:
        action = ViewpointAction(
            self._next_action_id,
            kind,
            camera,
            axial_direction=axial,
            aim_direction=aim,
            translation_scale=scale,
            angular_scale=angular,
            reason=reason,
        )
        self._next_action_id += 1
        if kind is not ActionKind.BASE_YAW:
            self._pending_yaw_id = None
        if kind is not ActionKind.CAMERA_ROLL:
            self._pending_roll_id = None
        return action

    def _terminate(
        self, kind: ActionKind, reason: str, camera: str | None = None
    ) -> ViewpointAction:
        self._terminal_action = ViewpointAction(
            self._next_action_id,
            kind,
            camera=camera,
            terminal=True,
            reason=reason,
        )
        self._next_action_id += 1
        return self._terminal_action
