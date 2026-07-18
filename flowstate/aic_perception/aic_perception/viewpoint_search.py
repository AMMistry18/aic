"""Deterministic, strictly ordered task-board viewpoint policy.

The planner is a small phase machine.  J1 first acquires and coarsely centres
the board, J6 then aligns the board's long edge, the wrapper levels joints
2--4 and acknowledges that motion with :meth:`mark_level_complete`, and only
then may the planner increase top-view clearance.  If J6 is unavailable or
displaces the projection, one bounded J1 fallback explicitly returns through
CENTER before alignment can be accepted; no J1 or J6 is used after LEVEL.

Completion is deliberately narrow and immediate.  Only a usable ``full``
report from ``center_camera`` can finish the search, and it can do so only
after leveling has completed.  Side cameras are acquisition hints, not goal
cameras.  The first qualifying post-level center frame returns ``DONE``;
there is no survey-context multiplier or extra confirmation delay.
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
    LEVEL = "j2_4_level"
    ASCEND = "ascend_clearance"
    DONE = "done"


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
    """ACQUIRE -> CENTER -> ALIGN -> LEVEL -> ASCEND -> DONE."""

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
        roll_align_threshold_deg: float = 2.0,
        # Eight 0.30-rad bounded corrections cover the full 90-degree
        # long-edge ambiguity plus re-alignment after later viewpoint
        # changes, without ever issuing a large one-shot wrist command.
        max_roll_moves: int = 8,
        # Permit fine corrections below two degrees instead of forcing the
        # previous 8.6-degree minimum wrist step.  Larger errors are corrected
        # in efficient ~26-degree transactions and remeasured after each one.
        roll_probe_scale: float = 0.20,
        max_roll_scale: float = 4.5,
        max_zoom_out_backoffs: int = 2,
        max_postlevel_translates: int = 4,
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
        if max_postlevel_translates < 0:
            raise ValueError("max_postlevel_translates must be non-negative")
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
        self.max_postlevel_translates = int(max_postlevel_translates)
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
        self._pending_clearance_id: int | None = None
        self._partial_clearance_reason: str | None = None
        self._center_streak = 0
        self._center_zoom_backoffs = 0
        self._recenter_entries = 0
        self._center_reject_fallbacks = 0
        self._force_ascend_once = False
        self._stall_streak = 0
        self._roll_moves = 0
        self._pending_roll_id: int | None = None
        self._roll_unavailable_reason: str | None = None
        self._roll_j1_fallback_allowed = True
        self._alignment_zoom_backoffs = 0
        self._clearance_zoom_backoffs = 0
        self._postlevel_translates = 0
        # Camera-plane +Y normally moves a fixed board upward in the image.
        # Keep this polarity learned from fresh post-motion frames instead of
        # assuming that every camera/controller calibration uses that sign.
        self._vertical_image_polarity = 1.0
        self._pending_vertical_sample: tuple[float, float] | None = None
        self._vertical_direction_reversals = 0
        self._roll_confirm_sign = 0.0
        self._roll_confirm_streak = 0
        self._roll_confirm_observes = 0
        self._aligned_confirm_streak = 0

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
        center = reports.get("center_camera")
        self._selected_camera = "center_camera"
        if self._terminal_action is not None:
            return self._terminal_action

        # Completion is evaluated only after the wrapper has acknowledged
        # the joints-2--4 leveling move.  This intentionally ignores a full
        # side-camera frame and a pre-level full center frame.
        if self._phase is _Phase.ASCEND and self._center_is_goal(center):
            return self._terminate(
                ActionKind.DONE,
                "first complete usable post-level center-camera view",
                "center_camera",
            )
        if deadline_reached:
            return self._terminate(
                ActionKind.DEADLINE,
                "viewpoint-search deadline reached before a complete view",
            )

        if self._phase is _Phase.LEVEL:
            return self._emit(
                ActionKind.OBSERVE,
                "center_camera",
                reason=(
                    "J1 centering and J6 long-axis alignment complete; "
                    "waiting for joints 2-4 leveling acknowledgement"
                ),
            )

        if self._phase is _Phase.ASCEND:
            return self._ascend(center)

        if self._phase is _Phase.ACQUIRE:
            if not self._board_evidence(center):
                return self._sweep(reports)
            self._phase = _Phase.CENTER
            self._sweep_reversals = 0

        if self._phase is _Phase.CENTER:
            if not self._board_evidence(center):
                # Losing the board before J6 is allowed to return to the J1
                # acquisition sweep.  Once ALIGN starts, this path is no
                # longer reachable, which enforces the one-way joint order.
                self._phase = _Phase.ACQUIRE
                return self._sweep(reports)

            error_x = self._steering_error(center)
            if self._force_ascend_once:
                self._force_ascend_once = False
                self._center_streak = 0
                return self._emit(
                    ActionKind.BACKOFF,
                    "center_camera",
                    axial=1.0,
                    scale=1.5,
                    reason=(
                        "J1 centering reached its envelope; make one "
                        "joints-2-4 zoom-out before retrying J1"
                    ),
                )
            if error_x is None or self._yaw_cannot_help(center):
                # A clipped/oversized plate does not provide a trustworthy
                # centroid.  Zoom out without changing phase, then retry J1;
                # the logo itself is intentionally not used as a center
                # target because it is offset on the board.
                self._center_streak = 0
                if self._center_zoom_backoffs >= self.max_zoom_out_backoffs:
                    return self._terminate(
                        ActionKind.STAGNATED,
                        "cannot establish a usable center-camera board "
                        "centroid after bounded joints-2-4 zoom-out",
                        "center_camera",
                    )
                self._center_zoom_backoffs += 1
                return self._emit(
                    ActionKind.BACKOFF,
                    "center_camera",
                    axial=1.0,
                    scale=1.5,
                    reason=(
                        "board centroid is clipped or unavailable; zoom out "
                        "with joints 2-4 before retrying J1 centering"
                    ),
                )
            if abs(error_x) > self.center_threshold:
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
            self._roll_confirm_sign = 0.0
            self._roll_confirm_streak = 0
            self._aligned_confirm_streak = 0

        # ALIGN owns J6.  A bounded fallback may explicitly return to CENTER,
        # but once alignment advances to LEVEL no later phase can issue J6 or J1.
        if self._phase is _Phase.ALIGN:
            return self._align(center)

        raise RuntimeError(f"unhandled viewpoint-search phase {self._phase.value}")

    def mark_level_complete(self) -> None:
        """Acknowledge that the wrapper finished its joints-2--4 leveling.

        The acknowledgement is deliberately explicit: a pre-level full frame
        must never terminate the policy.  The next fresh center frame is then
        evaluated immediately in ``ASCEND``.
        """

        if self._terminal_action is not None:
            return
        if self._phase is not _Phase.LEVEL:
            raise ValueError(
                "level completion is only valid in the j2_4_level phase"
            )
        self._phase = _Phase.ASCEND
        self._stall_streak = 0

    def request_relevel(self) -> None:
        """Return ASCEND to LEVEL when fresh TF says top-down was lost."""

        if self._terminal_action is not None:
            return
        if self._phase is not _Phase.ASCEND:
            raise ValueError(
                "re-leveling is only valid in the ascend_clearance phase"
            )
        self._phase = _Phase.LEVEL
        self._stall_streak = 0

    def request_recenter(self) -> None:
        """Re-enter visual J1/J6 correction after Cartesian IK drift.

        Leveling and clearance are requested through the Cartesian controller,
        which may distribute a pose correction through J1 or J6.  That is not
        a reason to fail the search: return to the measured visual loop and
        restore centering and two-degree long-axis alignment before continuing.
        """

        if self._terminal_action is not None:
            return
        if self._phase not in {_Phase.LEVEL, _Phase.ASCEND}:
            raise ValueError(
                "re-centering is only valid after alignment has completed"
            )
        self._phase = _Phase.CENTER
        self._center_streak = 0
        self._aligned_confirm_streak = 0
        self._roll_confirm_sign = 0.0
        self._roll_confirm_streak = 0

    def mark_clearance_partial(
        self, action: ViewpointAction, *, reason: str
    ) -> None:
        """Acknowledge a safe partial +Z move and replan from fresh vision."""

        if action.kind is not ActionKind.UP_CLEARANCE:
            raise ValueError(
                "partial clearance outcome does not match an UP_CLEARANCE action"
            )
        if self._terminal_action is not None:
            return
        if self._phase is not _Phase.ASCEND:
            raise ValueError(
                "partial clearance outcome is only valid during ASCEND"
            )
        if (
            self._pending_clearance_id is None
            or action.action_id != self._pending_clearance_id
        ):
            raise ValueError(
                "partial clearance outcome is stale or does not match the "
                "pending UP_CLEARANCE action"
            )
        self._pending_clearance_id = None
        self._partial_clearance_reason = str(reason)

    def _align(self, report: MaskReport | None) -> ViewpointAction:
        """Align the board long axis with J6, or use bounded zoom fallback."""

        if report is None or not report.seen:
            self._aligned_confirm_streak = 0
            return self._alignment_fallback(
                report,
                "center-camera plate estimate unavailable"
            )

        angle = float(report.orientation_deg)
        reliable = report.long_axis_ratio >= self.min_long_axis_ratio
        error_x = self._steering_error(report)
        if (
            reliable
            and abs(angle) <= self.roll_align_threshold_deg
            and error_x is not None
            and abs(error_x) > self.center_threshold
        ):
            self._aligned_confirm_streak = 0
            return self._alignment_fallback(
                report,
                "J6 changed the centered projection by "
                f"{error_x:+.3f}",
            )
        if reliable and abs(angle) <= self.roll_align_threshold_deg:
            self._aligned_confirm_streak += 1
            if self._aligned_confirm_streak < self.roll_confirmation_frames:
                return self._emit(
                    ActionKind.OBSERVE,
                    "center_camera",
                    reason=(
                        "J6 aligned candidate: confirm the long axis in a "
                        "second fresh center-camera frame"
                    ),
                )
            self._phase = _Phase.LEVEL
            return self._emit(
                ActionKind.OBSERVE,
                "center_camera",
                reason=(
                    f"J6 long axis aligned within {self.roll_align_threshold_deg:.1f} "
                    "deg; waiting for joints 2-4 leveling"
                ),
            )

        self._aligned_confirm_streak = 0
        if not reliable:
            return self._alignment_fallback(
                report,
                "long/short edge ambiguous at ratio "
                f"{report.long_axis_ratio:.2f}"
            )

        if self._roll_moves >= self.max_roll_moves:
            return self._alignment_fallback(
                report,
                self._roll_unavailable_reason
                or "bounded J6 alignment budget exhausted",
            )

        if self._roll_confirm_observes >= 6:
            return self._alignment_fallback(
                report,
                "signed J6 estimate did not stabilize across fresh frames",
            )

        self._update_roll_confirmation(report)
        if self._roll_confirmed():
            return self._emit_roll(report)
        return self._emit_roll_confirm_observe(
            "J6 long-side candidate: confirm the signed estimate in a "
            "second fresh frame"
        )

    def _alignment_fallback(
        self, report: MaskReport | None, detail: str
    ) -> ViewpointAction:
        error_x = self._steering_error(report) if report is not None else None
        if (
            self._roll_j1_fallback_allowed
            and error_x is not None
            and abs(error_x) > self.center_threshold
            and self._recenter_entries < self.max_recenter_entries
        ):
            self._recenter_entries += 1
            self._phase = _Phase.CENTER
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
                    f"J6 alignment fallback ({detail}); bounded J1 "
                    f"re-centering for horizontal error {error_x:+.3f}"
                ),
            )
        if self._alignment_zoom_backoffs < self.max_zoom_out_backoffs:
            self._alignment_zoom_backoffs += 1
            return self._emit(
                ActionKind.BACKOFF,
                "center_camera",
                axial=1.0,
                scale=1.5,
                reason=(
                    f"J6 alignment fallback: {detail}; bounded joints-2-4 "
                    "zoom-out and re-estimate"
                ),
            )
        return self._terminate(
            ActionKind.STAGNATED,
            "cannot establish a reliable, aligned center-camera long axis "
            f"after bounded J1/J2-4 fallback ({detail})",
            "center_camera",
        )

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
            self._phase = _Phase.CENTER
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
        safely reversed J6 command.  J6 is then permanently disabled for this
        search, but bounded J1/zoom fallback must still establish an aligned
        image before the one-way phase machine may advance to LEVEL.
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
        self._phase = _Phase.ALIGN
        self._stall_streak = 0
        self._aligned_confirm_streak = 0

    # ------------------------------------------------------------------
    # Phase bodies

    def _sweep(self, reports: Mapping[str, MaskReport]) -> ViewpointAction:
        self._phase = _Phase.ACQUIRE
        self._center_streak = 0
        self._aligned_confirm_streak = 0
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

    def _ascend(self, report: MaskReport | None) -> ViewpointAction:
        """Increase top-view clearance without revisiting J1 or J6."""

        if self._partial_clearance_reason is not None:
            # The reached measured pose is safe.  Do not blindly replace +Z
            # with optical back-away: once top-down, those vectors are nearly
            # identical.  Clear the diagnostic and let this fresh report pick
            # translation, clearance, observation, or immediate completion.
            self._partial_clearance_reason = None

        if report is None or not report.seen:
            self._pending_vertical_sample = None
            self._stall_streak = 0
            return self._emit(
                ActionKind.UP_CLEARANCE,
                "center_camera",
                axial=1.0,
                scale=2.0,
                reason=(
                    "post-level center view has no usable plate mask; "
                    "increase joints-2-4 top-view clearance"
                ),
            )

        response_note = self._consume_vertical_response(report)
        edges = report.edges
        bottom_blocked = (
            "bottom" in edges or report.artificial_bottom_contact
        )
        top_blocked = "top" in edges
        oversized = (
            report.area_frac > self.max_goal_area_frac
            or has_opposite_edges(edges)
        )
        clipped = bool(edges)

        # Once top-down, a plate that is high or low in the center image is a
        # camera-position error, not a standoff error.  Move the camera in its
        # image plane through the J2--J4 Cartesian path before zooming out.
        # This is deliberately bidirectional: positive center_error[1] means
        # the board is low in the frame, negative means it is high.  The first
        # fresh frame after every move validates the sign and flips the learned
        # polarity if the absolute vertical error got materially worse.
        error_y = float(report.center_error[1])
        vertical_clipped = top_blocked or bottom_blocked
        vertical_misaligned = (
            abs(error_y) > self.center_threshold or vertical_clipped
        )
        if (
            vertical_misaligned
            and not oversized
            and self._postlevel_translates < self.max_postlevel_translates
        ):
            direction_x = (
                (-1.0 if "left" in edges else 0.0)
                + (1.0 if "right" in edges else 0.0)
            )
            if abs(error_y) > 1e-6:
                raw_direction_y = math.copysign(1.0, error_y)
            else:
                raw_direction_y = (
                    (-1.0 if top_blocked else 0.0)
                    + (1.0 if bottom_blocked else 0.0)
                )
            direction_y = raw_direction_y * self._vertical_image_polarity
            direction_norm = math.hypot(direction_x, direction_y)
            if direction_norm > 1e-9:
                self._postlevel_translates += 1
                self._stall_streak = 0
                scale = min(1.5, max(1.0, 2.5 * abs(error_y)))
                action = self._emit(
                    ActionKind.TRANSLATE,
                    "center_camera",
                    image=(
                        direction_x / direction_norm,
                        direction_y / direction_norm,
                    ),
                    scale=scale,
                    reason=(
                        "post-level vertical visual servo: board is "
                        f"{'low' if error_y >= 0.0 else 'high'} in the frame "
                        f"(error {error_y:+.3f}); move the J2-4 camera "
                        "projection to re-center it"
                        + (f"; {response_note}" if response_note else "")
                    ),
                )
                self._pending_vertical_sample = (
                    error_y,
                    float(action.image_direction[1]),
                )
                return action

        # A lower-edge-only obstruction benefits from a bounded optical-axis
        # retreat only after the vertical visual servo has spent its bounded
        # correction budget.  It is a J2--J4 fallback, not an opportunity to
        # retry J6.
        if (
            bottom_blocked
            and not top_blocked
            and not oversized
            and self._clearance_zoom_backoffs < self.max_zoom_out_backoffs
        ):
            self._clearance_zoom_backoffs += 1
            self._stall_streak = 0
            return self._emit(
                ActionKind.BACKOFF,
                "center_camera",
                axial=1.0,
                scale=1.5,
                reason=(
                    "post-level lower-edge obstruction; bounded joints-2-4 "
                    "optical-axis zoom-out"
                ),
            )

        # Repair any remaining horizontal-only projection shift with bounded
        # camera-plane motion through the post-level joints-2--4 Cartesian
        # path before spending slow clearance moves.  Vertical clipping was
        # handled above so it can use signed image feedback.
        if (
            clipped
            and not oversized
            and self._postlevel_translates < self.max_postlevel_translates
        ):
            direction_x = (
                (-1.0 if "left" in edges else 0.0)
                + (1.0 if "right" in edges else 0.0)
            )
            direction_y = (
                0.0
            )
            direction_norm = math.hypot(direction_x, direction_y)
            if direction_norm > 1e-9:
                self._postlevel_translates += 1
                self._stall_streak = 0
                error_scale = max(
                    abs(float(report.center_error[0])),
                    abs(float(report.center_error[1])),
                )
                scale = min(2.0, max(1.0, 2.0 * error_scale))
                return self._emit(
                    ActionKind.TRANSLATE,
                    "center_camera",
                    image=(
                        direction_x / direction_norm,
                        direction_y / direction_norm,
                    ),
                    scale=scale,
                    reason=(
                        "post-level board projection is clipped on one side; "
                        "re-center with bounded joints 2-4 camera-plane motion"
                    ),
                )

        if clipped or oversized or bottom_blocked:
            self._stall_streak = 0
            scale = 1.5
            if oversized and self.max_goal_area_frac > 0.0:
                scale = min(
                    self.max_ascend_scale,
                    max(1.5, report.area_frac / self.max_goal_area_frac),
                )
            if (
                bottom_blocked
                and self._clearance_zoom_backoffs
                >= self.max_zoom_out_backoffs
            ):
                scale = max(scale, 2.5)
            return self._emit(
                ActionKind.UP_CLEARANCE,
                "center_camera",
                axial=1.0,
                scale=scale,
                reason=(
                    "ascend with joints 2-4 until the complete center-camera "
                    "board boundary fits"
                ),
            )

        if report.area_frac < self.min_goal_area_frac:
            return self._terminate(
                ActionKind.STAGNATED,
                "board is fully inside the center frame but below the detail "
                f"threshold (area {report.area_frac:.3f} < "
                f"{self.min_goal_area_frac:.3f})",
                "center_camera",
            )

        # No clipping remains, but MaskReport.full still rejects shape or
        # detail.  Motion cannot repair a transient segmentation result, so
        # re-observe briefly rather than adding unrequested context margin.
        self._stall_streak += 1
        if self._stall_streak >= self.max_stall_frames:
            return self._terminate(
                ActionKind.STAGNATED,
                "board boundary is framed but the center-camera full "
                "predicate still fails "
                f"({', '.join(report.failure_reasons) or 'unknown'})",
                "center_camera",
            )
        return self._emit(
            ActionKind.OBSERVE,
            "center_camera",
            reason=(
                "board boundary is framed but not yet usable; re-check a "
                "fresh center-camera frame"
            ),
        )

    def _consume_vertical_response(self, report: MaskReport) -> str:
        """Learn camera-plane vertical polarity from the next fresh frame."""

        pending = self._pending_vertical_sample
        if pending is None:
            return ""
        self._pending_vertical_sample = None
        previous_error, commanded_direction = pending
        current_error = float(report.center_error[1])
        worsened = (
            previous_error * current_error > 0.0
            and abs(current_error) > abs(previous_error) + 0.03
        )
        if not worsened:
            return (
                "previous vertical correction was validated by the fresh "
                f"frame ({previous_error:+.3f} -> {current_error:+.3f})"
            )
        self._vertical_image_polarity *= -1.0
        self._vertical_direction_reversals += 1
        return (
            "previous vertical correction worsened the fresh-frame error "
            f"({previous_error:+.3f} -> {current_error:+.3f}, commanded "
            f"image-y {commanded_direction:+.1f}); polarity reversed"
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

    def _center_is_goal(self, report: MaskReport | None) -> bool:
        """Completion is exactly the post-level full center-board mask.

        Phase ordering already proves that J1 centering, two-degree J6
        alignment, and physical top-down leveling completed.  Do not make the
        terminal image pass the noisy long-axis estimator a second time.
        """

        return bool(
            report is not None
            and report.seen
            and report.full
        )

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
        image: tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        angular: float = 0.0,
        reason: str,
    ) -> ViewpointAction:
        action = ViewpointAction(
            self._next_action_id,
            kind,
            camera,
            image_direction=image,
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
        if kind is ActionKind.UP_CLEARANCE:
            self._pending_clearance_id = action.action_id
        else:
            self._pending_clearance_id = None
        if action.moves_robot:
            # The viewpoint changes with every executed motion, so the
            # signed J6 estimate must be confirmed again from fresh frames.
            self._roll_confirm_observes = 0
            self._roll_confirm_sign = 0.0
            self._roll_confirm_streak = 0
            self._aligned_confirm_streak = 0
        return action

    def _terminate(
        self, kind: ActionKind, reason: str, camera: str | None = None
    ) -> ViewpointAction:
        if kind is ActionKind.DONE:
            self._phase = _Phase.DONE
        self._pending_yaw_id = None
        self._pending_roll_id = None
        self._pending_clearance_id = None
        self._terminal_action = ViewpointAction(
            self._next_action_id,
            kind,
            camera=camera,
            terminal=True,
            reason=reason,
        )
        self._next_action_id += 1
        return self._terminal_action
