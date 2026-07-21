"""Deterministic, strictly ordered task-board viewpoint policy.

The planner is a small phase machine.  J1 first acquires and coarsely centres
the board, J6 then aligns the board's long edge, the wrapper levels joints
2--4 and acknowledges that motion with :meth:`mark_level_complete`, and only
then may the planner increase survey clearance.  If J6 is unavailable or
displaces the projection, one bounded J1 fallback explicitly returns through
CENTER before alignment can be accepted.  Post-level Cartesian framing never
restarts that sequence merely because IK redistributed J1/J6; the fresh image
predicates remain authoritative and one final confirmed J6 trim is allowed at
the completed framing pose.

Completion is a synchronized three-camera survey contract.  The center camera
must retain the strict scale, long-axis, context, and gripper-clear view while
the wrapper independently enforces a controlled oblique optical-axis tilt for
IVM depth cues.  Every configured side camera must simultaneously retain board identity,
usable component context, and quantitative separation from its own calibrated
gripper mask.  Side cameras still cannot complete the policy by themselves;
they contribute required evidence and camera-plane correction directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np

from .board_visibility import (
    MaskReport,
    SurveyTargetMode,
    normalize_survey_target,
    survey_view_requirements,
)


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

# A 32-pixel plate-only margin still left one staged cable module close enough
# to a side-camera crop/occlusion boundary to disappear from IVM.  The staged
# SFP/LC cable modules mount on the board rails and protrude beyond the
# segmented plate silhouette, and the oblique survey tilt projects those
# protrusions further outward, so the clearance measured to the plate must
# reserve extra room for them.  At the live 1024x1152 resolution, 80 pixels
# keeps the whole equipment-and-pick-module footprint inside every camera
# while keeping the center projection in the documented IVM detail range.  All
# three cameras use this one value; raise it first if a live run still drops a
# module at a crop edge.
_MIN_COMPONENT_CONTEXT_PX = 80.0

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


def survey_tilt_correction(
    current_back_away: Sequence[float],
    image_right: Sequence[float],
    target_tilt_rad: float,
    tolerance_rad: float,
    *,
    max_step_rad: float = 0.12,
) -> tuple[np.ndarray, float, float]:
    """Return ``(axis, step, error)`` toward a deterministic IVM tilt.

    J6 first maps the board's long side to image-right. Projecting that axis
    into the base horizontal plane and tilting around it introduces perspective
    consistently along the board's short dimension. ``step`` is zero only when
    both tilt magnitude and azimuth are already inside the requested band.
    """

    values = (target_tilt_rad, tolerance_rad, max_step_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("survey tilt values must be finite")
    if target_tilt_rad < 0.0:
        raise ValueError("survey target tilt must be non-negative")
    if tolerance_rad < 0.0 or max_step_rad <= 0.0:
        raise ValueError("survey tilt tolerance/step is invalid")

    current = np.asarray(current_back_away, dtype=float)
    right = np.asarray(image_right, dtype=float)
    if current.shape != (3,) or right.shape != (3,):
        raise ValueError("survey axes must be three-vectors")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(right)):
        raise ValueError("survey axes must be finite")
    current_norm = float(np.linalg.norm(current))
    if current_norm < 1e-9:
        raise ValueError("current camera direction is degenerate")
    current /= current_norm

    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    horizontal_right = right - float(np.dot(right, vertical)) * vertical
    right_norm = float(np.linalg.norm(horizontal_right))
    if right_norm < 1e-9:
        raise ValueError("image-right survey axis is vertical")
    horizontal_right /= right_norm

    # Rodrigues rotation of +Z by -target_tilt around horizontal image-right.
    target = (
        vertical * math.cos(target_tilt_rad)
        - np.cross(horizontal_right, vertical) * math.sin(target_tilt_rad)
    )
    target /= float(np.linalg.norm(target))
    error = math.acos(float(np.clip(np.dot(current, target), -1.0, 1.0)))
    if error <= tolerance_rad + 1e-12:
        return horizontal_right, 0.0, error

    correction_axis = np.cross(current, target)
    axis_norm = float(np.linalg.norm(correction_axis))
    if axis_norm < 1e-9:
        correction_axis = horizontal_right
    else:
        correction_axis /= axis_norm
    return correction_axis, min(max_step_rad, error), error


class AdaptiveViewpointPlanner:
    """ACQUIRE -> CENTER -> ALIGN -> LEVEL -> ASCEND -> DONE."""

    def __init__(
        self,
        *,
        min_goal_area_frac: float = 0.04,
        max_goal_area_frac: float = 0.45,
        expected_cameras: Sequence[str] = ("center_camera",),
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
        max_occlusion_translates: int = 6,
        max_scale_adjustments: int = 3,
        min_gripper_clearance_px: float = 20.0,
        auxiliary_min_area_frac: float = 0.08,
        auxiliary_min_rectangularity: float = 0.55,
        # Side cameras are held to the same component-margin and gripper
        # clearance as the center camera: a module clipped in any one view is
        # invisible to multi-camera IVM, so no camera may pass on a looser bar.
        auxiliary_min_gripper_clearance_px: float = 20.0,
        auxiliary_context_scale: float = 1.50,
        auxiliary_max_center_error_x: float = 0.30,
        auxiliary_max_center_error_y: float = 0.30,
        max_auxiliary_translates: int = 8,
        survey_confirmation_frames: int = 2,
        # Live trace: the plate's true stable estimate sits at ratio
        # 1.16-1.24 at working standoff, while square-noise flicker stays
        # below ~1.10.  Trust above 1.15 and let the two-frame sign
        # confirmation carry the rest.
        min_long_axis_ratio: float = 1.15,
        roll_confirmation_frames: int = 2,
        survey_target: object = SurveyTargetMode.UNSPECIFIED,
    ) -> None:
        if not 0.0 <= min_goal_area_frac < max_goal_area_frac <= 1.0:
            raise ValueError("goal area limits must satisfy 0 <= min < max <= 1")
        cameras = tuple(str(item) for item in expected_cameras)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("expected_cameras must be non-empty and unique")
        if "center_camera" not in cameras:
            raise ValueError("expected_cameras must include center_camera")
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
        if max_occlusion_translates < 0:
            raise ValueError("max_occlusion_translates must be non-negative")
        if max_scale_adjustments < 0:
            raise ValueError("max_scale_adjustments must be non-negative")
        if min_gripper_clearance_px < 0.0:
            raise ValueError("min_gripper_clearance_px must be non-negative")
        if not 0.0 <= auxiliary_min_area_frac < 1.0:
            raise ValueError("auxiliary_min_area_frac must be in [0, 1)")
        if not 0.0 <= auxiliary_min_rectangularity <= 1.0:
            raise ValueError("auxiliary_min_rectangularity must be in [0, 1]")
        if auxiliary_min_gripper_clearance_px < 0.0:
            raise ValueError(
                "auxiliary_min_gripper_clearance_px must be non-negative"
            )
        if not 0.0 <= auxiliary_context_scale <= 2.0:
            raise ValueError("auxiliary_context_scale must be in [0, 2]")
        if not 0.0 < auxiliary_max_center_error_x <= 1.0:
            raise ValueError("auxiliary_max_center_error_x must be in (0, 1]")
        if not 0.0 < auxiliary_max_center_error_y <= 1.0:
            raise ValueError("auxiliary_max_center_error_y must be in (0, 1]")
        if max_auxiliary_translates < 0:
            raise ValueError("max_auxiliary_translates must be non-negative")
        if survey_confirmation_frames < 1:
            raise ValueError("survey_confirmation_frames must be positive")
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
        self.max_occlusion_translates = int(max_occlusion_translates)
        self.max_scale_adjustments = int(max_scale_adjustments)
        self.min_gripper_clearance_px = float(min_gripper_clearance_px)
        self.auxiliary_min_area_frac = float(auxiliary_min_area_frac)
        self.auxiliary_min_rectangularity = float(
            auxiliary_min_rectangularity
        )
        self.auxiliary_min_gripper_clearance_px = float(
            auxiliary_min_gripper_clearance_px
        )
        self.auxiliary_context_scale = float(auxiliary_context_scale)
        self.auxiliary_max_center_error_x = float(auxiliary_max_center_error_x)
        self.auxiliary_max_center_error_y = float(auxiliary_max_center_error_y)
        self.max_auxiliary_translates = int(max_auxiliary_translates)
        self.survey_confirmation_frames = int(survey_confirmation_frames)
        self.min_long_axis_ratio = float(min_long_axis_ratio)
        self.roll_confirmation_frames = int(roll_confirmation_frames)
        self.survey_target = normalize_survey_target(survey_target)
        self.survey_view = survey_view_requirements(self.survey_target)
        if self.survey_target is not SurveyTargetMode.UNSPECIFIED:
            # Each target declares its own terminal J6 tolerance. SFP and NIC
            # intentionally retain the exact two-degree yaw gate; accepting a
            # 20-degree residual here forced later J2-4 motion to compensate.
            self.roll_align_threshold_deg = max(
                self.roll_align_threshold_deg,
                self.survey_view.max_roll_error_deg,
            )
        if self.survey_target in {
            SurveyTargetMode.STAGED_SFP_MODULE,
            SurveyTargetMode.NIC_SFP_DESTINATION,
        }:
            # J1 owns the coarse target alignment; J6 only trims the remaining
            # image roll in bounded steps.
            self.yaw_gain = max(self.yaw_gain, 2.5)
            self.min_yaw_scale = max(self.min_yaw_scale, 0.25)
            self.max_yaw_scale = max(self.max_yaw_scale, 2.0)
            self.max_roll_scale = min(self.max_roll_scale, 2.0)
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
        self._occlusion_translates = 0
        self._auxiliary_translates = 0
        self._scale_adjustments = 0
        self._survey_ready_streak = 0
        self._gripper_motion_polarity = {
            camera: 1.0 for camera in self.expected_cameras
        }
        self._pending_gripper_sample: (
            tuple[str, int, float, float] | None
        ) = None
        self._pending_component_occlusion_sample: tuple[int, float] | None = None
        self._component_occlusion_reliefs = 0
        self._component_occlusion_use_backoff = False
        self._pending_yaw_feedback: tuple[float, float] | None = None
        self._yaw_error_per_scale: float | None = None
        self._yaw_relief_pending = False
        self._yaw_relief_moves = 0
        self._resume_center_after_yaw_relief = False
        self._resume_align_after_roll_relief = False
        self._last_roll_command_sign = 0.0
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
        if self._resume_center_after_yaw_relief:
            # The preceding bounded J2-4 +Z posture move existed only to break
            # a J1 sign-flip cycle. Resume yaw from this fresh image rather
            # than treating the relief pose as an ASCEND completion candidate.
            self._resume_center_after_yaw_relief = False
            self._partial_clearance_reason = None
            self._phase = _Phase.CENTER
            self._center_streak = 0
        if self._resume_align_after_roll_relief:
            # A J6 overshoot is allowed one direction crossing, then J2-J4
            # changes the posture before J6 may be considered again.  Resume
            # directly at ALIGN from the fresh post-clearance image.
            self._resume_align_after_roll_relief = False
            self._partial_clearance_reason = None
            self._phase = _Phase.ALIGN
            self._aligned_confirm_streak = 0

        # Completion is evaluated only after the wrapper has acknowledged
        # the joints-2--4 leveling move.  This intentionally ignores a full
        # side-camera frame and a pre-level full center frame.
        if self._phase is _Phase.ASCEND and self._survey_is_goal(reports):
            self._survey_ready_streak += 1
            if self._survey_ready_streak >= self.survey_confirmation_frames:
                return self._terminate(
                    ActionKind.DONE,
                    "strict synchronized multi-camera IVM survey confirmed in "
                    f"{self.survey_confirmation_frames} fresh center-camera "
                    "frames",
                    "center_camera",
                )
            return self._emit(
                ActionKind.OBSERVE,
                "center_camera",
                reason=(
                    "strict synchronized survey candidate: confirm center "
                    "alignment/scale plus side-camera context and gripper "
                    "clearance in one more fresh frame"
                ),
            )
        self._survey_ready_streak = 0
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
            return self._ascend(center, reports)

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
            yaw_response_note = (
                self._consume_yaw_response(error_x)
                if error_x is not None
                else ""
            )
            if self._yaw_relief_pending:
                self._yaw_relief_pending = False
                self._yaw_relief_moves += 1
                self._yaw_error_per_scale = None
                self._center_streak = 0
                self._phase = _Phase.ASCEND
                self._resume_center_after_yaw_relief = True
                return self._emit(
                    ActionKind.UP_CLEARANCE,
                    "center_camera",
                    axial=1.0,
                    scale=0.75,
                    reason=(
                        "J1 centering reversed across the target without "
                        "converging; make one bounded J2-4 clearance/roll "
                        "posture step before retrying yaw"
                    ),
                )
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
            center_limit = self._center_yaw_limit()
            if abs(error_x) > center_limit:
                self._center_streak = 0
                return self._emit_centering_yaw(
                    error_x,
                    reason=(
                        f"J1 proportional centering: horizontal error "
                        f"{error_x:+.3f}"
                        + (f"; {yaw_response_note}" if yaw_response_note else "")
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
        """Return ASCEND to LEVEL when fresh TF leaves the survey-tilt band."""

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
        current_roll_sign = (
            math.copysign(1.0, angle)
            if abs(angle) > self.roll_align_threshold_deg
            else 0.0
        )
        if (
            reliable
            and self._last_roll_command_sign != 0.0
            and current_roll_sign != 0.0
            and current_roll_sign != self._last_roll_command_sign
        ):
            # Do not command J6 back across the target. One measured crossing
            # is the maximum permitted oscillation; change J2-J4 posture and
            # re-estimate from a fresh image first.
            self._last_roll_command_sign = 0.0
            self._phase = _Phase.ASCEND
            self._resume_align_after_roll_relief = True
            return self._emit(
                ActionKind.UP_CLEARANCE,
                "center_camera",
                axial=1.0,
                scale=0.75,
                reason=(
                    "J6 correction crossed the aligned long axis once; stop "
                    "roll reversal and make one J2-4 clearance/posture step "
                    "before re-estimating yaw"
                ),
            )
        error_x = self._steering_error(report)
        center_limit = self._center_yaw_limit()
        if (
            reliable
            and abs(angle) <= self.roll_align_threshold_deg
            and error_x is not None
            and abs(error_x) > center_limit
        ):
            self._aligned_confirm_streak = 0
            return self._alignment_fallback(
                report,
                "J6 changed the centered projection by "
                f"{error_x:+.3f}",
            )
        if reliable and abs(angle) <= self.roll_align_threshold_deg:
            self._last_roll_command_sign = 0.0
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
            if (
                self._targeted
                and self.survey_target not in {
                    SurveyTargetMode.STAGED_SFP_MODULE,
                    SurveyTargetMode.NIC_SFP_DESTINATION,
                }
                and report.target_region_seen
                and float(report.target_region_visible_frac)
                >= 0.70
                and not has_opposite_edges(report.target_region_edges)
            ):
                # The plate min-area rectangle is intentionally unreliable
                # when a close target view clips the unrelated far edge.  If
                # the actual component ROI is already usable, do not zoom out
                # merely to recover cosmetic whole-board J6 geometry.
                self._phase = _Phase.LEVEL
                return self._emit(
                    ActionKind.OBSERVE,
                    "center_camera",
                    reason=(
                        "target component ROI is usable while the full-plate "
                        "long axis is ambiguous; keep the closer IVM view and "
                        "continue to target-specific J2-4 tilt"
                    ),
                )
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
            self.survey_target in {
                SurveyTargetMode.STAGED_SFP_MODULE,
                SurveyTargetMode.NIC_SFP_DESTINATION,
            }
            and report is not None
            and float(report.long_axis_ratio) < self.min_long_axis_ratio
            and self._recenter_entries < self.max_recenter_entries
        ):
            # A square/ambiguous plate estimate needs a changed azimuth, not
            # more height.  Probe with J1, then re-center and retry fine J6.
            self._recenter_entries += 1
            self._phase = _Phase.CENTER
            self._center_streak = 0
            direction = -math.copysign(1.0, error_x or 1.0)
            return self._emit_yaw(
                direction,
                0.75,
                reason=(
                    f"J6 alignment fallback ({detail}); change azimuth with "
                    "a bounded J1 probe before considering J2-4 backoff"
                ),
            )
        if (
            self._roll_j1_fallback_allowed
            and error_x is not None
            and abs(error_x)
            > self._center_yaw_limit()
            and self._recenter_entries < self.max_recenter_entries
        ):
            self._recenter_entries += 1
            self._phase = _Phase.CENTER
            self._center_streak = 0
            return self._emit_centering_yaw(
                error_x,
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
        self._pending_yaw_feedback = None
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
        # An intervening acquisition sweep is not a response to the previous
        # centering command, so it must not contaminate the learned J1 gain.
        self._pending_yaw_feedback = None
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

    def _ascend(
        self,
        report: MaskReport | None,
        reports: Mapping[str, MaskReport],
    ) -> ViewpointAction:
        """Converge the survey, returning to J1/J6 before Cartesian cleanup."""

        if self._partial_clearance_reason is not None:
            # The reached measured pose is safe.  Do not blindly replace +Z
            # with optical back-away. Clear the diagnostic and let this fresh
            # report pick translation, clearance, observation, or immediate
            # completion.
            self._partial_clearance_reason = None

        if report is None or not report.seen:
            self._pending_vertical_sample = None
            self._stall_streak = 0
            if self.survey_target in {
                SurveyTargetMode.STAGED_SFP_MODULE,
                SurveyTargetMode.NIC_SFP_DESTINATION,
            }:
                self._phase = _Phase.ACQUIRE
                return self._sweep(reports)
            return self._emit(
                ActionKind.UP_CLEARANCE,
                "center_camera",
                axial=1.0,
                scale=2.0,
                reason=(
                    "post-level center view has no usable plate mask; "
                    "increase joints-2-4 survey clearance"
                ),
            )

        response_note = self._consume_vertical_response(report)
        gripper_response_note = self._consume_gripper_response(reports)
        component_occlusion_note = self._consume_component_occlusion_response(
            reports
        )
        if component_occlusion_note:
            gripper_response_note = "; ".join(
                value
                for value in (gripper_response_note, component_occlusion_note)
                if value
            )
        if self._targeted and not report.target_region_seen:
            self._stall_streak = 0
            if self._clearance_zoom_backoffs < self.max_zoom_out_backoffs:
                self._clearance_zoom_backoffs += 1
                return self._emit(
                    ActionKind.BACKOFF,
                    "center_camera",
                    axial=1.0,
                    scale=1.0,
                    reason=(
                        f"{self.survey_target.value} region cannot be located "
                        "from the board/logo axes; make one bounded J2-4 "
                        "backoff and reacquire the fiducial"
                    ),
                )
            return self._emit(
                ActionKind.OBSERVE,
                "center_camera",
                reason=(
                    f"waiting for stable {self.survey_target.value} "
                    "board-relative region evidence"
                ),
            )
        # The staged cable modules sit close to the plate boundary. Preserve a
        # real image margin around that boundary so the full SFP/SC module row
        # remains visible to every IVM camera, without restoring the old large
        # scale-dependent empty-plate requirement.
        edges = self._framing_edges(report)
        center_error = self._framing_center_error(report)
        gripper_overlap, gripper_clearance, _ = self._gripper_metrics(report)
        bottom_blocked = (
            "bottom" in edges
            or (report.artificial_bottom_contact and not self._targeted)
        )
        top_blocked = "top" in edges
        if self._targeted:
            min_goal_area, max_goal_area = self._target_area_limits()
            framing_area = float(report.target_region_area_frac)
        else:
            min_goal_area, max_goal_area = (
                self.min_goal_area_frac,
                self.max_goal_area_frac,
            )
            framing_area = float(report.area_frac)
        oversized = (
            framing_area > max_goal_area
            or has_opposite_edges(edges)
        )
        clipped = bool(edges)

        # Leveling IK may redistribute J1 and shift the component horizontally.
        # SFP/NIC restore that error with coarse base yaw before asking J2-J4
        # to translate or increase height, then rerun the fine J6 alignment.
        error_x = float(center_error[0])
        horizontal_threshold = self._center_yaw_limit()
        if (
            self.survey_target in {
                SurveyTargetMode.STAGED_SFP_MODULE,
                SurveyTargetMode.NIC_SFP_DESTINATION,
            }
            and abs(error_x) > horizontal_threshold
            and not oversized
            and self._recenter_entries < self.max_recenter_entries
        ):
            self._recenter_entries += 1
            self._phase = _Phase.CENTER
            self._center_streak = 0
            self._aligned_confirm_streak = 0
            self._stall_streak = 0
            return self._emit_centering_yaw(
                error_x,
                reason=(
                    "post-level target drift: restore coarse alignment with "
                    f"J1 for horizontal error {error_x:+.3f} before J6/J2-4"
                ),
            )

        # A complete plate silhouette is not enough for downstream NIC/SC
        # perception when the camera-fixed gripper covers the task hardware.
        # Move the protected board envelope away from the calibrated ignore
        # mask before changing standoff.  The report supplies the desired board
        # image displacement; a static board moves opposite the camera, hence
        # the sign inversion below.
        required_gripper_clearance = (
            self.survey_view.min_gripper_clearance_px
            if self._targeted
            else self.min_gripper_clearance_px
        )
        gripper_blocked = (
            gripper_overlap > 0
            or gripper_clearance < required_gripper_clearance
            or (report.artificial_bottom_contact and not self._targeted)
        )
        if (
            gripper_blocked
            and not oversized
            and self._occlusion_translates < self.max_occlusion_translates
        ):
            if self.survey_target in {
                SurveyTargetMode.STAGED_SFP_MODULE,
                SurveyTargetMode.NIC_SFP_DESTINATION,
            }:
                return self._emit_component_occlusion_relief(
                    reports,
                    response_note=gripper_response_note,
                )
            self._occlusion_translates += 1
            self._stall_streak = 0
            overlap_scale = min(
                1.5,
                max(1.0, 1.0 + gripper_overlap / 20000.0),
            )
            return self._emit_mask_escape(
                "center_camera",
                report,
                scale=overlap_scale,
                response_note=gripper_response_note,
            )

        # Do not release IVM merely because the center projection is ideal.
        # Once it is strict, repair whichever auxiliary camera has the worst
        # mask/context evidence using that camera's calibrated image axes.
        if self._center_is_goal(report):
            auxiliary_action = self._auxiliary_survey_action(
                reports,
                response_note=gripper_response_note,
            )
            if auxiliary_action is not None:
                return auxiliary_action

        # At the controlled survey tilt, a plate that is high or low in the
        # center image is a camera-position error, not a standoff error. Move
        # the camera in its image plane through the J2--J4 Cartesian path
        # before zooming out.
        # This is deliberately bidirectional: positive center_error[1] means
        # the board is low in the frame, negative means it is high.  The first
        # fresh frame after every move validates the sign and flips the learned
        # polarity if the absolute vertical error got materially worse.
        error_y = float(center_error[1])
        vertical_clipped = top_blocked or bottom_blocked
        # Do not undo a deliberate mask-clear survey offset merely to put the
        # board centroid at image Y=0.  Repair true edge/context clipping (or a
        # gross >35% displacement); otherwise the protected-envelope servo is
        # the authority for vertical placement.
        vertical_threshold = (
            self.survey_view.center_max_error_y
            if self._targeted
            else 0.35
        )
        vertical_misaligned = vertical_clipped or abs(error_y) > vertical_threshold
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
                        "post-level vertical visual servo: survey target is "
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
        horizontal_misaligned = abs(error_x) > horizontal_threshold
        if (
            (clipped or horizontal_misaligned)
            and not oversized
            and self._postlevel_translates < self.max_postlevel_translates
        ):
            direction_x = (
                (-1.0 if "left" in edges else 0.0)
                + (1.0 if "right" in edges else 0.0)
            )
            if abs(direction_x) < 1e-9 and horizontal_misaligned:
                direction_x = math.copysign(1.0, error_x)
            direction_y = (
                0.0
            )
            direction_norm = math.hypot(direction_x, direction_y)
            if direction_norm > 1e-9:
                self._postlevel_translates += 1
                self._stall_streak = 0
                error_scale = max(
                    abs(float(center_error[0])),
                    abs(float(center_error[1])),
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
                        "post-level board projection needs horizontal framing; "
                        "re-center with bounded joints 2-4 camera-plane motion"
                    ),
                )

        if clipped or oversized or bottom_blocked:
            self._stall_streak = 0
            scale = 1.5
            if oversized and max_goal_area > 0.0:
                scale = min(
                    self.max_ascend_scale,
                    max(1.5, framing_area / max_goal_area),
                )
            if (
                bottom_blocked
                and self._clearance_zoom_backoffs
                >= self.max_zoom_out_backoffs
            ):
                scale = max(scale, 2.5)
            target_kind = (
                ActionKind.BACKOFF
                if self.survey_target in {
                    SurveyTargetMode.STAGED_SFP_MODULE,
                    SurveyTargetMode.NIC_SFP_DESTINATION,
                }
                else ActionKind.UP_CLEARANCE
            )
            return self._emit(
                target_kind,
                "center_camera",
                axial=1.0,
                scale=scale,
                reason=(
                    (
                        "bounded optical backoff after yaw alignment until the "
                        "center-camera component window fits"
                        if target_kind is ActionKind.BACKOFF
                        else "ascend with joints 2-4 until the center-camera "
                        "task component window fits"
                    )
                ),
            )

        if framing_area < min_goal_area:
            if self._scale_adjustments >= self.max_scale_adjustments:
                return self._terminate(
                    ActionKind.STAGNATED,
                    "board remains below the IVM detail scale after bounded "
                    f"approach (area {framing_area:.3f} < "
                    f"{min_goal_area:.3f})",
                    "center_camera",
                )
            self._scale_adjustments += 1
            return self._emit(
                ActionKind.APPROACH,
                "center_camera",
                axial=-1.0,
                scale=1.0,
                reason=(
                    "board is fully framed but too small for robust NIC/SC "
                    f"detail (area {framing_area:.3f}); bounded optical "
                    "approach before rechecking gripper clearance"
                ),
            )

        # Cartesian J2--J4 framing can redistribute a small amount of J6 even
        # while preserving the requested TCP orientation.  Do not throw away
        # the completed framing and rerun J1/leveling.  Once scale, context,
        # centering, mask separation, and survey tilt are ready, confirm the
        # fresh signed residual twice and trim J6 at this final pose.
        final_roll_needed = bool(
            report.survey_tilt_ready
            and report.logo_seen
            and float(report.rectangularity)
            >= (
                self.survey_view.min_rectangularity
                if self._targeted
                else 0.72
            )
            and float(report.long_axis_ratio) >= self.min_long_axis_ratio
            and abs(float(report.orientation_deg))
            > self.roll_align_threshold_deg
            and abs(float(center_error[0])) <= horizontal_threshold
            and not gripper_blocked
        )
        if final_roll_needed:
            if self._roll_moves >= self.max_roll_moves:
                return self._terminate(
                    ActionKind.STAGNATED,
                    "final-pose J6 alignment budget exhausted before the "
                    f"{self.roll_align_threshold_deg:.1f}-degree survey gate",
                    "center_camera",
                )
            self._update_roll_confirmation(report)
            if self._roll_confirmed():
                return self._emit_roll(report)
            return self._emit_roll_confirm_observe(
                "final-pose J6 trim candidate: confirm the signed residual in "
                "one more fresh framed center-camera image"
            )

        # No component-window clipping or scale error remains. Motion cannot
        # repair transient identity/shape evidence, so re-observe briefly
        # rather than adding unrequested empty plate context.
        self._stall_streak += 1
        if self._stall_streak >= self.max_stall_frames:
            return self._terminate(
                ActionKind.STAGNATED,
                "task component window is framed but the center-camera "
                "survey predicate still fails "
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

    def _consume_gripper_response(
        self, reports: Mapping[str, MaskReport]
    ) -> str:
        """Validate one camera's mask escape from its next fresh frame."""

        pending = self._pending_gripper_sample
        if pending is None:
            return ""
        self._pending_gripper_sample = None
        camera, previous_overlap, previous_clearance, commanded_y = pending
        report = reports.get(camera)
        if report is None or not report.seen:
            return f"{camera} fresh-frame mask response was unavailable"
        current_overlap, current_clearance, _ = self._gripper_metrics(report)
        worsened = (
            current_overlap > previous_overlap + 100
            or (
                previous_overlap == 0
                and current_overlap == 0
                and current_clearance + 2.0 < previous_clearance
            )
        )
        if not worsened:
            return (
                f"{camera} fresh frame validated mask escape "
                f"(overlap {previous_overlap}->{current_overlap}px, "
                f"clearance {previous_clearance:.1f}->{current_clearance:.1f}px)"
            )
        self._gripper_motion_polarity[camera] = -self._gripper_motion_polarity.get(
            camera, 1.0
        )
        return (
            f"{camera} fresh frame showed worse mask separation "
            f"(overlap {previous_overlap}->{current_overlap}px, "
            f"clearance {previous_clearance:.1f}->{current_clearance:.1f}px, "
            f"commanded image-y {commanded_y:+.1f}); polarity reversed"
        )

    def _emit_mask_escape(
        self,
        camera: str,
        report: MaskReport,
        *,
        scale: float,
        response_note: str = "",
    ) -> ViewpointAction:
        """Move through one camera's TF opposite its measured board escape."""

        overlap, clearance, escape = self._gripper_metrics(report)
        escape_x, escape_y = escape
        escape_norm = math.hypot(escape_x, escape_y)
        if escape_norm < 1e-9:
            escape_x, escape_y, escape_norm = 0.0, -1.0, 1.0
        polarity = self._gripper_motion_polarity.get(camera, 1.0)
        camera_x = -float(escape_x) / escape_norm * polarity
        camera_y = -float(escape_y) / escape_norm * polarity
        if self._targeted:
            # A single vector combines the target centering error and mask
            # separation.  This prevents the old frame-to-frame alternation
            # where centering moved the board into the gripper and the next
            # iteration immediately undid it with a mask-only correction.
            error_x, error_y = self._framing_center_error(report)
            camera_x = error_x + 0.75 * camera_x
            camera_y = (
                error_y * self._vertical_image_polarity + 0.75 * camera_y
            )
            norm = math.hypot(camera_x, camera_y)
            if norm > 1e-9:
                camera_x /= norm
                camera_y /= norm
        protected_name = (
            self.survey_target.value if self._targeted else "task-board envelope"
        )
        action = self._emit(
            ActionKind.TRANSLATE,
            camera,
            image=(camera_x, camera_y),
            scale=scale,
            reason=(
                f"{camera} protected {protected_name} intersects its "
                f"calibrated gripper mask (overlap="
                f"{overlap}px, clearance={clearance:.1f}px); move J2-4 "
                "with one coherent target-centering/mask-escape correction"
                + (f"; {response_note}" if response_note else "")
            ),
        )
        self._pending_gripper_sample = (
            camera,
            overlap,
            clearance,
            float(action.image_direction[1]),
        )
        return action

    def _component_occlusion_metrics(
        self, reports: Mapping[str, MaskReport]
    ) -> tuple[int, float]:
        metrics = [
            self._gripper_metrics(report)
            for report in reports.values()
            if report.seen and report.target_region_seen
        ]
        if not metrics:
            return 0, float("inf")
        return (
            sum(int(overlap) for overlap, _, _ in metrics),
            min(float(clearance) for _, clearance, _ in metrics),
        )

    def _consume_component_occlusion_response(
        self, reports: Mapping[str, MaskReport]
    ) -> str:
        pending = self._pending_component_occlusion_sample
        if pending is None:
            return ""
        self._pending_component_occlusion_sample = None
        previous_overlap, previous_clearance = pending
        current_overlap, current_clearance = self._component_occlusion_metrics(
            reports
        )
        improved = bool(
            current_overlap + 100 < previous_overlap
            or (
                current_overlap == previous_overlap == 0
                and current_clearance > previous_clearance + 2.0
            )
        )
        if not improved:
            # Do not reverse the shared J2-4 direction and create the observed
            # front/back roll cycle. Switch the next repair to standoff.
            self._component_occlusion_use_backoff = True
            return (
                "shared component mask relief did not improve all-camera "
                "separation "
                f"(overlap {previous_overlap}->{current_overlap}px, clearance "
                f"{previous_clearance:.1f}->{current_clearance:.1f}px); use "
                "optical backoff next"
            )
        return (
            "shared component mask relief improved all-camera separation "
            f"(overlap {previous_overlap}->{current_overlap}px, clearance "
            f"{previous_clearance:.1f}->{current_clearance:.1f}px)"
        )

    def _emit_component_occlusion_relief(
        self,
        reports: Mapping[str, MaskReport],
        *,
        response_note: str = "",
    ) -> ViewpointAction:
        """Clear an SFP/NIC row coherently without side-camera oscillation.

        A visual correction budget must not be interpreted as successful
        completion.  Keep using measured, same-direction relief until the
        synchronized predicate passes or the outer safety deadline/workspace
        guard stops the skill.
        """

        overlap, clearance = self._component_occlusion_metrics(reports)
        self._component_occlusion_reliefs += 1
        self._stall_streak = 0
        use_backoff = bool(
            self._component_occlusion_use_backoff
            or self._component_occlusion_reliefs % 3 == 0
        )
        self._component_occlusion_use_backoff = False
        target_label = self.survey_target.value
        if use_backoff:
            action = self._emit(
                ActionKind.BACKOFF,
                "center_camera",
                axial=1.0,
                scale=0.75,
                reason=(
                    f"{target_label} row remains close to a side-camera "
                    "gripper mask; "
                    "increase shared standoff without reversing J2-4 roll"
                    + (f"; {response_note}" if response_note else "")
                ),
            )
        else:
            # Every calibrated gripper occupies the lower image. Moving the
            # camera along center image-down shifts the static NIC row upward
            # in all three views while preserving the already-aligned TCP
            # orientation. IK realizes this mainly through the J2-J4 posture.
            action = self._emit(
                ActionKind.TRANSLATE,
                "center_camera",
                image=(0.0, 1.0),
                scale=0.75,
                reason=(
                    f"shift the {target_label} row upward in all cameras "
                    "with one shared "
                    "J2-4 posture correction so the gripper moves behind the "
                    "target view"
                    + (f"; {response_note}" if response_note else "")
                ),
            )
        self._pending_component_occlusion_sample = (overlap, clearance)
        return action

    def _auxiliary_survey_action(
        self,
        reports: Mapping[str, MaskReport],
        *,
        response_note: str = "",
    ) -> ViewpointAction | None:
        """Repair the worst configured side-camera survey projection."""

        auxiliary = [
            camera
            for camera in self.expected_cameras
            if camera != "center_camera"
        ]
        if not auxiliary:
            return None

        missing = [
            camera
            for camera in auxiliary
            if camera not in reports
            or not self._framing_evidence(reports[camera])
        ]
        if missing:
            self._stall_streak += 1
            if self.survey_target in {
                SurveyTargetMode.STAGED_SFP_MODULE,
                SurveyTargetMode.NIC_SFP_DESTINATION,
            }:
                self._auxiliary_translates += 1
                self._stall_streak = 0
                return self._emit(
                    ActionKind.BACKOFF,
                    "center_camera",
                    axial=1.0,
                    scale=1.0,
                    reason=(
                        "a configured side camera does not contain the full "
                        f"{self.survey_target.value} equipment band "
                        f"({missing}); widen the shared three-camera field "
                        "of view and re-observe"
                    ),
                )
            if (
                self._auxiliary_translates < self.max_auxiliary_translates
                and self._stall_streak == 1
            ):
                self._auxiliary_translates += 1
                return self._emit(
                    ActionKind.UP_CLEARANCE,
                    "center_camera",
                    axial=1.0,
                    scale=1.0,
                    reason=(
                        "configured IVM survey cameras lack board evidence "
                        f"({missing}); add one shared J2-4 clearance step"
                    ),
                )
            return self._emit(
                ActionKind.OBSERVE,
                "center_camera",
                reason=(
                    "waiting for board evidence in configured IVM survey "
                    f"cameras {missing}"
                ),
            )

        blocked = [
            camera
            for camera in auxiliary
            if self._auxiliary_gripper_blocked(reports[camera])
        ]
        if blocked:
            if self.survey_target in {
                SurveyTargetMode.STAGED_SFP_MODULE,
                SurveyTargetMode.NIC_SFP_DESTINATION,
            }:
                return self._emit_component_occlusion_relief(
                    reports,
                    response_note=response_note,
                )
            camera = max(
                blocked,
                key=lambda name: (
                    self._gripper_metrics(reports[name])[0],
                    -self._gripper_metrics(reports[name])[1],
                ),
            )
            if self._auxiliary_translates >= self.max_auxiliary_translates:
                return self._terminate(
                    ActionKind.STAGNATED,
                    "side-camera gripper separation did not converge after "
                    f"{self.max_auxiliary_translates} measured J2-4 steps",
                    camera,
                )
            self._auxiliary_translates += 1
            self._stall_streak = 0
            report = reports[camera]
            overlap, _, _ = self._gripper_metrics(report)
            scale = min(
                1.0,
                max(0.60, 0.60 + overlap / 30000.0),
            )
            return self._emit_mask_escape(
                camera,
                report,
                scale=scale,
                response_note=response_note,
            )

        # A side camera whose board component is clipped, or which lacks the
        # shared component margin, cannot be repaired by sliding within that
        # one camera: the board is simply too large for the offset side view,
        # and a lateral nudge only moves the clip to the opposite edge.  Back
        # the whole rig off instead so every camera gains margin at once and
        # the protruding pick-module row re-enters all three frames together.
        context_short = [
            camera
            for camera in auxiliary
            if self._auxiliary_context_edges(reports[camera])
        ]
        if context_short:
            if (
                self.survey_target not in {
                    SurveyTargetMode.STAGED_SFP_MODULE,
                    SurveyTargetMode.NIC_SFP_DESTINATION,
                }
                and self._auxiliary_translates >= self.max_auxiliary_translates
            ):
                return self._terminate(
                    ActionKind.STAGNATED,
                    "side-camera component coverage did not converge after "
                    f"{self.max_auxiliary_translates} shared backoff steps; "
                    "the board may exceed the shared field of view at this "
                    "survey tilt",
                    context_short[0],
                )
            self._auxiliary_translates += 1
            self._stall_streak = 0
            worst = max(
                context_short,
                key=lambda name: len(
                    self._auxiliary_context_edges(reports[name])
                ),
            )
            return self._emit(
                ActionKind.BACKOFF,
                "center_camera",
                axial=1.0,
                scale=1.5,
                reason=(
                    f"{worst} lacks the shared component margin at "
                    f"{sorted(self._auxiliary_context_edges(reports[worst]))}; "
                    "back off so every camera contains the whole board plus "
                    "the protruding pick-module row"
                ),
            )

        framing = [
            camera
            for camera in auxiliary
            if not self._auxiliary_camera_is_goal(reports[camera])
            and not self._auxiliary_context_edges(reports[camera])
            and (
                abs(float(self._framing_center_error(reports[camera])[0]))
                > (
                    self.survey_view.side_max_error_x
                    if self._targeted
                    else self.auxiliary_max_center_error_x
                )
                or abs(float(self._framing_center_error(reports[camera])[1]))
                > (
                    self.survey_view.side_max_error_y
                    if self._targeted
                    else self.auxiliary_max_center_error_y
                )
            )
        ]
        if framing:
            if self.survey_target in {
                SurveyTargetMode.STAGED_SFP_MODULE,
                SurveyTargetMode.NIC_SFP_DESTINATION,
            }:
                self._auxiliary_translates += 1
                self._stall_streak = 0
                return self._emit(
                    ActionKind.BACKOFF,
                    "center_camera",
                    axial=1.0,
                    scale=0.75,
                    reason=(
                        "side-camera component projection is offset while "
                        "the center is ready; widen the shared field of view "
                        "instead of applying opposing side-camera J2-4 axes"
                    ),
                )
            camera = max(
                framing,
                key=lambda name: (
                    len(self._auxiliary_context_edges(reports[name])),
                    abs(float(self._framing_center_error(reports[name])[0]))
                    / (
                        self.survey_view.side_max_error_x
                        if self._targeted
                        else self.auxiliary_max_center_error_x
                    ),
                    abs(float(self._framing_center_error(reports[name])[1]))
                    / (
                        self.survey_view.side_max_error_y
                        if self._targeted
                        else self.auxiliary_max_center_error_y
                    ),
                ),
            )
            report = reports[camera]
            edges = self._auxiliary_context_edges(report)
            error_x, error_y = self._framing_center_error(report)
            direction_x = (
                (-1.0 if "left" in edges else 0.0)
                + (1.0 if "right" in edges else 0.0)
            )
            if (
                abs(direction_x) < 1e-9
                and abs(float(error_x))
                > (
                    self.survey_view.side_max_error_x
                    if self._targeted
                    else self.auxiliary_max_center_error_x
                )
            ):
                direction_x = math.copysign(1.0, float(error_x))
            direction_y = (
                (-1.0 if "top" in edges else 0.0)
                + (1.0 if "bottom" in edges else 0.0)
            )
            if (
                abs(direction_y) < 1e-9
                and abs(float(error_y))
                > (
                    self.survey_view.side_max_error_y
                    if self._targeted
                    else self.auxiliary_max_center_error_y
                )
            ):
                direction_y = math.copysign(1.0, float(error_y))
            norm = math.hypot(direction_x, direction_y)
            if norm > 1e-9:
                if self._auxiliary_translates >= self.max_auxiliary_translates:
                    return self._terminate(
                        ActionKind.STAGNATED,
                        "side-camera component context did not converge after "
                        f"{self.max_auxiliary_translates} measured J2-4 steps",
                        camera,
                    )
                self._auxiliary_translates += 1
                self._stall_streak = 0
                return self._emit(
                    ActionKind.TRANSLATE,
                    camera,
                    image=(direction_x / norm, direction_y / norm),
                    scale=0.60,
                    reason=(
                        f"{camera} component survey needs framing at "
                        f"{sorted(edges)} with center error "
                        f"({float(error_x):+.3f},"
                        f"{float(error_y):+.3f}); apply a small "
                        "correction in that camera's image plane"
                    ),
                )

        not_ready = [
            camera
            for camera in auxiliary
            if not self._auxiliary_camera_is_goal(reports[camera])
        ]
        if not_ready:
            self._stall_streak += 1
            if self._stall_streak >= self.max_stall_frames:
                details = {
                    camera: self._auxiliary_rejection_reasons(reports[camera])
                    for camera in not_ready
                }
                return self._terminate(
                    ActionKind.STAGNATED,
                    f"side-camera survey evidence remained unusable: {details}",
                    not_ready[0],
                )
            return self._emit(
                ActionKind.OBSERVE,
                "center_camera",
                reason=(
                    "side-camera geometry is framed but needs a fresh stable "
                    f"observation: {not_ready}"
                ),
            )

        self._stall_streak = 0
        return None

    def _consume_yaw_response(self, error_x: float) -> str:
        """Learn center-error response per signed J1 command scale."""

        pending = self._pending_yaw_feedback
        if pending is None:
            return ""
        self._pending_yaw_feedback = None
        previous_error, signed_scale = pending
        if abs(signed_scale) < 1e-9:
            return ""
        center_limit = self._center_yaw_limit()
        crossed_without_converging = bool(
            previous_error * float(error_x) < 0.0
            and abs(previous_error) > center_limit
            and abs(float(error_x)) > center_limit
        )
        if crossed_without_converging:
            # Never spend another J1 move reversing the same miss.  The first
            # measured crossing immediately routes the next action through
            # J2-J4 clearance; subsequent crossings do the same rather than
            # falling back into an unbounded left/right limit cycle.
            self._yaw_relief_pending = True
        observed = (float(error_x) - previous_error) / signed_scale
        if not math.isfinite(observed) or observed <= 0.02:
            return "J1 response was not informative; retaining proportional gain"
        if self._yaw_error_per_scale is None:
            self._yaw_error_per_scale = observed
        else:
            self._yaw_error_per_scale = (
                0.65 * self._yaw_error_per_scale + 0.35 * observed
            )
        return f"learned J1 image response {self._yaw_error_per_scale:.3f}/scale"

    def _emit_centering_yaw(
        self, error_x: float, *, reason: str
    ) -> ViewpointAction:
        """Issue a bounded J1 correction, using live response when known."""

        direction = -math.copysign(1.0, error_x)
        center_limit = self._center_yaw_limit()
        minimum_scale = self.min_yaw_scale
        if self._targeted and abs(error_x) <= 2.0 * center_limit:
            # Keep the larger J1 steps for coarse alignment, but allow a small
            # final correction near the deadband so it cannot jump repeatedly
            # from one side of the target to the other.
            minimum_scale = min(minimum_scale, 0.10)
        scale = min(
            self.max_yaw_scale,
            max(minimum_scale, self.yaw_gain * abs(error_x)),
        )
        if self._yaw_error_per_scale is not None:
            desired_signed_scale = -0.85 * error_x / self._yaw_error_per_scale
            direction = math.copysign(1.0, desired_signed_scale)
            scale = min(
                self.max_yaw_scale,
                max(minimum_scale, abs(desired_signed_scale)),
            )
        action = self._emit_yaw(direction, scale, reason=reason)
        self._pending_yaw_feedback = (
            float(error_x),
            float(action.aim_direction[0]) * float(action.angular_scale),
        )
        return action

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
        self._last_roll_command_sign = direction
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

    @property
    def _targeted(self) -> bool:
        return self.survey_target is not SurveyTargetMode.UNSPECIFIED

    def _center_yaw_limit(self) -> float:
        """Return the J1 deadband, widened only after measured sign flips."""

        base = (
            self.survey_view.center_max_error_x
            if self._targeted
            else self.center_threshold
        )
        if self.survey_target in {
            SurveyTargetMode.STAGED_SFP_MODULE,
            SurveyTargetMode.NIC_SFP_DESTINATION,
        }:
            # Each bounded J2-4 relief grants 0.04 of hysteresis. After two
            # actual overshoots the gate is 0.26, still much tighter than the
            # historical 0.75 but wide enough to end a mechanical limit cycle.
            return min(0.26, base + 0.04 * self._yaw_relief_moves)
        return base

    def _framing_center_error(
        self, report: MaskReport
    ) -> tuple[float, float]:
        if self._targeted:
            return tuple(float(v) for v in report.target_region_center_error)
        return tuple(float(v) for v in report.center_error)

    def _framing_clearances(
        self, report: MaskReport
    ) -> tuple[float, float, float, float]:
        if self._targeted:
            return tuple(float(v) for v in report.target_region_clearance_px)
        return tuple(float(v) for v in report.clearance_px)

    @property
    def _sfp_targeted(self) -> bool:
        return self.survey_target is SurveyTargetMode.STAGED_SFP_MODULE

    def _sfp_geometry_ready(
        self,
        report: MaskReport,
        *,
        center: bool,
    ) -> bool:
        """Require a complete plate and complete sliding-module equipment band."""

        if not self._sfp_targeted:
            return True
        minimum_visible = (
            self.survey_view.center_min_visible_frac
            if center
            else self.survey_view.side_min_visible_frac
        )
        return bool(
            report.target_region_seen
            and float(report.target_region_visible_frac) >= minimum_visible
            # A single clipped equipment-band edge can be exactly the missing
            # fifth module.  Unlike other target ROIs, SFP completion therefore
            # requires the inferred full band to have physical margin on all
            # four image edges in every camera.
            and not report.target_region_edges
            and not has_opposite_edges(report.edges)
        )

    def _sfp_geometry_edges(self, report: MaskReport) -> frozenset[str]:
        edges = set(report.target_region_edges)
        if has_opposite_edges(report.edges):
            edges.update(report.edges)
        return frozenset(edges)

    def _framing_edges(self, report: MaskReport) -> frozenset[str]:
        if self._targeted:
            if self._sfp_targeted:
                if self._sfp_geometry_ready(report, center=True):
                    return frozenset()
                return self._sfp_geometry_edges(report)
            if (
                report.target_region_seen
                and float(report.target_region_visible_frac)
                >= self.survey_view.center_min_visible_frac
                and not has_opposite_edges(report.target_region_edges)
            ):
                # One target edge means only the steering context pad touches
                # the image boundary.  The actual component coverage is the
                # terminal authority; treating this as clipping caused r31's
                # repeated translate/up-clearance motions.
                return frozenset()
            return report.target_region_edges
        context_pad_px = float(report.context_pad_px)
        required = (
            max(_MIN_COMPONENT_CONTEXT_PX, 1.50 * context_pad_px)
            if context_pad_px > 0.0
            else 0.0
        )
        context_edges = {
            edge
            for edge, clearance in zip(
                ("left", "right", "top", "bottom"), report.clearance_px
            )
            if float(clearance) < required
        }
        return frozenset(set(report.edges) | context_edges)

    def _gripper_metrics(
        self, report: MaskReport
    ) -> tuple[int, float, tuple[float, float]]:
        if self._targeted:
            return (
                int(report.target_region_gripper_overlap_px),
                float(report.target_region_gripper_clearance_px),
                tuple(
                    float(v)
                    for v in report.target_region_gripper_escape_direction
                ),
            )
        return (
            int(report.gripper_overlap_px),
            float(report.gripper_clearance_px),
            tuple(float(v) for v in report.gripper_escape_direction),
        )

    def _framing_evidence(self, report: MaskReport | None) -> bool:
        if report is None or not report.seen:
            return False
        return not self._targeted or bool(report.target_region_seen)

    def _target_area_limits(self) -> tuple[float, float]:
        """Projected ROI scale bands calibrated for the 0.25-0.5 m IVM range."""

        return (
            self.survey_view.min_area_frac,
            self.survey_view.max_area_frac,
        )

    def _center_is_goal(self, report: MaskReport | None) -> bool:
        """Return whether center retains the task-component survey window.

        ``MaskReport.full`` includes a scale-dependent context pad around the
        entire black plate. IVM instead needs component-scale pixels and an
        uncropped plate footprint containing the SC/SFP zones and five NIC
        rails. Permit that dynamic context gate to be false at close range,
        while retaining a small physical image-border allowance.
        """

        if report is None or not report.seen or not report.survey_tilt_ready:
            return False
        if self._targeted:
            error_x, error_y = self._framing_center_error(report)
            overlap, clearance, _ = self._gripper_metrics(report)
            min_area, max_area = self._target_area_limits()
            return bool(
                report.survey_target == self.survey_target.value
                and report.target_region_seen
                # ``target_region_full`` includes an artificial 20--48 px
                # context pad.  It was useful for steering but kept moving
                # after the actual hardware was completely visible.  Require
                # the projected component region itself instead, retaining a
                # strict no-opposite-edges check so this cannot pass on a
                # narrow cropped slice of the target.
                and report.target_region_visible_frac
                >= self.survey_view.center_min_visible_frac
                and not has_opposite_edges(report.target_region_edges)
                and self._sfp_geometry_ready(report, center=True)
                # Target ROI scale is the image-space proxy for IVM's useful
                # 0.25-0.5 m range.  It avoids making the much smaller SC-port
                # zone obey an unrelated whole-board area threshold.
                and min_area
                <= float(report.target_region_area_frac)
                <= max_area
                and abs(error_x) <= self._center_yaw_limit()
                and abs(error_y) <= self.survey_view.center_max_error_y
                and float(report.long_axis_ratio) >= self.min_long_axis_ratio
                and abs(float(report.orientation_deg))
                <= self.roll_align_threshold_deg
                and overlap == 0
                and clearance >= self.survey_view.min_gripper_clearance_px
            )
        context_pad_px = float(report.context_pad_px)
        component_border_px = (
            max(_MIN_COMPONENT_CONTEXT_PX, 1.50 * context_pad_px)
            if context_pad_px > 0.0
            else 0.0
        )
        return bool(
            self.min_goal_area_frac
            <= float(report.area_frac)
            <= self.max_goal_area_frac
            and float(report.rectangularity) >= 0.72
            and abs(float(report.center_error[0])) <= self.center_threshold
            and report.logo_seen
            and float(report.long_axis_ratio) >= self.min_long_axis_ratio
            and abs(float(report.orientation_deg))
            <= self.roll_align_threshold_deg
            and not report.artificial_bottom_contact
            and int(report.gripper_overlap_px) == 0
            and float(report.gripper_clearance_px)
            >= self.min_gripper_clearance_px
            and min(report.clearance_px) >= component_border_px
        )

    def _survey_is_goal(self, reports: Mapping[str, MaskReport]) -> bool:
        """Require simultaneous usable evidence from every IVM camera."""

        if not self._center_is_goal(reports.get("center_camera")):
            return False
        if self._targeted:
            # IVM consumes the synchronized three-camera capture. Require the
            # selected component ROI in each configured image, while avoiding
            # the old and unnecessary requirement that every image contain the
            # full board.
            if set(self.expected_cameras) - set(reports):
                return False
            return all(
                camera == "center_camera"
                or self._auxiliary_camera_is_goal(reports.get(camera))
                for camera in self.expected_cameras
            )
        if set(self.expected_cameras) - set(reports):
            return False
        return all(
            camera == "center_camera"
            or self._auxiliary_camera_is_goal(reports.get(camera))
            for camera in self.expected_cameras
        )

    def _auxiliary_context_edges(self, report: MaskReport) -> frozenset[str]:
        if self._targeted:
            if self._sfp_targeted:
                if self._sfp_geometry_ready(report, center=False):
                    return frozenset()
                return self._sfp_geometry_edges(report)
            if (
                report.target_region_seen
                and float(report.target_region_visible_frac)
                >= self.survey_view.side_min_visible_frac
                and not has_opposite_edges(report.target_region_edges)
            ):
                return frozenset()
            return report.target_region_edges
        context_pad_px = float(report.context_pad_px)
        required = (
            max(
                _MIN_COMPONENT_CONTEXT_PX,
                self.auxiliary_context_scale * context_pad_px,
            )
            if context_pad_px > 0.0
            else 0.0
        )
        return frozenset(
            edge
            for edge, clearance in zip(
                ("left", "right", "top", "bottom"), report.clearance_px
            )
            if float(clearance) < required
        )

    def _auxiliary_gripper_blocked(self, report: MaskReport) -> bool:
        overlap, clearance, _ = self._gripper_metrics(report)
        minimum_clearance = (
            self.survey_view.min_gripper_clearance_px
            if self._targeted
            else self.auxiliary_min_gripper_clearance_px
        )
        return bool(
            (report.artificial_bottom_contact and not self._targeted)
            or overlap > 0
            or clearance < minimum_clearance
        )

    def _auxiliary_rejection_reasons(
        self, report: MaskReport | None
    ) -> tuple[str, ...]:
        if report is None or not report.seen:
            return ("board_not_seen",)
        reasons: list[str] = []
        if self._targeted:
            error_x, error_y = self._framing_center_error(report)
            min_area, max_area = self._target_area_limits()
            if report.survey_target != self.survey_target.value:
                reasons.append("wrong_survey_target")
            if not report.target_region_seen:
                reasons.append("target_region_not_seen")
            # A component ROI can lie inside the image while its *steering*
            # context pad touches an edge.  That is still an IVM-ready view
            # when the actual projected component coverage is high enough.
            if (
                float(report.target_region_visible_frac)
                < self.survey_view.side_min_visible_frac
                or has_opposite_edges(report.target_region_edges)
            ):
                reasons.append("target_region_insufficient_coverage")
            if self._sfp_targeted and not self._sfp_geometry_ready(
                report, center=False
            ):
                reasons.append("sfp_equipment_band_not_clear")
            if (
                float(report.target_region_area_frac)
                < self.survey_view.side_min_area_scale * min_area
            ):
                reasons.append("insufficient_detail")
            if (
                float(report.target_region_area_frac)
                > self.survey_view.side_max_area_scale * max_area
            ):
                reasons.append("target_too_large")
            if abs(error_x) > self.survey_view.side_max_error_x:
                reasons.append("horizontal_off_center")
            if abs(error_y) > self.survey_view.side_max_error_y:
                reasons.append("vertical_off_center")
            if self._auxiliary_gripper_blocked(report):
                reasons.append("gripper_mask_contact")
            return tuple(reasons)
        if not report.logo_seen:
            reasons.append("logo_not_seen")
        if float(report.area_frac) < self.auxiliary_min_area_frac:
            reasons.append("insufficient_detail")
        if float(report.rectangularity) < self.auxiliary_min_rectangularity:
            reasons.append("unstable_board_shape")
        if abs(float(report.center_error[0])) > self.auxiliary_max_center_error_x:
            reasons.append("horizontal_off_center")
        if abs(float(report.center_error[1])) > self.auxiliary_max_center_error_y:
            reasons.append("vertical_off_center")
        if self._auxiliary_context_edges(report):
            reasons.append("component_context_tight")
        if self._auxiliary_gripper_blocked(report):
            reasons.append("gripper_mask_contact")
        return tuple(reasons)

    def _auxiliary_camera_is_goal(self, report: MaskReport | None) -> bool:
        return not self._auxiliary_rejection_reasons(report)

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
        if self._targeted and report.target_region_seen:
            # Explicit surveys steer the component ROI, not the entire dark
            # plate.  A close board may touch both image edges while all five
            # plugs/ports still have a trustworthy target centroid; r31
            # incorrectly backed away roughly 18 cm in this state.
            return False
        edges = report.edges
        if "left" in edges and "right" in edges:
            return True
        if report.area_frac > self.max_goal_area_frac:
            return True
        # A credible plate/logo mask touching only the top or bottom edge still
        # has a useful horizontal centroid. Let J1 and J6 finish, then give
        # vertical framing to the signed J2-J4 image-plane servo. Treating the
        # live -0.40 top-edge view as unusable spent the zoom budget and ended
        # the search before that servo could run.
        return False

    def _steering_error(self, report: MaskReport) -> float | None:
        """Horizontal centering error of the board component, if usable."""

        if self._targeted and report.target_region_seen:
            return float(report.target_region_center_error[0])
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
            # Only a fresh frame immediately following J1 is valid response
            # evidence. Never attribute J6 or Cartesian motion to the yaw
            # learner or its oscillation detector.
            self._pending_yaw_feedback = None
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
