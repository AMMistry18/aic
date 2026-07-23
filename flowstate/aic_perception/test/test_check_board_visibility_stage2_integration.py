"""Source contracts for in-place geometric SFP Stage-2 integration."""

from __future__ import annotations

import ast
import copy
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "check_board_visibility_skill.py"
PROTO_PATH = Path(__file__).resolve().parents[1] / "check_board_visibility_skill.proto"


def _source_and_class():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    skill = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CheckBoardVisibilitySkill"
    )
    return source, skill


def _method(name: str):
    _, skill = _source_and_class()
    return next(
        node
        for node in skill.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(node: ast.AST, attribute: str) -> list[int]:
    return sorted(
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attribute
    )


def _named_calls(node: ast.AST, name: str) -> list[int]:
    return sorted(
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
    )


def test_stage1_done_hands_off_in_place_instead_of_returning_success():
    source, _ = _source_and_class()
    done_start = source.index("if action.kind == ActionKind.DONE:")
    terminal_start = source.index("if action.terminal:", done_start)
    done_branch = source[done_start:terminal_start]

    assert "_stage2_has_complete_landmark" in done_branch
    assert "_move_to_acquire_complete_logo" in done_branch
    assert "_run_sfp_geometric_stage2" in done_branch
    # The shared deployed skill keeps the legacy terminal contract for the
    # NIC/SC enum values, while SFP (0/1) continues into Stage 2.
    assert "if not staged_sfp_target" in done_branch
    assert "result.done = True" in done_branch
    assert "continue" in done_branch


def test_stage2_has_a_reserved_budget_inside_the_configured_total():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "stage2_reserve_sec" in method_source
    assert "overall_deadline = started_at + search_timeout_sec" in method_source
    assert "deadline = overall_deadline - stage2_reserve_sec" in method_source
    assert "deadline=overall_deadline" in method_source


def test_logo_acquisition_is_bounded_measured_and_never_a_blind_sweep():
    source, _ = _source_and_class()
    method = _method("_move_to_acquire_complete_logo")
    method_source = ast.get_source_segment(source, method)

    assert "max_logo_acquisition_moves = 5" in source
    assert "logo_acquisition_moves < max_logo_acquisition_moves" in source
    assert "result.moves_executed >= max_logo_acquisition_moves" in source
    assert "detect_purple_logo" in method_source
    assert "_camera_axes_in_base" in method_source
    assert "_gripper_pose" in method_source
    assert "would be blind" in method_source
    assert len(_calls(method, "move_smooth")) == 1
    assert not any(isinstance(item, ast.While) for item in ast.walk(method))


def test_stagnated_legacy_stage1_hands_staged_sfp_to_stage2_after_acquisition():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "if action.terminal:" in method_source
    assert "if staged_sfp_target:" in method_source
    assert "legacy Stage-1 planner stagnated" in method_source
    assert "handing off to Stage 2" in method_source
    assert "self._run_sfp_geometric_stage2(" in method_source


def test_stage2_requires_complete_unobstructed_logo_and_full_board_quad():
    source, _ = _source_and_class()
    landmarks = _method("_stage2_landmarks")
    method_source = ast.get_source_segment(source, landmarks)

    for contract in (
        "report.seen",
        "report.full",
        "logo_margin < 8",
        "purple logo intersects the gripper uncertainty mask",
        "board outline touches the physical image boundary",
        "approxPolyDP",
    ):
        assert contract in method_source


def test_stage2_consumes_all_camera_info_and_full_camera_tcp_tf():
    source, _ = _source_and_class()
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(source, method)
    transform = _method("_base_transform_at")
    transform_source = ast.get_source_segment(source, transform)

    assert "snapshot.calibrations" in method_source
    assert "calibration.camera_matrix" in method_source
    assert "calibration.distortion" in method_source
    assert "base_T_tcp_by_camera[name].inverse().compose(transform)" in method_source
    assert "lookup_transform" in transform_source
    assert "self.config.camera_frames.values()" in transform_source
    assert "self.config.gripper_frame" in transform_source
    assert "frames_are_approved_camera_pair" in method_source


def test_stage2_orders_pose_search_motion_fresh_triplet_and_verification():
    method = _method("_run_sfp_geometric_stage2")
    estimate = _named_calls(method, "estimate_board_pose")
    search = _named_calls(method, "search_survey_pose")
    motions = _calls(method, "move_smooth")
    fresh_grab = _calls(method, "grab")
    verification = _named_calls(method, "verify_survey_view")

    # Initial, fresh, and independently re-PnP'd confirmation triplets.
    assert len(estimate) == 3
    assert len(search) == 1
    assert len(fresh_grab) == 2
    # The two calls are inside per-camera loops, so each triplet is verified
    # with each camera's own timestamped pose rather than one shared pose.
    assert len(verification) == 2
    method_source = ast.get_source_segment(SOURCE_PATH.read_text(encoding="utf-8"), method)
    assert "for estimate in fresh_estimates" in method_source
    assert "for name, estimate in confirmation_estimates.items()" in method_source
    # Optional outward retreat, optional in-place orientation waypoint, then
    # the fixed-orientation survey translation.
    assert 1 <= len(motions) <= 3
    assert (
        estimate[0]
        < search[0]
        < motions[-1]
        < fresh_grab[0]
        < estimate[1]
        < verification[0]
    )
    assert verification[0] < fresh_grab[1] < verification[-1]


def test_stage2_uses_a_relaxed_pose_seed_before_strict_final_verification():
    source, _ = _source_and_class()
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(source, method)

    # Stage 2 must not reject a usable handoff merely because the initial
    # outline is noisier than the final-pose threshold.  Its physical motion
    # and terminal verification remain separately guarded below.
    assert "max_reprojection_error_px=20.0" in method_source
    assert "max_logo_error_px=120.0" in method_source
    assert "verify_survey_view" in method_source


def test_stage2_searches_inside_the_execution_workspace_guard():
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(SOURCE_PATH.read_text(encoding="utf-8"), method)

    assert "max_reach_m=1.20" in method_source
    assert "min_height_m=0.02" in method_source


def test_stage2_motion_uses_remaining_deadline_not_legacy_per_move_caps():
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(SOURCE_PATH.read_text(encoding="utf-8"), method)

    assert method_source.count("timeout_sec=remaining") == 3
    assert "max(move_timeout_sec, 8.0)" not in method_source
    assert "max(move_timeout_sec, 12.0)" not in method_source


def test_fresh_pixels_and_timestamp_skew_gate_terminal_done():
    source, _ = _source_and_class()
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(source, method)

    assert "min_cameras=len(expected)" in method_source
    assert "frames_within_skew(50_000_000)" in method_source
    assert "fresh_report = analyze_board" in method_source
    assert "fresh_report.gripper_overlap_px > 0" in method_source
    assert "fresh_estimates" in method_source
    assert len(_named_calls(method, "board_pose_set_is_consistent")) == 2
    assert "second settled three-camera triplet" in method_source
    assert method_source.index("verification_by_camera") < method_source.index(
        "result.done = True"
    )


def test_stage2_uses_image_timestamp_tf_and_samples_a_safe_cartesian_path():
    source, _ = _source_and_class()
    transform = _method("_base_transform_at")
    transform_source = ast.get_source_segment(source, transform)
    method_source = ast.get_source_segment(source, _method("_run_sfp_geometric_stage2"))

    assert "Time(nanoseconds=int(stamp_ns))" in transform_source
    assert "returned_ns" in transform_source
    assert "50_000_000" in transform_source
    assert method_source.count("_base_transform_at(") >= 6
    assert "path_is_safe" in method_source
    assert "sampled_cartesian_path_is_safe" in method_source
    assert "allow_outward_retreat=True" in method_source
    assert "rotation_clearance_m = 0.40" in method_source
    assert "safe orientation waypoint" in method_source
    assert "one and only retreat" in method_source


def test_stage2_rejection_is_a_normal_not_done_result():
    source, _ = _source_and_class()
    method = _method("_stage2_not_done")
    method_source = ast.get_source_segment(source, method)

    assert "result.success = True" in method_source
    assert "result.done = False" in method_source
    assert "result.component_coverage_ready = False" in method_source
    assert "raise" not in method_source


def test_target_dispatch_runs_geometry_only_for_staged_sfp_modes():
    # Execute the small extracted policy helper, rather than merely checking
    # source strings, so deployed NIC/SC enum values cannot silently route to
    # the loose-SFP Stage-2 path.
    helper = copy.deepcopy(_method("_uses_staged_sfp_stage2"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    dispatch = namespace["_uses_staged_sfp_stage2"]
    assert dispatch(0)
    assert dispatch(1)
    assert not dispatch(2)
    assert not dispatch(3)
    assert not dispatch(99)


def test_deployed_target_enum_and_compatibility_fields_are_preserved():
    proto = PROTO_PATH.read_text(encoding="utf-8")
    for declaration in (
        "STAGED_SFP_MODULE = 1;",
        "NIC_SFP_DESTINATION = 2;",
        "SC_DESTINATION_PORT = 3;",
        "double target_center_tilt_deg = 27;",
        "double center_tilt_tolerance_deg = 28;",
        "double ivm_min_center_board_area_frac = 29;",
        "double ivm_max_center_board_area_frac = 30;",
        "SurveyTarget survey_target = 31;",
    ):
        assert declaration in proto
