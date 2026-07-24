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

    # The DONE branch hands SFP off through the shared closure and keeps the
    # legacy terminal contract for the NIC/SC enum values.
    assert "handoff_to_stage2(snapshot, reports)" in done_branch
    assert "if not staged_sfp_target" in done_branch
    assert "result.done = True" in done_branch
    # Stage 1 no longer imposes a wall-clock deadline on the handoff.
    assert "deadline" not in done_branch


def test_stage1_has_no_wall_clock_deadline_and_hands_off_on_exposed_insignia():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    # Stage 1 imposes no wall-clock deadline; the planner terminates on its own
    # stall condition and an exposed insignia hands off to Stage 2 early.
    assert "deadline = started_at" not in method_source
    assert "stage2_reserve_sec" not in method_source
    assert "deadline_reached=False" in method_source
    assert "_stage2_has_complete_landmark(" in method_source
    assert "handoff_to_stage2(snapshot, reports)" in method_source


def test_stage1_does_not_inject_logo_acquisition_moves():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "max_logo_acquisition_moves" not in method_source
    assert "_move_to_acquire_complete_logo(" not in method_source


def test_stagnated_legacy_stage1_hands_its_final_triplet_to_stage2():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "if action.terminal:" in method_source
    assert "if staged_sfp_target:" in method_source
    assert "Stage-1 planner ended" in method_source
    assert "to geometric Stage 2" in method_source
    assert "handoff_to_stage2(snapshot, reports)" in method_source
    # The shared closure still invokes the geometric stage.
    assert "self._run_sfp_geometric_stage2(" in method_source


def test_stage2_seed_uses_insignia_not_the_board_outline_or_full_report():
    source, _ = _source_and_class()
    landmarks = _method("_stage2_landmarks")
    method_source = ast.get_source_segment(source, landmarks)

    for contract in (
        "detect_insignia_polygon",
        "insignia intersects the gripper uncertainty mask",
        "insignia touches the physical image boundary",
    ):
        assert contract in method_source
    # The board outline is no longer a handoff dependency: no dark-plate quad
    # recovery remains in the seed.
    assert "cv2.connectedComponentsWithStats" not in method_source
    assert "board outline contour is unavailable" not in method_source
    report_attributes = {
        node.attr
        for node in ast.walk(landmarks)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "report"
    }
    assert "full" not in report_attributes
    assert "seen" not in report_attributes


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


def test_sfp_stage2_is_perception_only_and_publishes_a_survey_pose():
    method = _method("_run_sfp_geometric_stage2")
    estimate = _named_calls(method, "estimate_board_pose_from_insignia")
    search = _named_calls(method, "search_survey_pose")
    motions = _calls(method, "move_smooth")
    grabs = _calls(method, "grab")
    method_source = ast.get_source_segment(
        SOURCE_PATH.read_text(encoding="utf-8"), method
    )

    # One insignia PnP and one survey search; no motion and no fresh-triplet grab.
    assert len(estimate) == 1
    assert len(search) == 1
    assert len(motions) == 0
    assert len(grabs) == 0
    # Publishes a pose instead of executing/confirming a move.
    assert "result.survey_pose" in method_source
    assert "sfp_survey_pose_published" in method_source
    for gone in (
        "move_smooth",
        "_confirm_coverage",
        "verify_survey_view",
        "path_is_safe",
        "needs_retreat",
    ):
        assert gone not in method_source, gone
    assert estimate[0] < search[0]


def test_sfp_survey_pose_is_the_native_intrinsic_pose_in_base_link():
    source, _ = _source_and_class()
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(source, method)

    assert "estimate_board_pose_from_insignia(" in method_source
    # Published as a native intrinsic_proto.Pose (position + orientation) so it
    # binds to Move Robot's Cartesian target_frame_offset.
    assert "result.survey_pose.position.x" in method_source
    assert "result.survey_pose.orientation.w" in method_source
    assert "result.target_frame = self.config.base_frame" in method_source


def test_result_proto_declares_intrinsic_pose_survey_pose_output():
    proto = PROTO_PATH.read_text(encoding="utf-8")
    assert 'import "intrinsic/math/proto/pose.proto";' in proto
    assert "intrinsic_proto.Pose survey_pose" in proto


def test_stage2_searches_inside_the_ur5e_reach_for_the_sfp_sector():
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(SOURCE_PATH.read_text(encoding="utf-8"), method)

    # Reach guard is the real UR5e envelope, and the survey frames one sector.
    assert "max_reach_m=0.85" in method_source
    assert "min_height_m=0.02" in method_source
    assert "sfp_sector_corners()" in method_source


def test_sfp_stage2_does_no_motion_and_has_no_time_budget():
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(SOURCE_PATH.read_text(encoding="utf-8"), method)

    assert "move_smooth" not in method_source
    assert "deadline" not in method_source
    assert "timeout_sec=remaining" not in method_source


def test_sfp_survey_pose_publish_sets_done_before_no_confirmation():
    source, _ = _source_and_class()
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(source, method)

    assert "result.done = True" in method_source
    assert "result.target_valid = True" in method_source
    assert "board_pose_set_is_consistent" not in method_source
    assert method_source.index("result.survey_pose.position.x") < method_source.index(
        "result.done = True"
    )


def test_stage2_uses_image_timestamp_tf_for_every_camera():
    source, _ = _source_and_class()
    transform = _method("_base_transform_at")
    transform_source = ast.get_source_segment(source, transform)
    method_source = ast.get_source_segment(source, _method("_run_sfp_geometric_stage2"))

    assert "Time(nanoseconds=int(stamp_ns))" in transform_source
    assert "returned_ns" in transform_source
    assert "50_000_000" in transform_source
    # base_T_tcp and base_T_cam are both resolved at each image timestamp.
    assert method_source.count("_base_transform_at(") >= 2


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
