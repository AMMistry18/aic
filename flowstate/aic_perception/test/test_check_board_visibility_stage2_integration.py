"""Source contracts for in-place geometric SFP Stage-2 integration."""

from __future__ import annotations

import ast
import copy
import math
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

    # Reachability is decided by the real UR5e IK gate (calibrated from the live
    # joint state), not the base-origin sphere -- which wrongly rejected the
    # reachable far, bore-facing poses and admitted unsolvable ones.  The sphere
    # survives only as a loose fallback at the full envelope, and the survey
    # frames exactly one sector, chosen from the target mode.
    assert "reachable=reachable_fn" in method_source
    assert "UR5eArm.autocalibrate(" in method_source
    assert "max_reach_m=0.85" in method_source
    assert "min_height_m=0.02" in method_source
    assert "self._sector_for_target(survey_target)" in method_source


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


def test_every_deployed_target_mode_uses_the_geometric_survey():
    # Execute the extracted policy helper rather than checking source strings.
    # All three deployed enum values now take the geometric sector survey; the
    # legacy adaptive search survives only as the Stage-1 insignia fallback.
    helper = copy.deepcopy(_method("_uses_geometric_survey"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    dispatch = namespace["_uses_geometric_survey"]
    assert dispatch(0)  # UNSPECIFIED (historical SFP default)
    assert dispatch(1)  # STAGED_SFP_MODULE
    assert dispatch(2)  # NIC_SFP_DESTINATION
    assert dispatch(3)  # SC_DESTINATION_PORT
    assert not dispatch(99)


def test_each_target_mode_maps_to_its_own_board_sector():
    source, _ = _source_and_class()
    selector = ast.get_source_segment(source, _method("_sector_for_target"))

    assert "nic_sector_corners()" in selector
    assert "sc_sector_corners()" in selector
    assert "sfp_sector_corners()" in selector
    # NIC=2, SC=3, everything else (0/1) is the staged-SFP rail.
    assert "target == 2" in selector
    assert "target == 3" in selector


def test_nic_view_looks_straight_down_the_port_bores_from_far_off():
    """The NIC SFP bores open straight up and are 45.8 mm deep behind a 16x12 mm
    aperture, so a port only shows its black depth to a ray within 7.5 deg of the
    board normal.  The view settings must therefore look straight down (no
    cross-rail tilt, tight obliquity) from the farthest reachable standoff -- a
    tilted view resolves *zero* of the ten ports.  Executed, not just grepped, so
    the policy cannot silently drift back onto the SC bore-tilt recipe.
    """
    helper = copy.deepcopy(_method("_survey_view_settings"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {"math": math}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    settings = namespace["_survey_view_settings"]

    nic = settings(2)
    assert nic["cross_rail_sign"] == 0.0, "a cross-rail tilt hides the bores"
    assert nic["max_obliquity_rad"] <= math.radians(3.0)
    assert nic["prefer_far_standoff"] is True  # outermost port needs the distance
    assert nic["require_all_cameras_frame"] is True
    # All-camera framing needs a smaller keep-out margin than the 40 px default;
    # the gripper mask already dilates the silhouette by 32 px underneath this.
    assert 20.0 <= nic["min_required_clearance_px"] <= 30.0
    # The wrist-camera self-collision gate (arm_ik.UR5eArm.flange_T_probes) can
    # rule out every candidate within the default 45 deg / 7-roll sample at some
    # board yaws; NIC widens both so a clear-of-forearm pose can still be found.
    assert nic["max_angular_motion_rad"] >= math.radians(89.0)
    assert len(nic["yaws_rad"]) >= 20

    # SC gets its own straight-down recipe; pinned by
    # test_sc_view_reads_five_adapters_from_inside_the_ivm_standoff_band.

    for sfp_target in (0, 1, 99):  # SFP modules keep the close all-camera view
        sfp = settings(sfp_target)
        assert sfp["require_all_cameras_frame"] is True
        assert sfp["prefer_far_standoff"] is False
        assert sfp["cross_rail_sign"] == 0.0

    method_source = ast.get_source_segment(
        SOURCE_PATH.read_text(encoding="utf-8"),
        _method("_run_sfp_geometric_stage2"),
    )
    assert "**self._survey_view_settings(survey_target)" in method_source
    assert "cross_rail_tilt_band_rad=None" in method_source


def test_no_sector_asks_for_a_cross_rail_tilt_any_more():
    """Every recessed port on this board opens along the board normal.

    NIC's SFP cages and SC's adapters both have bore axes within 1 deg of the
    board normal (measured from the workcell model), so tilting the camera
    across the rail reads them edge-on and the near wall occludes the bore.  The
    committed-band/flat-fallback ladder that used to serve SC is therefore gone,
    and the search is always called with no tilt band.
    """
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_bore_view_tilt_bands" not in source
    method_source = ast.get_source_segment(
        source, _method("_run_sfp_geometric_stage2")
    )
    assert "cross_rail_tilt_band_rad=None" in method_source
    # ...and no sector re-introduces a side commitment through cross_rail_sign.
    helper = copy.deepcopy(_method("_survey_view_settings"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {"math": math}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    settings = namespace["_survey_view_settings"]
    for target in (0, 1, 2, 3, 99):
        assert settings(target)["cross_rail_sign"] == 0.0


def test_sc_view_gets_close_enough_to_resolve_a_7mm_bore():
    """SC is pixel-limited, not geometry-limited, and needs its own band.

    The five adapters open straight up behind a 15.64 mm bore with a
    7.6 x 22.4 mm aperture -- a 13.7 deg cone, satisfied from 0.27 m, so nothing
    forces the standoff outward.  What limits it is resolution: at the 0.6 m
    that suits the NIC cards the bore spans only ~15 px and the first field run
    resolved 2 of 5 ports.  Two things follow, and both are load-bearing:

    All three cameras must frame the sector and stay gripper-clear.  Relaxing
    that to the reference camera alone, to chase a closer view, is a regression
    that has already been made once: the tool then sat on top of the ports in
    both side cameras (-13 to -32 px of gripper clearance) while the centre
    camera read +58 px, and the resulting 0.45 m pose drove the TCP to base
    z 0.24 m, reachable only through a contorted arm configuration.

    Resolution at the far end of this band is genuinely marginal -- ~17 px on
    the bore, and IVM resolved 2 of 5 ports on the field run -- but that has to
    be fixed without un-guarding the side cameras.
    """
    helper = copy.deepcopy(_method("_survey_view_settings"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {"math": math}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    sc = namespace["_survey_view_settings"](3)

    # Side cameras stay guarded: relaxing this put the tool over the ports.
    assert sc["require_all_cameras_frame"] is True
    assert sc["prefer_far_standoff"] is False  # cone is met; take the pixels
    assert sc["max_obliquity_rad"] <= math.radians(13.7)  # inside the bore cone
    standoffs = sc["standoffs_m"]
    assert min(standoffs) >= 0.50, "closer drives the TCP down onto the board"

    # The tool must clear the tallest thing standing on the board (the NIC card
    # tips), not merely the board plane -- the plane guard sits below them.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "BOARD_TALLEST_COMPONENT_Z" in source
    method_source = ast.get_source_segment(
        source, _method("_run_sfp_geometric_stage2")
    )
    assert "BOARD_TALLEST_COMPONENT_Z + TOOL_COMPONENT_CLEARANCE_M" in method_source


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
