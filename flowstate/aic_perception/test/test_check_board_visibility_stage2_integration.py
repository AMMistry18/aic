"""Source contracts for in-place geometric SFP Stage-2 integration."""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path

import pytest


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


def _stage2_source() -> str:
    """Source of the Stage-2 body *and* the per-tier search helper.

    ``_run_sfp_geometric_stage2`` used to hold the ``search_survey_pose`` call
    inline.  The relaxation ladder moved that call into
    ``_search_survey_pose_tier``, so the search-argument contracts below have to
    read both halves or they silently stop checking anything.
    """
    source, _ = _source_and_class()
    return "\n".join(
        ast.get_source_segment(source, _method(name))
        for name in ("_run_sfp_geometric_stage2", "_search_survey_pose_tier")
    )


def test_stage1_handoff_uses_the_existing_geometric_stage2_closure():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "self._run_sfp_geometric_stage2(" in method_source
    # The nested declaration plus its single call site: with Stage 1 removed
    # there is exactly one observation and one opportunity to hand off.
    assert method_source.count("handoff_to_stage2(snapshot, reports)") == 2
    assert method_source.count("_stage2_has_complete_landmark(") == 1
    # Only Stage 2 may declare done; Stage 1 no longer exists to try.
    assert "result.done = True" not in method_source


def test_stage1_has_no_wall_clock_deadline_and_hands_off_on_exposed_insignia():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    # Stage 1 is bounded by a move budget and per-move timeouts, not a
    # wall clock. An exposed insignia hands off to Stage 2 immediately.
    assert "deadline = started_at" not in method_source
    assert "stage2_reserve_sec" not in method_source
    assert "_stage2_has_complete_landmark(" in method_source
    assert "handoff_to_stage2(snapshot, reports)" in method_source
    assert "_stage2_has_complete_landmark(" in method_source
    # No search loop of any kind remains.
    assert "_seek_step(" not in method_source


def test_stage1_uses_wide_purple_only_as_a_cue_not_completion_authority():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "analyze_purple" in method_source
    assert "_move_to_acquire_complete_logo(" not in method_source
    assert "purple_report.full" in method_source  # diagnostics only
    assert "_stage2_has_complete_landmark(" in method_source


def test_missing_insignia_fails_without_calling_stage2():
    source, _ = _source_and_class()
    execute_inner = _method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "insignia_not_exposed" in method_source
    assert "Stage 1 acquisition has been removed" in method_source
    assert "result.success = False" in method_source
    assert "result.target_valid = False" in method_source


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


def test_sfp_stage2_is_perception_only_and_publishes_a_cartesian_target():
    method = _method("_run_sfp_geometric_stage2")
    tier = _method("_search_survey_pose_tier")
    estimate = _named_calls(method, "estimate_board_pose_from_insignia")
    # The relaxation ladder calls the search once per tier through the helper,
    # so the single ``search_survey_pose`` call site lives there now.
    search = _named_calls(tier, "search_survey_pose")
    motions = _calls(method, "move_smooth")
    grabs = _calls(method, "grab")
    method_source = ast.get_source_segment(
        SOURCE_PATH.read_text(encoding="utf-8"), method
    )

    # One insignia PnP and one survey search call site; no motion, no re-grab.
    assert len(estimate) == 1
    assert len(search) == 1
    assert _named_calls(method, "search_survey_pose") == []
    assert len(motions) == 0
    assert len(grabs) == 0
    # Publishes scalar Cartesian fields instead of executing/confirming a move.
    assert "result.target.x" in method_source
    assert "result.target.qw" in method_source
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
    assert estimate[0] < _calls(method, "_search_survey_pose_tier")[0]


def test_sfp_target_preserves_the_deployed_scalar_cartesian_interface():
    source, _ = _source_and_class()
    method = _method("_run_sfp_geometric_stage2")
    method_source = ast.get_source_segment(source, method)

    assert "estimate_board_pose_from_insignia(" in method_source
    # Flowstate exposes these seven scalar fields to the existing Python pose
    # packer.  The extra native pose remains compatible but is not required.
    assert "result.target.x" in method_source
    assert "result.target.y" in method_source
    assert "result.target.z" in method_source
    assert "result.target.qx" in method_source
    assert "result.target.qy" in method_source
    assert "result.target.qz" in method_source
    assert "result.target.qw" in method_source
    assert "result.survey_pose.position.x" in method_source
    assert "result.survey_pose.orientation.w" in method_source
    assert "result.target_frame = self.config.base_frame" in method_source
    assert "result.survey_joint_target" not in method_source
    assert "result.survey_joint_limits" not in method_source


def test_result_proto_declares_intrinsic_pose_survey_pose_output():
    proto = PROTO_PATH.read_text(encoding="utf-8")
    assert 'import "intrinsic/math/proto/pose.proto";' in proto
    assert "intrinsic_proto.Pose survey_pose" in proto
    assert "survey_joint_target" not in proto
    assert "survey_joint_limits" not in proto


def test_stage2_searches_inside_the_ur5e_reach_for_the_sfp_sector():
    method_source = _stage2_source()

    # Reachability is decided by the real UR5e IK gate (calibrated from the live
    # joint state), not the base-origin sphere -- which wrongly rejected the
    # reachable far, bore-facing poses and admitted unsolvable ones.  The sphere
    # survives only as a loose fallback at the full envelope, and the survey
    # frames exactly one sector, chosen from the target mode.
    assert "joint_motion=joint_motion_fn" in method_source
    assert "UR5eArm.autocalibrate(" in method_source
    assert "_arm.solve_ranked(" in method_source
    assert "joint_limits=" not in method_source
    assert "joints - seed" in method_source
    assert "select_clear_ik_solution(target)" in method_source
    assert "185.0 if int(survey_target) == 3 else 225.0" in method_source
    # The ladder passes the cap per tier; the strict tier is the sector cap.
    assert "joint_motion_limit_rad, total_joint_motion_limit_rad" in method_source
    assert "max_joint_motion_rad=max_joint_motion_rad" in method_source
    assert "max_reach_m=0.85" in method_source
    assert "min_height_m=0.02" in method_source
    assert "self._coverage_targets_for_target(survey_target)" in method_source


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
    method_source = _stage2_source()

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
    selector = ast.get_source_segment(
        source, _method("_coverage_targets_for_target")
    )

    assert "nic_sector_corners()" in selector
    assert "sc_sector_corners()" in selector
    # NIC=2, SC=3, everything else (0/1) is the staged-SFP strip.
    assert "target == 2" in selector
    assert "target == 3" in selector

    # The staged-SFP target must never be the one-rail sector: its centre sits
    # 112.5 mm off the module strip, which is what cropped a physically present
    # module on hardware (4 of 5 modules returned).
    assert "sfp_sector_corners()" not in selector  # docstring may still cite it
    assert "sfp_module_strip_corners()" in selector


def test_staged_sfp_coverage_straddles_both_rails():
    """Execute the selector: the SFP box must straddle board Y=0.

    The old box covered the +Y rail alone, so the survey aimed 112.5 mm off the
    middle of the strip and banked all its framing slack on one side.
    """
    helper = copy.deepcopy(_method("_coverage_targets_for_target"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(body=[helper], type_ignores=[])
    )
    namespace: dict[str, object] = {}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    select = namespace["_coverage_targets_for_target"]

    for sfp_target in (0, 1):
        targets = select(sfp_target)
        assert len(targets) == 1
        ys = targets[0][:, 1]
        assert abs(float(ys.mean())) < 1e-9
        assert abs(float(ys.max()) + float(ys.min())) < 1e-9
        # Same extent as the box it replaces -- placement is the fix, not size.
        assert float(ys.max()) - float(ys.min()) == pytest.approx(0.225)

    # NIC and SC remain single, already-validated sectors.
    assert len(select(2)) == 1
    assert len(select(3)) == 1


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
    assert nic["cross_rail_tilt_band_rad"] is None
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

    # SC gets its own small wide-axis obliquity; pinned by
    # test_sc_view_reads_five_adapters_from_inside_the_ivm_standoff_band.

    for sfp_target in (0, 1, 99):  # SFP modules keep the close all-camera view
        sfp = settings(sfp_target)
        assert sfp["require_all_cameras_frame"] is True
        assert sfp["prefer_far_standoff"] is False
        assert sfp["cross_rail_sign"] == 0.0
        assert sfp["cross_rail_tilt_band_rad"] is None

    method_source = _stage2_source()
    # Threaded through the relaxation ladder, which may override only the
    # clearance margin -- never the sector view geometry.
    assert "self._survey_view_settings(survey_target)" in method_source
    assert "**view_settings" in method_source
    assert "cross_rail_tilt_band_rad=None" not in method_source


def test_only_sc_uses_the_explicit_long_face_approach_axis():
    """SC must approach the long face, not infer an axis from the rail box.

    The mouth's 22.4 mm long face runs along board Y, so the view displacement
    must be along its board-X normal. Because that is the narrow bore direction,
    the angle remains 10-13 degrees. All three cameras stay fully framed while
    at least two per mouth must retain an open, strongly displaced back plane.
    """
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_bore_view_tilt_bands" not in source
    helper = copy.deepcopy(_method("_survey_view_settings"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {"math": math}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    settings = namespace["_survey_view_settings"]
    for target in (0, 1, 2, 3, 99):
        assert settings(target)["cross_rail_sign"] == 0.0
    for target in (0, 1, 2, 99):
        assert settings(target)["cross_rail_tilt_band_rad"] is None
    sc = settings(3)
    band = tuple(math.degrees(value) for value in sc["cross_rail_tilt_band_rad"])
    assert band == pytest.approx((16.0, 20.0))
    assert sc["directional_tilt_axis_board"] == (1.0, 0.0, 0.0)
    assert sc["max_along_rail_tilt_rad"] <= math.radians(2.0)
    assert sc["max_angular_motion_rad"] == pytest.approx(math.pi)


def test_sc_view_gets_close_enough_to_resolve_a_7mm_bore():
    """SC is pixel-limited, not geometry-limited, and needs its own band.

    The five adapters open straight up behind a 15.64 mm bore with a
    7.6 x 22.4 mm aperture. A 10-13 degree displacement normal to the long face
    gives the checked side view while the bore-margin gate proves every camera
    ray still passes the narrow dimension. Tilt along the long face stays below
    2 degrees so adjacent ports remain separated.
    Resolution is still limiting at roughly 15-17 px, so nearest-first remains
    load-bearing:

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
    low, high = sc["cross_rail_tilt_band_rad"]
    # Past the 13.66 deg back-centre cone on purpose: the gate below accepts a
    # displaced dark strip rather than a centred back plane, and that is where
    # the depth cue lives (3.3 px at 10-13 deg vs 8.0 px at 16-20 deg).
    assert math.radians(15.0) <= low <= high <= math.radians(22.0)
    assert sc["directional_tilt_axis_board"] == (1.0, 0.0, 0.0)
    assert sc["max_along_rail_tilt_rad"] <= math.radians(2.0)
    # A ladder, not a pin: closest-feasible wins and closer is also deeper.
    standoffs = sc["standoffs_m"]
    assert standoffs == tuple(sorted(standoffs))
    assert min(standoffs) >= 0.55, "0.45 put the tool over the ports"
    assert max(standoffs) <= 0.62

    # The tool must clear the tallest thing standing on the board (the NIC card
    # tips), not merely the board plane -- the plane guard sits below them.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "BOARD_TALLEST_COMPONENT_Z" in source
    method_source = _stage2_source()
    assert "BOARD_TALLEST_COMPONENT_Z + TOOL_COMPONENT_CLEARANCE_M" in method_source
    # All-camera framing alone does not guarantee that the separated side
    # camera origins can see through the SC mouth.  The diagonal-board hardware
    # failure selected an IK-clear pose with both side-camera rays outside the
    # narrow 7.6 mm aperture cone; SC must gate the physical bore margin before
    # accepting the first reachable pose.
    assert "rectangular_bore_visibility_margin" in method_source
    assert "sc_bore_sample_points()" in method_source
    # The narrow-axis tolerance is the *acceptance criterion*, and at the full
    # mouth width it accepts a displaced dark strip instead of demanding the
    # back-plane centre -- which is what lifts the long-face angle ceiling.
    assert "half_width_x_m=SC_BORE_X_TOLERANCE_M" in method_source
    assert "SC_BORE_X_TOLERANCE_M = 0.0076" in source
    assert "half_width_y_m=SC_BORE_HALF_WIDTH_Y_M" in method_source
    assert "depth_m=SC_BORE_DEPTH_M" in method_source
    assert "bore_margin < 0.0" in method_source
    assert method_source.count("required_camera_count=2") >= 2
    assert "rectangular_bore_depth_cue_px" in method_source
    assert "min_view_quality = 3.0" in method_source
    assert "view_quality_motion_tolerance = 0.1" in method_source
    assert "view_quality=view_quality_fn" in method_source
    assert "SC_MOVE_ROBOT_JOINT_LIMITS" not in method_source
    assert method_source.count("relative joint-motion") >= 2
    assert "relative_origin=live_joints" in method_source
    assert "joint_motion_preference=joint_motion_preference_fn" in method_source
    assert "joint_preference_motion_tolerance_rad=math.radians(30.0)" in method_source


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


def test_every_sector_searches_the_full_roll_family_from_any_start_pose():
    """A reorientation cap measured from the live TCP must not pick the poses.

    ``max_angular_motion_rad`` is measured against the *current* TCP, so it does
    not merely bound how far the arm turns -- it decides which candidates are
    ever scored.  Measured at the real hardware board distance (0.558 m
    horizontal) over 8 board yaws, the shipped 90 deg cap made availability a
    function of the Stage-1 exit wrist roll:

        live start pose      cap=45   cap=90   cap=180
        field 01:29            1/8      5/8      7/8
        sweep home             3/8      6/8      7/8
        home + J6 +90 deg      0/8      0/8      7/8
        chained start          5/8      7/8      7/8

    From the rolled-wrist start the 90 deg cap admitted 1036 framed candidates
    of which *zero* had any IK solution -- the field "BINDING GATE =
    reachability" refusal.  Bounding real motion is the live-seeded joint gate's
    job; this one only narrowed the search.  The 24-roll family then takes every
    one of those start poses from 7/8 to 8/8.
    """
    helper = copy.deepcopy(_method("_survey_view_settings"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {"math": math}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    settings = namespace["_survey_view_settings"]

    for target in (0, 1, 2, 3, 99):
        sector = settings(target)
        assert sector["max_angular_motion_rad"] == pytest.approx(math.pi), (
            f"target {target} reintroduced a reorientation cap that selects "
            "the candidate set instead of bounding motion"
        )
        assert len(sector["yaws_rad"]) >= 20, (
            f"target {target} lost the fine roll family"
        )


def test_one_insignia_view_is_enough_but_agreeing_views_are_fused():
    """Requiring two complete views was tried on hardware and reverted.

    The motivation was sound -- a single-view PnP of one small quad is a weak
    *range* measurement, and two invocations 7 s apart at the same arm pose
    disagreed enough to flip the near-standoff family across the 25 px
    clearance floor.  But two complete views refuses far too many real start
    poses: the field run rejected five consecutive invocations with "0 have
    one" at poses where the board was plainly in view.  Stage 1 acquisition no
    longer exists, so each of those is a dead stop rather than something the
    skill recovers from.

    What survives is the half that costs nothing: when two or more cameras do
    accept an estimate they must agree, and their origins are averaged.
    """
    source, _ = _source_and_class()
    assert "REQUIRED_INSIGNIA_CAMERAS = 1" in source

    gate = ast.get_source_segment(source, _method("_stage2_has_complete_landmark"))
    assert ">= REQUIRED_INSIGNIA_CAMERAS" in gate

    stage2 = ast.get_source_segment(source, _method("_run_sfp_geometric_stage2"))
    # A lone accepted estimate must still pass the agreement check.
    assert "len(pose_estimates) > 1 and len(consistent) < 2" in stage2
    # Opportunistic fusion: average the range whenever more than one view agrees.
    assert "cluster_translations.mean(axis=0)" in stage2
    assert "base_T_board=Transform(" in stage2
    # Rotation stays with the preferred view -- averaging orientations over a
    # near-square landmark can interpolate between mirror hypotheses.
    assert "board_pose.base_T_board.rotation, mean_translation" in stage2


def test_unreachable_and_camera_keepout_are_reported_as_different_gates():
    """``solve_ranked`` filters the keep-out before returning, so an empty list
    has two opposite meanings and reporting both as "no analytic IK solution at
    all" sent debugging after the arm's workspace when the pose was reachable.
    Measured at the hardware board distance, 231 of 926 "no IK" verdicts were
    keep-out rejections."""
    source, _ = _source_and_class()
    stage2 = ast.get_source_segment(source, _method("_run_sfp_geometric_stage2"))

    assert '"keepout": 0,' in stage2
    # Re-solved without the keep-out, and only on the failure path.
    assert "if _arm.solve_all(pose):" in stage2
    assert 'record["gate"] = "camera_keepout"' in stage2
    assert 'record["gate"] = "unreachable"' in stage2
    assert "BINDING GATE = wrist-camera keep-out" in stage2
    assert "camera_keepout=%d" in stage2


def test_nic_refuses_short_standoffs_that_frame_the_ports_without_reading_them():
    """NIC framing is not sufficiency -- the cage cone is.

    Each port is a 16 x 12 mm aperture at the top of a 45.8 mm recess, so it
    only shows the black depth the IVM keys on to a ray within
    ``atan(6/45.8) = 7.46 deg``.  Over 144 offline placements the search
    published 21 poses that framed all ten ports in all three cameras while the
    outermost ports lay outside that cone -- roughly a 6-of-10 view that reports
    success.  All of them had been pushed below 0.66 m because the arm could not
    reach farther at that placement.

    Poses that do resolve all ten sit in a tight band (0.66-0.76 m, worst cone
    7.27 deg), and it is the same band the superseded 90 deg cap produced
    whenever it worked.  Flooring the ladder there leaves the passing set
    unchanged at 105/144 and turns the 21 into honest refusals.
    """
    helper = copy.deepcopy(_method("_survey_view_settings"))
    helper.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, object] = {"math": math}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    nic = namespace["_survey_view_settings"](2)

    assert min(nic["standoffs_m"]) >= 0.66 - 1e-9, (
        "a rung below 0.66 m puts the outer NIC ports outside the bore cone"
    )
    # prefer_far_standoff must still have somewhere to climb.
    assert max(nic["standoffs_m"]) >= 1.0
    assert nic["prefer_far_standoff"] is True


def test_angled_view_is_the_last_resort_and_never_applies_to_sc():
    """NIC may trade view angle for existence -- but only after everything else.

    Section 9.3 records that tilting NIC across the rail resolved 0 of 10
    ports, because the cages show their black interior only near the board
    normal.  That experiment traded a *good* view for a tilted one.  This tier
    fires only when no pose within the normal-view cap is reachable at all, so
    the alternative is not a better view, it is no view.  Measured at the
    2026-07-28 04:34 refusal, where the ports sat 0.715 m out and every
    straight-down candidate needed >=0.914 m of reach:

        obliquity cap    found   ports in the 7.46 deg cone
         2 deg (ship)     no      -
         5 deg            no      -
         8 deg           YES     5/10

    SC is excluded: its directional tilt band *is* the depth measurement, so
    there is no degraded-but-useful angle to fall back to.
    """
    stage2 = _stage2_source()

    # Ordering: the angled tiers must come after every geometry-preserving one.
    strict = stage2.index('("strict"')
    clearance = stage2.index('("reduced clearance margin"')
    angled = stage2.index('("angled view (8deg off normal)"')
    assert strict < clearance < angled

    # SC opts out entirely.
    assert 'if view_settings.get("cross_rail_tilt_band_rad") is not None:' in stage2
    # Only ever widens, so SFP's existing 20 deg is untouched by an 8 deg tier.
    assert 'view_settings["max_obliquity_rad"] = max(' in stage2
    # A degraded view must announce itself.
    assert "ANGLED view" in stage2


def test_a_reachable_normal_view_is_never_affected_by_the_angled_tiers():
    """The tiers that existed before the angled fallback must be untouched.

    The deployed build searches four tiers and returns on the first that finds
    a pose.  Appending angled-view tiers must not change any outcome that those
    four already produced -- if tier 0 succeeds, nothing below it ever runs.
    This pins both halves of that: the original four tiers carry no obliquity
    relaxation, and the loop still breaks on the first candidate.
    """
    stage2 = _stage2_source()

    # The four pre-existing tiers, in order, each with obliquity relaxation
    # explicitly absent (the trailing ``None``).
    for tier in (
        '("strict", joint_motion_limit_rad, total_joint_motion_limit_rad,\n'
        '             False, None, None)',
        '("joint-travel caps lifted", math.radians(360.0), math.inf,\n'
        '             False, None, None)',
        '("any arm-clear IK branch", math.radians(360.0), math.inf,\n'
        '             True, None, None)',
        '("reduced clearance margin", math.radians(360.0), math.inf,\n'
        '             True, 12.0, None)',
    ):
        assert tier in stage2, tier

    # Tier 0 must still carry the *sector's own* caps, not a relaxed constant.
    assert "joint_motion_limit_rad, total_joint_motion_limit_rad" in stage2
    # And the search must stop at the first tier that yields a pose.
    assert "if candidate is not None:" in stage2
    body = stage2[stage2.index("if candidate is not None:"):]
    assert "break" in body[: body.index("if joint_motion_fn is None:")]
