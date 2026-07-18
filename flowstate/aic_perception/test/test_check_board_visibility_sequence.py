"""Source-level contracts for the ROS/Flowstate wrapper sequencing.

The runtime wrapper imports Intrinsic and ROS modules that are unavailable in
the lightweight unit-test environment.  These AST checks protect the ordering
regression without importing or mocking that external runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "check_board_visibility_skill.py"


def skill_source() -> tuple[str, ast.ClassDef]:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    skill = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CheckBoardVisibilitySkill"
    )
    return source, skill


def execute_inner_source() -> tuple[str, ast.FunctionDef]:
    source, skill = skill_source()
    execute_inner = next(
        node
        for node in skill.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_inner"
    )
    return source, execute_inner


def call_lines(node: ast.AST, attribute: str) -> list[int]:
    return [
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attribute
    ]


def named_call_lines(node: ast.AST, name: str) -> list[int]:
    return [
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
    ]


def test_planner_runs_before_any_cartesian_motion():
    _, execute_inner = execute_inner_source()

    planner_lines = call_lines(execute_inner, "next_action")
    cartesian_lines = call_lines(execute_inner, "move_smooth")

    assert len(planner_lines) == 1
    assert cartesian_lines
    assert planner_lines[0] < min(cartesian_lines)


def test_leveling_is_phase_gated_and_completion_has_no_extra_streak():
    source, _ = execute_inner_source()

    assert 'if planner.phase == "j2_4_level":' in source
    assert "planner.mark_level_complete()" in source
    assert "complete_streak" not in source
    assert "ivm_survey_ready" not in source


def test_only_center_camera_full_evidence_reaches_planner():
    source, execute_inner = execute_inner_source()

    full_keywords = [
        keyword.value
        for item in ast.walk(execute_inner)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "replace"
        for keyword in item.keywords
        if keyword.arg == "full"
    ]
    assert len(full_keywords) == 1
    assert isinstance(full_keywords[0], ast.Call)
    assert isinstance(full_keywords[0].func, ast.Name)
    assert full_keywords[0].func.id == "bool"
    assert len(full_keywords[0].args) == 1
    assert isinstance(full_keywords[0].args[0], ast.BoolOp)
    assert isinstance(full_keywords[0].args[0].op, ast.And)
    full_gate = ast.unparse(full_keywords[0])
    assert "name == 'center_camera'" in full_gate
    assert "item.full" in full_gate
    assert "center_top_down" in full_gate

    axes_lines = call_lines(execute_inner, "_camera_axes_in_base")
    planner_lines = call_lines(execute_inner, "next_action")
    assert min(axes_lines) < planner_lines[0]
    assert call_lines(execute_inner, "request_relevel")
    assert "the first qualifying frame exits in this same iteration" in source


def test_component_coverage_is_true_only_on_done():
    source, _ = execute_inner_source()

    assert "component_coverage_ready = bool" not in source
    assert source.count("result.component_coverage_ready = True") == 1
    assert "if action.kind == ActionKind.DONE:" in source
    assert source.index("if action.kind == ActionKind.DONE:") < source.index(
        "result.component_coverage_ready = True"
    )


def test_leveling_checks_local_j1_j6_drift_before_acknowledgement():
    source, _ = execute_inner_source()

    for name in (
        "pre_level_joint1",
        "pre_level_joint6",
        "post_level_joint1",
        "post_level_joint6",
        "level_joint1_drift",
        "level_joint6_drift",
        "level_joint_drift_tolerance_rad",
        "level_anchor_joint1",
        "level_anchor_joint6",
        "min_level_progress_rad",
    ):
        assert name in source
    assert "post_level_joint1 - level_anchor_joint1" in source
    assert "post_level_joint6 - level_anchor_joint6" in source
    assert "returning to visual correction" in source
    assert "planner.request_recenter()" in source
    assert source.index("level_joint1_drift =") < source.index(
        "planner.mark_level_complete()"
    )


def test_leveling_uses_fresh_vertical_image_feedback_and_can_reverse():
    source, _ = execute_inner_source()

    for contract in (
        "level_vertical_polarity",
        "pending_level_vertical_sample",
        "center_report.center_error[1]",
        "level_image_down",
        "level_center_delta",
        "reversing image-y polarity",
    ):
        assert contract in source
    assert source.index("pending_level_vertical_sample is not None") < source.index(
        'if planner.phase == "j2_4_level":'
    )


def test_force_is_acquired_only_after_planning_and_before_motion():
    source, execute_inner = execute_inner_source()

    planner_line = call_lines(execute_inner, "next_action")[0]
    force_lines = named_call_lines(execute_inner, "require_motion_force")
    assert len(force_lines) == 2
    assert planner_line < min(force_lines)
    assert source.count("require_motion_force(snapshot)") == 2


def test_legacy_motion_envelopes_do_not_terminate_viewpoint_search():
    source, _ = execute_inner_source()

    for name in (
        "max_travel_m",
        "max_displacement_m",
        "max_angular_displacement_rad",
        "max_angular_travel_rad",
    ):
        assert f"{name} = math.inf" in source
    assert "motion_envelopes=controller_native" in source


def test_faster_defaults_and_precise_j6_are_declared_by_wrapper():
    source, _ = execute_inner_source()

    assert "params.max_speed_mps or 0.05" in source
    assert "params.max_angular_speed_rps or 0.30" in source
    assert "params.move_timeout_seconds or 6.0" in source
    assert "params.search_timeout_seconds or 90.0" in source
    assert "j6_tolerance=%.1fdeg" in source


def test_execute_finalizer_prepares_handoff_before_unlocking():
    _, skill = skill_source()
    execute = next(
        node
        for node in skill.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    outer_try = next(node for node in execute.body if isinstance(node, ast.Try))
    finalizer = ast.Module(body=outer_try.finalbody, type_ignores=[])
    handoff_lines = call_lines(finalizer, "prepare_controller_handoff")
    release_lines = call_lines(finalizer, "release")

    assert handoff_lines and release_lines
    assert handoff_lines[0] < release_lines[0]
