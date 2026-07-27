"""Source-level contracts for deterministic Stage 1 and Flowstate cleanup."""

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


def method(name: str) -> tuple[str, ast.FunctionDef]:
    source, skill = skill_source()
    return source, next(
        node
        for node in skill.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def call_lines(node: ast.AST, attribute: str) -> list[int]:
    return sorted(
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attribute
    )


def named_call_lines(node: ast.AST, name: str) -> list[int]:
    return sorted(
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
    )


def test_stage2_landmark_is_checked_before_and_after_only_joint_motion():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)
    landmark_lines = call_lines(
        execute_inner, "_stage2_has_complete_landmark"
    )
    joint_lines = call_lines(execute_inner, "move_joint_target")

    assert len(landmark_lines) == 2
    assert len(joint_lines) == 1
    assert landmark_lines[0] < joint_lines[0] < landmark_lines[1]
    assert "move_smooth" not in method_source
    assert "next_action" not in method_source


def test_path_is_validated_before_any_joint_command():
    _, execute_inner = method("_execute_inner")
    path_lines = named_call_lines(
        execute_inner, "validate_observation_path"
    )
    waypoint_lines = named_call_lines(
        execute_inner, "interpolated_joint_waypoints"
    )
    motion_lines = call_lines(execute_inner, "move_joint_target")

    assert len(path_lines) == len(waypoint_lines) == len(motion_lines) == 1
    assert path_lines[0] < waypoint_lines[0] < motion_lines[0]
    assert call_lines(execute_inner, "_arm_clear_of_own_cameras")


def test_motion_requires_fresh_force_and_coherent_six_joint_state():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "wait_for_force_xyz" in method_source
    assert "current_joints" in method_source
    assert "baseline_force_xyz" in method_source
    assert "no fresh wrist-force sample" in method_source
    assert "no coherent fresh six-joint state" in method_source
    assert call_lines(execute_inner, "wait_for_force_xyz")[0] < call_lines(
        execute_inner, "move_joint_target"
    )[0]


def test_stage1_never_publishes_an_internal_motion_as_final_target():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "result.target_valid = False" in method_source
    assert "result.target.x" not in method_source
    assert "result.target.qw" not in method_source
    assert "result.target_frame = \"\"" in method_source


def test_deterministic_exhaustion_is_normal_not_done_at_observation_pose():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert 'result.last_action = "deterministic_observation_exhausted"' in method_source
    assert "deterministic observation pose reached safely" in method_source
    assert call_lines(execute_inner, "_stage2_not_done")
    assert "while True" not in method_source


def test_legacy_descriptor_inputs_are_still_validated_for_compatibility():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    for name in (
        "max_travel_m",
        "max_displacement_m",
        "max_angular_displacement_rad",
        "max_angular_travel_rad",
        "search_timeout_sec",
        "stable_frames",
    ):
        assert name in method_source
    assert call_lines(execute_inner, "_validate_parameters")


def test_old_phase_machine_is_retained_but_not_runtime_selected():
    source, skill = skill_source()
    names = {
        node.name
        for node in skill.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_execute_inner" in names
    assert "_execute_inner_legacy" in names

    execute = next(
        node
        for node in skill.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    assert len(call_lines(execute, "_execute_inner")) == 1
    assert not call_lines(execute, "_execute_inner_legacy")
    assert "AdaptiveViewpointPlanner" not in ast.get_source_segment(
        source, next(
            node
            for node in skill.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_execute_inner"
        )
    )


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
