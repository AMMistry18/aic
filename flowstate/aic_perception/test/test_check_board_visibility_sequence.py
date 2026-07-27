"""Source-level contracts for the Stage-1 image-plane seek and cleanup."""

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








def test_stage1_never_publishes_an_internal_motion_as_final_target():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "result.target_valid = False" in method_source
    assert "result.target.x" not in method_source
    assert "result.target.qw" not in method_source
    assert "result.target_frame = \"\"" in method_source


def test_stage1_is_gone_and_the_skill_never_commands_motion():
    """Stage 1 was removed entirely after three designs failed on hardware.

    The skill is now pure perception: the start pose either already exposes the
    insignia or the call fails.  Nothing here may command the arm.
    """
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    for banned in (
        "move_smooth",
        "move_joint_target",
        "move_joint1_yaw",
        "_seek_step",
        "select_work_target",
        "board_seek",
        "viewpoint_search",
        "stage1_acquisition",
        "MAX_CENTER_MOVES",
    ):
        assert banned not in method_source, banned
    assert "while " not in method_source
    assert "for step" not in method_source
    # One observation, one landmark check, one handoff call.
    assert len(call_lines(execute_inner, "_stage2_has_complete_landmark")) == 1
    assert len(named_call_lines(execute_inner, "handoff_to_stage2")) == 1
    assert len(named_call_lines(execute_inner, "observe")) == 1


def test_missing_insignia_raises_a_real_skill_error():
    """With Stage 1 removed there is nothing to retry, so it fails loudly."""
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert 'result.last_action = "insignia_not_exposed"' in method_source
    assert "result.success = False" in method_source
    assert "Stage 1 acquisition has been removed" in method_source
    assert "raise InsigniaNotExposedError(" in method_source
    # The failure says which cameras saw a partial insignia, so the operator
    # can tell "nothing there" from "nearly framed".
    assert "purple_reports" in method_source


def test_the_skill_error_is_raised_only_after_controller_cleanup():
    """Ordering is the whole point, and it is load-bearing.

    Raising before the handoff is published leaves the AIC controller bridge's
    ICON session holding `arm`, and the next Move Robot fails with
    "Part: 'arm' is already in use."  So `_execute_inner` raises a marker,
    `execute` holds it across the `finally`, and re-raises afterwards.
    """
    source, execute = method("execute")
    method_source = ast.get_source_segment(source, execute)

    assert "except InsigniaNotExposedError as error:" in method_source
    assert "pending_error = error" in method_source

    handoff = method_source.index("prepare_controller_handoff")
    reraise = method_source.index("raise skill_interface.SkillError(")
    assert handoff < reraise, "cleanup must publish before the skill error"

    # Cancellation still propagates untouched, and ordinary sensor/Stage-2
    # outcomes still come back as results rather than errors.
    assert "except skill_interface.SkillCancelledError:" in method_source
    assert "return result" in method_source


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


def test_the_scrapped_stage1_designs_are_gone_not_dormant():
    """Both superseded Stage 1s are deleted, not left behind as dead paths.

    The phase machine (`viewpoint_search.AdaptiveViewpointPlanner`) steered on
    a board orientation that is degenerate in clipped views, and the
    deterministic joint plan could not execute: the deployed controller drops
    joint target mode mid-segment.  Keeping either around invites a future
    session to re-select it.
    """
    source, skill = skill_source()
    names = {
        node.name
        for node in skill.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_execute_inner" in names
    assert "_execute_inner_legacy" not in names
    assert "AdaptiveViewpointPlanner" not in source
    assert "viewpoint_search" not in source
    assert "stage1_acquisition" not in source
    assert "move_joint_target" not in source

    execute = next(
        node
        for node in skill.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    assert len(call_lines(execute, "_execute_inner")) == 1


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


def test_every_runtime_import_symbol_actually_exists():
    """Resolve every ``from aic_perception...`` symbol the skill imports.

    The skill imports lazily inside methods, so a missing name is invisible at
    module import and only raises when that code path first runs -- on the
    robot.  That is exactly how a trimmed ``purple_insignia`` (no
    ``any_purple_seen`` / ``pick_purple_camera``) reached hardware and crashed
    Stage 1 while every source-level test passed.

    The generated protobuf module is excluded: it does not exist until build.
    """
    import importlib

    generated = "check_board_visibility_skill_pb2"
    sources = [SOURCE_PATH]

    missing = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("aic_perception"):
                continue
            if generated in node.module:
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if generated in alias.name:
                    continue
                if not hasattr(module, alias.name):
                    missing.append(
                        f"{path.name}:{node.lineno} {node.module}.{alias.name}"
                    )

    assert not missing, f"imported but undefined: {missing}"


def test_skill_error_is_raised_with_both_required_arguments():
    """`SkillError.__init__(self, status_code, message)` takes two positionals.

    The intrinsic SDK is not importable in this test environment, so nothing
    executes this call path locally and a wrong arity reaches the robot as
    "SkillError.__init__() missing 1 required positional argument: 'message'".
    Check the call shape in the source instead.
    """
    source, _ = skill_source()
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "SkillError"
    ]
    assert calls, "the skill should raise a real SkillError somewhere"
    for call in calls:
        assert len(call.args) == 2, (
            f"SkillError at line {call.lineno} needs (status_code, message), "
            f"got {len(call.args)} positional argument(s)"
        )
        code = call.args[0]
        assert isinstance(code, ast.Constant) and isinstance(code.value, int), (
            f"SkillError at line {call.lineno}: first argument must be an "
            "integer status code"
        )
