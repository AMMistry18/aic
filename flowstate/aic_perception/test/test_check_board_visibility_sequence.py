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


def test_stage2_landmark_is_rechecked_after_every_seek_move():
    """The loop exists only to reach a Stage-2-valid insignia.

    Checked once before moving at all, and again after each hop, so the search
    stops the instant Stage 2 can take over.
    """
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)
    landmark_lines = call_lines(
        execute_inner, "_stage2_has_complete_landmark"
    )
    seek_lines = call_lines(execute_inner, "_seek_step")

    assert len(landmark_lines) == 2
    assert len(seek_lines) == 1
    assert landmark_lines[0] < seek_lines[0] < landmark_lines[1]
    # The joint-target path is gone: the deployed controller drops joint
    # target mode mid-segment ("controller left joint target mode"), and the
    # large reconfigurations it planned could not execute inside a move
    # deadline.  Seeking is Cartesian and orientation-preserving.
    assert "move_joint_target" not in method_source
    assert "validate_observation_path" not in method_source
    assert "OBSERVATION_JOINTS_RAD" not in method_source


def test_seek_runs_on_progress_not_a_fixed_hop_count():
    """No arbitrary move budget: a corner start needs more hops than a
    near-framed one, so the search runs while it improves and stops when it
    stalls.  A termination backstop still exists because Stage 1 has no
    aggregate wall clock and an unbounded loop would hang the skill.
    """
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "MAX_CENTER_MOVES" not in method_source
    assert "while True" not in method_source
    assert "SEEK_STALL_MOVES" in method_source
    assert "SEEK_HARD_MOVE_CEILING" in method_source
    assert named_call_lines(execute_inner, "seek_progress_score")
    # Two selections: one before the loop, one after each move.
    assert len(named_call_lines(execute_inner, "select_work_target")) == 2


def test_motion_requires_a_fresh_force_baseline():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "wait_for_force_xyz" in method_source
    assert "baseline_force_xyz" in method_source
    assert "no fresh wrist-force sample" in method_source
    assert call_lines(execute_inner, "wait_for_force_xyz")[0] < call_lines(
        execute_inner, "_seek_step"
    )[0]


def test_stage1_never_publishes_an_internal_motion_as_final_target():
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert "result.target_valid = False" in method_source
    assert "result.target.x" not in method_source
    assert "result.target.qw" not in method_source
    assert "result.target_frame = \"\"" in method_source


def test_seek_stall_is_normal_not_done():
    """A search that runs out of budget is an expected outcome, not an error.

    Flowstate must still get success=true/done=false so it can release the AIC
    controller (handoff section 3).
    """
    source, execute_inner = method("_execute_inner")
    method_source = ast.get_source_segment(source, execute_inner)

    assert 'result.last_action = "seek_stalled"' in method_source
    assert "board seek stalled" in method_source
    assert 'result.last_action = "seek_ceiling_reached"' in method_source
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
    sources = [
        SOURCE_PATH,
        SOURCE_PATH.parent / "aic_perception" / "board_seek.py",
    ]

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
