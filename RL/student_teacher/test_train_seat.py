from collections import Counter

from RL.student_teacher.train_seat import (
    SeatCurriculumState,
    checkpoint_selection_score,
    fixed_eval_case_sequence,
    fixed_eval_class_sequence,
)


def test_fixed_evaluation_suite_has_exact_class_counts():
    assert Counter(fixed_eval_class_sequence(60, 90_000)) == {
        "live_shallow": 42,
        "centered_shallow": 9,
        "mid_tail": 6,
        "mastered_deep": 3,
    }
    assert Counter(fixed_eval_class_sequence(180, 190_000)) == {
        "live_shallow": 126,
        "centered_shallow": 27,
        "mid_tail": 18,
        "mastered_deep": 9,
    }
    assert fixed_eval_class_sequence(60, 90_000) == (
        fixed_eval_class_sequence(60, 90_000))


def test_fixed_evaluation_cases_balance_all_contact_variants():
    for episodes, seed in ((9, 1_600), (60, 90_000), (180, 190_000)):
        cases = fixed_eval_case_sequence(episodes, seed)
        assert cases == fixed_eval_case_sequence(episodes, seed)
        assert Counter(variant for _, variant in cases) == {
            0: episodes // 3,
            1: episodes // 3,
            2: episodes // 3,
        }

    periodic = fixed_eval_case_sequence(60, 90_000)
    for reset_class, count in {
            "live_shallow": 42,
            "centered_shallow": 9,
            "mid_tail": 6,
            "mastered_deep": 3,
    }.items():
        assert Counter(
            variant for name, variant in periodic if name == reset_class
        ) == {0: count // 3, 1: count // 3, 2: count // 3}


def test_sbc_updates_only_on_shared_hundred_episode_windows():
    state = SeatCurriculumState()
    assert state.record(["success"] * 99) == []
    events = state.record(["success"])
    assert events[-1]["success_rate"] == 1.0
    assert state.easy_max_mm == 37.0

    # The strict low threshold is <10%, and regression expands the easy side.
    state = SeatCurriculumState(easy_max_mm=20.0)
    events = state.record(["timeout"] * 100)
    assert events[-1]["success_rate"] == 0.0
    assert state.easy_max_mm == 23.0

    # Ten successes is exactly 10%, so neither boundary rule fires.
    state = SeatCurriculumState(easy_max_mm=20.0)
    state.record(["success"] * 10 + ["timeout"] * 90)
    assert state.easy_max_mm == 20.0


def test_sbc_state_round_trip_preserves_resume_window():
    state = SeatCurriculumState(easy_max_mm=27.0)
    state.record(["success"] * 41 + ["timeout"] * 18)
    restored = SeatCurriculumState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()
    restored.record(["success"] * 41)
    assert restored.easy_max_mm == 22.0


def test_checkpoint_selection_prioritizes_success_then_safety_then_force():
    base = {
        "eval/seat_success_rate": 0.9,
        "eval/bad_collision_rate": 0.02,
        "eval/force_abort_rate": 0.01,
        "eval/rotation_guard_rate": 0.0,
        "eval/max_force_n_p95": 12.0,
    }
    assert checkpoint_selection_score({
        **base, "eval/seat_success_rate": 0.91,
    }) > checkpoint_selection_score(base)
    assert checkpoint_selection_score({
        **base, "eval/bad_collision_rate": 0.0,
    }) > checkpoint_selection_score(base)
    assert checkpoint_selection_score({
        **base, "eval/max_force_n_p95": 10.0,
    }) > checkpoint_selection_score(base)
