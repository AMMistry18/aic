"""CPU-only acceptance checks for the wedged-start Seat RL v2 env."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from RL.student_teacher.seat_env import (
    DEPLOYMENT_RESET_CLASSES,
    FORCE_SOFT_START_N,
    ROTATION_GUARD_RAD,
    SEATED_SUCCESS_BONUS,
    STAGES,
    SeatEnv,
    make_seat_env,
)
from RL.student_teacher.parity.evaluate_guided_controller import guided_action
from RL.student_teacher.student_env_a import DEPLOY_POS_SCALE


def _sample(stage: str, seed: int, n: int = 4):
    rows = []
    for i in range(n):
        env = make_seat_env(
            stage, seed=seed + i, domain_randomization=True)
        try:
            obs, info = env.reset(seed=seed + i)
            assert obs["actor"].shape == (8, 34)
            assert obs["privileged"].shape == (32,)
            assert info["seat_reset_cache_hit"]
            assert info["seat_reset_validation_mode"] == "cached_pose_safety_check"
            assert info["seat_reset_pool_size"] >= 1
            probe = info["seat_reset_probe"]
            rows.append({
                "depth_m": float(env.scene._insertion_depth_m()),
                "compiled_seed": int(env._compiled_seed),
                "lateral_m": float(info["lateral_error_m"]),
                "rotation_rad": float(np.hypot(
                    info["plug_axis_error_rad"], info["plug_roll_error_rad"])),
                "force_n": float(info["contact_force_norm"]),
                "contacts": int(info["plug_port_contacts"]),
                "level": float(info["curriculum_level"]),
                "attempts": int(info["seat_reset_attempts"]),
                "used_fallback": bool(info["seat_reset_used_fallback"]),
                "direction": str(info["seat_reset_direction"]),
                "true_wedge": bool(info["seat_reset_true_lateral_wedge"]),
                "straight_progress_m": float(
                    probe["straight_probe"]["depth_progress_m"]),
                "nudge_progress_m": float(
                    probe["accepted_probe"]["depth_progress_m"]),
                "rejected": list(info["seat_reset_rejected"]),
            })
        finally:
            env.close()
    return rows


def _describe(stage: str, rows: list[dict]):
    def values(key):
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    depth = 1e3 * values("depth_m")
    lateral = 1e3 * values("lateral_m")
    rotation = np.degrees(values("rotation_rad"))
    attempts = values("attempts")
    rejected = [item for row in rows for item in row["rejected"]]
    flat_resamples = sum(
        item.get("reason") in {
            "contact_or_offset", "not_stalled_axially", "flat_or_dead_stall"
        } for item in rejected)
    total_candidates = int(np.sum(attempts))
    accepted_fraction = len(rows) / max(total_candidates, 1)
    print(
        f"{stage}: level min/mean/max={values('level').min():.6f}/"
        f"{values('level').mean():.6f}/{values('level').max():.6f}; "
        f"depth_mm min/mean/max={depth.min():.3f}/{depth.mean():.3f}/{depth.max():.3f}; "
        f"lateral_mm min/mean/max={lateral.min():.3f}/{lateral.mean():.3f}/{lateral.max():.3f}; "
        f"rotation_deg min/mean/max={rotation.min():.3f}/{rotation.mean():.3f}/{rotation.max():.3f}; "
        f"accepted_true_wedges={len(rows)}/{total_candidates} "
        f"({accepted_fraction:.3f}); resamples={total_candidates-len(rows)} "
        f"flat_resamples={flat_resamples}; "
        f"resampled_reset_fraction={np.mean(attempts > 1):.3f}; "
        f"fallbacks={sum(row['used_fallback'] for row in rows)}; "
        f"directions={sorted({row['direction'] for row in rows})}; "
        f"compiled_seeds={sorted({row['compiled_seed'] for row in rows})}"
    )


def test_seat_reset_distribution():
    samples = {
        "near_seated": _sample("near_seated", 3101),
        "mid": _sample("mid", 3201),
        "wedge": _sample("wedge", 3301),
    }
    for stage, rows in samples.items():
        _describe(stage, rows)
        assert all(row["true_wedge"] for row in rows)
        assert all(row["contacts"] > 0 for row in rows)
        assert all(row["force_n"] <= 10.0 for row in rows)
        assert all(0.24e-3 <= row["lateral_m"] <= 1.0e-3 for row in rows)
        assert all(row["straight_progress_m"] <= 1.0e-3 for row in rows)
        assert all(row["nudge_progress_m"] >= 2.0e-3 for row in rows)

    means = {
        stage: np.mean([row["depth_m"] for row in rows])
        for stage, rows in samples.items()
    }
    assert means["near_seated"] > means["mid"]
    wedge_depths = np.asarray([
        row["depth_m"] for row in samples["wedge"]])
    wedge_by_variant = {
        row["compiled_seed"]: row["depth_m"] for row in samples["wedge"]
    }
    assert wedge_depths.min() < means["mid"]
    assert np.ptp(wedge_depths) >= 15e-3
    assert 34e-3 <= means["near_seated"] <= 42e-3
    assert 25e-3 <= means["mid"] <= 35e-3
    assert set(wedge_by_variant) == {20260715, 20260740, 20260731}
    assert 4.5e-3 <= wedge_by_variant[20260715] <= 7.5e-3
    assert 24e-3 <= wedge_by_variant[20260740] <= 30e-3
    assert 39e-3 <= wedge_by_variant[20260731] <= 42e-3
    assert STAGES["tight"] is STAGES["near_seated"]
    assert STAGES["band"] is STAGES["mid"]
    assert STAGES["full"] is STAGES["wedge"]


def test_shallow_handoff_has_validated_bounded_fallback():
    # The old hot reset needed this after eight random failures.  It remains a
    # safe pool-construction candidate, but runtime resets never rerun probes.
    env = make_seat_env("full", seed=91002, domain_randomization=True)
    try:
        candidate, base_seed = env._validated_fallback_candidate()
        _obs69, prepared_info, _reset_info, ended = env._prepare_candidate(
            base_seed, candidate)
        assert not ended
        start_depth_m = env.scene._insertion_depth_m()
        validation = env._validate_candidate(
            base_seed, candidate, prepared_info)
        assert validation["true_lateral_wedge"]
        assert env._start_safety_reason(prepared_info) is None
        assert 4.5e-3 <= start_depth_m <= 7.5e-3
        assert validation["straight_probe"]["depth_progress_m"] <= 1.0e-3
        assert validation["accepted_probe"]["depth_progress_m"] >= 2.0e-3
        print(
            "shallow_fallback: "
            f"depth_mm={1e3 * start_depth_m:.3f} "
            f"lateral_mm={1e3 * prepared_info['lateral_error_m']:.3f} "
            f"nudge_progress_mm="
            f"{1e3 * validation['accepted_probe']['depth_progress_m']:.3f}"
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("stage", "requested_seed", "depth_bounds_m"),
    (
        ("near_seated", 0, (34e-3, 42e-3)),
        ("full", 1, (24e-3, 30e-3)),
        ("full", 2, (39e-3, 42e-3)),
    ),
)
def test_other_handoff_fallbacks_are_true_lateral_wedges(
        stage, requested_seed, depth_bounds_m):
    env = make_seat_env(stage, seed=requested_seed, domain_randomization=True)
    try:
        candidate, base_seed = env._validated_fallback_candidate()
        _obs69, prepared_info, _reset_info, ended = env._prepare_candidate(
            base_seed, candidate)
        assert not ended
        start_depth_m = env.scene._insertion_depth_m()
        validation = env._validate_candidate(
            base_seed, candidate, prepared_info)
        assert validation["true_lateral_wedge"]
        assert depth_bounds_m[0] <= start_depth_m <= depth_bounds_m[1]
        assert validation["straight_probe"]["depth_progress_m"] <= 1.0e-3
        assert validation["accepted_probe"]["depth_progress_m"] >= 2.0e-3
    finally:
        env.close()


def test_near_seated_repeated_resets_are_validated_and_never_raise():
    """Cached reset reconstruction stays safe without rerunning probe sweeps."""
    resets = 16
    env = make_seat_env("near_seated", seed=3101, domain_randomization=True)
    rows = []
    try:
        for _ in range(resets):
            obs, info = env.reset()
            probe = info["seat_reset_probe"]
            assert obs["actor"].shape == (8, 34)
            assert obs["privileged"].shape == (32,)
            assert info["seat_reset_true_lateral_wedge"]
            assert info["plug_port_contacts"] > 0
            assert info["contact_force_norm"] <= 10.0
            assert info["seat_reset_cache_hit"]
            assert info["seat_reset_validation_mode"] == "cached_pose_safety_check"
            assert probe["straight_probe"]["depth_progress_m"] <= 1.0e-3
            assert probe["accepted_probe"]["depth_progress_m"] >= 2.0e-3
            rows.append(info)
    finally:
        env.close()

    fallback_count = sum(bool(row["seat_reset_used_fallback"]) for row in rows)
    attempts = np.asarray(
        [row["seat_reset_attempts"] for row in rows], dtype=np.int64)
    print(
        "near_seated_repeated_resets: "
        f"validated={len(rows)}/{resets}; fallbacks={fallback_count}/{resets}; "
        f"mean_attempts={attempts.mean():.3f}; max_attempts={attempts.max()}"
    )
    assert len(rows) == resets


def test_cached_reset_does_not_rerun_physical_validator(monkeypatch):
    env = make_seat_env("near_seated", seed=3101, domain_randomization=True)
    try:
        _obs, first = env.reset(seed=3101)
        assert first["seat_reset_pool_size"] >= 1

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("hot reset reran the physical nudge validator")

        monkeypatch.setattr(env, "_validate_candidate", fail_if_called)
        for seed in range(4100, 4108):
            _obs, info = env.reset(seed=seed)
            assert info["seat_reset_true_lateral_wedge"]
            assert info["seat_reset_cache_hit"]
            assert env._start_safety_reason(info) is None
    finally:
        env.close()


def test_validator_rejects_unsafe_probe_terminations(monkeypatch):
    env = object.__new__(SeatEnv)
    env.stage = STAGES["near_seated"]
    env.scene = SimpleNamespace(cfg=SimpleNamespace(
        bad_collision_penetration_excess_m=1.5e-3,
        bad_collision_overinsert_m=2.0e-3,
        bad_collision_depth_gate=0.45,
        bad_collision_axis_rad=0.35,
        bad_collision_roll_rad=0.35,
    ))
    prepared = {
        "contact_force_norm": 3.0,
        "lateral_error_m": 0.35e-3,
        "plug_port_contacts": 2,
        "off_limit_contacts": 0,
        "plug_port_penetration_excess_m": 0.0,
        "overinsert_m": 0.0,
        "depth_norm": 0.8,
        "plug_axis_error_rad": 0.05,
        "plug_roll_error_rad": 0.05,
    }

    monkeypatch.setattr(env, "_straight_probe", lambda *_args: {
        "depth_progress_m": 0.0,
        "end_status": "bad_collision",
    })
    result = env._validate_candidate(1, {}, prepared)
    assert not result["true_lateral_wedge"]
    assert result["reason"] == "unsafe_straight_probe"

    monkeypatch.setattr(env, "_straight_probe", lambda *_args: {
        "depth_progress_m": 0.0,
        "end_status": None,
    })
    monkeypatch.setattr(env, "_nudge_probe", lambda *_args: {
        "depth_progress_m": 3.0e-3,
        "end_status": "force_abort",
        "unstick_success": True,
    })
    result = env._validate_candidate(1, {}, prepared)
    assert not result["true_lateral_wedge"]
    assert result["reason"] == "flat_or_dead_stall"

    penetrating = {**prepared, "plug_port_penetration_excess_m": 2.0e-3}
    assert env._start_safety_reason(penetrating) == "bad_collision_penetration"


def _reward_case(
        before_rel, rel, info, *, prev_f_lateral=0.0, commanded=None):
    """Exercise the numerical reward without constructing a MuJoCo scene."""
    env = object.__new__(SeatEnv)
    env._prev_action = np.zeros(6, dtype=np.float64)
    env._prev_f_lateral = float(prev_f_lateral)
    env._last_reward_terms = {}
    reward = env._seat_reward(
        np.asarray(before_rel, dtype=np.float64),
        np.asarray(rel, dtype=np.float64),
        (np.zeros(6, dtype=np.float64) if commanded is None
         else np.asarray(commanded, dtype=np.float64)),
        dict(info),
    )
    return reward, env._last_reward_terms


def test_seat_reward_signs():
    zero = np.zeros(6, dtype=np.float64)

    deeper = zero.copy()
    deeper[2] = 0.5e-3
    reward, terms = _reward_case(
        zero, deeper,
        {"f_z": 5.0, "f_lateral": 3.0, "contact_force_norm": 6.0,
         "term_status": None},
        prev_f_lateral=8.0,
    )
    assert reward > 0.5
    assert terms["depth"] > 0.0
    assert terms["force_direction"] > 0.0

    reward, terms = _reward_case(
        zero, zero,
        {"f_z": 18.0, "f_lateral": 0.0, "contact_force_norm": 18.0,
         "term_status": None},
    )
    assert FORCE_SOFT_START_N < 18.0
    assert reward < 0.0
    assert terms["force_soft_cap"] < 0.0

    before_lateral = zero.copy()
    after_lateral = zero.copy()
    before_lateral[0] = 1.0e-3
    after_lateral[0] = 0.5e-3
    reward, terms = _reward_case(
        before_lateral, after_lateral,
        {"f_z": 0.0, "f_lateral": 0.0, "contact_force_norm": 0.0,
         "term_status": None},
    )
    assert reward > 0.0
    assert terms["lateral"] > 0.0

    # With identical depth progress, unloading the lateral wedge must beat an
    # axial shove that keeps the side load and lateral pose error unchanged.
    relief_after = before_lateral.copy()
    relief_after[0] = 0.5e-3
    relief_after[2] = 0.5e-3
    shove_after = before_lateral.copy()
    shove_after[2] = 0.5e-3
    relief_reward, relief_terms = _reward_case(
        before_lateral, relief_after,
        {"f_z": 5.0, "f_lateral": 2.0, "contact_force_norm": 6.0,
         "term_status": None},
        prev_f_lateral=8.0,
    )
    shove_reward, shove_terms = _reward_case(
        before_lateral, shove_after,
        {"f_z": 5.0, "f_lateral": 8.0, "contact_force_norm": 9.5,
         "term_status": None},
        prev_f_lateral=8.0,
    )
    print(
        f"reward_unstick_vs_shove: relief={relief_reward:.6f} "
        f"shove={shove_reward:.6f}"
    )
    assert relief_reward > shove_reward
    assert relief_terms["lateral"] > shove_terms["lateral"]
    assert relief_terms["force_direction"] > shove_terms["force_direction"]
    assert relief_terms["lateral_force"] > shove_terms["lateral_force"]

    reward, terms = _reward_case(
        zero, zero,
        {"f_z": 0.0, "f_lateral": 0.0, "contact_force_norm": 0.0,
         "term_status": "success"},
    )
    assert terms["success"] == pytest.approx(SEATED_SUCCESS_BONUS)
    assert reward >= SEATED_SUCCESS_BONUS

    reward, terms = _reward_case(
        zero, zero,
        {"f_z": 0.0, "f_lateral": 0.0, "contact_force_norm": 0.0,
         "term_status": "force_abort"},
    )
    assert terms["failure"] == pytest.approx(-20.0)
    assert reward < 0.0


def test_squareness_penalty_starts_at_entry_and_is_sign_safe():
    zero = np.zeros(6, dtype=np.float64)
    base = {
        "f_z": 0.0,
        "f_lateral": 0.0,
        "contact_force_norm": 0.0,
        "term_status": None,
        "plug_axis_error_rad": np.radians(2.0),
        "plug_roll_error_rad": 0.0,
    }

    _reward, deep = _reward_case(
        zero, zero, {**base, "depth_norm": 0.45})
    _reward, shallow = _reward_case(
        zero, zero, {**base, "depth_norm": 0.13})
    _reward, square = _reward_case(
        zero, zero, {
            **base,
            "depth_norm": 0.45,
            "plug_axis_error_rad": 0.0,
            "plug_roll_error_rad": 0.0,
        })

    assert deep["squareness"] < -0.5
    assert shallow["squareness"] < 0.0
    assert deep["squareness"] < shallow["squareness"]
    assert square["squareness"] == pytest.approx(0.0)

    _reward, more_crooked = _reward_case(
        zero, zero, {**base, "depth_norm": 0.45,
                     "plug_axis_error_rad": np.radians(4.0)})
    assert more_crooked["squareness"] < deep["squareness"]


def test_inward_square_beats_inward_rotating_and_square_does_not_rotate():
    before = np.zeros(6, dtype=np.float64)
    after = before.copy()
    after[2] = 0.5e-3
    common = {
        "depth_norm": 0.14,
        "f_z": 2.0,
        "f_lateral": 1.0,
        "contact_force_norm": 2.5,
        "term_status": None,
        "plug_roll_error_rad": 0.0,
    }
    square_reward, _ = _reward_case(
        before, after, {**common, "plug_axis_error_rad": 0.0})
    rotating_reward, _ = _reward_case(
        before, after, {
            **common, "plug_axis_error_rad": 0.8 * ROTATION_GUARD_RAD})
    assert square_reward > rotating_reward

    hold_reward, _ = _reward_case(
        before, before, {**common, "plug_axis_error_rad": 0.0})
    needless_rotation_reward, _ = _reward_case(
        before, before, {**common, "plug_axis_error_rad": 0.0},
        commanded=np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]))
    assert hold_reward > needless_rotation_reward


@pytest.mark.parametrize("reset_class", tuple(DEPLOYMENT_RESET_CLASSES))
def test_deployment_reset_class_delivers_actor_and_physical_contract(reset_class):
    index = tuple(DEPLOYMENT_RESET_CLASSES).index(reset_class)
    env = make_seat_env("deployment", seed=index, domain_randomization=True)
    try:
        obs, info = env.reset(
            seed=91_000 + index,
            options={"seat_reset_class": reset_class},
        )
        spec = DEPLOYMENT_RESET_CLASSES[reset_class]
        assert info["seat_reset_validated"]
        assert info["seat_reset_compiled_seed"] == env._compiled_seed
        assert not info["seat_reset_used_fallback"]
        assert info["seat_reset_validation_mode"] == (
            "physical_and_actor_delivered_state")
        assert spec.depth_range_m[0] <= (
            info["seat_reset_delivered_depth_m"]) <= spec.depth_range_m[1]
        assert spec.actor_lateral_range_m[0] <= (
            info["seat_reset_delivered_actor_lateral_m"]
        ) <= spec.actor_lateral_range_m[1]
        assert info["seat_reset_delivered_physical_lateral_m"] <= 1.2e-3
        assert info["contact_force_norm"] <= 10.0
        assert info["off_limit_contacts"] == 0
        assert obs["actor"].shape == (8, 34)
        assert obs["privileged"].shape == (32,)
    finally:
        env.close()


def test_wedge_random_vs_lateral_nudge_feasibility():
    episodes = 4
    seed = 20260713
    rng = np.random.default_rng(seed)
    env = make_seat_env("wedge", seed=seed, domain_randomization=True)
    random_success = 0
    random_status: dict[str, int] = {}
    nudge_success = 0
    nudge_progress = []
    try:
        for episode in range(episodes):
            obs, _reset_info = env.reset(seed=seed + episode)
            for _ in range(env.scene.cfg.max_episode_steps):
                action = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(reward)
                assert all(np.isfinite(v) for v in info["seat_reward_terms"].values())
                if terminated or truncated:
                    break
            status = str(info.get("term_status") or "timeout")
            random_status[status] = random_status.get(status, 0) + 1
            random_success += int(status == "success")

        for episode in range(episodes):
            obs, reset_info = env.reset(seed=seed + 100 + episode)
            start_depth = float(env.scene._insertion_depth_m())
            max_depth = start_depth
            direction = SeatEnv._direction(
                reset_info["seat_reset_probe"]["accepted_probe"]["direction"])
            # Exact Phase-1 physical feasibility probe: one 0.75 mm lateral
            # correction, then guided descent. Random policy above still uses
            # the unchanged contact-scaled SeatEnv action path.
            obs69 = env._current_obs69.copy()
            action = guided_action(obs69).astype(np.float64)
            action[:2] = direction * (
                0.75e-3 / np.asarray(DEPLOY_POS_SCALE[:2], dtype=np.float64))
            obs69, _reward, terminated, truncated, info = env.contract_env.step(
                np.clip(action, -1.0, 1.0).astype(np.float32))
            max_depth = max(max_depth, float(info["insertion_depth_m"]))
            for _ in range(40):
                if terminated or truncated:
                    break
                obs69, _reward, terminated, truncated, info = (
                    env.contract_env.step(guided_action(obs69)))
                max_depth = max(max_depth, float(info["insertion_depth_m"]))
            progress = max_depth - start_depth
            success = bool(
                progress >= 2.0e-3 or info.get("term_status") == "success")
            nudge_success += int(success)
            nudge_progress.append(progress)
    finally:
        env.close()

    print(
        f"wedge_random_vs_nudge: random_success={random_success}/{episodes} "
        f"random_status={random_status}; lateral_nudge_success="
        f"{nudge_success}/{episodes}; nudge_progress_mm="
        f"{[round(1e3*x, 3) for x in nudge_progress]}"
    )
    assert random_success <= 1
    assert nudge_success >= 3


if __name__ == "__main__":
    test_seat_reset_distribution()
