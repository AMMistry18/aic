"""CPU-only acceptance checks for the wedged-start Seat RL v2 env."""
from __future__ import annotations

import numpy as np
import pytest

from RL.student_teacher.seat_env import (
    FORCE_SOFT_START_N,
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
    # This exact evaluation seed exhausted all 12 random candidates in the
    # Phase-3 smoke.  It must now recover via the validated shallow fallback,
    # not crash or accept a flat/axially-mobile start.
    env = make_seat_env("full", seed=91002, domain_randomization=True)
    try:
        obs, info = env.reset(seed=91002)
        probe = info["seat_reset_probe"]
        assert obs["actor"].shape == (8, 34)
        assert obs["privileged"].shape == (32,)
        assert info["seat_reset_used_fallback"]
        assert info["seat_reset_attempts"] == 9
        assert info["seat_reset_true_lateral_wedge"]
        assert 4.5e-3 <= env.scene._insertion_depth_m() <= 7.5e-3
        assert probe["straight_probe"]["depth_progress_m"] <= 1.0e-3
        assert probe["accepted_probe"]["depth_progress_m"] >= 2.0e-3
        print(
            "shallow_fallback: "
            f"attempts={info['seat_reset_attempts']} "
            f"depth_mm={1e3 * env.scene._insertion_depth_m():.3f} "
            f"lateral_mm={1e3 * info['lateral_error_m']:.3f} "
            f"nudge_progress_mm="
            f"{1e3 * probe['accepted_probe']['depth_progress_m']:.3f}"
        )
    finally:
        env.close()


def _reward_case(before_rel, rel, info, *, prev_f_lateral=0.0):
    """Exercise the numerical reward without constructing a MuJoCo scene."""
    env = object.__new__(SeatEnv)
    env._prev_action = np.zeros(6, dtype=np.float64)
    env._prev_f_lateral = float(prev_f_lateral)
    env._last_reward_terms = {}
    reward = env._seat_reward(
        np.asarray(before_rel, dtype=np.float64),
        np.asarray(rel, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
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


def test_squareness_penalty_is_depth_ramped_and_sign_safe():
    zero = np.zeros(6, dtype=np.float64)
    base = {
        "f_z": 0.0,
        "f_lateral": 0.0,
        "contact_force_norm": 0.0,
        "term_status": None,
        "plug_axis_error_rad": 0.3,
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

    assert deep["squareness"] < -2.0
    assert shallow["squareness"] == pytest.approx(0.0)
    assert square["squareness"] == pytest.approx(0.0)

    _reward, more_crooked = _reward_case(
        zero, zero, {**base, "depth_norm": 0.45,
                     "plug_axis_error_rad": 0.34})
    assert more_crooked["squareness"] < deep["squareness"]


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
