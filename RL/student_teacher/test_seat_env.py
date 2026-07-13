"""CPU-only reset-distribution acceptance check for the Stage-A seat env."""
from __future__ import annotations

import numpy as np
import pytest

from RL.student_teacher.seat_env import (
    FORCE_SOFT_START_N,
    SEATED_SUCCESS_BONUS,
    SEAT_RETRACTION_FROM_SEATED_M,
    SeatEnv,
    make_seat_env,
)


def _sample(stage: str, seed: int = 7, n: int = 20):
    env = make_seat_env(stage, seed=seed, domain_randomization=False)
    lateral, rotation, retraction, insertion = [], [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        assert obs["actor"].shape == (8, 34)
        assert obs["privileged"].shape == (32,)
        obs69 = env._current_obs69
        lateral.append(float(np.linalg.norm(obs69[32:38][:2])))
        rotation.append(float(np.linalg.norm(obs69[35:38])))
        retraction.append(float(env.scene.cfg.seated_depth_m - env.scene._insertion_depth_m()))
        insertion.append(float(env.scene._insertion_depth_m()))
    return (np.asarray(lateral), np.asarray(rotation), np.asarray(retraction),
            np.asarray(insertion))


def _describe(name, lateral, rotation, retraction, insertion):
    print(
        f"{name}: lateral_mm min/mean/max={lateral.min()*1e3:.3f}/"
        f"{lateral.mean()*1e3:.3f}/{lateral.max()*1e3:.3f}; "
        f"rotation_deg min/mean/max={np.degrees(rotation.min()):.3f}/"
        f"{np.degrees(rotation.mean()):.3f}/{np.degrees(rotation.max()):.3f}; "
        f"retraction_from_seated_mm min/mean/max={retraction.min()*1e3:.3f}/"
        f"{retraction.mean()*1e3:.3f}/{retraction.max()*1e3:.3f}; "
        f"insertion_depth_mm min/mean/max={insertion.min()*1e3:.3f}/"
        f"{insertion.mean()*1e3:.3f}/{insertion.max()*1e3:.3f}"
    )


def _zero_jitter_floor(seed: int = 101) -> tuple[float, float]:
    """Measure the reset/settle residual with commanded in-port jitter disabled."""
    env = make_seat_env("tight", seed=seed, domain_randomization=False)
    env.scene.cfg.jitter_xy_inport_m = 0.0
    env.scene.cfg.jitter_yaw_inport_rad = 0.0
    env.scene.cfg.jitter_tilt_inport_rad = 0.0
    env.reset(seed=seed)
    obs69 = env._current_obs69
    return (float(np.linalg.norm(obs69[32:38][:2])),
            float(np.linalg.norm(obs69[35:38])))


def test_seat_reset_distribution():
    tight = _sample("tight")
    full = _sample("full")
    _describe("tight", *tight)
    _describe("full", *full)
    floor_lat, floor_rot = _zero_jitter_floor()
    print(f"zero_jitter_floor: lateral_mm={floor_lat*1e3:.3f}; "
          f"rotation_deg={np.degrees(floor_rot):.3f}")

    tight_lat, tight_rot, tight_retract, tight_insert = tight
    full_lat, full_rot, full_retract, full_insert = full
    # Uniform radial-in-a-box samples include small values; require the tight
    # distribution to remain centred in the measured 0.7 mm hand-off regime.
    assert 0.25e-3 <= tight_lat.mean() <= 0.70e-3
    # ~1.1 degrees is the reset/settle floor, not commanded jitter.
    assert np.degrees(tight_rot.max()) <= 1.6
    assert full_lat.max() <= 2.0e-3
    assert np.degrees(full_rot.max()) <= 2.0
    assert np.radians(0.9) <= floor_rot <= np.radians(1.5)
    assert floor_lat < 0.7e-3
    for values in (tight_retract, full_retract):
        assert np.allclose(
            values, SEAT_RETRACTION_FROM_SEATED_M, atol=1.5e-3)
    for values in (tight_insert, full_insert):
        assert 4.5e-3 <= values.min()
        assert values.max() <= 7.0e-3
        assert 5.0e-3 <= values.mean() <= 6.5e-3


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


def test_random_policy_smoke_and_feasibility():
    """Five-episode safety smoke plus the 50-episode Stage-C random gate."""
    episodes = 50
    smoke_episodes = 5
    seed = 20260713
    rng = np.random.default_rng(seed)
    env = make_seat_env("tight", seed=seed, domain_randomization=False)
    seated_depth = float(env.scene.cfg.seated_depth_m)
    force_abort_n = float(env.scene.cfg.force_abort_n)

    seated_count = 0
    exact_depth_count = 0
    bonus_count = 0
    smoke_force_below = 0
    smoke_steps = 0
    smoke_max_force = 0.0
    total_steps = 0
    traces = []

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        start_depth = float(env.scene._insertion_depth_m())
        max_depth = start_depth
        episode_max_force = 0.0
        end_status = None
        episode_seated = False
        episode_exact_depth = False
        for _ in range(env.scene.cfg.max_episode_steps):
            action = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
            _obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            assert all(np.isfinite(v) for v in info["seat_reward_terms"].values())

            force = float(info["contact_force_norm"])
            assert np.isfinite(force)
            episode_max_force = max(episode_max_force, force)
            max_depth = max(max_depth, float(info["insertion_depth_m"]))
            end_status = info.get("term_status")
            total_steps += 1
            if episode < smoke_episodes:
                smoke_steps += 1
                smoke_force_below += int(force < force_abort_n)
                smoke_max_force = max(smoke_max_force, force)

            if float(info["insertion_depth_m"]) >= seated_depth:
                episode_exact_depth = True
            if info.get("term_status") == "success":
                episode_seated = True
            if info["seat_reward_terms"]["success"] == SEATED_SUCCESS_BONUS:
                bonus_count += 1
            if terminated or truncated:
                break
        seated_count += int(episode_seated)
        exact_depth_count += int(episode_exact_depth)
        if episode < 5:
            traces.append((episode, start_depth, max_depth, end_status,
                           episode_max_force))

    gentle_fraction = smoke_force_below / max(smoke_steps, 1)
    print(
        f"random_smoke_5: steps={smoke_steps} force_below_abort="
        f"{smoke_force_below}/{smoke_steps} ({gentle_fraction:.3f}) "
        f"max_force_n={smoke_max_force:.3f}"
    )
    print(
        f"random_feasibility_50: seated_success={seated_count}/{episodes} "
        f"exact_depth={exact_depth_count}/{episodes} "
        f"success_bonus_fires={bonus_count} total_steps={total_steps}"
    )
    for episode, start_depth, max_depth, status, max_force in traces:
        print(
            f"random_trace[{episode}]: start_depth_mm={start_depth*1e3:.3f} "
            f"max_depth_mm={max_depth*1e3:.3f} end_status={status} "
            f"max_force_n={max_force:.3f}"
        )
    assert gentle_fraction >= 0.90
    assert bonus_count == seated_count


if __name__ == "__main__":
    test_seat_reset_distribution()
