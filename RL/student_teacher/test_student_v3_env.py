import numpy as np

from RL.student_teacher.student_v3_env import StudentV3Env


def _bare_v3() -> StudentV3Env:
    env = object.__new__(StudentV3Env)
    env._contact_engaged = False
    env._residual_pos = np.zeros(3, dtype=np.float64)
    env._residual_rot = np.zeros(3, dtype=np.float64)
    return env


def test_residual_is_gated_then_bounded_and_allows_retreat():
    env = _bare_v3()
    outside = np.zeros(69, dtype=np.float32)
    outside[34] = -0.010
    _combined, residual, _physical = env._compose_action(outside, np.ones(6))
    assert np.array_equal(residual, np.zeros(6))
    assert np.array_equal(env._residual_pos, np.zeros(3))

    contact = np.zeros(69, dtype=np.float32)
    for _ in range(20):
        _combined, residual, _physical = env._compose_action(contact, np.ones(6))
    assert np.all(env._residual_pos <= env._ACCUM_POS)
    assert np.all(env._residual_rot <= env._ACCUM_ROT)

    before = env._residual_pos.copy()
    _combined, residual, physical = env._compose_action(contact, -np.ones(6))
    assert np.all(residual < 0.0)
    assert np.all(physical[:3] < 0.0)
    assert np.all(env._residual_pos < before)
