import numpy as np

from aic_model.seat_actor_adapter import SeatActorHistory, frame_from_obs69


def test_frame_matches_seat_training_layout_and_ema():
    obs = np.zeros(69, dtype=np.float32)
    obs[19:25] = np.arange(6, dtype=np.float32) + 10
    obs[32:38] = [0.003, -0.004, 0.012, 0.1, -0.2, 0.2]
    obs[51:57] = np.arange(6, dtype=np.float32) + 1
    previous = np.arange(6, dtype=np.float32) / 10

    frame, ema = frame_from_obs69(
        obs, wrench_ema=np.zeros(6), previous_action=previous, dt=0.05)

    np.testing.assert_allclose(frame[:6], obs[32:38])
    np.testing.assert_allclose(frame[6:12], obs[19:25])
    np.testing.assert_allclose(frame[12:18], obs[51:57])
    np.testing.assert_allclose(ema, 0.2 * obs[51:57])
    np.testing.assert_allclose(frame[18:24], ema)
    np.testing.assert_allclose(frame[24:30], previous)
    np.testing.assert_allclose(frame[30:], [0.005, 0.3, 0.012, 0.05])


def test_history_reset_repeats_first_frame_then_tracks_last_action():
    obs = np.zeros(69, dtype=np.float32)
    obs[51] = 5.0
    history = SeatActorHistory()
    initial = history.reset(obs, dt=0.1)
    assert initial.shape == (8, 34)
    np.testing.assert_allclose(initial, np.broadcast_to(initial[0], initial.shape))

    history.set_previous_action(np.full(6, 0.2, dtype=np.float32))
    obs[32] = 0.001
    next_history = history.append(obs, dt=0.2)
    np.testing.assert_allclose(next_history[-1, 0], 0.001)
    np.testing.assert_allclose(next_history[-1, 24:30], 0.2)
    np.testing.assert_allclose(next_history[-1, -1], 0.2)
