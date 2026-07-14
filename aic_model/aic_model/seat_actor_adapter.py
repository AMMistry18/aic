"""Deploy-time ABI adapter for the TACC force-reactive seat actor.

The actor exported by ``RL.student_teacher.train_seat`` consumes an eight-frame
history of 34-value frames, not the legacy 69-value last-inch actor input.  Keep
this tiny NumPy-only adapter separate from ``RLInsert`` so the on-robot history
matches the trainer exactly and can be unit-tested without ROS.
"""
from __future__ import annotations

from collections import deque

import numpy as np


HISTORY = 8
FRAME_DIM = 34
ACTION_DIM = 6


def frame_from_obs69(
    obs69: np.ndarray,
    *,
    wrench_ema: np.ndarray,
    previous_action: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact actor frame used by ``SeatEnv._frame``.

    ``obs69`` is the common deployment contract: relative tip pose occupies
    32:38, port-frame TCP velocities 19:25, and baseline-subtracted wrench
    51:57.  The caller owns history initialization and action timing.
    """
    obs69 = np.asarray(obs69, dtype=np.float64).reshape(69)
    previous_action = np.asarray(previous_action, dtype=np.float64).reshape(ACTION_DIM)
    ema = np.asarray(wrench_ema, dtype=np.float64).reshape(6)
    rel = obs69[32:38].copy()
    wrench = obs69[51:57].copy()
    ema = 0.8 * ema + 0.2 * wrench
    lateral = float(np.linalg.norm(rel[:2]))
    rotation = float(np.linalg.norm(rel[3:6]))
    frame = np.concatenate([
        rel,
        obs69[19:25],
        wrench,
        ema,
        previous_action,
        np.array([lateral, rotation, rel[2], max(float(dt), 1e-4)]),
    ])
    if frame.shape != (FRAME_DIM,) or not np.all(np.isfinite(frame)):
        raise ValueError(f"invalid seat actor frame: shape={frame.shape}")
    return frame.astype(np.float32), ema


class SeatActorHistory:
    """Eight-frame actor history with the trainer's reset/action semantics."""

    def __init__(self) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=HISTORY)
        self._wrench_ema = np.zeros(6, dtype=np.float64)
        self._previous_action = np.zeros(ACTION_DIM, dtype=np.float64)

    def reset(self, obs69: np.ndarray, *, dt: float) -> np.ndarray:
        first, self._wrench_ema = frame_from_obs69(
            obs69,
            wrench_ema=self._wrench_ema,
            previous_action=self._previous_action,
            dt=dt,
        )
        self._frames = deque((first.copy() for _ in range(HISTORY)), maxlen=HISTORY)
        return self.value()

    def append(self, obs69: np.ndarray, *, dt: float) -> np.ndarray:
        if len(self._frames) != HISTORY:
            raise RuntimeError("seat actor history must be reset before append")
        frame, self._wrench_ema = frame_from_obs69(
            obs69,
            wrench_ema=self._wrench_ema,
            previous_action=self._previous_action,
            dt=dt,
        )
        self._frames.append(frame)
        return self.value()

    def set_previous_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        if not np.all(np.isfinite(action)):
            raise ValueError("seat action must be finite")
        self._previous_action = action.copy()

    def value(self) -> np.ndarray:
        if len(self._frames) != HISTORY:
            raise RuntimeError("seat actor history is incomplete")
        history = np.stack(tuple(self._frames), axis=0).astype(np.float32)
        if history.shape != (HISTORY, FRAME_DIM):
            raise RuntimeError(f"seat actor history shape drifted: {history.shape}")
        return history


__all__ = ["ACTION_DIM", "FRAME_DIM", "HISTORY", "SeatActorHistory", "frame_from_obs69"]
