"""Frozen privileged-teacher observation contract.

The 100% SAC teacher was trained on this exact 21-dimensional vector.  The
maintained :mod:`RL.scene_env` no longer exposes a privileged observation mode,
so distillation wrappers build the vector read-only from the scene instead of
mutating the environment's public observation space.
"""
from __future__ import annotations

import numpy as np
import mujoco


TEACHER_OBS_DIM = 21


def build_teacher_obs21(scene) -> np.ndarray:
    """Return the exact GT state consumed by ``teacher_level1.zip``.

    Layout: lateral tip offset (2), axial remaining (1), axis/roll/depth (3),
    tip linear velocity in the port frame (3), raw six-axis wrench (6), and arm
    joint velocity (6).
    """
    d = scene.data
    tip = d.xpos[scene._plug_tip_id]
    _axial, lateral = scene._tip_port_errors(tip)
    axial_remaining = float(
        scene.cfg.seated_depth_m - scene._insertion_depth_m(tip)
    )

    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        scene.model,
        d,
        mujoco.mjtObj.mjOBJ_BODY,
        int(scene._plug_tip_id),
        velocity,
        0,
    )
    linear_world = velocity[3:6]
    linear_port = np.array(
        [
            np.dot(linear_world, scene._lat_x),
            np.dot(linear_world, scene._lat_y),
            np.dot(linear_world, scene._insert_axis),
        ],
        dtype=np.float64,
    )

    obs = np.concatenate(
        [
            np.asarray(lateral, dtype=np.float64),
            [axial_remaining],
            [scene._plug_axis_error(), scene._plug_roll_error(), scene._depth_norm()],
            linear_port,
            scene._raw_ft(),
            d.qvel[scene._arm_vadr],
        ]
    ).astype(np.float32)
    if obs.shape != (TEACHER_OBS_DIM,) or not np.all(np.isfinite(obs)):
        raise ValueError(f"invalid teacher observation: shape={obs.shape} obs={obs}")
    return obs


__all__ = ["TEACHER_OBS_DIM", "build_teacher_obs21"]
