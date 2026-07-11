"""No-training compute-node preflight for Gazebo-v1 student distillation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

try:
    from aic_model.rl_insert_contract import deploy_action_delta, port_frame
except ModuleNotFoundError:
    from aic_model.aic_model.rl_insert_contract import deploy_action_delta, port_frame
from RL.student_teacher.scripted_teacher_funnel import ScriptedTeacher
from RL.student_teacher.student_env_a import make_student_env_a
from RL.student_teacher.train_student_a import _teacher_target


TEACHER = Path("RL/student_teacher/weights/teacher_level1.zip")
TEACHER_SHA256 = "fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4"


def main() -> None:
    digest = hashlib.sha256(TEACHER.read_bytes()).hexdigest()
    assert digest == TEACHER_SHA256, (digest, TEACHER_SHA256)

    env = make_student_env_a(
        perception_noise=1.0,
        grasp_noise=1.0,
        level=1.0,
        action_convention="deploy",
        wrench_mode="baseline",
        seed=20260709,
    )
    try:
        obs, _ = env.reset(seed=20260709)
        privileged = env.priv_obs()
        assert obs.shape == (69,) and np.all(np.isfinite(obs)), obs
        assert privileged.shape == (21,) and np.all(np.isfinite(privileged)), privileged

        teacher = SAC.load(TEACHER, device="cpu")
        funnel = ScriptedTeacher(action_dim=6)
        funnel.reset()
        effective_sim, target = _teacher_target(
            env, funnel, teacher, residual_scale=0.15, convention="deploy"
        )
        assert effective_sim.shape == (6,) and np.all(np.isfinite(effective_sim))
        assert target.shape == (6,) and np.all(np.isfinite(target))

        port_quat = env.perceived_port_pose()[3:]
        delta_world, _ = deploy_action_delta([0, 0, 1, 0, 0, 0], port_quat)
        frame = port_frame(port_quat)
        axial = float(np.dot(delta_world, frame[:, 2]))
        lateral = float(np.linalg.norm(frame[:, :2].T @ delta_world))
        assert axial > 0.0 and abs(axial - 0.0035) <= 1e-9, axial
        assert lateral <= 1e-12, lateral

        next_obs, _, _, _, _ = env.step_sim(effective_sim)
        assert next_obs.shape == (69,) and np.all(np.isfinite(next_obs)), next_obs
    finally:
        env.close()

    print(json.dumps({
        "status": "passed",
        "wrench_mode": "baseline",
        "teacher_sha256": digest,
        "student_obs_shape": list(obs.shape),
        "teacher_obs_shape": list(privileged.shape),
        "teacher_target": target.tolist(),
        "positive_axial_delta_m": axial,
        "positive_axial_lateral_m": lateral,
        "step_sim_finite": True,
    }, indent=2))


if __name__ == "__main__":
    main()
