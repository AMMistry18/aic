from pathlib import Path
import sys

import numpy as np
import pytest
from geometry_msgs.msg import Pose
from rclpy.clock import Clock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aic_model"))

from aic_model.policy import Policy  # noqa: E402


class _Logger:
    def info(self, _message):
        pass


class _Node:
    def __init__(self):
        self._clock = Clock()
        self._logger = _Logger()

    def get_clock(self):
        return self._clock

    def get_logger(self):
        return self._logger


class _Policy(Policy):
    def insert_cable(self, _task, _get_observation, _move_robot, _send_feedback):
        return False


def _capture_motion_update():
    messages = []

    def move_robot(*, motion_update=None, joint_motion_update=None):
        assert joint_motion_update is None
        messages.append(motion_update)

    return messages, move_robot


def test_set_pose_target_serializes_full_cartesian_matrices_row_major():
    policy = _Policy(_Node())
    stiffness = np.array(
        [
            [130.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [20.0, 130.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 500.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        ]
    )
    damping = stiffness * 0.2
    messages, move_robot = _capture_motion_update()

    policy.set_pose_target(
        move_robot,
        Pose(),
        stiffness=stiffness,
        damping=damping.reshape(36),
    )

    assert len(messages) == 1
    np.testing.assert_allclose(
        np.asarray(messages[0].target_stiffness).reshape(6, 6), stiffness
    )
    np.testing.assert_allclose(
        np.asarray(messages[0].target_damping).reshape(6, 6), damping
    )


def test_set_pose_target_keeps_legacy_diagonal_gain_lists():
    policy = _Policy(_Node())
    stiffness = [90.0, 91.0, 92.0, 50.0, 51.0, 52.0]
    damping = [40.0, 41.0, 42.0, 20.0, 21.0, 22.0]
    messages, move_robot = _capture_motion_update()

    policy.set_pose_target(move_robot, Pose(), stiffness=stiffness, damping=damping)

    np.testing.assert_allclose(
        np.asarray(messages[0].target_stiffness).reshape(6, 6), np.diag(stiffness)
    )
    np.testing.assert_allclose(
        np.asarray(messages[0].target_damping).reshape(6, 6), np.diag(damping)
    )


@pytest.mark.parametrize(
    ("stiffness", "match"),
    [
        (np.eye(5), "shape"),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), "shape"),
        (np.array([[1.0, np.nan, 0.0, 0.0, 0.0, 0.0]] * 6), "finite"),
        (
            np.array(
                [
                    [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "symmetric",
        ),
        (np.diag([1.0, 1.0, -1.0, 1.0, 1.0, 1.0]), "positive semidefinite"),
    ],
)
def test_set_pose_target_rejects_unsafe_cartesian_impedance(stiffness, match):
    policy = _Policy(_Node())
    _messages, move_robot = _capture_motion_update()

    with pytest.raises(ValueError, match=match):
        policy.set_pose_target(move_robot, Pose(), stiffness=stiffness)
