#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import time

import numpy as np
from abc import ABC, abstractmethod
from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Vector3, Wrench
from rclpy.duration import Duration
from std_msgs.msg import Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from typing import Callable, Protocol

GetObservationCallback = Callable[[], Observation]


class MoveRobotCallback(Protocol):
    """Move the robot using either Cartesian or joint-space commands.

    This function is called by a policy to request robot motion. Either
    Cartesian or joint-space commands can be sent, but not both at the
    same time. One of the following must be set:
     - motion_update: cartesian motion commands
     - joint_motion_update: joint-space motion commands

    The MotionUpdate message contains a request to the Cartesian-space
    admittance controller. The details of this message are described
    in its message definition:
      https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_control_interfaces/msg/MotionUpdate.msg

    The JointMotionUpdate message contains commands to the joint-space
    controller. The details of this message are described in its message definition:
      https://github.com/intrinsic-dev/aic/blob/main/aic_interfaces/aic_control_interfaces/msg/JointMotionUpdate.msg

    As a convenience, a reasonable set of parameters is populated in a MotionUpdate
    message by the create_motion_update(pose) function in the Policy class.
    """

    def __call__(
        self,
        motion_update: MotionUpdate = None,
        joint_motion_update: JointMotionUpdate = None,
    ) -> None: ...


SendFeedbackCallback = Callable[[str], None]


def _cartesian_impedance_matrix(values, name: str) -> list[float]:
    """Serialize legacy Cartesian gains or a full 6x6 impedance matrix.

    Existing callers provide six diagonal gains.  A 6x6 nested sequence or
    NumPy array (and its 36-element row-major form) is also accepted so a
    caller can express directional compliance in a rotated insertion frame.
    The controller applies the matrix to a base-frame Cartesian error, so a
    full matrix must already be expressed in that frame; ``frame_id`` only
    controls the target pose.
    """
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as ex:
        raise ValueError(
            f"{name} must be six diagonal gains or a 6x6 Cartesian matrix"
        ) from ex

    if matrix.shape == (6,):
        matrix = np.diag(matrix)
    elif matrix.shape == (36,):
        matrix = matrix.reshape(6, 6)
    elif matrix.shape != (6, 6):
        raise ValueError(
            f"{name} must have shape (6,), (36,), or (6, 6); got {matrix.shape}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} matrix must be symmetric")

    # Cartesian stiffness and damping must not inject energy.  Scale the
    # tolerance to permit the tiny roundoff introduced by R @ K @ R.T.
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} matrix must be positive semidefinite")

    # MotionUpdate declares float64[36] in row-major order.  A Python list is
    # accepted by generated ROS messages on all supported rclpy versions.
    return np.ascontiguousarray(matrix).reshape(36).tolist()


class Policy(ABC):
    def __init__(self, parent_node):
        self._parent_node = parent_node
        self.get_logger().info("Policy.__init__()")

    def get_logger(self):
        return self._parent_node.get_logger()

    def get_clock(self):
        return self._parent_node.get_clock()

    def time_now(self):
        """Return the current time from the node's clock (sim-time aware)."""
        return self.get_clock().now()

    def sleep_for(self, duration_sec: float) -> None:
        """Sleep for the given duration using the node's clock (sim-time aware)."""
        self.get_clock().sleep_for(Duration(seconds=duration_sec))

    def set_pose_target(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        frame_id: str = "base_link",
        stiffness: list = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
        damping: list = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
    ) -> None:
        """Invoke the move_robot callback to request the supplied Pose.

        This is a convenience function which populates a MotionUpdate message
        with a reasonable set of default parameters, and invokes the move_robot
        callback to request motion to the supplied pose.

        The robot can be controlled in several different ways. This function
        is intended to be the simplest way to move the arm around, by sending
        a desired pose (position and orientation) for the gripper's
        "tool control point" (TCP), which is the "pinch point" between the very
        end of the gripper fingers. The rest of the control stack will take care
        of moving all the arm's joints to so that the gripper TCP ends up in
        the desired position and orientation.

        The constants defined in this function are intended to provide
        reasonable default behavior if the arm is unable to achieve the
        requested pose. Different values for stiffness, damping, wrenches, and
        so on can be used for different types of arm behavior. These values
        are only intended to provide a starting point, and can be adjusted as
        desired.
        """
        target_stiffness = _cartesian_impedance_matrix(stiffness, "stiffness")
        target_damping = _cartesian_impedance_matrix(damping, "damping")
        motion_update = MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=target_stiffness,
            target_damping=target_damping,
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

    @abstractmethod
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Called when the insert_cable task is requested by aic_engine"""
        pass
