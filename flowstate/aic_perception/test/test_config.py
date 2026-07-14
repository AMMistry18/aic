from __future__ import annotations

import pytest

from aic_perception.config import (
    CAMERA_IMAGE_TOPICS,
    CAMERA_OPTICAL_FRAMES,
    CHANGE_TARGET_MODE_SERVICE,
    CONTROLLER_STATE_TOPIC,
    POSE_COMMAND_TOPIC,
    PerceptionConfig,
)


def test_configuration_has_closed_robot_camera_tf_allowlist():
    config = PerceptionConfig()
    assert config.base_frame == "base_link"
    assert config.gripper_frame == "gripper/tcp"
    assert config.image_topics == CAMERA_IMAGE_TOPICS
    assert config.camera_frames == CAMERA_OPTICAL_FRAMES
    assert config.pose_command_topic == POSE_COMMAND_TOPIC
    assert config.controller_state_topic == CONTROLLER_STATE_TOPIC
    assert config.change_target_mode_service == CHANGE_TARGET_MODE_SERVICE
    assert all(name.endswith("_camera/optical") for name in config.camera_frames.values())


def test_privileged_or_environment_frame_override_is_rejected():
    with pytest.raises(ValueError):
        PerceptionConfig(camera_frames={"center_camera": "task_board"})
    with pytest.raises(ValueError):
        PerceptionConfig(base_frame="world")
    with pytest.raises(ValueError):
        PerceptionConfig(gripper_frame="task_board")
    with pytest.raises(ValueError):
        PerceptionConfig(image_topics={"center_camera": "/gazebo/board_image"})
    with pytest.raises(ValueError):
        PerceptionConfig(wrench_topic="/scoring/wrench")
    with pytest.raises(ValueError):
        PerceptionConfig(pose_command_topic="/joint_trajectory_controller/command")
    with pytest.raises(ValueError):
        PerceptionConfig(controller_state_topic="/gazebo/entity_state")
    with pytest.raises(ValueError):
        PerceptionConfig(change_target_mode_service="/scoring/change_mode")
