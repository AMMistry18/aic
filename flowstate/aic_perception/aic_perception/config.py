"""Runtime configuration and the explicit AIC data-access allowlist."""

from __future__ import annotations

from dataclasses import dataclass, field


CAMERA_NAMES = ("left_camera", "center_camera", "right_camera")

# These maps are intentionally closed. The skill accepts no topic or TF-frame
# override, so a process author cannot point it at a board, port, simulation,
# scoring, or other privileged source.
CAMERA_IMAGE_TOPICS = {
    "left_camera": "/left_camera/image",
    "center_camera": "/center_camera/image",
    "right_camera": "/right_camera/image",
}
CAMERA_OPTICAL_FRAMES = {
    "left_camera": "left_camera/optical",
    "center_camera": "center_camera/optical",
    "right_camera": "right_camera/optical",
}
POSE_COMMAND_TOPIC = "/aic_controller/pose_commands"
JOINT_COMMAND_TOPIC = "/aic_controller/joint_commands"
JOINT_STATE_TOPIC = "/joint_states"
CONTROLLER_STATE_TOPIC = "/aic_controller/controller_state"
CHANGE_TARGET_MODE_SERVICE = "/aic_controller/change_target_mode"


@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration restricted to documented participant interfaces."""

    base_frame: str = "base_link"
    gripper_frame: str = "gripper/tcp"
    wrench_topic: str = "/fts_broadcaster/wrench"
    pose_command_topic: str = POSE_COMMAND_TOPIC
    joint_command_topic: str = JOINT_COMMAND_TOPIC
    joint_state_topic: str = JOINT_STATE_TOPIC
    controller_state_topic: str = CONTROLLER_STATE_TOPIC
    change_target_mode_service: str = CHANGE_TARGET_MODE_SERVICE
    cameras: tuple[str, ...] = CAMERA_NAMES
    image_topics: dict[str, str] = field(
        default_factory=lambda: dict(CAMERA_IMAGE_TOPICS)
    )
    camera_frames: dict[str, str] = field(
        default_factory=lambda: dict(CAMERA_OPTICAL_FRAMES)
    )

    def __post_init__(self) -> None:
        if self.base_frame != "base_link":
            raise ValueError("target frame is fixed to the permitted base_link frame")
        if self.gripper_frame != "gripper/tcp":
            raise ValueError("TCP frame is fixed to the permitted gripper/tcp frame")
        if self.wrench_topic != "/fts_broadcaster/wrench":
            raise ValueError("wrench topic may not be overridden")
        if self.pose_command_topic != POSE_COMMAND_TOPIC:
            raise ValueError("pose command topic may not be overridden")
        if self.joint_command_topic != JOINT_COMMAND_TOPIC:
            raise ValueError("joint command topic may not be overridden")
        if self.joint_state_topic != JOINT_STATE_TOPIC:
            raise ValueError("joint state topic may not be overridden")
        if self.controller_state_topic != CONTROLLER_STATE_TOPIC:
            raise ValueError("controller state topic may not be overridden")
        if self.change_target_mode_service != CHANGE_TARGET_MODE_SERVICE:
            raise ValueError("target-mode service may not be overridden")
        if tuple(self.cameras) != CAMERA_NAMES:
            raise ValueError("camera set must match the three documented wrist cameras")
        if self.image_topics != CAMERA_IMAGE_TOPICS:
            raise ValueError("camera topic allowlist may not be overridden")
        if self.camera_frames != CAMERA_OPTICAL_FRAMES:
            raise ValueError("camera TF allowlist may not be overridden")
