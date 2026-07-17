"""Load and apply fixed per-camera gripper ignore masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MASK_FILES = {
    "left_camera": "left_mask.png",
    "center_camera": "center_mask.png",
    "right_camera": "right_mask.png",
}
CAMERA_NAMES = tuple(MASK_FILES)


def mask_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here / "masks", here.parent / "masks"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("gripper mask directory not found")


class GripperMaskBank:
    def __init__(self) -> None:
        root = mask_dir()
        self._masks: dict[str, np.ndarray] = {}
        for camera in CAMERA_NAMES:
            path = root / MASK_FILES[camera]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"missing gripper mask: {path}")
            self._masks[camera] = mask

    def apply(self, camera: str, image: np.ndarray) -> np.ndarray:
        if camera not in self._masks:
            raise KeyError(f"no gripper mask for camera {camera!r}")
        mask = self._masks[camera]
        height, width = image.shape[:2]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        output = np.asarray(image, copy=True)
        ignored = mask == 0
        if output.ndim == 2:
            output[ignored] = 255
        else:
            output[ignored] = (255, 255, 255)
        return output

    def ignored_pixels(self, camera: str, image_shape: tuple[int, ...]) -> np.ndarray:
        """Boolean mask of ignored pixels, resized to ``image_shape``."""
        if camera not in self._masks:
            raise KeyError(f"no gripper mask for camera {camera!r}")
        mask = self._masks[camera]
        height, width = image_shape[:2]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask == 0
