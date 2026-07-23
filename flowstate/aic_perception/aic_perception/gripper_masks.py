"""Load and apply the calibrated per-camera gripper ignore masks.

The three small PNG payloads are the exact masks added on the upstream
``board-search`` branch. Embedding them keeps overlay-image deployments
self-contained while preserving the source calibration byte-for-byte.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .config import CAMERA_NAMES


_MASK_PNG_BASE64 = {
    "left_camera": (
        "iVBORw0KGgoAAAANSUhEUgAAAO8AAADVCAAAAACa+viuAAAC4UlEQVR4AeXBUY4kxwHFQPL+h6bt"
        "hQ1D2h4JUGV2fbwIY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYU"
        "Y4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOM"
        "KcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGm"
        "GFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhi"
        "TDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDGmGFOMKcYUY4oxxZhiTDG+yXiX8T3yb/Em"
        "42vkl3iR8SXyP/Ee4yvk/+I9xn3yR/Ea4zr5k3iNcZv8SbzHuEt+E+8xbpIP4j3GRfJJvMe4Rz6K"
        "9xj3yEfxHuMa+SheZFwjH8WLjGvkk3iTcY18EK8yrpHfxbuMW+R38TLjDvkkXmbcID+IdxkXyE/i"
        "vyReYJwnP4pf5JeQ+CbjPPlJ/CJ/EF9jHCc/il/kN/EVxnHyo/gP+SS+wDhO/kKAfBbXGcfJXwrk"
        "B3GZcZz8pUB+EncZp8nfCPlRXGUcJs/ETcZZ8lRcZBwlj8VFxklyQNxjnCQnxDXGQXJG3GKcI6fE"
        "JcYxckxcYhwj58QdxilyUlxhHCMnxQ3GKXJWXGAcI0fFBcYpclicZ5wip8VxxiFyXBxnHCLnxWnG"
        "GXJBnGacITfEYcYRckUcZpwgl8RZxglySZxlnCC3xFHGAXJPnGQcIPfEScZzclMcZDwnN8VBxnNy"
        "VZxjPCaXxTHGY3JZHGM8JdfFKcZTcl2cYjwl98UhxlNyXxxiPCVfEGcYD8k3xBnGM/IdcYTxjHxH"
        "HGE8It8SJxiPyNfEAcYT8j1xgPHPyVfFc8Y/J98UB/wLlQEoEPzBFkEAAAAASUVORK5CYII="
    ),
    "center_camera": (
        "iVBORw0KGgoAAAANSUhEUgAAAO8AAADVCAAAAACa+viuAAACjElEQVR4AeXBQQrYBhTFQOn+h1Yh"
        "lCZddJH42xTejDHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHF"
        "mGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRj"
        "ijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wp"
        "xhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYY"
        "U4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJMMaYYU4wpxhRjijHFmGJM"
        "MaYYU4wpxhRjijHFmGJMMaYYU4wpxhTj/8h4h/G/Iz/EC4x3GH9I/hb3jBfID/EH5B9xzrgnf4vf"
        "Jz/FNeOc/CN+m/wU14xr8lP8LvlFXDOuyU/xu+RXccy4Jv8W/yIQ/0l+FceMY3IojhnH5FAcM47J"
        "oThmHJNDccy4JZfimHFLLsUx45Qci1PGJTkXl4xD8oa4Y9yRl8QV44y8Ja4YZ+Q1ccS4Iu+JI8YV"
        "eVHcMK7Ii+KGcUTeFDeMI/KmuGEckTfFDeOGvCpuGDfkVXHDOCEvixPGCXlZnDAuyOvignFBXhcX"
        "jAvyurhgHJD3xQXjgHwgDhjPyRfigPGcfCKeM56TL8QB4zH5SDxmPCZfiaeMx+Qr8ZTxlHwnHjKe"
        "ku/EQ8ZT8p14yHhKPhNPGQ/Jh+Ih4yH5UDxkPCRfimeMh+RL8YzxjHwqnjGekW/FI8Yz8q14xHhG"
        "vhWP/AVwJhMQDjyLHgAAAABJRU5ErkJggg=="
    ),
    "right_camera": (
        "iVBORw0KGgoAAAANSUhEUgAAAO8AAADVCAAAAACa+viuAAACyElEQVR4AeXBW64YSQEFwcz9LzqR"
        "QEI8jBnfrur+OBHGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnG"
        "FGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhT"
        "jCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwx"
        "phhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWY"
        "YkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYkwxphhTjCnGFGOKMcWYYnxKiBcZH5K/i9cY"
        "H5J/iLcY35F/incY35F/EW8wviP/Ju4zviP/IW4zviP/Je4yviO/EDcZ35FfinuM78gvxT3Gd+TX"
        "4hrjO/JrcY3xDQn5H+IW4wvyW3GL8QX5rbjF+ID8XtxifEB+L24xPiD/V1xhvE/+irjAeJ38JXGB"
        "8Sb5A3Ge8Rr5M3Ge8RL5U3Ge8Qr5c3Ge8QL5iTjPuE5+KI4zbpOfiuOMu+SBOM24Sp6I04yb5Jk4"
        "zLhIHorDjHvksTjLuEaei7OMW+SAOMu4RE6Is4w75Ig4y7hDjoizjCvkkDjKuEFOiaOMG+SYOMm4"
        "QA6Kg4wL5KQ4xzhPzopjjPPksDjFOE6Oi0OM4+S8OMM4TW6II4zD5Io4wjhM7ogTjLPkkjjBOEqu"
        "iQOMo+SeeM44SW6Kx4yD5Kp4zDhIrorHjHPkrnjMOEZui6eMY+S2eMo4Ra6Lp4xT5Lp4yjhE7oun"
        "jEPkBfGQcYa8IR4yjpBXxEPGEfKOeMY4QV4SzxgnyEviGeMEeUs8Yhwg74knjAPkVfFjxgHyrvgp"
        "4zl5WfzU3wAcbh8Qk0y7egAAAABJRU5ErkJggg=="
    ),
}


class GripperMaskBank:
    """Resize the calibrated gripper silhouettes to each incoming stream."""

    def __init__(self) -> None:
        missing = set(CAMERA_NAMES) - set(_MASK_PNG_BASE64)
        if missing:
            raise ValueError(
                f"missing calibrated gripper masks: {sorted(missing)}"
            )
        self._source_masks: dict[str, np.ndarray] = {}
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}
        for camera in CAMERA_NAMES:
            encoded = base64.b64decode(
                _MASK_PNG_BASE64[camera], validate=True
            )
            mask = cv2.imdecode(
                np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                raise ValueError(
                    f"invalid embedded gripper mask for {camera}"
                )
            self._source_masks[camera] = mask

    def ignored_pixels(
        self, camera: str, image_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Return true for pixels occupied by the camera's own gripper."""

        if camera not in self._source_masks:
            raise KeyError(f"no gripper mask for camera {camera!r}")
        if len(image_shape) < 2:
            raise ValueError("image_shape must contain height and width")
        height, width = int(image_shape[0]), int(image_shape[1])
        if height < 2 or width < 2:
            raise ValueError(
                "image dimensions must both be at least two pixels"
            )
        key = (camera, height, width)
        cached = self._cache.get(key)
        if cached is None:
            mask = self._source_masks[camera]
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
            cached = mask == 0
            cached.setflags(write=False)
            self._cache[key] = cached
        return cached.copy()

    def apply(self, camera: str, image: np.ndarray) -> np.ndarray:
        """Paint calibrated gripper pixels background-white in a copy."""

        output = np.asarray(image).copy()
        ignored = self.ignored_pixels(camera, output.shape)
        if output.ndim == 2:
            output[ignored] = 255
        else:
            output[ignored] = (255,) * output.shape[2]
        return output

