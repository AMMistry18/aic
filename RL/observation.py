"""
Observation builder for the residual SAC last-inch policy.

Turns a ROS `aic_model_interfaces.msg.Observation` (plus a known port pose)
into a flat dict that Stable-Baselines3's `CombinedExtractor` can consume.

The output mirrors REWARD_SPEC.md §4:
    image           (H, W, 3) uint8     three stacked 32x32 wrist cams
    force           (3,)      float32    wrist F/T (xyz), N
    tcp_pose        (7,)      float32    gripper/tcp pose in base_link
    port_pose       (7,)      float32    port entrance pose in base_link
    tcp_pose_err    (7,)      float32    tcp_pose - port_pose (port frame)
    last_action     (6,)      float32    previous residual action

The image is the *raw* AIC wrist-cam output resized to HxW (default 32x32
grayscale, stacked to 3 channels by repeating — keeps the channel count
stable when switching to 64x64 colour later).

`force` is read from `obs.wrist_wrench.wrench.force.{x,y,z}` (no F/T
smoothing here — that's the env's job; we want the raw value the AIC
scoring uses so the policy learns the same noise distribution it will
see at eval).

For the *training* env (MuJoCo) we bypass the ROS `Observation` and call
`build_obs_dict_from_arrays(...)` directly with numpy arrays produced by
the MuJoCo renderer. The two builders share the dict schema so the same
SB3 policy consumes both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

try:
    import cv2  # only needed for the ROS-image path; absent in pure-mujoco training
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObsConfig:
    image_h: int = 32
    image_w: int = 32
    n_cams: int = 3                  # left/centre/right stacked
    image_ch_per_cam: int = 3        # RGB (set 1 to stack 3× grayscale)
    pos_scale: tuple = (0.0015, 0.0015, 0.0035)
    rot_scale: tuple = (0.08, 0.08, 0.12)

    @property
    def image_channels(self) -> int:
        return self.n_cams * self.image_ch_per_cam


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ros_image_to_np(img_msg, target_hw: tuple[int, int]) -> np.ndarray:
    """Convert a sensor_msgs/Image to a (H, W) uint8 grayscale array."""
    if img_msg is None:
        return np.zeros(target_hw, dtype=np.uint8)
    arr = np.frombuffer(img_msg.data, dtype=np.uint8)
    if img_msg.encoding in ("mono8", "8UC1"):
        img = arr.reshape(img_msg.height, img_msg.width)
    else:
        # bgr8 / rgb8 — collapse to luminance via BT.601 weights
        if not _HAS_CV2:
            img = arr.reshape(img_msg.height, img_msg.width, 3).mean(axis=2).astype(np.uint8)
        else:
            bgr = arr.reshape(img_msg.height, img_msg.width, 3)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY if img_msg.encoding == "bgr8" else cv2.COLOR_RGB2GRAY)
    if img.shape != target_hw:
        if not _HAS_CV2:
            # simple nearest-neighbour fallback (no cv2 in some envs)
            ys = (np.linspace(0, img.shape[0] - 1, target_hw[0]) + 0.5).astype(int)
            xs = (np.linspace(0, img.shape[1] - 1, target_hw[1]) + 0.5).astype(int)
            img = img[ys[:, None], xs[None, :]]
        else:
            img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _stack_cams(images: list[np.ndarray], n_cams: int, target_hw: tuple[int, int]) -> np.ndarray:
    """Stack N grayscale images along the channel axis → (H, W, N) uint8."""
    out = np.zeros((target_hw[0], target_hw[1], n_cams), dtype=np.uint8)
    for i in range(n_cams):
        if i < len(images) and images[i] is not None:
            out[:, :, i] = images[i]
    return out


def _pose_err(tcp_xyz: np.ndarray, tcp_q: np.ndarray,
              port_xyz: np.ndarray, port_q: np.ndarray) -> np.ndarray:
    """Pose difference in the port frame: (Δxyz_port, Δaxis_angle_port).

    Returns a 7-vector: [dx, dy, dz, ax, ay, az] (axis-angle of the
    relative rotation, magnitude = angle in radians).
    """
    # numpy-only minimal quat math (no transforms3d import — keeps this
    # file usable in any env that has just numpy).
    def q_normalize(q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])

    def q_mult(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    def q_to_R(q):
        w, x, y, z = q_normalize(q)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])

    R_port = q_to_R(port_q)
    delta_xyz_world = tcp_xyz - port_xyz
    delta_xyz_port = R_port.T @ delta_xyz_world

    # relative rotation = q_port^-1 * q_tcp  → axis-angle
    q_port_inv = np.array([port_q[0], -port_q[1], -port_q[2], -port_q[3]])
    q_rel = q_mult(q_port_inv, q_normalize(tcp_q))
    q_rel = q_normalize(q_rel)
    # axis-angle from quaternion
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-6:
        axis = np.zeros(3)
    else:
        axis = q_rel[1:] / s
    return np.concatenate([delta_xyz_port, axis * angle]).astype(np.float32)


# --------------------------------------------------------------------------- #
# main builders
# --------------------------------------------------------------------------- #


def build_obs_dict_from_ros(
    obs_msg,
    port_xyz: np.ndarray,
    port_q_wxyz: np.ndarray,
    last_action: np.ndarray,
    cfg: ObsConfig = ObsConfig(),
) -> dict:
    """Build the obs dict from a live AIC `Observation` message + port pose.

    Image convention matches `ObsConfig.image_ch_per_cam`:
        ch=3 (RGB) — left/centre/right 3-channel stack → (H, W, 9)
        ch=1 (grey) — left/centre/right grayscale stack → (H, W, 3)
    """
    target_hw = (cfg.image_h, cfg.image_w)

    # images — left/centre/right stacked. Fill missing with zeros.
    cams = [getattr(obs_msg, "left_image", None),
            getattr(obs_msg, "center_image", None),
            getattr(obs_msg, "right_image", None)]
    imgs_gray = [_ros_image_to_np(c, target_hw) for c in cams]
    # Stack as (H, W, n_cams*ch) — if RGB, each cam contributes 3 chans.
    target_ch = cfg.image_channels
    if cfg.image_ch_per_cam == 3:
        # need to actually decode RGB from each ROS image
        imgs_rgb = []
        for cam_msg, gray in zip(cams, imgs_gray):
            if cam_msg is None or not _HAS_CV2:
                # fall back to replicating the grayscale
                imgs_rgb.append(np.stack([gray] * 3, axis=-1))
            else:
                arr = np.frombuffer(cam_msg.data, dtype=np.uint8)
                if cam_msg.encoding in ("mono8", "8UC1"):
                    rgb = np.stack([gray] * 3, axis=-1)
                else:
                    raw = arr.reshape(cam_msg.height, cam_msg.width, 3)
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB if cam_msg.encoding == "bgr8" else cv2.COLOR_RGB2BGR)
                    # resize if needed
                    if rgb.shape[:2] != target_hw:
                        rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
                imgs_rgb.append(rgb.astype(np.uint8))
        # Concatenate along channel axis → (H, W, 9)
        image = np.concatenate(imgs_rgb, axis=2)
    else:
        # grayscale — just stack → (H, W, 3)
        image = _stack_cams(imgs_gray, cfg.n_cams, target_hw)
    # final shape check
    if image.shape[2] != target_ch:
        raise ValueError(f"image channel count {image.shape[2]} != expected {target_ch}")

    # wrench — only force xyz, normalised to N
    force = np.zeros(3, dtype=np.float32)
    ww = getattr(obs_msg, "wrist_wrench", None)
    if ww is not None and getattr(ww, "wrench", None) is not None:
        w = ww.wrench
        force[:] = [w.force.x, w.force.y, w.force.z]

    # TCP pose — read from TF on the caller side and pass in via port_*? No:
    # this builder is pure-function of the Observation msg + port pose. We
    # expect the caller (the env wrapper) to populate tcp_xyz/q via TF
    # lookup and stash it on the msg if available. As a fallback we use
    # zeros and let the env fill them in.
    tcp_xyz = np.zeros(3, dtype=np.float32)
    tcp_q = np.array([1.0, 0, 0, 0], dtype=np.float32)
    if hasattr(obs_msg, "tcp_pose"):
        tp = obs_msg.tcp_pose
        if tp is not None:
            tcp_xyz[:] = [tp.position.x, tp.position.y, tp.position.z]
            tcp_q[:] = [tp.orientation.w, tp.orientation.x, tp.orientation.y, tp.orientation.z]

    tcp_pose_err = _pose_err(tcp_xyz, tcp_q, port_xyz, port_q_wxyz)
    tcp_pose = np.concatenate([tcp_xyz, tcp_q]).astype(np.float32)
    port_pose = np.concatenate([port_xyz, port_q_wxyz]).astype(np.float32)

    return {
        "image": image,
        "force": force,
        "tcp_pose": tcp_pose,
        "port_pose": port_pose,
        "tcp_pose_err": tcp_pose_err,
        "last_action": np.asarray(last_action, dtype=np.float32).reshape(-1),
    }


def build_obs_dict_from_arrays(
    image: np.ndarray,
    tcp_xyz: np.ndarray,
    tcp_q_wxyz: np.ndarray,
    port_xyz: np.ndarray,
    port_q_wxyz: np.ndarray,
    force_xyz: np.ndarray,
    last_action: np.ndarray,
    cfg: ObsConfig = ObsConfig(),
) -> dict:
    """Build the obs dict from raw numpy arrays (MuJoCo training path).

    `image` is one of:
        (H, W)         grayscale (will be replicated to n_cams × 3 chans)
        (H, W, 3)      RGB from a single cam (will be replicated)
        (H, W, n_cams) grayscale from each cam (will be repeated 3× for RGB)
        (H, W, n_cams*3) full stacked RGB from all cams
    """
    target_ch = cfg.image_channels
    if image.ndim == 2:
        # (H, W) grayscale → replicate across n_cams × image_ch_per_cam
        if image.shape != (cfg.image_h, cfg.image_w):
            ys = (np.linspace(0, image.shape[0] - 1, cfg.image_h) + 0.5).astype(int)
            xs = (np.linspace(0, image.shape[1] - 1, cfg.image_w) + 0.5).astype(int)
            image = image[ys[:, None], xs[None, :]]
        image = np.repeat(image[:, :, None], target_ch, axis=2)
    elif image.ndim == 3:
        if image.shape[2] == 3 and target_ch == cfg.n_cams * 3:
            # single RGB cam → replicate across n_cams
            image = np.repeat(image[:, :, :3], cfg.n_cams, axis=2)
        elif image.shape[2] == cfg.n_cams and target_ch == cfg.n_cams * 3:
            # n_cams grayscale → repeat each channel 3×
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] != target_ch:
            raise ValueError(
                f"image channel count {image.shape[2]} != expected {target_ch}"
            )

    tcp_pose_err = _pose_err(tcp_xyz, tcp_q_wxyz, port_xyz, port_q_wxyz)
    tcp_pose = np.concatenate([tcp_xyz, tcp_q_wxyz]).astype(np.float32)
    port_pose = np.concatenate([port_xyz, port_q_wxyz]).astype(np.float32)

    return {
        "image": image.astype(np.uint8),
        "force": np.asarray(force_xyz, dtype=np.float32).reshape(3),
        "tcp_pose": tcp_pose,
        "port_pose": port_pose,
        "tcp_pose_err": tcp_pose_err,
        "last_action": np.asarray(last_action, dtype=np.float32).reshape(-1),
    }


__all__ = [
    "ObsConfig",
    "build_obs_dict_from_ros",
    "build_obs_dict_from_arrays",
]