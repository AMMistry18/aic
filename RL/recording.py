"""
Recording format for the AIC last-inch residual-SAC trainer.

A recording is a single NumPy .npz file containing one episode from the
AIC engine / Gazebo sim. It captures everything the residual SAC needs
to learn the last-inch insertion:
    - 3 wrist camera frames (left/center/right, 32x32 RGB)
    - wrist force/torque
    - TCP pose and port pose
    - the action PerceptionInsert would have taken
    - the engine's /scoring/insertion_event (binary, latched)
    - the task success flag
    - the port type

The training env (RL/recorded_env.py) consumes these recordings and
lets the policy learn a residual correction offline — no MuJoCo
contact dynamics, no sim-to-real gap.

Recording API (used by the AIC sim while running PerceptionInsert):
    from RL.recording import Recorder
    r = Recorder(port_type="sc")
    r.start_episode(task)
    for obs_msg, action_6d, port_pose, insertion_event in roll_loop:
        r.append(obs_msg, action_6d, port_pose, insertion_event)
    r.finish_episode(success=True)
    r.save("outputs/rollouts/sc_run001.npz")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


class EpisodeRecorder:
    """Buffer one episode's worth of (obs, action, port_pose, event)."""

    def __init__(self, port_type: str, image_h: int = 32, image_w: int = 32,
                 n_cams: int = 3, image_ch_per_cam: int = 3):
        self.port_type = port_type
        self.image_h = image_h
        self.image_w = image_w
        self.n_cams = n_cams
        self.image_ch_per_cam = image_ch_per_cam
        self._frames: list[dict] = []

    def append(self, *, image: np.ndarray, force_xyz: np.ndarray,
               tcp_xyz: np.ndarray, tcp_q_wxyz: np.ndarray,
               port_xyz: np.ndarray, port_q_wxyz: np.ndarray,
               tcp_pose_err: np.ndarray, action_6d: np.ndarray,
               insertion_event: bool) -> None:
        """Append one frame. Image must be (H, W, n_cams*ch) uint8."""
        assert image.dtype == np.uint8, f"image must be uint8, got {image.dtype}"
        assert image.shape == (self.image_h, self.image_w,
                               self.n_cams * self.image_ch_per_cam)
        self._frames.append({
            "image": image,
            "force": np.asarray(force_xyz, dtype=np.float32).reshape(3),
            "tcp_xyz": np.asarray(tcp_xyz, dtype=np.float32).reshape(3),
            "tcp_q": np.asarray(tcp_q_wxyz, dtype=np.float32).reshape(4),
            "port_xyz": np.asarray(port_xyz, dtype=np.float32).reshape(3),
            "port_q": np.asarray(port_q_wxyz, dtype=np.float32).reshape(4),
            "tcp_pose_err": np.asarray(tcp_pose_err, dtype=np.float32).reshape(-1),
            "action": np.asarray(action_6d, dtype=np.float32).reshape(6),
            "insertion_event": bool(insertion_event),
        })

    def __len__(self) -> int:
        return len(self._frames)

    def save(self, path: str) -> None:
        """Dump the buffered episode to a .npz file."""
        if not self._frames:
            raise ValueError("no frames to save")
        n = len(self._frames)
        H, W, C = self.image_h, self.image_w, self.n_cams * self.image_ch_per_cam
        out = {
            "port_type": np.array(self.port_type),
            "image_h": np.array(self.image_h, dtype=np.int32),
            "image_w": np.array(self.image_w, dtype=np.int32),
            "n_cams": np.array(self.n_cams, dtype=np.int32),
            "image_ch_per_cam": np.array(self.image_ch_per_cam, dtype=np.int32),
            "task_success": np.array(any(f["insertion_event"] for f in self._frames)),
            "images": np.stack([f["image"] for f in self._frames], axis=0).astype(np.uint8),
            "forces": np.stack([f["force"] for f in self._frames], axis=0).astype(np.float32),
            "tcp_xyzs": np.stack([f["tcp_xyz"] for f in self._frames], axis=0).astype(np.float32),
            "tcp_qs": np.stack([f["tcp_q"] for f in self._frames], axis=0).astype(np.float32),
            "port_xyzs": np.stack([f["port_xyz"] for f in self._frames], axis=0).astype(np.float32),
            "port_qs": np.stack([f["port_q"] for f in self._frames], axis=0).astype(np.float32),
            "tcp_pose_errs": np.stack([f["tcp_pose_err"] for f in self._frames], axis=0).astype(np.float32),
            "actions": np.stack([f["action"] for f in self._frames], axis=0).astype(np.float32),
            "insertion_event": np.array(
                [f["insertion_event"] for f in self._frames], dtype=np.int8
            ),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path: str) -> dict:
        """Read a .npz file back into the same dict shape."""
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}


def make_synthetic_episode(
    port_type: str = "sc",
    n_frames: int = 200,
    seed: int = 42,
    image_h: int = 32,
    image_w: int = 32,
    n_cams: int = 3,
    image_ch_per_cam: int = 3,
    success: bool = True,
) -> dict:
    """Generate a fake episode for unit tests / dev without a real sim.

    The trajectory interpolates from a noisy start pose to a
    well-aligned goal pose over `n_frames` steps, with the
    `insertion_event` flag firing at the last frame if `success=True`.
    """
    rng = np.random.default_rng(seed)
    C = n_cams * image_ch_per_cam
    # start: random xy_err, slightly above port entrance
    start_xy = rng.uniform(-0.004, 0.004, size=2)
    start_z = rng.uniform(-0.001, 0.0005)
    # goal: aligned with port, seated at the insertion depth
    insertion_depth = 0.016 if port_type == "sc" else 0.051
    goal_z = -insertion_depth * 0.95
    # interpolate
    tcp_xyz = np.zeros((n_frames, 3), dtype=np.float32)
    tcp_xyz[:, 0] = np.linspace(start_xy[0], 0.0, n_frames) + rng.normal(0, 0.0005, n_frames)
    tcp_xyz[:, 1] = np.linspace(start_xy[1], 0.0, n_frames) + rng.normal(0, 0.0005, n_frames)
    tcp_xyz[:, 2] = np.linspace(start_z, goal_z, n_frames) + rng.normal(0, 0.0002, n_frames)
    tcp_q = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (n_frames, 1))
    port_xyz = np.zeros((n_frames, 3), dtype=np.float32)
    port_q = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (n_frames, 1))
    tcp_pose_err = np.zeros((n_frames, 6), dtype=np.float32)
    tcp_pose_err[:, :3] = port_xyz - tcp_xyz  # negative since tcp_xyz > port_xyz
    actions = np.zeros((n_frames, 6), dtype=np.float32)
    actions[:, 2] = -0.7  # press down
    # forces: 0 except for a contact bump in the last quarter
    forces = np.zeros((n_frames, 3), dtype=np.float32)
    if success:
        contact_start = int(n_frames * 0.6)
        forces[contact_start:, 2] = np.linspace(2.0, 8.0, n_frames - contact_start)
    # images: synthesize a moving "port" pattern + plug overlay
    # Goal image: plug seated at port centre (a circle in the middle)
    goal_img = _render_goal_image(image_h, image_w, C, port_type)
    images = np.zeros((n_frames, image_h, image_w, C), dtype=np.uint8)
    for t in range(n_frames):
        # interpolate from "no plug" to "goal"
        alpha = t / max(n_frames - 1, 1)
        if success:
            images[t] = (goal_img * alpha + _render_empty_image(image_h, image_w, C) * (1 - alpha)).astype(np.uint8)
        else:
            # freeze near the start (failed attempt)
            images[t] = (_render_empty_image(image_h, image_w, C) * 0.7 +
                         goal_img * 0.3).astype(np.uint8)
    insertion_event = np.zeros(n_frames, dtype=np.int8)
    if success:
        insertion_event[-1] = 1
    return {
        "port_type": np.array(port_type),
        "image_h": np.array(image_h, dtype=np.int32),
        "image_w": np.array(image_w, dtype=np.int32),
        "n_cams": np.array(n_cams, dtype=np.int32),
        "image_ch_per_cam": np.array(image_ch_per_cam, dtype=np.int32),
        "task_success": np.array(success),
        "images": images,
        "forces": forces,
        "tcp_xyzs": tcp_xyz,
        "tcp_qs": tcp_q,
        "port_xyzs": port_xyz,
        "port_qs": port_q,
        "tcp_pose_errs": tcp_pose_err,
        "actions": actions,
        "insertion_event": insertion_event,
    }


def _render_empty_image(H: int, W: int, C: int) -> np.ndarray:
    """Synthetic 'no plug' image: a uniform gray background."""
    img = np.full((H, W, C), 128, dtype=np.uint8)
    # add a small dark "port" rectangle in the middle
    cv = H // 2
    ch = W // 2
    img[cv-3:cv+3, ch-3:ch+3] = 50
    return img


def _render_goal_image(H: int, W: int, C: int, port_type: str) -> np.ndarray:
    """Synthetic 'plug seated' image: a bright disc in the middle of the port."""
    img = _render_empty_image(H, W, C).astype(np.float32)
    cv = H // 2
    ch = W // 2
    r = 3 if port_type == "sc" else 4
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cv) ** 2 + (xx - ch) ** 2 <= r * r
    img[mask] = 220
    return img.astype(np.uint8)


__all__ = [
    "EpisodeRecorder",
    "make_synthetic_episode",
]