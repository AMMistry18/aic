"""Offline harness for the paste-ready Flowstate ``filter_estimates_sc`` node.

Run from the repository root:

    python docs/reference/filter_estimates_sc_node_test.py

The node itself ends with a Flowstate-cell ``return``, so this harness executes
only its pure helper prefix and supplies minimal pose-estimate stand-ins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SOURCE = Path(__file__).with_name("filter_estimates_sc_node.py")
SOURCE_TEXT = SOURCE.read_text(encoding="utf-8")
HELPERS = SOURCE_TEXT.split(
    "# ---------------------------------------------------------------------------\n"
    "# Node body\n"
    "# ---------------------------------------------------------------------------",
    1,
)[0]
namespace: dict[str, object] = {}
exec(compile(HELPERS, str(SOURCE), "exec"), namespace)
fixed = SimpleNamespace(**namespace)


def assert_flowstate_output_contract():
  assert "output.sc_ports.append(best.root_t_target)" in SOURCE_TEXT
  assert "output.pose_estimates" not in SOURCE_TEXT
  assert "output.root_ts_target" not in SOURCE_TEXT


@dataclass
class _V:
  x: float
  y: float
  z: float


@dataclass
class _Q:
  x: float
  y: float
  z: float
  w: float


@dataclass
class _Pose:
  position: _V
  orientation: _Q


@dataclass
class _Est:
  root_t_target: _Pose
  score: float


PORTS_BOARD = np.array([
    [-0.120, 0.0295, 0.0301],
    [-0.075, 0.0295, 0.0301],
    [-0.030, 0.0295, 0.0301],
    [-0.100, 0.0705, 0.0301],
    [-0.045, 0.0705, 0.0301],
])


def rotation(yaw_deg, tilt_deg):
  yaw = math.radians(yaw_deg)
  tilt = math.radians(tilt_deg)
  cz, sz = math.cos(yaw), math.sin(yaw)
  cx, sx = math.cos(tilt), math.sin(tilt)
  rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
  rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
  return rx @ rz


def quaternion_from_matrix(matrix):
  matrix = np.asarray(matrix, dtype=float)
  w = math.sqrt(max(0.0, 1.0 + np.trace(matrix))) / 2.0
  x = math.copysign(
      math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) / 2.0,
      matrix[2, 1] - matrix[1, 2],
  )
  y = math.copysign(
      math.sqrt(max(0.0, 1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])) / 2.0,
      matrix[0, 2] - matrix[2, 0],
  )
  z = math.copysign(
      math.sqrt(max(0.0, 1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])) / 2.0,
      matrix[1, 0] - matrix[0, 1],
  )
  return _Q(x, y, z, w)


def estimate(xyz, matrix, score=0.9):
  return _Est(_Pose(_V(*xyz), quaternion_from_matrix(matrix)), score)


def detections(yaw_deg, tilt_deg=0.0, noise_m=0.0, seed=0):
  rng = np.random.default_rng(seed)
  matrix = rotation(yaw_deg, tilt_deg)
  origin = np.array([-0.3445, 0.2602, 0.0])
  return [
      estimate(matrix @ point + origin + rng.normal(0.0, noise_m, 3), matrix)
      for point in PORTS_BOARD
  ]


def confident_labels(estimates):
  confident = [
      item for item in estimates if float(item.score) >= fixed.MIN_SCORE
  ]
  labeled, ignored, _axes_from_orientation = fixed.positional_labels(
      fixed.deduplicate(confident)
  )
  return labeled, ignored


def assert_orientation_sweep():
  for tilt in (0.0, 8.0):
    for yaw in (0, 5, 10, 20, 45, 90, 140, 180, 250, 315):
      estimates = detections(yaw, tilt, noise_m=0.0015, seed=1)
      labeled, ignored = confident_labels(estimates)
      assert not ignored
      assert {
          label: estimates.index(item) for label, item in labeled
      } == {
          0: 0,
          1: 1,
          2: 2,
          3: 3,
          4: 4,
      }


def assert_non_alignment_never_rejects():
  estimates = detections(70, 8)
  # Destroy the nominal equal spacing, within-row alignment and 41 mm row
  # separation while preserving only positional order.
  distorted_board = np.array([
      [-0.150, 0.020, 0.020],
      [-0.082, 0.026, 0.038],
      [-0.018, 0.015, 0.028],
      [-0.130, 0.055, 0.012],
      [-0.030, 0.064, 0.045],
  ])
  matrix = rotation(70, 8)
  origin = np.array([-0.3445, 0.2602, 0.0])
  for estimate_item, point in zip(estimates, distorted_board):
    xyz = matrix @ point + origin
    estimate_item.root_t_target.position = _V(*xyz)
  labeled, ignored = confident_labels(estimates)
  assert not ignored
  assert {
      label: estimates.index(item) for label, item in labeled
  } == {
      0: 0,
      1: 1,
      2: 2,
      3: 3,
      4: 4,
  }


def assert_missing_slot_only_blocks_that_slot():
  estimates = detections(45, 8)
  estimates.pop(2)  # positional sc_port_2 is absent
  labeled, ignored = confident_labels(estimates)
  assert not ignored
  available = {label for label, _item in labeled}
  assert available == {0, 1, 3, 4}


def assert_background_scores_are_rejected():
  real = detections(0, 0)
  identity = np.eye(3)
  false = [
      estimate([0.2993, -0.0949, 1.1817], identity, 0.367),
      estimate([0.2991, -0.0448, 1.1848], identity, 0.363),
      estimate([0.1681, -0.1571, 1.0000], identity, 0.327),
  ]
  labeled, ignored = confident_labels(real + false)
  assert not ignored
  assert len(labeled) == 5
  assert all(item in real for _label, item in labeled)


if __name__ == "__main__":
  assert_flowstate_output_contract()
  assert_orientation_sweep()
  assert_non_alignment_never_rejects()
  assert_missing_slot_only_blocks_that_slot()
  assert_background_scores_are_rejected()
  print("filter_estimates_sc reference harness: PASS")
