import re

import numpy as np


# Recent physical SC detections are 0.77-0.91; observed remote background
# detections are 0.23-0.37. Keep them out before positional ordering.
MIN_SCORE = 0.4
DUPLICATE_RADIUS_M = 0.010

# Positional numbering in the signed board frame:
#   lower board-Y row, increasing board X -> sc_port_0/1/2
#   upper board-Y row, increasing board X -> sc_port_3/4
#
# There are deliberately NO alignment, rail-spacing, coplanarity, row-count,
# minimum-gap, or ambiguity gates. These signs are the only deployment knobs.
ALONG_RAIL_SIGN = 1.0
BETWEEN_RAIL_SIGN = 1.0
PORT_LABELS_BY_ROW = ((0, 1, 2), (3, 4))


def position(estimate):
  p = estimate.root_t_target.position
  return np.array([p.x, p.y, p.z], dtype=float)


def rotation_of(estimate):
  """Detection orientation as a 3x3 matrix, or None if unavailable."""
  pose = estimate.root_t_target
  quat = getattr(pose, "orientation", None)
  if quat is None:
    return None
  w = float(getattr(quat, "w", 0.0))
  x = float(getattr(quat, "x", 0.0))
  y = float(getattr(quat, "y", 0.0))
  z = float(getattr(quat, "z", 0.0))
  norm = (w * w + x * x + y * y + z * z) ** 0.5
  if norm < 1e-9:
    return None
  w, x, y, z = w / norm, x / norm, y / norm, z / norm
  return np.array(
      [
          [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
      ],
      dtype=float,
  )


def board_frame(estimates):
  """Average the detections' signed board frames in root coordinates."""
  mats = [r for r in (rotation_of(item) for item in estimates) if r is not None]
  if not mats:
    return None
  u, _s, vt = np.linalg.svd(sum(mats))
  frame = u @ vt
  if np.linalg.det(frame) < 0.0:
    u[:, -1] *= -1.0
    frame = u @ vt
  return frame


def deduplicate(estimates):
  """Keep the highest-score estimate at each physical position."""
  selected = []
  for estimate in sorted(
      estimates, key=lambda item: float(item.score), reverse=True
  ):
    xyz = position(estimate)
    if all(
        np.linalg.norm(xyz - position(existing)) >= DUPLICATE_RADIUS_M
        for existing in selected
    ):
      selected.append(estimate)
  return selected


def diagnostics(estimates):
  rows = []
  for estimate in estimates:
    xyz_mm = position(estimate) * 1000.0
    rows.append({
        "score": round(float(estimate.score), 3),
        "xyz_mm": [round(float(value), 1) for value in xyz_mm],
    })
  return rows


def positional_labels(candidates):
  """Assign labels only by board-relative position; never validate alignment."""
  if not candidates:
    return [], [], False

  frame = board_frame(candidates)
  axes_from_orientation = frame is not None
  if frame is None:
    # Fallback retains the original root-X/root-Y positional behavior. It does
    # not introduce a geometry failure if orientations are unavailable.
    along = np.array([1.0, 0.0, 0.0])
    between = np.array([0.0, 1.0, 0.0])
  else:
    along = frame[:, 0]
    between = frame[:, 1]
  along = along * ALONG_RAIL_SIGN
  between = between * BETWEEN_RAIL_SIGN

  ordered_between = sorted(
      candidates, key=lambda item: float(position(item) @ between)
  )
  if len(ordered_between) == 1:
    rows = (ordered_between, [])
  else:
    coords = np.array(
        [position(item) @ between for item in ordered_between], dtype=float
    )
    # Pure positional split. The gap size and resulting row counts are not
    # validated; this is only used to order the requested slots.
    cut = int(np.argmax(np.diff(coords))) + 1
    rows = (ordered_between[:cut], ordered_between[cut:])

  sorted_rows = tuple(
      sorted(row, key=lambda item: float(position(item) @ along))
      for row in rows
  )
  labeled = []
  ignored = []
  for labels, row in zip(PORT_LABELS_BY_ROW, sorted_rows):
    labeled.extend(zip(labels, row[:len(labels)]))
    ignored.extend(row[len(labels):])
  return labeled, ignored, axes_from_orientation


# ---------------------------------------------------------------------------
# Node body
# ---------------------------------------------------------------------------
output = code_execution_pb2.ReturnValue()

target_name = params.selected_module_name.strip()
match = re.fullmatch(r"sc_port_([0-4])", target_name)
if match is None:
  raise RuntimeError(
      f"Invalid selected_module_name {target_name!r}; "
      "expected sc_port_0 through sc_port_4"
  )
target_idx = int(match.group(1))

confident = [
    estimate
    for estimate in params.pose_estimates
    if float(estimate.score) >= MIN_SCORE
]
candidates = deduplicate(confident)
labeled, ignored, axes_from_orientation = positional_labels(candidates)

matches = [estimate for label, estimate in labeled if label == target_idx]
if not matches:
  raise RuntimeError(
      f"No detection occupied positional slot {target_name}. "
      f"Available labels={[f'sc_port_{label}' for label, _ in labeled]}. "
      f"Candidates={diagnostics(candidates)}"
  )
best = max(matches, key=lambda item: float(item.score))

print(
    "SC-port positional selection:",
    f"requested={target_name}",
    f"total_estimates={len(params.pose_estimates)}",
    f"score_filtered={len(confident)}",
    f"physical_candidates={len(candidates)}",
    f"axes_from_orientation={axes_from_orientation}",
    f"labels={[(f'sc_port_{label}', diagnostics([item])[0]) for label, item in labeled]}",
    f"ignored={diagnostics(ignored)}",
    f"selected_xyz_mm={[round(float(v), 1) for v in position(best) * 1000.0]}",
)

# This Flowstate node's ReturnValue schema exposes repeated Pose field
# ``sc_ports``. Append only the requested pose so the downstream belief step
# creates exactly one object, not one belief per detected port.
output.sc_ports.append(best.root_t_target)
return output
