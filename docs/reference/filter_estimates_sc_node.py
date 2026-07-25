import re
from itertools import combinations

import numpy as np


MIN_SCORE = 0.4
NUM_PORTS = 5

# Two SC rails, nominally 41 mm apart (board Y +0.0295 and +0.0705), each
# carrying ports that slide along the rail over board X -0.135..-0.020.
EXPECTED_RAIL_SEPARATION_M = 0.041
RAIL_SEPARATION_TOL_M = 0.015

# Detections assigned to one physical rail must agree on their between-rail
# coordinate. Measured in the *recovered board frame*, so this is a real
# perpendicular spread and no longer inflated by board yaw.
MAX_WITHIN_RAIL_SPREAD_M = 0.012
# Two adapters on one rail are only 9.3 mm wide in board X, so they can sit far
# closer than the 18 mm this used to demand -- at 18 mm a legitimately bunched
# rail is rejected outright once triangulation noise nudges a gap under the
# floor. Keep it just above DUPLICATE_RADIUS_M: below that the two detections
# would already have been merged, above it they are distinct ports.
MIN_ALONG_RAIL_SEPARATION_M = 0.012

# Must stay below MIN_ALONG_RAIL_SEPARATION_M or two genuinely adjacent ports
# get merged into one and the 5-port fit can never be satisfied.
DUPLICATE_RADIUS_M = 0.010

# The five ports are coplanar on the board face; a detection far off that plane
# is a false positive, not a port.
MAX_PLANE_RESIDUAL_M = 0.015

# ---------------------------------------------------------------------------
# NUMBERING (Phase 1 board, per the task-board diagram).
#
# SC_PORT_0/1/2 are on SC_RAIL_0 (the three-port rail, board Y +0.0295) and
# SC_PORT_3/4 on SC_RAIL_1 (the two-port rail, board Y +0.0705), with the index
# increasing the same way along both rails.
#
# Which rail carries three ports is decided by COUNT, not by a coordinate sign,
# so board rotation cannot swap the rails.  The direction the index runs ALONG
# each rail cannot be derived from the port pattern -- five ports in two rows
# are symmetric end-for-end -- so it comes from the board frame the detections
# report.  Its sign is the one bit that cannot be verified from geometry alone:
# if a run lands on the port at the wrong END of the correct rail, flip
# ALONG_RAIL_SIGN and change nothing else.
# ---------------------------------------------------------------------------
PORT_LABELS_BY_RAIL = ((0, 1, 2), (3, 4))
ALONG_RAIL_SIGN = 1.0


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
  """Signed board axes in root frame, from the detections' own orientations.

  Every detection is of a component rigidly fixed to the board, so they all
  carry (noisy copies of) the board orientation. Averaging the rotation
  matrices and re-orthonormalising with SVD gives a proper rotation whose
  columns are board-relative directions -- unlike a fixed world axis, this
  tracks the board wherever it is placed or however it is turned.

  Returns the 3x3 matrix, or None when no detection carries an orientation.
  """
  mats = [r for r in (rotation_of(item) for item in estimates) if r is not None]
  if not mats:
    return None
  u, _s, vt = np.linalg.svd(sum(mats))
  frame = u @ vt
  if np.linalg.det(frame) < 0.0:  # reflection -> flip the least-certain axis
    u[:, -1] *= -1.0
    frame = u @ vt
  return frame


def plane_axes(points):
  """In-plane directions of the best-fit plane through the port centres.

  Used only to decide *which* board axis runs along the rails; the signs come
  from the detection orientation, which a point cloud cannot supply.
  """
  centred = points - points.mean(axis=0)
  _u, _s, vt = np.linalg.svd(centred)
  return vt[0], vt[1], vt[2]  # primary, secondary, normal


def axis_candidates(candidates):
  """The three board axes, in root frame, to try as the between-rail direction.

  These come from the **detections' own orientations**, not from the shape of
  the point cloud.  That matters: a PCA of the port centres is skewed by any
  false-positive detection, and its axes stop lining up with the rails whenever
  the ports happen to bunch along the rail (a 36 mm along-rail span against the
  41 mm rail separation makes the principal axes diagonal).  The board
  orientation is unaffected by both.

  Falls back to a plane fit only when no detection carries an orientation, in
  which case the caller inherits those weaknesses -- but nothing worse than the
  fixed-world-axis version this replaces.
  """
  frame = board_frame(candidates)
  if frame is not None:
    return [frame[:, i] for i in range(3)], True
  points = np.array([position(item) for item in candidates], dtype=float)
  return list(plane_axes(points)), False


def deduplicate(estimates):
  """Keep the highest-score estimate for each physical SC port."""
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


def fit_five_port_layout(candidates, axes):
  """Find one unambiguous 3-port/2-port SC rail layout, in the board frame.

  Searches over both which five detections are the ports and which board axis
  runs between the rails, so neither a false positive nor an unknown detector
  frame convention has to be guessed in advance.
  """
  if len(candidates) < NUM_PORTS:
    raise RuntimeError(
        f"Only {len(candidates)} distinct SC-port candidates were detected; "
        f"all {NUM_PORTS} are required for unambiguous absolute labels. "
        f"Candidates: {diagnostics(candidates)}"
    )

  valid_layouts = []
  for subset in combinations(candidates, NUM_PORTS):
    pts = np.array([position(item) for item in subset], dtype=float)

    # Coplanarity: the five ports sit on the board face.
    _p, _s, normal = plane_axes(pts)
    residual = float(np.max(np.abs((pts - pts.mean(axis=0)) @ normal)))
    if residual > MAX_PLANE_RESIDUAL_M:
      continue

    for between_index, between in enumerate(axes):
      # Split into rails at the widest gap along this candidate axis, rather
      # than by taking "the first three" -- which of the two rails has the
      # lower coordinate depends on how the board is turned.
      between_coords = pts @ between
      order = np.argsort(between_coords)
      cut = int(np.argmax(np.diff(between_coords[order]))) + 1
      lower = [subset[i] for i in order[:cut]]
      upper = [subset[i] for i in order[cut:]]

      # Rail identity comes from the port COUNT, not from which side it is on:
      # three ports is always rail 0. A 180 deg board rotation therefore cannot
      # silently swap the labels, which a coordinate-sign test would allow.
      if len(lower) == 3 and len(upper) == 2:
        group_a, group_b = lower, upper
      elif len(lower) == 2 and len(upper) == 3:
        group_a, group_b = upper, lower
      else:
        continue  # not a 3/2 split; not the SC rail signature

      a_coords = np.array([position(i) @ between for i in group_a], dtype=float)
      b_coords = np.array([position(i) @ between for i in group_b], dtype=float)
      spread_a = float(np.ptp(a_coords))
      spread_b = float(np.ptp(b_coords))
      separation = abs(float(b_coords.mean() - a_coords.mean()))
      if spread_a > MAX_WITHIN_RAIL_SPREAD_M:
        continue
      if spread_b > MAX_WITHIN_RAIL_SPREAD_M:
        continue
      if abs(separation - EXPECTED_RAIL_SEPARATION_M) > RAIL_SEPARATION_TOL_M:
        continue

      # Of the two remaining board axes the rail runs along the one the ports
      # actually spread over; the board normal has essentially no spread.
      others = [axes[j] for j in range(len(axes)) if j != between_index]
      along = max(others, key=lambda a: float(np.ptp(pts @ a))) * ALONG_RAIL_SIGN

      rail_0 = sorted(group_a, key=lambda item: float(position(item) @ along))
      rail_1 = sorted(group_b, key=lambda item: float(position(item) @ along))

      rail_0_x = np.array([position(i) @ along for i in rail_0], dtype=float)
      rail_1_x = np.array([position(i) @ along for i in rail_1], dtype=float)
      gaps_0, gaps_1 = np.diff(rail_0_x), np.diff(rail_1_x)
      if np.any(gaps_0 < MIN_ALONG_RAIL_SEPARATION_M):
        continue
      if np.any(gaps_1 < MIN_ALONG_RAIL_SEPARATION_M):
        continue

      valid_layouts.append({
          "rails": (rail_0, rail_1),
          "rail_separation": separation,
          "rail_spreads": (spread_a, spread_b),
          "along_gaps": (gaps_0, gaps_1),
          "plane_residual": residual,
          "along": along,
          "between": between,
          "geometry_error": (
              spread_a
              + spread_b
              + abs(separation - EXPECTED_RAIL_SEPARATION_M)
              + residual
          ),
          "score_sum": sum(float(item.score) for item in subset),
      })

  if not valid_layouts:
    raise RuntimeError(
        "No valid SC 3-port/2-port rail layout was found. Expected rail "
        f"separation {EXPECTED_RAIL_SEPARATION_M * 1000.0:.1f} +/- "
        f"{RAIL_SEPARATION_TOL_M * 1000.0:.1f} mm measured in the board frame. "
        f"Candidates: {diagnostics(candidates)}"
    )

  valid_layouts.sort(key=lambda item: (item["geometry_error"], -item["score_sum"]))
  if (
      len(valid_layouts) > 1
      and valid_layouts[1]["geometry_error"] - valid_layouts[0]["geometry_error"]
      < 0.003
  ):
    summaries = [
        {
            "geometry_error_mm": round(layout["geometry_error"] * 1000.0, 1),
            "rail_separation_mm": round(layout["rail_separation"] * 1000.0, 1),
            "score_sum": round(layout["score_sum"], 3),
            "ports": diagnostics(layout["rails"][0] + layout["rails"][1]),
        }
        for layout in valid_layouts[:4]
    ]
    raise RuntimeError(
        "Multiple SC rail layouts fit the detections; refusing to guess. "
        f"Layouts: {summaries}"
    )

  return valid_layouts[0]


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
if not confident:
  raise RuntimeError(
      f"No SC-port detections with score >= {MIN_SCORE}. Received scores: "
      f"{[round(float(item.score), 3) for item in params.pose_estimates]}"
  )

candidates = deduplicate(confident)
axes, axes_from_orientation = axis_candidates(candidates)
print(
    "SC-port filter input:",
    f"requested={target_name}",
    f"total_estimates={len(params.pose_estimates)}",
    f"confident_estimates={len(confident)}",
    f"physical_candidates={len(candidates)}",
    f"axes_from_orientation={axes_from_orientation}",
    f"candidates={diagnostics(candidates)}",
)

layout = fit_five_port_layout(candidates, axes)
along = layout["along"]

labeled = []
for labels, rail in zip(PORT_LABELS_BY_RAIL, layout["rails"]):
  if len(labels) != len(rail):
    raise RuntimeError(
        f"SC label/layout mismatch: labels={PORT_LABELS_BY_RAIL}, "
        f"rail_sizes={[len(items) for items in layout['rails']]}"
    )
  labeled.extend(zip(labels, rail))

matches = [estimate for label, estimate in labeled if label == target_idx]
if len(matches) != 1:
  raise RuntimeError(
      f"SC port {target_idx} was not uniquely labeled; "
      f"labels={[label for label, _ in labeled]}"
  )
best = matches[0]

# Print the derived layout in the BOARD frame. If the along-rail ordering or
# the 3/2 rail split does not match the task-board diagram, the mapping is
# wrong -- fix PORT_LABELS_BY_RAIL / ALONG_RAIL_SIGN rather than letting the
# run quietly target a neighbouring port.
origin = np.mean([position(est) for _label, est in labeled], axis=0)
rows = []
for label, est in sorted(labeled, key=lambda item: item[0]):
  rail = 0 if any(est is item for item in layout["rails"][0]) else 1
  rows.append(
      f"sc_port_{label}(rail={rail},"
      f"along={float((position(est) - origin) @ along) * 1000.0:+.0f}mm,"
      f"score={float(est.score):.2f})"
  )
print(
    "SC-port selection:",
    f"requested={target_name}",
    f"rail_separation_mm={layout['rail_separation'] * 1000.0:.1f}",
    f"rail_spreads_mm={[round(v * 1000.0, 1) for v in layout['rail_spreads']]}",
    f"plane_residual_mm={layout['plane_residual'] * 1000.0:.1f}",
    f"along_gaps_mm={[[round(float(g) * 1000.0, 1) for g in gaps] for gaps in layout['along_gaps']]}",
    f"layout={' '.join(rows)}",
    f"selected_xyz_mm={[round(float(v), 1) for v in position(best) * 1000.0]}",
)

output.pose_estimates.append(best)
output.root_ts_target.append(best.root_t_target)
return output
