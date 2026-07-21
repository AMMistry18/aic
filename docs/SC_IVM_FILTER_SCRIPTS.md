# SC IVM filter scripts

> **Node mapping:** `filter_estimates_sfp` must use section 2 and the staged
> SC-module filter must use section 3. Section 1 is only for selecting a
> destination `sc_port_N`; its `sc_port_([0-4])` parser must never be pasted
> into either source-module node.

These snippets are Flowstate Code Execution function bodies. Their inputs are
different because only the destination has a task-selected numbered slot:

- Destination SC port: `params.pose_estimates` and
  `params.selected_module_name`
- Staged SFP module: `params.pose_estimates` and
  `params.selected_module_name`
- SC plug/module: `params.pose_estimates` and
  `params.selected_module_name`

All three return one selected estimate in `output.pose_estimates` and its pose
in `output.root_ts_target`, which can be connected directly to Create Object.

## 1. Destination SC port (`sc_port_0` through `sc_port_4`)

The illustrated layout is:

```text
SC_RAIL_0: sc_port_0, sc_port_1, sc_port_2  (root +X order)
SC_RAIL_1: sc_port_3, sc_port_4             (root +X order)
rail order: root +Y
```

```python
import re
from itertools import combinations

import numpy as np


MIN_SCORE = 0.4
DUPLICATE_RADIUS_M = 0.018
NUM_PORTS = 5

# Current task-board coordinates: ports slide along root +X, and the two SC
# rails are separated/ordered along root +Y.
ALONG_RAIL_AXIS_ROOT = np.array([1.0, 0.0, 0.0], dtype=float)
BETWEEN_RAIL_AXIS_ROOT = np.array([0.0, 1.0, 0.0], dtype=float)

# From the task-board model, the two SC rails are nominally 41 mm apart.
EXPECTED_RAIL_SEPARATION_M = 0.041
RAIL_SEPARATION_TOL_M = 0.015

# Detections assigned to one physical rail should agree on their between-rail
# coordinate. This allows IVM noise without allowing a candidate from the
# other rail into the group.
MAX_WITHIN_RAIL_SPREAD_M = 0.012
MIN_ALONG_RAIL_SEPARATION_M = 0.018

# Diagram numbering. If qualification publishes a different documented
# numbering permutation, change only this tuple; do not change the geometry.
PORT_LABELS_BY_RAIL = ((0, 1, 2), (3, 4))


def position(estimate):
  p = estimate.root_t_target.position
  return np.array([p.x, p.y, p.z], dtype=float)


def along_coordinate(estimate):
  return float(np.dot(position(estimate), ALONG_RAIL_AXIS_ROOT))


def between_coordinate(estimate):
  return float(np.dot(position(estimate), BETWEEN_RAIL_AXIS_ROOT))


def deduplicate(estimates):
  """Keep the highest-score estimate for each physical SC port."""
  selected = []

  for estimate in sorted(
      estimates,
      key=lambda item: float(item.score),
      reverse=True,
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
        "along_x_mm": round(along_coordinate(estimate) * 1000.0, 1),
        "rail_y_mm": round(between_coordinate(estimate) * 1000.0, 1),
    })
  return rows


def fit_five_port_layout(candidates):
  """Find one unambiguous 3-port/2-port SC rail layout."""
  if len(candidates) < NUM_PORTS:
    raise RuntimeError(
        f"Only {len(candidates)} distinct SC-port candidates were detected; "
        f"all {NUM_PORTS} are required for unambiguous absolute labels. "
        f"Candidates: {diagnostics(candidates)}"
    )

  valid_layouts = []

  for subset in combinations(candidates, NUM_PORTS):
    # Rail 0 has the three lower +Y coordinates; rail 1 has the two higher
    # +Y coordinates. The 41 mm gap makes this split insensitive to IVM noise.
    by_rail = sorted(subset, key=between_coordinate)
    rail_0 = list(by_rail[:3])
    rail_1 = list(by_rail[3:])

    rail_0_y = np.array(
        [between_coordinate(item) for item in rail_0], dtype=float
    )
    rail_1_y = np.array(
        [between_coordinate(item) for item in rail_1], dtype=float
    )

    rail_0_spread = float(np.ptp(rail_0_y))
    rail_1_spread = float(np.ptp(rail_1_y))
    rail_separation = float(np.mean(rail_1_y) - np.mean(rail_0_y))

    if rail_0_spread > MAX_WITHIN_RAIL_SPREAD_M:
      continue
    if rail_1_spread > MAX_WITHIN_RAIL_SPREAD_M:
      continue
    if (
        abs(rail_separation - EXPECTED_RAIL_SEPARATION_M)
        > RAIL_SEPARATION_TOL_M
    ):
      continue

    rail_0 = sorted(rail_0, key=along_coordinate)
    rail_1 = sorted(rail_1, key=along_coordinate)

    rail_0_x = np.array(
        [along_coordinate(item) for item in rail_0], dtype=float
    )
    rail_1_x = np.array(
        [along_coordinate(item) for item in rail_1], dtype=float
    )
    rail_0_gaps = np.diff(rail_0_x)
    rail_1_gaps = np.diff(rail_1_x)

    if np.any(rail_0_gaps < MIN_ALONG_RAIL_SEPARATION_M):
      continue
    if np.any(rail_1_gaps < MIN_ALONG_RAIL_SEPARATION_M):
      continue

    geometry_error = (
        rail_0_spread
        + rail_1_spread
        + abs(rail_separation - EXPECTED_RAIL_SEPARATION_M)
    )

    valid_layouts.append({
        "rails": (rail_0, rail_1),
        "rail_separation": rail_separation,
        "rail_spreads": (rail_0_spread, rail_1_spread),
        "along_gaps": (rail_0_gaps, rail_1_gaps),
        "geometry_error": geometry_error,
        "score_sum": sum(float(item.score) for item in subset),
    })

  if not valid_layouts:
    raise RuntimeError(
        "No valid SC 3-port/2-port rail layout was found. "
        f"Expected rail separation "
        f"{EXPECTED_RAIL_SEPARATION_M * 1000.0:.1f} +/- "
        f"{RAIL_SEPARATION_TOL_M * 1000.0:.1f} mm. "
        f"Candidates: {diagnostics(candidates)}"
    )

  # A false positive may produce another plausible subset. Refuse to assign
  # absolute names if two layouts are geometrically indistinguishable.
  valid_layouts.sort(
      key=lambda item: (item["geometry_error"], -item["score_sum"])
  )
  if (
      len(valid_layouts) > 1
      and valid_layouts[1]["geometry_error"]
          - valid_layouts[0]["geometry_error"]
          < 0.003
  ):
    summaries = [
        {
            "geometry_error_mm": round(
                layout["geometry_error"] * 1000.0, 1
            ),
            "rail_separation_mm": round(
                layout["rail_separation"] * 1000.0, 1
            ),
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
      f"No SC-port detections with score >= {MIN_SCORE}. "
      f"Received scores: "
      f"{[round(float(item.score), 3) for item in params.pose_estimates]}"
  )

candidates = deduplicate(confident)
print(
    "SC-port filter input:",
    f"requested={target_name}",
    f"total_estimates={len(params.pose_estimates)}",
    f"confident_estimates={len(confident)}",
    f"physical_candidates={len(candidates)}",
    f"candidates={diagnostics(candidates)}",
)

layout = fit_five_port_layout(candidates)

labeled = []
for labels, rail in zip(PORT_LABELS_BY_RAIL, layout["rails"]):
  if len(labels) != len(rail):
    raise RuntimeError(
        f"SC label/layout mismatch: labels={PORT_LABELS_BY_RAIL}, "
        f"rail_sizes={[len(items) for items in layout['rails']]}"
    )
  labeled.extend(zip(labels, rail))

matches = [
    estimate
    for label, estimate in labeled
    if label == target_idx
]
if len(matches) != 1:
  raise RuntimeError(
      f"SC port {target_idx} was not uniquely labeled; "
      f"labels={[label for label, _ in labeled]}"
  )

best = matches[0]
print(
    "SC-port selection:",
    f"requested={target_name}",
    f"labels={[label for label, _ in labeled]}",
    f"rail_separation_mm="
    f"{layout['rail_separation'] * 1000.0:.1f}",
    f"rail_spreads_mm="
    f"{[round(value * 1000.0, 1) for value in layout['rail_spreads']]}",
    f"along_gaps_mm="
    f"{[[round(float(gap) * 1000.0, 1) for gap in gaps] for gaps in layout['along_gaps']]}",
    f"selected_xyz_mm="
    f"{[round(float(value), 1) for value in position(best) * 1000.0]}",
    f"selected_score={float(best.score):.3f}",
)

output.pose_estimates.append(best)
output.root_ts_target.append(best.root_t_target)
return output
```

## 2. SFP module paired with the selected NIC rail

This is the source-side SFP selector. It preserves the existing calibrated
grasp and removal offsets, but replaces score-only or hard-coded selection with
the index parsed from `selected_module_name`.

```python
import re

import numpy as np


MIN_SCORE = 0.4
DUPLICATE_RAIL_RADIUS_M = 0.018
NUM_MODULES = 5
RAIL_AXIS_ROOT = np.array([0.0, 1.0, 0.0], dtype=float)

# Phase 1 pairing: nic_card_mount_N -> staged SFP module N.
MODULE_INDEX_FOR_NIC_RAIL = (0, 1, 2, 3, 4)

GRASP_OFFSET_POS = np.array([0.0, 0.027, 0.0], dtype=float)
GRASP_OFFSET_QUAT = np.array(
    [-0.568, 0.023, -0.015, -0.822], dtype=float
)
REMOVE_OFFSET_POS = np.array([0.0, 0.077, 0.0], dtype=float)


def position(estimate):
  p = estimate.root_t_target.position
  return np.array([p.x, p.y, p.z], dtype=float)


def orientation(estimate):
  q = estimate.root_t_target.orientation
  value = np.array([q.x, q.y, q.z, q.w], dtype=float)
  norm = float(np.linalg.norm(value))
  if norm < 1e-9:
    raise RuntimeError("SFP estimate contains a zero-length quaternion")
  return value / norm


def rail_coordinate(estimate):
  return float(np.dot(position(estimate), RAIL_AXIS_ROOT))


def deduplicate(estimates):
  """Retain the strongest observation of each physical SFP module."""
  selected = []
  for estimate in sorted(
      estimates,
      key=lambda item: float(item.score),
      reverse=True,
  ):
    coordinate = rail_coordinate(estimate)
    if all(
        abs(coordinate - rail_coordinate(existing))
        >= DUPLICATE_RAIL_RADIUS_M
        for existing in selected
    ):
      selected.append(estimate)
  return selected


def diagnostics(estimates):
  return [
      {
          "score": round(float(item.score), 3),
          "xyz_mm": [
              round(float(value), 1)
              for value in position(item) * 1000.0
          ],
          "rail_y_mm": round(rail_coordinate(item) * 1000.0, 1),
      }
      for item in estimates
  ]


def quaternion_multiply(first, second):
  x1, y1, z1, w1 = first
  x2, y2, z2, w2 = second
  return np.array([
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
  ], dtype=float)


def quaternion_rotate(quaternion, vector):
  xyz = quaternion[:3]
  return vector + 2.0 * np.cross(
      xyz,
      quaternion[3] * vector + np.cross(xyz, vector),
  )


output = code_execution_pb2.ReturnValue()

target_name = params.selected_module_name.strip()
match = re.fullmatch(r"nic_card_mount_([0-4])", target_name)
if match is None:
  raise RuntimeError(
      f"Invalid selected_module_name {target_name!r}; expected "
      "nic_card_mount_0 through nic_card_mount_4"
  )
nic_idx = int(match.group(1))
module_idx = MODULE_INDEX_FOR_NIC_RAIL[nic_idx]

confident = [
    estimate
    for estimate in params.pose_estimates
    if float(estimate.score) >= MIN_SCORE
]
if not confident:
  raise RuntimeError(
      f"No SFP-module detections with score >= {MIN_SCORE}. "
      f"Received scores: "
      f"{[round(float(item.score), 3) for item in params.pose_estimates]}"
  )

candidates = deduplicate(confident)
print(
    "SFP-module filter input:",
    f"requested_nic={target_name}",
    f"paired_module_index={module_idx}",
    f"total_estimates={len(params.pose_estimates)}",
    f"confident_estimates={len(confident)}",
    f"physical_candidates={len(candidates)}",
    f"candidates={diagnostics(candidates)}",
)

if len(candidates) != NUM_MODULES:
  raise RuntimeError(
      f"Expected {NUM_MODULES} physical SFP modules to preserve absolute "
      f"pairing with {target_name}, but found {len(candidates)}. "
      f"Candidates: {diagnostics(candidates)}"
  )

ordered = sorted(candidates, key=rail_coordinate)
best = ordered[module_idx]

sfp_position = position(best)
sfp_orientation = orientation(best)

offset_orientation = GRASP_OFFSET_QUAT.copy()
offset_orientation /= np.linalg.norm(offset_orientation)

grasp_position = sfp_position + quaternion_rotate(
    sfp_orientation,
    GRASP_OFFSET_POS,
)
grasp_orientation = quaternion_multiply(
    sfp_orientation,
    offset_orientation,
)
grasp_orientation /= np.linalg.norm(grasp_orientation)

remove_position = sfp_position + quaternion_rotate(
    sfp_orientation,
    REMOVE_OFFSET_POS,
)

print(
    "SFP-module selection:",
    f"requested_nic={target_name}",
    f"paired_module_index={module_idx}",
    f"ordered_y_mm="
    f"{[round(rail_coordinate(item) * 1000.0, 1) for item in ordered]}",
    f"selected_xyz_mm="
    f"{[round(float(value), 1) for value in sfp_position * 1000.0]}",
    f"selected_score={float(best.score):.3f}",
)

output.pose_estimates.append(best)
output.root_ts_target.append(best.root_t_target)

grasp_pose = output.grasp_poses.add()
grasp_pose.position.x = float(grasp_position[0])
grasp_pose.position.y = float(grasp_position[1])
grasp_pose.position.z = float(grasp_position[2])
grasp_pose.orientation.x = float(grasp_orientation[0])
grasp_pose.orientation.y = float(grasp_orientation[1])
grasp_pose.orientation.z = float(grasp_orientation[2])
grasp_pose.orientation.w = float(grasp_orientation[3])

remove_pose = output.remove_poses.add()
remove_pose.position.x = float(remove_position[0])
remove_pose.position.y = float(remove_position[1])
remove_pose.position.z = float(remove_position[2])
remove_pose.orientation.x = float(grasp_orientation[0])
remove_pose.orientation.y = float(grasp_orientation[1])
remove_pose.orientation.z = float(grasp_orientation[2])
remove_pose.orientation.w = float(grasp_orientation[3])

return output
```

## 3. SC plug/module paired with the selected NIC rail

This filter parses the index from the task's selected NIC name, orders the five
physical SC plug/module candidates along the same root-frame +Y rail axis, and
returns the candidate with that index. The same `selected_module_name` input
must be connected to both the SFP and SC filters. IVM may report one object from
several cameras, so estimates are first deduplicated by their rail coordinate.
It intentionally does not copy the SFP grasp/remove offsets; SC needs its own
calibrated offsets.

```python
import re

import numpy as np


MIN_SCORE = 0.4
DUPLICATE_RAIL_RADIUS_M = 0.018
NUM_MODULES = 5
RAIL_AXIS_ROOT = np.array([0.0, 1.0, 0.0], dtype=float)

# Explicit Phase 1 pairing: nic_card_mount_N -> SC module N. If inspection of
# the qualification board proves the physical ordering is reversed, change
# only this tuple to (4, 3, 2, 1, 0).
MODULE_INDEX_FOR_NIC_RAIL = (0, 1, 2, 3, 4)


def position(estimate):
  p = estimate.root_t_target.position
  return np.array([p.x, p.y, p.z], dtype=float)


def rail_coordinate(estimate):
  return float(np.dot(position(estimate), RAIL_AXIS_ROOT))


def deduplicate(estimates):
  """Merge multi-camera estimates of the same physical SC plug."""
  selected = []
  for estimate in sorted(
      estimates,
      key=lambda item: float(item.score),
      reverse=True,
  ):
    coordinate = rail_coordinate(estimate)
    if all(
        abs(coordinate - rail_coordinate(existing))
        >= DUPLICATE_RAIL_RADIUS_M
        for existing in selected
    ):
      selected.append(estimate)
  return selected


def diagnostics(estimates):
  return [
      {
          "score": round(float(item.score), 3),
          "xyz_mm": [
              round(float(value), 1)
              for value in position(item) * 1000.0
          ],
          "rail_y_mm": round(rail_coordinate(item) * 1000.0, 1),
      }
      for item in estimates
  ]


output = code_execution_pb2.ReturnValue()

target_name = params.selected_module_name.strip()
match = re.fullmatch(r"nic_card_mount_([0-4])", target_name)
if match is None:
  raise RuntimeError(
      f"Invalid selected_module_name {target_name!r}; expected "
      "nic_card_mount_0 through nic_card_mount_4"
  )
nic_idx = int(match.group(1))
module_idx = MODULE_INDEX_FOR_NIC_RAIL[nic_idx]

confident = [
    estimate
    for estimate in params.pose_estimates
    if float(estimate.score) >= MIN_SCORE
]
if not confident:
  raise RuntimeError(
      f"No SC-plug detections with score >= {MIN_SCORE}. "
      f"Received scores: "
      f"{[round(float(item.score), 3) for item in params.pose_estimates]}"
  )

candidates = deduplicate(confident)
print(
    "SC-plug filter input:",
    f"requested_nic={target_name}",
    f"paired_module_index={module_idx}",
    f"total_estimates={len(params.pose_estimates)}",
    f"confident_estimates={len(confident)}",
    f"physical_candidates={len(candidates)}",
    f"candidates={diagnostics(candidates)}",
)

if len(candidates) != NUM_MODULES:
  raise RuntimeError(
      f"Expected {NUM_MODULES} physical SC modules to preserve absolute "
      f"pairing with {target_name}, but found {len(candidates)}. "
      f"Candidates: {diagnostics(candidates)}"
  )

ordered = sorted(candidates, key=rail_coordinate)
best = ordered[module_idx]

print(
    "SC-plug selection:",
    f"requested_nic={target_name}",
    f"paired_module_index={module_idx}",
    f"ordered_y_mm="
    f"{[round(rail_coordinate(item) * 1000.0, 1) for item in ordered]}",
    f"selected_xyz_mm="
    f"{[round(float(value), 1) for value in position(best) * 1000.0]}",
    f"selected_score={float(best.score):.3f}",
)

output.pose_estimates.append(best)
output.root_ts_target.append(best.root_t_target)
return output
```

Do not reuse the SFP `GRASP_OFFSET_POS`, `GRASP_OFFSET_QUAT`, or
`REMOVE_OFFSET_POS` for the SC plug. Those fields can be added after measuring
the SC-specific transform; they are not needed by Create Object.
