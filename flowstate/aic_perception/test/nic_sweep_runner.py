"""Offline production-geometry sweep for the NIC destination survey policy.

Run from ``flowstate/aic_perception``:

    python test/nic_sweep_runner.py --workers 8

Like ``sc_sweep_runner.py`` and ``sfp_sweep_runner.py`` this is deliberately not
named ``test_*.py``: the full sweep takes minutes and is a pre-hardware
validation tool, not part of the unit suite.

Why this exists
---------------
It did not, until 2026-07-28.  NIC shipped its 25 px / 90 deg / 24-roll policy
and its ``prefer_far_standoff`` rule on the strength of three hardware board
orientations and no offline matrix at all, while SC and SFP each had 144 cases.
So when the 90 deg Cartesian reorientation cap was found to be selecting the
candidate set rather than bounding motion (handoff 21), NIC's share of that fix
had nothing to check it against.

What it measures that the search does not
-----------------------------------------
``search_survey_pose`` guarantees the coverage box is framed and gripper-clear.
For NIC that is necessary and *not* sufficient, because the thing the IVM reads
is the black depth inside a recessed cage:

* each of the ten ports is a 16 x 12 mm aperture at the top of a 45.8 mm recess
  whose axis is 0.7 deg off the board normal, so the bore only shows depth to a
  ray within ``atan(6/45.8) = 7.5 deg`` of that axis;
* the outermost port sits ~81 mm from the aimed centre, which is the whole
  reason ``prefer_far_standoff`` is right here -- staying inside the cone needs
  ``standoff >= 0.081 / tan(7.5 deg) ~= 0.62 m``.

So every case here runs an independent **port audit**: all ten entrances are
projected into all three cameras for framing, and the centre-camera ray to each
port is measured against the 7.5 deg cone.  A case passes only when a pose was
found, every port is framed in every camera, and every port is inside the cone.
That is the property the ten detections actually depend on, and none of it is
implied by the coverage box being in frame.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(PACKAGE_DIR))

from test_board_stage2 import (  # noqa: E402
    _production_camera_rig,
    axis_angle_rotation,
)

from aic_perception.arm_ik import (  # noqa: E402
    UR5eArm,
    _rot_z,
    capsule_intersects_camera_view,
)
from aic_perception.board_stage2 import (  # noqa: E402
    BoardPoseEstimate,
    Transform,
    nic_sector_corners,
    project_points,
    search_survey_pose,
)

YAW_DEG = (0, 45, 70, 90, 140, 180, 250, 315)
TILT_DEG = (0, 10)
PLACEMENTS_M = ((0.0, 0.0), (0.050, 0.030), (-0.050, -0.040))
# Same measured hardware board position the SC and SFP sweeps use; see
# ``sfp_sweep_runner.BOARD_CENTER_M``.
BOARD_CENTER_M = (-0.5189, 0.2054)
LEGACY_BOARD_CENTER_M = (-0.3445, 0.2602)
HOME_DEG = np.array([-9.15, -77.59, -95.39, -97.02, 90.01, 80.84])
EXIT_JOINTS = (
    np.radians(HOME_DEG),
    np.radians(HOME_DEG + np.array([0, 0, 0, 0, 0, 90.0])),
    np.radians([-17.5, -95.8, -19.5, -143.9, 82.9, 26.8]),
)

# The ten port entrances (``nic_sector_corners`` docstring / aic_world.xml):
# five cards at 40 mm board-Y pitch, two ports per card at board X -0.100 and
# -0.077, mouth plane board Z 0.1793.
PORT_X = (-0.100, -0.077)
PORT_Y = (-0.186, -0.146, -0.106, -0.066, -0.026)
PORT_Z = 0.1793
PORT_HALF_X = 0.008  # 16 mm aperture
PORT_HALF_Y = 0.006  # 12 mm aperture
# atan(6 / 45.8): the half-aperture over the recess depth.
PORT_CONE_RAD = math.atan2(0.006, 0.0458)
# The shipped ``search_survey_pose`` ladder, so ``--min-standoff-m`` can
# trim it rather than restate it.
DEFAULT_STANDOFFS_M = (
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.64,
    0.66, 0.68, 0.70, 0.73, 0.76, 0.80, 0.85, 0.90, 1.00, 1.15, 1.25,
)


def port_mouths() -> tuple[np.ndarray, ...]:
    """One board-frame rectangle per port entrance."""
    mouths = []
    for x in PORT_X:
        for y in PORT_Y:
            mouths.append(
                np.array(
                    [
                        (x - PORT_HALF_X, y - PORT_HALF_Y, PORT_Z),
                        (x + PORT_HALF_X, y - PORT_HALF_Y, PORT_Z),
                        (x + PORT_HALF_X, y + PORT_HALF_Y, PORT_Z),
                        (x - PORT_HALF_X, y + PORT_HALF_Y, PORT_Z),
                    ],
                    dtype=float,
                )
            )
    return tuple(mouths)


def _sector_image_regions(
    base_T_tcp,
    base_T_board,
    target_board,
    tcp_T_cam,
    cameras,
):
    """Project the NIC sector into each wrist camera."""
    regions = {}
    for name, camera in cameras.items():
        camera_from_board = (
            base_T_tcp.compose(tcp_T_cam[name])
            .inverse()
            .compose(base_T_board)
        )
        pixels, in_front = project_points(
            camera_from_board.apply(target_board), camera
        )
        if np.all(in_front) and np.all(np.isfinite(pixels)):
            regions[name] = (
                float(pixels[:, 0].min()),
                float(pixels[:, 1].min()),
                float(pixels[:, 0].max()),
                float(pixels[:, 1].max()),
            )
    return regions


def _arm_clear_of_own_cameras(
    base_T_tcp,
    joints,
    arm,
    tcp_T_cam,
    cameras,
    sector_regions=None,
    clearance_px=25.0,
):
    """Mirror production's exact, near-plane-safe link-capsule gate."""
    for name, camera in cameras.items():
        region = (
            sector_regions.get(name)
            if sector_regions is not None
            else None
        )
        if sector_regions is not None and region is None:
            continue
        bounds = None
        if region is not None:
            u_min, v_min, u_max, v_max = region
            bounds = (
                u_min - clearance_px,
                v_min - clearance_px,
                u_max + clearance_px,
                v_max + clearance_px,
            )
        camera_from_base = base_T_tcp.compose(tcp_T_cam[name]).inverse()
        for start, end, radius in arm.link_segments(joints):
            if capsule_intersects_camera_view(
                camera_from_base.apply(start),
                camera_from_base.apply(end),
                radius,
                camera,
                bounds,
            ):
                return False
    return True


def _port_audit(base_T_tcp, base_T_board, tcp_T_cam, cameras, edge_margin_px):
    """(worst framing margin px, worst centre-camera bore angle deg) per port.

    Framing is required in every camera.  The bore angle is measured on the
    centre camera only: the recessed cage shows its black interior to whichever
    camera looks down it, and the fused IVM keys on that dominant top-down view.
    """
    board_normal = base_T_board.rotation[:, 2]
    frame_margins = []
    cone_angles = []
    for mouth in port_mouths():
        worst_margin = math.inf
        for name, camera in cameras.items():
            cam_from_board = (
                base_T_tcp.compose(tcp_T_cam[name])
                .inverse()
                .compose(base_T_board)
            )
            pixels, in_front = project_points(cam_from_board.apply(mouth), camera)
            if not np.all(in_front) or not np.all(np.isfinite(pixels)):
                worst_margin = -math.inf
                break
            u, v = pixels[:, 0], pixels[:, 1]
            worst_margin = min(
                worst_margin,
                float(u.min()) - edge_margin_px,
                float(v.min()) - edge_margin_px,
                float(camera.width - 1 - u.max()) - edge_margin_px,
                float(camera.height - 1 - v.max()) - edge_margin_px,
            )
        frame_margins.append(worst_margin)

        centre = base_T_board.apply(mouth.mean(axis=0))
        camera_origin = base_T_tcp.compose(tcp_T_cam["center_camera"]).translation
        to_camera = camera_origin - centre
        norm = float(np.linalg.norm(to_camera))
        if norm < 1e-9:
            cone_angles.append(math.inf)
            continue
        cone_angles.append(
            math.degrees(
                math.acos(
                    float(np.clip(np.dot(to_camera, board_normal) / norm, -1.0, 1.0))
                )
            )
        )
    return frame_margins, cone_angles


def _run_case(case):
    yaw_deg, tilt_deg, placement_m, exit_index, policy = case
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    arm = UR5eArm(
        flange_T_tcp=Transform(np.eye(3), np.array([0.0, 0.0, 0.1971])),
        base=Transform(_rot_z(math.pi), np.zeros(3)),
        min_self_clearance_m=policy["keepout_mm"] / 1000.0,
    )
    arm = replace(
        arm,
        flange_T_probes=tuple(
            arm.flange_T_tcp.compose(extrinsic)
            for extrinsic in tcp_T_cam.values()
        ),
    )
    seed = EXIT_JOINTS[exit_index]

    rotation = axis_angle_rotation(
        [1, 0, 0], math.radians(tilt_deg)
    ) @ axis_angle_rotation([0, 0, 1], math.radians(yaw_deg + 2.7))
    board_center = policy["board_center_m"]
    board_pose = BoardPoseEstimate(
        Transform(
            rotation,
            np.array(
                [
                    board_center[0] + placement_m[0],
                    board_center[1] + placement_m[1],
                    0.0,
                ]
            ),
        ),
        0.3,
        math.inf,
        0.0,
        "center_camera",
    )

    def joint_motion(pose):
        solutions = arm.solve_ranked(pose, seed)
        sector_regions = (
            _sector_image_regions(
                pose,
                board_pose.base_T_board,
                nic_sector_corners(),
                tcp_T_cam,
                cameras,
            )
            if policy["arm_rule"] == "sector"
            else None
        )
        clear = [
            target
            for target in solutions
            if _arm_clear_of_own_cameras(
                pose,
                target,
                arm,
                tcp_T_cam,
                cameras,
                sector_regions,
                policy["min_clearance_px"],
            )
        ]
        # Production NIC checks every keep-out-valid ranked branch; the later
        # relaxation tier may accept one clear branch, but strict certification
        # should retain the all-ranked-branches rule.
        if not clear or len(clear) != len(solutions):
            return None
        return min(
            clear,
            key=lambda q: float(np.max(np.abs(q - seed))),
        ) - seed

    candidate, reason = search_survey_pose(
        board_pose,
        tcp_T_cam,
        cameras,
        grippers,
        reference_camera="center_camera",
        current_base_T_tcp=arm.fk(seed),
        coverage_targets=(nic_sector_corners(),),
        **(
            {
                "standoffs_m": tuple(
                    s
                    for s in DEFAULT_STANDOFFS_M
                    if s >= policy["min_standoff_m"] - 1e-9
                )
            }
            if policy.get("min_standoff_m")
            else {}
        ),
        cross_rail_tilt_band_rad=None,
        cross_rail_sign=0.0,
        require_all_cameras_frame=True,
        prefer_far_standoff=True,
        max_obliquity_rad=math.radians(policy["max_obliquity_deg"]),
        min_required_clearance_px=policy["min_clearance_px"],
        max_angular_motion_rad=math.radians(policy["max_angular_motion_deg"]),
        yaws_rad=tuple(
            math.radians(deg)
            for deg in range(-180, 180, int(360 / policy["roll_count"]))
        ),
        joint_motion=joint_motion,
        max_joint_motion_rad=math.radians(policy["max_joint_motion_deg"]),
        max_reach_m=0.85,
        min_height_m=0.02,
    )
    result = {
        "yaw_deg": yaw_deg,
        "tilt_deg": tilt_deg,
        "placement_m": placement_m,
        "exit_index": exit_index,
        "found": candidate is not None,
        "reason": reason,
    }
    if candidate is None:
        result["pass"] = False
        return result

    frame_margins, cone_angles = _port_audit(
        candidate.base_T_tcp,
        board_pose.base_T_board,
        tcp_T_cam,
        cameras,
        policy["port_edge_margin_px"],
    )
    selected_delta = joint_motion(candidate.base_T_tcp)
    if selected_delta is None:  # Defensive: the accepted pose just passed it.
        raise RuntimeError("selected NIC pose no longer passes its IK gate")
    framed = [m for m in frame_margins if m >= 0.0]
    in_cone = [a for a in cone_angles if a <= math.degrees(PORT_CONE_RAD)]
    result.update(
        standoff_m=candidate.standoff_m,
        clearance_px=candidate.min_clearance_px,
        joint_max_deg=math.degrees(candidate.max_joint_motion_rad),
        joint_total_deg=math.degrees(float(np.abs(selected_delta).sum())),
        ports_framed=len(framed),
        ports_in_cone=len(in_cone),
        port_margin_min_px=(
            min(frame_margins)
            if all(math.isfinite(m) for m in frame_margins)
            else None
        ),
        port_cone_max_deg=max(cone_angles),
    )
    result["pass"] = len(framed) == 10 and len(in_cone) == 10
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--max-obliquity-deg", type=float, default=2.0)
    parser.add_argument("--min-clearance-px", type=float, default=25.0)
    parser.add_argument("--max-angular-motion-deg", type=float, default=180.0)
    parser.add_argument("--max-joint-motion-deg", type=float, default=225.0)
    parser.add_argument(
        "--arm-rule",
        choices=("sector", "whole"),
        default="sector",
        help="Match production's NIC-sector rule or test whole-image exclusion.",
    )
    parser.add_argument(
        "--keepout-mm",
        type=float,
        default=140.0,
        help="Wrist-camera to forearm centreline keep-out.",
    )
    parser.add_argument("--port-edge-margin-px", type=float, default=12.0)
    parser.add_argument("--roll-count", type=int, default=24)
    parser.add_argument(
        "--min-standoff-m",
        type=float,
        default=0.66,
        help="Drop ladder rungs below this. The measured good-view band "
             "starts at 0.66 m; shorter poses frame all ten ports but put "
             "the outer ones outside the 7.5 deg bore cone.",
    )
    parser.add_argument(
        "--board-center-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Board origin in base_link, millimetres.  Default is the measured "
             "hardware position; pass -344.5 260.2 for the legacy pin.",
    )
    args = parser.parse_args()
    policy = {
        "max_obliquity_deg": args.max_obliquity_deg,
        "min_clearance_px": args.min_clearance_px,
        "max_angular_motion_deg": args.max_angular_motion_deg,
        "max_joint_motion_deg": args.max_joint_motion_deg,
        "arm_rule": args.arm_rule,
        "keepout_mm": args.keepout_mm,
        "port_edge_margin_px": args.port_edge_margin_px,
        "roll_count": args.roll_count,
        "min_standoff_m": args.min_standoff_m,
        "board_center_m": (
            (args.board_center_mm[0] / 1000.0, args.board_center_mm[1] / 1000.0)
            if args.board_center_mm is not None
            else BOARD_CENTER_M
        ),
    }
    cases = [
        (yaw, tilt, placement, exit_index, policy)
        for yaw in YAW_DEG
        for tilt in TILT_DEG
        for placement in PLACEMENTS_M
        for exit_index in range(len(EXIT_JOINTS))
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(_run_case, cases))

    found = [r for r in results if r["found"]]
    passed = [r for r in results if r["pass"]]
    summary = {
        "cases": len(results),
        "arm_rule": policy["arm_rule"],
        "keepout_mm": policy["keepout_mm"],
        "min_standoff_m": policy["min_standoff_m"],
        "pose_found": len(found),
        "passed": len(passed),
        "no_pose": len(results) - len(found),
        "found_but_port_audit_failed": len(found) - len(passed),
        "standoff_range_m": (
            [
                min(r["standoff_m"] for r in found),
                max(r["standoff_m"] for r in found),
            ]
            if found
            else None
        ),
        "clearance_range_px": (
            [
                min(r["clearance_px"] for r in found),
                max(r["clearance_px"] for r in found),
            ]
            if found
            else None
        ),
        "port_margin_min_px": (
            min(
                r["port_margin_min_px"]
                for r in found
                if r["port_margin_min_px"] is not None
            )
            if any(r["port_margin_min_px"] is not None for r in found)
            else None
        ),
        "port_cone_max_deg": (
            max(r["port_cone_max_deg"] for r in found) if found else None
        ),
        "port_cone_limit_deg": math.degrees(PORT_CONE_RAD),
        "joint_max_range_deg": (
            [
                min(r["joint_max_deg"] for r in found),
                max(r["joint_max_deg"] for r in found),
            ]
            if found
            else None
        ),
        "joint_total_range_deg": (
            [
                min(r["joint_total_deg"] for r in found),
                max(r["joint_total_deg"] for r in found),
            ]
            if found
            else None
        ),
        "ports_framed_histogram": {
            str(n): sum(1 for r in found if r["ports_framed"] == n)
            for n in sorted({r["ports_framed"] for r in found})
        },
        "ports_in_cone_histogram": {
            str(n): sum(1 for r in found if r["ports_in_cone"] == n)
            for n in sorted({r["ports_in_cone"] for r in found})
        },
        "failures": [
            {
                "yaw_deg": r["yaw_deg"],
                "tilt_deg": r["tilt_deg"],
                "placement_m": r["placement_m"],
                "exit_index": r["exit_index"],
                "reason": r["reason"],
                "ports_framed": r.get("ports_framed"),
                "ports_in_cone": r.get("ports_in_cone"),
            }
            for r in results
            if not r["pass"]
        ][:12],
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
