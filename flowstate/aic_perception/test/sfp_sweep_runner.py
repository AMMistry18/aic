"""Offline production-geometry sweep for the staged-SFP survey policy.

Run from ``flowstate/aic_perception``:

    python test/sfp_sweep_runner.py --workers 4

Like ``sc_sweep_runner.py`` this is deliberately not named ``test_*.py``: the
full sweep takes minutes and is a pre-hardware validation tool, not part of the
unit suite.

What it measures that the search itself does not
------------------------------------------------
``search_survey_pose`` only guarantees that the *coverage target it was handed*
is framed and gripper-clear.  The hardware failure this harness exists for is a
module that is physically present, outside that target, and therefore clipped:

    RuntimeError: Need 5 SFP modules after dedupe, got 4

So every case here also runs an independent **seat audit**.  The board has six
legal staged-module seats at 50 mm pitch across two rails
(``sfp_module_detail_boxes``); five are occupied, and which one is empty does
not matter -- the outermost seats at board Y -0.15625 and +0.15625 are the ones
that bind, and they are occupied.  The audit projects all six seat bodies into
all three cameras and reports how many are fully inside the usable image.  A
case passes only when a pose was found *and* every seat survives that audit --
the property downstream IVM actually depends on, and the one the superseded
``sfp_sector_corners`` target did not imply.

Headline numbers (144 cases: 8 board yaws x 2 tilts x 3 placements x 3 live
Stage-1 exits), both rows at the shipped 25 px / 90 deg settings:

    coverage      found   all-5 framed   clipped   standoff
    sector          96          0          96      0.62-0.80
    shipped         92         92           0      0.64-0.85

The superseded box never frames the whole strip: 35 of its 96 poses hold only
four of the six seats and the other 61 hold five, with the worst-placed seat
123.9 px outside the image.  It "finds" four more poses than the fix only
because it is grading itself against half the board.

Reproduce the regression with ``--coverage sector``.
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

from aic_perception.arm_ik import UR5eArm, _rot_z  # noqa: E402
from aic_perception.board_stage2 import (  # noqa: E402
    BoardPoseEstimate,
    Transform,
    project_points,
    sfp_module_detail_boxes,
    sfp_module_strip_corners,
    sfp_sector_corners,
    search_survey_pose,
)


YAW_DEG = (0, 45, 70, 90, 140, 180, 250, 315)
TILT_DEG = (0, 10)
PLACEMENTS_M = ((0.0, 0.0), (0.050, 0.030), (-0.050, -0.040))
HOME_DEG = np.array([-9.15, -77.59, -95.39, -97.02, 90.01, 80.84])
EXIT_JOINTS = (
    np.radians(HOME_DEG),
    np.radians(HOME_DEG + np.array([0, 0, 0, 0, 0, 90.0])),
    # Same chained live start the SC sweep uses: relative gating must accept
    # this state and judge only travel from it.
    np.radians([-17.5, -95.8, -19.5, -143.9, 82.9, 26.8]),
)

# Coverage targets under test.  ``sector`` is the shipped single-rail target
# that produced the 4-of-5 hardware failure and is kept so the regression stays
# reproducible.
COVERAGE_TARGETS = {
    # Exactly what ``_coverage_targets_for_target`` ships for targets 0/1.
    "shipped": lambda p: (sfp_module_strip_corners(),),
    # The superseded single-rail box, kept so the 4-of-5 hardware regression
    # stays reproducible from this harness.
    "sector": lambda p: (sfp_sector_corners(),),
    # Free exploration of the frontier: centred strip with a caller-chosen
    # half-span and board-X range (``--y-half-mm``, ``--x-min-mm/--x-max-mm``).
    "centered": lambda p: (
        _centered_span_corners(p["y_half_m"], p.get("x_range")),
    ),
}

DEFAULT_POLICY = {
    "coverage": "shipped",
    "y_half_m": 0.1125,
    "max_obliquity_deg": 20.0,
    "min_clearance_px": 25.0,
    "max_angular_motion_deg": 90.0,
    "max_joint_motion_deg": 225.0,
    "seat_edge_margin_px": 12.0,
    "aim_x_m": 0.0,
    "aim_y_m": 0.0,
    "roll_count": 7,
}


def _centered_span_corners(y_half_m, x_range=None):
    """Staged-SFP strip spanning +/-``y_half_m`` over the given board-X range."""
    from aic_perception.board_stage2 import SFP_SPAN_X, SFP_SPAN_Z

    corners = []
    for x in (x_range or SFP_SPAN_X):
        for y in (-y_half_m, y_half_m):
            for z in SFP_SPAN_Z:
                corners.append((x, y, z))
    return np.array(corners, dtype=float)


def _arm_clear_of_own_cameras(base_T_tcp, joints, arm, tcp_T_cam, cameras):
    for start, end, radius in arm.link_segments(joints):
        samples = np.array(
            [start + (end - start) * t for t in np.linspace(0.0, 1.0, 25)]
        )
        for name, camera in cameras.items():
            camera_pose = base_T_tcp.compose(tcp_T_cam[name])
            local = camera_pose.inverse().apply(samples)
            pixels, in_front = project_points(local, camera)
            for pixel, ahead, point in zip(pixels, in_front, local):
                if not ahead or not np.all(np.isfinite(pixel)):
                    continue
                margin = radius * float(camera.K[0, 0]) / max(
                    float(point[2]), 1e-6
                )
                if (
                    -margin <= pixel[0] <= camera.width + margin
                    and -margin <= pixel[1] <= camera.height + margin
                ):
                    return False
    return True


def _seat_audit(base_T_tcp, base_T_board, tcp_T_cam, cameras, edge_margin_px):
    """Per-seat worst-camera image-boundary margin, in pixels.

    Independent of whatever coverage target the search was given: this is the
    physical question "would a module sitting in this seat be wholly inside
    every camera image".  Returns one margin per legal seat; negative means the
    seat body leaves at least one image.
    """
    margins = []
    for seat in sfp_module_detail_boxes():
        worst = math.inf
        for name, camera in cameras.items():
            cam_from_board = (
                base_T_tcp.compose(tcp_T_cam[name]).inverse().compose(base_T_board)
            )
            pixels, in_front = project_points(cam_from_board.apply(seat), camera)
            if not np.all(in_front) or not np.all(np.isfinite(pixels)):
                worst = -math.inf
                break
            u, v = pixels[:, 0], pixels[:, 1]
            worst = min(
                worst,
                float(u.min()),
                float(v.min()),
                float(camera.width - 1 - u.max()),
                float(camera.height - 1 - v.max()),
            )
        margins.append(
            worst - edge_margin_px if math.isfinite(worst) else -math.inf
        )
    return margins


def _run_case(case):
    yaw_deg, tilt_deg, placement_m, exit_index, *policy_items = case
    policy = dict(DEFAULT_POLICY)
    if policy_items:
        policy.update(policy_items[0])
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    arm = UR5eArm(
        flange_T_tcp=Transform(np.eye(3), np.array([0.0, 0.0, 0.1971])),
        base=Transform(_rot_z(math.pi), np.zeros(3)),
    )
    arm = replace(
        arm,
        flange_T_probes=tuple(
            arm.flange_T_tcp.compose(extrinsic)
            for extrinsic in tcp_T_cam.values()
        ),
    )
    seed = EXIT_JOINTS[exit_index]
    current_base_T_tcp = arm.fk(seed)

    rotation = axis_angle_rotation(
        [1, 0, 0], math.radians(tilt_deg)
    ) @ axis_angle_rotation([0, 0, 1], math.radians(yaw_deg + 2.7))
    board_pose = BoardPoseEstimate(
        Transform(
            rotation,
            np.array([-0.3445 + placement_m[0], 0.2602 + placement_m[1], 0.0]),
        ),
        0.3,
        math.inf,
        0.0,
        "center_camera",
    )

    def joint_motion(pose):
        solutions = arm.solve_ranked(pose, seed)
        clear = [
            target
            for target in solutions
            if _arm_clear_of_own_cameras(pose, target, arm, tcp_T_cam, cameras)
        ]
        if not clear:
            return None
        target = min(
            clear,
            key=lambda joints: (
                float(np.max(np.abs(joints - seed))),
                float(np.sum(np.abs(joints - seed))),
            ),
        )
        return target - seed

    candidate, reason = search_survey_pose(
        board_pose,
        tcp_T_cam,
        cameras,
        grippers,
        reference_camera="center_camera",
        current_base_T_tcp=current_base_T_tcp,
        coverage_targets=COVERAGE_TARGETS[policy["coverage"]](policy),
        aim_offset_board_m=(policy["aim_x_m"], policy["aim_y_m"], 0.0),
        cross_rail_tilt_band_rad=None,
        cross_rail_sign=0.0,
        require_all_cameras_frame=True,
        prefer_far_standoff=False,
        max_obliquity_rad=math.radians(policy["max_obliquity_deg"]),
        min_required_clearance_px=policy["min_clearance_px"],
        max_angular_motion_rad=math.radians(policy["max_angular_motion_deg"]),
        yaws_rad=(
            tuple(
                math.radians(deg)
                for deg in range(-180, 180, int(360 / policy["roll_count"]))
            )
            if policy["roll_count"] != 7
            else None
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
    if candidate is not None:
        margins = _seat_audit(
            candidate.base_T_tcp,
            board_pose.base_T_board,
            tcp_T_cam,
            cameras,
            policy["seat_edge_margin_px"],
        )
        covered = [m for m in margins if m >= 0.0]
        result.update(
            standoff_m=candidate.standoff_m,
            clearance_px=candidate.min_clearance_px,
            obliquity_deg=math.degrees(
                math.acos(
                    min(
                        1.0,
                        abs(
                            float(
                                np.dot(
                                    candidate.base_T_tcp.rotation[:, 2],
                                    board_pose.base_T_board.rotation[:, 2],
                                )
                            )
                        ),
                    )
                )
            ),
            joint_max_deg=math.degrees(candidate.max_joint_motion_rad),
            seats_covered=len(covered),
            seat_margin_min_px=(
                min(margins) if all(math.isfinite(m) for m in margins) else None
            ),
            seat_margins_px=[
                round(m, 1) if math.isfinite(m) else None for m in margins
            ],
        )
        result["pass"] = len(covered) == len(margins)
    else:
        result["pass"] = False
        result["seats_covered"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--coverage",
        choices=sorted(COVERAGE_TARGETS),
        default=DEFAULT_POLICY["coverage"],
        help="'shipped' is the deployed ladder; 'sector' reproduces the "
             "4-of-5 hardware regression.",
    )
    parser.add_argument("--y-half-mm", type=float, default=112.5)
    parser.add_argument("--x-min-mm", type=float, default=None)
    parser.add_argument("--x-max-mm", type=float, default=None)
    parser.add_argument("--max-obliquity-deg", type=float, default=20.0)
    parser.add_argument("--min-clearance-px", type=float, default=25.0)
    parser.add_argument("--max-angular-motion-deg", type=float, default=90.0)
    parser.add_argument("--max-joint-motion-deg", type=float, default=225.0)
    parser.add_argument("--seat-edge-margin-px", type=float, default=12.0)
    parser.add_argument("--aim-x-mm", type=float, default=0.0)
    parser.add_argument("--aim-y-mm", type=float, default=0.0)
    args = parser.parse_args()
    policy = {
        "coverage": args.coverage,
        "y_half_m": args.y_half_mm / 1000.0,
        "x_range": (
            (args.x_min_mm / 1000.0, args.x_max_mm / 1000.0)
            if args.x_min_mm is not None and args.x_max_mm is not None
            else None
        ),
        "max_obliquity_deg": args.max_obliquity_deg,
        "min_clearance_px": args.min_clearance_px,
        "max_angular_motion_deg": args.max_angular_motion_deg,
        "max_joint_motion_deg": args.max_joint_motion_deg,
        "seat_edge_margin_px": args.seat_edge_margin_px,
        "aim_x_m": args.aim_x_mm / 1000.0,
        "aim_y_m": args.aim_y_mm / 1000.0,
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
    clipped = [r for r in found if not r["pass"]]
    summary = {
        "coverage": args.coverage,
        "cases": len(results),
        "pose_found": len(found),
        "passed": len(passed),
        "no_pose": len(results) - len(found),
        "clipped_a_seat": len(clipped),
        "standoff_range_m": (
            [min(r["standoff_m"] for r in found), max(r["standoff_m"] for r in found)]
            if found
            else None
        ),
        "clearance_range_px": (
            [min(r["clearance_px"] for r in found), max(r["clearance_px"] for r in found)]
            if found
            else None
        ),
        "seat_margin_min_px": (
            min(
                r["seat_margin_min_px"]
                for r in found
                if r["seat_margin_min_px"] is not None
            )
            if any(r["seat_margin_min_px"] is not None for r in found)
            else None
        ),
        "joint_max_range_deg": (
            [min(r["joint_max_deg"] for r in found), max(r["joint_max_deg"] for r in found)]
            if found
            else None
        ),
        "seats_covered_histogram": {
            str(n): sum(1 for r in found if r["seats_covered"] == n)
            for n in sorted({r["seats_covered"] for r in found})
        },
        "failures": [
            {
                k: r[k]
                for k in (
                    "yaw_deg",
                    "tilt_deg",
                    "placement_m",
                    "exit_index",
                    "found",
                    "reason",
                    "seats_covered",
                    "seat_margins_px",
                )
                if k in r
            }
            for r in results
            if not r["pass"]
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
