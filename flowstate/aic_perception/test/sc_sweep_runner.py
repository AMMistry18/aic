"""Offline production-geometry sweep for the SC survey policy.

Run from ``flowstate/aic_perception``:

    python test/sc_sweep_runner.py --workers 4

This is intentionally not named ``test_*.py`` because the full sweep takes
minutes and is a pre-hardware validation tool, not part of the two-minute unit
suite.  It exercises board yaw, tilt, placement, two Stage-1 wrist-roll exits,
the exact chained hardware start that exposed the old absolute-window failure,
the production camera intrinsics/extrinsics and gripper masks, the analytic IK
and camera/forearm gates, SC bore visibility, and live-relative joint travel.
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

from aic_perception.arm_ik import JOINT_LIMITS, UR5eArm, _rot_z  # noqa: E402
from aic_perception.board_stage2 import (  # noqa: E402
    BoardPoseEstimate,
    Transform,
    project_points,
    rectangular_bore_depth_cue_px,
    rectangular_bore_visibility_margin,
    sc_bore_sample_points,
    sc_sector_corners,
    search_survey_pose,
)


YAW_DEG = (0, 45, 70, 90, 140, 180, 250, 315)
TILT_DEG = (0, 10)
PLACEMENTS_M = ((0.0, 0.0), (0.050, 0.030), (-0.050, -0.040))
HOME_DEG = np.array([-9.15, -77.59, -95.39, -97.02, 90.01, 80.84])
EXIT_JOINTS = (
    np.radians(HOME_DEG),
    np.radians(HOME_DEG + np.array([0, 0, 0, 0, 0, 90.0])),
    # Exact live start from the 2026-07-26 chained NIC/SFP -> SC failure.  The
    # former absolute window rejected J4=-143.9 deg before evaluating a single
    # SC pose.  Relative gating must accept this state and judge only travel
    # from it.
    np.radians([-17.5, -95.8, -19.5, -143.9, 82.9, 26.8]),
)
DEFAULT_POLICY = {
    "tilt_min_deg": 16.0,
    "tilt_max_deg": 20.0,
    "aim_x_m": 0.0,
    "min_standoff_m": 0.55,
    "max_standoff_m": 0.62,
    "min_depth_cue_px": 3.0,
    "depth_cue_motion_tolerance_px": 0.10,
    "required_bore_cameras": 2,
    "min_bore_margin": 0.0,
    # Acceptance criterion on the narrow bore axis.  The full mouth width
    # (0.0076) accepts a displaced dark strip; the half width (0.0038) demands
    # the back-plane centre and caps the long-face angle at 13.66 deg.
    "bore_x_tolerance_m": 0.0076,
    "required_depth_cameras": 2,
    "max_joint_motion_deg": 185.0,
    "j6_preference_motion_tolerance_deg": 30.0,
}


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
    j6_low, j6_high = JOINT_LIMITS[5]
    requested_flips = (seed[5] + math.pi, seed[5] - math.pi)
    legal_flips = tuple(
        float(value)
        for value in requested_flips
        if j6_low - 1e-9 <= value <= j6_high + 1e-9
    )
    preferred_j6 = (
        legal_flips[0]
        if legal_flips
        else float(np.clip(requested_flips[0], j6_low, j6_high))
    )

    rotation = axis_angle_rotation(
        [1, 0, 0], math.radians(tilt_deg)
    ) @ axis_angle_rotation([0, 0, 1], math.radians(yaw_deg + 2.7))
    board_pose = BoardPoseEstimate(
        Transform(
            rotation,
            np.array(
                [
                    -0.3445 + placement_m[0],
                    0.2602 + placement_m[1],
                    0.0,
                ]
            ),
        ),
        0.3,
        math.inf,
        0.0,
        "center_camera",
    )
    sc_view_samples = sc_bore_sample_points()

    def view_quality(pose):
        bore_margin = rectangular_bore_visibility_margin(
            sc_view_samples,
            board_pose.base_T_board,
            pose,
            tcp_T_cam,
            half_width_x_m=policy["bore_x_tolerance_m"],
            half_width_y_m=0.0112,
            depth_m=0.01564,
            camera_names=tuple(cameras),
            required_camera_count=policy["required_bore_cameras"],
        )
        if bore_margin < policy["min_bore_margin"]:
            return -math.inf
        return rectangular_bore_depth_cue_px(
            sc_view_samples,
            board_pose.base_T_board,
            pose,
            tcp_T_cam,
            cameras,
            depth_m=0.01564,
            camera_names=tuple(cameras),
            required_camera_count=policy["required_depth_cameras"],
        )

    ik_counts = {"solved": 0, "arm_clear": 0, "within_180": 0}

    def joint_motion(pose):
        solutions = arm.solve_ranked(
            pose,
            seed,
        )
        ik_counts["solved"] += len(solutions)
        clear = [
            target
            for target in solutions
            if _arm_clear_of_own_cameras(
                pose, target, arm, tcp_T_cam, cameras
            )
        ]
        ik_counts["arm_clear"] += len(clear)
        if not clear:
            return None
        target = min(
            clear,
            key=lambda joints: (
                float(np.max(np.abs(joints - seed))),
                abs(float(joints[5] - preferred_j6)),
                float(np.sum(np.abs(joints - seed))),
            ),
        )
        delta = target - seed
        if float(np.max(np.abs(delta))) <= math.pi + 1e-9:
            ik_counts["within_180"] += 1
        return delta

    candidate, reason = search_survey_pose(
        board_pose,
        tcp_T_cam,
        cameras,
        grippers,
        reference_camera="center_camera",
        current_base_T_tcp=current_base_T_tcp,
        coverage_targets=(sc_sector_corners(),),
        cross_rail_tilt_band_rad=(
            math.radians(policy["tilt_min_deg"]),
            math.radians(policy["tilt_max_deg"]),
        ),
        directional_tilt_axis_board=(1.0, 0.0, 0.0),
        max_along_rail_tilt_rad=math.radians(2.0),
        cross_rail_sign=0.0,
        require_all_cameras_frame=True,
        prefer_far_standoff=False,
        min_required_clearance_px=25.0,
        max_angular_motion_rad=math.pi,
        yaws_rad=tuple(math.radians(deg) for deg in range(-180, 180, 15)),
        aim_offset_board_m=(policy["aim_x_m"], 0.0, 0.0),
        standoffs_m=tuple(
            value
            for value in (
                0.50,
                0.55,
                0.58,
                0.60,
                0.62,
                0.65,
                0.68,
                0.70,
                0.73,
                0.76,
                0.80,
            )
            if (
                value >= policy["min_standoff_m"] - 1e-9
                and value <= policy["max_standoff_m"] + 1e-9
            )
        ),
        view_quality=view_quality,
        min_view_quality=policy["min_depth_cue_px"],
        view_quality_motion_tolerance=policy[
            "depth_cue_motion_tolerance_px"
        ],
        joint_motion=joint_motion,
        max_joint_motion_rad=math.radians(policy["max_joint_motion_deg"]),
        joint_motion_preference=lambda delta: abs(
            float(seed[5] + delta[5] - preferred_j6)
        ),
        max_joint_preference_error=math.inf,
        joint_preference_motion_tolerance_rad=math.radians(
            policy["j6_preference_motion_tolerance_deg"]
        ),
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
        "ik_counts": ik_counts,
    }
    if candidate is not None:
        selected_bore_margin = rectangular_bore_visibility_margin(
            sc_view_samples,
            board_pose.base_T_board,
            candidate.base_T_tcp,
            tcp_T_cam,
            half_width_x_m=policy["bore_x_tolerance_m"],
            half_width_y_m=0.0112,
            depth_m=0.01564,
            camera_names=tuple(cameras),
            required_camera_count=policy["required_bore_cameras"],
        )
        selected_delta = joint_motion(candidate.base_T_tcp)
        selected_target = (
            seed + np.asarray(selected_delta, dtype=float)
            if selected_delta is not None
            else None
        )
        result.update(
            standoff_m=candidate.standoff_m,
            depth_cue_px=candidate.view_quality,
            bore_margin=selected_bore_margin,
            cross_tilt_deg=math.degrees(candidate.cross_rail_tilt_rad),
            along_tilt_deg=math.degrees(candidate.along_rail_tilt_rad),
            clearance_px=candidate.min_clearance_px,
            joint_max_deg=math.degrees(candidate.max_joint_motion_rad),
            joint_total_deg=math.degrees(candidate.total_joint_motion_rad),
            seed_joints_deg=np.degrees(seed).tolist(),
            target_joints_deg=(
                np.degrees(selected_target).tolist()
                if selected_target is not None
                else None
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--tilt-min-deg", type=float, default=16.0)
    parser.add_argument("--tilt-max-deg", type=float, default=20.0)
    parser.add_argument("--aim-x-mm", type=float, default=0.0)
    parser.add_argument("--min-standoff-m", type=float, default=0.55)
    parser.add_argument("--max-standoff-m", type=float, default=0.62)
    parser.add_argument("--min-depth-cue-px", type=float, default=3.0)
    parser.add_argument(
        "--depth-cue-motion-tolerance-px", type=float, default=0.10
    )
    parser.add_argument("--required-bore-cameras", type=int, default=2)
    parser.add_argument("--min-bore-margin", type=float, default=0.0)
    parser.add_argument("--bore-x-tolerance-m", type=float, default=0.0076)
    parser.add_argument("--required-depth-cameras", type=int, default=2)
    parser.add_argument("--max-joint-motion-deg", type=float, default=185.0)
    parser.add_argument(
        "--j6-preference-motion-tolerance-deg", type=float, default=30.0
    )
    args = parser.parse_args()
    policy = {
        "tilt_min_deg": args.tilt_min_deg,
        "tilt_max_deg": args.tilt_max_deg,
        "aim_x_m": args.aim_x_mm / 1000.0,
        "min_standoff_m": args.min_standoff_m,
        "max_standoff_m": args.max_standoff_m,
        "min_depth_cue_px": args.min_depth_cue_px,
        "depth_cue_motion_tolerance_px": (
            args.depth_cue_motion_tolerance_px
        ),
        "required_bore_cameras": args.required_bore_cameras,
        "min_bore_margin": args.min_bore_margin,
        "bore_x_tolerance_m": args.bore_x_tolerance_m,
        "required_depth_cameras": args.required_depth_cameras,
        "max_joint_motion_deg": args.max_joint_motion_deg,
        "j6_preference_motion_tolerance_deg": (
            args.j6_preference_motion_tolerance_deg
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

    found = [result for result in results if result["found"]]
    missed = [result for result in results if not result["found"]]
    summary = {
        "cases": len(results),
        "found": len(found),
        "missed": len(missed),
        "standoff_range_m": (
            [min(r["standoff_m"] for r in found), max(r["standoff_m"] for r in found)]
            if found
            else None
        ),
        "depth_cue_range_px": (
            [min(r["depth_cue_px"] for r in found), max(r["depth_cue_px"] for r in found)]
            if found
            else None
        ),
        "bore_margin_range": (
            [min(r["bore_margin"] for r in found), max(r["bore_margin"] for r in found)]
            if found
            else None
        ),
        "clearance_range_px": (
            [min(r["clearance_px"] for r in found), max(r["clearance_px"] for r in found)]
            if found
            else None
        ),
        "joint_max_range_deg": (
            [min(r["joint_max_deg"] for r in found), max(r["joint_max_deg"] for r in found)]
            if found
            else None
        ),
        "misses": missed,
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
