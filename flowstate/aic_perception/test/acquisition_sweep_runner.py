#!/usr/bin/env python3
"""Backtest deterministic Stage 1 over the production 144-case matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import itertools
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
    INSIGNIA_RECT_CORNERS,
    Transform,
    evaluate_camera_coverage,
    project_points,
)
from aic_perception.stage1_acquisition import (  # noqa: E402
    OBSERVATION_JOINTS_RAD,
    validate_observation_path,
)


YAW_DEG = (0, 45, 70, 90, 140, 180, 250, 315)
TILT_DEG = (0, 10)
PLACEMENTS_M = ((0.0, 0.0), (0.050, 0.030), (-0.050, -0.040))
HOME_DEG = np.array([-9.15, -77.59, -95.39, -97.02, 90.01, 80.84])
LIVE_STARTS = (
    np.radians(HOME_DEG),
    np.radians(HOME_DEG + np.array([0, 0, 0, 0, 0, 90.0])),
    np.radians([-17.5, -95.8, -19.5, -143.9, 82.9, 26.8]),
)


def _arm_clear_of_own_cameras(
    base_T_tcp, joints, arm, tcp_T_cam, cameras
):
    for start, end, radius in arm.link_segments(joints):
        samples = np.array(
            [start + (end - start) * t for t in np.linspace(0.0, 1.0, 25)]
        )
        for name, camera in cameras.items():
            local = (
                base_T_tcp.compose(tcp_T_cam[name]).inverse().apply(samples)
            )
            pixels, in_front = project_points(local, camera)
            for pixel, ahead, point in zip(pixels, in_front, local):
                if not ahead or not np.all(np.isfinite(pixel)):
                    continue
                margin = radius * camera.fx / max(float(point[2]), 1e-6)
                if (
                    -margin <= pixel[0] <= camera.width + margin
                    and -margin <= pixel[1] <= camera.height + margin
                ):
                    return False
    return True


def _run_case(case):
    yaw_deg, tilt_deg, placement_m, start_index = case
    cameras, tcp_T_cam, grippers = _production_camera_rig()
    arm = UR5eArm(
        flange_T_tcp=Transform(
            np.eye(3), np.array([0.0, 0.0, 0.1971])
        ),
        base=Transform(_rot_z(math.pi), np.zeros(3)),
    )
    arm = replace(
        arm,
        flange_T_probes=tuple(
            arm.flange_T_tcp.compose(extrinsic)
            for extrinsic in tcp_T_cam.values()
        ),
    )
    rotation = axis_angle_rotation(
        [1, 0, 0], math.radians(tilt_deg)
    ) @ axis_angle_rotation([0, 0, 1], math.radians(yaw_deg + 2.7))
    board = Transform(
        rotation,
        np.array(
            [
                -0.3445 + placement_m[0],
                0.2602 + placement_m[1],
                0.0,
            ]
        ),
    )
    start = LIVE_STARTS[start_index]
    target = np.asarray(OBSERVATION_JOINTS_RAD)
    target_tcp = arm.fk(target)
    path = validate_observation_path(
        arm,
        start,
        board_transforms=(board,),
        endpoint_arm_clear=lambda pose, joints: _arm_clear_of_own_cameras(
            pose, joints, arm, tcp_T_cam, cameras
        ),
    )

    visible = []
    for name, camera in cameras.items():
        cam_T_board = (
            target_tcp.compose(tcp_T_cam[name]).inverse().compose(board)
        )
        coverage = evaluate_camera_coverage(
            INSIGNIA_RECT_CORNERS,
            None,
            cam_T_board,
            camera,
            grippers[name],
            edge_margin_px=15.0,
            required_clearance_px=8.0,
            min_pixel_scale=0.0,
            module_envelopes_board=(),
            min_module_pixel_scale=0.0,
        )
        if coverage.feasible:
            visible.append((name, coverage.clearance))
    return {
        "yaw_deg": yaw_deg,
        "tilt_deg": tilt_deg,
        "placement_m": placement_m,
        "start_index": start_index,
        "path_safe": path.safe,
        "path_reason": path.reason,
        "visible": bool(visible),
        "best_clearance_px": max(
            (clearance for _, clearance in visible), default=-math.inf
        ),
        "worst_joint_deg": math.degrees(path.travel.worst_rad),
        "total_joint_deg": math.degrees(path.travel.total_rad),
        "min_board_clearance_m": path.min_board_clearance_m,
    }


def run_sweep(workers: int = 4):
    cases = list(
        itertools.product(
            YAW_DEG,
            TILT_DEG,
            PLACEMENTS_M,
            range(len(LIVE_STARTS)),
        )
    )
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_run_case, cases))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    results = run_sweep(args.workers)
    safe = [result for result in results if result["path_safe"]]
    visible = [
        result
        for result in results
        if result["path_safe"] and result["visible"]
    ]
    print(
        f"deterministic Stage 1: {len(visible)}/{len(results)} acquired, "
        f"{len(safe)}/{len(results)} safe"
    )
    print(
        "worst physical margins: "
        f"image={min(r['best_clearance_px'] for r in visible):.1f}px "
        f"board={min(r['min_board_clearance_m'] for r in safe):.3f}m "
        f"max_joint={max(r['worst_joint_deg'] for r in safe):.1f}deg "
        f"total_joint={max(r['total_joint_deg'] for r in safe):.1f}deg"
    )
    failures = [
        result
        for result in results
        if not result["path_safe"] or not result["visible"]
    ]
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
