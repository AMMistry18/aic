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
all three cameras and asks whether each is **visible**: inside the usable image
*and* not behind any camera's gripper silhouette.  A case passes only when a pose
was found and every seat survives both halves.

**Both halves, because each has already shipped a bug.**  The audit originally
scored image edges only, and that is how the 2026-07-28 17:03 field run got
through: at coverage half-span +/-0.1125 the outer seats sit 122-159 px *inside*
every frame -- so the harness reported 144/144 with room to spare -- while the +Y
one lands squarely behind the centre camera's tool silhouette and never reaches
IVM.  Edge margin was not the binding constraint, so measuring only edge margin
could not see it.

Headline numbers (144 cases: 8 board yaws x 2 tilts x 3 placements x 3 live
Stage-1 exits) at the current policy -- 25 px clearance, 180 deg reorientation,
12 rolls, standoffs floored at 0.70, summed travel capped at 400 deg, and the
per-seat gate active:

    coverage   found     passed    standoff     joint max     joint total
    sector      22/144    22/144   0.73-0.85     18-151deg      43-374deg
    narrow     138/144   138/144   0.76-0.90      9-174deg      36-347deg
    shipped    138/144   138/144   0.76-0.90      9-168deg      33-347deg

These are **strict-tier** numbers.  A case reported as no pose here is one the
skill recovers on its next relaxation tier, so 138/144 is a floor on availability,
not a prediction of failures.

Note what the seat gate does to the two regression rows: nothing passes with a
hidden seat any more, because the gate rejects those candidates and the search
simply steps back to a farther standoff until every seat is visible.  That is the
point of it -- and it means ``--coverage narrow`` no longer *reproduces* the
2026-07-28 17:03 failure, it fixes it a different way.  The regressions were
measured before the gate existed:

* the one-rail ``sector`` box clips a module in 96 of its 96 found poses, 35 of
  them holding only four of the six seats, worst seat 123.9 px outside the image;
* the +/-0.1125 ``narrow`` box frames all six seats -- 122-159 px inside every
  image -- and hides the +Y outer one behind the centre camera's tool silhouette
  in **81 of 81** swept placements, at 0.64-0.66 m standoff.

Reproducing either now needs the gate switched off, which this harness cannot yet
do; adding that switch is worth doing before the next policy change.

The summary separates ``seat_outside_an_image`` from ``seat_behind_the_tool``
because the fixes are opposite: framing wants a different standoff or aim,
occlusion wants a wider coverage box.
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
    SFP_COVERAGE_HALF_Y,
    SFP_FALLBACK_COVERAGE_HALF_Y,
    BoardPoseEstimate,
    Transform,
    project_points,
    sfp_module_detail_boxes,
    sfp_module_strip_corners,
    sfp_seat_bodies,
    sfp_sector_corners,
    search_survey_pose,
)


YAW_DEG = (0, 45, 70, 90, 140, 180, 250, 315)
TILT_DEG = (0, 10)
PLACEMENTS_M = ((0.0, 0.0), (0.050, 0.030), (-0.050, -0.040))
# Board origin in ``base_link``.
#
# Recovered from the 2026-07-28 hardware run: FK on the logged survey joint
# target reproduces the published pose to 0.4 mm, and its centre-camera axis at
# the logged 0.640 m standoff aims at (-0.5189, 0.2054, 0.0355) -- 0.558 m
# horizontally from base_link.  This harness had been pinned at 0.4317 m, 13 cm
# nearer than the cell it is supposed to certify, which is why a start pose that
# sweeps clean can still fail in the field: the survey viewpoint sits 0.64 m
# *above* the board, so board distance spends the arm's envelope directly.
#
# The origin is the aim point rather than the plate centre, so it carries the
# coverage centroid's ~0.07 m in-plane offset at unknown board yaw; the swept
# yaws and +/-50 mm placements already cover that.  ``--board-center-mm`` takes
# any other position, and the legacy pin is kept for comparing recorded numbers.
BOARD_CENTER_M = (-0.5189, 0.2054)
LEGACY_BOARD_CENTER_M = (-0.3445, 0.2602)
# Seat centres, for labelling audit output.  The bodies themselves come from
# ``board_stage2.sfp_seat_bodies`` so the harness and the skill cannot diverge.
SFP_SEAT_Y = tuple(
    float(box[:, 1].mean()) for box in sfp_module_detail_boxes()
)
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
    # Exactly what ``_coverage_targets_for_target`` ships for targets 0/1: the
    # +/-0.145 box that clears the tool, then the +/-0.1125 box that shipped, kept
    # only so availability cannot regress.
    "shipped": lambda p: (
        sfp_module_strip_corners(),
        sfp_module_strip_corners(SFP_FALLBACK_COVERAGE_HALF_Y),
    ),
    # The superseded single-rail box, kept so the 4-of-5 hardware regression
    # stays reproducible from this harness.
    "sector": lambda p: (sfp_sector_corners(),),
    # The +/-0.1125 box alone -- the 2026-07-28 17:03 regression, where all six
    # seats are framed and the outer one hides behind the tool.
    "narrow": lambda p: (
        sfp_module_strip_corners(SFP_FALLBACK_COVERAGE_HALF_Y),
    ),
    # Free exploration of the frontier: centred strip with a caller-chosen
    # half-span and board-X range (``--y-half-mm``, ``--x-min-mm/--x-max-mm``).
    "centered": lambda p: (
        _centered_span_corners(p["y_half_m"], p.get("x_range")),
    ),
}

# Standoff ladder, matching the skill.  The +/-0.145 coverage box cannot be framed
# and gripper-cleared nearer than ~0.75 m, so the default 0.30 m opening spent
# nine rungs of full-resolution mask work on poses that cannot exist.
STANDOFFS_M = (0.70, 0.73, 0.76, 0.80, 0.85, 0.90)

DEFAULT_POLICY = {
    "coverage": "shipped",
    "y_half_m": SFP_COVERAGE_HALF_Y,
    "max_obliquity_deg": 20.0,
    "min_clearance_px": 25.0,
    # Shipped values.  The cap is measured from the current TCP and therefore
    # selects the candidate set rather than bounding motion, so at 90 deg a
    # rolled-wrist start could leave only unreachable candidates in scope; 24
    # rolls then recover the board yaws the 7-sample family skips.
    "max_angular_motion_deg": 180.0,
    "max_joint_motion_deg": 225.0,
    "max_total_joint_motion_deg": 400.0,
    "seat_edge_margin_px": 12.0,
    "seat_gate": True,
    "aim_x_m": 0.0,
    "aim_y_m": 0.0,
    # 12, matching the skill: the grid is standoffs x 25 offsets x rolls, and 24
    # was measured costing 160 s in the field for the last 1-of-8 board yaws.
    "roll_count": 12,
    "board_center_m": BOARD_CENTER_M,
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


def _seats_visible(
    base_T_tcp, base_T_board, tcp_T_cam, cameras, grippers, edge_margin_px
):
    """The skill's ``_staged_seats_are_visible`` gate, for parity."""
    margins, blockers = _seat_audit(
        base_T_tcp, base_T_board, tcp_T_cam, cameras, grippers, edge_margin_px
    )
    return all(m >= 0.0 and not b for m, b in zip(margins, blockers))


def _arm_clear_of_own_cameras(base_T_tcp, joints, arm, tcp_T_cam, cameras):
    """The skill's staged-SFP arm-in-view rule: no limb anywhere in any image.

    Matching the deployed gate matters in both directions.  This harness applied
    the whole-image rule while the skill had relaxed to a sector-region keep-out,
    so its "N of 144" was a lower bound on a policy nobody was running.  Staged SFP
    is now back on the whole-image rule in the skill, and it is free: at the
    0.80-0.85 m standoff the wider coverage box forces, 144 swept cases select the
    same poses whether the keep-out is the coverage box, the rail column, the whole
    board face, or the entire image.  NIC and SC keep the region rule.
    """
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


def _seat_bodies():
    """The shared seat geometry -- ``board_stage2.sfp_seat_bodies``.

    This used to be a local copy, which is precisely how the harness and the skill
    came to disagree about what a "seat" is.  The skill now gates on the same
    function, so a change to the geometry moves both together or neither.
    """
    return sfp_seat_bodies()


def _seat_audit(base_T_tcp, base_T_board, tcp_T_cam, cameras, grippers, edge_margin_px):
    """Per-seat visibility: image-boundary margin **and** tool occlusion.

    Independent of whatever coverage target the search was given: this is the
    physical question "would a module sitting in this seat be visible in every
    camera".  Returns ``(margins, blockers)`` -- one image margin per legal seat,
    negative when the seat body leaves an image, and the cameras in which the
    gripper silhouette covers the seat.

    **The gripper term is why this harness exists in its current form.**  It used
    to score image edges only, and reported 144/144 with >100 px to spare while
    the field lost the outer module every run.  Reconstructed from the 2026-07-28
    17:03 pose, the +Y outer seat sits 122 px inside the frame and squarely behind
    the centre camera's tool silhouette.  Edge margin was never the binding
    constraint, so a harness that measured only edge margin could not see the bug
    it was built to catch.  Score both, always.
    """
    margins = []
    blockers = []
    for seat in _seat_bodies():
        worst = math.inf
        blocked = []
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
            mask = grippers[name].mask
            if mask is not None and any(
                0 <= int(round(float(vv))) < mask.shape[0]
                and 0 <= int(round(float(uu))) < mask.shape[1]
                and mask[int(round(float(vv))), int(round(float(uu)))]
                for uu, vv in pixels
            ):
                blocked.append(name)
        margins.append(
            worst - edge_margin_px if math.isfinite(worst) else -math.inf
        )
        blockers.append(sorted(blocked))
    return margins, blockers


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
        # The skill's seat gate: framing the coverage box is necessary and not
        # sufficient, so every candidate is checked against the module seats
        # themselves before any IK work.  ``--no-seat-gate`` removes it, which is
        # how both shipped regressions stay reproducible from this harness -- with
        # the gate on, the search simply steps back until the seats are visible.
        if policy["seat_gate"] and not _seats_visible(
            pose,
            board_pose.base_T_board,
            tcp_T_cam,
            cameras,
            grippers,
            policy["seat_edge_margin_px"],
        ):
            return None
        solutions = arm.solve_ranked(pose, seed)
        clear = [
            target
            for target in solutions
            if _arm_clear_of_own_cameras(pose, target, arm, tcp_T_cam, cameras)
        ]
        # Hard stop: every branch must be arm-clear, because Move Robot
        # re-solves the published Cartesian pose and may take any of them.
        if not clear or len(clear) != len(solutions):
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
        # Summed six-joint travel, now gated for SFP as well as SC: the worst-joint
        # cap says nothing about three joints swinging 170 deg at once, which is
        # the "arm contorts and gets between the cameras and the board" report.
        # This harness measures the *strict* tier only, so a case it marks as no
        # pose is one the skill would recover on its next relaxation tier.
        max_total_joint_motion_rad=math.radians(
            policy["max_total_joint_motion_deg"]
        ),
        max_reach_m=0.85,
        min_height_m=0.02,
        standoffs_m=STANDOFFS_M,
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
        selected_delta = joint_motion(candidate.base_T_tcp)
        margins, blockers = _seat_audit(
            candidate.base_T_tcp,
            board_pose.base_T_board,
            tcp_T_cam,
            cameras,
            grippers,
            policy["seat_edge_margin_px"],
        )
        # A seat counts as covered only when it is inside every image *and* no
        # camera's tool silhouette sits on it.
        covered = [m for m, b in zip(margins, blockers) if m >= 0.0 and not b]
        framed = [m for m in margins if m >= 0.0]
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
            joint_total_deg=math.degrees(candidate.total_joint_motion_rad),
            # Per-joint target/seed, so the selected branch can be audited
            # against the *workcell's* real position limits rather than the
            # ones ``arm_ik.JOINT_LIMITS`` assumes.  A branch this model
            # believes is legal but the planner cannot use does not fail --
            # Move Robot re-solves the same Cartesian pose on another branch,
            # which is how a survey move becomes a whole-arm reconfiguration.
            seed_joints_deg=np.degrees(seed).tolist(),
            target_joints_deg=(
                np.degrees(seed + selected_delta).tolist()
                if selected_delta is not None
                else None
            ),
            seats_covered=len(covered),
            seats_framed=len(framed),
            seat_margin_min_px=(
                min(margins) if all(math.isfinite(m) for m in margins) else None
            ),
            seat_margins_px=[
                round(m, 1) if math.isfinite(m) else None for m in margins
            ],
            seats_occluded_by_tool={
                f"{y * 1000:+.0f}": b
                for y, b in zip(SFP_SEAT_Y, blockers)
                if b
            },
        )
        result["pass"] = len(covered) == len(margins)
    else:
        result["pass"] = False
        result["seats_covered"] = None
        result["seats_framed"] = None
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
             "4-of-5 one-rail regression; 'narrow' reproduces the 17:03 "
             "tool-occlusion regression.",
    )
    parser.add_argument(
        "--y-half-mm", type=float, default=SFP_COVERAGE_HALF_Y * 1000.0
    )
    parser.add_argument("--x-min-mm", type=float, default=None)
    parser.add_argument("--x-max-mm", type=float, default=None)
    parser.add_argument("--max-obliquity-deg", type=float, default=20.0)
    parser.add_argument("--min-clearance-px", type=float, default=25.0)
    parser.add_argument(
        "--max-angular-motion-deg",
        type=float,
        default=DEFAULT_POLICY["max_angular_motion_deg"],
    )
    parser.add_argument("--max-joint-motion-deg", type=float, default=225.0)
    parser.add_argument(
        "--max-total-joint-motion-deg",
        type=float,
        default=DEFAULT_POLICY["max_total_joint_motion_deg"],
        help="Summed six-joint travel cap. Pass a large value to measure the "
             "un-gated behaviour the SC-only cap used to leave SFP with.",
    )
    parser.add_argument("--seat-edge-margin-px", type=float, default=12.0)
    parser.add_argument(
        "--no-seat-gate",
        action="store_true",
        help="Drop the per-seat visibility gate, leaving only coverage-box "
             "framing. Reproduces both shipped regressions: with '--coverage "
             "sector' the one-rail crop, with '--coverage narrow' the 17:03 "
             "tool-occlusion run.",
    )
    parser.add_argument("--aim-x-mm", type=float, default=0.0)
    parser.add_argument("--aim-y-mm", type=float, default=0.0)
    parser.add_argument(
        "--roll-count", type=int, default=DEFAULT_POLICY["roll_count"]
    )
    parser.add_argument(
        "--board-center-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Board origin in base_link, millimetres.  Default is the "
             "measured hardware position; pass -344.5 260.2 for the legacy "
             "harness pin the recorded numbers were taken at.",
    )
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
        "max_total_joint_motion_deg": args.max_total_joint_motion_deg,
        "seat_edge_margin_px": args.seat_edge_margin_px,
        "aim_x_m": args.aim_x_mm / 1000.0,
        "aim_y_m": args.aim_y_mm / 1000.0,
        "roll_count": args.roll_count,
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
    clipped = [r for r in found if not r["pass"]]
    summary = {
        "coverage": args.coverage,
        "cases": len(results),
        "pose_found": len(found),
        "passed": len(passed),
        "no_pose": len(results) - len(found),
        "lost_a_seat": len(clipped),
        # Split the two ways a seat is lost.  They look identical downstream --
        # one module missing from IVM -- and need opposite fixes: framing wants a
        # different standoff or aim, tool occlusion wants a wider coverage box.
        "seat_outside_an_image": sum(
            1 for r in found if r["seats_framed"] != len(SFP_SEAT_Y)
        ),
        "seat_behind_the_tool": sum(
            1
            for r in found
            if r["seats_framed"] == len(SFP_SEAT_Y)
            and r["seats_covered"] != len(SFP_SEAT_Y)
        ),
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
        "joint_total_range_deg": (
            [
                min(r["joint_total_deg"] for r in found),
                max(r["joint_total_deg"] for r in found),
            ]
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
                    "seats_framed",
                    "seat_margins_px",
                    "seats_occluded_by_tool",
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
