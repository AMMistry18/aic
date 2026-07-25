# Board-search handoff

Updated: 2026-07-25

## Pinned implementation

The current implementation is the **insignia-driven deterministic survey**. It
supersedes the outline-PnP geometric Stage 2 pinned at `525eb40`. Record the
exact commit here when this change merges; until then the pin is the working
tree on `main`. Legacy adaptive search is now Stage 1 only -- the fallback that
runs when the insignia is not in view at all. All deployed `survey_target`
values (SFP, NIC, SC) route to the geometric sector survey.

Note on branches: the sector-survey work was authored on
`agent/sync-model-overlay-and-skill-bundling` (`252e8ac`, `f33ce83`) while the
fixed build script and the current teammate work landed on `main`. The working
tree carries the union; reconcile the branches before pinning a commit here.

Authoritative modules:
- `flowstate/aic_perception/aic_perception/board_visibility.py`
  (`detect_insignia_polygon`)
- `flowstate/aic_perception/aic_perception/board_stage2.py`
  (`estimate_board_pose_from_insignia`, `board_coverage_corners`,
  `module_coverage_corners`, `sfp_sector_corners`, `sc_sector_corners`,
  `nic_sector_corners`, `search_survey_pose`, `verify_survey_view`)
- `flowstate/aic_perception/check_board_visibility_skill.py`
  (`_stage2_landmarks`, `_execute_inner`, `_uses_geometric_survey`,
  `_sector_for_target`, `_run_sfp_geometric_stage2`)

## Behavior contract

Stage 1 is a short, low-constraint exposure search. It has **no wall-clock
deadline**: the planner terminates on its own stall condition and every move is
force- and per-move-timeout-guarded. For staged SFP modules, as soon as the
insignia is cleanly visible in a calibrated camera (or on the planner's own
`DONE`/terminal), Stage 1 hands its freshest triplet to Stage 2. This
guarantees Stage 2 always runs; it is never pre-empted by a timeout.

Stage 2 is deterministic:

- it consumes exact CameraInfo intrinsics and image-timestamped TCP/camera TF;
- it estimates the full 6-DoF board pose by planar PnP of the **asymmetric
  purple insignia** (bracket corners against `INSIGNIA_RECT_CORNERS`, the mask
  centroid resolving the rectangle ambiguity). This is clip-proof: it does not
  require a fully visible plate outline or a "full" Stage-1 report;
- it computes one board-relative TCP survey pose by inverting the production
  three-camera URDF rig, searching standoff, both board-plane offsets, look
  direction, and roll, filtered by the execution workspace (reach 0.85 m, the
  UR5e envelope; height 0.02 m) and sampled Cartesian path clearance. Height
  and lateral placement both fall out of the estimated `base_T_board`, so the
  pose tracks a board that moves or tilts;
- coverage is **per sector**, selected by `survey_target`: SFP modules (0/1),
  NIC cards (2), SC ports (3). Each sector is a board-frame box covering that
  component group's full rail travel; the whole sector must be framed in all
  three cameras, because IVM pose estimation needs every camera to see it. The
  only per-camera acceptance is target-in-frame plus positive gripper
  clearance;
- selection is tuned for **the way the IVM reads each part**. The SFP pick
  modules use the standard **all-camera** near-overhead framing at ~0.66 m,
  closest-standoff-wins. NIC cards and SC ports are detected by looking straight
  **down into their recessed ports** -- the ports face out essentially along the
  board normal (within ~1 degree), so a top-down view looks down the port axis
  and the recess shows its **full depth**, which is what the model match needs;
  a tilted view foreshortens the recess (it reads shallow) and the part is
  missed. Two consequences: (a) the three splayed wrist cameras cannot frame the
  tall protruding cards together at all, so these sectors require only the
  **center camera** to frame the sector (`require_all_cameras_frame=False`);
  (b) the pose is placed **high, not close** (`prefer_far_standoff=True`, reach
  capped at 0.80 m -> ~0.65-0.90 m standoff, camera ~0.9-1.0 m up, in the
  0.5-1 m optimal band). Height matters because the 145 mm cards protrude toward
  the lens: up close they foreshorten and the edge cards' ports are seen
  off-axis (partial depth) while the tool occludes an end card; a high, near-
  orthographic view shows every port's full depth undistorted with the tool
  clear. The NIC sector box also spans the **full protruding card height**
  (board Z to 0.175) so the cages -- at Z ~= 0.13 -- are framed, not just the
  mount bases. (`search_survey_pose` also supports a rail-aware **cross-rail
  tilt** band for parts whose pose needs side-on depth cues, but the
  recessed-port NIC/SC detector deliberately does not use it.)
- it allows at most 45 degrees of orientation change and performs any
  meaningful wrist reorientation only after retreating beyond a conservative
  0.40 m rig sweep radius; and
- the skill is **perception-only**: it publishes the result as a native
  `intrinsic_proto.Pose` on `result.survey_pose` (with `result.target_frame =
  base_link`) for a downstream Move Robot Cartesian target, and does not move
  to the survey pose itself. There is **no** aggregate Stage-2 time budget and
  **no** two-triplet consistency gauntlet.

Any calibration, geometry, reach, path, or confirmation failure returns
`success=true, done=false` so the Flowstate process can decide whether to retry.
Cancellation still uses the process cancellation path; every motion is
force-guarded.

## Reachability gate + bore commitment (2026-07-25)

This supersedes the "reach capped at 0.80 m" and "deliberately does not use
cross-rail tilt" notes above for the NIC/SC sectors.

**Problem.** The survey search judged a candidate TCP pose reachable by a single
base-origin sphere (`norm(base_T_tcp) <= max_reach`). That is not what the arm
can do: it both **admitted kinematically-impossible poses** (Move Robot then
reported `IK not computable`, e.g. at a ~140 deg board yaw) and **rejected
genuinely reachable far, bore-facing poses**, so when the recessed ports faced
away from the base the search settled for a near pose that framed the cards from
the wrong (closed-back) side and the IVM could not read the ports.

**Real reachability.** New module `flowstate/aic_perception/aic_perception/
arm_ik.py` provides exact forward kinematics for the workcell UR5e (the chain
taken verbatim from `aic_utils/aic_mujoco/mjcf/aic_robot.xml`) and the
**closed-form analytic** IK for that chain, used as a boolean reachability gate:
"does a joint-limit-valid solution exist?". The MJCF chain is bit-for-bit the
classical UR5e DH chain (asserted by
`test_arm_ik.test_mjcf_chain_is_the_classical_ur5e_dh_chain`), so the textbook
eight-branch UR solution applies directly with no adapter transform. Analytic,
not iterative, for two reasons: the verdict is exact in both directions (no
seeds, no local minima -- a rejection means the pose is genuinely outside the
workspace), and at ~0.2 ms it is cheap enough to gate *every* framed candidate. The `base_link<->kinematic-base` convention *and* the flange->TCP tool offset
are **recovered together from the live, static (joint-state, base_T_tcp)
sample** by `UR5eArm.autocalibrate`: it tries each candidate base rotation and
keeps the one whose resulting tool offset is physically plausible (~0.15-0.30 m,
roughly along the flange axis). This is necessary because the workcell `base_link`
TF differs from the UR kinematic base by the classic **180-deg-about-Z flip** --
without correcting it the recovered "tool offset" was a nonsensical 634.6 mm
(exactly `2*horizontal-flange-distance`, the signature of the Rz(180) mismatch)
and the gate fell back to the sphere, which then published a pose Move Robot
rejected as unreachable. If no candidate is plausible (or joints are
unavailable) the skill logs every candidate's offset and falls back to the
sphere. Only
the robot's own kinematics are used -- no task-board TF -- so the permitted-TF
policy is unchanged. `search_survey_pose` takes an optional
`reachable(base_T_tcp) -> bool`; it collects every framed, gripper-clear
candidate, ranks them, and commits to the **best-ranked pose that is actually
reachable**, returning `framed N but none reachable` when applicable. The IK is
intentionally *more lenient* than Move Robot (full joint limits, no collision),
so anything it rejects is genuinely unsolvable -- it cannot regress a
previously-working pose, and a rejection yields a graceful `done=false`, never a
bad move.

**The gate must scan the whole ranking, not a shortlist.** This is what broke the
first deployment (`255 framed ... but none had a reachable joint-limit-valid IK
solution` -> `done=false` -> Move Robot crashed on the empty pose). For the NIC
sector, framing all five cards in the centre camera needs a 0.66-1.3 m standoff,
and the board sits at the height of the arm's own base -- so **only the closest
handful of framed poses are inside the UR5e envelope**, while
`prefer_far_standoff` ranks the far ones *first*. Offline, at the nominal board
placement, the single reachable pose sat at rank 262 of 263; a gate that checked
only the top 24 was guaranteed to miss it. Gating the full list restores the
intent of `prefer_far_standoff` -- "the farthest standoff the arm can actually
reach" -- and finds a pose in 96/96 swept scenarios (board yaw 0-315 deg, tilt
0-10 deg, +/-50 mm placement, two Stage-1 exit poses).

**NIC looks straight DOWN the port bores -- the cross-rail tilt was backwards.**
Measured from `aic_world.xml`: each NIC SFP port is a **16 x 12 mm aperture at the
top of a 45.8 mm recess whose axis points straight up** -- board-frame bore axis
`(0.001, -0.013, -0.9999)`, i.e. **0.7 deg off the board normal**, entrance at
board Z 0.1793. So a port only shows the black depth the IVM keys on to a ray
within `atan(6/45.8) = 7.5 deg` of the board normal; past that the cage wall
occludes the backstop and the port reads as a flat grey rectangle. The old code
assumed the mouths opened sideways toward board -X and deliberately tilted the
camera 12-22 deg onto that side. Scored against the port cone, that committed
band resolved **0 of 10 ports** at the board yaws where it was reachable. NIC
therefore leaves `_bore_view_tilt_bands` (returns `(None,)`) and takes
`max_obliquity_rad = 2 deg` instead. Only **SC** keeps the cross-rail tilt: its
ports really are recessed sideways.

Two more consequences, both now in `_survey_view_settings`:

* **`nic_sector_corners()` is centred on the ten port entrances**, not on the card
  bodies. The search aims the optical axis at the sector centroid, and the old
  box's centroid sat 16 mm off the port cluster -- enough to push the outermost
  port past the 7.5 deg cone, worth two of the ten ports.
* **`prefer_far_standoff` is right here for a real reason.** The ten ports span
  160 mm, so the outermost sits `atan(0.081/d)` off axis and needs `d >= 0.62 m`
  above the port plane. Preferring the *nearest* framing resolves only 6/10.

Scored over 96 scenarios in isolation (no collision model): **96/96 poses found,
all ten ports resolved in 96/96**, worst ray 7.2 deg, standoff 0.66-0.80 m. All
three cameras frame the sector, which needs `min_required_clearance_px=25` rather
than the 40 px default -- the gripper mask already dilates the silhouette by
32 px, so the cards still stay 57 mm-equivalent clear of true gripper pixels.

**The gate must know about the wrist cameras.** A purely kinematic gate published
a pose the workcell planner then refused outright: `IK could not find a collision
free configuration ... robot.forearm_link vs left_camera.camera_link`, every
solution colliding. The three cameras stick ~108 mm off the flange axis and swing
onto the forearm whenever the wrist folds back. `UR5eArm` therefore takes
`flange_T_probes` (the camera extrinsics the skill already has from permitted TF)
and `min_self_clearance_m`; `reachable()`/`solve()` now require a branch that
keeps every probe clear of the elbow->wrist_1 segment. Calibrated against that
failure: the refused pose's best branch put a camera **111 mm** from the forearm
centreline, so the threshold is **140 mm** (26% margin).
`test_wrist_camera_keep_out_rejects_a_pose_the_workcell_planner_refused` pins it,
and also confirms the planner's own configurations reproduce the published TCP
through this model **to 0.0 mm** -- the first end-to-end validation of the DH
chain, `base=Rz180` and the 197.1 mm tool against the real robot.

Adding the collision keep-out costs poses: at the default 45 deg reorientation
cap / 7-roll sample it drops to 82/96 -- some orientations have no
collision-clear candidate within that budget. NIC therefore widens its own
search (`_survey_view_settings`, target 2 only -- SFP/SC untouched):
`max_angular_motion_rad=90 deg`, `yaws_rad` = 24 values (15 deg steps) instead of
the 7-value/45-deg default. **Final measured result, full production settings
including the collision gate: 90/96 poses found, 80/90 resolving all ten ports**,
camera-forearm clearance 141-207 mm, reorientation used 0-90 deg. The 6 misses
cluster at board yaw 45 deg and 70 deg (jittered placement); confirmed a genuine
geometric conflict, not a sampling gap -- even a 72-value (5 deg) roll sweep at
yaw 45 tops out at 140.7 mm of clearance, and that one candidate still only
resolves 6/10 ports. The fixed camera-rig splay (yaws 90/30/150 deg) and that
board orientation leave a near-empty intersection of reachable / collision-clear
/ port-cone poses. Not worth shaving the ground-truth-calibrated 140 mm threshold
to chase it. This is exactly what the graceful `done=false` path plus the
still-outstanding BT `result.done` gate exist for.

**Bore commitment (SC only).** `_bore_view_tilt_bands` returns a committed
`(12 deg, 22 deg)` cross-rail band then a flat `(0, 22 deg)` fallback for the SC
sector, whose ports are recessed sideways; `cross_rail_sign=-1` keeps the camera
on the board -X (mouth) side, which is board-relative and tracks the insignia. The
0.80 m NIC reach cap is removed; reach is decided by the IK gate, with the 0.85 m
sphere surviving only as a loose fallback. The published log line reports
`bore_band=` so a run shows which tier won (`none` for NIC).

Tests: `test/test_arm_ik.py` (FK/IK round-trip, calibration, limits, rejection),
reachability-gate plumbing in `test/test_board_stage2.py`, and updated source
guards in `test/test_check_board_visibility_stage2_integration.py`. **Not yet
hardware-validated** -- on the next run watch for the log
`arm IK reachability gate active: base=Rz180 tool=NNNmm axial=1.00` (confirms
autocalibrate found the base convention and the gate is live), and on any
failure the `arm IK calibration failed (...)` candidate list or the
`framed ... but none ... reachable` reason.

## Latest deployment

The insignia-driven implementation was built and installed **in place** as the
existing `check_board_visibility_skill_v4` asset (not a new skill) into Flowstate
solution `9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH` ("Work on this",
org `tar-2@xfa-prod-aic-us`) on 2026-07-23 as:

```text
ai.tar2.check_board_visibility_skill_v4.0.0.1+aac828ea836a05056fc5ab0b1fe10a6fcc112ec7c18ce7bb0a317b81c9dc99f6
```

Provenance: built from the edited working tree via
`flowstate/scripts/build_check_board_visibility_skill.sh` (full colcon rebuild,
gRPC smoke test passed). Bundle SHA-256
`ca45a13cac0ce099f664f1dbcddd9a39754fd93a4284ff5a694dab0dffaade44`. It replaces
the prior outline-PnP build `...+03b1f018...`; the outline-clip failure mode is
eliminated because pose is driven by the insignia rather than the plate outline.
The change is not yet committed to git -- record the commit under Pinned
implementation when it lands. Build note: strip CRLF from
`flowstate/scripts/*.sh` before running the build on a Windows-checkout workspace.

## Authoritative source

- `aic_model/aic_model/board_search.py`
- `aic_model/test/test_board_search.py`
- `flowstate/aic_perception/`
- `flowstate/resources/`
- `flowstate/scripts/`
- `deploy/flowstate/aic_model_v38.manifest.textproto`
- `scripts/flowstate/inctl.sh`

To verify that the implementation files have not drifted, diff against the
commit recorded under Pinned implementation once this change has merged:

```bash
git diff --exit-code <pinned-commit> -- flowstate/aic_perception
```

## Build and install

Use a Linux/AMD64 workspace with `src/aic` and `src/sdk-ros`:

```bash
cd ~/ws_aic_phase1
bash src/aic/flowstate/scripts/build_check_board_visibility_skill.sh
```

Install the generated bundle only after re-reading the active cluster:

```bash
inctl asset install \
  --org tar-2@xfa-prod-aic-us \
  --cluster "$CLUSTER" \
  images/check_board_visibility_skill/check_board_visibility_skill.bundle.tar
```

Recommended serial wiring is:

```text
Move Robot
-> Switch To AIC Controller
-> Check Board Visibility
-> Switch To Default Controller
-> require result.success && result.done
-> downstream IVM
```

Do not run another motion session in parallel with this skill.

## Validation

Run the model helper test and the complete Flowstate package suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="aic_model:${PYTHONPATH}" \
  .pixi/envs/default/bin/python -m pytest -q \
  aic_model/test/test_board_search.py \
  flowstate/aic_perception/test
```

The insignia-driven implementation passes 224 Flowstate perception tests
(216 prior + 8 new insignia-PnP / two-tier-coverage cases). Any intentional
board-search change must update this handoff and its pinned implementation.
