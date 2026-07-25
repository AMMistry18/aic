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

## SC destination ports -- full reference (2026-07-25)

SC was rebuilt from scratch this session after being reported "completely
broken". Everything below is measured from the workcell model or from field
runs; where something is inferred rather than measured it says so.

### Intrinsics -- the five SC adapters

Authoritative source: upstream `aic_description/urdf/task_board.urdf.xacro`
(branch `phase_1`). **Our copy was stale and defined only two ports**; it has
been synced. Note the sync also **moved `sc_port_1` from board Y +0.0705 to
+0.0295** -- downstream SC perception code that assumed the old layout must be
rechecked.

| port | rail | board X | board Y | board Z |
|---|---|---|---|---|
| `sc_port_0/1/2` | SC_RAIL_0 | `-0.075 + t` | **+0.0295** | 0.0165 |
| `sc_port_3/4` | SC_RAIL_1 | `-0.075 + t` | **+0.0705** | 0.0165 |

* rail translation `t` in **-0.060 .. +0.055** (`task_board_limits.sc_rail`), so
  the cluster spans board X **-0.135 .. -0.020**, i.e. 115 x 41 mm over the two
  rows, with the five entrances coplanar at **board Z 0.0301**.
* adapter body: **9.3 (X) x 25.8 (Y) x 27.4 (Z) mm**, spanning board Z
  0.0028..0.0302.
* **bore axis (0.001, 0.000, -0.9999) in board frame -- 0.00 deg off the board
  normal.** The ports open *straight up*; the bore runs 15.64 mm *down* from the
  entrance. The pre-existing code assumed they were recessed *sideways* and
  tilted the camera 12-22 deg across the rail onto the board -X face. That is
  almost certainly why the sector never worked.
* receptacle opening **7.6 x 22.4 mm**, derived from the adapter's own collision
  primitives (side walls at local |x| 12.05 mm x 1.69 mm thick; plates at local
  z +/-4.2 mm). Limiting geometric cone `atan(3.8/15.64)` = **13.7 deg** across
  the narrow axis (35.6 deg across the wide one).
* the adapters are **blue**: the shipped texture is 88.1% saturated blue
  (mean BGR 211/137/27, hue ~102). A colour mask is viable as a *diagnostic* but
  the SC **plug** housings are blue too (`housing blue` geoms in
  `aic_assets/models/SC Plug/model.sdf`), so any blob count needs board-region
  and size gating. Not implemented -- IVM does the detection.

Relative to NIC, SC should be the easier sector: **149 mm lower** on the board
(entrances at Z 0.0301 vs 0.1793), a **13.7 deg** cone vs 7.5, and the worst
port only 61 mm off the cluster centre vs 81. Reach and the wrist-camera
keep-out are not binding here.

### How the survey pose is chosen today

`_survey_view_settings(3)` in `check_board_visibility_skill.py`:

| setting | value | why |
|---|---|---|
| `cross_rail_tilt_band_rad` | `None` | bores open along the normal; any tilt occludes them |
| `cross_rail_sign` | `0.0` | no side commitment |
| `require_all_cameras_frame` | **`True`** | IVM needs all three cameras framing the sector (confirmed by the user) |
| `prefer_far_standoff` | `False` | cone is met from 0.27 m, so take the pixels |
| `max_obliquity_rad` | `8 deg` | keep the view down the bores |
| `min_required_clearance_px` | `25` | gripper keep-out margin |
| `max_angular_motion_rad` | `90 deg` | inherited from the NIC recipe |
| `yaws_rad` | 24 values, 15 deg | roll rotates the sector clear of the gripper silhouette |
| `standoffs_m` | `0.50 .. 0.80` | nearest-first inside the band |

`sc_sector_corners()` is centred on the five **entrances**, not the adapter
bodies -- the search aims the optical axis at the box centroid, and the old box
(X -0.14..-0.01, Y -0.02..0.10, Z 0.01..0.05) predated the 3-port row and swept
in ~47 mm of empty board on the -Y side, pulling the aim point 10 mm off.

Typical output: standoff 0.58-0.62 m, obliquity <8 deg, TCP at base z ~0.38 m,
gripper clearance +38..+71 px in all three cameras, all five ports inside the
cone. Offline that is 96/96 scenarios (board yaw 0-315 deg x tilt 0-10 deg x
+/-50 mm placement x two Stage-1 exit poses).

### The arm can stand in its own cameras -- and does

The gripper keep-out is a **fixed image-space silhouette**. That is correct only
for geometry rigidly attached to wrist_3. The **upper arm and forearm are
upstream of the wrist**, so where they land in the image depends entirely on the
joint configuration, and no static mask can represent them.

Measured on the deployed build: at board yaw **70 deg** -- the orientation
reported as still failing -- the published pose puts the **upper arm in the
centre camera** (and clipping the left), while every existing check passes:
obliquity 0.0 deg, collision-free, gripper-clear, fully framed. At yaw 90 and
250 deg arm links land in **all three** cameras.

Fix: `UR5eArm.link_segments()` returns the upper arm and forearm as base-frame
capsules (60 mm / 50 mm tubes), and `_arm_clear_of_own_cameras()` rejects any
pose whose IK solution projects a link into any camera. Cost: 1 of 8
orientations (yaw 90 loses its pose); yaw 70/140/250 move to a clean pose.
**Approximate by construction** -- it checks the IK branch nearest the current
joints, and Move Robot may pick another.

### What was tried and reverted -- do not repeat

**Closer standoff with reference-camera-only framing (0.40-0.45 m).** Motivated
by resolution: at 0.6 m the bore spans only 15-17 px (NIC's cage gets 20-25 px)
and IVM resolved 2 of 5 ports. All-camera framing is what pins the standoff at
~0.6 m -- it cannot be met closer regardless of the floor -- so the requirement
was dropped and the band lowered. **This broke the run.** With only the
reference camera checked, the tool sat *on top of* the ports in both side
cameras (gripper clearance **-13 to -32 px**) while the centre camera reported a
healthy +58 px; and the 0.45 m pose put the TCP at base z 0.24 m, reachable only
through a contorted configuration.

The flawed check behind it: all five entrances were verified to **project inside
the image bounds** in all three cameras, and that was taken as proof coverage
was not lost. **In-frame is not unoccluded.** Any future attempt to relax
all-camera framing must test gripper clearance per camera, not projection.

**Clearance-bucketed ranking.** An attempt to stop the objective trading a 90 deg
wrist roll for a few px of extra clearance. It did not change the outcome
(standoff dominates the objective, and at the nearest framing standoff only
rolled poses clear the gripper) and it perturbed NIC/SFP ranking, so it was
reverted.

### Open issues

1. **The transit path.** Move Robot is reported taking a violent route to the
   pose. The skill publishes a Cartesian pose only; Move Robot owns the joint
   path and configuration choice. Related: an earlier run reached a NIC pose
   through a ~360 deg joint-6 swing -- the planner picked the co-terminal branch
   (`226.2 deg` vs `-133.8 deg`). Capping `max_angular_motion_rad` is the lever
   on our side; measured cost: 90 -> 60/45 deg takes SC from 7/8 to 5/8
   orientations. Not applied -- it is a guess at a problem one layer down.
2. **Does the detector actually want a top-down view?** The whole SC view is
   built on "look down the bore so it shows its black depth". The user reports
   70 deg still failing *while looking straight down with full port depth*. If
   that survives the arm-in-view fix, the premise is wrong: `best_sc_pose.pt`
   labels an 8.8 x 6.0 mm rectangle at the **mouth**, not the bore (see the SC
   size-gate commit), and a head-on view of a symmetric rectangle is the worst
   conditioning for 6-DoF pose -- which would argue for deliberate obliquity.
   Resolve with the `SFP Stage 2 published survey pose ...` line from a run
   known to work; that gives exact standoff and obliquity for a good pose.
3. **Resolution is marginal regardless.** ~17 px on the bore at 0.62 m, and the
   SC recess is shallower than NIC's relative to its width (depth/width 2.1 vs
   3.8), so it is a weaker dark-hole cue at equal pixel count.
4. **BT `done` gate still missing.** SC can legitimately return `done=false`
   (no arm-clear pose at some orientations); without the gate that becomes the
   `norm(quat)==0` crash.

### Downstream `filter_estimates_sc` (Flowstate code node, not in the build)

Rewritten this session; **not yet deployed**. Working copy:
`filter_estimates_sc_NODE.py` in the session scratchpad.

Defects found in the original:

* **fixed world axes** (`ALONG_RAIL_AXIS_ROOT=[1,0,0]`,
  `BETWEEN_RAIL_AXIS_ROOT=[0,1,0]`). Board yaw inflates the within-rail spread by
  `along_extent * sin(yaw)`; with a 115 mm rail against a 12 mm spread gate it
  fails at **~6 deg of board yaw**. Measured: passes at 0-5 deg, fails at 10 deg
  and every yaw beyond.
* **silent mislabel on a flipped board** -- `by_rail[:3]` assumes the 3-port rail
  has the lower +Y; a 180 deg rotation inverts that with no error raised.
* `DUPLICATE_RADIUS_M` (18 mm) equalled `MIN_ALONG_RAIL_SEPARATION_M`, so two
  genuinely adjacent ports merged and the 5-port fit could never be satisfied.
* the 18 mm along-rail floor is too tight for 9.3 mm-wide adapters.
* no coplanarity check.

The rewrite takes its axes from the **detections' own orientations** (averaged
rotation matrices, re-orthonormalised by SVD), searches all three board axes for
the one giving a clean 3/2 split at 41 mm, and assigns rail identity by **port
count** so no rotation can swap the rails. Verified 20/20 board orientations,
plus ghost detection, missing port (refuses rather than mislabels),
no-orientation fallback, and 4 mm noise. PCA of the point cloud was tried first
for the axes and rejected -- a single false positive skews it, and it fails when
the ports bunch along the rail.

Numbering knobs (Phase 1 board): `PORT_LABELS_BY_RAIL = ((0,1,2),(3,4))` and
`ALONG_RAIL_SIGN`. The along-rail *direction* cannot be derived from the pattern
-- five ports in two rows are symmetric end-for-end -- so if a run lands on the
port at the wrong END of the correct rail, flip `ALONG_RAIL_SIGN` and nothing
else.


## Tests and log lines

Tests: `test/test_arm_ik.py` (closed-form FK/IK round-trip, DH-chain identity,
autocalibrate, wrist-camera keep-out against the planner's own reported
collision configs, `link_segments` tracking the elbow rather than the wrist),
reachability-gate plumbing and the full-ranking regression in
`test/test_board_stage2.py`, and per-sector view settings in
`test/test_check_board_visibility_stage2_integration.py`. 256 passing.

Log lines that tell you what is actually live:

```
arm IK reachability gate active: base=Rz180 tool=197.1mm axial=1.00
    wrist-camera keep-out 140mm over 3 probes; arm-in-view rejection over 3 cameras
SFP Stage 2 published survey pose ... standoff=0.580m ... obliquity=<deg>
```

* `obliquity` is measured on the **reference camera's optical axis**, not the
  TCP +Z. The wrist cameras are pitched **15 deg** off the tool axis, so a
  TCP-based reading is ~15 deg high and does not correspond to the limit the
  search enforces. An earlier version of this log line had that bug and reported
  13.1 deg for a pose that was actually inside an 8 deg cap.
* on failure, the reason string carries the framed/evaluated counts, or
  `arm IK calibration failed (...)` with every candidate base convention.

## Latest deployment

**Current deploy target (2026-07-25):** solution
`dc50ce22-2362-4345-85b3-89945912e761_BRANCH`, org `tar-2@xfa-prod-aic-us` --
this is the default in `C:\tmp\ws_aic_phase1\install_skill.sh` and supersedes
the `9b9e6784-...` solution named below. Override per-install with
`AIC_SOLUTION=<id> bash install_skill.sh`; note the variable persists in the
shell, so pass it explicitly rather than relying on the script default.

The **SC/NIC work described above is written and unit-tested but NOT yet
hardware-validated**: the arm-in-view rejection in particular has never run on
the robot. Judge a run on whether the three camera views are clear of the arm.

Historical -- the insignia-driven implementation was built and installed **in
place** as the existing `check_board_visibility_skill_v4` asset (not a new
skill) into Flowstate solution
`9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH` ("Work on this",
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
