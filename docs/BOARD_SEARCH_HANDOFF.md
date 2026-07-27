# Board-search handoff

Updated: 2026-07-26

For the consolidated end-to-end v4 handoff covering shared Stage 1/Stage 2,
SFP, NIC, SC, Flowstate wiring, downstream filters, testing, and deployment,
start with `docs/CHECK_BOARD_VISIBILITY_V4_HANDOFF.md`. This document remains
the deeper behavior contract and SC reasoning trail.

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
  base per-camera acceptance is target-in-frame plus positive gripper
  clearance; recessed-port sectors add their target-specific bore/depth gates;
- selection is tuned for **the way the IVM reads each part**. SFP uses the
  standard all-camera near-overhead framing. NIC stays nearly normal to the
  board. SC requires all three cameras, the closest valid standoff, and an
  explicit 10-13 degree displacement along board X: the normal to the adapter's
  board-Y long face. Tilt along that long face stays at most 2 degrees. All
  three cameras fully frame the sector; at least two cameras per mouth retain a
  positive back-plane margin and at least 3.0 px of projected depth cue. This
  avoids inferring the view axis from the longer three-/two-port cluster box.
  The target-specific settings and measurements are authoritative in the
  sections below;
- the skill is **perception-only**: it publishes the existing scalar Cartesian
  target on `result.target.{x,y,z,qx,qy,qz,qw}` for the deployed Python pose
  packer and does not execute the survey move itself. IK branches are expressed
  at the physical winding nearest the measured live joint vector and judged by
  relative travel from that origin; no artificial absolute joint window, joint
  target, or joint-limit output is used. There is **no** aggregate Stage-2 time
  budget and **no** two-triplet consistency gauntlet.

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
therefore takes `cross_rail_tilt_band_rad=None` and
`max_obliquity_rad = 2 deg`. **SC also opens along the board normal**, but its
long face is the 22.4 mm board-Y edge. The checked hardware view stands off that
face, so the in-plane camera displacement is explicitly board X and limited to
10-13 degrees by the 7.6 mm narrow bore. All-camera framing plus the two-camera
rectangular-bore/depth gates remain authoritative.

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
port only 61 mm off the cluster centre vs 81. Ordinary reach is not binding,
but the arm-in-view gate limits how aggressively the black-depth cue can be
increased at board yaw 180 deg.

### How the survey pose is chosen today

`_survey_view_settings(3)` in `check_board_visibility_skill.py`:

| setting | value | why |
|---|---|---|
| `cross_rail_tilt_band_rad` | **`10..13 deg`** | removes every visibly head-on 7-8 deg pose; 11..13 loses 2/96, while 10..13 keeps the strongest fully reachable band |
| `directional_tilt_axis_board` | **`(1,0,0)`** | explicit board-X normal to the board-Y long face; transformed through `base_T_board.rotation`, so board yaw/tilt rotate the offset rather than leaving it in world X |
| `max_along_rail_tilt_rad` | **`2 deg`** | limits motion along the 22.4 mm long face / port-row direction |
| `cross_rail_sign` | `0.0` | search both sides; the rectangular-bore score chooses the healthier all-camera rays |
| `require_all_cameras_frame` | **`True`** | IVM needs all three cameras framing the sector (confirmed by the user) |
| `prefer_far_standoff` | `False` | cone is met from 0.27 m, so take the pixels |
| `min_required_clearance_px` | `25` | gripper keep-out margin |
| `max_angular_motion_rad` | **`180 deg`** | do not discard camera-clear rolls after a rolled Stage-1 exit; the separate live-relative IK gate constrains physical joint travel |
| `yaws_rad` | 24 values, 15 deg | roll rotates the sector clear of the gripper silhouette |
| `standoffs_m` | **`(0.62,)`** | selected in all 144 scenarios by the stronger-view sweep |
| rectangular-bore view margin | **>= 0.0 in at least 2 cameras per mouth** | all three still frame the sector, but requiring all three to see through every bore made every >=10 deg pose impossible |
| projected bore-depth cue | **>= 3.0 px in at least 2 cameras per mouth** | requires a detector-meaningful displaced dark interior rather than the prior nearly head-on ~1.4 px cue |

`sc_sector_corners()` is centred on the five **entrances**, not the adapter
bodies -- the search aims the optical axis at the box centroid, and the old box
(X -0.14..-0.01, Y -0.02..0.10, Z 0.01..0.05) predated the 3-port row and swept
in ~47 mm of empty board on the -Y side, pulling the aim point 10 mm off.

The current full production sweep is **144/144** (board yaw 0-315 deg x tilt
0-10 deg x +/-50 mm placement x three live starts: two Stage-1 wrist-roll
exits plus the exact chained predecessor state that exposed the old absolute
window failure), including exact live-relative IK, wrist-camera/forearm
collision, arm-in-view rejection, all three gripper masks, and the 185 deg SC
relative-motion cap. With the explicit long-face approach, depth-cue gate and J6 preference,
selected ranges are: standoff **0.62 m**, two-camera projected depth cue
**3.343..4.451 px**, normalized two-camera bore margin **+0.0135..+0.2674**,
all-camera clearance **37.8..74.0 px**, board-X tilt **10..13 deg**, board-Y tilt
**<=2 deg**, worst-joint motion **27.4..182.4 deg**. Reproduce with
`python test/sc_sweep_runner.py --workers 4` from
`flowstate/aic_perception`.

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
capsules (60 mm / 50 mm tubes), and `_arm_clear_of_own_cameras()` rejects a
branch whose links project into any camera. `UR5eArm.solve_ranked()` now exposes
**every** finite, forearm-clear analytic branch; the selector tests arm-in-view
on all of them and keeps the lowest-relative-motion clear branch. The earlier
version tested only the one branch nearest the seed and could reject a pose
without checking its other exact IK branches. For SC, every branch is lifted
to the physically equivalent winding nearest the live seed before its relative
delta is scored; only the resulting Cartesian TCP pose is published.

### All-camera framing is not all-camera bore visibility

Two further hardware cases isolated a second roll-dependent failure after the
arm-in-view work: a diagonal-board configuration failed to return all five SC
ports, while an axis-aligned configuration returned all five. The world-view
screenshots alone do not expose the camera pixels, but the production
camera/arm sweep reproduced the split.

The missing constraint was physical image formation in the **side cameras**.
The three camera origins are separated by about 115 mm. Wrist roll rotates that
baseline relative to the SC mouth, whose limiting dimension is only 7.6 mm
across board X (22.4 mm across board Y). The old search constrained the centre
camera's obliquity and required every mouth to be framed and gripper-clear, but
never checked whether a ray from each separated camera origin could pass through
the rectangular mouth to the bore back plane.

At the analogous 45 deg board yaw, full production geometry selected:

* old first IK/arm-clear candidate: standoff 0.583 m, normalized bore margin
  **-0.270** in both side cameras (a wall hides the back plane);
* first zero-margin bore-valid candidate: 0.621 m, margin only **+0.006**;
* first candidate with the new 5% robustness buffer: 0.651 m, margin
  **+0.053**.

Fix: `rectangular_bore_visibility_margin()` evaluates the ray from every
required camera origin to six conservative SC mouth samples (both rail rows at
the minimum/midpoint/maximum rail X). A non-negative score means the ray reaches
the back plane within both aperture half-widths. SC now requires
`view_quality >= 0.05` before the ranked list reaches the IK/arm-clear gate, then
keeps nearest-standoff-first among the usable poses. Within that nearest
standoff it now **maximizes view quality before joint motion**. A later hardware
failure exposed why the second clause matters: the selected pose was only
`+0.054`, and IVM returned three geometrically valid ports plus one off-plane
false pose. The valid detections form exact rail coordinates
`(-40,-41), (0,0), (+40,0)` mm: two detections on one rail and one on the
other. The error omits the detection orientations and randomized task
configuration, so that is **not enough to assign absolute `sc_port_N` labels**
or decide whether the missing pair is both on the three-port rail versus one
per rail. In the failing images two end mouths degrade toward thin bright cyan
rims while the three surviving mouths retain more dark rectangular interior,
but do not turn that visual inference into an absolute label without the task
configuration.

At the analogous 45 deg board pose, the old first-passing roll was `+0.053`;
the production camera geometry contained healthier rolls at the same
resolution, proving the score had to remain in the ranking after its hard
floor.

The next hardware run disproved the original 4-8 deg detector-conditioning
band.
The skill selected its maximum -- standoff `0.580 m`, cross/along tilt
`8.0/1.0 deg`, all-camera clearance `41.6 px`, modeled bore margin `+0.221` --
yet IVM returned only **2/5** ports. The camera frames show nearly flat cyan
rims, not the visibly oblique blue-mouth/dark-depth cue in the working image.
The bore margin is therefore a necessary occlusion guard, not a sufficient IVM
quality score.

The attempted response was 12-16 deg along the 22.4 mm board-Y dimension. The
next camera images exposed the axis mistake: one camera approached the short end
of the adapter and failed, while the checked view stands off the **long face**.
Standing off a face means translating along its normal, so the correct in-plane
axis is board X, not board Y. Because board X is also the narrow 7.6 mm bore
dimension, the corrected interim angle returned to **4-8 deg**, now with
`directional_tilt_axis_board=(1,0,0)` rather than the sector-box heuristic.
Along-face/board-Y tilt remains <=2 deg and the physical bore gate still covers
every mouth/camera. That interim production sweep was **96/96**. NIC and SFP
were untouched.

The latest hardware run returned four strong coplanar candidates:
`[146.5,-260.4,1156.7]`, `[160.3,-222.7,1156.3]`,
`[108.1,-246.4,1156.6]`, and `[121.9,-208.7,1156.5]` mm. Their vectors close
to an exact **40.15 x 40.87 mm 2x2 rectangle** with only **0.30 mm closure
error**. Extending the three-port rail predicts the missing end port at
`[132.7,-298.1,~1157]` mm, just **0.42 mm** from the earlier independent
detection `[132.8,-298.0,1156.7]`. This verifies the user's hypothesis at the
physical-port level: the single end mouth is omitted while the other four form
the expected lattice. It still does not assign an absolute `sc_port_N` label
without the randomized rail configuration.

The new quality model projects each mouth centre and its 15.64 mm-deep back
centre into every camera. Their pixel displacement explicitly measures the
dark-depth cue that the geometric margin could not. The first version required
every camera to see through every bore. That hidden intersection made
**all 10-13 deg candidates impossible** and forced the selected hardware view
back to 7-8 deg / ~1.4 px, which the latest images correctly exposed as nearly
head-on. Full framing and bore visibility are different requirements.

The corrected fused-view policy keeps full-sector framing and gripper clearance
in all three cameras, while requiring a positive bore margin and >=3.0 px depth
cue in at least two cameras for every mouth. The sweep found:

* 10-13 deg, fixed 0.62 m, no aim bias: **144/144** including the chained hardware start;
* selected worst two-camera cue **3.343 px**;
* on the original two-start sweep, 11-13 deg: **94/96**; 12-13 deg: **74/96**;
* the 185 deg live-relative SC joint cap preserves **144/144**; 182 deg loses
  2/96 on the original two-start sweep.

Only scores within **0.1 px** of the best reachable cue at one standoff are
treated as perception-equivalent before joint/J6 ranking. This stops the
preferred wrist roll from trading away the feature the new model preserves.

The desired arm picture also places the wrist/tool on the opposite side. SC now
computes a preferred target at live J6 +/-180 deg, choosing an exact half-turn
inside the modeled physical J6 limits. This is a **preference, not a validity
escape**: the 185 deg live-relative cap remains, and the preferred roll may buy
no more than 30 deg additional worst-joint travel over the safest
perception-equivalent route. A fixed board-X sign was swept and rejected (only
72/96), because board yaw changes which world-side pose is reachable; both
signs remain available and the J6/arm-clear ranking chooses.

The sparse IVM coordinates cannot identify absolute missing labels because the
randomized rail translations and detection orientations are absent. The two
surviving candidates (`[132.8,-298.0,1156.7]` and
`[107.9,-246.5,1153.1]` mm) are the same two high-confidence physical
detections that survived the earlier 4-candidate run, so the failure is
repeatable view bias rather than a filter timeout.

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
   pose. The 2026-07-25 SC run made this unambiguous: a 0.374 m Cartesian move
   produced **1193 trajectory points / 29.44 s**, versus 91-111 points /
   2.75-3.16 s for ordinary moves in the same log. An earlier NIC run used a
   ~360 deg joint-6 swing -- the planner picked the co-terminal branch
   (`226.2 deg` vs `-133.8 deg`).

   The deployed process cannot expose a joint target from this skill. It reads
   only `result.target.{x,y,z,qx,qy,qz,qw}`, packs those seven scalars into a
   Cartesian TCP pose in a Python node, and sends that pose to Move Robot.

   The fixed absolute-window mitigation was removed on 2026-07-26. It judged a
   global coordinate rather than motion from the live start. In the hardware
   failure that exposed the problem, a preceding valid survey pose ended at
   J4 `-143.9 deg`; the SC window started at `-127.7 deg`, so SC rejected the
   current state before evaluating any pose. The ungated BT then sent the empty
   result to Move Robot and produced `norm(quat)==0`.

   The current policy uses the measured six-joint vector as the origin:

   * `UR5eArm.solve_ranked(pose, seed=live_joints)` lifts every analytic branch
     to the physically equivalent winding nearest that live seed. No
     task-specific absolute position box is passed.
   * Every physical shoulder/elbow/wrist branch is checked for wrist-camera
     collision and arm-in-view.
   * At the nearest reachable standoff, the best **reachable** projected
     depth-cue score defines a 0.1 px perception plateau. Inside that plateau
     the selector
     minimizes worst-joint and then total travel. An unreachable high-score roll
     must not suppress a slightly lower-score reachable one.
   * The SC internal worst-relative-joint cap is **185 deg**. The selected
     maximum is 182.4 deg; 182 deg loses 2/96 of the original Stage-1-start
     sweep. Other sectors retain the prior 225 deg budget.
   * The selected-pose log prints current/target/delta vectors plus
     `joint_max`, `joint_total`, and `relative_origin=live_joints`.

   The live-relative production sweep is **144/144** over board yaw, tilt,
   placement, both Stage-1 exits, and the exact chained hardware start above.
   This validates the skill's selected endpoint branch offline. It does **not**
   force Move Robot to use that branch: a Cartesian pose contains no joint
   winding, so Move Robot remains authoritative for the actual collision-free
   path. Remove the old custom absolute position bounds from the SC Move Robot
   segment, retain conservative velocity/acceleration limits, and validate the
   planned transit on hardware.
2. **Validate the corrected long-face axis and J6 preference on hardware.** The
   code now uses board X explicitly, keeps all three cameras fully framed, and
   biases the wrist toward the best legal half-turn inside a bounded motion
   plateau. The actual detector frames remain authoritative. Capture the
   published-pose line (including `preferred_j6_deg`/`j6_error`) and all three
   camera frames on the rerun.
3. **Resolution is marginal regardless.** ~17 px on the bore at 0.62 m, and the
   SC recess is shallower than NIC's relative to its width (depth/width 2.1 vs
   3.8), so it is a weaker dark-hole cue at equal pixel count.
4. **BT `done` gate still missing.** SC can legitimately return `done=false`
   (no arm-clear pose at some orientations); without the gate that becomes the
   `norm(quat)==0` crash.

### Downstream `filter_estimates_sc` (Flowstate code node, not in the build)

Rewritten this session; **not yet deployed**. Paste-ready copy checked in at
`docs/reference/filter_estimates_sc_node.py`, with its offline harness at
`docs/reference/filter_estimates_sc_node_test.py` (synthetic detections at
arbitrary board yaw/tilt; run it directly with python, it needs only numpy).
Neither is part of the skill build.

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
* the hardware-pasted copy had `MIN_SCORE=0.0`. Both recent runs contain four
  high-score physical ports (0.77-0.91) plus remote false positives at
  0.23-0.37. With zero threshold and exactly five candidates, the only
  five-element combination is forced to include the false positive and reports
  a misleading rail-separation failure. The reference keeps `MIN_SCORE=0.4`.

Per the latest process requirement, the paste-ready node is now deliberately
**positional and permissive**. It no longer validates rail separation,
coplanarity, within-row spread, 3/2 row counts, minimum gaps, or competing
layouts. It:

1. score-filters and deduplicates detections;
2. recovers signed board X/Y from their orientations (root X/Y fallback);
3. splits at the largest board-Y positional gap without validating its size or
   resulting counts;
4. sorts each row by board X;
5. assigns positional slots `0/1/2` to the lower row and `3/4` to the upper;
6. returns the slot named by `selected_module_name`.

Non-alignment never causes failure. The only remaining selection failure is
that the requested positional slot has no detection. Extra items beyond the
three/two output slots are logged and ignored. This intentionally accepts the
risk that a missing or spurious detection can shift a positional rank; the
strict geometry refusal was removed by request.

The self-contained harness verifies 20/20 rotated/tilted board orientations,
deliberately invalid spacing/alignment, the observed 0.23-0.37 background false
positives, and partial detections where only the missing positional slot is
unavailable.

Numbering knobs: `ALONG_RAIL_SIGN`, `BETWEEN_RAIL_SIGN`, and
`PORT_LABELS_BY_ROW = ((0,1,2),(3,4))`.


## Tests and log lines

Tests: `test/test_arm_ik.py` (closed-form FK/IK round-trip, DH-chain identity,
autocalibrate, wrist-camera keep-out against the planner's own reported
collision configs, `link_segments` tracking the elbow rather than the wrist),
reachability-gate plumbing and the full-ranking regression in
`test/test_board_stage2.py`, and per-sector view settings in
`test/test_check_board_visibility_stage2_integration.py`. **282 passing**.

### 2026-07-26 timing split and Stage-2 optimization

The combined skill/IVM logs settle the reported "perception took more than a
minute" delay:

* geometric Stage 2: `01:28:56.864` IK-gate start to `01:30:00.690` published
  target = **63.83 s**;
* IVM: three captures plus request/inference = **6.15 s**, of which cloud
  request-to-response was **5.71 s**.

The minute was our exhaustive pose search, not cloud IVM. A representative
offline SC case evaluated 7,920 poses and 23,760 full-resolution gripper-mask
clearances. The search now applies the cheap rectangular-bore hard gate first,
checks the reference camera first, rejects insufficient image-boundary margin
before rasterizing the gripper hull, and stops after the first failed required
camera. This is only a reordering/short-circuit of the same hard gates. The
representative case fell from about **8.0 s to 3.6 s** locally and selected the
same pose/IK branch; deployment timing remains to be measured.

The stronger-view sweep selects standoff 0.62 m in all 144 scenarios.
Restricting SC to that one value preserves **144/144** and removes the remaining
unused distance groups. The complete 144-case sweep takes about **23 s** with
four local workers on the current Windows host. Hardware timing still needs to
be measured after deployment.

Log lines that tell you what is actually live:

```
arm IK joint-motion gate active: base=Rz180 tool=197.1mm axial=1.00
    wrist-camera keep-out 140mm over 3 probes; arm-in-view rejection over 3 cameras;
    max predicted joint move 185deg
survey IK motion current_deg=[...] target_deg=[...] delta_deg=[...]
    max=<deg> total=<deg> relative_origin=live_joints
SC survey image geometry required_depth_cameras=2
    bore_margin_2cam=<margin> depth_cue_2cam=<pixels>px
SFP Stage 2 published survey pose ... standoff=0.620m ... obliquity=<deg>
    cross_tilt=<deg> along_tilt=<deg>
    joint_max=<deg> joint_total=<deg>
```

* `obliquity` is measured on the **reference camera's optical axis**, not the
  TCP +Z. The wrist cameras are pitched **15 deg** off the tool axis, so a
  TCP-based reading is ~15 deg high and does not correspond to the limit the
  search enforces. An earlier version of this log line had that bug and reported
  13.1 deg for a pose that was actually inside an 8 deg cap.
* on failure, the reason string carries the framed/evaluated counts, or
  `arm IK calibration failed (...)` with every candidate base convention.

### 2026-07-25 IVM failure was not a perception miss

The IVM invocation after the 1193-point move captured all three images and
constructed the `aic_sc_port` request, then failed inside `ModelInfer` while
fetching the IPC identity token:

```
Failed to get IPC token from AccountsTokensService
StatusCode.DEADLINE_EXCEEDED
```

No model result existed, so this run says nothing about whether the selected
view would detect five ports. It is distinct from the earlier 20:05 run, which
did reach inference and returned `No poses detected` / status `10100`.

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

The insignia-driven implementation passes 279 Flowstate perception tests.
Any intentional
board-search change must update this handoff and its pinned implementation.
