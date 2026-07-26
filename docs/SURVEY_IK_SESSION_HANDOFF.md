# NIC/SC Survey-Pose + Reachability — Full Session Handoff

**Written:** 2026-07-25, for a fresh session picking up mid-problem.
**Updated:** 2026-07-25 (same day, third pass) — **RESOLVED in code, not yet
deployed / not yet hardware-validated.** Three rounds this session:

1. **Reachability shortlist bug** (original write-up, §8): the gate only
   checked the top 24 ranked candidates while `prefer_far_standoff` ranks the
   unreachable far poses first — the single reachable pose sat at rank 262 of
   263. Fixed by closed-form UR5e IK + gating the whole ranked list.
2. **After a hardware run** (this pass, §9): the fix from round 1 was deployed
   and mostly worked (70/250 deg board yaw), but (a) NIC's view geometry was
   backwards — the SFP ports open straight up, not sideways, so the code's
   deliberate cross-rail tilt resolved **zero** ports wherever it was reachable
   — and (b) the reachability gate had no collision model, so it published a
   pose the workcell planner refused outright as a self-collision
   (`robot.forearm_link` vs `left_camera.camera_link`). Both fixed; see §9.
   Measured result over 96 scenarios: **90/96 poses found, 80/90 resolving all
   10 ports** — the remaining 6 are a genuine geometric conflict at specific
   board yaws (§9.4), not a bug.
3. **SC view + Cartesian-only Move Robot handoff:** SC now uses an explicit
   10-13 deg board-X approach normal to the adapter's board-Y long face. All
   three cameras fully frame the sector; at least two cameras per mouth pass
   rectangular-bore and projected-depth-cue gates. All-branch arm-in-view
   rejection and an internal IK gate mirror fixed J1..J6
   position limits configured directly on Move Robot. The deployed
   `result.target` seven-scalar -> Python Cartesian pose interface is unchanged.
   The exact-window SC sweep is **96/96**; see §9.3.

The BT `done` guard (§8 Failure A) is **still outstanding** and is not skill
code — it is the single most important remaining step before a real run, since
even the improved search returns `done=false` at some orientations by design.

---

## 0. TL;DR

- **Goal:** a Flowstate skill (`check_board_visibility`) that looks at a task
  board, estimates its pose from the purple insignia, and publishes a Cartesian
  **survey pose** for a downstream **Move Robot** so that the cloud **IVM**
  perception can detect all **5 NIC cards** (recessed SFP ports). A downstream
  code node `filter_estimates_sfp` labels the 5 cards.
- **What we changed this session:** replaced the survey search's crude
  base-origin *sphere* reachability test with a **real UR5e IK reachability
  gate** (`arm_ik.py`), because the sphere both published unreachable poses
  (Move Robot "IK not computable") and rejected reachable far/bore-side poses.
- **Where it broke, and why (measured, §8):** the gate self-calibrated correctly
  (`base=Rz180`, tool 197 mm) but reported 0 of 255 framed candidates reachable.
  The cause was a **ranking/budget interaction**, not the IK and not the view
  geometry alone: framing the NIC sector needs a 0.66–1.3 m standoff and the
  board sits at the height of the arm's own base, so only the **closest** framed
  poses are inside the envelope — while `prefer_far_standoff` ranks the far ones
  **first**. The gate then checked only the top `max_reach_checks=24`, which were
  all far and all unreachable, and gave up with a good pose available.
- **Fix (shipped in this repo, untested on hardware):** (1) replaced the
  damped-least-squares IK with the **closed-form eight-branch UR5e solution** —
  exact, seedless, ~0.2 ms instead of ~260 ms; (2) the gate now scans **every**
  framed candidate in rank order, so `prefer_far_standoff` means "the farthest
  standoff the arm can actually reach". Offline this finds a pose in **96/96**
  swept scenarios. 253 tests green (was 248).
- **Immediate crash** (`norm(quat)==0`) is a **behavior-tree bug**: it runs Move
  Robot on the empty `survey_pose` when `done=false`. The BT must gate Move
  Robot on `result.done` (and `result.success`). This is not skill code, and it
  is **still outstanding** — the fix above makes `done=false` rare, not
  impossible.

---

## 1. Competition context & hard constraints

- **Robot:** UR5e (6-DoF) + Hand-E gripper, **three wrist cameras** ("left",
  "center", "right"), splayed (yaws 90/30/150°, pitch −75°-ish), fixed to
  wrist_3. Reach ~0.85 m.
- **Task board:** a plate 0.30 × 0.425 m with component rails. We care about the
  **NIC card rail**: 5 cards, board **X = −0.081418**, **Y = −0.1745 … −0.0145**
  (40 mm pitch), Z = 0.012 mounts; cards are 145 mm tall fins protruding +Z with
  **SFP cage ports** near the tips (Z ≈ 0.13–0.17). Ports are recessed "black
  holes" that must be viewed roughly down their axis to read depth. There is an
  **asymmetric purple insignia** (open 3-sided bracket) in a board corner used
  as the planar PnP target.
- **TF policy (competition rule — do not violate):** the skill may request TF
  **only** for `base_link` ← the three camera optical frames and
  `gripper/tcp`. It must **not** request task-board, port, module, cable,
  Gazebo, entity-state, or scoring transforms. Board pose is derived by
  detecting the insignia in the allowed camera frames (legitimate perception),
  **not** by querying board TF. The UR5e's own kinematics (used by `arm_ik.py`)
  are the robot's own frames and are allowed.
- **Repo rule:** never commit generated image tarballs, bundles, upload logs,
  API keys, or credentials.
- **Deploy target:** org `tar-2@xfa-prod-aic-us`, solution
  `dc50ce22-2362-4345-85b3-89945912e761_BRANCH` (see `install_skill.sh`). NOTE:
  memory/older docs also mention `9b9e6784-…aaa40_BRANCH` ("Work on this") — the
  authoritative value is whatever is in `C:\tmp\ws_aic_phase1\install_skill.sh`
  (`AIC_SOLUTION`), currently `dc50ce22-…`.

---

## 2. System architecture — the full pipeline

```
Flowstate behavior tree (Move to Board process):
  1. check_board_visibility skill  ── publishes result.survey_pose (intrinsic_proto.Pose, base_link)
        Stage 1: expose the insignia (short, no deadline)
        Stage 2: insignia PnP -> board pose -> search a board-relative TCP survey pose -> publish
  2. [BT should gate on result.done && result.success]   <-- MISSING GATE = the norm(quat)==0 crash
  3. Move Robot (move_to_visible) ── Cartesian move gripper/tcp to survey_pose (its planner does the REAL IK)
  4. estimate_pose_ivm_cloud ── cloud IVM detects components, returns pose_estimates (root_t_target + score)
  5. filter_estimates_sfp (code_execution node) ── labels the 5 NIC cards (3-D lattice fit) and picks the target
```

The skill is **perception + geometry only**; it never moves to the survey pose
itself. Move Robot owns motion + the authoritative IK/collision planner.

---

## 3. The algorithm in detail

### 3.1 Stage 1 — expose the insignia
Short, low-constraint search (`AdaptiveViewpointPlanner`), **no wall-clock
deadline** (planner stall-terminal + per-move timeout bound it). Terminal
condition: the insignia is cleanly visible in a calibrated camera → hand the
freshest 3-camera triplet to Stage 2. In the recent logs Stage 1 finishes at
`iteration=0` (insignia already exposed, no motion), so the arm is **static**
during Stage 2 — important for calibration (§3.4).

### 3.2 Stage 2 — insignia PnP → board pose
`estimate_board_pose_from_insignia` (in `board_stage2.py`): planar `cv2.solvePnP`
of the detected insignia bracket corners against known CAD `INSIGNIA_RECT_CORNERS`,
with the mask centroid resolving the rectangle ambiguity and a
`camera_origin_board[2] > BOARD_TOP_Z` disambiguation + reprojection rejection.
Multiple cameras' estimates are clustered for consistency (≤5 cm / 8°); the
center-camera estimate is preferred. Output: `base_T_board` (full 6-DoF), which
**tracks the board wherever the insignia places it** (this is why the bore-side
logic is board-relative and "follows" a flipped board — confirmed working).

### 3.3 Survey-pose search — `search_survey_pose` (board_stage2.py)
Given `base_T_board`, `tcp_T_cam` (the 3 camera extrinsics recovered from TF at
image time), camera models, gripper keep-out masks, and the current TCP:

1. **Coverage target:** a board-frame box for the requested sector. NIC uses
   `nic_sector_corners()` (a cage-focused box over the SFP cage band, roughly
   board X (−0.14,−0.03), Y (−0.19,0.01), Z (0.07,0.17)). SFP/SC have their own.
2. **Candidate generation:** aim the reference camera's optical axis at the
   sector centroid; sweep `standoffs_m` (0.30 … 1.25 m), board-plane offsets,
   look-direction, and roll (`yaws_rad`). For NIC/SC a **directional cross-rail
   tilt** is used: the camera is tilted *across* the rail (toward the bore-facing
   board −X side, `cross_rail_sign=-1`) by an angle inside a band, holding the
   along-rail tilt ~0.
3. **Rig inversion ("IK" in the geometric sense — NOT joint IK):**
   `base_T_tcp = base_T_refcam ∘ (ref_tcp_T_cam)⁻¹`. This is exact transform
   composition; it yields the TCP pose that realises the desired camera pose.
4. **Per-camera acceptance:** project the sector through every required camera;
   accept only if the target is in front, fully in-frame with a positive
   boundary margin, and gripper-clear with ≥ `min_required_clearance_px` (40 px).
   - NIC/SC: `require_all_cameras_frame=False` (only the **center** camera must
     frame — the splayed rig cannot hold all 5 tall cards together) and
     `prefer_far_standoff=True` (a high, far, undistorted view was believed best
     for the model match / to see full port depth).
   - SFP: all-camera near-overhead framing, closest standoff wins.
5. **Reachability gate (NEW this session, see §3.4):** among all framed
   candidates, ranked by the objective, commit to the **best one that is
   actually reachable**; else return `"framed N but none reachable"`.
6. **Tiered bore band (NEW):** `_bore_view_tilt_bands` returns
   `((12°,22°),(0°,22°))` for NIC/SC — try the *committed* bore tilt first, fall
   back to flat only if nothing reachable (so `prefer_far` can't trade the bore
   tilt away for a far, flat, wrong-side view). SFP → `(None,)`.
7. **Objective (lexicographic, deterministic):** standoff (far-preferred for
   NIC/SC), then cross-rail tilt nearest band centre (directional) or most
   overhead (isotropic), then clearance, then least motion.

### 3.4 Reachability gate — `arm_ik.py` (NEW this session)
Replaces the old base-origin sphere (`‖base_T_tcp.translation‖ ≤ max_reach`),
which was wrong: it admitted kinematically-impossible poses (→ Move Robot "IK
not computable") and rejected reachable far/bore-side poses (→ search settled
near/wrong-side). Contents:

- **Exact FK** for the UR5e wrist chain, taken **verbatim** from the workcell
  MuJoCo model `aic_utils/aic_mujoco/mjcf/aic_robot.xml` (d1=0.1625, a2=−0.425,
  a3=−0.3922, d4=0.1333, d5=0.0997, d6≈0.0996; joint limits: elbow ±π, rest
  ±2π; joint order = `ARM_JOINT_NAMES`). Verified: zero-config flange =
  `(-0.817, -0.233, 0.063)` = exactly `(-(a2+a3), -(d4+d6), d1-d5)`.
- **Closed-form IK** (replaced the damped-least-squares solver). The MJCF chain
  above is bit-for-bit the classical UR5e DH chain (`a`, `d`, `alpha` in
  `_DH_A/_DH_D/_DH_ALPHA`; equality asserted to 1e-12 by
  `test_mjcf_chain_is_the_classical_ur5e_dh_chain`), so the textbook UR solution
  runs directly in the model frame with **no adapter transform**. `solve_all`
  enumerates all 8 branches (shoulder × wrist × elbow); `solve(seed)` returns the
  branch nearest the seed; `reachable` is `bool(solve_all(...))`. Measured over
  3000 random configs: **0 misses**, worst residual 2.3e-12 m / 0 rad,
  mean 7.07 branches per pose, ~0.2 ms/pose (133 µs to reject an unreachable
  one). The old DLS was ~260 ms/check and missed ~3 % of poses cold — which is
  why the gate could only afford a 24-candidate shortlist, and that shortlist is
  what actually broke the run (§8).
- **`autocalibrate`** (KEY): the workcell `base_link` TF differs from the UR
  kinematic base by the classic **180°-about-Z flip**. From ONE static
  `(measured_joints, base_T_tcp)` sample it tries candidate base rotations
  {identity, Rz180, Rz±90, Rx180, Rx180·Rz180} and keeps the one whose recovered
  flange→TCP offset is physically plausible (0.05–0.35 m, ≥0.6 axial along the
  flange +Z). It correctly finds **`base=Rz180`, tool=197.1 mm, axial=1.00**.
  The 634.6 mm garbage offset under the identity assumption was diagnosed as
  exactly `2·(horizontal flange distance)` = the Rz180 signature.
- The gate is **fail-safe**: on implausible calibration / missing joints it logs
  every candidate and **falls back to the sphere**; and it is *more lenient*
  than Move Robot (full joint limits, no collision model) so a rejection means
  genuinely unsolvable — it cannot regress a truly-reachable pose *provided the
  numerical IK actually finds the solution* (the weak point, §9).

### 3.5 Downstream `filter_estimates_sfp` (Flowstate code node — NOT rebuilt)
This is a separate code_execution node the user pastes into Flowstate. It takes
the IVM `pose_estimates` and labels the 5 NIC cards. **Earlier fix this session
(WORKING per the user):** it used to project each detection onto a hard-coded
world axis `RAIL_AXIS_ROOT=[0,1,0]` to measure the 40 mm rail pitch, which
**compressed the pitch below the gate** whenever the board was tilted
(40·cos(tilt) → ~26 mm < 28 mm floor) — so a perfect 5-card detection failed.
Replaced with an **orientation-invariant 3-D line fit**: fit a 3-D line through
the candidate card centers (SVD), measure pitch *along* that line, reject
off-rail detections by perpendicular distance, dedup in full 3-D, and order
labels by board +Y derived from the detection orientation (0 = board −Y edge,
4 = toward insignia). The user says **this already works — leave it.** Full
corrected node saved at
`…/scratchpad/filter_estimates_sfp_fixed.py` (this session's scratch).

---

## 4. Files & functions (map)

Repo root: `c:\Users\anshu\College\aic\aic` (the real git repo + code live in
`aic/aic`; the outer `.git` is empty — run git/pytest from `aic/`).

- `flowstate/aic_perception/aic_perception/arm_ik.py` **(NEW)** — UR5e FK,
  numerical IK, `UR5eArm`, `calibrated_from`, `autocalibrate`, `_BASE_CANDIDATES`.
- `flowstate/aic_perception/aic_perception/board_stage2.py` — `Transform`,
  `estimate_board_pose_from_insignia`, `nic_sector_corners`/`sc_`/`sfp_`,
  `board_coverage_corners`, `module_coverage_corners`, **`search_survey_pose`**
  (now takes `reachable: Callable[[Transform],bool] | None` and
  no longer `max_reach_checks` — deleted), `verify_survey_view`.
- `flowstate/aic_perception/check_board_visibility_skill.py` —
  `_run_sfp_geometric_stage2` (Stage-2 runner; builds the `reachable` callable
  via `UR5eArm.autocalibrate`, loops `_bore_view_tilt_bands`, calls
  `search_survey_pose`), `_single_camera_top_view`, `_bore_view_tilt_bands`,
  `_sector_for_target`, `_stage2_not_done`, `_stage2_landmarks`.
- `flowstate/aic_perception/aic_perception/robot_motion.py` — `RobotMotion`
  (`current_joint(i)`, `ARM_JOINT_NAMES`); TCP/joint feedback.
- `aic_utils/aic_mujoco/mjcf/aic_robot.xml` — ground-truth UR5e kinematics.
- `aic_description/urdf/task_board.urdf.xacro` — NIC/LC/SFP/SC rail positions.
- `docs/reference/nic_card_mount.sdf` — NIC card + SFP port geometry.
- `docs/BOARD_SEARCH_HANDOFF.md` — prior behavior contract (has a "Reachability
  gate" section from this session).
- Tests: `flowstate/aic_perception/test/test_arm_ik.py` (**NEW**, 8 tests),
  `test_board_stage2.py` (has 3 new reachability-gate plumbing tests),
  `test_check_board_visibility_stage2_integration.py` (source guards updated).
- Build/upload: `flowstate/scripts/build_check_board_visibility_skill.sh`;
  `C:\tmp\ws_aic_phase1\install_skill.sh` (retry loop).

Packaging: `CMakeLists.txt` globs `aic_perception/*.py`, so `arm_ik.py` is
picked up automatically — no CMake edit needed.

---

## 5. Build & upload commands (exact, WSL)

```bash
cd /mnt/c/tmp/ws_aic_phase1
# 1. sync the edited tree into the colcon workspace
rsync -a --delete --exclude .git /mnt/c/Users/anshu/College/aic/aic/ src/aic/
# 2. strip CRLF (Windows checkout breaks `set -euo pipefail` and python3 shebangs)
find src/aic/flowstate -type f \( -name '*.py' -o -name '*.sh' \) -exec sed -i 's/\r$//' {} +
# 3. build (full colcon rebuild + gRPC smoke test + single-step inbuild bundle)
INBUILD_BIN=$PWD/inbuild bash src/aic/flowstate/scripts/build_check_board_visibility_skill.sh
# 4. install with bounded retries (stops on AlreadyExists = bundle unchanged)
bash install_skill.sh
```

- The build script does: colcon build → `docker save` the skill image →
  `inbuild skill bundle --manifest <textproto> --file_descriptor_set <desc>
  --oci_image <tar> --output <bundle.tar>` (SINGLE step — the two-step
  `inbuild skill manifest --augmented_*` flow is NOT supported by the installed
  inbuild). Watch the `Bundle:` SHA line — if unchanged, CRLF or a stale bundle
  means your code did not get in.
- Install: `./inctl asset install <bundle> --org tar-2@xfa-prod-aic-us
  --solution dc50ce22-2362-4345-85b3-89945912e761_BRANCH`.
- `AlreadyExists` / "already is installed" = the bundle is byte-identical
  (rebuild; retry won't help). Transient transfer errors auto-retry every 120 s
  ×5.

---

## 6. Test commands (Windows local Python has numpy+cv2)

```bash
cd c:/Users/anshu/College/aic/aic/flowstate/aic_perception
python -m pytest test/ -q                      # full suite (~2 min); last green = 279 passed
python -m pytest test/test_arm_ik.py -q         # FK/IK/autocalibrate unit tests (fast)
python -m pytest test/test_check_board_visibility_stage2_integration.py -q   # source guards (fast)
```
The full colcon test flow the user runs in WSL:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=aic_model:$PYTHONPATH
.pixi/envs/default/bin/python -m pytest -q flowstate/aic_perception/test`.

---

## 7. Chronology — what worked and what didn't

1. **Original (before this session):** survey reachability = base-origin sphere
   `max_reach_m` (0.80 NIC / 0.85 SFP). NIC detection got only 2–4 cards; long
   iteration on view geometry (near-overhead vs cross-rail tilt vs high). SFP
   generally OK.
2. **`filter_estimates_sfp` fix (WORKS):** 3-D lattice fit replaced the fixed
   world-Y projection; NIC labeling now orientation-invariant. **Keep.**
3. **Sphere-reachability failures:** at some board yaws the sphere published
   poses Move Robot rejected: "IK not computable" (e.g. 140° yaw; and yaw-0 poses
   `(-0.28,0.07,0.76)` std 0.90 and `(-0.09,0.41,0.74)` std 0.85). Proven the
   sphere accepts genuinely-unreachable poses: a published pose was **33 mm/38°
   from any UR5e config** in a 400k brute-force.
4. **Key user correction:** "the pose needed to get the view + IVM to work is
   ALWAYS reachable; the failure is our IK, not the robot." And: the board-side
   logic is correct (flipping the board to put the bore side toward the arm
   WORKED); the arm just didn't *go to* the far side when the ports faced away.
5. **IK gate built (`arm_ik.py`) + wired.** Full suite green (247→248). BUT:
6. **First deploy:** `arm IK tool offset 634.6 mm implausible; using sphere
   reach` → gate fell back to sphere → sphere published unreachable pose → Move
   Robot "IK not computable". (Base-frame convention not yet handled.)
7. **`autocalibrate` added** (auto-detect base=Rz180). Validated offline.
8. **Second deploy (current):** `arm IK reachability gate active: base=Rz180
   tool=197.1mm axial=1.00` — calibration FIXED. **But now the gate rejects ALL
   255 framed candidates** at yaw 0° → `done=false` → empty `survey_pose` → Move
   Robot `norm(quat)==0`. The user also noted the 70° ("original") case worked
   even though it wasn't fully around the cards (possibly it was reach-limited).

---

## 8. Root cause (measured offline, 2026-07-25)

**Failure A (the crash):** `move_robot ... norm(quat)==0`. Stage 2 returned
`done=false`, so `survey_pose` was empty, and the **behavior tree ran Move Robot
on it anyway**. FIX (BT-side, not skill): gate the Move Robot branch on
`result.success && result.done`. **Still outstanding.**

**Failure B (`255 framed but none reachable`): SOLVED.** Neither hypothesis in
the original §9 was the whole story; the reproduction settles it.

Reproduced offline with the real workcell geometry, read out of
`aic_utils/aic_mujoco/mjcf/aic_world.xml`: the kinematic base ("tabletop") sits
at world `(-0.2, 0.2, 1.14)` and `task_board_base_link` at
`(0.1445, -0.0602, 1.14)`, yaw `-177.26 deg`. Through the `base=Rz180`
convention that is a board at **`(-0.3445, 0.2602, 0.0)` in `base_link`, yaw
+2.7 deg** — i.e. the "board yaw 0" failing case, and it explains why the
published poses had negative X. Note the board plate is at **exactly the height
of the arm's own base**.

Feeding that through the production NIC search, with the gate instrumented to
scan every candidate rather than 24:

| band | framed | reachable | rank of the first reachable |
|---|---|---|---|
| committed bore (12-22 deg) | 389 | 0 | — |
| flat fallback (0-22 deg) | 263 | 1 | **262 of 263** |

- Framing the five cards in the centre camera requires a **0.66-1.3 m
  standoff**; the framed set contains nothing closer.
- Because the board is level with the arm's base, those standoffs put the TCP
  0.79-1.15 m out and up to **1.05 m above the base, looking down** — which no
  UR5e configuration achieves. Only the very closest framed poses are reachable.
- `prefer_far_standoff=True` ranks the **farthest first**, so the 24-candidate
  shortlist consisted entirely of ~1.1 m poses. Guaranteed miss.

So: **H2 was largely right** (the framed population really is mostly out of
reach — an honest geometry conflict), **H1 was largely wrong** (the closed-form
solver reproduces the DLS verdicts on this population; the DLS was not
manufacturing false negatives here), and the actual bug was the **shortlist**,
which neither hypothesis named.

## 9. The fix

1. **Closed-form UR5e IK** (`arm_ik.solve_all`) replaces the DLS — see §3.4.
   Exact and ~1300x faster, which is what makes step 2 affordable.
2. **Gate every framed candidate in rank order** (`max_reach_checks` deleted from
   `search_survey_pose`). `prefer_far_standoff` now means what it says: the
   farthest standoff that is genuinely reachable.

Offline sweep — board yaw 0/45/70/90/140/180/250/315 deg x tilt 0/10 deg x
placement (0,0)/(+50,+30)/(-50,-40) mm x two Stage-1 exit poses = **96/96
scenarios produce a pose** (before: 0/96 at yaw 0). Regression tests:
`test_reachability_gate_scans_every_framed_candidate_not_a_shortlist`
(board_stage2) and the rewritten exactness tests in `test_arm_ik.py`.

## 10. SUPERSEDED — the bore-side tilt question (kept for history)

Everything below this line described the pre-fix NIC recipe, which tilted the
camera 12-22 deg across the rail on the assumption the SFP port mouths opened
sideways. **That assumption was wrong** (see §9.1) and the whole premise of "buy
the bore tilt at yaw 0" no longer applies — NIC does not tilt at all anymore.
Left in place only so the reasoning trail isn't lost; §9.1-§9.3 are current.

---

### 9.1 The view geometry was backwards (found from a real run, 2026-07-25)

A hardware run at 70/250 deg board yaw worked but the 250 deg case took the arm
through an ugly ~360 deg joint-6 swing to reach the published pose (a downstream
IK-branch-selection question, not a skill bug — see §9.3). A third run, at the
board orientation in the user's screenshots, failed outright:

```
IK could not find a collision free configuration.
Collision reported: robot.forearm_link vs left_camera.camera_link (all 4 solutions)
published pose: (-0.1001, 0.4162, 0.6859), bore_band=12-22deg
```

Measuring the actual SFP port geometry from `aic_world.xml` (mount 2, the
`nic_card_mount` -> `nic_card_link` -> `sfp_port_N_link` -> `..._entrance` chain)
settled a question the code had been guessing at: each port is a **16 x 12 mm
aperture at the top of a 45.8 mm recess whose axis points straight *up*** —
board-frame bore axis `(0.001, -0.013, -0.9999)`, i.e. 0.7 deg off the board
normal, entrance at board Z 0.1793. A port only shows the black depth the IVM
keys on to a ray within `atan(6/45.8) = 7.5 deg` of that axis; past that the
cage wall occludes the backstop and the port reads as a flat grey rectangle.

The code's cross-rail bore tilt (12-22 deg, committed) was built on the opposite
assumption — that the mouths opened sideways, the way the SC ports genuinely do.
Scored against the measured cone, that band resolved **0 of 10 ports** wherever
it was reachable. This — not reachability — is the reason NIC framing never got
more than 2-4 cards historically, and it explains the user's own annotated
screenshot: tilted views (X'd) read the cages edge-on; straight-down views
(checked) show the black holes.

**Fix (`check_board_visibility_skill.py::_survey_view_settings`, target 2):**
look straight down (`max_obliquity_rad=2 deg`, no cross-rail tilt — NIC dropped
out of `_bore_view_tilt_bands`, which now only serves SC), all three cameras
(`require_all_cameras_frame=True`), farthest reachable standoff
(`prefer_far_standoff=True` — now justified by real geometry: the ten ports
span 160 mm, so the outermost needs `d >= 0.62 m` above the port plane to stay
in the 7.5 deg cone), and `min_required_clearance_px=25` (down from 40 — needed
for the three cameras to fit together; the gripper mask already dilates the
silhouette by 32 px underneath this, so cards still stay 57 mm-equivalent clear
of real gripper pixels). `nic_sector_corners()` is now centred on the ten port
entrances rather than the card bodies (the old centroid sat 16 mm off the port
cluster, enough to push the outermost port past the cone on its own).

### 9.2 The reachability gate had no collision model

The `robot.forearm_link` collision above is real and the IK gate — purely
kinematic — had no way to see it. Its four reported joint configurations
reproduce the published TCP through `arm_ik`'s FK to **0.0 mm**, which is also
the first end-to-end validation of the whole model (DH chain, `base=Rz180`,
197.1 mm tool) against the real robot, not just self-consistency tests. The
planner's best branch put a wrist camera **111 mm** from the forearm centreline
— all four solutions collided.

**Fix:** `UR5eArm` gained `flange_T_probes` (points rigidly attached to the
flange — the three wrist cameras, populated from the same TF-derived extrinsics
the skill already has) and `min_self_clearance_m` (calibrated to **140 mm**, a
26% margin over the 111 mm ground truth). `solve()`/`reachable()` now require a
branch clearing every probe from the elbow->wrist_1 segment.
`test_wrist_camera_keep_out_rejects_a_pose_the_workcell_planner_refused` in
`test_arm_ik.py` carries the planner's exact four configs as a permanent
regression.

### 9.3 The ~360 deg joint-6 swing at yaw 250

The skill publishes a **Cartesian** `survey_pose`; Move Robot's planner chooses
the actual joint configuration and path, including which branch of joint 6
(`-133.8 deg` vs its co-terminal `226.2 deg`) to use. A later SC run confirmed
the same class of failure at larger scale: a 0.374 m Cartesian displacement
became 1193 trajectory points and 29.44 s, while normal moves in the same log
were 91-111 points and about 3 s.

The skill previously had one avoidable blind spot: `solve(seed)` measured
`wrap_pi(q - seed)`, so configurations separated by a full revolution had zero
cost, and `search_survey_pose` discarded the chosen IK solution after its
boolean reachability test. Current mitigation:

1. lift every analytic IK solution to the closest co-terminal value inside the
   real joint limits before comparing it with the live seed;
2. return physical joint deltas to the survey search;
3. among equal camera-quality candidates, minimize worst-joint travel and then
   total joint travel;
4. enumerate every finite forearm-clear IK branch, express it inside the same
   fixed absolute J1..J6 position window configured directly on Move Robot,
   reject branches that put the upper arm/forearm in any wrist camera, and use
   the lowest-motion clear branch under the 220 deg SC internal cap;
5. log current, target and delta joint vectors for the selected pose;
6. keep the deployed output unchanged:
   `result.target.{x,y,z,qx,qy,qz,qw}` -> existing Python Cartesian pose packer
   -> Move Robot;
7. configure these constant J1..J6 position bounds directly on Move Robot:

   ```
   min deg = [-53.6, -187.0, -122.4, -127.7, -116.1, -71.5]
   max deg = [170.1,  -28.9,   94.1,   43.8,  114.8, 180.8]
   ```

The skill mirrors exactly those limits in its analytic gate and returns
`done=false` if the live start or every arm-clear target branch falls outside
them. No joint target or limit message is exposed by the skill. Each configured
interval is narrower than 253 deg, so Move Robot cannot place two co-terminal
representations of one joint inside the same window and cannot plan a complete
360 deg winding. Joint speed/acceleration caps remain useful backstops but do
not replace these absolute position bounds.

The matching SC 96-case production sweep (yaw x tilt x placement x two rolled
Stage-1 exits) is now **96/96** with the explicit long-face approach, two-camera
3.0 px depth-cue floor and soft J6 half-turn preference: standoff 0.62 m,
selected depth cue 3.343..4.451 px, bore margin +0.0135..+0.1572, all-camera
clearance 37.8..74.0 px, and worst-joint motion 27.6..219.5 deg. The
J6 preference may buy at most 30 deg additional worst-joint motion and never
weakens the fixed position window. A 215 deg cap loses 1/96 under this stronger
view policy, so 220 deg is the lowest tested production cap that preserves all
scenarios.
`test/sc_sweep_runner.py` is the reproducible harness.

### 9.4 Result — measured over 96 scenarios (board yaw 0-315 deg x tilt 0-10 deg
x +/-50 mm placement x two Stage-1 exit poses), full production settings
including the collision gate:

| metric | value |
|---|---|
| pose found | **90/96** |
| all 10 ports resolved (of found) | 80/90 |
| worst port ray | 8.7 deg (cone limit 7.5 — the degraded fallback poses below) |
| standoff | 0.55-0.80 m |
| camera-forearm clearance | 141-207 mm (threshold 140) |
| reorientation used | 0-90 deg (cap 90) |

The 6 misses cluster at board yaw 45 deg (nominal and jittered placement) and
70 deg (jittered only). **Confirmed a hard geometric ceiling, not a sampling
gap:** at yaw 45 even a 72-value roll sweep (5 deg steps, vs the production
24-value/15 deg) tops out at **140.7 mm** of forearm clearance, and that single
candidate resolves only 6/10 ports — the fixed camera-rig splay (yaws 90/30/150
deg) and this specific board orientation squeeze the reachable envelope, the
7.5 deg port cone, and the 140 mm keep-out into a near-empty intersection. Not
worth shaving the collision threshold to grab it: the ground-truth-calibrated
140 mm should not move, and the one pose that clears a lower bar still doesn't
resolve all the ports. This is exactly the case the graceful `done=false` +
BT `result.done` gate (§8, Failure A — **still outstanding**) exists for: no
good pose, so no move, rather than a bad move.

`NIC_VIEW` settings that produced this (see `_survey_view_settings(2)`):
`max_obliquity_rad=2 deg`, `min_required_clearance_px=25`,
`max_angular_motion_rad=90 deg`, `yaws_rad` = 24 values (15 deg steps, vs the
7-value/45-deg-band default other sectors keep) — search cost is correspondingly
higher (tens of seconds per candidate set vs ~3 s pre-collision-gate) but this
runs once per Stage-2 handoff, not in a control loop.

## 11. Key numbers & geometry reference

- UR5e: d1=0.1625, a2=−0.425, a3=−0.3922, d4=0.1333, d5=0.0997, d6≈0.0996;
  elbow ±π, others ±2π; **base_link = Rz(180°) · kinematic-base** (autocalibrate
  confirmed); tool `gripper/tcp` ≈ 197 mm along flange +Z.
- NIC mounts: board X=−0.081418, Y ∈ {−0.1745,−0.1345,−0.0945,−0.0545,−0.0145}
  (40 mm pitch), Z=0.012; cards 145 mm tall; SFP cage/ports Z≈0.13–0.17; ports
  open toward board −X ("red arrow" side), recessed ~45.8 mm.
- Filter thresholds (`filter_estimates_sfp`): NOMINAL_PITCH 40 mm, PITCH_TOL
  12 mm (⇒ 28–52 mm gate), MAX_LATTICE_RESIDUAL 10 mm, PERP_TOL 12 mm,
  MIN_SCORE 0.4, DUP_RADIUS 18 mm (now 3-D).
- Survey search NIC/SC: `cross_rail_sign=-1`, bands ((12°,22°),(0°,22°)),
  `require_all_cameras_frame=False`, `prefer_far_standoff=True`,
  `min_required_clearance_px=40`, `max_reach_m=0.85`, `min_height_m=0.02`,
  standoffs 0.30…1.25 (the gate now checks every framed candidate).
- Observed failing published poses (sphere, pre-gate): `(-0.28,0.07,0.76)` std
  0.90; `(-0.09,0.41,0.74)` std 0.85 → Move Robot IK reject.
- Workcell placement (from `aic_world.xml`, for offline reproduction): kinematic
  base at world `(-0.2, 0.2, 1.14)`; `task_board_base_link` at
  `(0.1445, -0.0602, 1.14)` quat `(w=0.0239514, z=-0.999713)` → in `base_link`,
  board origin `(-0.3445, 0.2602, 0.0)` at yaw `+2.7°`, level with the arm base.
- Post-fix pick at that placement: standoff 0.73 m, TCP `(-0.396,0.183,0.626)`,
  |t| 0.763 m, cross-tilt 0° (flat band).

---

## 12. Gotchas

- **Nested repo:** work in `aic/aic`; outer `.git` is empty.
- **CRLF:** always `sed -i 's/\r$//'` before building (shebangs / `set -euo`).
- **`inbuild` is single-step** (`skill bundle`), not the two-step
  manifest/`--augmented_*` flow.
- **Docker storage** bloats; keep both the aic_model container and the skill
  build cached; `defaultKeepStorage` ~40 GB; WSL vhdx compaction needs Docker
  Desktop fully quit + `Stop-Service com.docker.service` before diskpart.
- **`autocalibrate` assumes a static arm** at Stage-2 handoff (true when Stage 1
  ends at iteration 0). If Stage 1 moves right before handoff, the (joints, TCP)
  sample could be inconsistent — the plausibility check guards this (falls back
  to sphere) but watch the log.
- **The IK gate is fail-safe but the search returns `done=false` when nothing is
  reachable** — the BT MUST handle not-done (Failure A).
- Do not revert the user's edits to `build_check_board_visibility_skill.sh` or
  the integration test unless genuinely broken.
```
