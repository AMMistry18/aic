# NIC/SC Survey-Pose + Reachability — Full Session Handoff

**Written:** 2026-07-25, for a fresh session picking up mid-problem.
**Status:** BLOCKED. The reachability gate is correctly calibrated but now
rejects *every* framed survey pose at some board orientations, so Stage 2
returns `done=false`, and the downstream Move Robot then crashes on the empty
pose. Read §8 (what fails now) and §9 (the core tension) first, then decide
§10.

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
- **Where it broke:** the IK gate now correctly self-calibrates the base frame
  (`base=Rz180`, tool 197 mm) but then finds **0 of 255 framed candidates
  reachable** at board yaw 0°. Two open possibilities: (a) the numerical IK is
  too weak (misses reachable far-side branches → false negatives), or (b) the
  framed candidates are *genuinely* all unreachable because framing the tall NIC
  cards forces a far/high standoff that is at/beyond reach — a **fundamental
  conflict** between "frame the cards" and "stay reachable". See §9.
- **Immediate crash** (`norm(quat)==0`) is a **behavior-tree bug**: it runs Move
  Robot on the empty `survey_pose` when `done=false`. The BT must gate Move
  Robot on `result.done` (and `result.success`). This is not skill code.

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
- **Numerical IK** = damped least-squares (Levenberg) with a geometric Jacobian,
  seeded at the current joints + a **10-seed** pan×elbow grid, joint-limit
  clamping, a stall early-out, and a wrist-center annulus pre-filter. Used as a
  boolean gate ("does a joint-limit-valid solution exist?"). `max_iters=80`,
  `damping=0.06`, `max_reach_checks=24` (only the top-24 ranked candidates get
  the numerical solve). ~1 ms/solve with a near seed; ~130 ms worst-case
  unreachable; near-seed round-trip 100 %, **cold round-trip only ~96.7 %**
  (this matters — see §9).
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
  `max_reach_checks`), `verify_survey_view`.
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
python -m pytest test/ -q                      # full suite (~2 min); last green = 248 passed
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

## 8. What fails RIGHT NOW (precise)

**Failure A (the immediate crash):**
`ai.intrinsic.move_robot:10601 … Failed to create Pose from proto which contains
a non-unit quaternion with norm(quat)==0 … position { } orientation { }`.
Root cause: Stage 2 returned `done=false` (no reachable pose), so `survey_pose`
is empty (identity/zero), and the **behavior tree ran Move Robot on it anyway**.
FIX (BT-side, not skill): gate the Move Robot branch on
`result.success && result.done`; on not-done, skip/replan instead of moving.

**Failure B (the real problem):** at board yaw 0°, Stage 2 logs
`no safe all-camera survey pose: 255 pose(s) framed the target in all required
cameras but none had a reachable joint-limit-valid IK solution (294 candidates
evaluated)`. The gate is calibrated correctly (base=Rz180, tool 197 mm) yet
finds **zero** reachable framed poses. The user is (reasonably) frustrated: "why
is it failing to compute successful IK solutions at all poses — this seems
inherently broken." It reproduces at yaw 0 and "all poses."

---

## 9. Leading hypotheses & the CORE TENSION (read this)

Two non-exclusive explanations for Failure B:

**H1 — the numerical IK is too weak (false negatives).** The far/bore-side
survey poses require arm configurations *far* from the current joints (shoulder
rotation + possible elbow flip to reach the far side). The DLS is seeded at the
current joints + only 10 canonical seeds; its **cold round-trip is only ~96.7 %**,
and for these specific far configs the miss rate is plausibly much higher — so
it rejects poses the real robot *can* reach. If true, the gate is the bug, not
the geometry. **Strong candidate.**

**H2 — the framed candidates are genuinely (near-)unreachable.** NIC framing
uses `prefer_far_standoff=True` + the cage box (tall cards) → the center camera
only frames the sector at HIGH standoff (0.85–1.0 m); at that standoff the TCP
is ~0.8+ m from base — at/over the reach edge. The bore-side −X shift pushes it
further. So *every framed candidate* may sit outside the reachable envelope.
This is a **fundamental conflict**: "frame the tall cards well" wants far/high;
"stay reachable" wants near. The sphere hid this (it happily passed far poses);
the IK gate exposes it honestly. At 70° a reachable pose existed; at 0° maybe
not. **Also strong.**

These have OPPOSITE fixes, so the next session must distinguish them first:
- If **H1**: make IK trustworthy — implement **closed-form analytic UR5e IK**
  (8 branches, exact, no seed dependence). This is now safe because
  `autocalibrate` gives the exact base (Rz180) + tool, so the closed-form
  solver can run in the calibrated model frame. Then reachability is definitive.
- If **H2**: change the VIEW STRATEGY — allow the NIC sector to be framed from a
  CLOSER, reachable standoff (drop or cap `prefer_far`, shrink the cage box,
  relax `require_all_cameras`/`min_required_clearance`), i.e. accept a nearer
  view that the IVM can still use. This loops back to the view-geometry question
  the user spent a long time on; the constraint now is "must be reachable."

**How to distinguish (offline, no rebuild):** in
`c:/Users/anshu/College/aic/aic/flowstate/aic_perception`, reproduce a realistic
scenario (see the sweep scripts this session used: import
`test/test_board_stage2.py::_production_camera_rig`, build a board pose, a
`UR5eArm` with a 0.16 m tool and `base=Rz180`, seed IK with a plausible
current-joint config). For the exact failing poses, do a **400k brute-force**
over joint space (as done this session) to check whether ANY config reaches them
within a few mm — if yes and DLS said no → H1; if no → H2. Better: get the real
`(measured_joints, base_T_tcp)` and the 255 candidate `base_T_tcp` from the live
run (add a one-line log dumping the top candidate's pose + the brute-force
result) to settle it in one deploy.

**The user's stated preference:** "go back to when perception actually got
proper IK and derive from there." Concretely that means: consider reverting the
hard IK gate to the sphere (or making the gate *soft* — prefer reachable but
fall back to the best geometric pose rather than `done=false`), get moves
happening again, and rebuild reachability more carefully. A **soft gate** +
the BT `done` guard would at least stop the crash and restore motion while H1/H2
is resolved.

---

## 10. Recommended next steps (in order)

1. **Stop the crash (BT):** gate Move Robot on `result.success && result.done`.
   Non-negotiable regardless of H1/H2.
2. **Decide the gate posture** with the user: hard gate (current, honest but can
   produce `done=false`) vs **soft gate** (publish best reachable, else best
   geometric — restores motion, risks an occasional Move-Robot IK reject). A
   soft gate is the pragmatic bridge.
3. **Settle H1 vs H2** with the brute-force test / a one-deploy pose dump.
4. If H1 → **closed-form UR5e IK** (replace DLS). If H2 → **reachable-first view
   strategy** (cap standoff to reachable, closer NIC view; verify IVM still
   detects). Likely BOTH are partly true → do both.
5. Re-run the full suite (`pytest test/ -q`), update `docs/BOARD_SEARCH_HANDOFF.md`,
   rebuild + install (§5).

---

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
  `max_reach_checks=24`, standoffs 0.30…1.25.
- Observed failing published poses (sphere, pre-gate): `(-0.28,0.07,0.76)` std
  0.90; `(-0.09,0.41,0.74)` std 0.85 → Move Robot IK reject.

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
