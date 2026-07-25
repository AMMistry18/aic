# Agent Handoff — SC (duplex fibre) insertion: controller built, perception unresolved

**Repo:** `/Users/satya_anandh/Developer/aic` · **Local branch:** `observe-seat-wrench-micro-correction` (pushes to `main`) · **Date:** 2026-07-25
**Companion doc:** `.artifacts/HANDOFF_seat_windup.md` — the SFP story. Read §1, §2 and §6 of it for repo conventions; they apply here unchanged.

---

## 1. Repo conventions you must not get wrong

**The deployed code is the overlay, not the main tree.** `docker/aic_model/Dockerfile` line 12
copies `aic_model/`, then line 35 copies `docker/aic_model/v50_overlay/` over the top.
**Edit BOTH copies of any file that exists in both.** For SC that means:
```
aic_model/aic_model/sc_controller.py
docker/aic_model/v50_overlay/aic_model/sc_controller.py     <- must stay byte-identical
aic_model/aic_model/RLInsert.py
docker/aic_model/v50_overlay/aic_model/RLInsert.py          <- these two differ legitimately
```
`sc_controller.py` is currently identical in both; verify with `diff` after every edit.
The two `RLInsert.py` copies differ by design (the overlay has the v50 plug-priming path).

The Dockerfile copies `v50_overlay/aic_model/*.py` wholesale, so **new modules are picked up
automatically** — no Dockerfile change was needed for `sc_controller.py`, and none should be.

**Do not modify or commit `docker/aic_model/Dockerfile`.** It is the user's local-source build
hack, intentionally uncommitted (md5 `49fa78a848809227e7dc2322adb8535a`). It blocks rebases:
stash it, rebase, pop, and byte-verify it came back.

**Test command** — plain `pytest` fails with `PluginValidationError`:
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .pixi/envs/default/bin/python -m pytest \
  aic_model/test/test_sc_controller.py aic_model/test/test_v50_controller.py -q
```
Currently **45 pass** (41 before `a93e315`). Five other files in `aic_model/test/` have collection errors that
**pre-exist on HEAD** (verified by stashing) — `test_board_search`, `test_rl_insert_contract`,
`test_sc_plug_pose_geometry`, `test_sfp_plug_pose`, `test_visual_gap`. Never run the directory
as a whole; it also fails to collect.

`origin/main` receives pushes from a collaborator doing board-search / perception / IK work
under `flowstate/`. No file overlap so far. Fetch and rebase before pushing.

---

## 2. State — everything below is pushed, HEAD == origin/main == `a93e315`

```
a93e315 fix SC size gate: it was rejecting the model's own label rectangle
9709610 fix SC port perception: select detections before triangulating
39d979e strip the cable-instance prefix from scoring insertion events
e534a15 add scripted SC insertion to the insert_cable skill
```
(`fbd09c8`, `99db35c`, `3b305ba`, `2272dc0` in between are the collaborator's board-search work.)

**None of this has run on hardware.** The friend who rebuilds/deploys was asleep when this
handoff was written. The SC path has had exactly **one** field run, before `9709610`.

### What to upload
**`9709610` as pushed — nothing else.** No model/weights change: `best_sc_pose.pt` is unchanged
and already baked into the image, and the retrain was decided against (see 6d-PRE). No Dockerfile
change: it copies `v50_overlay/aic_model/*.py` wholesale, so `sc_controller.py` is picked up
automatically. The next build carries three unvalidated changes at once — SFP alignment law, SFP
event normalisation, SC insertion + perception fix — so read the logs for all three.

### SFP side, for context
Two SFP changes are also unvalidated in the field:
- `f07d3a1` + a follow-up made the seat alignment law a **low-pass of a clamped proportional
  target**, not an accumulator. The accumulator saturated at its clamp ~3 samples after first
  chamfer contact (at only 1.7 N) and jammed the plug; shrinking the clamp had only moved the
  jam from 37 mm to 2 mm. Replaying field logs through the new law: zero saturated samples
  where the old law saturated 6/60, 32/56 and 41/95.
- `39d979e` fixed `_normalize_event`, which stripped whitespace and slashes but **not** the
  `cable_N#0#` prefix the scoring topic publishes. The equality test in `_event_status` could
  therefore never match — every correctly-seated SFP run reported a wrong-port hard failure.
  **Expect SFP success numbers to change once this deploys**; prior logs undercount.

---

## 3. SC — what exists

`aic_model/aic_model/sc_controller.py` (+ overlay copy). Dispatched from `RLInsert._run` when
`task.plug_type == "sc"`; in the overlay the branch is taken **before** `prime_v50_plug_pose`,
which loads the SFP-only plug-pose model and fails closed.

It is deliberately **fixed-grasp**: there is no SC plug-pose model, so the tip comes from the
TCP through a static transform.

Reuses from `v50_controller`: `WallProgressWatch`, `clamp_vector_norm`, `axis_angle`,
`rotation_from_axis_angle`, and the status constants. `next_sc_depth` is a deliberate copy of
`next_persistent_depth` rather than a parameterisation, because that one closes over the SFP
`INSERT_DEPTH_M` module constant and the SFP build was mid-deploy.

### Task fields — verified against `aic_engine/config/generated_boards/*.yaml` and `sc_eval_config.yaml`
```
plug_type           = sc
plug_name           = sc_tip
port_type           = sc
port_name           = sc_port_base
target_module_name  = sc_port_0        (or sc_port_1)
```
`target_module_name/port_name` must reproduce the Gazebo TouchPlugin `<namespace>`, which for
SC is `${prefix}/sc_port_base` (`aic_assets/models/SC Port/sc_port_macro.xacro`). Confirmed
independently by the SDF the user pulled from the Intrinsic repo.
**Gotcha:** the dispatch reads `(task.plug_type or "sfp").lower()` — a blank `plug_type`
silently runs the **SFP** path.

---

## 4. Geometry ground truth — derived, verified, do not re-derive

From `aic_assets/models/SC Port/model.sdf`, cross-checked against the upstream Intrinsic SDF
(identical, every box):

| Quantity | Value | Source |
|---|---|---|
| Seated depth | **15.64 mm** | `sc_port_base_link_entrance` at −0.01564 of `sc_port_base_link` |
| Bore opening | 9.708 × 8.10 mm | divider faces ±1.496, wall inner faces ±11.204; plates ±4.05 |
| Duplex inner | 22.408 × 8.10 mm | same |
| Outer face | 25.78 × 9.30 mm | `cube_collider_box.001` x-size 0.025781; wall height 0.0093 |
| Bore pitch | 12.70 mm | matches the real SC duplex standard — sanity check on the reading |
| Plug body | 20.0 × 6.4 mm | `sc_plug_pose_geometry.py` keypoints |
| **Clearance** | **1.2 mm lateral, 0.85 mm vertical per side** | 8× looser than SFP |

**Insertion axis is board −Z, same as SFP.** Composing the board pose `rpy(1.57, 0, 1.57)` with
the port-base `rpy(π/2, π, 0)` gives `R_board_from_port = [[0,0,1],[1,0,0],[0,1,0]]`, mapping
the port-base +Z to board −Z. This is why the SFP entrance-frame estimator
(`_estimate_sfp_port_orientation`, which hardcodes "insertion axis is world −Z") is reusable
for SC unchanged. `sc_port_link` +X maps to board +Y, so the opening's long axis is board Y.

**Why the constants were rescaled and not copied:** SFP's force lead is 8 N / 500 N/m = **16 mm**,
longer than the entire SC insertion. Copied verbatim it would command the plug through the back
of the port on the first stall. SC caps the lead at 5 mm, uses a 5/7/12 N ladder, halves approach
speed, and keeps overtravel (1.5 mm) and mouth zone (2 mm) at SFP's *fractions* of the bore.
`SCConfig.validated()` rejects any lead ≥ the bore, and a test asserts the SFP numbers fail.

---

## 5. The one field run, and the perception fix

```
[rl] task: cable=cable_0 plug=sc_tip (sc) port=sc_port_base module=sc_port_0
[sc] no candidate under 5.0px select gate (best 79.3px) -- rejecting frame     x7
[sc] perception consensus failed: only 0/7 frames passed reproj (need 3)
```
79.3 px **stable to a tenth of a pixel across all 7 frames** — deterministic geometry, not noise.

**Diagnosis.** `detect_nic` returns ONE detection carrying 8 keypoints (both SFP cages), so the
SFP path has no cross-camera correspondence ambiguity. `detect_sc_pose` returns **one detection
per SC adapter**, and the scene holds ~6 identical adapters. The old code took the top 5 per
camera by confidence and ran `itertools.product` over all cross-camera combinations; pairing
adapter A in one camera with adapter B in another gives non-intersecting rays whose triangulated
midpoint reprojects tens of pixels away. If the target fell outside the top 5 in even one camera,
no correct pairing existed at all.

**Fix shipped in `9709610`** (implemented by Codex, reviewed and amended):
- Select detections **before** combining: project an anchor into each image, keep only detections
  within `SC_MAX_DETECT_PX_FROM_TIP` (250 px), then triangulate survivors. Cap raised 5 → 8,
  applied after filtering. Falls back to confidence ranking if the anchor is unavailable.
- **The anchor is the gripper TCP, deliberately not `sc_tip_pose_from_tcp`** — the SC tip
  transform is the uncalibrated SFP default, and a perception gate must not be centred on a
  constant known to be wrong. A test with a real pinhole fixture fails if anyone reverts this;
  the pre-existing `_FakeProjectionCore` maps every point to one pixel and cannot catch it.
- Diagnostics: `SC_PERCEPT_CAMERA` (per-camera counts before/after, confidences, centroids,
  anchor pixel) and `SC_PERCEPT_BEST` (per-camera keypoints, per-keypoint reproj residuals,
  triangulated width/height) — the latter emitted **even when the candidate is rejected**.
- A camera emptied by the pre-filter now says so explicitly instead of surfacing as a generic
  "no candidates".

### Field run 2 (2026-07-25, with `9709610` deployed) — pre-filter WORKED, new failure downstream

```
[sc] SC_PERCEPT_CAMERA cam=left_camera   mode=tcp_filter radius=250.0px before=5  after=2 tcp_px=[576.3, 829.3]
[sc] SC_PERCEPT_CAMERA cam=center_camera mode=tcp_filter radius=250.0px before=10 after=2 tcp_px=[575.5, 821.8]
[sc] SC_PERCEPT_CAMERA cam=right_camera  mode=tcp_filter radius=250.0px before=3  after=2 tcp_px=[576.4, 828.2]
[sc] multiview matching found no SC opening candidates          x7
[sc] perception consensus failed: only 0/7 frames passed reproj (need 3)
```

**The 79.3 px reprojection failure is gone.** The pre-filter narrowed every camera to 2 survivors
and no candidate was rejected on reproj. That part of `9709610` is validated.

**`SC_PERCEPT_BEST` never appears**, which is decisive: that line is emitted whenever the
candidate list is non-empty, so *every* combination was discarded by an in-loop gate before a
candidate existed. Remaining gates in `sc_multiview_candidates` are: triangulate exception,
`X[2] < -0.05 or > 0.25`, `_estimate_sfp_port_orientation` returning None, and the size gate.

**Cause: the size gate, and it was my bug. FIXED in `a93e315` (2026-07-25), not yet field-tested.**
`SC_MIN_OPENING_M` was `0.005`, sized to bracket the duplex (22.41 × 8.10) and bore
(9.71 × 8.10) hypotheses, both ≥8.1 mm tall. But the model emits an **8.8 × 6.0 mm** box
(see 6a), so the measured height sat **1 mm above the floor** and ordinary triangulation noise
dropped it under. Floor lowered to `0.002`, and every rejection path now logs which gate fired
and what it measured (`SC_PERCEPT_REJECT`).

Regression test: `test_size_gate_survives_a_realistic_underestimate_of_the_short_axis` builds a
real two-camera DLT fixture using RLInsert's own reprojection and orientation code, and a 6.0 mm
axis measured 20% short. It **fails against the old 5 mm floor and passes at 2 mm** — i.e. it
reproduces the field failure rather than merely asserting the new constant. 41 → 45 passing.

**How to confirm on the next run:** `SC_PERCEPT_BEST` appearing at all is the proof, since that
line only prints once a candidate survives. If `no SC opening candidates` persists, the new
`SC_PERCEPT_REJECT` line names the gate — no more inference from source.

Still worth doing later: derive the gate from the *matched* hypothesis rather than a fixed pair of
absolute bounds (see 6a).

**The correct correspondence does appear to be surviving the filter.** Among the survivors:
left `(650.2, 676.9)`, center `(573.8, 634.9)`, right `(499.6, 681.7)` — x decreasing
monotonically left→center→right, consistent with one 3D point across a horizontal rig. The
geometry is there; the gate is discarding it.

**~~Unexplained~~ RESOLVED — the TCP projecting to nearly the same pixel in all three cameras**
(`(576.3, 829.3)`, `(575.5, 821.8)`, `(576.4, 828.2)`, and 576 = half of 1152) **is correct
behaviour, not an extrinsics bug.** `aic_description/urdf/ur_gz.urdf.xacro:195-218` mounts all
three cameras on the wrist `cam_mount_link`, symmetric about the tool axis (x = −0.09326, 0,
+0.09326) and each yawed to converge on it (0.5236, 1.5708, 2.6180 rad). The TCP lies on that
axis, so it lands at the image centre in all three by construction. The ±93 mm baseline is real
and triangulation is well conditioned. **Do not spend time on the extrinsics.**

**~~Missing diagnostic~~ SHIPPED in `a93e315`.** `sc_multiview_candidates` now records every
discarded combination and emits `SC_PERCEPT_REJECT` with a per-reason count, the active gate
bounds, and the measured width/height of the rejects — e.g.
`counts={'size': 8} size_gate=[2.0, 30.0]mm sample=['size(0.8x0.5mm)', ...]`.

### How to read the next run
- Reproj now low → perception solved; move to `SC_TIP_IN_TCP` calibration.
- Any `SC_PERCEPT_CAMERA … after=0` → radius too tight; raise `RL_INSERT_SC_MAX_DETECT_PX_FROM_TIP`.
- Still ~79 px with residuals spread evenly over all 4 keypoints → correspondence still ambiguous.
- Still ~79 px with residuals **concentrated on 2 of 4 keypoints** → keypoint order flipping
  between viewpoints. Not fixable by filtering; needs a retrain. Note `sc_plug_pose_geometry`
  already carries `SC_PLUG_FLIP_IDX = [1,0,3,2,5,4,7,6]`, so flips are a known SC issue.

---

## 6. OPEN WORK — this is what to pick up

**Start here (2026-07-25, after field run 2).** In priority order:

1. ~~Size gate blocking every candidate~~ and ~~add rejection diagnostics~~ — **both shipped
   together in `a93e315`, awaiting a field run.** See §5. Read the next run's logs first: if
   `SC_PERCEPT_BEST` appears, perception is unblocked and the next item is 6c.
2. **6c — `SC_TIP_IN_TCP` calibration.** The hard blocker, and the only remaining one that
   *needs the robot*. SC cannot seat until this is solved. Start it as soon as perception
   produces a pose.
3. Then 6a (label-convention constants, which the size gate should be derived from) and 6b
   (lateral selection gate). Both are cleanups that make the pipeline robust rather than
   unblock it.

**Note there is no runtime env knob in Flowstate** — the service manifest has no env field, so
`RL_INSERT_SC_*` overrides only take effect if baked into the image via a Dockerfile `ENV`. Since
that costs a rebuild either way, prefer changing the default in both `sc_controller.py` copies.
And Flowstate keys services by manifest identity: **reinstalling under the same asset name does
not replace the running image** — bump the version in `aic_model.manifest.textproto` or the
deploy silently runs the old code.

### 6a. The label-convention mismatch (a real bug in shipped code)

There are **two competing SC label definitions** in the repo:

| Collector | Rectangle | long/short | Traceable to geometry? |
|---|---|---|---|
| `DataCollectorScPoseGT.py` (`SC_HALF_WIDTH_M=0.0044`, `SC_HALF_HEIGHT_M=0.0030`) | 8.8 × 6.0 mm | 1.467 | **no — matches nothing** |
| `DataCollectorPoseSC.py` (`SC_FULL_WIDTH_M=0.02578`, `SC_FULL_LENGTH_M=0.00927`) | 25.78 × 9.27 mm | 2.781 | yes — outer face |

**The shipped `best_sc_pose.pt` follows the 8.8 × 6.0 convention.** Established empirically:
running the model locally on `testing/check_sc_previews/*.png` gave quad long/short ratios of
**1.46, 1.53, 1.50** across three detections. Model metadata: task `pose`, one class `sc_port`,
`kpt_shape [4, 3]`.

**Refined 2026-07-25 — corners, and a small CENTRED patch. Rendered evidence:
`.artifacts/sc_detections.png`.**
- **Corners, not edge-midpoints.** The aspect ratio alone cannot distinguish these: a diamond
  with 8.8 × 6.0 *diagonals* also gives 1.467. The **diagonals** settle it — measured equal to
  within 1–3% while adjacent sides differ 1.44–1.51, which is a rectangle. (Worth knowing what
  the diamond reading would have implied: `width` and `height` in `sc_multiview_candidates` both
  collapse to √(4.4² + 3.0²) = 5.33 mm, i.e. 0.33 mm above the old floor and failing every frame.
  It is a *better* fit to the observed failure than the rectangle reading, so rule it out by
  measurement rather than by assumption.)
- **The label covers ~a quarter of the part.** Segmenting the cyan adapter and comparing extents
  gives quad/adapter = **0.25–0.26** in all three frames. The label is a small rectangle in the
  *middle* of the adapter, aligned with its long axis — visibly not the port outline, not a bore.
- Detection quality itself is fine: conf 0.90–0.92, box squarely on the adapter every time.
- **Look at the render before theorising.** Two rounds of this were argued from ratios alone; the
  picture settled it in one pass.

`sc_controller.SC_OPENING_HYPOTHESES` currently holds **neither** — it has my SDF-derived
guesses (duplex 22.41 × 8.10 and single-bore 9.71 × 8.10). Consequences, both live:
1. `classify_opening` reports `single_bore` and logs a warning telling the operator the pose is
   offset by 6.35 mm and needs correcting. **Acting on that would push the plug half a bore
   off-centre.** Both conventions project from `sc_port_base_link_entrance`, i.e. the duplex
   centre, so the correct offset is **zero** in both cases.
2. The candidate score adds `shape_residual × 250`. With a ~3 mm residual on every genuine
   detection, a ghost with accidentally-closer dimensions can outrank a real port.

**Planned fix (agreed with the user, not yet written):** replace the two SDF guesses in
`SC_OPENING_HYPOTHESES` with the two **real label conventions** (8.8 × 6.0 and 25.78 × 9.27),
both with zero bore offset. `classify_opening` then auto-detects which model is loaded and logs
it, so one build works with the current weights *and* any retrain, and the user can A/B via
`AIC_SC_POSE_WEIGHTS` without a code change. Also update `LOCAL_SC_PORT_KPS` (the single-view
PnP fallback rectangle) to match whichever is selected.

### 6b. Selection gate is on the wrong axis

`SC_MAX_HANDOFF_SELECT_M = 0.030` measures **3D** distance from the tip to a candidate, which
mixes in handoff height. Adapters cannot be closer than their own width, **25.78 mm**, so at a
15 mm handoff a shoulder-to-shoulder neighbour sits at √(25.78² + 15²) ≈ 30 mm — inside the gate.
Tightening it instead risks rejecting the target when the macro hands off higher.

**Planned fix:** gate on **lateral (board XY) distance**, since insertion is straight down
board −Z. Target is then ~0–3 mm lateral and any neighbour ≥25.78 mm, independent of handoff
height. ~10 mm gate, huge margin. Add `SC_MAX_LATERAL_SELECT_M`.

Keep the 250 px pixel pre-filter **coarse** — the TCP sits ~58 mm from the tip, ≈190 px at
working distance, so a tight pixel radius would exclude the target. Division of labour: pixels
kill gross cross-board mis-pairing, the lateral gate picks between neighbours.

Note: the qualification xacro places only two SC ports, 41 mm apart in Y (`sc_port_0` at
y=0.0295, `sc_port_1` at y=0.0705, each sliding independently in X over −0.06…+0.055). The
user's screenshots show ~6 adapters, so **the Phase 1 board differs from the qualification
xacro** and the 41 mm figure must not be relied on. 25.78 mm is the only safe bound.

### 6c. `SC_TIP_IN_TCP` is uncalibrated — the hard blocker

Defaults to the SFP grasp transform; `run()` logs a warning every run. **SC will not seat until
this is solved.** Re-solve with `RL_INSERT_CALIB_DUMP=1` exactly as the SFP transform was, then
set `RL_INSERT_SC_TIP_IN_TCP_POS` / `_QUAT` / `RL_INSERT_SC_TIP_CALIBRATED=1`.

**Do this over ~10 grasps, not one.** The spread of the solved transform is grasp repeatability,
which is the measurement that decides 6d. Needs the robot, so it cannot happen before morning.

### 6d-PRE. READ THIS BEFORE 6d — the retraining decision was CLOSED on 2026-07-25: **do not retrain**

6d below records the reasoning as it stood mid-evening. It was overtaken. Final position:

**Yaw does not need to come from perception, so the model's weak yaw conditioning does not matter.**
- The task board is level in every trial (`roll: 0.0`, `pitch: 0.0` across all 13 configs), so the
  insertion axis is world −Z a priori. Only board yaw varies.
- The upstream macro hands off already aligned to the port. Measured SFP field values for the
  yaw component of `rot_err_deg`: **−1.92, −3.92, −0.80, −4.22, −0.19, −3.65, −0.09** degrees —
  worst case 4.2°, against the 6.9° budget from the 1.2 mm lateral clearance on a 20 mm plug.

**The 8.8 × 6.0 box being non-physical does NOT corrupt position.** `LOCAL_SC_PORT_KPS` is
symmetric about the origin and projected from `sc_port_base_link_entrance`, so the four labelled
points are centred on the true mouth regardless of how wrong their spread is. The pipeline
triangulates all four and takes the mean, which lands on the entrance. Position is trained on a
correct target; only the orientation baseline is fictional.

Net: wrong-looking box, usable position, rotation we take from the handoff instead. Retraining
would have cost ~5 h for no expected gain.

**A likely bug this exposed, NOT yet fixed:** `ScInsertionController` commands
`target_rotation = self.Rp` in both `_align` and `_seat` — i.e. it rotates the plug to match the
*perceived* port yaw, discarding the macro's good handoff yaw. If perception yaw error exceeds
~4°, the controller actively degrades an alignment that arrived inside tolerance. Position
servoing is unaffected (the lateral correction reconstructs the same world vector regardless of
frame yaw), so the fix is narrow: build the seat frame with the axis from perception and the yaw
preserved from the tip orientation at handoff.

**Deliberately not implemented yet**, because the measurement that justifies it does not exist:
the first SC run that gets past perception prints `[sc] handoff check: ... rot_err_deg=[...]`.
- yaw component under ~5° → the handoff is good, make the change, no model work ever needed.
- over 6.9° → the SC macro is worse than the SFP one; then yaw perception matters and 6d reopens.

**Dataset risks worth five minutes before ever retraining** (both would genuinely corrupt
position, unlike the box size):
1. `DataCollectorScPoseGT.py` lines 172-176 try five TF frames in order and fall back from
   `sc_port_base_link_entrance` to `sc_port_base_link` — the **seat**, 15.64 mm deeper than the
   mouth. If the entrance frame was not published during collection, every label is
   systematically 15.64 mm too deep.
2. Around line 259 a second path labels from the **HSV blue-blob detector's** corners when GT is
   unavailable. Those are not a projected rectangle at all, so any samples produced that way
   follow a different convention inside the same dataset.
Check the per-sample JSON for which path/frame was used before trusting the set.

---

### 6d. Retraining — reasoning as of mid-evening 2026-07-25 (SUPERSEDED by 6d-PRE above)

The user has compute and wanted to use the night. Agreed position:

- **Retrain the PORT model: yes.** Justified on two independent grounds. (1) The current
  8.8 × 6.0 box matches no physical feature in the SDF — likely hand-entered. (2) It is badly
  conditioned for yaw: at ~0.3 m and fx ≈ 1000 px, an 8.8 mm long axis is ~29 px, so 1–2 px
  keypoint noise gives 4–8° yaw error, against a budget of `asin(1.2/10)` ≈ **6.9°** set by the
  lateral clearance. The 25.78 mm box is ~86 px → 1.3–2.7°, ~5× margin. It is also the crispest
  visible edge on the part.
- **Do NOT retrain the plug model yet.** SC clearance is 1.2 mm/side against SFP's near-zero, so
  a once-calibrated fixed grasp may simply suffice. The deciding measurement is grasp
  repeatability from 6c, which needs the robot. Also, the SFP plug-pose estimator has a known
  failure where it latches the *other cable end* (~370 px camera jump → 148 px reproj rejects);
  on an `sfp_sc` cable an SC plug model inherits that problem by construction.
- **Ship any retrain as a NEW weights file**, never over `best_sc_pose.pt`. `AIC_SC_POSE_WEIGHTS`
  already exists as an env override, so old vs new can be A/B'd on hardware without a rebuild.

**Blocking practical issue:** the SC dataset is **not on this machine** — only `sfp_plug_pose`
survives under `~/aic_perception_data/`. `train_sc.py` expects
`~/aic_perception_data/pose_sc/aic_sc_pose.yaml` and asserts it exists. Changing the rectangle
means **relabelling**, not just retraining. The user was told to check their compute box:
- per-sample JSON kept the camera↔port transforms → re-project labels onto existing images,
  minutes, skip collection;
- only pixel labels kept → re-collect by running the Gazebo sim with `DataCollectorScPoseGT`
  (viewpoints sampled ±60 mm laterally, −10…+160 mm height — close range *is* in-distribution,
  so data coverage is not the weakness; the convention is).

Collection is the multi-hour step; training is the fast one. `sc_pose_sanity_check.py` audits
labels and writes a CSV/markdown report — worth running before retraining.

---

### 6e. Upstream facts from a prior session that MUST be checked before acting on 6d

These come from the `sc-port-insertion-plan` memory (2026-07-16, sourced from
`intrinsic-dev/aic-phase-1` and `intrinsic-dev/aic`, **not present in this local repo**, so they
could not be verified here). Two of them materially affect the retraining case:

1. ~~SC port yaw is evaluation-fixed at 0°~~ — **RESOLVED 2026-07-25, and the memory note is
   misleading. Do not act on it.** The port is `<static>true</static>` with only fixed joints and
   zero yaw *relative to the board*, which is what that note meant. But **the task board itself
   is spawned at a random yaw every trial**: `aic_engine/config/generated_boards/*.yaml` give
   board yaw values of 2.9833 … 3.3157 rad, i.e. roughly **±10° around π**. The ports are rigid
   on the board, so their yaw **in the robot frame varies per trial by more than the 6.9°
   tolerance** set by the 1.2 mm lateral clearance on the 20 mm plug. Yaw therefore must be
   measured from vision every trial, and cannot be replaced by a constant prior.
   **The yaw-conditioning argument for retraining (6d) stands.** The xacro also exposes
   `sc_port_N_yaw` args (default 0.0), so per-port yaw is possible in principle though unused in
   the configs seen here. `PerceptionInsert._choose_sc_yaw_by_tip_error` (line 1158) exists and
   is about resolving the 180° ambiguity of a near-symmetric rectangle, not about yaw being fixed.
2. **Insertion triggers at 1 mm tip proximity, not contact** (since the 2026-07-09 assets,
   aic#593). `sc_controller` currently drives to `seat_candidate_depth_m` (15.2 mm) and then
   waits for an event. If the trigger is proximity-based the event may fire far earlier, and the
   force ladder may be doing work it does not need to. Check against the first successful run.
3. **OPEN tunneling bug** (#121/#137): fast approach lets the plug pass *through* the thin SC
   port colliders and drift, blocking neighbouring ports in later runs. Slow final approach is
   mandatory, ≤0.3 mm/step near the mouth. SC already halves approach speed and applies a 0.25×
   mouth slowdown, but verify the per-step figure against this.
4. **3 SC ports on rail 0, 2 on rail 1 = 5 total.** This explains the ~6 adapters in the user's
   screenshots and confirms the local xacro (2 ports) is not the Phase 1 layout. Reinforces 6b:
   do not rely on the 41 mm figure.
5. **Phase 1 Tier 3 scores the average of both cable ends, and runs 2–5 only execute if the
   prior run fully inserts BOTH ends** — so SC gates the whole sequence. SC is not optional
   polish; it is on the critical path.
6. FT tare bug (#112): tare affects `/fts_broadcaster/wrench` only, not the `/observations`
   composite.

---

## 7. Things already ruled out — do not re-investigate

- **Insertion axis.** SC is board −Z, same as SFP. Verified twice from the SDFs.
- **Bore offset.** Both label conventions centre on `sc_port_base_link_entrance` = duplex centre.
  The duplex plug enters as one unit; there is no bore to choose. The SC port declares **one**
  TouchPlugin where the NIC declares two, confirming this.
- **Training-set viewpoint coverage.** `DataCollectorScPoseGT.py:78-80` samples −10…+160 mm in
  height, so close-range views are included. The `check_sc_previews` frames that detect poorly
  are distant board overviews (~11 px ports) and are not representative.
- **`Dockerfile.plug_relative_v50`** — the narrow second build path, copies three named files and
  omits `board_search.py` as well as `sc_controller.py`. Looks superseded; confirm with the user
  before spending time syncing it.

## 8. Known-but-parked

- **SFP wrong port.** Every SFP insertion event named `sfp_port_1` when `sfp_port_0` was
  requested. `RLInsert._select_candidate` picks the port **nearest the tip and ignores the
  requested slot**. The user explicitly said their other Flowstate processes handle this — do
  not "fix" it without asking.
- **Plug-pose estimator instability** (SFP) — left camera can jump ~370 px latching the other
  cable end. Separate bug.
- `RL_INSERT_REPORT_MISS_AS_SUCCESS=1` means the skill reports success on a failed insertion so
  the 5-run process continues. Truth is only in the logs.

## 9. Relevant memories
`aic-repo-gotchas`, `flowstate-deploy-recipe`, `aic-phase-1-flowstate`, `aic-phase-1-task`,
`sc-port-insertion-plan` (partly stale — the `docs/SC_INSERTION_PLAN.md` it names no longer
exists), `seat-align-windup`, `port-58px-reproj-root-cause`.
