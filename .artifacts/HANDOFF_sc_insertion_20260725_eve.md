# Agent Handoff — SC insertion, evening 2026-07-25

**Supersedes the SC-specific parts of `.artifacts/HANDOFF_sc_insertion.md`.** That
document's §1 (repo conventions) and §3 (what exists) still hold. Its §4 geometry
table, §5 diagnosis and §6a/6c plans contain errors corrected below — read this
first, then that one for background.

**HEAD:** `5b5f478`, pushed to `origin/main`. 73 tests pass.

---

## 1. Read this before touching anything

Same as the previous handoff, restated because it still bites:

- **Edit BOTH copies.** `aic_model/aic_model/sc_controller.py` and
  `docker/aic_model/v50_overlay/aic_model/sc_controller.py` must stay
  byte-identical. `diff` them after every edit. Current md5 is shared.
- **Test command** (plain `pytest` fails with `PluginValidationError`):
  ```
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .pixi/envs/default/bin/python -m pytest \
    aic_model/test/test_sc_controller.py aic_model/test/test_v50_controller.py -q
  ```
  73 pass. Never run the whole test directory; five other files have collection
  errors that pre-date all of this.
- **`deploy/flowstate/Dockerfile.aic_model_service` and
  `aic_model.manifest.textproto` are untracked local files.** Do not
  `git add -A` — I did once and swept them into a commit. Stage explicitly.
- **`docker/aic_model/Dockerfile` is TRACKED and clean**, contradicting the
  previous handoff's claim that it is "intentionally uncommitted" with md5
  `49fa78a8…`. Actual tracked md5 is `539c75a9…`, last touched by `89b6ede`.
  The stash-rebase-pop dance that handoff prescribes is not needed.
- **No runtime env knobs in Flowstate.** `RL_INSERT_SC_*` only takes effect if
  baked into the image. Prefer changing defaults in both `sc_controller.py`
  copies. Bump `aic_model.manifest.textproto` or the deploy silently reuses the
  old image.

---

## 2. Commits this session

```
5b5f478  resolve the SC keypoint corner order instead of assuming it matches
def336e  run the grasp calibration dump before perception, not after
4d3d4a0  probe the SC tip frame names the sim actually publishes
4e367a6  make RL_INSERT_CALIB_DUMP actually work on the SC path
cbd78bc  correct why the SC opening is 7.85mm, and cite upstream on the rails
b6d9fd6  refuse an impossible handoff depth; use the real SC label conventions
b92c502  don't rotate the SC plug onto the perceived port yaw
```
(`f275de4`…`28e639f` in between are the collaborator's board-search work, which
added `sc_port_2..4`.)

**Only `b92c502` has been field-validated.** Everything after it is untested on
hardware.

---

## 3. Three field runs, and what each proved

| run | epoch | deployed | outcome |
|---|---|---|---|
| A | 1785005084 | `a93e315` | perception unblocked; alignment timed out at 15 s |
| B | 1785013590 | `b92c502` | alignment converged; fake seat at 21.13 mm |
| C | 1785017165 | `4d3d4a0` | perception failed 0/7; no calib dump |

### Run A — the size-gate fix worked
`SC_PERCEPT_BEST` appeared, 7/7 consensus, reproj 4.35 px. `a93e315` validated.
Then `[sc] alignment did not converge in 15.0s`.

Cause: `rot_err_deg=[3.19, -4.37, -89.55]` — 89.71° about an axis 3.46° off the
insertion axis, stable across all 7 frames. `_align` slews at 1.5°/iteration
against a 15 s budget, so 89.7° needs ≥60 iterations.

**That 90° is a frame-convention offset, not a perception error.** Three
independent reasons:
1. `_estimate_sfp_port_orientation` builds its in-plane axis from exactly the
   vector `sc_multiview_candidates` calls `width`, and width > height, so
   `Rp[:,0]` really is the opening's long axis.
2. A non-square rectangle pins that axis to within 180°, never 90°.
3. The plug is 20.0 mm across and the opening 7.85 mm tall — a plug genuinely
   turned 90° could not enter, yet the handoff was only 3.74/1.75 mm off
   laterally.

Removing an exact −90° about port Z leaves **4.89°**, inside the 6.9° budget.

**Do not apply 6d-PRE's ">6.9° → retrain" rule to the raw 89.55°.** It would
trigger an unnecessary retrain. The macro's handoff twist is fine.

**Fix (`b92c502`, validated):** `Rp` drives position and the wrench frame; a new
`Rs` (`seat_frame()`) drives rotation targets — same insertion axis, in-plane
twist taken from the plug as handed over. They share column 2, so the lateral
plane and every position correction are unchanged. `acc_tilt` is measured about
`Rp`'s axes, so `_seat` composes `Rp @ tilt @ R_yaw`, not `Rs @ tilt`.

**Never "fix" an alignment timeout by raising `align_timeout_wall_s` or
`align_max_rotation_step_rad`.** That lets the robot complete a 90° turn it
should not make and drive a 20 mm plug at a 7.85 mm opening.

### Run B — alignment fixed, fake seat exposed
```
[sc] seat frame: preserve_handoff_yaw=True twist_vs_perceived_yaw_deg=-89.56
[sc] aligned: lateral=0.95mm rot=0.09deg depth=21.13mm
```
Converged in ~6.5 s. But `depth=21.13 mm` is past the 15.64 mm fully-seated
depth, so `_seat` saw `depth >= seat_candidate_depth_m` (15.20 mm) on entry,
skipped the whole approach, and waited for an event that could not arrive.
`SEAT_WRENCH axial_N=-0.22` — the plug touched nothing.

Root cause: the handoff check read **+6.99 mm** *before any motion* (+7.04 mm in
run A). The plug cannot be inside a port it has not been pushed into. That is the
uncalibrated `SC_TIP_IN_TCP_POS` — the SFP 58 mm offset — placing the computed
tip somewhere the plug is not. **This is 6c.**

**Fix (`b6d9fd6`, untested):** `run_sc_insertion` refuses a handoff depth above
`SC_MAX_HANDOFF_DEPTH_M` (2 mm) and names the tip transform. This matters because
`RL_INSERT_REPORT_MISS_AS_SUCCESS=1` was reporting the fake seat as success.

### Run C — the collaborator's five ports went live
`sc_port_2..4` are now in the sim. The target port moved from `[-0.325, 0.130]`
to `[-0.285, 0.170]`, detection counts rose, and the TCP pre-filter culled hard
(left 6→2, centre 7→1, right 2→1) leaving **2 candidates** where run B had 28.

Perception failed 0/7 at 11.5 px. Cause: **keypoint corner-order flip.** The left
and centre cameras saw the same physical port (centroids 1.1 px apart) but the
left camera labelled it rotated by two:

```
left kps vs run B, mean per-corner distance
  as-is  18.58 px
  roll 1 12.65 px
  roll 2   2.38 px   <- true correspondence
  roll 3 13.42 px
centre camera, same comparison: 1.46 px (order unchanged)
```

Signature: residuals uniformly high on all four corners (13.3/17.6/16.8/14.5),
and the opening measured 3.35 × 7.14 mm — transposed.

**Fix (`5b5f478`, untested):** `_best_keypoint_correspondence` tries each cyclic
relabelling and keeps whichever reprojects. First camera held as reference (a
global roll is a pure rename), so 4^(n−1) = 16 assignments for three cameras.
53 ms/frame at field scale, 370 ms across 7 frames, against a 45 s budget. The
same freedom normalises `width` to the long axis.

The previous handoff said this needed a retrain. **It does not** — the ambiguity
is resolvable from reprojection alone. But watch the new `SC_KEYPOINT_ROLL` log
line: if the count climbs toward every combination, the detector's corner order
is genuinely degrading and a retrain becomes the honest answer.

---

## 4. Corrections to the previous handoff's "do not re-derive" facts

**§4 bore opening height: 8.10 mm → 7.85 mm.** The channel is not constant along
its depth:
- `cube_collider_box_mid/02/03` run the full 27.432 mm depth at z 4.050–4.650,
  so the ceiling is +4.050 over most of the channel.
- `cube_collider_box.001` is a **lip, not a plate**: full width (x 25.781) but
  only **10.8 mm deep**, centred at y=0, spanning z 3.800–4.650.
- Floor is `.002` at −4.050, full depth.

The plug traverses the lip (15.64 mm of insertion from the y=+13.716 face reaches
y=−1.92; the lip spans y=−5.4…+5.4), so **7.85 mm is the binding height**. 8.10 mm
is real — it is the height clear of the lip — but it is not a tolerance budget.

**Knock-on: vertical clearance is 0.725 mm/side, not 0.85.** Lateral stays
1.205 mm. **Vertical is the binding axis.** Budget grasp repeatability against
0.725 mm.

**§5 "the measured height sat 1 mm above the floor and noise dropped it under":**
the triangulated height is ~3.97 mm, a full 1 mm **below** the old 5.0 mm floor.
The gate was rejecting **deterministically every frame**, which is why the
failure reproduced exactly ×7.

**§6a "8.8 × 6.0 matches nothing":** it is the clear bore (9.71 × 7.85) inset 8.6%
in width and 23.8% in height. Close to the bore, not arbitrary.

**§6a "the label covers ~a quarter of the part":** the 0.25–0.26 fraction is
against the **full visible FOA-005A body, 34.671 mm** including mounting flanges
(`sc_port_visual.glb` node 16, extents 34.671 × 27.432 × 9.271), **not** the
25.78 mm outer face. 0.256 × 34.671 = 8.88 mm ✓. Against 25.78 the same fraction
computes to 6.6 mm and sends you hunting a convention that does not exist. Two
sessions made that substitution.

**§4 "41 mm apart":** stale. Upstream `docs/task_board_description.md` says the
board *"supports up to five SC ports, distributed across two rails"* and that
ports *"slide along their rails"* over **[0, 0.115] m**. There is no fixed pitch.
Adapters cannot overlap, so **25.78 mm is the only safe lower bound**.

**§6e.4 "5 SC ports":** confirmed by upstream docs and by the user's render.

---

## 5. The measurement that settled the label convention

`best_sc_pose.pt` emits the **8.8 × 6.0 mm** rectangle
(`DataCollectorScPoseGT`), not `DataCollectorPoseSC`'s 25.78 × 9.27 — even though
`train_sc.py --data` defaults to `pose_sc`, which is the *second* collector's
output dir. So the shipped weights were not produced by the default invocation.

Method: run the weights over `testing/check_sc_previews`, scale by the **SC duplex
bore pitch, 12.70 mm** — a fixed mechanical dimension, independent of segmenting
anything. Results: 9.00 × 5.94 and 8.73 × 6.01 mm. Diagonals agree to 1–3% while
adjacent sides differ 1.44–1.51, so rectangle not diamond.

**Centred on the duplex centre**, not a bore: centroid sits 1.2–1.9 mm from the
bore-pair midpoint versus 5.3–5.4 mm from either bore. Hence zero bore offset.

**Field-triangulated it measures 7.09–7.40 × 3.95–4.07 mm** — materially smaller
than the label, because the target is detected at only ~0.25–0.45 confidence in
the left and right cameras (a *different* adapter scores 0.91 there) and weak
corners pull the quad toward its own centroid. Residual against the label is
~3.4 mm, so 6a halves the score bias but does not remove it.

Scripts that produced these numbers are in the session scratchpad and are
disposable; the numbers are reproducible from `testing/check_sc_previews` and
`aic_assets/models/SC Port/model.sdf`.

---

## 6. Do this next: 6c, the calibration campaign

**This is the only remaining blocker to seating, and it needs the robot.**

`RL_INSERT_CALIB_DUMP=1` **never worked on the SC path** before `4e367a6`. Three
reasons, all now fixed:
1. Unreachable — `RLInsert._run` dispatches `plug_type == "sc"` at line 946 and
   returns; the `if CALIB_DUMP:` block is at 972 on the SFP-only path.
2. Wrong frames — `CALIB_PLUG_FRAMES` is SFP-only and every name lacks both the
   `cable_N/` prefix and the `_link` suffix the sim publishes.
3. Wrong output names — printed `SFP_TIP_IN_TCP_*`.

`dump_sc_grasp_calibration()` now probes, in order:
```
1. cable_0/sc_tip_link      (from task.cable_name / task.plug_name)
2. sc_tip_link
```
mirroring `DataCollectorScPlugPoseGT._tip_frame_candidates`, which is the naming
known to work. `sc_tip_link` is declared in `aic_assets/models/SC Plug/model.sdf`
and merged into the cable model.

It runs **before perception and before both handoff gates** (`def336e`). That
ordering is load-bearing twice over: the depth gate refuses every run until the
transform is calibrated, and perception can fail independently — either would
otherwise cost the sample. There are tests pinning both.

### Procedure
1. Bake `RL_INSERT_CALIB_DUMP=1` into the image. Bump the manifest version.
2. Run **~10 times, re-grasping each time**. The spread is the measurement, not
   the mean.
3. Expect every run to end at the depth gate. **That refusal is correct** — it is
   the fake seat being caught. Harvest the `[sc-calib]` block.
4. Median of the 10 → new `SC_TIP_IN_TCP_POS` / `_QUAT` defaults in **both**
   copies of `sc_controller.py`.
5. **Spread across the 10 decides whether an SC plug-pose model is needed:**
   if any axis exceeds ~**0.4 mm** (against 0.725 mm vertical clearance, the
   binding axis), build the pipeline; under it, a fixed transform holds.

### Expected healthy log
```
[sc-calib] GROUND-TRUTH frame 'cable_0/sc_tip_link' RESOLVED
[sc-calib]   >>> SOLVED RL_INSERT_SC_TIP_IN_TCP_POS =[...]
[sc-calib]   >>> SOLVED RL_INSERT_SC_TIP_IN_TCP_QUAT=[...]
[sc] SC_KEYPOINT_ROLL n combination(s) needed a corner relabelling
[sc] SC_OPENING convention=gt_label width=~7.1mm height=~4.1mm expected=8.80x6.00mm
[sc] seat frame: preserve_handoff_yaw=True twist_vs_perceived_yaw_deg=~-89.6
[sc] handoff depth is +6.99mm -- ...INSIDE the port... Refusing to seat
```

Predictions worth checking: the solved quat should differ from the SFP default by
~90° about the plug axis (confirming the convention story), and `handoff depth`
should flip from **+7 mm to a small negative** once the calibration is applied.

Failure modes: no `[sc-calib]` block at all → the env var did not reach the image
(manifest version). `"no ground-truth frame resolved"` → the warning prints the
list tried; override with `RL_INSERT_SC_CALIB_PLUG_FRAMES`.

---

## 7. Open, in priority order

1. **6c calibration** (above). Needs the robot. Blocks seating.
2. **6b — the selection gate is on the wrong axis, and now under-margined.**
   `SC_MAX_HANDOFF_SELECT_M = 30 mm` is a **3D** distance, so handoff height is
   mixed into a lateral decision. It was sized against the stale 41 mm spacing.
   With five ports sliding over 115 mm, a shoulder-to-shoulder neighbour at a
   15 mm handoff sits at √(25.78² + 15²) ≈ **29.8 mm — inside the gate**.
   Tightening the number alone would start rejecting the real target on higher
   handoffs. **Fix is to gate on lateral board-XY distance** (insertion is
   straight down board −Z): target ~0–3 mm, any neighbour ≥25.78 mm, independent
   of height. Add `SC_MAX_LATERAL_SELECT_M`, ~10 mm. Documented in the code but
   deliberately not changed — needs its own field run. Run C had only 2
   candidates, so it is not today's blocker, but it is the next real risk.
3. **Watch `SC_KEYPOINT_ROLL`.** If it approaches every combination, revisit the
   retrain question.
4. **Weak outer-camera detection.** The target scores ~0.25–0.45 in left/right
   while a different adapter scores 0.91. This is what shrinks the triangulated
   quad by ~16%/34%. Not blocking, but it is the root of the size residual and
   would be the first thing a port-model retrain should target.

---

## 8. Collector fixes (`b6d9fd6`) — relevant before ANY retrain

`DataCollectorScPoseGT` had three silent label-poisoning paths. All removed:

1. **Seat-frame fallthrough.** `_candidate_sc_frames` fell through from the
   entrance frames to `sc_port_base_link` (the SEAT, 15.64 mm deeper) and
   `sc_port_link`. If the entrance frames were not published, every label in the
   run was off by more than the entire insertion depth.
2. **HSV pseudo-labels.** A blue-blob fallback wrote labels whenever TF was
   missing. Those corners are a blob's min-area box, not a projected rectangle,
   so such frames followed a different convention *inside the same dataset*.
3. **Two-slot loop.** `for slot in (0, 1)` predates `sc_port_2..4`. An image with
   five ports and two labels teaches the detector the other three are
   *background*. Now iterates `SC_SLOTS` (0–4, `AIC_SC_POSE_SLOTS` to override);
   absent slots fail TF and are skipped.

New `no_gt_label` counter, and an error when every camera fails.

`~/aic_perception_data` **does not exist on this machine**. Any retrain means
collecting from scratch.

---

## 9. Things I did NOT change, deliberately

- `SC_MAX_HANDOFF_SELECT_M` — see 6b. Comment updated, value untouched.
- `SC_OPENING_WIDTH_M` / `SC_BORE_WIDTH_M` / `SC_BORE_PITCH_M` — still correct,
  now definition-only since the hypotheses use label conventions.
- `LOCAL_SC_PORT_KPS` **is** now 8.8 × 6.0 (was 22.41 × 8.10, a 2.5× PnP scale
  error), but **nothing calls `solvePnP` for SC** — the only one in the tree is
  SFP's. It is a trap for whoever wires that fallback up, not a live bug.
- `bore_offset_m` is still on the candidate dict and always 0.0; only the
  misleading warning was removed.
