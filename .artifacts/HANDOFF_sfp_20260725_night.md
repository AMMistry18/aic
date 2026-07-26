# Agent Handoff — SFP focus, night 2026-07-25

Written for someone picking up **SFP** work. SC continues in parallel with
another agent; §6 carries the SC state so nothing is lost, but SFP is the point
of this document.

**Companion docs that still hold:** `docs/INSERTION_HANDOFF.md` (SFP V50
baseline, build, validation gate), `docs/HANDOFF.md` (repo baseline),
`docs/WAYS_TO_MAKE_YOLO_POSE_BETTER.md` and
`docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md` (ranked perception interventions —
written for SC, but Items 1–3 apply to SFP unchanged).

**HEAD:** `8254648`. **There are uncommitted changes in the working tree — see
§0 before doing anything else.**

---

## 0. Uncommitted work, commit it first

```
 M aic_model/aic_model/sc_controller.py
 M aic_model/test/test_sc_controller.py
 M docker/aic_model/v50_overlay/aic_model/sc_controller.py
?? deploy/flowstate/Dockerfile.aic_model_service
?? deploy/flowstate/aic_model.manifest.textproto
```

The two modified `sc_controller.py` copies are byte-identical
(`0d45f589c1e00b9308c4657b2747bc97`) and 80 tests pass. The two untracked
`deploy/flowstate/` files are **intentionally untracked** — do not `git add -A`,
stage explicitly.

---

## 1. Conventions that still bite

- **Edit BOTH copies.** `aic_model/aic_model/X.py` and
  `docker/aic_model/v50_overlay/aic_model/X.py` must stay byte-identical. `diff`
  after every edit.
- **Test command** (plain `pytest` fails with `PluginValidationError`):
  ```
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="aic_model:${PYTHONPATH}" \
    .pixi/envs/default/bin/python -m pytest -q \
    aic_model/test/test_sfp_plug_pose.py \
    aic_model/test/test_sfp_plug_pose_trials.py \
    aic_model/test/test_v50_controller.py \
    aic_model/test/test_sc_plug_pose_geometry.py \
    aic_model/test/test_sc_plug_pose_trials.py \
    testing/sfp_v50_validation/tests
  ```
  Never run the whole test directory; several files have collection errors that
  pre-date all of this.
- **No runtime env knobs in Flowstate.** `RL_INSERT_*` only takes effect if baked
  into the image. Prefer changing defaults in source.

---

## 2. The finding that matters most for SFP: the sim's real TF inventory

A full TF enumeration was run in the field on 2026-07-25 (three runs). **The sim
publishes 63 frames.** This has never been recorded before and it invalidates a
standing assumption on the SFP path.

### 2a. `RL_INSERT_CALIB_PLUG_FRAMES` defaults are all fictional

`RLInsert.py:253` probes:

```
sfp_tip, sfp_plug, plug, cable_0, sfp, gripper/sfp_tip, gripper/plug, tool/tip
```

**None of these exist.** The SFP grasp-calibration dump (`_dump_grasp_calibration`,
`RL_INSERT_CALIB_DUMP=1`) therefore cannot ever have resolved ground truth. If
you were planning to recalibrate `SFP_TIP_IN_TCP_*` that way, it will silently
find nothing.

### 2b. `selected_sfp/sfp_tip_link` exists — and is almost certainly a trap

It resolves, at **235 mm from the TCP**. Its SC twin was investigated in depth
and *proven* to be a static frame:

| | run A | run B | moved |
|---|---|---|---|
| TCP | `[-0.3252, 0.1762, 0.0811]` | `[-0.3242, 0.1371, 0.0809]` | **39.1 mm** |
| `selected_sc/sc_tip_link` | `[-0.5209, 0.3573, 0.0167]` | `[-0.5208, 0.3572, 0.0174]` | **0.7 mm** |

It is a genuine SC plug — `selected_sc/sc_tip_link` and
`selected_sc/sc_plug_link` are separated by 11.65 mm, exactly the `sc_tip_joint`
offset in `aic_assets/models/SC Plug/model.sdf`. It is simply **not the plug in
the gripper**: a second instance, or a spawn-time pose, sitting at table height
~30 cm away and never moving.

`selected_sfp/*` sits in the same namespace at a similarly implausible distance.
**Verify before trusting it.** The one-line test: sample it across two runs with
different TCP poses; if it doesn't move, it's dead.

This cost three field runs to establish on the SC side. Don't repeat it.

### 2c. Frames that ARE useful

| frame | what it is | use |
|---|---|---|
| `arm_link_tip`, `flange_to_tip_joint` | identity to the TCP (< 0.2 mm) | confirms TCP convention |
| `sc_port/sc_port_base_link_entrance` | **ground-truth port entrance pose** | validate perception against truth |
| `sc_port/sc_port_link`, `_sensor`, `_base_link` | port body / seat | seat-depth reference |
| `selected_sfp/sfp_tip_link` | static, ~235 mm off | **do not calibrate against** |
| `selected_sc/sc_tip_link`, `sc_plug_link` | static, ~275–302 mm off | **do not calibrate against** |

`sc_port/sc_port_base_link_entrance` is the interesting one. It is a truth
oracle for port perception — usable offline to measure the perception error
directly instead of inferring it from reprojection residuals. There is very
likely an `sfp_port/...` equivalent; the enumeration only printed frames matching
sc/tip/plug/cable, so the SFP port frames were filtered out before printing.
**Re-run the enumeration with the filter widened** and you get the SFP truth
frames for free.

---

## 3. `SFP_TIP_IN_TCP_*` is hand-tuned, and that should be checked

`rl_insert_contract.py:29-38`:

```python
SFP_TIP_IN_TCP_POS  = [-0.0017771781, -0.0188744563, 0.0547221980]
SFP_TIP_IN_TCP_QUAT = [0.9840750466, 0.1756266707, -0.0115567892, -0.0248599222]
```

The comment above the quaternion records a manual `+1.16 deg` nudge, a finding
that it "made the tilt WORSE", and a reversal to the same magnitude in the
opposite direction. That is hand-tuning, not a solved transform — and per §2a its
calibration path has never worked.

Two things worth establishing early:

1. **Is it still on a live path?** `sfp_plug_pose.py` explicitly has "no
   fixed-grasp/bias fallback", and V50 uses the estimator, so this constant may be
   legacy. If it is dead, delete it rather than leave a hand-tuned trap. If it is
   live anywhere, that path is running on a fudge factor.
2. **Competition rules forbid hardcoding.** The user raised this directly. A
   hand-tuned tool transform is the most exposed thing in the tree on that front.
   The SFP plug-pose model is the defensible answer and it already exists —
   `best_sfp_plug_pose.pt`, trained, wired, fail-closed.

---

## 4. Tooling built this session that generalises to SFP

All in `aic_model/aic_model/sc_controller.py`, all tested. Porting cost is low.

**Geometric frame identification** (`_probe_tf_frames_for_tip`,
`_tf_frame_names`, `parse_tf_frame_names`). Probes every TF frame, sorts by
distance from the TCP, and identifies the held plug **by where it is, not what
it is called**. Name-guessing failed three times; geometry did not. Handles both
`all_frames_as_yaml()` and `all_frames_as_string()`.

**Plausibility band** (`SC_HELD_PLUG_MIN_M` = 20 mm, `SC_HELD_PLUG_MAX_M` =
120 mm). A resolved frame outside the band is *rejected*, not reported. This
exists because a frame resolving cleanly is not evidence it is the right frame —
`selected_sc/sc_tip_link` resolved perfectly and printed `>>> SOLVED` with a
number 30 cm wrong. Pinned by
`test_calibration_refuses_a_frame_too_far_away_to_be_the_held_plug`.

**`_env_vector` bracket parsing.** The calibration dump prints transforms via
`.tolist()` → `[0.1, 0.2, 0.3]`, but the parser could not read brackets:
`float("[0.1")` raised and it **silently returned the uncalibrated default**.
Copy-pasting from the log was indistinguishable from never setting the variable.
Now accepts both forms and complains loudly on a bad value. **Check whether the
SFP path has the same pattern** — `v50_controller.py` has no `_env_vector`, but
any env-parsed vector with a silent `except: return default` is the same bug.

---

## 5. Where to push SFP quality

`docs/WAYS_TO_MAKE_YOLO_POSE_BETTER.md` and
`docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md` already rank the interventions. Written
for SC, but the mechanics are shared:

- **Item 1** — rigid-shape PnP instead of triangulate-then-average.
- **Item 2** — crop-refine two-stage inference (called out as the biggest win).
- **Item 3** — angle-diverse data + rotation augmentation.

One field observation from this session that supports Item 2/3: **outer-camera
detection is weak.** On the SC port the target scored 0.25–0.41 in left/right
while center scored 0.93, and a *different* adapter scored 0.91 in the same
outer frames. Weak corners pull the triangulated quad toward its own centroid,
shrinking it ~16%/34%. That is a detector problem, not a geometry problem, and it
is the root of the size residual. Whether SFP shows the same asymmetry is
unmeasured — worth checking first, since it decides whether Item 2 or Item 3 is
the higher-value fix.

Validation contract lives in `testing/sfp_v50_validation/README.md`: 300 unique
trials, 300 correct events within 45 s, zero wrong-port / off-limit-contact /
force-penalty. Evidence under `results/` (gitignored).

---

## 6. SC state, for the parallel track

Continuing with a separate agent. Summary so you don't collide:

**Working:** keypoint corner-order resolution (7/7 perception at ~4.0 px, was
0/7 at 11.5 px); the impossible-depth gate, which correctly refuses every run at
+6.8 to +7.0 mm.

**Blocked on:** the TCP→tip transform. SC currently borrows
`SFP_TIP_IN_TCP_POS`, which places the tip 57.8 mm below the TCP — an SFP plug's
length on an SC plug. That constant is what produces the phantom "+7 mm inside
the port before any motion" and the resulting fake seat.

**Next SC run** is the decisive one: the geometric frame probe now always runs
and can no longer be short-circuited. Either a frame appears 20–120 mm from the
TCP — that is the held plug and the transform is solved — or nothing does, which
proves the sim never publishes the grasped plug's pose and the SC plug-pose model
(`d860470`, already built by the parallel agent) is the only path.

**Deliberate SC behaviour change:** port selection is nearest-to-tip, and the
seat check no longer hard-fails when the insertion event names a different port
(`SC_STRICT_PORT_EVENT`, default off). Steering to the *requested* port is the
Flowstate macro's job. Previously a physically correct insertion into the port
the macro parked over was reported as `HARD_FAILURE`, which made seating
impossible to demonstrate. It still logs both port names loudly, because scoring
credits only the requested port. **Set `RL_INSERT_SC_STRICT_PORT_EVENT=1` before
submission.**

**Do not "fix" an alignment timeout** by raising `align_timeout_wall_s` or
`align_max_rotation_step_rad`. The ~90° rotation error is a frame-convention
offset handled by `seat_frame()`; letting the robot complete that turn drives a
20 mm plug at a 7.85 mm opening.

---

## 7. Open, in priority order

1. **Re-run the TF enumeration with the name filter widened** (§2c). Cheapest
   possible win: it yields the SFP port truth frames and confirms or kills
   `selected_sfp/sfp_tip_link` in one run.
2. **Establish whether `SFP_TIP_IN_TCP_*` is on a live path** (§3). If yes, it is
   both a hardcoding exposure and a hand-tuned fudge.
3. **Measure SFP outer-camera detection asymmetry** (§5). Decides Item 2 vs
   Item 3 ordering.
4. **Port the geometric frame prober and plausibility band** to the SFP
   calibration path (§4), or delete that path if the model supersedes it.
5. **Check the SFP path for silent env-parse fallbacks** (§4, last paragraph).

---

## 8. Things deliberately NOT changed

- `SFP_TIP_IN_TCP_*` — flagged, not touched. Changing it without knowing whether
  it is live would be a blind edit to the qualified V50 baseline.
- `RL_INSERT_CALIB_PLUG_FRAMES` defaults — left fictional. The right fix is the
  geometric prober, not a better guess list, and that belongs to whoever owns the
  SFP calibration decision in §7.2.
- `SC_MAX_HANDOFF_SELECT_M` — the previous handoff's §6b item (gate on lateral
  rather than 3D distance) is now **void**. It was about choosing the *right*
  port among neighbours; port choice is macro territory. Nearest-to-tip is the
  intended behaviour.
