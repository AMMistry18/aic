# Agent Handoff — SC insertion, 2026-07-26

**Supersedes `.artifacts/HANDOFF_sc_insertion_20260725_eve.md`** for everything
about the tip transform, port selection and frame naming. That document's §1
(conventions), §4 (geometry corrections) and §5 (label convention) still hold and
are not repeated here. Its §6c procedure and §7.2 selection-gate item are
**superseded** — see §5 and §8 below.

**HEAD:** `f362b7d`, tree clean, `main == origin/main`. SC work landed as
`668fb0e`; `8757b7e` extended the frame probe. 89 tests pass.

---

## 1. The one blocker, stated precisely

SC does not know where the plug tip is relative to the gripper.

`SC_TIP_IN_TCP_POS` currently falls back to `SFP_TIP_IN_TCP_POS`
(`rl_insert_contract.py:29`), which places the tip **57.8 mm** below the TCP —
measured live: TCP `z=0.081111`, assumed tip `z=0.023324`. That is an *SFP*
plug's length applied to an *SC* plug.

Everything downstream inherits the error. The handoff check computes the tip as
already **+6.8 to +7.0 mm inside the port before any motion**, which is
physically impossible, so `_seat` would skip its entire approach and wait for an
insertion event that cannot arrive. Measured across four runs:

| run | epoch | handoff depth |
|---|---|---|
| A | 1785005084 | +7.04 mm |
| B | 1785013590 | +6.99 mm |
| C | 1785018404 | +6.92 mm |
| D | 1785021488 | +6.82 mm |
| E | 1785022840 | +7.01 mm |

The tight spread is itself the evidence: this is a constant transform error, not
noise. `b6d9fd6`'s depth gate now refuses these runs. **That refusal is correct
behaviour** — it is the fake seat being caught, and it is why
`RL_INSERT_REPORT_MISS_AS_SUCCESS=1` no longer scores a miss as a win.

---

## 2. The frame hunt — what is settled, so nobody repeats it

Three field runs went into this. The conclusions are firm.

### 2a. Name-guessing does not work

Every guessed name has missed:

- `cable_0/sc_tip_link`, `sc_tip_link` (from `DataCollectorScPlugPoseGT`) — **do
  not exist.** The previous handoff called this "the naming known to work"; that
  was inferred from the collector, and the SC collector has never actually been
  run, so it was never verified.
- SFP's list (`RLInsert.py:253`: `sfp_tip, sfp_plug, plug, cable_0, sfp,
  gripper/sfp_tip, gripper/plug, tool/tip`) — **none exist either.**

The sim publishes **63 frames**. The naming convention is `<model>/<link>`, and
the models are not called what anyone assumed.

### 2b. `selected_sc/*` resolves, and is a decoy

`selected_sc/sc_tip_link` and `selected_sc/sc_plug_link` both resolve cleanly.
They are a **genuine SC plug** — their separation is 11.65 mm, exactly the
`sc_tip_joint` offset in `aic_assets/models/SC Plug/model.sdf`.

They are also **static**:

| | run D | run E | moved |
|---|---|---|---|
| TCP | `[-0.3252, 0.1762, 0.0811]` | `[-0.3242, 0.1371, 0.0809]` | **39.1 mm** |
| `selected_sc/sc_tip_link` | `[-0.5209, 0.3573, 0.0167]` | `[-0.5208, 0.3572, 0.0174]` | **0.7 mm** |

~30 cm away, at table height, not tracking the gripper. A second instance or a
spawn-time pose. `selected_sfp/sfp_tip_link` sits in the same namespace at a
similar distance and should be assumed to be the same trap until measured.

**This cost a full field run**, because `selected_sc/sc_tip_link` was briefly
added to the candidate list and resolving it short-circuited the geometric probe,
printing `>>> SOLVED` with a number 30 cm wrong. Hence §3's plausibility band.

### 2c. Frames that are real and useful

| frame | what it is |
|---|---|
| `arm_link_tip`, `flange_to_tip_joint` | identity to the TCP (< 0.2 mm) — confirms the TCP convention |
| `sc_port/sc_port_base_link_entrance` | **ground-truth port entrance pose** |
| `sc_port/sc_port_link`, `_sensor`, `_base_link` | port body / seat |
| `selected_sc/sc_tip_link`, `sc_plug_link` | static decoys, ~275–302 mm off |

`sc_port/sc_port_base_link_entrance` is worth exploiting: it is a truth oracle
for port perception, letting you measure perception error directly rather than
inferring it from reprojection residuals. Use it offline for validation only —
reading it at runtime would be exactly the hardcoding the rules forbid.

---

## 3. What changed in the code (`668fb0e`, `8757b7e`)

All in `sc_controller.py` (both copies), all tested.

**Geometric frame identification.** `_probe_tf_frames_for_tip()` probes every TF
frame, sorts by distance from the TCP, and identifies the held plug **by where it
is, not what it is called**. Name matching failed three times; geometry cannot.
`parse_tf_frame_names()` handles both `all_frames_as_yaml()` and
`all_frames_as_string()`. `8757b7e` removed a nearest-20 truncation so the full
inventory prints.

**Plausibility band.** `SC_HELD_PLUG_MIN_M` = 20 mm, `SC_HELD_PLUG_MAX_M` =
120 mm. A frame outside the band is **rejected, not reported**, even in the
explicit-candidate loop. A frame resolving is not evidence it is the right frame.
Pinned by `test_calibration_refuses_a_frame_too_far_away_to_be_the_held_plug`.

**`_env_vector` bracket parsing.** The calib dump prints transforms via
`.tolist()` → `[0.1, 0.2, 0.3]`, but the parser could not read brackets:
`float("[0.1")` raised and it **silently returned the uncalibrated SFP default** —
i.e. copy-pasting the solved value out of the log was indistinguishable from
never setting the variable. Now accepts bracketed and bare forms and complains on
stderr for a bad value instead of vanishing.

**Wrong-port seat events no longer hard-fail.** `SC_STRICT_PORT_EVENT`, default
off. See §8.

---

## 4. Do this next — it no longer costs an insertion run

`8757b7e` added **`scripts/enumerate_tf_frames.py`**, a standalone rclpy TF
harvester. It attaches to a live sim, needs no insertion, and does the two-sample
motion check automatically — which is exactly the measurement that unmasks static
decoys.

```bash
.pixi/envs/default/bin/python scripts/enumerate_tf_frames.py \
  --settle 5 --gap 3 --json /tmp/tf_inventory.json
```

Defaults: `--base-frame base_link`, `--tcp-frame arm_link_tip`. It prints the
full inventory sorted by TCP distance, flags frames that did not move between
samples, and sections out port/sfp/sc frames and anything in the 20–120 mm
held-plug band.

**Run it with the robot holding the plug.** That is the whole question: is there
a frame 20–120 mm from the TCP that moves with the gripper?

---

## 5. Decision tree on the result

**A frame appears in the 20–120 mm band and moves with the TCP.**
That is the held plug. Set `RL_INSERT_SC_CALIB_PLUG_FRAMES` to it, bake
`RL_INSERT_CALIB_DUMP=1`, and harvest. Then:

1. Apply the **first** solved sample immediately (not the median) plus
   `RL_INSERT_SC_TIP_CALIBRATED=1`. This opens the depth gate.
2. From then on every run does double duty — a real insertion attempt *and*
   another calibration sample. The 10-grasp spread comes free from runs you were
   doing anyway. **The spread is the measurement, not the mean.**
3. Spread > ~0.4 mm on any axis (against 0.725 mm vertical clearance, the binding
   axis) → the grasp is not repeatable and the plug-pose model is mandatory.
   Under it, a fixed transform would hold — but see the rules note below.

**Nothing appears in the band.**
The sim does not publish a pose that tracks the grasped plug. Ground-truth
calibration is unavailable, and the SC plug-pose model is the only path. That
work already exists — `d860470` added the estimator, trainer and validation, on
top of `sc_plug_pose_geometry.py`, `generate_sc_plug_pose_trials.py` and
`DataCollectorScPlugPoseGT.py`. What is missing is trained weights;
`~/aic_perception_data` does not exist, so collection starts from zero.

**Either way, the shipped answer is the model, not a constant.** The competition
forbids hardcoding, and `sfp_plug_pose.py` already establishes the house pattern:
"There is deliberately no fixed-grasp/bias fallback." A solved constant is
legitimate as a **debug scaffold and validation oracle** — it is what lets you
exercise alignment and seating while the model trains — but it must not be the
default at submission.

---

## 6. What is working

- **Keypoint corner-order resolution** (`5b5f478`). Perception recovered from
  0/7 at 11.5 px to **7/7 at ~4.0 px**. `SC_KEYPOINT_ROLL` fires 3–20 times per
  frame, so the relabelling is doing real work, not passing by luck.
- **The depth gate** (`b6d9fd6`). Refusing correctly, five runs running.
- **Alignment** (`b92c502`, field-validated). The ~90° rotation error is a
  frame-convention offset, not a perception error; `seat_frame()` handles it.
  Run B converged in ~6.5 s to lateral 0.95 mm / rot 0.09°.

---

## 7. Known risk: weak outer-camera detection

Not currently blocking, but it is the next thing to break. On the SC port the
target scores **0.25–0.41** in left/right while center scores **0.93** — and a
*different* adapter scores 0.91 in those same outer frames. Weak corners pull the
triangulated quad toward its own centroid, shrinking it ~16%/34%; the opening
measures 6.2–7.5 × 3.4–4.0 mm against an 8.80 × 6.00 mm label.

This has already caused a hard failure once (run C-equivalent at the y≈0.170
port): reproj 5.06–5.31 px against a 5.0 px select gate rejected 6/7 frames and
perception failed 1/7.

**Do not fix this by loosening the 5.0 px gate.** The honest fix is to stop
letting a 0.26-confidence detection contribute to the pose fit at all — two views
are already legal (`min_views=2`) and center+right are usually both strong. That
is a contribution threshold, not a quality-bar change. `scripts/measure_sfp_camera_asymmetry.py`
(from `8757b7e`) measures per-camera detection stats offline and works for the
port model too.

---

## 8. Deliberate behaviour changes, and what to revert before submission

**Port selection is nearest-to-tip, by design.** The controller inserts into
whatever port the macro parked it over; steering to the *requested* port is the
Flowstate macro's job and is owned elsewhere.

The seat check used to contradict this — it demanded the task's named port and
returned `HARD_FAILURE` otherwise, so a physically correct insertion into the
nearest port was indistinguishable from never seating. It now counts as seated
and logs both port names loudly, because scoring credits only the requested port.

> **Set `RL_INSERT_SC_STRICT_PORT_EVENT=1` before submission.**

Consequently the previous handoff's **§7.2 selection-gate item is void**.
Gating on lateral rather than 3D distance was about choosing the right port among
neighbours; that is no longer this controller's problem.

**Never "fix" an alignment timeout** by raising `align_timeout_wall_s` or
`align_max_rotation_step_rad`. That lets the robot complete a 90° turn it should
not make and drive a 20 mm plug at a 7.85 mm opening.

---

## 9. Open, in priority order

1. **Run `scripts/enumerate_tf_frames.py` while holding the plug** (§4). Cheap,
   no insertion needed, and it decides §5 outright.
2. **Follow the §5 branch it selects.** Either harvest the transform, or commit
   fully to the plug-pose model.
3. **SC plug-pose training data.** Zero exists. Smoke-test
   `DataCollectorScPlugPoseGT` on a handful of trials *before* launching a long
   collection — it has never been run for SC and everything downstream assumes it
   works.
4. **Outer-camera contribution threshold** (§7).
5. **Validate perception against `sc_port/sc_port_base_link_entrance`** (§2c).
   Offline only.

---

## 10. Conventions (unchanged, still bite)

- **Edit BOTH copies.** `aic_model/aic_model/sc_controller.py` and
  `docker/aic_model/v50_overlay/aic_model/sc_controller.py` byte-identical.
  `diff` after every edit.
- **Tests:**
  ```
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .pixi/envs/default/bin/python -m pytest \
    aic_model/test/test_sc_controller.py aic_model/test/test_v50_controller.py -q
  ```
  89 pass. Never run the whole test directory.
- **Never `git add -A`.** `deploy/flowstate/Dockerfile.aic_model_service` and
  `aic_model.manifest.textproto` are intentionally untracked. Stage explicitly.
- **No runtime env knobs in Flowstate.** `RL_INSERT_SC_*` only takes effect baked
  into the image. Bump the manifest or the deploy silently reuses the old image.
