# SC plug-pose: handoff to the TACC agent

**Written 2026-07-25.** Everything that can be built without a working GPU is
built, tested and pushed. What remains is data collection, training, and an
honest accuracy measurement — all of which need a GPU that can render.

> **This work is on `main`** (commit `d860470`), so a plain clone has it:
> ```bash
> git clone https://github.com/AMMistry18/aic.git
> ```
> It is also on the branch `sc-plug-pose` at the same commit; the two are
> identical, use either.
>
> **What is deliberately NOT here:** the in-progress `sc_controller.py` seating
> work. That was being edited in parallel and was excluded from the commit, so
> `sc_controller.py` on `main` is the pre-existing version that still uses the
> hardcoded `SC_TIP_IN_TCP_*` constant. Nothing in this handoff depends on that
> work, and §7 explains why the wiring is left unapplied.

---

## 1. The job in one paragraph

The SC insertion path computes the plug tip from a hardcoded TCP→tip constant
borrowed from the SFP plug (`SC_TIP_IN_TCP_POS/_QUAT` default to
`SFP_TIP_IN_TCP_*`). It is wrong — it causes a +7 mm phantom depth reading that
blocks seating — and hardcoding it is disallowed by the competition rules. The
fix is to estimate `sc_tip_link` from the wrist cameras on every run, exactly
the way SFP already does. Train an SC plug-pose YOLO model, fuse its keypoints
across cameras, and hand the controller a measured tip pose or nothing at all.

**Accuracy target: ~0.4 mm.** The SC port opening leaves 0.725 mm of vertical
clearance per side (vertical is the binding axis) and 1.205 mm lateral.

---

## 2. Why this is being handed over

The dev workstation (`jarvis`) cannot render. Its loaded NVIDIA kernel module
is 595.71.05 (from the removed `libnvidia-compute-595-server`) while installed
userspace is 595.84 (`nvidia-driver-595-open`), so NVIDIA EGL enumerates **zero
devices**. CUDA still works — torch runs on the RTX 5090 — but Gazebo falls back
to llvmpipe at a real-time factor of ~0.0003, which puts a 4-hour collection at
somewhere between 2 weeks and 1.5 years. That machine needs a reboot; TACC does
not have the problem.

Nothing about the code is TACC-specific. If `jarvis` gets rebooted,
`scripts/sc_plug_pose_collect_local.sh` runs the same pipeline locally.

---

## 3. What is already done (do not rebuild)

| File | Status |
|---|---|
| `aic_model/aic_model/sc_plug_pose_geometry.py` | pre-existing; 8 SC keypoints in `sc_tip_link`, dataset-yaml writer |
| `generate_sc_plug_pose_trials.py` | pre-existing; trial generator, tested |
| `aic_example_policies/.../ros/DataCollectorScPlugPoseGT.py` | pre-existing; auto-labelled collection from sim TF |
| `aic_model/aic_model/sc_plug_pose.py` | **new** — `ScPlugPoseEstimator`, fail-closed |
| `train_sc_plug_pose.py` | **new** — SC trainer |
| `validate_sc_plug_pose.py` | **new** — accuracy vs sim TF GT, plus a synthetic error budget |
| `aic_model/test/test_sc_plug_pose.py` | **new** — 15 tests |
| `scripts/sc_plug_pose_collect_local.sh` | **new** — local (non-TACC) collection runner |
| `docs/SC_PLUG_POSE_WIRING_PATCH.md` | **new** — the controller change, written up, NOT applied |
| `.tacc/sc_plug_pose_datagen.slurm` | pre-existing, unchanged |
| `.tacc/train_sc_plug_pose.slurm` | **updated** — now calls `train_sc_plug_pose.py` and runs validation |

`ScPlugPoseEstimator` is a *parameterisation* of `SfpPlugPoseEstimator`, not a
fork. `SfpPlugPoseEstimator.__init__` gained an optional `local_keypoints_m`
(default = SFP keypoints, so the SFP path is byte-for-byte unaffected in
behaviour); the SC class passes the SC keypoints. A test pins the SFP default.

**Test status:** 97 passed when measured on the dev workstation.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .pixi/envs/default/bin/python -m pytest \
  aic_model/test/test_sc_plug_pose.py aic_model/test/test_sc_controller.py \
  aic_model/test/test_v50_controller.py aic_model/test/test_sc_plug_pose_trials.py -q
```

A **fresh clone gives 92**, and that is the expected number. The 97 was measured
in a working tree that also held the colleague's in-progress `sc_controller.py`
changes, which are deliberately not committed. Of the 92, 15
(`test_sc_plug_pose.py`) and 4 (`test_sc_plug_pose_trials.py`) are this work and
must pass; the `test_sc_controller.py` / `test_v50_controller.py` counts float
with whatever state `sc_controller.py` is in. What matters is that nothing
**fails**.

Never run the whole test dir — five other files have pre-existing collection
errors (the installed `aic_model` in `.pixi` predates the plug-pose modules).

---

## 4. Your job, in order

### Step 0 — stage the repo and prerequisites on TACC

Transport is **GitHub, not rsync** (the repo laptop is on Tailscale and TACC
needs UT VPN; the two can't be held at once).

```bash
cd $WORK
git clone https://github.com/AMMistry18/aic.git aic-sc
cd aic-sc
```

Then satisfy what the slurm scripts require. **Both are hard prerequisites and
neither is in git:**

1. **A pinned evaluation SIF** at `${job_root}/containers/aic_eval_pinned.sif`.
   `sc_plug_pose_datagen.slurm` deliberately refuses to pull a mutable
   `:latest`. Build it once with
   `apptainer build aic_eval_pinned.sif docker://ghcr.io/intrinsic-dev/aic/aic_eval:latest`
   on a node with outbound internet.
2. **A built pixi env** at `${repo_root}/.pixi/envs/default/bin/python`. Use
   `PIXI_HOME=$WORK/pixi` and cache-dir `$WORK/.pixi-cache` — the `$HOME` quota
   is tiny. The monolith env (robostack-kilted ROS + Gazebo) is what the repo
   expects.

Both slurm scripts hardcode
`/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260719` as
`job_root` and `...-src-20260719` as `repo_root`. **Update those paths** to
wherever you stage, or recreate the dated directories.

Put the repo in `$WORK` — `$SCRATCH` purges after ~10 days.

### Step 1 — SMOKE TEST FIRST. This is not optional.

`DataCollectorScPlugPoseGT` **has never been run for SC**. Not once. No SC data
exists anywhere. Everything downstream assumes it works. Run a handful of
trials and confirm it writes non-empty labels before committing hours of
allocation to it.

```bash
SC_MODE=smoke SC_TRIAL_COUNT=10 sbatch .tacc/sc_plug_pose_datagen.slurm
```

Then check, and do not proceed until all of these hold:

```bash
# 10 trials x 3 cameras x 3 frames = up to 90 images
find $job_root/smoke_dataset/images -name '*.png' | wc -l   # > 0, ideally ~90
find $job_root/smoke_dataset/labels -name '*.txt' | wc -l   # equal to image count
# every label must be exactly 29 tokens (1 class + 4 bbox + 8*3 keypoints)
find $job_root/smoke_dataset/labels -name '*.txt' -exec awk 'NF!=29{print FILENAME}' {} +
```

The smoke run sets `AIC_SC_PLUG_POSE_SAVE_DEBUG=1`, which writes keypoint
overlays to `smoke_dataset/debug/`. **Look at two or three of them.** The eight
dots must sit on the blue duplex housing in a consistent order. If they are
scattered, on the wrong body, or the plug isn't in frame, stop — the collector
or the frame resolution is wrong and no amount of training fixes it.

Likely failure modes, in rough order of probability:
- **TF frame not found.** The collector tries `<cable>/<plug>_link`, then
  `<cable>/sc_tip_link`, `cable_0/sc_tip_link`, `sc_tip_link`
  (`_tip_frame_candidates`). If none resolve it silently saves nothing and you
  get zero images with no error. Check the collector log for trials that report
  `saved=0`.
- **Task filter rejects everything.** `insert_cable` skips any task that is not
  exactly `plug_type=sc`, `plug_name=sc_tip`, `port_name=sc_port_base`,
  `target_module_name` in `{sc_port_0, sc_port_1}` — it logs "Skipping
  noncanonical SC plug task". `generate_sc_plug_pose_trials.py` should produce
  only canonical trials, but verify.
- **Fewer than 6 keypoints visible** (`MIN_VISIBLE_KEYPOINTS`) → frame dropped.
  A few dropped frames is normal; all frames dropped is a geometry problem.

### Step 2 — full collection

```bash
SC_MODE=full SC_TRIAL_COUNT=450 SC_SEED=20260725 sbatch .tacc/sc_plug_pose_datagen.slurm
```

450 trials × 3 cameras × 3 frames = **4,050 images**, roughly 6 GiB. The script
carries a storage watchdog (15 GiB cap), a rosbag janitor, and a simulator
health watchdog that kills the collector if Gazebo dies — so a crashed sim
fails fast instead of hanging the allocation.

Split is a stable 80/10/10 **by trial index**, so images from one trial never
straddle train and test. If you collect in batches, pass `SC_TRIAL_START` so
indices keep counting (`451`, `901`, …) — restarting at 1 will collide
filenames and corrupt the split.

Set `SC_TRAIN_AFTER_COLLECTION=1` to chain training onto the same allocation
once every integrity check passes (TACC caps queued jobs at two).

### Step 3 — train

```bash
sbatch .tacc/train_sc_plug_pose.slurm
```

Defaults are `yolo11s-pose.pt`, 120 epochs, `imgsz 960`, batch 8, patience 30 —
identical to the SFP run so the two stay comparable. The dataset gate refuses
to start if any split has under 300 images or the total is under 3,600.
Training runs validation automatically at the end (see next step).

### Step 4 — measure accuracy honestly

This is the deliverable that matters, more than the checkpoint.

```bash
.pixi/envs/default/bin/python validate_sc_plug_pose.py \
  --mode dataset --weights <best_sc_plug_pose.pt> \
  --data <dataset>/aic_sc_plug_pose.yaml --split test --device 0
```

Read these three numbers and report them plainly:

| Field | Meaning | Target |
|---|---|---|
| `position_error_mm.median` | tip error vs sim TF ground truth | **≤ 0.4 mm** |
| `lateral_error_mm.p95` | error perpendicular to the insertion axis | **≤ 0.725 mm** |
| `group_miss_rate` | fraction of frames yielding no pose at all | ≤ 0.01 |

`lateral` is the one that has to fit the port. Error is decomposed in the
plug's own frame (local +Z is the insertion axis), so it is meaningful
regardless of wrist orientation.

**If it misses the target, say so plainly — that changes the whole strategy.**
Do not paper over it. See §6 for what to try next.

---

## 5. What the error budget says before you start

I ran a synthetic study (`validate_sc_plug_pose.py --mode synthetic`) built
from the real rig: three Basler cameras, f = 1236.6 px, 1152×1024, baselines
116 mm / 201 mm, plug at 280 mm (derived through
`cam_mount → ATI 0.0265+0.0245 → Hand-E 0.172 → tcp → tip`).

| keypoint σ (px) | tip median (mm) | lateral p95 (mm) |
|---|---|---|
| 0.5 | 0.087 | 0.154 |
| 1.0 | 0.181 | 0.328 |
| **2.0** | **0.343** | **0.581** |
| 3.0 | 0.527 | 0.965 |

**So keypoints need σ ≲ 2 px to hit 0.4 mm.** The SFP model's held-out gate is
p95 keypoint error ≤ 4 px (σ ≈ 1.6 px), so a comparable SC model should make
it. `validate_sc_plug_pose.py` reports `keypoint_error_px` — check it against
this table to see whether a miss is a *keypoint* problem or a *geometry*
problem.

**The caveat that matters.** This assumes zero-mean independent Gaussian
keypoint noise. Triangulation averages that out; it does **not** average out
*bias*. `docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md` documents exactly that
failure mode for YOLO regression heads ("consistent directional keypoint
shift", no sub-pixel decoding). A systematic shift feeds straight through to
the tip. Treat σ ≤ 2 px as necessary, not sufficient — the real number is
Step 4.

I also verified the label-formatting path offline on 12,000 synthetic frames:
100% produced valid 29-token labels with all 8 keypoints visible. That derisks
the projection/visibility/bbox math but **not** TF resolution, image delivery,
or the task filter — which is why Step 1 exists.

---

## 6. If the 0.4 mm target is missed

In rough order of expected gain per unit effort. Most of this is already
argued out in `docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md` and
`docs/WAYS_TO_MAKE_YOLO_POSE_BETTER.md`.

1. **Check `imgsz` is 960 everywhere at inference.** Ultralytics defaults to
   640. This was leak #1 for the SC *port* model. `ScPlugPoseEstimator`
   defaults to 960; make sure nothing overrides it.
2. **More data / more angle diversity.** The collector's `sample_viewpoints`
   is currently modest. Zero angle diversity was leak #4 for the port model.
3. **Crop-refine two-stage** — coarse detect, padded crop from the native-res
   frame, re-run pose on the crop, remap to full-frame coords before fusing.
   Historically the largest single win (~5–10× effective resolution). Get the
   `cx,cy` offset and `fx,fy` scaling right or it biases the pose.
4. **Joint multi-camera bundle adjustment** — one 6-DoF pose minimising summed
   reprojection error across all three cameras, replacing per-keypoint
   triangulate-then-fit.
5. **A larger backbone** (`yolo11m-pose`) — cheap to try, uncertain gain.

If the median lands somewhere between 0.4 and 0.725 mm, that is worth
reporting as a partial result: it may still be usable with a compliant seating
strategy, since lateral clearance is 1.205 mm and only vertical is at 0.725 mm.

---

## 7. Wiring it into the controller — DO NOT DO THIS YET

`docs/SC_PLUG_POSE_WIRING_PATCH.md` contains the full proposed change:
where the seam is (`ScSeatAction._tip_pose()`), how to build the estimator at
configure time, how to reuse `_plug_views_from_observation()` from
`v50_controller.py`, and — most importantly — the contract change, because
`_tip_pose()` gains the ability to return `None` and all four call sites must
handle that by **stopping**, never by falling back to the constant.

It is deliberately not applied. `sc_controller.py` is being edited in parallel.
Apply it after that work lands and after Step 4 gives a number worth trusting.

---

## 8. House rules (these bit me; they will bite you)

- **Never `git add -A`.** `deploy/flowstate/Dockerfile.aic_model_service` and
  `deploy/flowstate/aic_model.manifest.textproto` are untracked on purpose.
  Stage explicitly.
- **Do not modify `aic_model/aic_model/sc_controller.py`.** Parallel work.
- **Dual copies must stay byte-identical.** `aic_model/aic_model/X` and
  `docker/aic_model/v50_overlay/aic_model/X`. Currently mirrored:
  `sfp_plug_pose.py`, `sfp_plug_pose_geometry.py`, `sc_plug_pose.py`,
  `sc_plug_pose_geometry.py`, `sc_controller.py`, `rl_insert_contract.py`.
  `diff` them after every edit.
- **Force-add meshes.** New scene meshes under `aic_utils/aic_mujoco/mjcf/` are
  `.gitignore`d; `git add -f` them or a TACC clone gets a scene that won't load.
- The real repo is `aic_0/aic`. The top level has a stray empty `.git` and
  `aic_1/` is a stale copy — ignore both.

---

## 9. Open questions I could not resolve

1. **Does the collector actually work?** Unknown. Never run. Step 1.
2. **What is the true SC grasp transform?** Also unknown, and deliberately so —
   the whole point is to stop depending on it. Note that
   `dump_sc_grasp_calibration()` reports the simulator publishes no frame that
   tracks the grasped plug, so the old `RL_INSERT_CALIB_DUMP=1` route cannot
   solve it anyway.
3. **Are the SC keypoint rejection gates right?** `SC_MAX_REPROJECTION_ERROR_PX
   = 6.0` and `SC_MAX_KEYPOINT_RMSE_M = 0.0035` were inherited from SFP because
   the two plugs are of comparable size. They are rejection gates, not accuracy
   claims. Re-tune them against the measured distributions from Step 4.
4. **The installed `aic_model` in `.pixi` is stale.** It predates the plug-pose
   modules, which is why `test_sfp_plug_pose.py` and
   `test_sc_plug_pose_geometry.py` fail to collect. More concerning:
   `test_sc_controller.py` is therefore testing a *stale installed copy*, not
   the source being edited. Worth fixing (rebuild the ROS package), but I left
   the baseline alone rather than change what "73 tests pass" meant.
