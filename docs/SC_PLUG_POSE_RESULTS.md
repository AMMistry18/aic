# SC plug-pose: collection, training and measured accuracy

**Run 2026-07-25/26 on Stampede3 (`rtx-small`, account IRI26004).** This records
the completed training and evaluation run; the earlier planning handoff has
been retired.

**Result: the 0.4 mm working target is met, with margin, on a held-out split.**

---

## 1. Headline numbers

Measured by `validate_sc_plug_pose.py --mode dataset` against simulator TF
ground truth, with the crop-refine second pass enabled.

| Gate | val split | test split | Limit |
|---|---|---|---|
| `position_error_mm.median` | **0.2733** | 0.2633 | ≤ 0.4 |
| `lateral_error_mm.p95` | **0.3291** | 0.3434 | ≤ 0.725 |
| `group_miss_rate` | **0.0** | 0.0 | ≤ 0.01 |
| `axis_p95_deg` | 1.88 | 1.81 | ≤ 3.0 |
| `all_gates_pass` | **true** | true | — |

The val split was never used to tune anything (see §4), so 0.2733 mm is the
number to quote. Val and test agree to within 0.01 mm.

Reports are committed under `docs/reports/`.

## 2. What was produced

| Artifact | Where |
|---|---|
| Trained checkpoint | `aic_example_policies/.../ros/weights/best_sc_plug_pose.pt` (mirrored to the v50 overlay) |
| Dataset | 4,050 images / 450 trials, splits 3240/405/405, 0 malformed labels, 1.2 GiB |
| Training | `yolo11s-pose`, 120 epochs, imgsz 960, batch 8 — 2 h 08 m, `mAP50-95(P) = 0.995` |
| Collection | 56 min for 215 trials (a second batch; see §5) |

## 3. The accuracy result is a bias story, not a noise story

Straight YOLO inference on full frames gave **0.456 mm median (test)** — a miss.
The handoff's error budget says σ ≈ 2 px should yield ~0.343 mm, and measured
keypoint error was **1.47 px median**, i.e. *better* than budget. Better
keypoints, worse tip error: that is the signature of bias, which triangulation
cannot average away.

`kpt_bias_analysis.py` (committed) measured per-keypoint signed error against
the GT labels on the test split:

```
kpt  plane  mean_dx  mean_dy   std_dx  std_dy
  0  near    -0.065   -0.782    0.650   0.658
  1  near    -0.008   -1.446    1.389   1.103
  2  near     0.313   -1.254    1.362   1.343
  3  near     0.131   -0.715    0.719   0.635
  4  REAR    -0.171   -0.781    1.127   1.329
  5  REAR     0.131   -1.433    1.250   1.442
  6  REAR     0.315   -1.323    1.307   1.404
  7  REAR     0.159   -0.693    1.255   1.250
```

`mean_dx ≈ 0` everywhere; `mean_dy` is negative for **all eight**, averaging
**−1.05 px**. The whole predicted cuboid sits about one pixel high in every
image. At f = 1236.6 px and 280 mm working distance one pixel is 0.226 mm, so
the bias alone is worth ~0.238 mm — and the observed error *floor* across 135
groups was 0.272 mm, with no group ever doing better. That floor is the bias.

**Two hypotheses were checked and refuted first.** Recording them so they are
not re-tried:

1. *"`SC_PLUG_LOCAL_KEYPOINTS_M` z-placement disagrees with the mesh."* It
   cannot matter. `DataCollectorScPlugPoseGT` builds the labels by projecting
   that same constant, and `ScPlugPoseEstimator` fits against it. The constant
   cancels — a mismatch would move the labels and the fit target together and
   recover an identical frame.
2. *"The rear-plane keypoints (4-7) are occluded inside the gripper, so the
   model guesses them badly."* Measured rear/near bias ratio: **1.01×**. The
   rear plane is no more biased than the visible one.

## 4. The fix: crop-refine, and the trap in it

Second pass: take the first-pass box, crop a padded region from the
**native-resolution** frame, re-run the same model, remap. Because Ultralytics
returns coordinates in the frame of the image it was given, remapping is a pure
translation by the crop origin — no rescaling, so no opportunity to introduce a
new offset.

`crop_pad_scale` is a genuine tradeoff and **the default matters enormously**:

```
   pad   miss_rate   pos_med   pos_min   lat_p95   kpt_px   n_pose
  none      0.0000    0.4562    0.2719    0.4121   1.4698      135
     2      0.9481         —         —         —   4.0268        7
     3      1.0000       nan       nan         —   8.6825        0
     4      0.0000    0.2683    0.0099    0.5119   1.7533      135
     6      0.0000    0.2633    0.0350    0.3434   1.2579      135  <- default
     8      0.0000    0.2936    0.0210    0.3887   1.2640      135
    12      0.0000    0.4562    0.2719    0.4121   1.4698      135
    16      0.0000    0.4562    0.2719    0.4121   1.4698      135
```

- **Too tight is catastrophic and does not look like a failure.** At pad 2.0 the
  plug appears ~10× larger than anything in training; 94.8% of groups produce no
  pose, and the surviving 5% report a *better* median (0.411 mm) purely by
  survivorship. Read `group_miss_rate` before reading any error metric.
- **Too loose is a silent no-op.** Pad 12 and 16 reproduce the uncropped result
  bit for bit because the crop exceeds the frame. This is also a useful
  correctness check on the remapping.
- Pad 4-8 all hold. **6.0 sits mid-plateau and measured best**, so the default is
  not perched on the cliff between 3 and 4.

`pos_min` is the column that proves the mechanism: the 0.272 mm floor collapses
to 0.035 mm. The bias is removed, not merely averaged down.

Enabled with `AIC_PLUG_POSE_CROP_REFINE=1` (env) or `crop_refine=True`.
Default is **off**, so the SFP path is untouched; `crop_pad_scale` defaults to
6.0 when it is on.

## 5. Pipeline bugs found and fixed

These are why the earlier SC attempts (jobs 3329045 cancelled at 8 h 22, 3329874
**TIMEOUT at 12 h 00**) never produced data. Both were silent.

1. **The collector never exits.** It completes every trial, writes the dataset
   yaml, runs its lifecycle down to `on_shutdown` — and the process stays alive.
   The script blocked in `wait` until Slurm killed the job. The data in those
   runs was almost certainly already on disk. Fixed with a reaper that detects
   the completed shutdown, allows 45 s of grace, then forces termination and
   marks the non-zero exit as legitimate so the integrity checks still gate.
2. **`aic_engine` loses a startup race.** It gives up if `/clock` is not
   published within 10 s. When Gazebo's render falls back to headless
   (`libEGL: failed to create dri2 screen`) it misses that window, the engine
   dies, and *nothing drives any trial*. Its message is `aic_engine-7: process
   has died`, which the existing crash pattern (`component_container.*process
   has died`) did not match. Pattern extended.
3. **No liveness guard at all.** Added a progress watchdog: no trial completed
   within 7 min of start, or a 7 min stall mid-run, stops the job with a
   diagnosable reason instead of burning 12 h.

Also hardened for space, and `SC_TRAIN_AFTER_COLLECTION=1` fixed — it referenced
`${repo_root}/train_sc_plug_pose.slurm`, but the file is at `.tacc/`.

**Gazebo degrades across trials.** Measured per-trial time climbs 9.4 s → 18.7 s
over ~175 trials, then plateaus. Restarting gives a fresh ~7 s/trial, so
collecting in batches via `SC_TRIAL_START` is faster as well as safer. Note
`qvrtx` allows only **one running job per user**, so batches are sequential.

## 6. What is still open

- **Rotation is unimproved.** 1.03° median, essentially identical to the
  uncropped baseline. The bias has an x-dependent component (keypoints at local
  x = +10 mm carry ~−1.36 px, those at −10 mm ~−0.74 px) which is a small roll
  and survives cropping. It passes the axis gate with margin.
- **Crop-refine costs a second inference pass** per view. Fine offline; not
  benchmarked for the control loop.
- **Controller wiring is complete.** The active SC path is documented in
  `docs/CURRENT_SYSTEM.md`; this file remains the measured model-accuracy
  record rather than the runtime contract.
- **The rejection gates were never re-tuned** (`SC_MAX_REPROJECTION_ERROR_PX =
  6.0`, `SC_MAX_KEYPOINT_RMSE_M = 0.0035`, both inherited from SFP). They now
  have measured distributions to tune against — open question 3 in the handoff.

## 7. Reproducing

```bash
cd $WORK && git clone https://github.com/AMMistry18/aic.git aic-sc
cd aic-sc
PIXI_HOME=$WORK/pixi PIXI_CACHE_DIR=$WORK/.pixi-cache \
  RATTLER_AUTH_FILE=$WORK/.rattler_auth.json pixi install --frozen
```

Two things that cost an hour if you miss them:

- **`--frozen` matters.** A full solve is slow; installing from `pixi.lock` took
  4 minutes.
- **`RATTLER_AUTH_FILE` matters.** Without it pixi reaches for the system
  keyring, spawns `gnome-keyring-daemon`, tries to talk to a terminal that does
  not exist in a batch job, and leaves the process group **stopped** (`STAT=T`)
  until the walltime expires. It looks exactly like a very slow install.

Then build `containers/aic_eval_pinned.sif`, update the dated `job_root` /
`repo_root` paths in both `.tacc` scripts, and:

```bash
SC_MODE=smoke SC_TRIAL_COUNT=10 sbatch .tacc/sc_plug_pose_datagen.slurm
SC_MODE=full  SC_TRIAL_COUNT=450 SC_SEED=20260725 sbatch .tacc/sc_plug_pose_datagen.slurm
sbatch .tacc/train_sc_plug_pose.slurm
```

Note `.tacc/train_sc_plug_pose.slurm` runs validation automatically but **does
not enable crop-refine**, so it reports the ~0.45 mm uncropped number. Set
`AIC_PLUG_POSE_CROP_REFINE=1` to get the 0.27 mm figure.
