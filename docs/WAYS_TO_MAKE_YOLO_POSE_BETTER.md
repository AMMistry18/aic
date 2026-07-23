# Ways to Make YOLO Pose Better (SC port perception)

Implementation handoff — written for someone with no prior context. Three
work items, in priority order. Background/audit is in
docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md; this file is the how-to.

**Context in one paragraph**: the SC port (an 8.8×6.0 mm rectangle) is
detected by a YOLOv8m-pose model (`best_sc_pose.pt`, 4 corner keypoints,
weights in `aic_example_policies/aic_example_policies/ros/weights/`) from
three 1152×1024 wrist cameras. The 2D keypoints are turned into a 3D port
pose that the insertion controller servos against. We need ≤1 mm median
lateral / ≤1.5° rotation error. An `imgsz` inference fix is already applied
(perception_core.py `detect_sc_pose` now runs at 960 — do not undo).
NOTE: the ACTIVE perception file is
`aic_example_policies/aic_example_policies/ros/perception_core.py`; the copy
at repo root has no YOLO SC path and is NOT the one to edit.

**Measure before/after every item.** The eval dataset
(`~/aic_perception_data/pose_sc/`) does not currently exist on this machine —
regenerate it with `DataCollectorScPoseGT` (see Item 3, which modifies the
collector anyway) and get a baseline via `eval_sc_pose_model.py
--weights aic_example_policies/aic_example_policies/ros/weights/best_sc_pose.pt`
plus a 3D-error check before starting. No item is "done" without a number.

---

## Item 1 — Rigid-shape PnP for SC (replace triangulate-then-average)

**Problem.** SC 3D pose is currently computed by triangulating each of the 4
corner keypoints independently (linear DLT, `perception_core.py:180-204` of
the example-policies copy) and averaging
(`PerceptionInsert.py:1395-1406`, `_make_sc_pose_multiview_candidates`).
Nothing enforces that the 4 points form the known rigid 8.8×6.0 mm
rectangle, so per-corner pixel noise flows directly into the position AND the
orientation estimate. (Contrast: the SFP path at least has a PnP fallback,
`RLInsert.py:643-669`.)

**Fix.** Per camera, solve a planar PnP against the known local corner model,
refine it, and only then fuse across cameras/frames.

1. Define the local model (must match the label convention used by the data
   collector — TL, TR, BR, BL order, `DataCollectorScPoseGT.py:43-51`):
   ```python
   SC_HALF_W, SC_HALF_H = 0.0044, 0.0030   # meters; verify against collector
   LOCAL_SC_PORT_KPS = np.array([
       [-SC_HALF_W, +SC_HALF_H, 0.0],   # TL
       [+SC_HALF_W, +SC_HALF_H, 0.0],   # TR
       [+SC_HALF_W, -SC_HALF_H, 0.0],   # BR
       [-SC_HALF_W, -SC_HALF_H, 0.0],   # BL
   ], dtype=np.float64)
   ```
   ⚠️ There are TWO conflicting SC keypoint conventions in the repo:
   `DataCollectorScPoseGT.py:38-39` (4.4×3.0 mm half-extents) vs
   `DataCollectorPoseSC.py:116-117` (12.89×4.635 mm). `best_sc_pose.pt` was
   trained on ONE of them. Before trusting PnP output, verify which one by
   overlaying predicted keypoints on a ground-truth frame and checking the
   corner spacing. Using the wrong local model produces a silently
   scaled/shifted pose.

2. Per-camera solve — use the planar-specific solver, NOT the default:
   ```python
   ok, rvec, tvec = cv2.solvePnP(
       LOCAL_SC_PORT_KPS, kps_px.astype(np.float64), K, dist,
       flags=cv2.SOLVEPNP_IPPE)          # planar analytic solver
   if ok:
       rvec, tvec = cv2.solvePnPRefineVVS(
           LOCAL_SC_PORT_KPS, kps_px, K, dist, rvec, tvec,
           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     20, 1e-6))
   ```
   - `SOLVEPNP_IPPE` (not `IPPE_SQUARE` — the target is a rectangle, not a
     square). IPPE is analytic for planar point sets and surfaces the planar
     pose ambiguity instead of silently picking a wrong local minimum, which
     `SOLVEPNP_ITERATIVE` can do at near-fronto-parallel viewing angles.
   - Cap RefineVVS iterations (shown) — rare non-convergence reports exist.
   - `dist`: currently zeros everywhere; read `CameraInfo.D` once — if sim
     publishes non-zero distortion, pass it.

3. Use per-keypoint confidence (currently discarded — only `keypoints.xy` is
   read at `perception_core.py:161`): also fetch `result.keypoints.conf`,
   and skip PnP for a camera if any corner conf < ~0.5, or better, run
   PnP with all 4 but store `min_kp_conf` and weight the fusion step by it.

4. Fusion across cameras: keep the existing multi-frame consensus machinery
   (median cluster, `RLInsert.py:714-763`) but feed it per-camera PnP poses
   (weighted by min-kp-conf and reprojection error) instead of the
   DLT-averaged point. Sanity-gate each PnP pose with reprojection error
   (reuse the existing `MAX_PORT_REPROJ_PX` idea, `RLInsert.py:72`).

**Acceptance.** On a held-out GT val set (≥200 frames at handoff-like
viewpoints): rotation error median vs the DLT baseline must improve; target
≤1.5° median. Position median ≤1 mm. Also verify the recovered rectangle
side lengths (back-projected) match 8.8×6.0 mm — that's the rigidity check.

**Effort.** ~half a day + eval time. Touches:
`aic_example_policies/.../perception_core.py` (or a new helper),
`PerceptionInsert.py:1386-1444`, and the SC path being added to
`aic_model/aic_model/RLInsert.py`.

---

## Item 2 — Crop-refine two-stage inference (biggest accuracy win)

**Problem.** The port occupies a tiny fraction of the 1152×1024 frame. Even
at imgsz=960 the 8.8 mm mouth spans only a few dozen pixels; keypoint error
in *pixels* is roughly constant, so accuracy in *millimeters* is limited by
pixels-per-mm on the object. A second pass on a zoomed crop multiplies
effective resolution 5–10×. This is the standard high-precision keypoint
pattern (used by G-RMI/CPN-style human pose and 6D object-pose pipelines,
e.g. Pix2Pose crops bbox×1.5 then resizes to a fixed square).

**Design.**
1. **Coarse pass** (existing): `detect_sc_pose(bgr)` at imgsz=960 → bbox.
2. **Crop**: expand bbox by 1.5× (padding absorbs coarse-detection jitter),
   clamp to image bounds, keep fixed aspect (square is simplest). Crop from
   the ORIGINAL full-res frame — never from a resized copy.
3. **Refine pass**: run the pose model on the crop with imgsz set to the
   crop's resized size (e.g. resize crop to 640×640). Because the object now
   fills most of the input, effective px/mm is far higher.
4. **Remap to full-frame coordinates** (critical, easy to get wrong):
   ```python
   # crop taken at (x0, y0), size (cw, ch), resized to (sw, sh) for the model
   kp_full_x = x0 + kp_crop_x * (cw / sw)
   kp_full_y = y0 + kp_crop_y * (ch / sh)
   ```
   Then run PnP (Item 1) with the ORIGINAL camera intrinsics K on the
   remapped full-frame pixels. Do NOT pass crop-local pixels to PnP with the
   full-frame K, and do NOT adjust K instead of remapping — mixing the two
   conventions biases the pose (this is the img2pose crop/intrinsics caveat).

5. **Train/test scale match (decides whether a retrain is needed).** The
   current model was trained on full frames at 960, so objects appeared
   small; in the crop they appear large — out of the trained scale range
   (`scale=0.5` augmentation covers 0.5–1.5×, not 5–10×). Two options:
   - **Option A (preferred): retrain on crops.** Dataset images are saved at
     full 1152×1024 native res by the collector, so generate a crop dataset
     offline: for each labeled image, cut the 1.5×-padded GT-bbox crop,
     transform labels into crop coordinates, write a new YOLO-pose dataset.
     Train with `train_sc.py` pointed at it (imgsz=640 on crops is plenty).
     Randomize the crop center/scale slightly per sample so the model
     tolerates coarse-stage jitter. Ship as a SECOND weights file (e.g.
     `best_sc_pose_crop.pt`) — keep the coarse model unchanged.
   - **Option B (test first, zero training):** just run the existing model on
     the crop and measure. Sometimes works acceptably; if val numbers are
     good, skip the retrain. Measure, don't assume.

6. Runtime: two YOLO passes ≈ 2× inference cost per frame. The pipeline
   samples 7 frames × 3 cameras per perceive (`RLInsert.py:78`); at ~10-20 ms
   per pass this stays well inside budget.

**Acceptance.** ≥2× reduction in median keypoint pixel error on the val set
vs single-stage; 3D median lateral ≤1 mm at handoff viewpoints. Also test
robustness: perturb the coarse bbox by ±10 px and confirm refine output is
stable (that's what the 1.5× padding is for).

**Effort.** Option B: ~2 hours. Option A: ~1 day including dataset generation
and a retrain (150 epochs at imgsz=640 on crops is fast).

---

## Item 3 — Angle-diverse data + rotation augmentation (fixes holding-angle sensitivity)

**Problem.** The model systematically mis-localizes keypoints when the camera
views the port from a slightly different angle than it was trained on —
observed in practice as pose error that varies with the cable's grasp/holding
angle. Root cause is twofold and both halves must be fixed:
- The GT data collector holds camera ORIENTATION fixed for all 18 viewpoints —
  it only offsets position (dx,dy ∈ ±0.06 m, dz ∈ [-0.01, 0.16] m;
  `DataCollectorScPoseGT.py:73-83`, `_move_to_offset` :333-351 never changes
  orientation).
- Training disables all rotation augmentation: `degrees=0.0, shear=0.0,
  perspective=0.0` (`train_sc.py:86,89-90`).
So the network has literally never seen the port at a tilted viewing angle.

**Fix — collector** (`aic_example_policies/.../DataCollectorScPoseGT.py`):
1. In `sample_viewpoints` / `_move_to_offset`, add a random orientation
   jitter to each viewpoint: sample roll/pitch/yaw offsets uniform in
   ±10–15° and compose them onto the TCP target orientation. Keep the port
   in frame (the existing ≥2-visible-keypoints filter at :226 already rejects
   bad frames).
2. **Oversample the deployment distribution**: bias ~60-70% of viewpoints to
   the actual handoff range (tip 5–30 mm from the mouth, near-normal
   incidence ±15°) rather than the uniform ±6 cm box. Keypoint accuracy only
   matters where inference actually happens.
3. **Disable the HSV pseudo-label fallback** (:256-285): when TF ground truth
   is unavailable it silently labels corners from the blue-blob color
   detector — color-detector-quality "ground truth" poisons training. Make
   the frame fail loudly / be skipped instead.
4. While in there (free wins): add a 5th keypoint at the port center
   (`kpt_shape` [4,3]→[5,3], `flip_idx` [1,0,3,2]→[1,0,3,2,4]) —
   over-determined PnP is strictly more robust; and randomize sim lighting
   between trials if the harness allows.
5. Collect ≥3-5k labeled frames (the collector does 18 viewpoints × 3 cams
   per trial = 54 images/trial → ~60-90 trials). Hold out ≥10% of TRIALS
   (not random frames — frames within a trial are correlated) as val.

**Fix — training** (`train_sc.py`):
```python
degrees=10.0,        # was 0.0  — in-plane rotation
perspective=0.0005,  # was 0.0  — mild out-of-plane warp
shear=2.0,           # was 0.0
# keep: translate=0.1, scale=0.5, fliplr=0.5 (flip_idx handles kp swap)
# consider mosaic=0.5 (from 1.0): mosaic shrinks objects; our object is
# already tiny. Measure both.
```
Retrain (`yolov8m-pose`, imgsz=960, 150 epochs as before), eval with
`eval_sc_pose_model.py`, and compare against the pre-retrain baseline on the
SAME val split.

**Acceptance.** Build a val subset binned by viewing angle (0-5°, 5-10°,
10-15° off-normal — derivable from GT TF at collection time). The old model
will show error growing with angle; the new model's error must be flat across
bins and ≤ the old model's 0-5° bin. That flatness is the direct fix for the
"holding angle" symptom.

**Effort.** ~1 day: collector changes (~2 h), collection run in sim
(hours, unattended), retrain (~hours on the 5090), eval.

---

## Order of work & interaction between items

1. Item 1 first (no training, immediate, and its PnP is needed to *measure*
   3D error properly for the others).
2. Item 3's collector changes next, because BOTH retrains (Items 2A and 3)
   should train on the new angle-diverse, fallback-free dataset — do the
   collection once, use it for both.
3. Item 2 Option B (test existing model on crops) can be measured the same
   day; decide on Option A after seeing its numbers.

Log all metrics to `outputs/sc_pose_pipeline/` (referenced by
`getting_started.md:200` and the eval scripts but currently empty).
