# YOLO-Pose Accuracy Playbook (SC + SFP) — 2026-07-16

Goal: get SC port pose to ≤1 mm median lateral / ≤1.5° rotation error at the
handoff viewpoint, so the insertion pipeline needs no hand-tuned biases.

## Why the "holding angle" bias existed (root cause chain)

Audit of the full accuracy chain (2026-07-16) found five concrete leaks.
Together they explain angle-dependent systematic error that a constant bias
was papering over:

| # | Leak | Evidence |
|---|---|---|
| 1 | **Inference runs at imgsz=640** (ultralytics default — no `imgsz` arg passed), while training used 960 and cameras deliver 1152×1024. ~1.8× resolution thrown away on an 8.8×6.0 mm target, AND train/test scale mismatch. | `aic_example_policies/aic_example_policies/ros/perception_core.py:153`, `tools/perception/sc_port/train.py:33` |
| 2 | **No rigid-shape constraint for SC**: pose = triangulate 4 corners independently (linear DLT, no distortion, no refinement) then average. The known 8.8×6.0 mm rectangle is never enforced; corner noise passes straight into position and yaw. SFP at least has an IPPE PnP fallback; SC has none. | `PerceptionInsert.py:1395-1406`, `aic_example_policies/aic_example_policies/ros/perception_core.py:180-204` |
| 3 | **Per-keypoint confidence discarded** — only `keypoints.xy` read; occluded/uncertain corners weighted equally. | `aic_example_policies/aic_example_policies/ros/perception_core.py:117` |
| 4 | **Zero angle diversity end-to-end**: GT collector holds camera orientation fixed across all 18 viewpoints (XYZ offsets only), and training uses `degrees=0, shear=0, perspective=0`. Any tilt of the wrist/cable at inference is out-of-distribution → systematic keypoint shift that varies with holding angle. **This is the "holding angle" sensitivity.** | `DataCollectorScPoseGT.py:73-83,342-347`, `tools/perception/sc_port/train.py:86-90` |
| 5 | Lens distortion ignored (zeros to PnP, raw K in DLT). In sim cameras are near-ideal so low priority now; becomes real in Phase 2. Check `CameraInfo.D` once. | `CableInsertionPolicy.py:648` |

Architecture context (methods research): ultralytics YOLO-pose uses a
regression head with no sub-pixel decoding, and regression heads degrade most
in exactly the strict-precision regime we need (documented community reports
of consistent directional keypoint shift: ultralytics#19284; DARK/UDP CVPR
2020 papers quantify the heatmap-decoding gap). So the model family has a
ceiling — but we are far below that ceiling today because of leaks 1–4.

## Blocking prerequisite: the dataset does not exist on this machine

`~/aic_perception_data` is absent — eval (`tools/perception/sc_port/evaluate_model.py`) and any
retrain need it regenerated via `DataCollectorScPoseGT` in sim. When
regenerating:

- **Disable the HSV pseudo-label fallback** (`DataCollectorScPoseGT.py:256-285`
  labels from the blue-blob detector when TF GT is missing — silent label
  poisoning; fail loudly instead).
- **Add viewpoint-angle jitter** to the collector (small random camera tilts,
  ±10-15°, plus the existing XYZ offsets) — fixes leak 4 at the source.
- **Oversample the deployment distribution**: most frames should come from the
  actual handoff approach range (close, near-normal incidence), not a uniform
  box.
- **Add a 5th keypoint (port center)** — labels are free in sim;
  over-determined PnP is strictly more robust. Update `kpt_shape` to [5,3]
  and `flip_idx` accordingly.
- Keep full-res 1152×1024 frames (collector already does).

## Interventions, ranked by gain ÷ effort

### Tier 0 — no retraining, do first (hours total)

1. **Pass `imgsz` at inference** (`model(bgr, imgsz=960, ...)` — test 960 vs
   1152 empirically; 960 matches training scale, 1152 preserves native pixels;
   FixRes says scale-match usually wins). One line in `detect_sc_pose` and
   `detect_nic`. Keep the package source and Docker overlay copies synchronized;
   the obsolete repository-root duplicate has been removed.
2. **Use `keypoints.conf`**: drop keypoints below ~0.5 conf from
   triangulation/PnP; weight the rest.
3. **Rigid PnP for SC**: per-camera `solvePnP` with `SOLVEPNP_IPPE` (planar
   rectangle — use IPPE, not IPPE_SQUARE, since 8.8×6.0 is not square) using
   the resolved `LOCAL_SC_PORT_KPS`, then `solvePnPRefineVVS` polish
   (~10 lines, capped iterations). Keep the existing multi-frame consensus on
   top. Do the same refine step for SFP's PnP fallback.
4. **Baseline measurement**: regenerate a val set (above), run
   `tools/perception/sc_port/evaluate_model.py` + a 3D-error variant at handoff viewpoints. Every
   intervention below gets accepted/rejected against this number.

### Tier 1 — this week (1–3 days)

5. **Crop-refine two-stage** (highest expected gain: ~5–10× effective
   resolution on the port): coarse detect with existing model → padded crop
   (~1.5× bbox, fixed aspect) from the **native-res** frame → re-run pose on
   the crop → remap keypoints to full-frame pixel coords before
   PnP (adjust cx,cy by crop offset; scale fx,fy if resized — getting this
   wrong biases the pose, img2pose caveat). Train the refine pass on crops cut
   from the regenerated native-res dataset so train/test crop statistics
   match (FixRes). The refine model can be the same YOLO-pose retrained on
   crops — no new architecture.
6. **Retrain with angle diversity**: regenerated angle-jittered data +
   `degrees≈10, perspective≈0.0005` in tools/perception/sc_port/train.py. Directly kills the
   holding-angle sensitivity. (Keep `close_mosaic`; consider `mosaic=0.5`.)

### Tier 2 — only if Tier 0+1 miss the ≤1 mm gate

7. **Joint multi-camera bundle PnP**: one 6-DoF pose minimizing summed
   reprojection error across all 3 cameras simultaneously (scipy
   least_squares), replacing per-camera-then-average. Directionally superior
   (Multi-View Keypoints for 6D Pose, arXiv:2303.16833); 1–2 days custom
   work.
8. **RTMPose (MMPose) swap**: SimCC sub-pixel decoding, documented
   custom-4-keypoint path (MMPose issue #1029). Highest single-model ceiling,
   2–4 days, integration risk — parallel bet only, not a replacement for #5.
9. `cv2.cornerSubPix` / rendered-template NCC refinement on full-res crops
   seeded by CNN keypoints — cheap to try (~hours) but unverified on molded
   plastic edges; test early, keep only if it measurably helps.

### Not worth the week
YOLO11-pose swap alone (unverified marginal gain), pose/kobj loss tuning
alone (community consensus: low impact), ViTPose (cost/risk), P2-head surgery
(crop stage captures the same benefit cheaper).

## Acceptance gates

- After Tier 0: expect a large step from imgsz fix + rigid PnP alone;
  re-measure before starting Tier 1.
- Gate for downstream control work: **median 3D lateral
  ≤1 mm AND rotation ≤1.5° at handoff-distribution viewpoints, p95 lateral
  ≤2 mm**, measured on ≥200 held-out sim frames.
- Keep the per-run measured numbers in `outputs/sc_pose_pipeline/` (currently
  empty — referenced by 3 files but never populated).
