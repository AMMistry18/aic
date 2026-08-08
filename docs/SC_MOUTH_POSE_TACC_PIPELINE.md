# SC physical-mouth pose — TACC collection and training

The trained physical-mouth model is now the canonical SC port-pose contract for
the Docker `InsertionPolicy -> run_sc_insertion` runtime:

- checkpoint: `weights/best_sc_mouth_pose.pt`;
- schema: one `sc_mouth` class, four corners plus explicit centre;
- geometry: 22.407 x 8.10 mm physical front mouth; and
- motion target: multiview triangulation of KP4, with the corners retained for
  orientation, cyclic-order resolution, and geometry diagnostics.

The legacy `best_sc_pose.pt` remains only as an explicit A/B artifact. The
runtime rejects its four-keypoint output instead of guessing a geometry.
The 7.85 mm binding-throat height remains unchanged in the seating controller;
it is a mechanical clearance, not pose-label geometry.

## Target and data contract

`DataCollectorScMouthPoseGT` labels the physical SC **front-mouth** outline:

| Keypoints | Geometry in the entrance plane |
|---|---|
| 0–3 | four corners of the 22.407 x 8.100 mm front mouth |
| 4 | mouth centre |

It intentionally does not label the 7.85 mm binding throat.  The first smoke
overlays must be reviewed to confirm that these front-face edges are actually
the strongest observable feature at deployment handoff angles.  If they are
not, collect a separate throat-labelled dataset; do not mix its labels here.

The collector writes only timestamp-aligned entrance-frame projections.  It
has no HSV/pseudo-label fallback.  All five points must be visible, and the
split is deterministic by complete global trial index: 80% train, 10% val,
10% test.  Each metadata record includes `K` and `T_camera_mouth`, so the
trainer can evaluate metric single-view PnP error against TF ground truth.

## TACC prerequisites

Stage a source tree that contains this pipeline at:

```text
/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-src-20260727
```

The scripts default to the existing pinned evaluation image from the successful
SC plug-pose run:

```text
/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260725/containers/aic_eval_pinned.sif
```

Build the source tree's Pixi environment on `/work2`, never `$HOME`:

```bash
cd /work2/11590/satya_a/stampede3/aic-sc-mouth-pose-src-20260727
export PIXI_HOME=/work2/11590/satya_a/stampede3/pixi
export PIXI_CACHE_DIR=/work2/11590/satya_a/stampede3/.pixi-cache
export RATTLER_AUTH_FILE=/work2/11590/satya_a/stampede3/.rattler_auth.json
printf '{}' > "$RATTLER_AUTH_FILE"
pixi install --frozen
```

Before submitting, create the Slurm output directories (Slurm opens its output
file before the script gets to its own `mkdir -p`):

```bash
mkdir -p /work2/11590/satya_a/stampede3/aic-sc-mouth-pose-datagen-20260727/{logs,training/logs}
```

`RATTLER_AUTH_FILE` prevents noninteractive Pixi from stopping in a keyring
prompt.  Do not add `set -u` to either Slurm script: TACC's module setup reads
optional unset variables.

## 1. Smoke collection — required first

```bash
cd /work2/11590/satya_a/stampede3/aic-sc-mouth-pose-src-20260727
SC_MODE=smoke SC_MOUTH_TRIAL_COUNT=10 sbatch .tacc/sc_mouth/collect.slurm
```

Each smoke job writes a fresh dataset directory
`aic-sc-mouth-pose-datagen-20260727/smoke_dataset/<job-id>/`, preventing stale
files from a prior run from passing its checks.  After it succeeds:

```bash
smoke_root=/work2/11590/satya_a/stampede3/aic-sc-mouth-pose-datagen-20260727/smoke_dataset/<job-id>
find "$smoke_root/images" -name '*.png' | wc -l
find "$smoke_root/labels" -name '*.txt' | wc -l
find "$smoke_root/labels" -name '*.txt' -exec awk 'NF != 20 { print FILENAME ":" FNR }' {} +
```

The first two counts must agree, there must be no 20-token failures, and the
debug overlays must be inspected.  Yellow quads and green centre dots must lie
on the physical outer mouth edge—not the old small central virtual rectangle,
not the blue-face sticker, and not the seat behind the lip.  Stop if this is
not true; bad labels cannot be repaired by training.

The job includes the three protections that proved necessary for SC plug pose:
bag cleanup plus a dataset/engine-home storage watchdog, a simulator-start/crash watchdog, and a collector
reaper for the known post-`on_shutdown` hang.  It fails quickly on a 7-minute
startup or progress stall rather than consuming the 12-hour allocation.

## 2. Full collection

Collect four sequential 25-trial batches.  Restarting between batches avoids
Gazebo's measured slowdown, while `SC_MOUTH_TRIAL_START` makes names and
train/val/test membership continuous:

```bash
SC_MODE=full SC_MOUTH_TRIAL_START=1  SC_MOUTH_TRIAL_COUNT=25 sbatch .tacc/sc_mouth/collect.slurm
SC_MODE=full SC_MOUTH_TRIAL_START=26 SC_MOUTH_TRIAL_COUNT=25 sbatch .tacc/sc_mouth/collect.slurm
SC_MODE=full SC_MOUTH_TRIAL_START=51 SC_MOUTH_TRIAL_COUNT=25 sbatch .tacc/sc_mouth/collect.slurm
SC_MODE=full SC_MOUTH_TRIAL_START=76 SC_MOUTH_TRIAL_COUNT=25 sbatch .tacc/sc_mouth/collect.slurm
```

Run only one at a time on `qvrtx`; submit the next after the current job's
integrity summary succeeds.  At 14 viewpoints and three cameras per trial,
100 healthy trials give up to 4,200 images and nominally 3,360/420/420
train/val/test images.  The train job refuses fewer than 3,000 total or 300 in
any split, so it cannot silently train a partial batch.  The full job uses 14
independent randomized viewpoints per trial, yielding up to 3,600 images over
100 trials; it does not inflate the count with duplicate captures at one pose.

## 3. Training and held-out report

```bash
sbatch .tacc/sc_mouth/train.slurm
```

This trains `yolo11s-pose` at 960 px, stores the candidate checkpoint at:

```text
aic-sc-mouth-pose-datagen-20260727/training/best_sc_mouth_pose.pt
```

It then writes `reports/validate_sc_mouth_pose_test.json`, including:

- model miss rate and 2-D physical-mouth centre/corner error;
- single-camera PnP translation error in mm and rotation error in degrees
  versus held-out simulator TF;
- inference time.

These numbers are an A/B evaluation artifact, not an automatic deployment
decision.  Before switching, compare the candidate against the legacy model
on a clean held-out entrance-frame set and update the checkpoint, local
geometry, PnP/multiview assumptions, and operator overlay together.
