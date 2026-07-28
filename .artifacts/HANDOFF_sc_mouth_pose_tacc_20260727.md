# Handoff — physical SC-mouth pose data collection and RTX training

**Completed:** 2026-07-27 13:59 CDT  
**Workspace:** `/home/rschnurr/satya/aic`  
**Worktree:** intentionally dirty. Preserve all pre-existing user changes;
do not reset, checkout, or bulk-overwrite the tree.

## Outcome

The physical SC-mouth data pipeline and a dedicated five-keypoint pose model
are complete on TACC. The live SC controller was deliberately **not** changed:
the new checkpoint must remain an evaluation artifact until a separate runtime
geometry/calibration review approves an atomic migration.

| Item | Result |
|---|---|
| Final dataset | 3,464 matched image/label pairs; 0 malformed label lines |
| Splits | train 2,706; val 355; test 403 |
| Collection job | `3349284`, `COMPLETED` |
| RTX training job | `3349588`, `COMPLETED` in 24m 13s on `rtx-small` |
| Training duration | Early-stopped at epoch 60 (configured maximum: 120) |
| Checkpoint | `training/best_sc_mouth_pose.pt` (19 MB) |
| Test mouths | 799 matched; 2 missed; 0 PnP failures |
| Test 2D mouth-centre error | median 1.06 px; p95 2.56 px |
| Test 2D corner error | median 1.41 px; p95 3.79 px |

The pipeline state is recorded remotely as:

```text
COMPLETE training_job=3349588 checkpoint_and_heldout_report_present
```

## Remote TACC locations

```text
host:      satya_a@stampede3.tacc.utexas.edu
source:    /work2/11590/satya_a/stampede3/aic-sc-mouth-pose-src-20260727
data:      /work2/11590/satya_a/stampede3/aic-sc-mouth-pose-datagen-20260727
dataset:   .../aic-sc-mouth-pose-datagen-20260727/dataset
checkpoint:.../aic-sc-mouth-pose-datagen-20260727/training/best_sc_mouth_pose.pt
reports:   .../training/reports/train_sc_mouth_pose.json
           .../training/reports/validate_sc_mouth_pose_test.json
logs:      .../logs and .../training/logs
```

The pre-existing pinned simulator image and Pixi environment were reused:

```text
/work2/11590/satya_a/stampede3/aic-sc-plug-pose-datagen-20260725/containers/aic_eval_pinned.sif
/work2/11590/satya_a/stampede3/aic-sc-plug-pose-src-20260725/.pixi
```

Use the authenticated control socket when it exists:

```bash
ssh -S /home/rschnurr/.ssh/cm-stampede3 -o BatchMode=yes \
  satya_a@stampede3.tacc.utexas.edu
```

## Data/model contract

The old `best_sc_pose.pt` predicts a nonphysical 8.8 × 6.0 mm virtual
rectangle. The new dataset instead uses the physical front-mouth outline:

```text
physical front mouth: 22.407 × 8.10 mm
keypoints:             four physical corners + centre
YOLO label:            one class, five keypoints, 20 tokens per row
split rule:            deterministic whole-trial 80/10/10 split
```

The defining implementation files are:

```text
aic_example_policies/aic_example_policies/ros/sc_mouth_pose_geometry.py
aic_example_policies/aic_example_policies/ros/DataCollectorScMouthPoseGT.py
train_sc_mouth_pose.py
validate_sc_mouth_pose.py
```

Data comes only from simulator entrance-frame TF projection; there is no
pseudo-label fallback. Dataset filenames embed global trial numbers, preventing
train/validation/test leakage between adjacent synchronized viewpoints.

## Validation interpretation

The held-out 2D keypoint result is strong, including `mAP50(P)=0.98795` and
`mAP50-95(P)=0.97449` at the final recorded epoch. The test report also has:

```text
single-view translation error: median 13.55 mm, p95 36.39 mm
single-view rotation error:    median 13.38°, p95 31.73°
```

Those PnP figures are not yet adequate evidence to deploy this model as a
drop-in controller pose source. They likely need camera/depth calibration and
multi-view or temporal pose refinement. Do **not** merely replace the old
weight while retaining its 8.8 × 6.0 mm geometry; model weights, mouth
keypoints, PnP geometry, and deployed overlay must migrate together.

## Important TACC operational history

The successful staged collection jobs were:

```text
3348396  smoke collection — completed, 471 pairs
3348399  full trials 1–25 — completed
3348418  full trials 26–50 — completed
3348868  full trials 51–75 — completed
3349284  full trials 76–100 — completed, 920 additional pairs
3349588  RTX training + held-out validation — completed
```

The following failures did not add conflicting final-dataset samples:

- `3348832`: cold policy/lifecycle startup race.
- `3349164`: policy reached the engine after its lifecycle deadline.
- `3349224`: engine hit its fixed 10-second `/clock` timeout.

The final collector script removes both races by:

1. launching Gazebo with `start_aic_engine:=false`;
2. retrying until `/clock` is published;
3. launching and verifying the Python policy;
4. launching the engine separately only after both clock and policy are ready.

An isolated one-trial smoke (`3349270`) exercised this sequence before the
successful final full batch. Simulator logs are capped at 512 MB and the final
successful logs stayed small.

Do not try to parallelize this data collection on this allocation: TACC's RTX
QoS allowed only one active collection job for this user
(`QOSMaxJobsPerUserLimit`), even while other RTX nodes appeared idle. Splitting
the last batch would have queued jobs serially and increased startup overhead.

## Local files added for this pipeline

```text
.tacc/sc_mouth_pose_datagen.slurm
.tacc/train_sc_mouth_pose.slurm
.tacc/overnight_sc_mouth_pipeline.sh
.tacc/parallel_final_sc_mouth_pipeline.sh
aic_example_policies/aic_example_policies/ros/DataCollectorScMouthPoseGT.py
aic_example_policies/aic_example_policies/ros/sc_mouth_pose_geometry.py
train_sc_mouth_pose.py
validate_sc_mouth_pose.py
aic_model/test/test_sc_mouth_pose_geometry.py
docs/SC_MOUTH_POSE_TACC_PIPELINE.md
```

These are currently uncommitted alongside unrelated user edits. The TACC
source received the necessary pipeline files. Do not stage broad paths or
assume all dirty files belong to this work.

## Verification performed

Locally, before collection:

```bash
bash -n .tacc/sc_mouth_pose_datagen.slurm .tacc/train_sc_mouth_pose.slurm
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run pytest -q \
  aic_model/test/test_sc_mouth_pose_geometry.py
# 4 passed
```

On TACC, the final runner checked image/label equality, 20-token labels, and
the final minimums of 3,000 total plus 300 images in every split before it
submitted training. It also required both the checkpoint and held-out report
before writing the final `COMPLETE` state.

## Recommended next step

Review the held-out geometry report and run a controlled calibration/multi-view
evaluation before any controller migration. Keep the old virtual-rectangle
model operational meanwhile. When deploying, update the checkpoint and all
physical-mouth geometry as one reviewed atomic change, then rerun SC insertion
integration tests.
