# Gazebo redistillation handoff

## Stop condition

Do not spend another full run on `student_env_a.py` until a paired MuJoCo/Gazebo
contract check passes. The current TorchScript network is not an export failure:
its eight parameter tensors are byte-for-byte identical to
`weights/student_a.pt`. The failure is in the values presented to and consumed
from that network.

## Assets that remain authoritative

- `weights/teacher_level1.zip`: frozen privileged residual teacher.
- `scripted_teacher_funnel.py`: base action that must be combined with the SAC
  residual using the same residual scale as teacher evaluation.
- `weights/student_a.pt` and `models/final_insert_sfp_contractA_v1.ts`: useful
  regression artifacts, but not deployable scoring artifacts.

## Confirmed contract problems and local corrections

1. **Deploy home state differed from training (corrected in this worktree).** Training uses the fixed AIC home
   vector. `RLInsert.py` instead uses the first handoff joint state when
   `RL_INSERT_HOME_QPOS` is unset, and the Docker image does not set it. It now
   defaults to the same fixed training vector.
2. **Deploy plug-tip reconstruction was not the training transform (corrected in this worktree).** Composing
   the exported MuJoCo weld and fixed SFP child transforms gives approximately
   `T_tcp_tip.translation = [-0.001777, -0.018874, 0.054722] m` and
   `T_tcp_tip.quaternion_wxyz = [0.985287, 0.168862, -0.004258, -0.026029]`.
   The former `RLInsert.GRASP_TCP_IN_PLUG` reconstruction differed by up to
   about 29 mm in translation and about 89 degrees in orientation. The SFP path
   now uses the composed `T_tcp_tip` directly. This still needs a paired Gazebo
   assertion because evaluation grasps include small perturbations.
3. **Observation and action conversion are now shared.**
   `aic_model/aic_model/rl_insert_contract.py` is a NumPy-only source of truth
   used by both `student_env_a.py` and `RLInsert.py`. It owns quaternion sign,
   the canonical `[lat_x, lat_y, inward]` basis, TCP-to-tip reconstruction, the
   69-vector layout, and deploy action scaling.
4. **Wrench equivalence is unproven, so Gazebo-v1 zeros it.** MuJoCo reads the force and torque sensor
   arrays directly. Gazebo reports a sensor-frame joint wrench with noise. Frame,
   action/reaction sign, baseline, force/torque ordering, and scale must be
   measured before re-enabling it. New distillation and deployment therefore use
   `wrench_mode=zero` by default.
5. **The active policy now runs the bundled pose estimator.** `RLInsert` inherits
   the `PerceptionInsert` multiview SFP path and loads `best.pt`; object GT TF is
   not used by the policy. A 25 px reprojection gate rejects inconsistent
   multiview matches instead of moving toward them.

## No-training Gazebo validation (2026-07-09)

The current TorchScript was tested with the real AIC Engine Gazebo backend,
attached cables, bundled `best.pt` perception, zeroed student wrench fields,
and automatic 12 mm lateral / 15 mm retreat aborts.

- In the first capture, perception was not the failure: the estimated SFP mouth
  differed from scoring GT by approximately `[0.02, 0.04, -1.09] mm`.
- With the raw action convention, a near-level-1 handoff immediately commanded
  outward axial motion and eventually retreated hundreds of millimetres before
  that exploratory run was stopped.
- Flipping only axial action sign at a zero-variance handoff advanced depth from
  `-21.9 mm` to `-13.5 mm`, but lateral error grew from `1.4 mm` to the `12.0 mm`
  safety limit in roughly 1.8 seconds.
- Keeping the axial flip but disabling all learned rotation advanced depth from
  `-21.4 mm` to `-5.3 mm`; lateral error still grew from `2.6 mm` to `12.1 mm`.
  This rules out TCP rotation alone and confirms a broader observation/action
  contract mismatch.
- A later task produced a 157 px multiview reprojection while detections still
  reported 0.88--0.96 confidence. The new reprojection gate catches this case;
  detector validation metrics alone do not certify requested-slot pose quality.

Therefore `student_a.pt` / `final_insert_sfp_contractA_v1.ts` are regression
assets, not a deployable baseline. Do not run them unguarded or interpret their
MuJoCo success rate as Gazebo insertion reliability.

## Repaired adapter preflight (no training)

- The exact frozen-teacher 21-D observation was recovered from repository
  history into `teacher_contract.py`; it is no longer coupled to the removed
  `SceneEnvConfig.privileged_obs` API.
- The frozen teacher scored 10/10 at level 1 through the repaired wrapper.
- `student_env_a.py` constructs a finite 69-D observation, teacher inference
  works, and one teacher-labeled physics step succeeds.
- TCP-derived SFP tip position agrees with MuJoCo GT within 0.031 mm.
- A `+1` deploy axial action maps to exactly +3.5 mm inward and effectively zero
  lateral displacement.
- The old `student_a.pt`, evaluated without training under the corrected
  Gazebo-v1 producer contract, scored 0/10: 9 timeouts and 1 bad collision.
  This is expected because those weights encode the old contract.
- `gazebo_v1` student models mask absolute TCP and port world-pose fields inside
  the exported network. The policy must use port-relative error, velocity,
  alignment, joint state, and action history instead of memorizing a fixed
  MuJoCo board placement.
- The student observation adds a per-episode hidden TCP-to-tip perturbation
  (base sigma 2/2/4 mm and 1.5 degrees) while teacher labels remain GT. This
  covers the approximately 4 mm grasp/reconstruction discrepancy seen in the
  first Gazebo capture; the multiplier is recorded as `grasp_noise`.

The nominal perceived SFP basis itself is close to the exported Gazebo/MuJoCo
entrance basis: labeled keypoint +X, `+Z = world down`, and
`+Y = cross(+Z, +X)`. On the checked scene, forcing perceived +Z exactly down
removes only about 0.72 degrees of asset tilt. This does not clear the whole
contract; paired snapshots are still required for randomized boards.

## Required parity capture

Capture the same physically aligned states in development Gazebo with object GT
temporarily enabled and in MuJoCo. Log raw producer values, not only `obs69`:

- arm joint names, positions, and velocities;
- TCP pose and twist in `base_link`;
- port mouth pose from GT and from perception;
- plug-tip pose from GT and from the deploy reconstruction;
- raw wrench, message frame ID, and a free-space baseline;
- previous action;
- the final port basis and world-space action delta.

At the mouth, 10 mm inserted, and fully seated, enforce these gates:

- lateral `delta_port` is below 0.5 mm when physically centered;
- axial `delta_port` is positive inward and agrees within 0.5 mm;
- aligned `rot_err_port` is below 0.02 rad;
- aligned tip axes are near `[1,0,0]` and `[0,0,1]`;
- a positive axial policy action produces a world displacement whose dot product
  with the GT inward axis is positive and whose lateral component is negligible;
- zero-contact baseline-subtracted wrench is near zero in both simulators;
- a controlled axial contact has the same signed axial wrench in both.

## Redistillation sequence

1. Put observation construction and action conversion behind one pure numerical
   contract used by both the MuJoCo wrapper and `RLInsert`.
2. Make the student consume the deploy reconstruction of the tip and the
   perception-frame port pose. The privileged teacher continues to consume GT.
3. Fit perception/grasp/noise randomization from paired Gazebo captures. Include
   board yaw and grasp-offset randomization; the current MuJoCo board pose is
   otherwise effectively fixed.
4. Generate a fresh teacher-driven BC dataset. Never reuse Contract-A shards.
5. Train BC and certify closed-loop MuJoCo performance.
6. Run DAgger rounds: roll the current student, label every visited state with
   the frozen teacher action, aggregate those samples with all prior data, and
   refit. The trainer now uses unique per-round shard prefixes so DAgger data is
   actually aggregated instead of overwriting earlier worker-0 shards.
7. Export a versioned TorchScript artifact containing contract metadata, then
   pass the paired observation test and closed-loop Gazebo scoring test before
   treating it as deployable.

## Remote distillation commands

No local training is required. On the training machine, start from a fresh
output directory and run BC plus two DAgger rounds:

```bash
export PYTHONPATH="$PWD"
export WANDB_MODE=offline
export MUJOCO_GL=egl

python -m RL.student_teacher.train_student_a \
  --teacher-zip RL/student_teacher/weights/teacher_level1.zip \
  --transitions 150000 --epochs 40 --num-envs 12 \
  --action-convention deploy --wrench-mode zero --feature-mode gazebo_v1 \
  --perception-noise 1.0 --grasp-noise 1.0 --level 1.0 \
  --dagger-iters 2 --dagger-transitions 30000 --dagger-epochs 10 \
  --eval-episodes 100 --regen \
  --out RL/output/student_teacher/student_gazebo_v1 --seed 0
```

Export only after the closed-loop MuJoCo evaluation is acceptable:

```bash
python -m RL.student_teacher.export_student_a \
  --checkpoint RL/output/student_teacher/student_gazebo_v1/student_a.pt \
  --out models/final_insert_sfp_gazebo_v1.ts
```

The exporter embeds the `gazebo_v1` input mask in TorchScript, checks numerical
parity on random inputs, and writes a neighboring `.contract.json`. Copy the
`.pt`, `.ts`, `.contract.json`, and `metrics.jsonl` back together.

## Is DAgger required?

DAgger cannot repair a frame mismatch. Start with BC because it is the quickest
contract check. DAgger is still recommended after BC passes: pure BC only sees
teacher-visited states, while a student encounters its own small errors and can
drift into states absent from the dataset. The previous approximately 80% BC
versus approximately 96% DAgger result is direct evidence that this distribution
shift matters here. Two or three aggregation rounds are a reasonable starting
point; stop based on closed-loop validation rather than training loss.

## Perception deployment

Object GT TF must not be a runtime input. Extract the SFP keypoint/triangulation
logic from `PerceptionInsert` into the active policy (or a publisher started in
the same container) and return a port-mouth pose with the exact canonical basis.
The plug-tip pose should come from robot kinematics plus a calibrated
`T_tcp_tip`, with grasp perturbations represented during distillation. If the
robot TCP TF is unavailable, compute the same TCP pose from joint encoders and
the robot model; do not substitute an object TF.
