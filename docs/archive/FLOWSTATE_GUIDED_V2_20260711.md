> **ARCHIVED / SUPERSEDED (2026-07-12).** Describes the GUIDED scripted controller. The deployed policy now runs `RL_INSERT_CONTROL_MODE=rl`; guided is only a fallback. See `docs/FLOWSTATE_STATUS.md`.
> Kept for history only — do not follow as current instructions.

# Flowstate guided insertion v2

Updated: 2026-07-11

## Observed failure

The first `flowstate_v1` deployment fixed the simulator-specific joint input,
but its first live closed-loop run still failed. Perception and prepositioning
were correct:

```text
handoff delta_port_mm=[0.03, 0.06, -21.89]
handoff rot_err_deg=[-0.0, 0.0, 0.0]
```

The learned loop then increased lateral error from 0.1 mm to 13.2 mm and tilted
the connector until the contract safety guard aborted. The first lateral action
was `-0.585`, despite the same packaged handoff fixture producing `-0.111`.

The remaining observation mismatch was wrench timing. Deployment sampled the
`baseline` wrench before perception and a roughly 40-second preposition move.
MuJoCo samples it after reset at the final handoff. Cable gravity and tension
therefore appeared to the deployed model as false contact forces.

## Corrected controller

The new deployment:

1. Resamples the six-axis wrench baseline after prepositioning and settling.
2. Keeps the TorchScript model active for diagnostics.
3. Uses an absolute, perception-guided SFP tip target for motion.
4. Centers the tip on the perceived port axis and locks its orientation to the
   port frame, eliminating cumulative lateral and rotational command drift.
5. Advances 1.5 mm per target outside the mouth and 0.75 mm near/inside contact.
6. Caps the target at 4 mm in free space. Within 8 mm of the mouth it permits
   a 20 mm lead with 500 N/m translational stiffness, bounding the requested
   contact push at roughly 10 N instead of the ineffective 0.4 N seen with the
   original 100 N/m, 4 mm configuration.
7. Aborts at 6 mm lateral error, 0.20 rad rotation error, or sustained 18 N
   baseline-subtracted force.

The old incremental learned mode remains available with
`RL_INSERT_CONTROL_MODE=rl`, but the Flowstate image explicitly selects
`RL_INSERT_CONTROL_MODE=guided`.

## Validation

Contract tests cover centered/aligned targets, near-contact speed, and target
lead limits. A 10-episode exact-pose smoke run succeeded 10/10.

TACC Slurm job `3296234` runs the fixed held-out seeds used for student
selection. The calibrated-pose result is:

```text
300/300 success
0 timeout
0 bad_collision
seeds: 10001, 20002, 30003 (100 episodes each)
```

Machine-readable evidence:

```text
RL/student_teacher/parity/guided_exact_pose_evaluation_300.json
/scratch/11590/satya_a/aic/guided_flowstate_v2_20260711
```

Synthetic perception/grasp noise is evaluated separately because those hidden
perturbations are deliberately unobservable to a geometric controller. Do not
conflate that stress test with the captured Flowstate handoff, whose measured
lateral and rotational errors were effectively zero.

The official local Gazebo run confirmed the post-handoff wrench reset and
maintained about 1 mm lateral and zero rotational error, but the first version
stalled 4.9 mm before the mouth because its impedance target could request only
0.4 N. The contact-authority correction above addresses that measured cause.
An accelerated rerun was invalidated before policy execution when the emulated
evaluator's `aic_controller` spawner timed out three times; it is not counted as
a policy trial.

## Flowstate deployment

Installed on 2026-07-11:

```text
solution: 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
cluster: vmp-f5ed-cjm9g9wx
asset: ai.intrinsic.aic_model.0.0.1+c84d8e248aa372bfa959e0e0b790f6150d96ffd1900226879d6da3798741d393
service instance: aic_model
bundle: /private/tmp/aic-flowstate-guided-v4/images/aic_model/aic_model.bundle.tar
```

The stale pre-existing `aic_model` service instance was deleted and the exact
asset version above was added successfully. The bundle and authentication data
remain local-only and must not be committed.
