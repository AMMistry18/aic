# Residual Student-v3 / Gate-0 handoff — 2026-07-12

## Current remote state

- TACC user/host: `satya_a@stampede3.tacc.utexas.edu`.
- The reusable ControlMaster at `/tmp/codex-tacc-%r@%h:%p` was healthy for
  staging and submission, then expired. A new password + MFA authentication is
  required before any remote status check or pilot submission.
- Gate-0 confirmation job: Slurm `3299028` (`aic-v3-gate0`), submitted to
  `rtx-small` under `IRI26004`.
- Gate source snapshot (new; base snapshot untouched):
  `/work2/11590/satya_a/stampede3/aic-residual-v3-gate0-20260712`
- Gate output: `/scratch/11590/satya_a/aic/residual_v3_gate0_20260712`
- Gate log: `/scratch/11590/satya_a/aic/slurm/aic-v3-gate0-3299028.out`

The job validates the frozen teacher before evaluation. Its preserved hash is:

```text
fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4
```

It uses the existing Pixi symlink and `MUJOCO_GL=egl`; it does not run
Distrobox, modify the frozen teacher, datasets, snapshots, or the shared Pixi
environment.

## Gate-0 implementation and local evidence

`RL/scene_env.py` now keeps nominal behavior off by default and provides an
opt-in domain-randomized variant:

- compiled (safe) collision/port-pose variants, including a passable internal
  contact ridge centered at 6--7.5 mm insertion depth;
- per-episode friction, contact solver, controller, cable wrench, command
  latency/cadence, tracking, wrench noise/bias/delay, and contact margins;
- cadence spans 2.5--20 Hz; unsafe runtime changes to MuJoCo collision size or
  fixed-body pose are deliberately not used.

`RL/student_teacher/gate0_contact_jam.py` establishes nominal guided success
and detects either a sustained contact stall or an in-band force abort. The
local confirmation passed:

```text
command:
  MUJOCO_GL=disable /tmp/aic-gate0-venv/bin/python -m \
    RL.student_teacher.gate0_contact_jam \
    --controller guided --nominal-episodes 2 --randomized-episodes 12 \
    --output RL/output/student_v3/gate0_guided_smoke_final.json

nominal: 2/2 success
randomized: 1 sustained jam
seed: 20260722
stall: 1.26 s, 0.95 mm depth progress, 4.89--5.84 mm depth band,
       >= 8.49 N contact force
final: 5.93 mm insertion depth, 15.11 N, 50 plug-port contacts
```

The remote job is the authoritative 10 nominal / 40 randomized confirmation.
Do not submit a pilot unless both exist:

```bash
test -f /scratch/11590/satya_a/aic/residual_v3_gate0_20260712/GATE0_PASSED
python - <<'PY'
import json
p = "/scratch/11590/satya_a/aic/residual_v3_gate0_20260712/gate0_report.json"
assert json.load(open(p))["passed"] is True
PY
```

## Student-v3 local implementation

New, uncommitted files:

- `student_v3_env.py`: 8x48 deployable history actor observation, 32-D
  privileged critic state, guided acquisition then bounded accumulated residual
  contact control, and contact/stall/recovery reward shaping.
- `student_v3_sac.py`: asymmetric actor/critic SAC policy, zero-initialized
  residual actor, immutable prior plus online RLPD sampling (50:50 -> 20:80),
  and early trusted-prior BC auxiliary loss.
- `student_v3_prior.py`: replay-complete teacher/old-student/failure-boundary
  shard collector. Old Contract-A `(obs, action)` shards are not reused as SAC
  replay because they lack reward, next observation, done, and privileged state.
- `train_student_v3.py`: contact-stage trainer/evaluator.
- `tacc/train_student_v3_pilots.slurm`: one allocation runs seed 0/1 on the
  two GPUs, then seed 2; it hard-requires the Gate-0 JSON pass and teacher hash.

Local verification completed:

```text
6 passed: Gate classifier, v3 residual bounds, asymmetric actor/critic,
          prior replay schema
tiny 12-step RLPD SAC update: passed
Student-v3 MuJoCo wrapper reset/step smoke: passed
bash -n both Slurm scripts: passed
```

The replay-prior collector was also smoke-tested in `/tmp` with 48 transitions:
one teacher-success episode, one old-student-success episode, and one
randomized guided failure-boundary timeout. It produced validated replay-complete
NPZ shards and a manifest; no preserved dataset was changed.

## Resume after re-authentication

Establish a fresh reusable session locally (do not put secrets in chat):

```bash
ssh -fNM -S /tmp/codex-tacc-%r@%h:%p -o ControlPersist=12h \
  satya_a@stampede3.tacc.utexas.edu
```

Then check Gate-0 with:

```bash
ssh -S /tmp/codex-tacc-%r@%h:%p satya_a@stampede3.tacc.utexas.edu \
  'sacct -j 3299028 --format=JobIDRaw,State,ExitCode,Elapsed -n -P; \
   cat /scratch/11590/satya_a/aic/residual_v3_gate0_20260712/gate0_report.json'
```

If it passed, copy the Gate source snapshot to the separate pilot snapshot,
sync the current Student-v3 overlay without `--delete`, collect the prior
manifest under `/scratch/11590/satya_a/aic/residual_v3_prior_20260712`, then
submit `RL/student_teacher/tacc/train_student_v3_pilots.slurm`. The intended
pilot outputs are:

```text
/scratch/11590/satya_a/aic/residual_v3_pilot_seed0_20260712
/scratch/11590/satya_a/aic/residual_v3_pilot_seed1_20260712
/scratch/11590/satya_a/aic/residual_v3_pilot_seed2_20260712
```

If Gate-0 did not pass, do not train. Preserve the report and refine the
contact-mode model first.

## Frozen teacher under Gate-0 randomization (authoritative)

This independent check used the unmodified frozen teacher
`RL/student_teacher/weights/teacher_level1.zip` with SHA-256
`fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4`.
It ran the same Gate-0 regime as the guided-controller confirmation: 10
nominal and 40 randomized episodes, including the contact, friction,
cadence, perception, and grasp randomization. The strict jam predicate was
at least 20 N for at least 1.2 s with less than 1 mm insertion progress in
the 5--9 mm depth band.

```text
run: /scratch/11590/satya_a/aic/teacher_gate0_20260712
nominal success:       10/10 = 100.0%
randomized success:     9/40 = 22.5%
randomized jam:         0/40 = 0.0%  (strict >=20 N predicate)
randomized outcomes:    29 timeout, 2 bad_collision, 9 success
classification:         nominal_regime_only
```

The frozen hash matched before execution and the TACC Pixi environment and
existing snapshots were not changed. The teacher does not reproduce the
specific high-force jam under this stricter predicate, but its randomized
success is far below the 80% hard-contact-valid threshold. Therefore its
demonstrations remain useful for early nominal warm-up only; they are not
hard-contact-expert demonstrations. Do not alter the currently running pilot
for this result. Before continuing the selected seeds to full scale, reduce
the teacher-success prior share from the initial 50% mixture (for example to
25%) and rely more on the student recovery and failure-boundary experience.
Interpret any pilot ranking with this qualification.

## Pilot retry after TensorBoard pre-training failure

Pilot `3299496` completed prior collection but failed before any optimizer step
because Stable Baselines3 was configured with a TensorBoard log directory
while the Pixi environment did not contain `tensorboard`. This was not caused
by the MuJoCo QACC warnings emitted during failure-boundary collection.

The retained immutable prior is complete and hash-verified:

```text
/scratch/11590/satya_a/aic/residual_v3_prior_20260712/manifest.json
60,000 rows: teacher_success 30,000; old_student_success 15,000;
             failure_boundary 15,000
prior_0000.npz sha256 407879800cf480bb702aaec2d9ceacc2e77d02ae9f4f08b974187759f3282309
prior_0001.npz sha256 8a8604b2c520fc253c955f8b5acac2e0ac8b10c625a7db87fbdd6d01b288a8c3
prior_0002.npz sha256 9edfc3bf9e52e09ed0c46d0a65d3d086f5358689bf4d4dfef2bb9694e5fc60a9
```

`tensorboard >=2.9` was added to `pixi.toml`; the regenerated Linux lock
installs TensorBoard `2.21.0`, and the actual TACC Pixi interpreter imports it
successfully with Stable Baselines3 `2.9.0`. The failed pre-training seed
directories were preserved as
`residual_v3_pilot_seed{0,1}_20260712_failed_3299496_tensorboard`.

Replacement pilot `3299715` was submitted against the existing manifest. Its
preflight passed (5 tests), then seeds 0 and 1 entered CUDA training and
created TensorBoard logs. It does not recollect or alter the retained prior;
seed 2 remains sequenced after seeds 0 and 1.
