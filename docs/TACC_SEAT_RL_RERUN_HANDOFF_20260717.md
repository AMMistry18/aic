# TACC seat-RL rerun handoff — 2026-07-17

## 2026-07-18 execution update (supersedes the pre-implementation plan below)

The required reset-mixture, curriculum, checkpoint-selection, and evaluator
changes are implemented, tested, committed, and pushed. The source of truth for
all new work is:

```text
branch: flowstate-rl-deploy-and-docs
pipeline commit: fe1bc1a3cfa1837773be6318a5f7727c033e36f6
remote checkout: /work2/11590/satya_a/stampede3/aic-seat-rerun-20260718-cb0a209
TACC user/account: satya_a / IRI26004
partition: rtx-small
```

Do not reimplement the older clean/mild/wedge proposal later in this document.
It was replaced after examining the deployment handoff logs and measuring what
the simulator actually delivers. The implemented deployment-stage mixture is:

| Reset class | Probability | Depth | Actor-visible lateral error |
| --- | ---: | ---: | ---: |
| `live_shallow` | 70% | 3–8 mm | 0.9–2.7 mm |
| `centered_shallow` | 15% | 3–8 mm | 0.0–0.9 mm |
| `mid_tail` | 10% | 8–22 mm | 0.9–2.7 mm |
| `mastered_deep` | 5% | 22–42 mm | 0.0–1.2 mm |

The bootstrap stage uses synchronized reverse curriculum over depth. It starts
with a 42 mm easy boundary and samples down to the hard 3 mm boundary. Across a
shared 100-episode window, success above 80% moves the easy boundary 5 mm toward
the hard boundary; success below 10% moves it 3 mm back. The boundary is clamped
to 8–42 mm and persisted in `curriculum_state.json`.

Other implemented safety and selection behavior:

- The 5-degree deployment rotation guard is applied during training and fixed
  evaluation, with an explicit `rotation_guard` termination and failure cost.
- Periodic evaluation uses the same frozen 60-case suite for every checkpoint;
  final selection uses a frozen 180-case suite. Seeds do not depend on training
  timestep, and each suite balances all three compiled contact geometries both
  overall and within every reset class.
- Checkpoints are versioned and resume-compatible. `best_model.zip` is ranked by
  success, then safety-failure rate, then p95 force, and exports to
  `best_seat_actor.ts` with numerical parity validation.
- `evaluate_seat_checkpoints.py` compares SB3 checkpoints and TorchScript actors
  on the identical frozen cases.
- `live_metrics.jsonl` preserves update ratio, reset class, compiled contact
  variant, termination, and reward telemetry independently of W&B. The smoke
  writes `smoke_gate.json` and fails unless update:data is 0.9–1.1 and every
  required reset class and compiled contact variant was observed.
- The deploy-side action contract now matches training's seat-action scaling,
  port-frame residual accumulation, residual clamp, and 1-degree yaw clip.

Local validation completed at this commit: 28 focused tests passed, Python and
Slurm syntax checks passed, forced examples of all four reset classes passed,
and an online W&B end-to-end plumbing smoke produced versioned/best/final
artifacts. That tiny random-policy smoke is not policy-quality evidence.

### W&B routing

W&B rejects direct run logging to the organization slug. The authenticated
writable run entity is the `tar2` team, while the requested organization is
recorded in configuration and run metadata:

```text
WANDB_ENTITY=tar2
WANDB_ORGANIZATION=anshulmistry1-the-university-of-texas-at-austin-org
WANDB_PROJECT=aic-seat-rl
WANDB_MODE=online
```

Online authentication check:
`https://wandb.ai/tar2/aic-seat-rl/runs/kne12nvy`

Local end-to-end plumbing smoke:
`https://wandb.ai/tar2/aic-seat-rl/runs/4lp5xqd0`

The checked-in launchers require online W&B and fail instead of silently
falling back offline.

### TACC environment and live queue state

The fresh checkout's `pixi.lock` and `pixi.toml` exactly match the previously
validated environment at
`/work2/11590/satya_a/stampede3/aic-seat-rl-20260713/.pixi`. A fresh login-node
Pixi backend build exceeded login-node memory/thread limits, so the new checkout
uses a symlink to that lock-identical environment and launchers set
`SKIP_PIXI_INSTALL=1`. Do not remove the symlink or retry the ROS build on a
login node.

The mandatory 16-environment smoke is submitted:

```text
job: 3324435
name: seat-smoke16
source commit: fe1bc1a3cfa1837773be6318a5f7727c033e36f6
stdout: /scratch/11590/satya_a/aic/slurm/seat-smoke16-3324435.out
stderr: /scratch/11590/satya_a/aic/slurm/seat-smoke16-3324435.err
```

It is pending with `QOSMaxJobsPerUserLimit` because unrelated job `3319555`
(`simdist2`) already occupies the account's single running GPU-job slot. Do not
cancel or alter `3319555`. The QOS also prevents a second pending submission, so
job `3324435` runs the 16-env smoke and then the identical 8-env comparison
sequentially inside the same allocation. Monitor with:

```bash
squeue -j 3324435 -o '%.18i %.12P %.28j %.8u %.2t %.10M %.10l %.6D %R'
tail -n 100 -f /scratch/11590/satya_a/aic/slurm/seat-smoke16-3324435.out
sacct -j 3324435 --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

The comparison writes its separate artifacts under
`/scratch/11590/satya_a/aic/seat_smoke_8env_3324435`. Select 16 workers only if
the smoke is stable, the update:data ratio is
0.9–1.1, and its useful post-warmup throughput is at least 1.25 times the
identical 8-worker smoke. Test 12 only if 8 versus 16 leaves the choice unclear.
Run seed 0 through the first-hour gate before submitting seeds 1 and 2.

### Deploy candidate staging

The actor-enabled deploy integration is staged separately in the local
`aic-board-search` worktree at commit
`617970c7be5f2141979a7abfbec7e797a6d6db1b`. It is intentionally not pushed or
deployed yet. Preserve the current script-only runtime until a trained actor
wins the fixed held-out evaluation and passes guarded local/Flowstate checks.

---

The remainder of this document is the original pre-implementation handoff. It
is retained as provenance; wherever it conflicts with the execution update
above, the update above wins.

This is the execution handoff for the next Codex agent and the person running
the job on their own TACC account. The objective is a new plain-SAC seat policy
that continues straight down when the plug is already centered in the port but
can still recover mild and hard lateral wedges.

The warning below described the trainer before the 2026-07-18 implementation.
Those changes are now complete; use the execution update above for launch work.

## Source and immutable boundaries

- Repository: `https://github.com/AMMistry18/aic.git`
- Branch: `flowstate-rl-deploy-and-docs`
- Seat environment: `RL/student_teacher/seat_env.py`
- Trainer: `RL/student_teacher/train_seat.py`
- Full launcher: `RL/student_teacher/tacc/train_seat.slurm`
- Worker smoke: `RL/student_teacher/tacc/smoke_train_seat.slurm`
- Previous run for comparison only: TACC job `3305828`, W&B run
  `https://wandb.ai/tar2/aic-seat-rl/runs/1t7ucme9`

Do not modify the calibrated contact physics in `RL/scene_env.py`. Keep this
lane plain Stable-Baselines3 SAC: no teacher policy, BC loss, RLPD prior, or
demonstration replay. Do not deploy any new policy from TACC; return the
artifacts for held-out review and Flowstate deployment separately.

## Why this rerun is needed

The previous full actor was trained almost entirely as a wedge-recovery policy.
In the latest real run, the script handed it a plug that was already centered
and about 4.6 mm into the port. The actor then increased lateral error instead
of simply seating the plug. More rollout workers alone will not correct that
training-distribution error.

There was also no fair checkpoint comparison: the evaluation seed included the
current training timestep, so every checkpoint saw a different test set. The
periodic evaluation used only eight episodes in the submitted job, and the
final `model.zip` was deployed even though an earlier checkpoint may have been
better.

## Required implementation before the full run

The next Codex agent owns these changes and must add focused tests for them.

### 1. Train the correct handoff mixture

Preserve the current actor/critic/action ABI and the three calibrated full-stage
contact geometries. Change reset generation so the learner sees these explicit
start classes:

| Class | Delivered lateral error | Delivered tilt | Nominal mixture | Full mixture | Intended behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| clean | 0.00–0.25 mm | 0.00–0.25 deg | 70% | 50% | Continue inward without lateral searching |
| mild | 0.25–0.65 mm | 0.25–0.50 deg | 30% | 30% | Small compliant centering correction |
| wedge | 0.65–1.00 mm | 0.50–1.00 deg | 0% | 20% | Existing validated lateral-unstick behavior |

Requirements:

- Full-stage depth coverage must retain all three currently measured handoffs:
  shallow approximately 4.5–7.5 mm, middle 24–30 mm, and deep 39–42 mm.
- Balance `+X`, `-X`, `+Y`, and `-Y` for non-clean offsets.
- Do not pass clean or mild states through a validator that requires them to be
  a true lateral wedge. Give each class its own safety/behavior acceptance rule.
- A clean start must be safe, in contact where appropriate, and make inward
  progress under the existing guided straight action without lateral growth.
- A mild start must be safe and recoverable with a bounded centering action.
- A wedge start must retain the current straight-stall plus lateral-unstick
  proof and all unsafe-probe rejection.
- Record `seat_reset_class` in `info` and W&B. Never hardcode
  `seat_reset_true_lateral_wedge=True` for clean or mild starts.
- Measure at least 30 delivered resets per class and per compiled contact
  variant. Report actual depth, lateral error, tilt, force, contact count,
  fallback rate, and unsafe rejection rate. Nominal constants are not proof.

### 2. Make checkpoint evaluation comparable

- Remove `num_timesteps` from the evaluation seed. Every checkpoint must run the
  same frozen held-out cases.
- Stratify the suite over clean/mild/wedge, shallow/middle/deep, and all three
  compiled contact variants.
- Use 45 fixed periodic episodes: five per reset-class/depth stratum with the
  compiled variants rotated evenly. Run this every 40,000 training steps.
- Run a separate fixed 180-episode selection evaluation at the end: 20 cases
  per reset-class/depth stratum, balanced across contact variants.
- Log success, jam, bad-collision, force-abort, maximum force, final lateral
  error, and steps-to-seat both overall and per reset class.
- Treat any evaluation exception or missing stratum as a failed evaluation, not
  as permission to use the final checkpoint.

### 3. Preserve and select checkpoints

- Save versioned policy checkpoints such as
  `checkpoints/model_000040000_steps.zip`; do not only overwrite `model.zip`.
- Keep `model.zip` and `replay_buffer.pkl` for resume compatibility.
- Maintain `best_model.zip`, `best_evaluation.json`, and the exact best training
  timestep. Selection priority is: highest success, then lowest combined
  bad-collision/force-abort rate, then lower p95 force.
- Export `best_seat_actor.ts` from `best_model.zip`. Do not automatically call
  the final 400k full-stage actor the deploy candidate.
- Parameterize the full launcher with `SEED`, `LR_NOMINAL`, and `LR_FULL` so
  seeds and stage-specific learning rates do not require editing the script.

## Connect the friend's TACC and W&B accounts

Replace angle-bracket placeholders locally; never commit credentials.

```bash
ssh <TACC_USER>@stampede3.tacc.utexas.edu
printf 'USER=%s\nWORK=%s\nSCRATCH=%s\n' "$USER" "$WORK" "$SCRATCH"
```

Determine the allocation and GPU partition that this user is actually allowed
to charge. Do not copy `IRI26004` blindly from Satya's launcher. TACC's current
Stampede3 guide is `https://docs.tacc.utexas.edu/hpc/stampede3/`; it requires
compute work to be submitted through Slurm and currently rejects GPU GRES flags.

Clone a fresh source tree in `$WORK`:

```bash
export PROJECT_DIR="$WORK/aic-seat-rerun-20260717"
git clone --branch flowstate-rl-deploy-and-docs --single-branch \
  https://github.com/AMMistry18/aic.git "$PROJECT_DIR"
cd "$PROJECT_DIR"
git pull --ff-only
git status --short
git rev-parse HEAD
```

If the repository is private, the friend must first be granted GitHub access
and authenticate using their own GitHub credential. Do not send or store another
person's token.

Install the pinned workspace environment:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
pixi install --locked
export PYTHONPATH="$PROJECT_DIR"
```

For W&B, the friend needs membership in the `tar2` team if the run should appear
beside the previous run. Obtain a personal W&B key from User Settings, then use
the interactive login so the key is not placed in the repository, Slurm file,
shell command arguments, chat, or logs:

```bash
cd "$PROJECT_DIR"
pixi run --as-is wandb login --verify
pixi run --as-is wandb status
export WANDB_ENTITY=tar2
export WANDB_PROJECT=aic-seat-rl
export WANDB_MODE=online
```

W&B documents both interactive login and `WANDB_API_KEY`, but API keys must be
treated as passwords and never committed. If the friend cannot join `tar2`, log
to their own entity under project `aic-seat-rl` and send the run URL. The trainer
falls back to offline mode if online initialization fails; later sync the exact
offline directory printed in `train.log` with:

```bash
pixi run --as-is wandb sync <offline-run-directory>
```

Official references:

- `https://docs.wandb.ai/models/ref/cli/wandb-login`
- `https://docs.wandb.ai/platform/app/settings-page/user-settings`

## Mandatory 16-environment feasibility test

Do this before the full run and do it on a compute node, never on a login node.
First finish the reset/evaluation/checkpoint implementation above and run its
focused unit tests. Then submit the checked-in smoke with at least 20 CPU cores:

```bash
cd "$PROJECT_DIR"
mkdir -p "$SCRATCH/aic/slurm"

export TACC_ACCOUNT=<FRIEND_TACC_ALLOCATION>
export TACC_GPU_PARTITION=<FRIEND_GPU_PARTITION>
export PROJECT_DIR
export NUM_ENVS=16
export SMOKE_STEPS=20000
export SMOKE_EVAL_EPISODES=9
export SMOKE_TAG="seat_smoke_16env_$(git rev-parse --short HEAD)"
export SMOKE_OUT="$SCRATCH/aic/$SMOKE_TAG"

sbatch \
  -A "$TACC_ACCOUNT" \
  -p "$TACC_GPU_PARTITION" \
  --exclusive \
  --cpus-per-task=20 \
  -J seat-smoke16 \
  -o "$SCRATCH/aic/slurm/seat-smoke16-%j.out" \
  -e "$SCRATCH/aic/slurm/seat-smoke16-%j.err" \
  RL/student_teacher/tacc/smoke_train_seat.slurm
```

Monitor it:

```bash
squeue -u "$USER"
tail -f "$SCRATCH/aic/slurm/seat-smoke16-<JOB_ID>.out"
sacct -j <JOB_ID> --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

The 16-env smoke passes only if:

- all 16 workers reset and step without worker death, OOM, NaN, or exception;
- CUDA is visible and SAC updates run on the GPU;
- `train/update_data_ratio` settles between 0.9 and 1.1 after learning starts;
- all three reset classes and contact variants appear in telemetry;
- no class silently falls back to a different class;
- the run produces `model.zip`, `evaluation.json`, `seat_actor.ts`, and
  `smoke_meta.txt`.

After 16 environments pass, repeat the identical smoke with `NUM_ENVS=8` and a
new `SMOKE_TAG`. Compare post-warmup W&B `time/fps` and `smoke_meta.txt`. Use 16
for the full run only if it is stable and at least 1.25x the useful throughput
of 8 while retaining an update:data ratio near 1.0. Otherwise test 12 or return
to 8. More workers are not a goal by themselves.

## Full run specification

After all implementation tests and the worker-count gate pass:

| Setting | Required value |
| --- | --- |
| algorithm | plain asymmetric SB3 SAC |
| seeds | 0, 1, and 2 |
| environments | winner of the 8/12/16 smoke gate |
| nominal stage | 200,000 steps, learning rate `3e-4`, 70/30 clean/mild |
| full stage | 400,000 steps, warm-start from nominal best, learning rate `1e-4`, 50/30/20 clean/mild/wedge |
| gradient steps | `-1` with `train_freq=(1, "step")` |
| batch/buffer | 256 / 500,000 |
| checkpoint | every 20,000 steps, versioned |
| fixed periodic evaluation | 45 episodes every 40,000 steps |
| fixed final selection | 180 episodes |
| video | every 50,000 steps |
| W&B | online with offline fallback; report every run URL |

Parameterize `SEED`, `LR_NOMINAL`, and `LR_FULL` in the launcher before using
the following pattern. Submit seed 0 first. Start seeds 1 and 2 only after seed
0 proves the reset mix, W&B telemetry, checkpoint preservation, and walltime
projection are healthy.

```bash
cd "$PROJECT_DIR"
export PROJECT_DIR
export NUM_ENVS=<PASSED_ENV_COUNT>
export STEPS_NOMINAL=200000
export STEPS_FULL=400000
export LR_NOMINAL=3e-4
export LR_FULL=1e-4
export GRADIENT_STEPS=-1
export CHECKPOINT_FREQ=20000
export EVAL_FREQ=40000
export EVAL_EPISODES=45
export VIDEO_FREQ=50000
export WANDB_LOG_FREQ=1000
export BUDGET_FPS_FLOOR=<MEASURED_CONSERVATIVE_FPS>

for SEED in 0 1 2; do
  export SEED
  export TAG="seat_mix_seed${SEED}_$(git rev-parse --short HEAD)"
  sbatch \
    -A "$TACC_ACCOUNT" \
    -p "$TACC_GPU_PARTITION" \
    --exclusive \
    --cpus-per-task=$((NUM_ENVS + 4)) \
    -J "seat-mix-s${SEED}" \
    -o "$SCRATCH/aic/slurm/seat-mix-s${SEED}-%j.out" \
    -e "$SCRATCH/aic/slurm/seat-mix-s${SEED}-%j.err" \
    RL/student_teacher/tacc/train_seat.slurm
done
```

Do not submit all three simultaneously unless the friend's allocation permits
three exclusive GPU nodes and seed 0 has already passed the health gate. TACC's
maximum individual job runtime is 48 hours, so use the measured smoke throughput
to confirm the 600k-step curriculum plus evaluation fits. If it does not fit,
resume from the saved `model.zip` in a dependent second job; do not reduce the
held-out suite or update:data ratio to hide the walltime problem.

## First-hour health gate

For seed 0, capture and report:

- Slurm job ID and exact git SHA;
- W&B URL and whether it is online or offline;
- measured `time/fps` and projected completion time;
- `train/update_data_ratio` (target 0.9–1.1);
- observed clean/mild/wedge reset proportions;
- reset pool/fallback/rejection rates per class;
- termination mix and reward-component means;
- GPU visibility/utilization and worker memory;
- confirmation that versioned checkpoints and the frozen evaluation suite are
  being used.

Stop the run instead of burning allocation if workers die, resets collapse into
one class, the update:data ratio is wrong, evaluation cases change between
checkpoints, forces become non-finite, or the projected runtime exceeds the
available walltime without a tested resume path.

## Acceptance and return artifacts

Do not choose a deploy candidate from training reward alone. A candidate must:

- pass the fixed 180-episode suite with results reported overall and by reset
  class/contact variant;
- show no systematic lateral motion from clean centered starts;
- have zero bad-collision and force-abort events in the clean stratum;
- beat the old full actor on both success and p95 force;
- remain finite under TorchScript export with max parity error at most `1e-6`;
- reproduce its selection metrics when the saved best checkpoint is reevaluated.

Return these together for review:

```text
best_model.zip
best_seat_actor.ts
best_seat_actor.ts.contract.json
best_evaluation.json
evaluation_history.jsonl
config.json
progress.json
train.log
submitted.slurm
live_metrics.jsonl
smoke_gate.json for 8 and 16 environments
smoke_meta.txt for 8 and 16 environments
exact git SHA, Slurm job IDs, W&B URLs, and scratch paths
```

The final decision to deploy remains on the local Flowstate side after reviewing
these artifacts and running guarded real insertions.
