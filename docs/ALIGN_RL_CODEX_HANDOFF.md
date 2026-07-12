# Align-RL — Codex handoff: submit the TACC training run

Created 2026-07-12. Updated: the env, trainer, AND slurm are all WRITTEN and
locally smoke-checked (py_compile passes; trainer↔env info-key contract verified).
YOUR ONLY JOB: sync the repo to `$WORK/aic-align-rl-20260712` on TACC and submit
`RL/student_teacher/tacc/train_align.slurm`, then report results. Do NOT redesign
the env, reward, trainer, or slurm — they are settled. Ask before deviating.

## Files already written (do NOT modify)

- `RL/student_teacher/align_env.py` — the align-only env (reward + termination).
- `RL/student_teacher/train_align.py` — plain-SAC trainer (no RLPD prior; reuses
  `AsymmetricSACPolicy` from `student_v3_sac.py`, which is dim-agnostic;
  align-metric `_evaluate` reading `info["align_term_status"]`).
- `RL/student_teacher/tacc/train_align.slurm` — SBATCH job: compile gate →
  SMOKE run (`--steps 2000 --num-envs 2`, must emit evaluation.json) → FULL run
  (3 seeds, `--stage align`, 300k steps, one per GPU) → rank seeds by align
  success. Mirrors `train_student_v3_pilots.slurm` scaffolding, minus Gate-0 /
  teacher-SHA / prior collection.

## What Codex does

1. Sync/checkout this repo into `$WORK/aic-align-rl-20260712` (mirror how the v3
   slurm's `PROJECT_DIR` is populated). Ensure the three files above are present.
2. `sbatch RL/student_teacher/tacc/train_align.slurm`.
3. The slurm's SMOKE step is the MuJoCo gate — if it fails (shape error, no
   evaluation.json), STOP and report the traceback; do not force the full run.
4. Report back: smoke evaluation.json, full-run job ID + out dirs
   (`$SCRATCH/aic/align_rl_seed{0,1,2}_20260712/`), and the first ~50k-step
   progress/eval so we can see if the reward is learnable early.

---

## Reference (design context — NOT action items)

The material below documents the design so Codex understands what it is running.
The trainer and slurm already implement all of it.

## What the align RL is (context)

Align-first pipeline (`docs/INSERTION_PIPELINE_DESIGN.md`). This RL's ONLY job is
to SQUARE the plug over the perceived port (fix lateral x/y + rotation
roll/pitch/yaw, HOLD z at a standoff). It does NOT insert — once it reports
"aligned", the base script takes over the slow descent. So there is NO depth
reward. Success == alignment tolerance reached.

This is a DIFFERENT env from `student_v3_env.py` (which is the residual-at-contact,
depth-rewarded policy). Leave v3 untouched; it stays as a fallback.

## The env (already written — do not modify)

`RL/student_teacher/align_env.py`
- `make_align_env(stage, *, seed, domain_randomization=True) -> AlignEnv`
  - stages: `"align"`, `"small"`, `"medium"`, `"full"`, `"robust"` (see `STAGES`).
- Obs is a Dict: `{"actor": (HISTORY=8, ACTOR_FRAME_DIM=34), "privileged": (32,)}`
  — SAME asymmetric structure as v3, only the actor frame dim differs (34 vs 48).
- Action: `Box(-1,1,(6,))` — the RL IS the whole controller (deploy convention,
  no guided base, no residual accumulator). It directly commands 6-DoF deltas.
- Reward (in `_align_reward` / `_timeout_penalty`): lateral+rotation error
  REDUCTION (progress-shaped) + gentle hold penalty; z held at `Z_STANDOFF_M`;
  descend-while-misaligned penalty; +50 on aligned; -20 on base-env hard failure;
  proximity-scaled timeout penalty (0 near-aligned → 20 far-off).
- Termination: `terminated=True` with `info["align_term_status"]=="aligned"` when
  aligned (lat<1mm AND rot<1.5°); base-env hard failures (bad_collision/
  force_abort/off_limit) also terminate; timeout → `truncated=True`.
- **Success metric for eval: `info["align_term_status"] == "aligned"`.** Not the
  base env's `term_status` (that requires seating, which this RL never does).

Verified locally: syntax parses, frame dim (34) consistent, timeout penalty
scales correctly. NOT run in MuJoCo yet (no pixi env on the dev Mac) — your first
job on TACC is a smoke run.

## Reuse from v3 (do NOT rewrite)

`RL/student_teacher/student_v3_sac.py`:
- `AsymmetricSACPolicy`, `ActorHistoryExtractor`, `PrivilegedCriticExtractor` —
  these read dims DYNAMICALLY from `observation_space`, so they work with the
  align env's 34-dim frame with ZERO changes. Reuse them directly.

## Trainer: create `RL/student_teacher/train_align.py`

Copy `train_student_v3.py` as the template, then make these changes:

1. **Drop the RLPD prior entirely.** The align RL trains FROM SCRATCH — there is
   no guided base to distill a prior from. So:
   - Remove `--prior-manifest`, `PriorReplayDataset`, `PriorReplayDataset(...)`.
   - Use plain `stable_baselines3.SAC` (not `RLPDSAC`) with `AsymmetricSACPolicy`.
   - `learning_starts` should be a real warmup now (e.g. 5000), NOT 0 (0 only made
     sense because RLPD pre-filled the buffer with priors).
2. **Swap the env factory:** `from RL.student_teacher.align_env import make_align_env`
   and call `make_align_env(seed=..., stage=args.stage, domain_randomization=True)`.
3. **Stage choices:** `("align","small","medium","full","robust")`, default `"full"`.
4. **Rewrite `_evaluate`** to report ALIGN metrics from `info["align_term_status"]`:
   - `align_success_rate` = fraction ending `"aligned"`
   - `fail_rate` = fraction ending in bad_collision/force_abort/off_limit
   - `timeout_rate`
   - `final_lat_err_p50_mm`, `final_rot_err_p50_deg` (from `align_lat_err_m` /
     `align_rot_err_rad`) — median final errors
   - `steps_to_align_mean` (episode length on aligned episodes)
   Keep it simple; these drive stage promotion decisions.
5. Keep: asymmetric policy, checkpointing, wandb (project `"aic-align-rl"`),
   SubprocVecEnv, resume logic. Keep hyperparams (lr 3e-4, gamma 0.99, tau 0.005,
   net_arch [256,256], `share_features_extractor=False`, ent_coef auto).

## TACC slurm: create `RL/student_teacher/tacc/train_align.slurm`

Model it on `train_student_v3_pilots.slurm` but SIMPLER (no prior collection, no
Gate-0 gate). Keep the proven scaffolding:
- Same SBATCH header (account IRI26004, partition rtx-small, the `%x-%j` out/err
  paths, `MUJOCO_GL=egl`, `WANDB_MODE=offline`, `OMP_NUM_THREADS=1`,
  `PATH=$HOME/.pixi/bin:$PATH`).
- Sync/checkout the repo into a fresh `$WORK/aic-align-rl-20260712` project dir
  (mirror how the v3 slurm sets `PROJECT_DIR`, `PYTHONPATH`).
- Teacher-weight SHA guard is NOT needed (align RL doesn't use the teacher policy),
  but `build_teacher_obs21` IS imported for the privileged critic — that's a pure
  state function, needs no weights. So no SHA check required.
- **First: a SMOKE step** — `pixi run --as-is python -m py_compile
  RL/student_teacher/align_env.py RL/student_teacher/train_align.py` then a tiny
  `--steps 2000 --num-envs 2` run to a throwaway dir to confirm it steps in MuJoCo
  and produces an evaluation.json. Only proceed to the full run if smoke passes.
- **Full run:** 2-3 seeds, `--stage align` first (contact-level curriculum),
  `--steps 300000 --num-envs 12`, one per GPU like the v3 slurm's `run_seed`.
  Write outputs under `$SCRATCH/aic/align_rl_seed${seed}_20260712/`.

## Physics + perception status (why now)

- Sim contact physics was just stabilized (ridge solref, QACC ejection fixed) —
  see `RL/student_teacher/MUJOCO_CONTACT_PHYSICS_CALIBRATION_20260712.md`. The
  align RL depends on stable contact, so this is the right time.
- Wrong-port perception fix is applied in `aic_model/aic_model/RLInsert.py`
  (deployment side) — irrelevant to sim training, but it's the deploy prerequisite.

## What to report back

1. Smoke result (did it step in MuJoCo? shapes OK? any errors).
2. Full-run kickoff confirmation (job IDs, seeds, out dirs).
3. First eval after ~50k steps: align_success_rate + median final lat/rot errors,
   so we can see if the reward is learnable before burning the full budget.

## Guardrails (verbatim, preserve)

- Do NOT modify/delete the frozen teacher
  `RL/student_teacher/weights/teacher_level1.zip`
  (sha256 fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4).
- Do NOT delete TACC datasets, snapshots, teacher weights, training outputs, or
  the Pixi environment.
- No secrets/tokens in git/chat/scripts.
- Do NOT change the align env's reward or termination design — it is settled.
