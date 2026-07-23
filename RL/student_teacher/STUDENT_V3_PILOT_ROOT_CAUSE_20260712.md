# Student-v3 pilot root-cause diagnosis (2026-07-12)

## Decision

Do **not** scale the current Student-v3 design. The primary failure is **(c):
the guided base is unsafe in the randomized training environment because the
hard-contact regime produces numerical ejections**. A secondary reward-sign
bug weakens lateral-error shaping. The evidence does not support “just early at
300k.”

No frozen teacher, snapshot, or Pixi environment was changed during this
diagnosis.

## Residual initialization and bounds: not the cause

- `student_v3_sac.py:64-70` zeroes the deterministic actor mean and initializes
  exploration log standard deviation to `-4`.
- `student_v3_env.py:191-197` clips the accumulated residual to
  `[2.5, 2.5, 6.0] mm` and `[2, 2, 2] deg`. The environment applies the change
  in that accumulator because the underlying command is a pose delta.
- The residual/actor tests pass: `4 passed` for
  `test_student_v3_env.py` plus `test_student_v3_sac.py`.
- An authoritative TACC comparison used 30 identical randomized contact-stage
  episodes. The fresh deterministic actor's maximum absolute action was
  exactly `0.0`; `combined - guided` was exactly `0.0`; the maximum residual
  accumulator was exactly `[0, 0, 0] m`.

The fresh policy and direct guided base therefore produced identical physical
results:

| controller | success | collision | peak lateral mean / p95 | force p95 |
|---|---:|---:|---:|---:|
| direct guided base | 4/30 (13.3%) | 21/30 (70.0%) | 95.5 / 435.9 mm | 9834.6 N |
| fresh zero-init Student-v3 | 4/30 (13.3%) | 21/30 (70.0%) | 95.5 / 435.9 mm | 9834.6 N |

Both runs emitted MuJoCo `QACC` instability warnings. Existing guided Gate-0
traces independently show the same problem: randomized guided control had
61.2 mm mean peak lateral and 306.8 mm p95 versus 0.79 mm nominal. Several
episodes jump by 50--342 mm in one policy step immediately after contact. This
is a numerical contact ejection, not accumulated residual drift.

## Training curves: degrading, not learning

TensorBoard contains return/loss/prior-ratio scalars but does **not** contain
training-time lateral, force, or collision scalars. Those safety trends cannot
be reconstructed from this run; only final evaluation reports them.

| seed | rollout return early / middle / late | actor loss early / late | critic loss early / late |
|---|---:|---:|---:|
| 0 | -219 / -266 / -271 | 5.59 / 64.76 | 14.48 / 39.45 |
| 1 | -264 / -280 / -277 | 5.68 / 69.56 | 13.53 / 43.69 |
| 2 | -223 / -250 / -211 | 4.01 / 56.95 | 13.17 / 41.73 |

Seed 0 final evaluation was 30% success, 53.3% collision, 71.4 mm mean peak
lateral, and -109.6 return. Seed 1 was 0% success, 46.7% collision, 161.3 mm
mean peak lateral, and -359.7 return. Seed 2 was 6.7% success, 63.3%
collision, 206.1 mm mean peak lateral, 212.2 N force p95, and -379.1 return.
Pilot job `3299715` completed normally after all three evaluations.

The seed 0/1/2 training logs contain 18/20/26 `QACC` warnings respectively.
Returns do not consistently improve, and actor/critic losses grow
substantially. The generated selection file lists seeds 1 and 0, but that
ranking is invalid for continuation: it sorts collision rate before success
and therefore ranks a 0%-success seed first. No full-scale job is active.

## Reward signs

The base reward signs are correct: depth progress is positive; XY, axis, force,
lateral load, collision, and action changes are negative; failure terminals
are negative and success is positive (`RL/reward.py:255-266`). Student-v3 force,
residual magnitude, residual change, stall, and large-retreat terms also have
the intended sign.

There is one real sign bug at `student_v3_env.py:232-235`:

```python
shaping -= 0.7 * breakdown.xy
```

`breakdown.xy` is already negative, so this adds a positive refund and removes
70% of the base XY penalty during plug-port contact. It does not explain the
fresh zero-residual physical ejections, but it makes learned lateral behavior
less constrained and must be corrected before another pilot.

## Prior replay

The prior is not dominated by failure trajectories:

- raw rows: 30k nominal teacher success, 15k nominal old-student success, 15k
  randomized guided failure boundary;
- category weighting makes only 13.89% of prior samples failure-boundary rows
  and 86.11% BC-eligible nominal success/recovery rows;
- the actual prior:fresh ratio anneals as coded from about 0.50 to 0.20, making
  failure-boundary rows only about 6.94% of the full batch initially and 2.78%
  at 300k;
- all failure-boundary replay actions are zero.

The contamination risk is instead stale nominal success replay. Teacher rows
are 50% of the raw prior even though the teacher achieved only 22.5% under hard
randomization. Source identity is not stored per transition, so the current
sampler cannot separately downweight teacher versus old-student success rows.

## Required gate before another pilot

1. Stabilize the randomized contact model until guided-only episodes no longer
   show `QACC` warnings or one-step 50--342 mm ejections, and add an immediate
   spatial/off-limit termination so an invalid simulation cannot continue.
2. Fix the XY shaping sign and log lateral, force, collision, residual
   accumulator, and outcome rates throughout training.
3. Preserve a per-transition source ID and downweight the nominal-only teacher
   prior as previously decided.
4. Re-run the fresh zero-residual/guided baseline gate before any new 300k pilot.
   Do not submit a 2--3M run from the current results.
