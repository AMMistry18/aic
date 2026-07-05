# Curriculum + Entropy Tuning for Last-Inch Insertion SAC

Research-grounded tuning guide for making the reverse curriculum actually climb
(2026-07-04). Written against the observed failure data of the two training
runs (`RL/output/residual_sac_sfp_curriculum/metrics.jsonl`): 0.05↔0.10
oscillation, `ent_coef` collapsing 1.0 → 0.07 within ~30k steps, and returns
plateauing while per-level success kept improving.

---

## 0. How our training actually parallelizes (common confusion)

`batch_size=1024` is **not** 1024 parallel runs. The data flow is:

- **6 parallel `SceneInsertEnv` workers** (`--num-envs 6`, SubprocVecEnv) step
  simultaneously → 6 transitions per vec step go into ONE shared **replay
  buffer** (`--buffer-size 20000` transitions ≈ the last ~20k env steps).
- Every vec step, SAC does `--gradient-steps 4` updates, each on a minibatch of
  **1024 transitions sampled uniformly at random from the buffer** (UTD =
  4/6 ≈ 0.67 updates per env step… with 6 envs: 4 updates per 6 env steps
  = 0.67; at 16 envs it would be 0.25).
- A *successful* episode therefore is not "exploited once": its transitions
  sit in the buffer for ~20k steps and are expected to be resampled
  `batch × grad_steps / buffer ≈ 1024×4/20000 ≈ 0.2` times per vec step —
  ~hundreds of gradient contributions before they age out.

Consequences: (a) exploitation of successes is controlled by **buffer
composition + sampling**, not by batch size; (b) with a 20k buffer, data from
a level the curriculum left ~20k steps ago is *gone* — this is the real
"forgetting" channel, not the network (nothing in the network resets between
levels).

---

## 1. What the papers say, mapped to our failure data

### 1.1 Reverse curriculum: train in the success band, never re-train solved levels
**Florensa et al., "Reverse Curriculum Generation for RL" (CoRL 2017,
[arXiv:1707.05300](https://arxiv.org/abs/1707.05300)).** Their key-insertion
task is directly analogous to ours. The algorithm trains only on **"good
starts"**: states where the current policy's success rate is *strictly
between* `R_min` and `R_max` — "samples more heavily nearby the start states
that need more training to be mastered and avoiding start states that are yet
too far to receive any reward under the current policy." Mastered starts
(`R > R_max`) are **dropped from the training set**; to avoid catastrophic
forgetting they re-append `N_old` starts sampled from a replay list of
previous good starts.

Mapping to us:
- **"If the lower level already succeeds, going back changes nothing"** — this
  is exactly Florensa's drop-mastered-starts rule. Our scheduler's old
  unsafe-retreat rule violated it (retreated to 0.05 at 89% success). Fixed on
  disk: retreat fires **only** when success ≤ `retreat_threshold` (0.15 ≈
  their `R_min` 0.1); collision/abort rates only gate *advancement*.
- Our frontier band (`curriculum_band=0.25`) + easy replay
  (`curriculum_easy_frac=0.2`) is the sanctioned analogue of their
  `SampleNearby` + `N_old` replay. Keep both.
- Their `R_min/R_max` interpretation says our `advance_threshold=0.6` is
  conservative-but-fine; the important part is **holding at the frontier**
  when success is between 0.15 and 0.6 — which the fixed scheduler now does
  (that's where the learning happens).

### 1.2 Entropy: fixed low target ⇒ premature determinism; schedule it
**Xu et al., "Target Entropy Annealing for SAC" (TES-SAC,
[arXiv:2112.02852](https://arxiv.org/abs/2112.02852))**: with a fixed target
entropy set low, "the policy entropy will quickly drop and become overly
deterministic in an early stage of training" — precisely the `ent_coef`
1.0→0.07 collapse you observed. They anneal target entropy high→low over
training.

**"Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary RL"
([arXiv:2601.19624](https://arxiv.org/html/2601.19624v2))**: when the
environment (here: the start distribution) shifts, a stationary entropy
setting either wastes samples or under-explores; entropy should be
**scheduled off drift signals**. A curriculum advance is a *known, discrete
drift event* — we don't need to detect it, we cause it.

Prescription (implemented as `--ent-reheat`): **on every curriculum advance,
re-heat alpha** — `log_ent_coef := log(max(alpha_now, reheat))` with
`reheat ≈ 0.15` (≈2× the converged 0.07) — and let SAC's auto-temperature
re-anneal. Optionally also anneal the *target* per level
(`target_H(level) = −1.5 − 1.5·level`), which is TES-SAC's schedule keyed to
curriculum progress instead of wall-clock. The expected `ent_coef` curve
becomes a **sawtooth**: decay within a level, jump at each advance. A
monotone-decaying curve after this change means advances stopped, not that
exploration is healthy.

### 1.3 Exploiting successes harder: success-biased replay
**Oh et al., "Self-Imitation Learning" (ICML 2018,
[arXiv:1806.05635](https://arxiv.org/abs/1806.05635))**: learning to
reproduce the agent's own past good trajectories "indirectly drives deep
exploration" — exploitation of successes is itself an exploration mechanism.
**Vecerík et al. (DDPGfD, [arXiv:1707.08817](https://arxiv.org/abs/1707.08817))**
— on real **peg/clip insertion** — keep demonstration/valuable transitions in
the replay buffer with elevated sampling priority so they are never crowded
out.

Mapping to us (implemented as `--success-mix`): keep a **side buffer of
transitions from successful episodes** (they survive after the main 20k ring
overwrites them) and compose each minibatch as `(1−ρ)` uniform-main +
`ρ` success-buffer, `ρ ≈ 0.2–0.3`. This is the "higher degree of exploitation
off good episodes" you asked for, without the instability of copying SIL's
extra loss. It also fixes cross-level transfer through the buffer: successes
from level L keep training the critic while the curriculum works on L+1.

### 1.4 If you crank exploitation (UTD), pair it with resets
**Nikishin et al., "The Primacy Bias in Deep RL" (ICML 2022,
[PDF](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf))** and
**D'Oro et al. (ICLR 2023)**: high replay ratios overfit early data; periodic
re-initialization of the last layers (keeping the buffer) fixes it — SAC at
replay ratio 32 *with resets* beats low ratios. Our UTD is modest (0.67 at 6
envs), so this is **not** needed now; it becomes relevant only if you raise
`--gradient-steps` ≥ 2×. **Reset & Distill
([arXiv:2403.05066](https://arxiv.org/html/2403.05066v1))** likewise supports
occasional actor resets under task shift. Keep in the back pocket for
plateaus; don't apply preemptively.

### 1.5 Forgetting across levels
**TeachMyAgent ([arXiv:2103.09815](https://arxiv.org/pdf/2103.09815))** shows
curricula collapse with "forgetting students" unless earlier tasks are
replayed. Our defenses, in order of leverage: (1) `curriculum_easy_frac=0.2`
easy-start replay (Florensa's `N_old`), (2) the success side-buffer (§1.3),
(3) buffer sized so ≥2 levels of data coexist (20k ≈ OK at current episode
lengths; don't shrink it further).

**Note:** the policy network itself carries everything forward between levels
— nothing is re-initialized on advance. What "resets" at an advance is only
the start distribution. The two leaks are buffer aging (§1.3 fixes) and
post-advance under-exploration (§1.2 fixes).

---

## 2. Concrete apply list for this repo (ordered)

| # | Change | Where | Status |
|---|---|---|---|
| P0 | Retreat only on success ≤ 0.15; unsafe rate gates advance only | `logging_utils.CurriculumScheduler._maybe_update` | **on disk**, needs restart |
| P0 | Recovery bounds: axis kill 0.20→0.35 rad, pen-excess kill 3→5 mm (success gates unchanged) | `scene_env.SceneEnvConfig` | **on disk**, needs restart |
| P0 | `--max-episode-steps 300` (wanderers wasted 600) | `train.py` | **on disk**, needs restart |
| P1 | `--ent-reheat 0.15`: on curriculum advance, `log_ent_coef := log(max(α, 0.15))` | `CurriculumScheduler` (has `self.model`) | **on disk** (this doc's commit), needs restart |
| P1 | optional `--target-entropy-schedule`: `target_H = −1.5 − 1.5·level` on advance | same callback | **on disk**, off by default |
| P2 | `--success-mix 0.25 --success-buffer-size 4000`: success-transition side buffer mixed into every minibatch | `RL/success_buffer.py` + `train.py` `replay_buffer_class` | **on disk**, on by default for `--scene` |
| P3 | If plateau at high level AND you raise gradient-steps: periodic last-layer resets (Nikishin) | not implemented | back pocket |

### Verification signals after restart
1. `train/curriculum_level` should pass 0.10 within ~30–50k steps and keep a
   staircase shape (holds are fine; bounces back to 0.05 should be gone).
2. `loss/entropy_coef` should be a **sawtooth** synced to level advances.
3. Per-level success in the `[curriculum]` log lines: each new level should
   start ~0.3–0.5 and climb; if a level starts at <0.1 the step size (0.05) or
   band (0.25) is too aggressive *for that region* — halve `--curriculum-step`
   before touching anything else.
4. `eval/tier3` (pinned at level 1.0) is the end-goal curve — it should start
   moving well before the curriculum reaches 1.0, because of the success
   buffer + frontier-band generalization.

---

## Sources
- Florensa et al., *Reverse Curriculum Generation for RL*: https://arxiv.org/abs/1707.05300 (alg. details from the CMU PDF)
- Xu et al., *Target Entropy Annealing for SAC*: https://arxiv.org/abs/2112.02852
- *Variation-Aware Entropy Scheduling for Non-Stationary RL*: https://arxiv.org/html/2601.19624v2
- Oh et al., *Self-Imitation Learning*: https://arxiv.org/abs/1806.05635
- Vecerík et al., *DDPGfD — Leveraging Demonstrations for Insertion*: https://arxiv.org/abs/1707.08817
- Nikishin et al., *The Primacy Bias in Deep RL*: https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf
- D'Oro et al., *Breaking the Replay Ratio Barrier* (via Nikishin follow-up)
- *Reset & Distill*: https://arxiv.org/html/2403.05066v1
- Portelas et al., *TeachMyAgent*: https://arxiv.org/pdf/2103.09815
- Narvekar et al., *Curriculum Learning for RL Domains* (JMLR 2020): https://www.jmlr.org/papers/volume21/20-212/20-212.pdf
