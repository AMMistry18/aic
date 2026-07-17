# SC Port Insertion Plan (2026-07-16, rev 2 — judging-aware)

Goal: a deployed SC-plug insertion capability in `aic_model` (RLInsert.py,
currently v41 on Flowstate) that fully inserts the SC end of `sfp_sc_cable`.
Phase 1 Tier 3 averages BOTH cable ends and runs 2–5 only execute if the prior
run fully inserts both ends — SC gates the whole 5-run sequence.

**Rev 2 change**: rev 1 optimized purely for the quantitative score and leaned
on sim-specific facts (fixed yaw, 1 mm proximity trigger, tuned biases). The
judging research below shows 30% of Phase 1 is panel-scored *explicitly against*
"hardcoded policies overfitting the task," prizes are decided in Phase 2 on
real hardware, and the submission deadline is ~July 28 (12 days). The revised
plan keeps the reliability backbone but builds it as ONE connector-parameterized
learned-perception pipeline, purges sim-quirk load-bearing assumptions, and
runs a learned SC alignment policy as a parallel track.

## What the competition actually rewards (researched 2026-07-16)

From `aic-phase-1/docs/scoring_and_submissions.md` §2.3 — Phase 1 = 70%
quantitative (AIC Engine) + 30% panel, three criteria (0–10 each), verbatim:

1. **Motion intent and fluidity** — "Does the robot move with intent, or is it
   'jittery' and random? … Reward solutions that demonstrate stable, robust
   behavior rather than brute-force or lucky attempts."
2. **Innovation** — "Does the solution use a novel approach (e.g., unique RL
   policy, clever use of perception) **or does it rely on hardcoded policies
   overfitting the task?**"
3. **Scalability & real-world viability** — "**Is the solution overfitted to
   the simulation**, or would it operate effectively in a factory environment?"

Supporting organizer statements (office-hour Q&As, June 25 / July 1 / July 9):

- "We expect your policy should be able to generalize for any insertion
  sequence." / "Don't make assumption on insertion order either for plugs or
  ports."
- "We recommend participants to not make assumptions and build an
  implementation which is generalizable."
- Grasp variation (~2 mm, ~0.04 rad): "We expect participants to come up with
  approaches to handle such variations."
- "The judges will look at your solution to verify the objective scores make
  sense. **You can be penalized**" (re: engine scores that don't reflect a
  legitimately good policy).
- `challenge_rules.md` prohibits "Exploitative Hardcoding: Hardcoding sensor
  data or environment configurations to exploit knowledge of the specific
  simulation setup"; top teams get container audits + behavioral verification.
- Classical control is explicitly *allowed* (`phases.md`: "Classical control
  algorithms" listed as a permitted approach) — sensor-driven feedback control
  is legitimate; the target of the rubric is *overfit, open-loop, sim-specific*
  hardcoding.

What judges actually see: **a one-page technical writeup + results + videos
from the highest-scoring submission** (due with final submission, ~July 28).
Not source code (audits are separate compliance checks). So the panel scores
the *narrative and the observed behavior*.

Funnel & stakes: Qualification top 30 → Phase 1 **top 10** (announced ~Jul 22
per overview.md, but July 9 Q&A says final submission July 28 — ⚠️ confirm the
real deadline) → Phase 2 on **real robots at Intrinsic HQ** decides the
$180k prize pool (top 5). Anything that only works because of sim quirks dies
in Phase 2. Organizers' own sim philosophy: "Signal over Precision … rather
than over-indexing on hyper-specific insertion physics"; they encourage
multi-sim domain randomization.

### Design implications (the honest read)

- The 70% quant + both-ends sequence gate still dominates raw scoring:
  reliability remains priority #1. A panel-pleasing approach that misses
  insertions loses more than it gains.
- But "reliability via sensor-driven feedback" and "reliability via tuned
  magic constants" score identically on the engine and very differently with
  the panel and in Phase 2. Choose the former wherever the cost is days, not
  weeks.
- **Purge as load-bearing assumptions**: the 1 mm proximity trigger (design for
  true mechanical seating; the trigger just makes it easier), the
  fixed-yaw-0° fact (use as prior/sanity clamp, keep perception-driven
  rotation active), SFP-style per-port bias constants (SFP's −0.8/+7 mm/−7°
  is exactly "hardcoded policy overfitting the task" — don't replicate for
  SC), any insertion-order assumptions.
- **Keep and present proudly**: learned multi-view YOLO pose perception,
  force-gated compliant search (Archimedean spiral is a *classic industrial
  assembly strategy*, not a hack — it reads as factory-viable in the writeup),
  RL contact policies, one architecture parameterized by connector spec.
- Motion fluidity actually favors the deliberate align→descend backbone over
  jittery RL — slow ≠ bad on the panel, and duration costs at most 12 quant
  points.

## What already exists (audited 2026-07-16)

| Item | State |
|---|---|
| RLInsert.py SC control | Not started — hard-refuses SC at `RLInsert.py:892-894`; SC YOLO weights loaded but unused |
| PerceptionInsert.py SC path (example policy, not deployed) | Substantially built: SC descent, snag recovery (lift+twist), Archimedean spiral seat search (`PerceptionInsert.py:105-110, 3852+`), `INSERTION_DEPTH["sc"]=0.016` |
| SC YOLO pose weights `best_sc_pose.pt` | Bundled (53 MB, YOLOv8m-pose, 4 kpts) — **no committed accuracy metrics anywhere** |
| SC pose train/eval/sanity scripts | Exist (`train_sc.py`, `eval_sc_pose_model.py`, `sc_pose_sanity_check.py`); datasets off-repo in `~/aic_perception_data/` |
| `final_insert_sc_model73.ts` (learned SC seat) | Referenced at `PerceptionInsert.py:324` but **missing** — no trained SC RL exists |
| SC MuJoCo collision geometry | Does not exist (SC port/plug visual-only); SC contact teacher blocked at gate 0 (`docs/SC_PORT_TEACHER_HANDOFF.md`) |
| Align-RL pipeline (SFP) | Env + trainer + slurm exist (align-only, free-space — does NOT need SC collision geometry to adapt) |
| Eval configs | `aic_engine/config/sc_eval_config.yaml` (2 trials, sc_port_0/1, time_limit=180 s), `sc_data_collect.yaml` |

⚠️ **Keypoint convention conflict**: `DataCollectorScPoseGT.py:38-39` uses half
extents 4.4×3.0 mm (8.8×6.0 mm mouth); `DataCollectorPoseSC.py:116-117` uses
12.89×4.635 mm (25.78×9.27 mm plug-face box). `best_sc_pose.pt` was trained on
one of them — PnP with the wrong local keypoint model silently produces a
scaled/shifted pose. Resolve before trusting any SC PnP output.

## Key facts (verified)

- **Geometry**: mouth ≈ 8.8×6.0 mm (GT collector convention); bore depth
  15.6 mm (entrance frame z=-0.01564, `SC Port/model.sdf:219`); plug ferrules
  Ø2.5 mm ×14 mm at y=±6.35 mm; chamfered lead-ins on the plug. Scoring seat
  verified by contact sensor `sc_port_sensor` (`model.sdf:206-210`).
  Clearance ~1–2 mm per side (tight vs SFP's roomy cage) — measure exactly in
  Stage 0.
- **Randomization**: board pose (x,y,z,yaw) once per 5-run sequence; SC ports
  translate along rails (3 on rail 0, 2 on rail 1); port yaw documented fixed
  at 0° — treat as *prior*, not assumption.
- **Grasp**: SC grasp is a steep ~89° pitch wrist config; grasp deviations
  ~2 mm / 0.04 rad are documented and must be handled (organizer statement
  above). SC port colliders trimmed July 9 (`aic#590`) for gripper clearance.
- **Sensors**: 3 wrist Baslers (1152×1024 @ 20 fps), ATI Axia80 FT on
  `/fts_broadcaster/wrench` (gravity-uncompensated; harness tares pre-spawn;
  tare service disabled during eval — keep RLInsert's software baseline),
  joint states, TF, ControllerState, Observation ≤20 Hz. IVM cloud pose
  supports `sc_plug`/`sc_port` (optional wrong-port cross-check).
- **Scoring (quant)**: port-type-agnostic. Full insertion +75/end; partial
  (5 mm XY box) 38–50 by depth; wrong port −12; >20 N >1 s up to −12;
  off-limit contact up to −24; duration ≤5 s→12 … ≥60 s→0.

## Upstream risks (as of 2026-07-16)

1. **SC tunneling bug — OPEN** (#121/#137): fast approach can pass the plug
   through the thin port colliders; may score "full insertion" then drift and
   block neighboring ports for later runs. Slow final approach (≤0.3 mm/step
   near the mouth) is mandatory — and happens to be the *right* real-world
   behavior anyway (low-force compliant insertion), so this is not a
   sim-quirk tune, it's convergent.
2. **FT tare bug (#112)**: tare only affects `/fts_broadcaster/wrench`, not
   `/observations`. RLInsert already reads the right topic.
3. **Cable physics (#98)**: cable contact can produce unrealistic torques —
   keep force gates conservative.
4. **Asset version**: be on the 2026-07-09 release or later.

## Plan — 12 days to ~July 28

Architecture principle: **one insertion pipeline, parameterized by a connector
spec** `{plug_type, local_port_keypoints, tip_in_tcp, insert_depth,
force_gates, search_radii}` — SFP and SC are two instantiations, not two code
paths. This is the qualification "Trial 3: Generalization (SC)" story made
literal, it is the writeup's central claim, and it's also just better code.

### Track A (critical path): reliable SC insertion in RLInsert.py

**A0 — Ground truth & calibration (Jul 16–17, blockers)**
- Update to latest assets; empirically verify trigger semantics + tunneling
  status with one slow and one fast scripted probe (`sc_eval_config.yaml`).
  Design for full mechanical seating regardless of trigger behavior.
- Resolve the keypoint convention; run `eval_sc_pose_model.py` for committed
  accuracy numbers. **Gate: median lateral error ≤1 mm** at handoff distance.
  If failed → retrain `best_sc_pose.pt` (`train_sc.py`, GT collector) before
  touching control.
- Measure `SC_TIP_IN_TCP` over ~10 grasps **with the documented ±2 mm /
  0.04 rad grasp jitter injected** — the policy must tolerate it, so
  calibration must characterize the distribution, not one sample.
- Measure true lateral clearance at the mouth (repo values conflict).

**A1 — Connector-parameterized refactor + SC instantiation (Jul 17–20)**
- Refactor RLInsert's perceive→align→descend→seat flow to take a connector
  spec; SFP spec reproduces current behavior bit-for-bit (regression-check
  against SFP eval before proceeding).
- SC spec: `detect_sc_pose` (already in `perception_core.py:138-195`, weights
  already loaded), resolved keypoints, measured tip transform, depth ≈15.6 mm,
  contact 3–4 N / seat cap 8–10 N, near-step ≤0.3 mm.
- Rotation: perception-driven with the rail-orientation prior as a sanity
  clamp (reject PnP rotations that disagree wildly with the multi-frame
  consensus median; do NOT hard-code yaw=0).
- **Zero pre-engage bias for SC.** If a systematic catch appears, fix it with
  the force-adaptive layer (A2), not a tuned offset.
- Keep multi-frame consensus + nearest-tip wrong-port rejection unchanged
  (SC ports are clustered; wrong port = −12 and likely kills the sequence).
- Eval ≥20 randomized trials → failure taxonomy {mouth miss, bore snag, wrong
  port, tunnel, force abort, timeout}.

**A2 — Force-adaptive recovery layer (Jul 19–22)**
- Port from PerceptionInsert.py: snag recovery (lift 2–3 mm → re-perceive →
  re-align → re-descend, bounded retries) and force-gated Archimedean spiral
  search (r 0.3→~measured-clearance+chamfer; `PerceptionInsert.py:105-110`
  defaults, shrink r_max). Frame in code and writeup as *compliant search*,
  which it is.
- Re-perception between retries = the legitimate version of "keep using the
  cameras during insertion."
- **Gate: ≥90% full insertion over ≥30 randomized trials, and 5-run sequences
  passing end-to-end.**

### Track B (parallel, time-boxed): learned SC alignment policy

Adapt the existing align-RL pipeline (env/trainer/slurm already written for
SFP) to SC — **not blocked** by missing SC collision geometry since alignment
is free-space above the mouth. Submit to TACC by ~Jul 18; decide by **Jul 24**:
- If it beats or matches script-align on the SC eval (success rate AND
  smoothness), deploy it as the alignment layer → strengthens Innovation
  narrative with a real learned SC component.
- If not converged by Jul 24, ship Track A's align and mention the approach in
  the writeup as the Phase 2 direction.

**Explicit non-goal**: SC contact-RL teacher (MuJoCo collision build + gate-0
physics + training) does not fit 12 days. Only revisit for Phase 2 — where it
becomes the *right* investment, since real SC seating won't have the sim's
proximity trigger.

### Track C: judge-facing deliverables (Jul 22–28)

Judges score writeup + videos + results — treat these as first-class:
- **One-page technical writeup** (due with final submission): architecture
  story = learned multi-view pose perception (YOLOv8-pose, multi-frame
  consensus, wrong-port rejection) + connector-parameterized compliant
  insertion primitive (impedance control, force-gated search) + RL contact
  policy (SFP) / learned alignment (SC if Track B lands) + domain
  randomization across MuJoCo/Gazebo/Flowstate. Emphasize: one pipeline, two
  connectors, documented-variation robustness (grasp jitter, board pose,
  insertion order), factory viability of low-force compliant insertion.
- **Videos come from the highest-scoring submission** — verify the submitted
  runs *look* intentful: no dithering at the standoff, no visible spiral
  flailing on easy insertions (spiral should trigger only on miss), clean
  retreat-and-retry rather than grinding.
- Confirm the actual deadline (overview.md says eval Jul 14–21 / top-10
  Jul 22; July 9 Q&A says final submission Jul 28 — these conflict).

## Answers to the open questions

- **Same technique as SFP?** Same *pipeline*, literally the same code path
  parameterized per connector — that's now a scoring asset, not just an
  engineering choice. SC's contact layer is force-adaptive search rather than
  a trained seat RL (SFP needed one because of its 45.8 mm cage; SC's 15.6 mm
  chamfered bore shouldn't), with learned SC alignment as a time-boxed
  parallel bet.
- **"Video stream + RL if it can't insert cleanly"?** No pixel-RL — but
  re-perception between recovery attempts keeps vision in the loop during
  insertion, which is both the robust and the panel-friendly version.
- **Robust/creative vs hardcoding?** Confirmed with rubric text in hand: 30%
  of Phase 1 is scored on exactly this, prizes are decided on real hardware in
  Phase 2, and the plan is revised accordingly — sensor-driven feedback
  everywhere, zero magic offsets in the SC path, sim facts demoted to priors,
  and the generalization story built into the code structure itself.
