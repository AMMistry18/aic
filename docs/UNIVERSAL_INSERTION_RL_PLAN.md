# Universal Insertion RL — Plan (2026-07-16)

Goal: one RL policy (or one pipeline producing one deployable policy) that
inserts ANY connector — conditioned on inputs, no per-port hand-tuning.
Grounded in (a) a SOTA survey of 2021–2026 generalist-insertion work and
(b) an audit of the existing RL/ pipeline. Companion: docs/SC_INSERTION_PLAN.md.

## The state of the art, in one paragraph

The field is essentially one NVIDIA research lineage plus a few satellites:
Factory (RSS'22, GPU contact sim for assembly) → IndustReal (RSS'23, sim-to-real
insertion, 83–99% over 600 real trials) → **AutoMate (RSS'24 — the direct
answer to this question)**: ~100 per-part specialist policies distilled into
ONE geometry-conditioned generalist that hits 84.5% real-world success across
20 assemblies, within ~2 points of the specialists. Around it: FORGE
(force-threshold-conditioned contact policies), SRSA (retrieve+finetune the
nearest existing skill for a new part, 90% real success), MatchMaker
(auto-generating training assets), InsertionNet (Bosch: one supervised net,
16 connectors, >97.5% — no RL), RLDG (RL specialists → VLA generalist).

## The consensus recipe (what everyone converged on)

1. **Specialists first, generalist by distillation** (BC → DAgger → RL
   fine-tune). Nobody reports joint multi-task RL from scratch beating this.
2. **Geometry conditioning = a learned latent from the part's point
   cloud/mesh** (AutoMate: PointNet autoencoder latents z_plug‖z_socket in the
   obs). Task-ID one-hots don't transfer to new parts.
3. **Reverse curriculum from the seated state** (AutoMate reverses
   disassembly trajectories; IndustReal's sampling-based curriculum). Nobody
   trains "from far away and hope."
4. **Asymmetric actor-critic** (privileged critic, sensor-realistic actor),
   single-stage — the modern replacement for two-phase teacher→student.
5. **Dense reward in geometry space** (IndustReal: SDF distance-to-goal), not
   EE-pose L2.
6. **Pose noise calibrated to MEASURED real perception error** (FORGE holds
   >80% to 2.5 mm noise; IndustReal's SAPU down-weights reward where pose
   uncertainty makes sim contact unreliable).
7. **Force threshold as a policy INPUT** with penalty −β·max(0,‖F‖−F_th)
   (FORGE) — one policy, gentleness dial-able at deploy without retraining.
8. **Success-prediction head** on the policy for self-aware early
   termination/retry (FORGE).
9. **PLAI**: deployment-time integration of action deltas to cancel
   impedance-controller steady-state error — free sim-to-real patch.
10. All NVIDIA code is Isaac-Lab-native (AutoMate is in Isaac Lab v2.2.0;
    FORGE/Factory as `Isaac-Forge-*`/`Isaac-Factory-*` tasks). The
    **algorithms are simulator-agnostic; the code is not.**

## Audit: what the RL/ pipeline already has vs lacks

Already built (matches consensus — do NOT rebuild):
- Asymmetric actor-critic SAC (`student_v3_sac.py:57-77`), RLPD prior replay
  + BC aux loss, success-oversampling buffer.
- Reverse curriculum with frontier-band sampling (`scene_env.py:1044-1095`),
  validated-start machinery (seat_env).
- Distillation code incl. DAgger-style relabel (`distill_dataset.py:81-83`)
  — the exact AutoMate stage-2 mechanism, already written.
- Rich DR: physics, controller gains, F/T bias/noise/delay, tracking bias,
  perception+grasp noise at the deploy-obs layer (`student_env_a.py:114-122`).
- Geometry mutation at compile time via MjSpec (`scene_env.py:282-367`) —
  the mechanism procedural port randomization needs.
- **Per-episode target-port switching already wired but unused**:
  `insert_target_bodies` + `randomize_target_body` (`scene_env.py:99-101`,
  `_select_target_body` :709-714). SC port bodies already in aic_world.xml
  (entrance −0.01564) — visual-only, no collision geometry yet.

Missing (the actual work):
- **Geometry is never an observation** — depth/mouth/tolerances are scalars
  baked into `SceneEnvConfig` and `rl_insert_contract.py`. No conditioning
  input exists anywhere.
- No per-port spec table (seated_depth_m etc. are single SFP scalars; plug
  body is hardcoded `sfp_tip_link`).
- SC collision primitives in MuJoCo (gate-0, per SC_PORT_TEACHER_HANDOFF —
  primitives only, solref ≥5–8 ms, guided-stability gate before training).
- Force-threshold conditioning, success head, PLAI — none present.

## Decision: conditioning representation

At N=2 connectors, a PointNet mesh autoencoder (AutoMate) is overkill and
adds failure modes. **Use an explicit normalized geometry vector** —
`[seated_depth, mouth_half_w, mouth_half_h, clearance, chamfer_angle,
tip_length]` (≈6 dims, unit-normalized) — appended to the obs. At tiny N an
explicit parameterization spans the same space a latent would learn, is
interpretable, and lets procedural randomization sample it directly. Swap in
PointNet latents only when the connector family grows past what a hand
parameterization captures (that upgrade is obs-layer-only; policies retrain
but the pipeline doesn't change). This is the one deliberate deviation from
AutoMate, justified by scale.

## Build plan

### Phase A — plumbing (simulator-agnostic, ~2–3 days, can start now)
1. **PortSpec dataclass**: `{name, target_body, plug_tip_body, tip_in_tcp,
   seated_depth_m, mouth_half_extents, clearance_m, success_tol, force_gates,
   geometry_vec()}`. Registry: SFP spec (current constants), SC spec
   (depth 0.0156, mouth 8.8×6.0 mm — pending keypoint-convention
   resolution). Replace the scalars in `SceneEnvConfig` /
   `make_student_env_a` / reward normalization (depth progress ÷
   seated_depth so w_depth means the same thing for a 15.6 mm bore as a
   45.8 mm cage).
2. **Geometry obs**: append `spec.geometry_vec()` (≈6 dims) to the align/v3
   actor frames and to a new obs69→obs75 deploy contract rev. Keep the
   privileged critic's channel as-is (it can also take the true geometry).
3. Enable `insert_target_bodies=(sfp_port_*, sc_port_*)` +
   `randomize_target_body=True`; per-reset spec lookup in
   `_configure_port_frame`.
4. **FORGE force conditioning**: add F_th to obs (sampled per episode
   6–20 N), reward −β·max(0,‖F‖−F_th). Cheap, high value for the scoring
   force penalty too.
5. Set `PERCEPTION_POS/ROT_SIGMA` from MEASURED YOLO numbers once the
   perception eval exists (docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md) — not
   guessed values. Consensus point #6.

### Phase B — SC contact capability (blocker for contact generality)
6. Build SC port collision primitives in aic_world.xml (box walls + chamfer
   wedges; NEVER raw mesh collisions), pass the guided-only stability gate
   (zero QACC events, forces in tens of N) per SC_PORT_TEACHER_HANDOFF
   before any training.
7. Extend the MjSpec compile path to **procedurally randomize port geometry**
   per compiled variant (mouth size ±20%, depth 10–50 mm, clearance,
   chamfer) with the geometry obs reflecting the sampled values — this is
   Factory/AutoMate-style "no one true port to memorize," using machinery
   the contact-ridge work already proved.

### Phase C — train specialists, then distill (the AutoMate path on your rig)
8. **Specialists** (existing trainers, now spec-parameterized): SFP contact
   specialist (have it), SC contact specialist (seat/align pipelines with SC
   spec), each asymmetric SAC + reverse curriculum. 2–3 seeds × ~300k on
   TACC as usual.
9. **Generalist**: roll each specialist ~5k episodes → BC on
   (deploy_obs+geometry, action) via `distill_dataset.py`/`train_student*`
   → DAgger relabel rounds (code exists) → RL fine-tune with RLPD using the
   distill shards as prior, curriculum raising start difficulty
   (SBC-style: raise the minimum start offset, keep max fixed).
10. **Success head + PLAI** at deploy: auxiliary success output for retry
    logic in RLInsert; verify the pose-command integration path matches
    PLAI (integrate deltas at the controller target, don't re-anchor to
    measured pose each step).

### Sequencing vs the competition
- **Before July 28 (Phase 1 submission)**: Phase A only, applied to the
  ALIGN policy — a single geometry-conditioned align policy trained jointly
  on SFP+SC (align is free-space; not blocked by missing SC collisions).
  This is real, defensible "universal" progress and feeds the writeup.
  Contact insertion for SC ships as script+recovery per SC_INSERTION_PLAN.
- **Phase 2 (real robots, ~6 weeks)**: Phases B+C in full. This is where the
  generalist matters most — FORGE-style force conditioning and calibrated
  pose-noise training are exactly the sim-to-real levers, and AutoMate's
  real-world numbers (84.5%) show the recipe transfers.
- **New connector later (LC?)**: SRSA pattern — embed, retrieve nearest
  specialist (SFP/SC), fine-tune; don't start from scratch.

## What NOT to do
- Don't train one multi-task policy from scratch across ports as the primary
  plan — the entire field went specialist→distill (at N=2 joint training
  MAY work; treat it as a cheap ablation, not the plan).
- Don't port to Isaac Lab now. The MuJoCo pipeline is calibrated against
  Flowstate physics and the deploy contract; AutoMate/FORGE code won't drop
  in, and every idea above is portable. Revisit only if MuJoCo contact
  fidelity becomes the proven bottleneck.
- Don't build the PointNet autoencoder for 2 connectors.
- Don't skip the SC guided-stability gate before SC contact training (the
  SFP gate-0 QACC blowups are the cautionary tale).

## Sources (key)
- AutoMate: arXiv 2407.08028 (Isaac Lab v2.2.0 module)
- IndustReal: arXiv 2305.17110 (SAPU, SDF reward, SBC, PLAI)
- FORGE: arXiv 2408.04587 (force-threshold conditioning, success head)
- Factory: arXiv 2205.03532 · SRSA: arXiv 2503.04538 (NVlabs/SRSA)
- InsertionNet 1/2: arXiv 2104.14223 / 2203.01153 · RLDG: arXiv 2412.09858
- Residual RL: Johannink ICRA'19 · Inoue IROS'17 (force-history LSTM)
