# Insertion pipeline redesign — align-first, single alignment RL

Created 2026-07-12. Design decision to implement AFTER the sim contact physics is
stable (see the QACC/ejection fix in `RL/scene_env.py` + the physics-match loop).
Not implemented yet — this captures the agreed direction.

## Motivation

The current design (`RL/student_teacher/student_v3_env.py`) is a residual RL that
only activates AT CONTACT: `combined = guided_base + residual`, with
`residual = 0` until `_contact_engaged`. Two problems observed in real Flowstate:

1. The guided base (`RL/student_teacher/parity/evaluate_guided_controller.py`,
   `guided_action`) descends WHILE still misaligned — it commands
   `[-delta_x, -delta_y, axial_step]` every step, i.e. it cancels lateral error
   AND steps down simultaneously. So the plug reaches the chamfer at an angle and
   jams. It cannot "hold depth until aligned".
2. The contact-only residual then has to fix alignment UNDER LOAD (chamfer
   fighting it, rising force) — the hardest moment to correct. Deploy runs showed
   the plug engaging then jamming (lateral/rotation spiking, force to ~170 N).

Insight (user): align BEFORE descending, so insertion becomes a clean push.
Prevention beats recovery.

## Agreed pipeline (single alignment RL)

```
1. Handoff -> perception (MUST be the FIXED perception; see prerequisite below)
2. Alignment RL  -- priority: orientation + lateral. GATES descent: does not
                    step down until orientation/lateral error is small.
3. Base script descent  -- straight down the perceived port axis once aligned.
4. Base script slows near the port mouth.
5. Gentle FORCE-LIMITED descent to seated (45.8 mm for SFP).  NO second RL
   initially -- if a well-aligned plug still jams in the (fixed) sim, THEN add a
   contact RL here as stage 5b. Build the one-RL version first.
```

Decision: **start with ONE RL** (stage 2 alignment with descent-gating). Add a
contact-phase RL only if empirically needed. Fewer policies, rewards, and
handoffs to train/debug; may be sufficient given "if aligned, insertion is
trivial".

The key thing this RL does that the guided base CANNOT: withhold descent until
aligned. That is the whole point — arrive squared, not grind in at an angle.

## Camera / perception sufficiency (evidence from 2026-07-12 Flowstate runs)

Cameras ARE good enough for the alignment RL:
- **Orientation:** perceived port quaternion was rock-stable across runs
  (`q ~ [0, 0.005, 1.0, 0]`, reproj < 2 px). Trustworthy for squaring the plug.
- **Lateral:** on correct-port runs, position clustered within ~0.6 mm (SFP mouth
  is 8.8 mm wide, so 0.6 mm noise is fine).

## HARD PREREQUISITE: fix the wrong-port perception bug first

This pipeline servos every stage toward the stage-1 perceived port. If perception
returns the WRONG port, the pipeline confidently, precisely drives into the wrong
hole. In the 3x probe, ~1/3 of runs perceived a neighboring NIC ~40 mm off with
CLEAN reproj (1.2 px). So `docs/PERCEPTION_TODO.md` (multi-frame median consensus
+ max-distance gate) is a HARD PREREQUISITE, not optional, for this design.

## Build order

1. Stabilize sim contact physics (in progress) — align-first is pointless if the
   sim ejects the plug.
2. Fix wrong-port perception (`PERCEPTION_TODO.md`).
3. Implement the alignment RL (stage 2) with descent gating; reward = orientation
   + lateral reduction, penalize descending while misaligned, success = aligned
   within tolerance at the mouth. Keep the base script for descent + gentle
   force-limited seat.
4. Evaluate: do well-aligned plugs seat cleanly? If yes, done. If they still jam,
   add a bounded contact RL as stage 5b (the current residual approach, but now
   starting from a genuinely-aligned pose).

## Related

- Current residual env: `RL/student_teacher/student_v3_env.py`
- Guided base: `RL/student_teacher/parity/evaluate_guided_controller.py`
- Physics failure + fix: `RL/student_teacher/STUDENT_V3_PILOT_ROOT_CAUSE_20260712.md`
- Perception prerequisite: `docs/PERCEPTION_TODO.md`
- Reward internals (for whichever RL): `RL/reward.py` + shaping in `student_v3_env.py`
  (note the `-=0.7*breakdown.xy` sign bug — fix before reusing that shaping).
