# Agent handoff — 2026-07-16 (AIC Phase 1, seat-RL deploy contract)

Entry point for the next Claude agent. Branch: `flowstate-rl-deploy-and-docs`.
**The single most important doc to read first is
`docs/SEAT_RL_DEPLOY_HANDOFF_20260714.md`** — it is the authoritative, detailed
state of the current problem. This file is the thin top layer: current status,
what changed since that doc, and the open decision.

## One-paragraph status

The pipeline is **script aligns + pre-seats → force-reactive seat RL finishes**.
Script alignment is solved. The **seat RL trains fine in sim (~0.85) but flails on
Flowstate** — it wanders in/out and never seats. Root cause is NOT the policy: it
is a **deploy-contract mismatch** — the obs band, action scale, and action
integration we feed the policy at handoff differ from what it trained on. Full
diagnosis + fix plan is in `SEAT_RL_DEPLOY_HANDOFF_20260714.md`. Since that doc,
work continued on the training side (seat curriculum, SAC update ratio, validated
wedge starts, walltime/checkpoints) and a `_nudge_to_unstick` (+Y stall recovery)
was added. The core deploy-contract bugs (#2 action integration, #3 scales) are
still the live blockers.

## Two worktrees (critical — don't edit the wrong one)

- **Training / main repo:** `/Users/satya_anandh/Developer/aic` (this branch).
  `RL/scene_env.py`, `RL/student_teacher/seat_env.py`, `train_seat.py`.
- **Deploy code:** branch `board-search`, separate worktree at
  `/Users/satya_anandh/Developer/aic-board-search`. **Codex owns the Flowstate
  deploys from there.** The seat-RL deploy handoff (`RLInsert.py`,
  `rl_insert_contract.py`) is edited THERE. Do not fix deploy bugs in the main repo.

## What changed since SEAT_RL_DEPLOY_HANDOFF_20260714.md

- New commits (training side): `e37425d`→`3347f8e` — Phase 1-3 contact-ridge
  recalibration to variable-depth lateral wedges (match Flowstate), validated
  wedge-start reset with bounded fallback (fixes a near_seated RuntimeError),
  seat SAC update-ratio + live W&B fix, curriculum fit within walltime with 50k
  checkpoints. Net: the seat training env + run are more robust; success ~0.85.
- **Uncommitted in main repo:** `aic_model/aic_model/RLInsert.py` (+106 lines) =
  `_nudge_to_unstick` — on a mouth stall, step **+Y only** (0.4 mm steps, 2 mm cap)
  with a gentle inward push to try to drop the plug in before handing to the seat
  RL. Knobs: `RL_INSERT_SCRIPT_NUDGE_*`. This is the same idea already committed on
  `board-search` (c05fa53); the main-repo copy is uncommitted — decide whether to
  commit it here or treat board-search as the source of truth (likely the latter).

## The live blockers (from the 0714 doc — still open, in priority order)

1. **Action integration bug ("the wander") — NOT fixed.** Deploy free-integrates
   `cmd_pos = cmd_pos + dp` with no clamp / no base pose; training uses a
   **clamped residual around a base pose**. Same action → tiny bounded move in sim,
   unbounded drift in deploy. Live at `RLInsert.py:1540` (and check `:1202`) on
   board-search. **Fix this first.**
2. **Action scales UNVERIFIED.** Deploy `DEPLOY_POS_SCALE/ROT_SCALE` vs training
   scene `cart_trans_scale_m/cart_rot_scale_rad`. Verify they match before touching
   (seat_env imports DEPLOY scales, which hints they do — but confirm, don't assume).
3. **8 mm handoff offset vs ≤1 mm training band.** The script's +7 mm/-7° pre-engage
   bias parks the plug ~8 mm off the true port → OOD obs for RL. The entry-gate +
   sweep (committed on board-search) converts flail into a clean "sweep failed" but
   the 2 mm sweep can't cover 8 mm. **KEEP the bias (user was emphatic) — do not
   shrink/delete it.** Reconciling this is a DESIGN decision (see below).

## Open decision for the user (ASK — do not guess)

How to land the handoff ≤1.5 mm true WITHOUT deleting the +7 mm bias: (a) a bounded
"de-bias to entry" as the script's last move before handoff, (b) extend the sweep
budget toward the true port, or (c) confirm the intended design is script-seats-
via-bias and RL only takes over once truly in-port. The 0714 doc §Step-3 lays this
out. This gates whether deploy v37+ can seat.

## Hard-won facts — do NOT repeat

- Do NOT re-perceive the port (7/7 agree, ~1.2 px — perception is not the problem).
- Do NOT delete/shrink the +7 mm pre-engage bias without the user.
- Do NOT change reward/training to fix deploy behavior — it's a deploy-contract bug.
- Nearest-tip port selection is correct/parked — fix is the Flowstate handoff
  position, not the selector. See [[script-align-seat-findings]].
- `seat-RL success_mean >1.0` is a shaped mean, NOT a 0-1 rate — get a real eval.
- Deploy: FULL `Dockerfile.student_flowstate` (thin overlay crash-loops), BUMP the
  asset name each time, `--name aic_model` on rebind, ~26 GB export is normal, auth
  is browser device-login only (never paste tokens). See `docs/FLOWSTATE_DEPLOY_RECIPE.md`.

## Key docs

- `docs/SEAT_RL_DEPLOY_HANDOFF_20260714.md` — READ FIRST (full diagnosis + steps).
- `docs/SEAT_RL_V2_PLAN.md`, `docs/INSERTION_PIPELINE_DESIGN.md`,
  `docs/FLOWSTATE_DEPLOY_RECIPE.md`, `docs/SEAT_HANDOFF_ENTRYGATE_CODEX_PROMPT.md`.
- `docs/AGENT_HANDOFF_20260713.md` — prior handoff (script align findings; now
  superseded by the seat-RL work but still valid on the align conclusions).
- Memories: [[seat-rl-v2-physics-recalibration]], [[script-align-seat-findings]],
  [[seat-env-start-depth-and-nic-geometry]], [[flowstate-deploy-recipe]].
