# Codex task — gate the seat-RL handoff behind a TRUE-frame "is it in the port?"
# check + a sweep-to-enter, then redeploy to Flowstate (v37)

**Branch:** `board-search` (this is where `_run_seat_rl` and the
`script_then_seat_rl` control mode live). Make the change there, commit, redeploy.

**Model:** GPT-5 sol-tier is fine — this is a targeted control-flow edit in one file
plus the known deploy recipe.

---

## Why (read this — it explains last night's failure)

In the last Flowstate run the script pre-engaged the plug and stalled, then handed off
to the seat RL. The seat RL immediately reported:

```
[seat_rl] handoff: depth=3.76mm lateral=8.00mm rot_deg=7.00 force=1.05N
```

`8.00 mm` and `7.00°` are exactly the script's pre-engage bias
(`SCRIPT_BIAS_Y_M = +7 mm`, `SCRIPT_BIAS_RX_RAD = -7°`). The seat RL correctly measures
pose vs the **TRUE** perceived port (`actual_port_pos`), so `8 mm` is the *real*
true-frame lateral offset — the bias physically parks the plug ~8 mm off the true port
centerline. That is fine for the SCRIPT (the bias lets the script do the easy
pre-seating), but the seat policy was **trained on a plug already inside the port with
true-frame lateral ~0.3–1.0 mm** (see `RL/student_teacher/seat_env.py` `_CANONICAL_STAGES`
`wedge`: `accepted_lateral_range_m = (0.30e-3, 1.00e-3)`, depth `level_range (0.05, 0.45)`
i.e. anywhere 5–40 mm deep, tilt ≤ ~1°). Handing it an 8 mm start is ~8× out of
distribution → it flailed and tripped the contract guard.

**Fix:** only hand off to the seat RL once the plug is genuinely INSIDE the port in the
TRUE frame (true-frame lateral small, past the mouth lip, low force). If it is not inside
yet, first **sweep the plug laterally to make it enter** the port, then hand off.

Perception does NOT need to be redone — the run showed `perception consensus: 7/7 agree,
reproj=1.17px` (sub-mm). The port location is known; the catch is a plug-side/geometry
issue resolved by feel (contact), so a lateral sweep is the right tool. Do NOT add a
re-perception step.

---

## The change — in `_run_script` at the handoff point (`aic_model/aic_model/RLInsert.py`)

Currently, at the depth-stall handoff (around the block that logs
`[script] HANDOFF (...)` and then, when `handoff_to_seat_rl`, calls `_run_seat_rl(...,
port_pos=actual_port_pos, ...)`), the handoff fires whenever depth stalls at low force —
**regardless of the true-frame lateral.** That is the bug.

Replace that with: on a low-force depth-stall (the `handoff_to_seat_rl and not force_jam`
path only), do this BEFORE handing off:

1. **Compute the TRUE-frame pose** of the tip vs `actual_port_pos` (NOT the biased
   `port_pos`):
   ```python
   true_delta = Rp.T @ (tip_pos - actual_port_pos)   # port frame
   true_depth = float(true_delta[2])
   true_lateral = float(np.linalg.norm(true_delta[:2]))
   ```
2. **Entry gate `is_in_port`** — the plug counts as inside the port (and in the seat
   policy's training band) iff ALL of:
   - `true_depth >= SEAT_ENTRY_MIN_DEPTH_M` (past the mouth lip; default 0.004 m),
   - `true_lateral <= SEAT_ENTRY_MAX_LATERAL_M` (the wedge band; default 0.0015 m — a
     little above the 1.0 mm training max to allow handoff slack, tunable via env),
   - `f_mag <= SEAT_RL_MAX_HANDOFF_FORCE_N` (already checked; low force).
3. **If `is_in_port`:** hand off to `_run_seat_rl(..., port_pos=actual_port_pos, ...)`
   exactly as today (it is in-distribution now). Log the TRUE-frame numbers:
   `[script] IN-PORT (true lateral X.XXmm, depth Y.Ymm) -> handing to seat RL`.
4. **If NOT `is_in_port`:** call a new helper `_sweep_into_port(...)` that tries to make
   the plug drop into the true port by lateral feel. THEN re-check `is_in_port`:
   - entered → hand to `_run_seat_rl`,
   - still not entered after the whole sweep → hold pose (no retreat) and `return False`
     as today (log `[script] sweep failed to enter port -- holding, no handoff`).

Keep the existing `force_jam` path unchanged (a hard force jam still holds/aborts; do not
sweep on a real jam).

### New helper `_sweep_into_port(self, get_observation, move_robot, *, Rp, actual_port_pos, R_seat, insert_axis)`

Model it on the existing `_nudge_to_unstick` (same file), but:

- **Sweep multiple directions in the TRUE port frame**, order **−Y first, then +Y, then
  −X, then +X** (config `SEAT_SWEEP_DIRS = [("-Y",0,-1),("+Y",0,1),("-X",-1,0),("+X",1,0)]`).
  (User chose −Y first.)
- For each direction: step the plug laterally in small increments
  (`SEAT_SWEEP_STEP_M`, default 0.0004 m) while keeping a gentle inward push
  (`SCRIPT_SEAT_STEP_M` along `insert_axis`) and compliant gains (`STIFFNESS`/`DAMPING`),
  a few settle sub-steps per increment (`SEAT_SWEEP_SETTLE_STEPS`, default 6).
- After each increment, recompute the TRUE-frame `is_in_port` (via `actual_port_pos`).
  **The moment `is_in_port` becomes true, STOP and return True** (entered).
- Bound each direction's cumulative lateral excursion by `SEAT_SWEEP_MAX_PER_DIR_M`
  (default 0.0020 m). When a direction hits its cap without entering, **return that
  direction's excursion to ~0 (step back to the pre-sweep lateral)** before trying the
  next direction, so excursions don't accumulate across directions and the plug can't
  walk off the port. Then try the next direction.
- Apply the lateral step only on the first settle sub-step of each increment (then zero
  it and keep only the inward push) — same anti-runaway pattern as `_nudge_to_unstick`.
- Return True if any direction achieves `is_in_port`, else False. Log each direction and
  the entered/failed outcome with TRUE-frame lateral/depth.

**Entry detection nuance:** "entered the port" = `is_in_port` (true_lateral small AND
past the lip). Equivalently you will observe true_depth advance past the mouth lip while
true_lateral drops as the plug drops into the throat. Use `is_in_port` as the single
authoritative check so the gate and the sweep agree.

### The existing `+Y` `_nudge_to_unstick`

Leave `_nudge_to_unstick` as-is (it runs earlier, before the handoff block, to break a
mid-descent stall). The NEW sweep is specifically the pre-handoff "get it into the port"
search and supersedes the old "hand off regardless of lateral" behavior. If the +Y nudge
already broke the stall and descent resumed, this block is simply reached later (or not at
all) — no conflict.

### New config constants (near the other `SEAT_RL_*` / `SCRIPT_*` env reads)

```python
SEAT_ENTRY_MIN_DEPTH_M   = float(os.environ.get("RL_INSERT_SEAT_ENTRY_MIN_DEPTH_M", "0.004"))
SEAT_ENTRY_MAX_LATERAL_M = float(os.environ.get("RL_INSERT_SEAT_ENTRY_MAX_LATERAL_M", "0.0015"))
SEAT_SWEEP_STEP_M        = float(os.environ.get("RL_INSERT_SEAT_SWEEP_STEP_M", "0.0004"))
SEAT_SWEEP_MAX_PER_DIR_M = float(os.environ.get("RL_INSERT_SEAT_SWEEP_MAX_PER_DIR_M", "0.0020"))
SEAT_SWEEP_SETTLE_STEPS  = int(os.environ.get("RL_INSERT_SEAT_SWEEP_SETTLE_STEPS", "6"))
SEAT_SWEEP_DIRS = [("-Y", 0.0, -1.0), ("+Y", 0.0, 1.0), ("-X", -1.0, 0.0), ("+X", 1.0, 0.0)]
```

---

## Guardrails / correctness

- All sweep motion in the TRUE port frame uses `Rp[:,0]*dx + Rp[:,1]*dy` (same idiom as
  `_nudge_to_unstick`), orientation held at `R_seat`, inward push along `insert_axis`.
- Never exceed the per-direction excursion cap; step back to baseline lateral between
  directions so total excursion stays bounded (no walking off the port).
- Do NOT re-perceive the port. Do NOT change `_run_seat_rl` internals, the seat model,
  the ABI, or any reward/training. This is purely the handoff GATE + sweep.
- Compile-check: `python3 -m py_compile aic_model/aic_model/RLInsert.py`.
- With `CONTROL_MODE` not `script_then_seat_rl`, behavior is unchanged (the gate/sweep
  only run on the `handoff_to_seat_rl` path).

---

## Expected result (state this back to the user)

On the next run, the script should either hand off with a small TRUE-frame lateral
(≤ ~1.5 mm) — logged as `[script] IN-PORT (true lateral X.XXmm ...) -> seat RL` — or run
`[script] SWEEP -Y/+Y/... entered port` first, and only then start the seat RL. The seat
RL's `[seat_rl] handoff:` line should now read a SMALL lateral (≤ ~1.5 mm), NOT 8 mm.
That puts the policy in its training distribution, which is the whole point.

---

## Deploy to Flowstate (v37) — follow `docs/FLOWSTATE_DEPLOY_RECIPE.md` exactly

- Build from the FULL `docker/aic_model/Dockerfile.student_flowstate` (not a thin overlay
  — see the recipe's crash-loop GOTCHA). Keep the existing env
  (`RL_INSERT_CONTROL_MODE=script_then_seat_rl`, board-search env, seat model path, etc.).
  No new ENV is required (the new knobs default correctly).
- **Bump the asset name** to `aic_model_v37` (next free) — reinstalling the same name does
  NOT replace the running image.
- `inbuild service bundle` → `inctl asset install` → `service delete aic_model` +
  `service add ai.intrinsic.aic_model_v37 --name aic_model` (keep `--name aic_model`).
- Verify: `inctl service state list` shows `aic_model`; run one trial; the logs show the
  IN-PORT / SWEEP lines and a seat-RL handoff with SMALL true-frame lateral.
- Token: the user pastes the auth token into the interactive `inctl auth login` prompt.
  Never put tokens in chat/scripts/Git.

## Report back

The v37 asset name, the final `SEAT_ENTRY_MAX_LATERAL_M` used, and — from one trial — the
`[seat_rl] handoff:` lateral value (should be ≤ ~1.5 mm now, not 8 mm), plus whether the
seat RL seated or held.
