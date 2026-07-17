# Codex note — include the +Y stall-nudge when building/deploying v36

When you build the next asset (v36) with the board-search changes you added, **also
include an UNCOMMITTED change already made to `aic_model/aic_model/RLInsert.py`** in the
working tree. Do NOT overwrite or discard it.

## What the uncommitted change is

A `+Y` stall-nudge recovery in the scripted insertion (`_run_script`). When the plug
stalls at the mouth without entering the port at low force, it now steps a little in `+Y`
(port frame) to drop it in, BEFORE giving up / handing off. Details:

- New helper `_nudge_to_unstick(...)` (defined just above `_run_script`).
- New config knobs near the other `SCRIPT_*` env reads:
  `SCRIPT_NUDGE_ENABLE` (env `RL_INSERT_SCRIPT_NUDGE_ENABLE`, default `1`),
  `SCRIPT_NUDGE_STEP_M`, `SCRIPT_NUDGE_MAX_TOTAL_M`, `SCRIPT_NUDGE_SETTLE_STEPS`,
  `SCRIPT_NUDGE_PROGRESS_M`, `SCRIPT_NUDGE_DIRS = [("+Y", 0.0, 1.0)]` (+Y only).
- A hook in the depth-stall branch of `_run_script` that calls `_nudge_to_unstick`
  before the hold-and-handoff.

## What you must do

1. **Before building v36, check the working tree is intact.** These must all be present
   in `aic_model/aic_model/RLInsert.py`:
   - `def _nudge_to_unstick(` exists,
   - `SCRIPT_NUDGE_ENABLE`, `SCRIPT_NUDGE_DIRS` exist,
   - the depth-stall branch calls `self._nudge_to_unstick(`.
   ```bash
   grep -n "_nudge_to_unstick\|SCRIPT_NUDGE_ENABLE\|SCRIPT_NUDGE_DIRS" \
     aic_model/aic_model/RLInsert.py
   ```
   If any are missing (e.g. a checkout/stash wiped them), STOP and tell me — do not
   rebuild without them.

2. **Commit it together with your board-search work** so v36's image contains both. (It
   is a real feature, keep it in the same branch/commit set as board-search.)

3. **Keep it ENABLED in v36.** `RL_INSERT_SCRIPT_NUDGE_ENABLE` defaults to `1`, so it is
   on unless disabled. When you bake env vars into the full
   `docker/aic_model/Dockerfile.student_flowstate` build, you do NOT need to add a new
   ENV for this (the default is already on) — but make sure nothing you added sets it to
   `0`. If you want it explicit, bake `ENV RL_INSERT_SCRIPT_NUDGE_ENABLE=1`.

4. **Verify in the built image before bundling** that the code is present:
   ```bash
   docker run --rm <v36-image> python3 -c \
     "import aic_model.aic_model.RLInsert as m; assert hasattr(m.RLInsert,'_nudge_to_unstick'); print('nudge OK')"
   ```
   (Adjust the import path to however the package is installed in the image.)

5. Deploy v36 per `docs/FLOWSTATE_DEPLOY_RECIPE.md` (full Dockerfile, bump asset name,
   `service delete` + `service add --name aic_model`).

## Confirm back

Report: the v36 asset name, that `grep` found all three markers, and that the in-image
check printed `nudge OK`. In a scripted run that stalls at the mouth, the log should show
`[script] NUDGE +Y ...` lines before any handoff.
