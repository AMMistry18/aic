# Agent Handoff — SFP seating: alignment windup fix + non-terminating results

**Repo:** `/Users/satya_anandh/Developer/aic` · **Local branch:** `observe-seat-wrench-micro-correction` (pushes to `main`) · **Date:** 2026-07-24
**Supersedes** `.artifacts/HANDOFF_seat_p1_p3.md` (that doc's "enable the P1 gains" step is DONE and its sign hypothesis turned out to be the wrong framing — see §4).

---

## 1. Project

SFP cable-insertion robot policy (`aic_model`) deployed on Intrinsic **Flowstate**.
- `aic_model/aic_model/RLInsert.py` — perception + macro approach/handoff.
- `aic_model/aic_model/v50_controller.py` — plug-relative seating state machine (align → descend → seat).
- `aic_model/aic_model/aic_model.py` — ROS 2 action server; owns the goal result.

**The deployed code is the overlay**, not the main tree. `docker/aic_model/Dockerfile`
line 12 copies `aic_model/`, then line 35 copies `docker/aic_model/v50_overlay/` over the
top. **Edit BOTH copies of any file that exists in both.**

The two `v50_controller.py` copies **intentionally differ**: the overlay has an extra
`PLUG_POSE_INPUT` diagnostics block in the plug-priming function. Apply the same *logical*
edit to each; **never** copy one file over the other. Verify parity with:
```
diff aic_model/aic_model/v50_controller.py docker/aic_model/v50_overlay/aic_model/v50_controller.py
```
(only the PLUG_POSE_INPUT block should appear).

---

## 2. Current state — SHIPPED, commit `f07d3a1` on `origin/main`

Friend rebuilds/deploys; user pushes to `main`. **Not yet run on hardware — no field data
for this build yet.** The next ~10 runs are the test of §4.

### Seat config as shipped (`V50Config`, both copies)
```
seat_align_force_gain      = 0.00003   (was 0.00015)
seat_align_moment_gain     = 0.004     (was 0.02)
seat_align_max_lat_m       = 0.0004    (was 0.0015)
seat_align_max_tilt_rad    = 0.0087    (was 0.0175)
seat_align_release_decay   = 0.7       (NEW)
seat_mouth_speed_scale     = 0.25      (was 1.0 = feature was dead code)
seat_stall_grace_s         = 1.5
seat_overtravel_m          = 0.005
seat_candidate_depth_m     = 0.0445    INSERT_DEPTH_M = 0.0458
target_axial_force_n = 8.0   seat_force_cap_n = 10.0   force_abort_n = 18.0
contact_force_n = 3.0   free_speed_m_s = 0.015   axial_stiffness_n_m = 500.0
stall_timeout_wall_s = 2.5   stall_progress_m = 0.0008
insertion_event_timeout_wall_s = 6.0
```

### Results now non-terminating
`aic_model.py` used to call `goal_handle.abort()` on an unconfirmed insertion, which
**terminates the enclosing Flowstate process** — fatal for the 5-sequential-insertion task.
Now: confirmed → succeed; miss → succeed + WARN log + message
`"Cable insertion ended safely without confirmation"`. Gated by
`RL_INSERT_REPORT_MISS_AS_SUCCESS` (default ON). Set it to `0` to restore strict aborting.
This restores the *original upstream* behaviour; the abort was a later "truthful result" patch.

⚠️ **This means the skill reports success even when insertion failed.** For the
`AIC Phase 1 Submission` process, if the grader reads action results rather than the
simulator's `/scoring/insertion_event`, this matters. Truth is still in the logs.

---

## 3. Failure history (two field datasets, both analysed)

Analysis scripts: `.artifacts/analyze_diag.py`, `.artifacts/analyze_diag2.py`.
Logs: `diagnostics logs.txt` (15 runs, gains=0) and `~/Downloads/diagnositics 2.0.txt` (10 runs, gains on).

| | Old log (15 runs) | New log (10 runs) |
|---|---|---|
| Reached full depth | 33% | 50% |
| Insertion event fired | 20% | 40% |
| Mouth stall | 33% | 20% |
| Deep stall | 33% | 30% |

**None of that is statistically significant** (Fisher p = 0.38–1.00). Do not claim the
success rate improved. Two things ARE proven deterministically:
- **P2 (overtravel) works.** Deep-stall axial force went 2.29/4.70/4.72 N → 6.95/7.56/8.73 N,
  exactly matching the arithmetic `(INSERT_DEPTH_M − depth) × 500 N/m` prediction. The old
  deep stalls were *position-capped force starvation*; that is fixed.
- **Recovery-ladder removal works.** 42 thrash events (14 visual-no-consensus + 14
  stall-after-rescue + 9 lift-failed + 5 percept-failed) → **0**. This is what the user saw
  as "much better" — no more pointless lifting.

---

## 4. THE CORE DIAGNOSIS (this is what `f07d3a1` fixes — verify it next)

`_seat_alignment_sample()` was an integrator with **no leak and no reset**. At the observed
~4.5 N lateral it added **0.68 mm per sample** against a **1.5 mm** clamp → saturated in
**two samples**, then held a constant 1.5 mm lateral + 1° tilt bias for the rest of the seat.

Field run 8 of 10, one row per control sample:
```
depth   axial    |lat|   |applied|
 -1.2   -0.58    0.64     0.00    free descent
 +0.2  -11.81    6.41     0.96    <- ONE chamfer tap fires the correction
 +1.4   -7.78    4.03     1.50    <- SATURATED at the clamp
 +5.8   -0.78    0.11     1.50    <- contact GONE, bias persists
+18.5   -0.34    0.35     1.50    <- 30 mm of free travel, zero force, still pinned
+35.1   -1.34    0.41     1.50
+37.8   -6.62    3.02     1.50    <- jams permanently where the bore narrows
```
- **P1 engaged → 0/6 runs succeeded. P1 never engaged → 4/4 succeeded.**
- The "35–38 mm obstruction" is **not real** — runs 3/4/6/9 sail through it at ~0.3 N. It
  only exists when the plug carries the off-axis bias.
- Ground truth (`docs/reference/nic_card_mount.sdf`): SFP cage latch fingers are **0.25 mm**
  features. A 1.5 mm clamp is ~6× the geometry it is aligning to.

**On the "correction sign" question from the old handoff:** it cannot be resolved from this
data and is probably the wrong question. Under 7 N axial load the commanded lateral offset
is largely **not achieved** — friction locks it. That's why axial relief (Batch 2) matters
more than sign.

### What to check in the next ~10 runs
1. Do deep stalls at 35–38 mm **disappear**? (primary hypothesis test)
2. Does `nudge_applied_mm` stay **well under 0.4 mm** instead of pinning at the clamp?
3. Does the bias **wash out** during free travel (applied → ~0 when `|lat|` < 3 N)?
4. Do the −11.8/−13.3 N mouth impact spikes shrink (mouth slowdown now live)?
5. Does the process now **continue past a miss** instead of halting?

---

## 5. ON HOLD — Batches 2 and 3

Written and ready in `.artifacts/CODEX_PROMPTS_windup_fix.md`. **Deliberately not shipped.**

Reason: Batch 1 may make them unnecessary. Run 1 hit full depth (44.8 mm) but the latch
never registered — plausibly *because* of the 1.5 mm/1° bias. Run 5's late event followed a
deep stall the bias caused. Get 10 runs first, then re-decide.

- **Batch 2 — mouth unjam back-off.** For the 2/10 mouth stalls (2.3, 2.9 mm) where the plug
  friction-locks at 7 N. ⚠️ **Trap:** `next_persistent_depth()` (line 321) starts with
  `commanded_depth = max(commanded_depth, current_depth)` — you *cannot* retract by lowering
  `command_depth`; it clamps straight back. Retract must bypass it via `set_pose_target`.
  Only patch the wall-time stall paths (lines 961, 972) — **not** line 929, which is the
  `lateral_safety_m` safety abort.
- **Batch 3 — event dwell.** Run 1 sat 6.06 s at 44.8 mm and got no event; run 5's event
  arrived **2.6 s after** the controller gave up. Raise the timeout, drop the bias and
  re-check for a late event before failing.

---

## 6. Gotchas that will bite you

1. **Test command** — plain `pytest` fails. Use:
   ```
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .pixi/envs/default/bin/python -m pytest aic_model/test/test_v50_controller.py -q
   ```
   Without the prefix, `launch_testing` raises `PluginValidationError`. A collection error in
   `test_board_search.py` (`No module named 'aic_model.board_search'`) **pre-exists on HEAD**.
   Currently 15/15 pass.
2. **`docker/aic_model/Dockerfile` is intentionally uncommitted** (user's local-source build
   hack; their Mac Docker builds break on QEMU-amd64). **Never commit or modify it.** It
   blocks `git rebase` — stash it, rebase, pop, and byte-verify it came back.
3. **Build tripwire:** `Dockerfile.plug_relative_v50` line 56 greps the patched source for
   the literal `"Cable insertion failed: no correct-port event"`. Don't reword/delete it.
4. **Two build paths must stay in sync:** the overlay files AND
   `docker/aic_model/patch_v49_plug_relative_v50.py` (used by `Dockerfile.plug_relative_v50`).
5. `origin/main` gets pushes from a collaborator doing board-search/perception work. Fetch
   and rebase before pushing; no file overlap so far.
6. Model must be **ACTIVATED** on Flowstate or you get "Skill Goal Rejected by Server".
7. Deploy recipe: `docs/FLOWSTATE_DEPLOY_RECIPE.md`. `inctl`/`inbuild` live only in
   `/private/tmp` (wiped on reboot).

---

## 7. Known-but-parked issues

- **Wrong port.** All 10 runs targeted the same physical point (spread < 0.5 mm) and every
  insertion event reported `sfp_port_1` when `sfp_port_0` was requested. It is a consistent
  selection/naming offset, **uncorrelated with jamming**. User handles this in Flowstate.
- **Plug-pose estimator instability** — left cam can jump ~370 px latching the other cable
  end → 148 px plug reproj rejects. Separate bug.
- Port "58px reproj" regression was fixed earlier via `RL_INSERT_MAX_SELECT_REPROJ_PX=5`
  (commit `0fc17a3`).

## 8. Relevant memories
`seat-rl-deploy-contract-bug`, `seat-rl-v47-retrain-verdict`, `flowstate-deploy-recipe`,
`port-58px-reproj-root-cause`, `aic-phase-1-flowstate`, `aic-phase-1-task`.
