# Agent Handoff — SFP seating (P1 force-guided align + P3 mouth-zone/stall diagnostics)

**Repo:** `/Users/satya_anandh/Developer/aic`  ·  **Branch:** `main`  ·  **Date:** 2026-07-23

## 1. What this project is
SFP cable-insertion robot policy (`aic_model`) deployed on Intrinsic **Flowstate**.
- `aic_model/aic_model/RLInsert.py` — perception + macro approach/handoff.
- `aic_model/aic_model/v50_controller.py` — plug-relative seating state machine (align → descend → seat, with lift/visual recovery).
- **Deployed copy is the overlay:** `docker/aic_model/v50_overlay/aic_model/*.py`. The Dockerfile COPies the overlay over the main tree at build (`docker/aic_model/Dockerfile` lines 35-57). **Edit BOTH copies of any file** (`aic_model/aic_model/` AND `docker/aic_model/v50_overlay/aic_model/`).

## 2. Current problem
Plug **enters the port consistently but stalls partway (won't fully seat)**. Root cause from diagnostics: a **strongly lateral wedge** — at the catch (depth ≈ 0), |lateral force| ≈ 4.3–5.5N vs ~6.5N axial, |M| ~0.5–0.64 N·m. It's a **true jam** (dDepth≈0 with high force), not just contact-spring engagement.

## 3. What was built this session (observe-first, SHIPPED, gains = 0)
Committed `5ea3c82 "insert diagnostics"` → on `origin/main`. Friend already rebuilt/deployed it.
- **P1 — force-guided micro-alignment:** computes a lateral+tilt nudge from the plug-frame wrench and applies it to `target_tip`/`target_rotation` during seat. Currently **gain 0 → no motion**, but logs what it *would* do (`nudge_would`).
- **P3 — mouth-zone slowdown + stall grace + slope logging:** slows descent inside `seat_mouth_zone_m`; adds stall-grace loop; logs `SEAT_SLOPE` (windowed dForce/dDepth) to distinguish jam vs spring.
- **Align timeout 5s → 15s** applied (code default in both copies + Dockerfile `ENV RL_INSERT_V50_ALIGN_TIMEOUT_S=15`).

### Key constants in `v50_controller.py` (both copies)
- Observe gains (used only for `_would` logging): `SEAT_ALIGN_OBSERVE_FORCE_GAIN = 0.00015`, `SEAT_ALIGN_OBSERVE_MOMENT_GAIN = 0.02`.
- `V50Config` live gains (SHIP AS 0): `seat_align_force_gain = 0.0`, `seat_align_moment_gain = 0.0`. Also `seat_align_enable = True`, `seat_align_max_lat_m = 0.0015`, `seat_align_max_tilt_rad = 0.0175`, `seat_mouth_zone_m = 0.006`, `seat_mouth_speed_scale = 1.0`, `seat_stall_grace_s = 0.0`, `align_timeout_wall_s = 15.0`, `lift_timeout_wall_s = 5.0`.
- Frame map wrist→plug: `_wrench_plug_frame()` uses `wrist_to_plug = self.Rp.T @ R_tcp`.

## 4. Diagnostics verdict (from 1 observe run — high within-run consistency)
- Frame mapping SANE: |lat| ≈ 0.2–0.7N in free descent, no gross offset.
- Wedge is **lateral-dominated** (~4.5–5.5N, ~10 consecutive samples) → **P1 is the right lever**. This is settled; more observe runs won't add.
- P3 slope confirms **true jam**. Catch at the mouth (stall depth ≈ −0.3mm, best −0.8mm, force 7.87N). `nudge_would` stable ~[−0.5, +0.35] mm.
- **UNKNOWN: correction SIGN.** Cannot be resolved from gain-0 data (no motion). Only enabling the gain answers it.

## 5. THE NEXT STEP (recommended, not yet done)
**Enable the P1 gains** and rebuild. This single change teaches the sign AND whether it breaks the wedge — worth more than any number of gain-0 runs.
- One-line change in **both** `v50_controller.py` copies: `seat_align_force_gain: float = 0.00015`, `seat_align_moment_gain: float = 0.02` in `V50Config` defaults (and matching `from_env` fallbacks).
- On next runs, watch whether `|lat|` **trends down** (right sign) or **up** (flip the sign).
- Data guidance already given to user: observe phase is done (1 run sufficient for "is it lateral"); optionally 2–3 more STALL runs to see if `nudge_would` direction is systematic vs varying; **~10–20 runs to validate** wedge-rate drop once gains are on (seating is stochastic — already seats sometimes).
- **User is observe-first / data-driven — do NOT flip gains until they confirm.**

## 6. Build / deploy notes
- User's Mac Docker builds are broken (QEMU-amd64 emulation hangs on heavy imports). `docker/aic_model/Dockerfile` is modified for LOCAL-source build (COPY working tree instead of `git fetch` from private `intrinsic-dev/aic`) and **left UNCOMMITTED** on purpose (friend doesn't need it). The **friend rebuilds/deploys**; user pushes to `main`.
- Deploy recipe: `docs/FLOWSTATE_DEPLOY_RECIPE.md`. inctl/inbuild live only in `/private/tmp` (wiped on reboot).
- Note: model must be ACTIVATED on Flowstate or you get "Skill Goal Rejected by Server" (a lifecycle/runtime issue, not code).

## 7. Also fixed earlier this session (done, on main)
Port "58px reproj" regression after adding plug YOLO. NOT the model — ghost cross-matched candidates beat the true port under ungated nearest-tip selection. Fixed with `MAX_SELECT_REPROJ_PX` (env `RL_INSERT_MAX_SELECT_REPROJ_PX`, default 5.0) select gate in RLInsert.py (both copies). Commit `0fc17a3 "port pose fix"`. Memory: `port-58px-reproj-root-cause.md`.
- Open sub-issue: plug-pose estimator instability (left cam can jump ~370px latching the other cable end → 148px plug reproj rejects). Separate bug.
- Wrong-port (port_1 vs requested port_0) is being handled by user in Flowstate.

## 8. Relevant memories
`seat-rl-deploy-contract-bug`, `seat-rl-v47-retrain-verdict`, `flowstate-deploy-recipe`, `port-58px-reproj-root-cause`, `aic-phase-1-flowstate`.
