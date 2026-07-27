# Prompt for the next agent — rebuild Stage 1 of check_board_visibility v4

Copy everything below the line into the next session.

---

Your job this session is to **replace Stage 1 of the
`ai.tar2.check_board_visibility_skill_v4` perception skill**. Stage 1 is the
insignia-acquisition search. It is structurally broken, not mistuned, and it is
to be scrapped and rebuilt. Stage 2 (the geometric survey) is validated and
hardware-confirmed — **do not touch it**.

## Start here

Read `docs/CHECK_BOARD_VISIBILITY_V4_HANDOFF.md` in full before writing any
code. It is the consolidated entry point. Sections that matter most for this
task:

- **§19** — the hardware evidence that Stage 1 is broken, and why the design is
  wrong rather than mistuned.
- **§20** — the replacement plan, the prior art on `origin/navigate-to-purple`,
  and the suggested order of work.
- **§5** — what the current (broken) Stage 1 actually does. You need this to
  replace it safely.
- **§2, §3** — the closed input allowlist and the Flowstate contract. Both
  constrain the rebuild.
- **§17** — experiments already run and rejected. Do not repeat them.

The real git repo and all commands live in `aic\aic` (the outer `.git` is
empty). `HEAD` is `467636d`, tree clean.

## The problem in one paragraph

Stage 1 is an open-loop phase machine (`ACQUIRE -> CENTER -> ALIGN -> LEVEL ->
ASCEND`, 1614 lines in `aic_perception/viewpoint_search.py`) that nudges the
arm and re-measures. It never forms a hypothesis about where the board or the
insignia actually is, so when the insignia is out of frame in all three cameras
there is no gradient to follow and it wedges. On 2026-07-27 it published a pose
requiring 501 degrees of total joint travel, the arm ended up in contact with
the board, and every subsequent invocation force-aborted with all three cameras
reporting `logo=False` and byte-identical readings. All three downstream
sectors then returned ~1 detection out of 5 — not a per-sector perception
problem, a Stage-1 problem.

## What to build

Make acquisition a **search over commanded poses**, the way Stage 2 already
works, instead of a sequence of greedy nudges:

1. **Hypothesise the board pose from what is visible.** The plate is reliably
   visible (`area` 0.30-0.37 even in the failing runs); it is the *insignia*
   that is missing. `estimate_board_pose()` (`board_stage2.py:839`) PnPs the
   plate outline but **must not be used naively** — the plate is clipped in
   exactly these situations and a clipped mask yields a frame-aligned
   degenerate rectangle (`long_ratio ~= 1.00`, `long_axis_error = +0.0deg`).
   Reject that case explicitly. Consider the coloured landmarks that stay in
   frame instead — blue SC adapters, green NIC cards — at their known
   board-frame positions.
2. **Solve for a pose that would expose the insignia.** `INSIGNIA_RECT_CORNERS`
   is a known board-frame box, so this is the same problem Stage 2 solves for
   SFP/NIC/SC. Reuse the trusted machinery: `UR5eArm.solve_ranked` seeded from
   live joints, `self_clearance`, `_arm_clear_of_own_cameras`, workspace and
   component-clearance guards, relative joint-travel caps. Command one move,
   not a chain.
3. **Deterministic joint-ladder fallback** when there is no hypothesis: a fixed,
   precomputed set of IK-valid, arm-clear configurations tiling the plausible
   board region, visited in a fixed order. This replaces `ACQUIRE`.
4. **A safe home, always.** On exhaustion, force abort, or any terminal
   failure, return to a known-good observation pose before releasing the
   controller. Its absence is why one bad exit poisoned every later invocation.

## Prior art to mine, not to copy wholesale

`origin/navigate-to-purple`, tip `4a20097`, skill `move_to_board_skill`:

- `aic_perception/purple_insignia.py` — a ROS-free, unit-testable HSV purple
  detector whose band is copied from the proven
  `PerceptionInsert._sc_purple_logo_centroid_px`. **Port this.**
- `move_to_board_skill.py` — a flat greedy loop with no phase machine, pure
  image-plane translation at fixed orientation, and an explicit terminal
  condition (`purple_done`: all three cameras see the unclipped insignia and
  the centre camera is within 10% of centre). The flatness and the fixed
  orientation are the good ideas.
- `test/test_move_to_board_loop.py` — 448 lines of existing tests.

Its limitation: it is still a greedy servo with no board-pose hypothesis and no
reachability gate, so when nothing sees purple it falls back to centring the
board — the same gradient-free situation that wedges v4. Take its detector and
its flat structure; supply the hypothesis and the pose search yourself.

## How to work

**Build the offline sweep harness first and score the *current* Stage 1 with
it, so you have a baseline number to beat.** Model it on
`test/sc_sweep_runner.py` and `test/sfp_sweep_runner.py`: board yaw x tilt x
placement x live start, using the production camera rig, the real gripper
masks, the analytic IK and the arm-in-view gate. Score whether the insignia
ends up unclipped in a calibrated camera. Do not delete `viewpoint_search.py`
until the replacement beats that baseline.

This codebase's standard is: measure the workcell geometry offline, sweep the
full matrix, then validate real camera frames. Two bugs fixed last cycle both
hid behind a metric that averaged over the wrong thing — score the **physical
worst case over the component's legal placement range**, not fixed samples.

Constraints:

- Keep Stage 2 untouched.
- Keep the input allowlist closed (§2): no ground-truth board transform, no
  component pose, no scoring state.
- Keep returning expected failures as `success=true, done=false` so Flowstate
  can always release the AIC controller (§3).
- Preserve force-guard semantics, but treat a force abort as a state to
  **recover from**, not just a reason to return.

Validate with:

```powershell
cd C:\Users\anshu\College\aic\aic\flowstate\aic_perception
python -m pytest test/ -q                    # currently 286 passed
python test/sc_sweep_runner.py --workers 8   # 144/144, cue 7.36-8.55 px
python test/sfp_sweep_runner.py --workers 8  # 92/144 found, 92/92 seats, 0 clipped
```

Those three must not regress. The user builds and installs; do not deploy.

## Two smaller things, if there is time after

- **Total joint-travel cap.** The 185 degree limit is on the worst joint only;
  `total_motion` is computed for ranking and never gated, which let a
  `total=501.3deg` pose through. Suggested 250-300 degrees, swept to confirm no
  coverage loss.
- **`UR5eArm.autocalibrate()` instability.** One session logged `tool=197.1mm`
  four times and `tool=201.3mm` once. A 4.2 mm shift moves every projected
  candidate.
