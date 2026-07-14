# Codex task — Taskboard "get the whole board in view" search (build + deploy on Flowstate)

**Model recommendation:** GPT-5 Sonnet-tier / Codex "sol" is enough. This is a
self-contained OpenCV + arm-move task with a small, well-defined interface — no ML,
no physics, no RL. Use a stronger model only if the deploy (inctl/inbuild) trips you up.

**Repo:** `/Users/satya_anandh/Developer/aic`  (branch: make a new branch off `main`,
e.g. `board-search`). Commit when done.

---

## 0. Goal (read this twice — it is the WHOLE job)

Before the existing insertion runs, the robot's cameras may only see a **corner** of
the taskboard. Build a small, standalone perception+motion routine that:

1. **Finds the taskboard** in a camera image by color/brightness (the board is a dark
   charcoal/near-black plate on a light-gray floor).
2. **Moves the arm** so the camera's view covers the **ENTIRE** taskboard (board fully
   inside the frame, roughly centered), in **AT MOST 3 arm moves**.
3. Returns success once the whole board is in view.

That is it. **Do NOT** do any NIC-card / SFP-port perception, IVM, triangulation, or
insertion in this routine. This routine runs FIRST; whatever runs after is out of scope.

Success criterion, concretely: the detected board blob is fully inside the image
(not touching any image border) AND its centroid is within a center tolerance of the
image center, in one chosen camera.

---

## 1. What already exists (reuse it — do not reinvent)

File: `aic_model/aic_model/RLInsert.py` (class `RLInsert(Policy)`). It already has:

- `CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]` — three cameras.
- `self._get_cam_data(obs, cam_name)` → returns `(bgr, K)` (a BGR `np.uint8` image and
  the 3×3 intrinsics) or `None`. **Use this to get images.**
- `self._lookup_cam_from_base(cam_name)` → 4×4 `T_cam_from_base` (or `None`). Available
  but NOT required for visual servoing (only needed if you add a back-projection path).
- `self._tcp()` → `(tcp_pos[3], tcp_quat_wxyz[4])` current TCP pose in `base_link`.
- `self._tcp_target_for_tip(...)` / `self._tcp_target_for_tip` and
  `self.set_pose_target(move_robot, pose, frame_id="base_link", ...)` — **the way you
  command arm motion.** `pose` is a `geometry_msgs.msg.Pose` (position + wxyz-quat) for
  the TCP in `base_link`. Building a `Pose` from a numpy pos+quat is done all over this
  file (see `_tcp_target_for_tip` at ~line 488 for the exact idiom).
- `self.sleep_for(seconds)`, `self.get_logger()`.
- Entry: `_run(self, task, get_observation, move_robot, send_feedback)` (~line 860).
  `get_observation()` returns an `Observation` with the camera fields used by
  `_get_cam_data`. The first observation is fetched at ~line 874.

The Flowstate deploy recipe is in `docs/FLOWSTATE_DEPLOY_RECIPE.md` — follow it exactly
for build + install + rebind. Read it before deploying.

---

## 2. Build a new module `aic_model/aic_model/board_search.py`

Put ALL the new logic here so `RLInsert.py` stays clean. Suggested shape (adapt names as
you see fit, keep it readable and matching the file's existing numpy/cv2 style):

```python
class BoardSearch:
    """Move the arm until the whole dark taskboard is in a camera's view (<=3 moves)."""
    def __init__(self, policy):        # policy is the RLInsert instance (for _get_cam_data,
        self.p = policy                # _tcp, set_pose_target, sleep_for, get_logger, etc.)

    # --- detection -------------------------------------------------------
    def detect_board(self, bgr):
        """Return (found, cx, cy, area_frac, bbox, touches_border, mask) for the board
        blob in one BGR image, or found=False. cx,cy in pixels; area_frac = blob_area /
        image_area; touches_border True if bbox hits any image edge."""

    # --- top-level -------------------------------------------------------
    def run(self, get_observation, move_robot):
        """Do the search. Return True once the whole board is in view (or False if it
        genuinely cannot be found). At most 3 arm moves."""
```

### Detection (`detect_board`) — the board is dark, floor is light

Robust recipe (tune constants against the real frame; see §4 for a sample frame's HSV):

1. Convert BGR→HSV.
2. Board mask = **low Value (dark) AND low Saturation (gray/black)**:
   `V in [~20, ~95]` and `S in [0, ~80]` (OpenCV HSV, V,S in 0–255). Floor is bright
   (`V > ~180`) so it is excluded automatically.
3. **Remove the arm/gripper and shadows** (they are also dark). Do this by:
   - `cv2.morphologyEx` OPEN then CLOSE (kernel ~7–11px) to kill speckle/shadow fringe.
   - `cv2.findContours`, keep the largest, and **require it be board-like**: area above a
     min fraction (e.g. > 3% of image), and reasonably solid/rectangular
     (`contourArea / boundingRectArea > ~0.55`). Shadows are large but ragged → low
     solidity → rejected. The arm is a blob too, but the board is a big compact quad; if
     the arm competes, prefer the more rectangular / higher-solidity contour.
   - Optionally mask out a known lower-center image band where the gripper usually sits
     (only if it helps — do not over-fit).
4. Return the chosen contour's centroid, area fraction, bounding box, and whether the
   bbox touches any image border.

Keep detection **pure** (image in → result out) so it is unit-testable offline.

### Camera choice

Detect in all three cameras; pick the one with the **largest** valid board blob (most of
the board visible). If none detect, that camera set does not see the board — return False
with a clear log (do not move blindly).

### Motion — center + fully-frame the board in ≤3 moves (DO THIS RIGHT)

Do NOT do slow blind nudging. Use a **one-probe image-Jacobian**, then jump:

- **Move 1 (probe):** command a small, known TCP translation in `base_link` — e.g.
  `+dx` then observe, or do a single diagonal probe `Δp = (PROBE_M, PROBE_M, 0)` with
  `PROBE_M ≈ 0.02` (2 cm). Re-image. Measure how the board centroid moved in pixels:
  `Δpix = (dcx, dcy)`. This gives a 2×2 Jacobian `J` mapping base-plane arm motion (x,y)
  → pixel motion (u,v). (With a single diagonal probe you get one column; do two small
  orthogonal probes if you want the full 2×2 — still within budget if you fold the probe
  into the corrective move. Simplest robust version: two tiny orthogonal probes as part
  of "move 1", then one corrective jump = counts as your first real move.)
- **Move 2 (center):** solve `Δp_xy = J^{-1} · (center_pixel − current_centroid_pixel)`
  and command that TCP translation (keep Z and orientation fixed). This lands the board
  centroid near image center in one shot.
- **Move 3 (fit / refine):** re-image. If the board bbox still **touches a border**
  (board overflows the frame → camera too close), **raise the arm** (increase standoff:
  move TCP up along `base_link +Z`, or back along the camera's viewing axis) by a step
  sized from `area_frac` (bigger overflow → bigger raise) so the whole board fits. Also
  apply a small residual centering correction from the same `J`. Re-check.

Stop as soon as: **board fully inside frame (no border touch) AND centroid within
`CENTER_TOL_FRAC` (e.g. 0.08) of image center.** If not achieved after 3 moves, log the
final state and return whether the whole board is at least fully in-frame.

**Guardrails:** clamp every commanded TCP delta to a safe magnitude (e.g. ≤5 cm/move),
keep orientation fixed (do not rotate the wrist during search), keep Z within sane
bounds, and hold `set_pose_target` stiffness/damping at the `Policy` defaults. Sleep
briefly (e.g. `self.p.sleep_for(0.5–1.0)`) after each move before re-imaging so the arm
settles and a fresh `get_observation()` reflects the new pose.

### Config constants (top of module, easy to tune)

`BOARD_V_MAX`, `BOARD_S_MAX`, `MIN_BLOB_AREA_FRAC`, `MIN_SOLIDITY`, `PROBE_M`,
`CENTER_TOL_FRAC`, `MAX_MOVE_M`, `RAISE_STEP_M`, `MAX_MOVES=3`.

---

## 3. Wire it into `RLInsert.py` behind a flag (don't break existing behavior)

- Add an env flag near the other env-config reads (top of file, e.g.
  `BOARD_SEARCH = os.environ.get("RL_INSERT_BOARD_SEARCH", "0") == "1"`).
- In `_run`, **after** the first observation is obtained (~line 883, right after the
  `obs is None` guard) and **before** `perceive_port_pose_consensus`, add:

  ```python
  if BOARD_SEARCH:
      from .board_search import BoardSearch
      ok = BoardSearch(self).run(get_observation, move_robot)
      log.info(f"[board_search] whole board in view: {ok}")
      # For now, continue regardless (search is a pre-positioning aid). If you want the
      # task to abort when the board can't be framed, return False here on not-ok — but
      # default to continuing so existing runs are unaffected when the flag is off.
  ```

- When `RL_INSERT_BOARD_SEARCH` is unset/`0`, behavior must be **byte-for-byte the old
  path**. This flag is the only behavioral switch.

---

## 4. Sample frame reference (for tuning the HSV thresholds)

A real Flowstate camera frame shows: **light-gray floor (bright)**, a **dark charcoal
taskboard plate (target)** at the right, a **bright magenta square marking on the board**
(ignore it — it is on the board, inside the dark region), and the **dark robot arm/gripper
bottom-center** plus **soft gray shadows** on the floor (these are the false positives the
solidity/size filters must reject). Rough HSV of the board: very low V (~0.1–0.25 → ~25–65
of 255), low S. Floor V ~0.75–0.9 (~190–230). Start thresholds there and adjust.

There is NO reliable color value in `taskboarinfo/` (the `.glb` is a binary mesh; the
`.xacro` only gives geometry: the board footprint is ~0.30 m × 0.425 m — useful only as a
loose sanity scale on blob size, not required). Tune HSV from actual camera frames.

---

## 5. Self-check before deploy

- `detect_board` on a saved frame returns the board (not the arm, not a shadow). Add a
  tiny offline test / `__main__` in `board_search.py` that loads an image path and prints
  the detection + saves a debug overlay (mask + chosen contour + centroid), so tuning is
  fast and does not require the sim.
- Reason through the ≤3-move budget and the Jacobian sign/inverse (a wrong sign sends the
  board off-frame — verify the probe direction vs. pixel-shift sign, and guard against a
  near-singular `J`).
- With the flag OFF, confirm no diff in the existing run path.
- Lint/import cleanly (`cv2`, `numpy` already used in this file).

---

## 6. Deploy on Flowstate (follow `docs/FLOWSTATE_DEPLOY_RECIPE.md` exactly)

- Build from the FULL `docker/aic_model/Dockerfile.student_flowstate` (NOT a thin overlay
  — see the recipe's crash-loop GOTCHA). Bake `ENV RL_INSERT_BOARD_SEARCH=1` (plus keep
  the existing control-mode env as-is) so the search runs on deploy.
- **Bump the asset name** (e.g. `aic_model_v36` or next free) — reinstalling the same
  name does NOT replace the running image.
- Build bundle with `inbuild`, `inctl asset install`, then
  `service delete aic_model` + `service add ai.intrinsic.aic_model_vNN --name aic_model`
  (keep `--name aic_model`).
- Verify: `inctl service state list` shows `aic_model`; `inctl logs ... --service
  aic_model` prints the `[board_search]` lines; run one trial in the Flowstate UI and
  confirm the arm repositions and the log reports "whole board in view: True".

**Auth/token:** the user pastes the auth token into the interactive `inctl auth login`
prompt themselves. Never put tokens in chat, scripts, or Git.

---

## 7. Deliverables

1. `aic_model/aic_model/board_search.py` (detector + ≤3-move visual-servo search + offline
   `__main__` test/debug-overlay).
2. Minimal wiring in `aic_model/aic_model/RLInsert.py` behind `RL_INSERT_BOARD_SEARCH`.
3. Committed on a `board-search` branch.
4. Deployed as a new bumped asset, bound to `aic_model`, with logs showing the search
   running and framing the board within 3 moves.

Report back: the final HSV/solidity constants you settled on, how many moves it took to
frame the board in the test trial, and the new asset name/version.
