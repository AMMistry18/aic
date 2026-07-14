# Taskboard board-search handoff (v38)

Date: 2026-07-14

Branch: `board-search`

Base branch: `flowstate-rl-deploy-and-docs`
Flowstate status: v38 is installed and starts, but the board-only motion is not
working yet.

## Start here

The perception/search code is implemented, unit-tested, built into a full
Flowstate image, and deployed as v38. The detector finds the board in the live
camera image. The current blocker is downstream of perception: Flowstate accepts
the Cartesian mode/target, but the measured TCP never moves. The next task is to
fix Flowstate controller/session ownership, then rerun the existing v38
`board_search_only` skill call. Do not start by retuning HSV or rewriting the
image Jacobian.

Use this serial process while debugging (no parallel motion session):

1. Tare force/torque.
2. Run `switch_to_aic_controller`.
3. Configure lifecycle node `aic_model`.
4. Activate lifecycle node `aic_model`.
5. Run **Insert Cable Skill** with `id = board_search_only`.
6. Run `switch_to_default_controller` only after the board-only skill finishes.

Disable/remove the second normal **Insert Cable Skill** node during this test.
The last trial started a normal insertion immediately after the failed board-only
goal, which makes controller ownership and logs harder to interpret.

## What is implemented

- `aic_model/aic_model/board_search.py`
  - Pure OpenCV detector for the dark, low-saturation taskboard.
  - Chooses the camera with the largest valid board blob.
  - Uses the selected camera optical U/V axes transformed into `base_link` for
    probe motion.
  - Waits for measured TCP settling and for a camera frame with a newer header
    timestamp before estimating image motion.
  - Uses measured TCP displacement, not the requested displacement.
  - Refuses blind movement and caps each move at 5 cm, with at most 3 moves.
  - Includes an offline image/debug-overlay CLI.
- `aic_model/aic_model/RLInsert.py`
  - `Task.id == "board_search_only"` runs only `BoardSearch` and returns.
  - Normal task IDs retain the existing insertion behavior.
  - `RL_INSERT_BOARD_SEARCH=1` optionally prepends search to a normal task; the
    deployed image intentionally leaves this off.
  - Includes the pre-existing, formerly uncommitted scripted `+Y` stall-nudge
    recovery. It is enabled by default and was not overwritten.
- `docker/aic_model/Dockerfile.student_flowstate`
  - Copies `board_search.py` into both source and installed package locations.
  - Uses `RL_INSERT_CONTROL_MODE=script`.
  - Sets `RL_INSERT_BOARD_SEARCH=0` and reserves `board_search_only`.
  - Uses the full router-safe entrypoint, not a thin overlay image.
- `aic_model/test/test_board_search.py`
  - Detector, camera selection, fresh-frame synchronization, motion bounds, and
    success/failure behavior are covered offline.

Current detector defaults in `board_search.py`:

```text
HSV V:              15..95
HSV S max:          80
minimum area:       0.03 of image
minimum solidity:   0.55
morphology kernel:  9 px
center tolerance:   0.08 of image dimensions
probe:              0.020 m
maximum move:       0.050 m
maximum moves:      3
TCP settle:         0.003 m tolerance, 3.0 s timeout
fresh frame:        2.0 s timeout
```

## What was verified

Local tests:

```bash
PYTHONPATH=aic_model \
  .pixi/envs/default/bin/python -m pytest -q \
  aic_model/test/test_board_search.py \
  aic_model/test/test_rl_insert_contract.py
# 10 passed
```

`py_compile` and `git diff --check` also passed. A saved real camera frame was
detected as the board (area about 0.283 of the image, clipped at the top), and the
synthetic servo test completes within three moves.

The v38 container starts the correct policy in Flowstate:

```text
Loading policy module: aic_model.RLInsert
Loaded policy module aic_model.RLInsert
Using policy: RLInsert
```

## What is not working

The latest live `board_search_only` run produced:

```text
[board_search] selected center_camera: area=0.200 centroid=(489.6,821.7) border=True
[board_search] center_camera probe axes in base: u=[-0.365, 0.904, 0.223] v=[0.908, 0.399, -0.13]
Setting cartesian mode...
Successfully set target mode
[board_search] TCP did not settle within 3.0s; remaining_error=0.0200m
[board_search] probe motion was clipped to zero
[board_search] board-only task complete: False
```

This proves the detector and camera-axis lookup ran, and the target-mode request
was accepted. It also proves the commanded 2 cm probe produced effectively zero
measured TCP displacement. Earlier variants were corrected for base-axis
singularity (v36) and stale camera frames (v37); v38 now stops safely when the arm
does not move.

Controller logs around the same trials contained both of these signals:

```text
Part 'arm' ... entering Safety Action
Contexts are not equal. This can happen if parallel sessions exist.
```

Therefore the leading issue is a competing/expired motion session, wrong
controller ownership, or a safety-state transition in the Flowstate process. A
one-shot MotionUpdate is supported by the controller and is held, so changing
board search to continuously publish is not the first fix.

### Next debugging checklist

1. Build a Flowstate process containing only the six serial steps listed at the
   top of this document.
2. Confirm there is only one `agent_bridge`/motion session and no parallel
   `switch_to_default_controller` path.
3. Confirm the arm is out of Safety Action and hardware motion is enabled before
   invoking `board_search_only`.
4. Run v38 unchanged and watch measured TCP. The first probe should move about
   20 mm along the logged camera U axis.
5. If TCP moves, inspect the next `[board_search]` line for pixel displacement,
   measured displacement, and the estimated Jacobian. Only then tune search.
6. Success requires a final log reporting `board-only task complete: True` and a
   visually centered, fully contained board. This has **not** been achieved live.

## Current Flowstate deployment

```text
organization:     tar-2@xfa-prod-aic-us
solution:         582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
service instance: aic_model
asset name:       ai.intrinsic.aic_model_v38
installed asset:  ai.intrinsic.aic_model_v38.0.0.1+144f4ebff271742bd2c971ae709011b6ba279a45a7f32b31a5dc6026c91c619b
local image tag:  my-solution:student-flowstate-script-v38
bundle SHA-256:   127e3338ab8ceca17029b1cb78c44450deafae7a0ca7c0db3abf6fd5182ab15a
image-tar SHA-256: 2fb125ac454731d99b3a9974c430dd067fa8d458cb4c14859699da3393d0752b
```

The local v38 bundle was staged at
`/private/tmp/aic-script-v38/aic_model_v38.bundle.tar`. That directory and the
current CLI binaries are temporary and will disappear on reboot. The manifest
itself is preserved in Git at
`deploy/flowstate/aic_model_v38.manifest.textproto`.

Assets v35 through v38 remain installed for rollback. The live `aic_model`
binding points to v38. `inctl service state list/get` has sometimes omitted a
healthy service in this solution, so use startup logs as the authoritative
startup check.

## Set up `inctl` and `inbuild` on a new laptop

These are Intrinsic Linux/AMD64 tools, not repository dependencies. There is no
stable public direct download URL. Sign into:

<https://flowstate.intrinsic.ai/o/tar-2@xfa-prod-aic-us>

Open **Set up development environment** / developer tools and download both the
Linux AMD64 `inctl` and `inbuild` binaries. Keep them outside `/private/tmp`:

```bash
mkdir -p ~/.aic-flowstate/bin ~/.aic-flowstate/inctl-home
cp ~/Downloads/<inctl-download> ~/.aic-flowstate/bin/inctl-linux-amd64
cp ~/Downloads/<inbuild-download> ~/.aic-flowstate/bin/inbuild
chmod 755 ~/.aic-flowstate/bin/inctl-linux-amd64 ~/.aic-flowstate/bin/inbuild
```

The checked-in wrapper runs `inctl` in a Linux/AMD64 Debian container and keeps
auth state in `~/.aic-flowstate/inctl-home`:

```bash
scripts/flowstate/inctl.sh version
scripts/flowstate/inctl.sh auth login --no_browser \
  --org tar-2@xfa-prod-aic-us
```

The login command is interactive. Open the URL it prints and paste the generated
token into the CLI prompt. Never save the token in Git, a script, or chat.

`inbuild` is a separate required binary; this version of `inctl` does not have a
working `bundle` command. See `docs/FLOWSTATE_DEPLOY_RECIPE.md` for the complete
v39 build/bundle/install/rebind commands.

## Moving this work to another repository

For another clone/fork of this repository:

```bash
git remote add board-search-source https://github.com/AMMistry18/aic.git
git fetch board-search-source board-search
git switch -c board-search --track board-search-source/board-search
```

If the destination repository has diverged, cherry-pick these implementation
commits in order and resolve `RLInsert.py` against its insertion code:

```text
4f93231  Add bounded taskboard framing search (also preserves +Y stall nudge)
191b36e  Probe taskboard in camera image axes
b49ddb9  Synchronize board probes with fresh camera frames
```

Then also take the later handoff/deploy commit on this branch. The files that
must survive a manual port are:

```text
aic_model/aic_model/board_search.py
aic_model/test/test_board_search.py
aic_model/aic_model/RLInsert.py              (small board dispatch + +Y nudge)
docker/aic_model/Dockerfile.student_flowstate
deploy/flowstate/aic_model_v38.manifest.textproto
scripts/flowstate/inctl.sh
docs/BOARD_SEARCH_V38_HANDOFF.md
docs/FLOWSTATE_DEPLOY_RECIPE.md
```

The taskboard geometry/mesh is already tracked under
`aic_assets/models/Task Board Base/`; the separate untracked `taskboarinfo/` copy
is redundant and is not required by the policy.
