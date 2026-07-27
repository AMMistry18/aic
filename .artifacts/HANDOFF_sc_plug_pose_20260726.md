# Agent Handoff — SC plug-pose, 2026-07-26

**What this session did:** collected the SC plug-pose dataset that had never
existed, trained the model, diagnosed why it missed spec, fixed that, and
measured **0.2733 mm median on a held-out split with all gates passing**.

Technical detail lives in `docs/SC_PLUG_POSE_RESULTS.md` (committed). This
document is state and next actions.

---

## 1. Repository state — read this first

```
origin/main          d499d5f  2026-07-26 01:33  ← authoritative branch
origin/sc-plug-pose  139fcc9  2026-07-26 02:39  ← this work lives ONLY here
```

**The two have diverged.** `main` contains `d860470` (the plug-pose base) but
**not** `139fcc9`. Meanwhile `main` gained SFP Waves 0–3, the Flowstate deploy
service, and `.artifacts/HANDOFF_sc_insertion_20260726.md`.

`docs/HANDOFF.md` says *"`main` is the authoritative development branch. Do not
infer the active implementation from an old feature branch."* By that rule this
work is currently invisible to anyone following the documented workflow.

> **Open decision: merge `sc-plug-pose` into `main`.** Not done here because
> `main` is shared and `sc_controller.py` is under active edit there. The files
> in `139fcc9` do not overlap what landed on `main`, so it should be clean.

Note `8254648` (on the branch) asserts "work is on main, not a branch." That was
true when written at 21:20; `main` has moved since.

## 2. The result

Measured by `validate_sc_plug_pose.py --mode dataset` against simulator TF
ground truth, **crop-refine enabled**.

| Gate | val | test | Limit |
|---|---|---|---|
| `position_error_mm.median` | **0.2733** | 0.2633 | ≤ 0.4 |
| `lateral_error_mm.p95` | **0.3291** | 0.3434 | ≤ 0.725 |
| `group_miss_rate` | **0.0** | 0.0 | ≤ 0.01 |
| `axis_p95_deg` | 1.88 | 1.81 | ≤ 3.0 |
| `all_gates_pass` | **true** | true | |

Quote the **val** number: the crop parameter was swept on test, never on val.
They agree to 0.01 mm.

Weights: `aic_example_policies/.../ros/weights/best_sc_plug_pose.pt`, mirrored
into the v50 overlay. Reports in `docs/reports/`.

This closes §9 item 3 of `HANDOFF_sc_insertion_20260726.md` ("SC plug-pose
training data. Zero exists.") and satisfies its §5 threshold — lateral p95
0.329 mm sits inside the 0.725 mm vertical clearance that document names as the
binding axis.

## 3. Settled — do not re-derive

**The miss was bias, not noise.** Uncropped inference gives 0.456 mm median
while keypoint error is 1.47 px — *better* than the 2 px the handoff's budget
asks for. Better keypoints, worse tip error. Per-keypoint signed error
(`kpt_bias_analysis.py`, committed) shows `mean_dx ≈ 0` but `mean_dy` negative
for **all eight** keypoints, averaging **−1.05 px**. At f = 1236.6 px and 280 mm
that is 0.238 mm, and the observed error floor was 0.272 mm with no group ever
beating it. The floor *was* the bias.

**Two hypotheses were tested and refuted.** Recorded so they are not retried:

1. *"`SC_PLUG_LOCAL_KEYPOINTS_M` z-placement disagrees with the mesh."*
   Impossible. `DataCollectorScPlugPoseGT` builds labels by projecting that same
   constant and `ScPlugPoseEstimator` fits against it — it cancels exactly. A
   mismatch moves labels and fit target together and recovers an identical frame.
2. *"Rear-plane keypoints 4–7 are occluded in the gripper, so they are worse."*
   Measured rear/near bias ratio **1.01×**. No.

**`crop_pad_scale` is load-bearing and fails deceptively.**

```
   pad   miss_rate   pos_med   pos_min   lat_p95   kpt_px   n_pose
  none      0.0000    0.4562    0.2719    0.4121   1.4698      135
     2      0.9481         —         —         —   4.0268        7
     3      1.0000       nan       nan         —   8.6825        0
     4      0.0000    0.2683    0.0099    0.5119   1.7533      135
     6      0.0000    0.2633    0.0350    0.3434   1.2579      135  <- default
     8      0.0000    0.2936    0.0210    0.3887   1.2640      135
    12      0.0000    0.4562    0.2719    0.4121   1.4698      135
    16      0.0000    0.4562    0.2719    0.4121   1.4698      135
```

- **Too tight looks like success.** At pad 2.0, 94.8% of groups yield no pose and
  the surviving 5% report a *better* median (0.411 mm) purely by survivorship.
  **Read `group_miss_rate` before any error metric.**
- **Too loose is a silent no-op** — pad 12/16 reproduce the uncropped result bit
  for bit. That also validates the remapping introduces no offset of its own.
- Pad 4–8 all hold; 6.0 is mid-plateau, not on the cliff between 3 and 4.

Enable with `AIC_PLUG_POSE_CROP_REFINE=1` (or `crop_refine=True`). **Default is
off**, so the SFP path is unchanged; `crop_pad_scale` defaults to 6.0 when on.

## 4. Pipeline bugs fixed — this is why old runs "failed"

Jobs `3329045` (cancelled 8h22) and `3329874` (**TIMEOUT 12h00**) almost
certainly had their data on disk already. Two silent bugs:

1. **The collector never exits.** It completes every trial, writes the dataset
   yaml, runs its lifecycle to `on_shutdown` — then the process stays alive. The
   script blocked in `wait` until Slurm killed the job. Fixed with a reaper that
   detects the completed shutdown, allows 45 s grace, forces termination, and
   marks the non-zero exit legitimate so integrity checks still gate.
2. **`aic_engine` loses a startup race.** It gives up if `/clock` is not up
   within 10 s, which it misses when Gazebo falls back to headless rendering
   (`libEGL: failed to create dri2 screen`). It dies and *nothing drives any
   trial*. Its message is `aic_engine-7: process has died`, which the existing
   crash pattern (`component_container.*process has died`) did not match.
3. **No liveness guard existed.** Added a progress watchdog: no trial within
   7 min of start, or a 7 min stall mid-run, stops with a diagnosable reason.

Also fixed: `SC_TRAIN_AFTER_COLLECTION=1` exec'd `${repo_root}/train_sc_plug_pose.slurm`
but the file is at `.tacc/`. That path had clearly never run.

**Gazebo degrades across trials:** 9.4 s → 18.7 s per trial over ~175 trials,
then plateaus. A restart resets it to ~7 s. Collect in batches via
`SC_TRIAL_START` — faster *and* safer. `qvrtx` allows only **one running job per
user**, so batches are sequential; `amd-rtx`/`h100` were full and `pvc` is Intel
so Gazebo cannot use it.

## 5. Corrections to `SC_PLUG_POSE_HANDOFF.md`

That document is otherwise accurate and worth reading. Three things in it are
wrong and cost time:

- **"`DataCollectorScPlugPoseGT` has never been run for SC. Not once."** False.
  `sacct` shows four SC datagen jobs on 2026-07-20 (0:42, 2:41, 8:22, 12:00
  TIMEOUT). The job directories were deleted, so only Slurm accounting survives.
  The author had no way to see it.
- **"The eight dots must sit on the blue duplex housing."** The sim renders the
  grasped plug **beige**; the blue duplex parts are the panel-mounted *ports*.
  Following this literally would lead you to reject good data. The keypoints
  derive from SDF collision bounds, so material colour is irrelevant.
- **Test counts** (97, later pinned at 92) do not match a fresh branch clone.
  Only the 19 plug-pose tests were run this session; they pass.

## 6. TACC environment

Everything is staged and reusable. **The dataset alone is ~2 h of node time.**

```
$WORK = /work2/11590/satya_a/stampede3
  aic-sc-plug-pose-src-20260725/       repo + built pixi env (.pixi/envs/default)
  aic-sc-plug-pose-datagen-20260725/
    containers/aic_eval_pinned.sif     1.4 GiB
    dataset/                           4,050 imgs, 3240/405/405, 1.2 GiB
    training/best_sc_plug_pose.pt      also committed to git
    training/reports/                  all validation JSON
```

**Access needs an MFA-authenticated socket.** Key-only auth is refused. Have the
user run, then reuse `-S`:

```bash
ssh -M -S ~/.ssh/cm-stampede3 -o ControlPersist=8h -fN stampede3.tacc.utexas.edu
ssh -S ~/.ssh/cm-stampede3 stampede3.tacc.utexas.edu '<cmd>'
```

Two traps that cost ~45 min each:

- **`pixi install` without `RATTLER_AUTH_FILE` hangs forever.** It reaches for
  the system keyring, spawns `gnome-keyring-daemon`, tries to talk to a terminal
  that does not exist in a batch job, and leaves the process group **stopped**
  (`STAT=T`) until walltime. Looks exactly like a slow install. Set
  `RATTLER_AUTH_FILE=$WORK/.rattler_auth.json` (containing `{}`).
- **Use `pixi install --frozen`.** From `pixi.lock` it takes 4 minutes.
- Do **not** `set -u` in a slurm script that does `module load tacc-apptainer` —
  its bash-completion reads unbound variables and kills the job in 0 s.

`/work2` quota reports ~300 GB more than `du` finds — files owned by this UID
outside the user's own tree. ~680 GB headroom, not a constraint, but it makes
quota numbers untrustworthy.

## 7. Open, in priority order

1. **Merge `sc-plug-pose` → `main`** (§1). Until then this is invisible.
2. **Wire the estimator into the controller.** `docs/SC_PLUG_POSE_WIRING_PATCH.md`
   still applies unchanged, including the contract change where `_tip_pose()`
   may return `None` and **all four call sites must stop, never fall back to the
   constant**. Do this after `sc_controller.py`'s parallel work lands.
3. **`.tacc/train_sc_plug_pose.slurm` does not enable crop-refine**, so its
   automatic validation reports the ~0.45 mm uncropped number. Set
   `AIC_PLUG_POSE_CROP_REFINE=1` there, or expect to be confused.
4. **Re-tune the rejection gates.** `SC_MAX_REPROJECTION_ERROR_PX = 6.0` and
   `SC_MAX_KEYPOINT_RMSE_M = 0.0035` are still SFP inheritances. Measured
   distributions now exist in `docs/reports/` — this is open question 3 of the
   original handoff.
5. **Rotation is unimproved** — 1.03° median, same as uncropped. The bias has an
   x-dependent component (keypoints at local x = +10 mm carry ~−1.36 px vs
   ~−0.74 px at −10 mm) which is a small roll and survives cropping. Passes the
   axis gate with margin; only matters if tighter angular accuracy is needed.
6. **Crop-refine costs a second inference pass per view.** Fine offline, never
   benchmarked in the control loop.
7. From `HANDOFF_sc_insertion_20260726.md`, still untouched here: run
   `scripts/enumerate_tf_frames.py` while holding the plug (§9.1), the
   outer-camera contribution threshold (§7), and
   `RL_INSERT_SC_STRICT_PORT_EVENT=1` before submission (§8).

## 8. Conventions (unchanged, still bite)

- **Edit BOTH copies.** `aic_model/aic_model/X` and
  `docker/aic_model/v50_overlay/aic_model/X` byte-identical. `diff` after every
  edit. `sfp_plug_pose.py` was mirrored this session.
- **Never `git add -A`.** `deploy/flowstate/*` is intentionally untracked, as is
  `docker/aic_model/Dockerfile`. Stage explicitly. Watch for `chmod` mode flips
  sneaking into the diff.
- **Do not modify `sc_controller.py`** — parallel work.
- Tests actually run this session:
  ```
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .pixi/envs/default/bin/python -m pytest \
    aic_model/test/test_sc_plug_pose.py aic_model/test/test_sc_plug_pose_trials.py -q
  ```
  19 pass. Never run the whole test directory.
- **The shipped answer must be the model, not a constant.** A solved fixed
  transform is legitimate as a debug scaffold only; hardcoding is disallowed and
  `sfp_plug_pose.py` sets the house pattern with no fixed-grasp fallback.
