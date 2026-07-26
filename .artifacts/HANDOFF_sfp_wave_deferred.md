# Deferred SFP items — decided 2026-07-25/26 wave planning

Decisions made while implementing Wave 0+1 (see HANDOFF_sfp_20260725_night.md for
the source analysis). These are deliberate deferrals, not omissions.

## TODO next sim session: TF harvest + camera frame collection (Wave 0 execution)

Code and tooling are committed (8757b7e); only the *runs* are pending. Blocked
on 2026-07-26 by an NVIDIA driver/library mismatch (kernel module 595.71.05 vs
userspace 595.84, persistenced dead) — **reboot the workstation first**, then:

1. `nvidia-smi` must succeed; the launcher preflight's container EGL probe
   should report 1-2 devices.
2. **TF harvest** (~2 min, no insertion): launch the sim per
   `scripts/sc_plug_pose_collect_local.sh`'s docker recipe (bridge net, root,
   `-p 7447:7447`, `--gpus all`), then run `scripts/enumerate_tf_frames.py
   --json <out>` with the collector's Zenoh env
   (`RMW_IMPLEMENTATION=rmw_zenoh_cpp`,
   `ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'`).
   Yields: the `sfp_port/...` ground-truth frames (perception-error oracle) and
   the two-sample motion verdict on `selected_sfp/sfp_tip_link` (expected:
   static decoy).
3. **Frame collection** (~5 trials, ~45 images): same sim, run
   `DataCollectorSfpPlugPoseGT` via `generate_sfp_plug_pose_trials.py` +
   `ros2 run aic_model aic_model -p policy:=aic_example_policies.ros.DataCollectorSfpPlugPoseGT`
   (mirror `.tacc/sfp_pose_datagen_after_simdist2.slurm`, output dir via
   `AIC_SFP_PLUG_POSE_OUTPUT_DIR`). Then
   `scripts/measure_sfp_camera_asymmetry.py --images <dir> --json <out>` for
   the per-camera scorecard. Decides crop-refine vs angle-diverse retrain
   (playbook Item 2 vs Item 3). Zero camera images exist anywhere locally
   today, so this collection is the only path to the measurement.

## Deferred: standoff re-perception before mouth entry (was item #13)

Re-solving the grasp transform via `_activate_plug_pose` at the aligned standoff
would bound kinematic drift between priming and seating. **Deliberately not
implemented**: at the aligned position the port is most likely occluded by the
cable itself, so a re-perception there would either fail its freshness gates or,
worse, lock onto a cable-biased estimate. May become viable later if re-perception
is done during the retry lift (which raises the plug clear of the port) — the
Wave-3 lift/re-perceive retry covers most of the same drift window from a
vantage point that is actually visible.

## Deferred: port selection honoring the requested slot (was item #1)

Every 2026-07 field run inserted into `sfp_port_1` when `sfp_port_0` was
requested; scoring credits only the requested port. **Not fixed in the
controller by user decision**: the Flowstate macro layer owns steering to the
requested port and will account for it. The controller keeps nearest-to-tip
selection. If macro-side handling changes, revisit `_select_sfp_candidate`
(RLInsert.py) which ignores the requested slot today.

## Wave 2 (approved, not yet implemented)

- Event-dwell hold at `INSERT_DEPTH_M + seat_overtravel_m` and carry
  `acc_lat/acc_tilt` corrections into `fixed_tip` (pad sits ~1 mm past 45.8 mm).
- Axial-force gating for the 10 N freeze (norm-based today; axial already
  computed and logged).
- Seat-align authority raise: cap 0.4 -> ~0.7 mm (NOT 1.0 — the 0.4 cap exists
  because aggressive correction previously hurt), keep low-pass, add per-tick
  rate limit.

## Wave 3 (approved, not yet implemented)

- One bounded lift + re-perceive + re-seat retry on STALLED (never on
  HARD_FAILURE). Lift must be high enough to re-run YOLO pose for BOTH plug and
  port (user: time budget is not the constraint). Tests pinning no-recovery
  behavior (test_v50_controller.py:400-426) must be consciously updated.
