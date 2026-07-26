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

## Wave 2 — DONE (58aff8c), unvalidated in sim

Motivated by `diagnositics sfp 5` (2026-07-26, 9 goals): 4 full insertions
(events fired, all named `sfp_port_1` — macro's problem, not the policy's),
3 stalls at ~36 mm believed depth, 2 mouth stalls at ~1.8 mm from a Flowstate
scene bug (port spawned inside a port — **discount these, not a policy issue**).

The ~36 mm stalls decoded cleanly: axial 6.3-7.2 N with 4.3-4.8 N lateral bind,
nudge_applied pinned near 0.1 mm. 7.4 N is exactly `5 mm overtravel x 500 N/m`,
i.e. the setpoint ceiling's entire spring budget — the plug was pressing as hard
as the config allowed and could not advance. Not an obstruction; a budget.

Shipped: dwell holds `INSERT_DEPTH_M + overtravel` with corrections carried
(the old fixed 45.8 mm hold sat BEHIND a plug at the pad, so the impedance loop
pulled it off); target force 8->10 N, cap 10->12 N, overtravel 5->8 mm;
freeze/contact gated on plug-frame axial force (18 N norm abort untouched);
seat-align gains 3e-5/0.004 -> 1e-4/0.01 with caps 0.7 mm / 0.7 deg plus new
per-sample slew limiters (`seat_align_max_step_m`, `_tilt_step_rad`).

**Trap found while doing this:** both release Dockerfiles baked
`RL_INSERT_V50_TARGET_FORCE_N=8.0` / `SEAT_FORCE_CAP_N=10.0` as ENV. Baked ENV
beats source defaults and Flowstate takes no runtime knobs, so the retune would
have been a silent no-op in the image. Pins moved; a test now asserts the
Dockerfile ENV matches `V50Config`. **Any future seat-force change must move
both.**

Unverified predictions to check on the next field run: a plug that reaches the
pad now presses ~3.5 N through the dwell instead of retreating; press at a 36 mm
stall rises ~7.0 -> ~8.6 N; SEAT_WRENCH `nudge_applied_mm` should reach
~0.3-0.45 mm under 4-5 N lateral instead of ~0.1 mm. If deep stalls persist at
the SAME depth with the higher press, the cause is geometric (alignment or
believed-depth error), not force budget — go to Wave 3 rather than raising
force further.

## Wave 3 — DONE (37a02a2), unvalidated in sim

Built after diag-6: full insertion otherwise solved, 1 wedge in 5 runs, and a
wedge was terminal because `run()` was single-shot.

**Outcome semantics changed — read this before touching the seat path.**
`WEDGED` = stuck short of the seat (excursion check, or no advance while still
in the bore). `STALLED` = at seat depth, event never arrived. Only `WEDGED`
retries; backing out of a `STALLED` would discard an insertion that may already
be physically complete. `HARD_FAILURE` (wrong-port event, sustained over-force)
never retries.

Order on a wedge: **rescue first, retract only if there is no rescue.** One
rescue per retract. The rescue is measured against the *original* port
perception so its 8 mm excursion cap bounds total drift across all retries.

Retract is two-phase on purpose: straight out along the port axis holding the
*measured* rotation (correcting a cocked plug in place cams it into the cage),
then back to the run-start pose. It drives `set_pose_target` directly because
`next_persistent_depth` clamps to at least the current depth — **the seat
setpoint physically cannot retract.** Clearing the mouth but stopping short of
the start pose still counts as success.

Retries are unbounded by count; the action deadline is the only terminator and
every cycle consults it.

**Budget raised 45 -> 150 s** (overlay `RLInsert.ACTION_TIME_BUDGET_S` + both
Dockerfiles). Justification: engine per-task `time_limit` is 180 s in every
shipped config, and the scorer's 60 s is where the duration bonus reaches zero,
**not** a cutoff (`ScoringTier2.cc:945` feeds only
`GetTaskDurationScore`). So a retry costs ~3 remaining duration points and can
convert a 38-50 point partial into 75. **If the Flowstate macro's pre-insert
phase is long, lower this** — a run that never returns scores nothing.

Not done, deliberately: no lateral jitter between attempts. A retry re-perceives
and re-aligns, so it is not a bit-identical repeat, but if retries are observed
reproducing the *same* wedge, adding a small per-attempt offset is the next
lever.
