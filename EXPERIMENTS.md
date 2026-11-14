# Reproducing the research artifacts

November 14: frozen final44 continues at 92 primary attempts (SC44/46, SFP31/46),
with 80 ablations pending. All 110 frozen source/model hashes match. A separate
faster lip-search development insertion passed the unchanged independent
seating/contact/2000 Hz force checks: action 47.0710 sim s, raw peak 8.881041 N.
`experiments/export_lip51_evidence.py` completed; all 31 exported checksums in
`experiments/evidence/lip51/` pass. Original headless final imagery was inspected;
arm occlusion means depth still requires the independent traces. Snapshot and
failure classification: `experiments/checkpoints_20261114/`. No final trial was
replaced and no development result is pooled into final44.

Latest SC cache pair completed: action 26.1890 sim s versus 27.6735 baseline,
port measurement 45.833 versus 53.654 wall s, raw peak 3.060299 N and complete
2000 Hz coverage. The original independent seat/contact checks passed.
`experiments/export_cache49_evidence.py` preserves 29 checksummed raw/camera/code
files in `experiments/evidence/cache49/`. It is one reused development case,
separate from final44. The next bounded SFP lip-search check remains pending
under `experiments/lip_followup51/progress.json`.

November 13 actual motion result: runtime47v3 SFP seating independently passed
with complete 2000 Hz coverage; side-wall registration 11.8295 sim s, action
51.9735 sim s, raw peak 4.684120 N. The same-case baseline was 16.8725 /56.1585
sim s. Exported camera, raw-force/contact and source evidence:
`experiments/evidence/motion48/` (29 checksummed files), reproduced by
`experiments/export_motion48_evidence.py`. This is one reused development pair;
wall time did not improve. SC cache physical qualification is queued through
`continue_rim49_v2.py`; `continue_lip51.py` follows with bounded axial search,
which has passed synthetic controller checks only. The original failed BOM
source comparison remains recorded; no scientific threshold was relaxed.

Later November 13: complete SC/SFP GPU runtime checks passed, with 54 compact
checksummed evidence files exported to `experiments/evidence/runtime47/`.
Development now runs between final-trial boundaries through versioned
`development_slot47.py` leases, never beside another complete world. The frozen
main coordinator resumes unchanged after each slot. Active finite follow-ups,
offline speed measurements, retained failed startup and rejected angular
candidates are documented in `experiments/efficiency_followup48/README.md`.

November 13 short development checks: see `experiments/parallel_dev46/README.md`
for measured CPU/GPU cropped-pipeline comparisons, angular fits, actual-model
thread-guard integration, CAD feasibility and independent scalar SAT checks.
The finite offline queue completed all six image comparisons and the thread
check. `experiments/candidates/efficiency46/` contains the reviewable runtime
repair; only `.artifacts/dev46_workspace` received it. Production/final sources,
weights, thresholds, physics and force settings are unchanged. No new physical
insertion success is claimed. The second Gazebo renderer was stopped after
setup problems and a final sensor-recording gap during overlap; the short
physical driver now refuses to run beside the active final evaluator. All
five development setup/cancellation attempts and the failed final attempt
remain recorded. Offline work is bounded to one CPU core; unrelated WSL work
is untouched. Current final status remains in the separate final44 ledger.

November 13 final-study checkpoint: eight completed final policy trials
independently verified, SC 4/4 and SFP 4/4, all with complete 2000 Hz force
coverage. Maximum raw peak: SC 3.150219 N, SFP 3.404254 N. Successful action
medians: SC 28.33825 simulated /293.179765 wall seconds; SFP 58.41375
simulated /631.072275 wall seconds. These are partial final results, not a
completed reliability study.

The ninth selected case, SC 700026, never reached controller-ready or policy
execution. Its scene and logs are retained as a setup failure. A Docker-stop
30s timeout escaped cleanup and stopped the original coordinator; the initial
startup error was not persisted, so no further startup cause is asserted.
All-attempt accounting is SC 4/5 and SFP 4/4; completed-policy denominators
remain SC 4/4 and SFP 4/4. This selected case is not retried or replaced.

`experiments/resume_full_evaluation44.py` resumed at the next frozen manifest
entry, SFP 710039. It preserves the exact policy, model, physics, verifier and
manifest hashes and separately records its own coordinator source/hash. It
reconciles uncommitted outcomes and transport cleanup without changing policy
action budgets. Two bookkeeping regressions reject skipped failures, altered
grasps and duplicate execution. No preliminary simulation was added. Current
status and partial results remain in `experiments/full_evaluation44/progress.json`;
91 primary cases including the active trial and 80 ablations remain. The
manuscript's older snapshot is preserved until a reviewed results update.

November 12 final-study launch: at the user's explicit request, remaining
preliminary simulations were cancelled and the recovered baseline entered the
proper final evaluation. `--start-final-now` froze source/checkpoint hashes and
a 100-case manifest before any final outcome: 50 SC and 50 SFP, followed by
80 paired ablation trials on a 40-case subset fixed before outcomes. The first
selected case is SC 700029. Four development ablation simulations were skipped.
The contact-control simulation had already completed; its retained recording
qualified with 1334 physical requested contact pairs, without another run.
All per-trial setup, independent seating/contact checks, continuous dwell and
complete 2000 Hz force auditing with the 20 N measured ceiling remain enabled.
No final-source changes, tuning, replacement of failures or success-dependent
stopping are permitted. Freeze/manifest and live results are preserved under
`experiments/full_evaluation44/`. Earlier preliminary-check requirements below
are superseded by this recorded user instruction. The manuscript's older
snapshot is unchanged; final outcomes will be reported separately.

November 12 late-evening checkpoint (supersedes earlier running snapshots):
The user stopped speed43 after five completed development trials: SC 0/3,
SFP 1/2 independently verified. SFP dev_45002 was not attempted. The fifth
SC trial completed its policy/recorder before the coordinator disappeared;
it was independently analysed and archived without rerunning. Its complete
2000 Hz raw-force audit measured 109.895620 N, exceeding the 20 N ceiling;
the slower policy stream reported only 2.090200 N. No force qualification
is claimed for this candidate. All five outcomes and camera/log evidence
are preserved under `experiments/recovery_20261112/speed43_development/`.
The seven candidate source files were restored from saved baseline copies
and their hashes checked. No further speed43 trials will run.

The recovered baseline replay remains SC 9/9 and SFP 6/8 completed policy
trials verified; one additional SFP bootstrap failure and one cancelled
SC duplicate setup are retained separately. SFP failures were the unchanged
2 px perception gate (measured 2.393 px) and inadequate continuous recording.
A separate dev_45006 recording-coverage regression passed with a 1.52 s
continuous tail, complete raw-force coverage and 3.116151 N peak. This rerun
does not relabel the failed replay or convert the replay into SFP 7/8.

The full-study coordinator was resumed at the unopened-final development
boundary using the restored baseline, with no repeat of the speed study.
Measurement, fresh physical contact-control and component-removal checks
remain required before the planned 100 primary and 80 paired final trials.
Live state: `experiments/full_evaluation44/progress.json`. No final result
is inferred from these development outcomes. The manuscript keeps its
explicit older frozen snapshot until a separately reviewed results update.

Full-study launch, November 12: `experiments/run_full_evaluation44.py` owns the
serial queue. Its live ledger is `experiments/full_evaluation44/progress.json`.
The recovered historical replay finished with SC 9/9 policy trials verified
and SFP 6/8, plus one SFP controller-bootstrap failure and a cancelled duplicate
SC setup retained separately. The two completed SFP failures were a 2.393px
rim rejection against the unchanged 2px gate, and insufficient independently
recorded dwell. The latter had a 1.68s pose-stream gap followed by only 0.99s
of contiguous terminal evidence; its original failed verdict is preserved.

The new recorder collects a fixed coverage-only post-policy tail: 1.5s of
continuous pose timestamps, <=30ms gaps, with a 45s wall budget. It does not
inspect geometry to select a successful tail. The original 1s seating gate
is unchanged. Controller activation now allows 120s for cold renderer/sensor
startup; no physical tolerance changed. A fresh dev_45006 regression is
running before the speed comparison. 78 focused current measurement/force/
verification checks passed; one legacy replay test requiring a missing old
artifact was explicitly deselected.

The full queue then runs six speed-development cases, selects by the frozen
10-percent speed/all-six-verifications rule, qualifies a fresh physical SFP
collision control and four development ablation trials, and freezes sources
and weights before generating any final cases. The final stage contains 100
primary trials (50 SC, 50 SFP) and 80 matched component-removal trials. The
explicit final runner calls the existing isolated trial executor; the legacy
development-only CLI remains restricted to development seeds. Final outcome
data cannot select or tune the policy. Source changes, inadequate disk space,
or repeated environment failures stop the job for inspection. Raw results and
camera evidence are exported outside `.artifacts`; immutable weight objects
are shared by hard link across new source snapshots to avoid repeated copies.
The manuscript's older immutable snapshot is deliberately not overwritten by
this live study.

November 12 recovery: new perception checkpoints are preserved with SHA256
records under `experiments/recovery_20261112/checkpoints/`. Original missing
checkpoint identities are not claimed. Fresh SC dev_42000 and dev_43001 and
SFP dev_45005 independently passed; the remaining historical cases are running.
Live ledger: `experiments/recovery_20261112/historical_batch_replay/progress.json`.
Initial raw evidence export: `experiments/recovery_20261112/verified_evidence/`.

The baseline source freeze remains active. `experiments/prepare_speed43.py`
prepared seven candidate files outside production; eight checks in
`experiments/test_speed43.py` pass. `experiments/run_speed43_after_replay.py`
waits for all 17 baseline outcomes, checks original/candidate hashes, preserves
baseline files, then runs six matched development cases and exports each result.
It stops if the baseline stops; it never opens final seeds. Its progress is
`experiments/recovery_20261112/speed43_development/progress.json`.
The proposed 100-case final protocol is
`experiments/configs/recovered_robustness43_protocol.json`; it is not an executed
result or an automatically started final evaluation. The historical snapshots
below describe earlier checkpoints.

Latest checked November 6: `qualification41v2` stopped automatically on a
setup/hold-validation failure for SC case dev_42002, before its insertion
policy ran. **16 verified insertions: SC 8, SFP 8**; **17 attempts total**
(SC 8 verified +1 setup failure; SFP 8 verified), with four planned cases
unattempted. All sixteen completed insertion attempts passed independent
seating and complete 2000 Hz force checks; maximum raw peak 3.4331 N.
The 21-case screen is incomplete and no research worker is running.
Both connector types have independently verified insertions, but combined
robustness, matched ablations and held-out evaluation remain incomplete.
The setup failure remains in the ledger. Actual stopped-state snapshot:
`.artifacts/finish41/qualification41v2/stopped_status.json`.
Earlier running/12-completed status statements below are historical.

November 6 checkpoint: the frozen V38 combined screen `qualification41v2`
has completed **12/12 independently verified insertions: SC 6/6, SFP 6/6**.
Nine of the prescribed 21 development cases remain, with the next SC case
running. Every completed attempt has complete 2000 Hz raw-force coverage;
the maximum raw peak so far is 3.4331 N. These are partial development counts,
not a finished study or held-out reliability estimate. No policy sources or
thresholds changed during the screen. Snapshot:
`.artifacts/finish41/qualification41v2/status_12_completed.json`.
Earlier completed results and active-status statements remain historical.

Current checkpoint, November 5 late evening (local): V37's completed SFP
screen is **7/10**, with three exposure-time TF failures before insertion
motion. The deployed V38 acquisition-time transform repair passed **4/4**
selected SFP cases: all three failures plus one prior-pass regression.
Complete 2000 Hz raw-force peaks were 2.9993?3.1833 N; median action time
58.9248 simulated seconds and median recorded wall time 565.4294 s.
The descriptive Wilson interval for 4/4 is 51.0?100%; these selected reused
development cases are not a population-reliability estimate. Do not pool
successes from different revisions. Earlier broad SC remains V25 10/11.

The frozen same-version 11-SC/10-SFP screen is executing serially:
`.artifacts/finish41/qualification41v2/progress.json`. Production sources and
adapted original checkpoint hashes match the V38 pilot freeze. The first
launch (`qualification41v1`) was rejected before simulation because its SC
manifest purpose lacked the required `development_only` prefix; that empty
attempt and log are preserved. V2 corrects this metadata, uses new identities
and preserves all physical criteria. No final seeds have run. Matched
current-method ablations, SC contact-negative checks and held-out evaluation
remain pending. Do not call this paper-ready robustness.

The combined SC/SFP manuscript is now `paper/MANUSCRIPT_DRAFT.md`, with a
preserved historical draft and a dated combined copy. It contains actual
measurements and explicit pending results; four-page layout is unverified.
Auditable V38 trial table: `experiments/evidence/exposure40/trials.csv`;
summary: `experiments/evidence/exposure40/summary.json`. Reproduce with
`python experiments/summarize_exposure40.py`. Earlier running/pending
statements below are historical snapshots superseded by this checkpoint.

November 5 evening progress: V37 `lip39v1` repair pilots finished **2/2**
independently verified (peaks 3.211243 /3.494833 N, complete 2000 Hz coverage).
The broader unchanged `qualification39v1` screen currently has **3/6**
verified, with four of ten cases still pending. Completed cases 45000, 45004,
45005 passed; 45001, 45002, 45003 failed before motion with past-TF
extrapolation while resolving the exposure-time TCP after expensive fitting.
Requested/earliest available times were 13.2/13.31, 12.8/14.2 and
13.0/13.002 s. All six have complete force coverage. Keep every failure:
no partial-denominator or motion-only reliability claim is supported, and
the ten-of-ten development gate is already unmet.

The running screen stays frozen. An undeployed fix retains copied exposure-time
TCP transforms before inference/CAD fitting, while preserving the original
camera stamps, age checks and final wrist-stationarity gate. Missing exposure
data still rejects; no latest-pose substitute is permitted. Five contract checks
pass, including computation beyond TF retention and rejection of wrist movement.
Candidate: `.artifacts/repair40/exposure_candidate/`.
A finite boundary worker waits for all ten frozen outcomes, preserves their
snapshot, checks source hashes, then installs/tests the repair and runs the three
failed development cases plus known-pass 45000 at the same 10/2 mm/s profile.
It stops on execution/check/storage errors and opens no final seeds.
Status: `.artifacts/repair40/resume_progress.json`; measured current snapshot:
`.artifacts/repair40/qualification39_status.json`.
SC's earlier broad result is still V25 10/11; final combined SC/SFP qualification
and matched ablations are pending. Earlier active counts below are historical.

November 5 resumed checkpoint: the previously pending V36 `causal38v1`
batch actually completed **1/2** before the user pause. Case 45000 passed
independent held-seat verification at 45.831325-45.841541 mm, complete 2000 Hz
peak 3.915038 N; case 45001 failed the second entrance-lip search before
insertion (`SFP mouth stop unobservable`), peak 2.931267 N. Preserve both.
Both runs had zero future-sample bypasses, so their lack of clock rejection
does not by itself physically exercise the ahead-of-clock repair.

Offline replay found a lost search event: the first stationary blocked window
had a 52.5 ms gap (limit 50 ms) while the command still advanced 207.6 um.
Sampling subsequently recovered, but the already saturated 0.3 mm command lead
then had no recent advance. The old detector never triggered in 27.986 s.
V37 separates executed bounded search intent from the current stationary
measurement window; it becomes ready at 3.7235 s on the saved policy
observations. This is replay readiness only, not a simulated repaired seat.
The 50 ms gap, 0.77 s minimum span, 20 samples, 25 um motion, 150 um lead,
and 100 um search movement limits remain. One fixed reversible force test
with the same force/SEM/stationarity gates is still mandatory; no retries,
truth inputs or seating tolerance changes were introduced.

V37 `lip39v1` is now running 45001 then 45000 using the same adapted original
weights and 10/2 mm/s travel. All 205 selected checks pass, including current
controller noisy-reaction and no-reaction controls. Source freeze and replay:
`.artifacts/repair39/lip39v1_source_freeze.json` and
`.artifacts/repair39/lip_search_replay.json`. A ten-case SFP development screen
is gated on both pilots passing:
`.artifacts/repair39/qualification39v1/progress.json`.
SC's prior broad result remains V25 10/11; fresh same-revision qualification,
contact controls and matched ablations remain. No final cases opened.
Docker/WSL have been restarted at the user's request.
Earlier running/pending statements below are historical.

Latest completed speed comparison V35 `speed38v1`: **1/2 verified SFP
development attempts**. Case 45000 failed before travel because the newest
independent wrist sample led the local ROS clock by 15.5 ms. Keep this failure;
it is not a test of faster travel. Case 45001 completed and independently
verified a held clearance seat with 45.828973-45.854275 mm depth, maximum
lateral error 0.023186 mm, maximum angle 0.003647 deg and maximum terminal
penetration 0.000102 mm. Complete 2000 Hz peak was 3.251675 N; zero samples
exceeded 8 N or the prospective 20 N ceiling.

| Same development seed 45001 | V34 fast37 | V35 fast38 |
|---|---|---|
| Independent median interior mesh speed (mm/s) | 5.006828 | 10.039321 |
| Full travel including estimated dwell (sim s) | 11.6505 | 6.8570 |
| Entire action (sim s) | 62.4860 | 58.3065 |
| Recorded wall including configure (s) | 572.677558 | 519.188105 |

This selected comparison gives 41.14% shorter travel, 6.69% shorter action sim
time and 9.34% shorter recorded wall time. Separate images/host scheduling
preclude a held-out throughput claim. No 3 N travel pause occurred. A low
force peak was not the speed-setting mechanism. Headless footage:
`experiments/evidence/original_repair/video/sfp_speed38v1_verified/attempt.mp4`;
its actual camera streams, timestamps and evaluator-only overlays are hashed.
Gripper occlusion limits visual inspection of the final plug/port interface;
independent source geometry and force/contact records remain necessary.

V36 `causal38v1` now repeats 45000/45001 at the same 10/2 mm/s speed, using
a bounded wrist-history cache that selects the newest sample at or before
the local ROS clock. The 100 ms stale limit remains; future samples are
withheld, not relabelled or accepted. The graph has one clock publisher;
precise callback-delay attribution was not recorded. All 164 selected host
checks and 16 real ROS message checks pass (overlapping tests; do not sum as
unique checks). Physical causal-repair outcomes are pending. All prior bytes
and the original failure remain archived. Freeze:
`.artifacts/speed38/causal_source_freeze.json`. Full measured comparison and
failure record: `.artifacts/speed38/speed_attribution.json`. No final trials
opened; both-connector paper robustness remains unqualified.
Earlier active statuses below are historical.

Speed verification, November 3 local /November 4 UTC: V34's fixed SFP screen
was stopped deliberately at a completed-case boundary after **3/3 verified**
cases (45000, 45001, 45002), to follow the user's speed-optimization request.
Seven remaining cases were not opened; this is not a completed ten-case
qualification. The active third trial finished normally before the queue
orchestrator was terminated and before any frozen source was changed.
Its complete batch is preserved with the preceding two in
`.artifacts/repair37/qualification37v1/`.

Across these three and the two selected V34 repairs, the travel controller
recorded **zero iterations at or above its 3 N pause condition**. Travel
control-observed peaks were 1.094-1.544 N; whole-attempt complete 2000 Hz
peaks were 3.109-3.443 N. Those whole-attempt peaks are not a force target.
Full travel took 11.263-11.844 sim s. The two baseline matched seeds used
63.4345 /62.486 action sim s and 568.764654 /572.677558 recorded wall s.
Consequently increasing travel speed alone cannot halve the entire cycle;
probing, settling, perception and simulation/compute time remain.

V35 `speed38v1` now compares the same development seeds 45000 and 45001
using `fast38`: 10/2 mm/s travel/terminal versus 5/1 previously. Fixed
contact-measurement windows, position leads, 3 N travel pause, 6 N abort,
20 N outer ceiling, and independent seat criteria remain unchanged. Axial
damping/static-drag compensation authority is bounded at 1.5 N, combined at
1.75 N, sufficient for the faster path; this is not a 20 N force command.
Free waypoints and contact trajectories are unchanged. All 150 selected
checks pass, including five ideal load cases; physical speed outcomes are
pending and must not be inferred from those ideal checks. Source and original
model lineage are rechecked. Freeze: `.artifacts/speed38/source_freeze.json`;
measurements: `.artifacts/speed38/speed_attribution.json`. No final seeds opened.
Earlier active statuses below are historical.

Latest complete V34 repair queue: **2/2 selected SFP development trials
independently verified**; both failed their prior unchanged V33 deadlines.

| Case | Stable depth (mm) | Max lateral error (mm) | Raw peak (N) | Action sim (s) | Recorded wall including configure (s) | Waypoint sim (s), prior -> current |
|---|---|---|---|---|---|---|
| 45006 | 45.803222-45.840971 | 0.005772 | 3.191373 | 60.9680 | 660.940853 | 22.2640 -> 13.4390 |
| 45005 | 45.803350-45.834752 | 0.007091 | 3.401211 | 62.7645 | 707.300836 | 22.5035 -> 14.1500 |

Both have complete 2000 Hz coverage and zero samples over 8 N or the
prospectively declared 20 N ceiling. Maximum angular errors were 0.037647 and
0.020210 deg. Configuration took 16.465224 /14.898465 wall s; measured
full-travel segments were 11.263 /11.276 sim s. Positioning was descriptively
39.6% /37.1% faster on these known cases, with different image captures.
Do not interpret this as a matched final throughput estimate. Earlier V33
45006 crossed host sleep; V34 45006 policy finished before a separate sleep
during verification/postprocessing. Preserve both interruptions and distinguish
policy-recorded time from calendar throughput.

V34's abort-compensation clearing has command-path tests but was **not exercised
by these successful insertions**; physical forced-abort validation remains pending.
No threshold or force ceiling was relaxed. Both pilots passed the prerequisite
gate and the serial ten-case SFP development screen **has started**, covering
five NICs and both slots with fixed source/checkpoint hashes, no final seeds or
case replacement, and a 5 GiB free-space stop. Current status:
`.artifacts/repair37/qualification37v1/progress.json`. The SC broad-screen
record is still 10/11 on V25; neither connector has final qualification.
Full V34 measurements: `.artifacts/repair37/waypoint37v1/analysis.json`.
Camera evidence:
`experiments/evidence/original_repair/video/sfp_waypoint37v1_verified/attempt.mp4`.
The historical pending statements below refer to earlier checkpoints.

Latest complete V33 fixed regression queue: **1/3** verified SFP
insertions. `dev_45003` passed (raw peak 3.250855 N); `dev_45006` and
`dev_45005` passed contact registration but exceeded the unchanged 720 s action
budget during travel (peaks 3.128375 and 2.906452 N). All three have complete
2000 Hz coverage. Keep both failures. Their source/checkpoint maps match the
earlier `pulse36v1` selected success exactly; they remain development cases,
not final reliability. Analysis/CSV: `.artifacts/repair36/pulse36regv1/`.
`dev_45006` contains a confirmed host sleep interval; its recorded process wall
duration cannot stand for uninterrupted calendar time. The interruption is
preserved separately and is not grounds to erase the failed insertion.

Offline truth/command diagnosis found post-deadline forward travel of 0.489695
mm over 0.3125 sim s (45006), and 2.913399 mm over 0.3805 sim s (45005).
The prior send hook could inject its last 0.124375 /0.605089 N axial motion
feedforward into the generic 90 N/m hold. The source-audited mechanism is
consistent with the observed drift; a counterfactual physical abort test has
not yet been run. Diagnostic artifact:
`.artifacts/repair37/timeout_hold_diagnosis.json`.

V34 `waypoint37v1` now tests bounded smooth positioning plus cleared motion
compensation before deadline/failure holds, on 45006 then 45005. Default free
waypoints use a quintic path bounded at 5 mm/s and 20 mm/s^2, native base-frame
damping feedforward, and the same 0.15 N axial integral /0.5 mm position-lead
bounds. Explicit 1 mm/s entry moves retain their earlier controller. The
40 um arrival tolerance, 0.3 s dwell, 12 sim s waypoint budget, force guards,
720 s action budget and independent seat criteria remain. 142 checks pass;
physical outcome pending. Source freeze:
`.artifacts/repair37/waypoint37v1_source_freeze.json`. No model was retrained.

The checkpoint/article audit is complete: original SC plug weights unchanged;
SC mouth, SFP plug and NIC weights are fine-tuned descendants of the tracked
originals, with matching training and archived runtime hashes. See
`paper/ORIGINAL_PIPELINE_TRACEABILITY.md`. An undeployed support-priority
search optimization returned exactly identical selected pose dictionaries on
three saved RGB captures, with 19 selector tests passing and 0.36-0.70 s
selection savings in quiet-container replay. It is outside V34, which isolates
the larger positioning repair. No final trial has been opened.

Latest completed V33 `pulse36v1` outcome: **1/1**
verified SFP insertion on selected development case `dev_45002`.
Policy reason: `estimated_datum_depth_dwell_completed`; independent reason: `stable_geometric_clearance_seat_collision_control_validated`.
Complete raw-force audit: True;
2000 Hz peak 3.284503 N under
the prospectively labelled 20 N ceiling. Wall 681.779624 s includes
configuration 12.922353 s; action sim 75.292500 s.
Recorded insertion segment: 11.853500000000011 sim s.
Actual lip-excitation measurements, independent terminal ranges and all audit
details: `.artifacts/repair36/pulse36v1_summary.json`; frozen full batch:
`.artifacts/repair36/pulse36v1_completed.json`. Do not pool different revisions.
Prior running statuses below are historical. Final cases remain unopened and
combined SC/SFP robustness is not yet qualified.

November 3 latest status: V32 `dev_45002_fast36v1` completed **0/1**,
rejecting the first lip as unobservable before the faster travel segment ran.
Duration 416.564511 wall /44.248 sim s; complete 2000 Hz raw peak 2.766287 N,
zero samples over 8 N or the prospectively selected 20 N ceiling. This is not
a physical test of the new travel speed. Archive:
`.artifacts/repair36/fast36v1_completed.json`. Preserve the earlier separate
V31 verified seat and its 8 N protocol.

V33 `pulse36v1` now tests a fixed reversible 0/1/0 N entrance-lip excitation.
Unchanged kinematic blocking checks only trigger the measurement; they cannot
accept contact. Each of the three phases has a fixed 0.2 s settling period
and about 0.4 s collection, using unique actual wrist timestamps. Acceptance
requires stationary tip position, resolved force rise AND fall (at least
25 mN and 3-SEM), and recovery to the baseline, with a 0.2 N source-noise floor
for uncertainty. No repeated searches for a favourable window are permitted.
Opposed-stop agreement, capture limits, independent depth/alignment/penetration
and dwell criteria remain. The 20 N profile retains inner 6 N and travel-pause
3 N guards; the largest additional axial force is 1 N, not a 20 N push.
The legacy 8 N SFP branch keeps its previous mouth method. SC is unchanged.

118 selected checks pass: the new controller's ideal fixtures include 32/32
noisy physical-reaction positives, 0/32 no-reaction clipped-motion acceptances,
and rejection of free air. Sensor checks also reject stale/duplicate samples,
motion and monotonic force drift. These are synthetic fixture controls, not
Gazebo reliability. V33 physical outcome remains pending. Sources are frozen
at `.artifacts/repair36/pulse36v1_source_freeze.json`; do not edit while active.
The previous undeployed status for this candidate is historical.

Publication scope clarification from the user, November 3: **this special
collection allows simulation-only work**. Proceed on that clarification;
do not claim physical hardware evidence. The general author page's real-world
rule was checked, but must not be treated as a reason to stop this explicitly
authorized simulation-only collection study. The manuscript has four main
pages, with optional supplementary-description and reference pages under the
published 4+1+1 format. All claimed outcomes remain conditional on prescribed
rigid grasp and upstream angular alignment. No final held-out trial is open.

Latest completed result, November 3: V31 `tracking36v1` SFP **1/1** on
the selected previously failing development case `dev_45002`. Independent
terminal depth **45.845407-45.867269 mm**, lateral error **0.002890-0.018865 mm**,
angle **0.042741-0.043818 deg**; complete 2000 Hz raw peak **3.005526 N**, zero
samples over the original 8 N ceiling. Terminal dwell and source-clearance
seat checks passed against the qualified matching collision positive control.
This is a geometrically stable seat while held, not latch/electrical/release
retention validation. Duration **686.127787 wall /79.466 sim s** including
configuration. Terminal static-drag integral was **0.221127 N**. This supports
the diagnosed tracking repair on one known case; do not pool with V25-V30 or
describe it as final robustness. Full frozen batch:
`.artifacts/repair36/tracking36v1_completed.json`.

Actual headless camera footage was encoded, decoded without errors, and
inspected at middle/terminal frames:
`experiments/evidence/original_repair/video/sfp_tracking36v1_verified/attempt.mp4`.
Frames are sparse captures held causally between captures, without fabricated
interpolation. The gripper partly occludes the close view; independent recorded
geometry and force coverage complement the visual evidence. Truth overlays
are diagnostic-only and never policy inputs.

V32 `fast36v1` is a separate development test with the user-authorized 20 N
ceiling. It schedules 5 mm/s travel and 1 mm/s terminal motion, adds explicit
100 Ns/m damping feedforward plus bounded 0.5 N static-drag correction, and
retains 0.8/0.3 mm pose leads. Combined extra force is bounded at 1.2 N, with
the existing inner 6 N guard and 3 N insertion pause still active. This does
not command a 20 N push. The evaluator owns the prospective ceiling and checks
complete full-rate coverage; old archives default to 8 N. Fifty selected tests
passed after supplying both local package paths; the first invocation had
three import-path errors, resolved without changing policy code. Physical
speed benefit remains unmeasured until the new trial completes. Contact gates,
geometric tolerances and dwell requirements are unchanged.

An undeployed fixed low/high 1 N excitation candidate has synthetic controls:
0/32 no-reaction noise acceptances and 32/32 one-newton-reaction acceptances
for fixed seeds 7000-7031, plus duplicate/stale/moving/no-block rejections.
It retains the 25 mN minimum rise and adds a 3-SEM stationary contrast margin
with a source-noise floor. These are ideal sensor results, NOT Gazebo trials
or a calibrated probability. It remains outside the live frozen policy.
The old lip-gate noise limitation below therefore remains unresolved in V32.
No held-out final trial is opened; combined SC/SFP paper readiness stays false.

November 3, 2026 current checkpoint (supersedes historical running statuses below):
V30 `sampling36v1` completed SFP 0/1: both lip and side-wall measurements
passed, but the 720 wall-second action deadline stopped insertion. Independent
terminal depth was 43.397542-44.077837 mm versus the 45.8 mm target; complete
2000 Hz force peak was 3.209270 N. Duration: 736.377079 wall /80.558 sim s
including configuration. Archive: `.artifacts/repair36/sampling36v1_completed.json`.
Own-observation traces show terminal drag could stall the old controller near
44 mm before its command-at-goal integral activation; terminal mode could also
chatter. V31 `tracking36v1` is testing bounded axial feedforward throughout
travel and latched terminal braking, with unchanged pose leads and 8 N audit.
Its outcome remains pending; 55 selected regressions passed before launch.

An ideal sensor-negative control exposed a separate lip-detector limitation:
with the public sensor's 0.2 N Gaussian noise and physical reaction disabled,
31/32 fixed seeds passed the old 25 mN reaction-rise gate. This is NOT a
Gazebo insertion false-positive rate; it shows that the reaction-rise check
alone does not certify contact under this noise model. Independent geometric
seat verifications remain separate. Preserve this negative result and require
controlled excitation / stationary uncertainty checks before claiming reliable
force-based lip identification. The simple unload/reload candidate remains
undeployed. Artifact: `.artifacts/repair36/lip_noise_negative_control.json`.

The user authorizes up to 20 N for future trials. Adopt a new explicitly
labelled prospective force protocol; do not change the running 8 N freeze or
relabel prior results. The ceiling is a tared resultant wrist force limit,
not a nominal push target. Preserve full 2000 Hz coverage and unchanged depth,
lateral, angular, penetration and dwell checks. Stronger useful bounded control
and faster measurement/travel must be tested, rather than merely raising aborts.
No held-out final trial has run. Combined SC/SFP robustness is not qualified.

V29 `clearance36v1` finished unsuccessfully at a noisy side-wall measurement;
its source-frozen batch is `.artifacts/repair36/clearance36v1_completed.json`.
V30 `dev_45002_sampling36v1` is active, using actual sensor stamps and a
0.4 s stationary force window plus the bounded faster axial waypoint gain.
Its source map is `.artifacts/repair36/sampling36v1_source_freeze.json`.
46 selected tests pass. Preserve the active freeze. The separate lip retry
candidate based on V30 is `.artifacts/repair36/lip_retry_v31`, with five
passing ideal tests and no physical outcome. No final cases are opened.

November 3: V27 finished SC 1/1, SFP 1/3. V28 finished SFP 0/2; its second
failure was recovered after the host worker exited. Source hashes and complete
raw sensor coverage were checked; the failed verdict is preserved with an
explicit recovery note. Analyze the observation-only missed-contact windows
with `python experiments/diagnose_lip36.py`. Both completed studies have
`analysis.json` /`trials.csv` under their own repair35 directories. Final seeds
remain closed.

V29 `dev_45002_clearance36v1` is executing through the existing seeded runner;
its source map is `.artifacts/repair36/clearance36v1_source_freeze.json`.
Do not edit frozen production sources or run a second simulator. The all-slot
public-CAD path check is `python experiments/audit_sfp_cross_lip_clearance36.py`.
The prospective lip retry is prepared under
`.artifacts/repair36/lip_retry_candidate`; its five ideal tests run via
`experiments/test_lip_retry36.py`. This candidate has not been installed.

New camera evidence is saved in
`experiments/evidence/original_repair/video/sfp_waypoint35v1_verified/attempt.mp4`
and `video/sc_confirmation35v1_verified/attempt.mp4` under the same evidence
root. Both decoded without errors and sampled frames were viewed. Gripper
occlusion limits direct visual depth measurement in these views; numerical
insertion verdicts also require the separate geometry/contact/raw-force logs.
The videos retain sparse source-camera capture times and repeat frames
causally, with no interpolated motion. Their truth overlays are diagnostic.

V27 SFP `dev_45001` now passed independently (45.739129-45.755050 mm depth,
complete 2000 Hz peak 3.320212 N); `dev_45003` is running. The finite
`python experiments/run_axial_waypoint35_queue.py` follow-up is already active,
waiting for the four-case V27 completion marker. Do not start another copy.
It checks all prior hashes and a stopped evaluator before installing the
reviewed V28 free-waypoint axial candidate, running regression unit tests,
then testing only `dev_45002`/`dev_45003` under new `axial35v1` identities.
No held-out cases, threshold changes, faster probe path or longer force window
are included. Status: `.artifacts/repair35/axial35v1/progress.json`.

Subsequent V26 waypoint repair passed the known SFP `dev_45000` regression,
**1/1**, separately from the frozen V25 development screen. V27 four-case
regressions are running through `python experiments/run_repair35_queue.py`;
progress `.artifacts/repair35/regression35v1/progress.json`. Do not launch a
second simulator worker or edit its frozen sources while active. This queue
creates its output directory once and preserves completed failures.

V27 SFP `dev_45002` completed unsuccessfully before the first lip measurement,
with complete 2000 Hz peak 1.019469 N. Run
`python experiments/diagnose_waypoint35.py` for the own-command/hold diagnostic:
the final approach and failure-hold targets differ by 60.387 um axially and
7.571 um laterally in the visual port frame. This is a timestamp-separated
proxy (25 ms), not an exact synchronous tip error. The first V26 lateral
repair is insufficient for this case. Frozen V27 trials continue unchanged.

`python experiments/check_waypoint_axial35.py` exercises a prospective bounded
axial integral in ten ideal damped-load fixtures. All ten candidates passed
the unchanged 40 um/.3 s/12 s arrival checks; this is not a Gazebo result and
the candidate is not deployed. `python experiments/audit_sfp_cross_lip_clearance35.py`
finds zero plug/NIC box intersections in 81 canonical lateral traverses over
the sampled clearance/offset/angle combinations. That geometry-only result
does not qualify the proposed shorter physical inter-probe path. Current
physics, force guards, deadlines and success criteria remain unchanged.

The completed V25 plot can be regenerated with
`python experiments/plot_validation34.py`. The exported PNG/SVG/PDF and
source-data hashes are under
`experiments/evidence/original_repair/validation34v1`. Recorded-force window
diagnosis: `python experiments/diagnose_force_windows35.py`; it uses distinct
sensor stamps, visual port axes and calibrated own-robot orientation, with no
simulator object/target/grasp transform. Longer windows are unqualified runtime
hypotheses, not a relabelling of historical noisy-contact failures.

Evening November 2 update: `validation34v1` finished all 21 cases, SC **10/11**
and SFP **4/10** complete verified trials. Its original source/checkpoint maps,
verdicts and failures remain unchanged. Final qualification failed; final seeds
remain unopened. Recompute the completed-case table with
`python experiments/analyze_contact34_validation.py`. The read-only watcher
`experiments/watch_contact34_analysis.py` exits when the queue finishes; it
does not mutate the policy or simulator.

V26 `waypoint35v1` now runs a separate `dev_45000` development regression with
free-waypoint tracking enabled and actual contact-probe freezes retained.
The geometry/force/deadline/uncertainty thresholds are unchanged. Its outcome
is now verified above. Do not restart the completed V25 queue against changed sources.
The new robustness and four-page paper plan is
[`paper/ROBUSTNESS_AND_PAPER_PLAN.md`](paper/ROBUSTNESS_AND_PAPER_PLAN.md).

2026-11-02 19:13 UTC: V25 SFP `datum34v5` is independently verified **1/1**
nominal case. It is a source clearance seat, without latch/release claims.
The source-frozen 21-case development queue is now active (11 SC, ten SFP):

```powershell
python experiments/run_contact34_validation.py
python experiments/analyze_contact34_validation.py
```

The queue creates a new `.artifacts/repair34/validation34v1` directory and
requires the archived `datum34v5` qualification. Do not launch a second copy
while it runs or change frozen sources/checkpoints. Progress and completed
per-case batches are saved in that directory. The analyzer reports completed
cases only, connector-specific exploratory intervals, all-attempt/successful
times, raw force coverage/peaks and failure causes. Selected SC regression
intervals are not population reliability guarantees. No final seeds are opened.
Exact nominal invocation:

```powershell
python experiments/run_seeded_evaluation.py --case-manifest experiments/configs/sfp_visual33_development.json --case-ids dev_44000 --methods original_perception_sfp_contact_datum_handoff --label datum34v5 --nic-weights .artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt --sfp-plug-weights .artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt --physics-backend source_float64_damped --physics-step 0.0005 --wrist-rate 0 --solver-iterations 200 --positive-control-proof .artifacts/sfp_damped27/contact_proof.json
```

Use a new archive label for a repetition; archived run identities are never
overwritten. Video and measured qualifications are listed in
[`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md).

Latest subsequent development: SC `axial33v1` **2/3**, separate V22
`partial33v1` **1/1**; do not pool revisions. SFP adapted weights and static
errors, valid/invalid lighting diagnostics, videos, and the experimental
`datum34v1/v2` camera/contact controller are recorded in
[`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md). `datum34v1` failed
configuration before movement; V2 reduces the inherited recovery pull lead
to 4 mm (nominal 2 N at 500 N/m), below the unchanged 6 N policy abort.
Final insertion cases remain unopened. The earlier repair33 queue has finished;
new development run status is `.artifacts/active_original_perception29.json`.

2026-11-02: SC development batch `persistent32v1` is complete, **8/11** verified;
the final32 study did not start. SC/SFP repairs now proceed through
`experiments/run_repair33_queue.py`; status `.artifacts/repair33/progress.json`.
See [`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md) for the preserved
failures, SFP visual diagnosis and separate training/development data. No final
insertion case is accessed by this queue.

<!-- current-checkpoint -->
## Current checkpoint - 2026-11-01, shoulder braking and force measurement

The original-perception SC pipeline retains the original confidence-weighted
plug network and uses an adapted mouth network, public-CAD rim/body image fits,
and bounded own-robot lateral PI tracking. It estimates grasp translation from
RGB and receives no live object/target truth. The held connector begins aligned upstream
under a prescribed rigid grasp. General visual angular recovery and repaired
SFP perception remain unqualified.

| Frozen development batch | Verified insertion | Remaining failures |
|---|---:|---|
| v13, one nominal setup repeated | 3/3 | Same seed/setup; not randomized reliability |
| v14, cable mask + earlier mouth weights | 1/5 | Three mouth rejections; one learned angular-prior rejection |
| v14, longer mouth adaptation (`transfer30v1`) | 2/5 | Three learned angular-prior rejections |
| v15, angular-screen diagnostic ablation | 5/5 | No physical failures in these five cases; angular alignment assumed |
| Matched conditional body + PI, `component30v1` | 5/5 | Same five development cases; distinct frozen batch |
| Matched control: no body correction | 0/5 | Three shallow stalls, two force aborts |
| Matched control: no lateral PI | 0/5 | Four lateral misses and one shallow stall |
| V19 high-rate qualification, stopped after two | 0/2 | One missing-sample audit failure; one 24.1285 N shoulder impact |

Each five-case batch uses the fixed development targets 0-4 (seeds 3000, 3002,
3004, 3006, 3008), not fresh held-out cases. Do not pool versions. The conditional
ablation and component comparisons omit the learned angular veto; they are
**not** repaired guarded-policy results and do not certify upstream alignment.
Guarded production results remain 2/5 for the latest mouth checkpoint. No final
100000-series case has been used; these screens do not establish broad reliability.
Older rows use their frozen geometry/contact protocol and 50 Hz sampled force
peaks. They do not establish compliance with the newer every-step force audit.

All five ablation insertions passed the unchanged independent criteria:
15.64 +/-0.25 mm depth, <=0.25 mm lateral and <=0.25 deg full angular error,
continuous one-simulation-second dwell with valid contact evidence, and <=0.1 mm
whole-trial penetration. Across terminal windows they held 15.54785-15.56677 mm
depth, 0.02961-0.13204 mm lateral error and 0.11650-0.23988 deg angular error.
Sampled 50 Hz wrist peaks were 1.4970-3.7933 N, maximum observed penetrations
0.000064-0.002060 mm, and durations 146.883-385.425 wall /14.751-37.323 sim s.
The largest terminal angle is close to the unchanged limit and warrants broader
testing. Per-case records: `varied_screen_v14.json`,
`varied_screen_transfer30v1.json`, `varied_screen_angular_ablation30v1.json`
under `experiments/evidence/original_repair`.

The longer mouth model passes all five target-perception stages. The original
plug angular estimate rejects three physically aligned cases; removing only
that veto permits all five conditional insertions. This supports the translation/
control path under the declared angular condition, not arbitrary 6D perception.

The orange-cable occlusion defect was repaired in the rim fitter; recorded
cross-view disagreement fell from 1.869/2.073 to 0.108/0.091 mm without changing
calibration or acceptance thresholds. Direct rotation fitting, keypoint
calibration, visibility filtering, line/silhouette fits, augmentation ensembles
and longer plug training have not qualified a rotation repair. On six static
development scenes, original median angular error was 1.310 deg; standard plug
adaptation best/final were 2.709/1.086 deg, projected-edge loss best/final
1.960/1.625 deg, and affine-rotation loss best/final 1.986/1.825 deg. None was
substituted into these physical results.

The source-frozen 15-trial component comparison is complete, with identical
source/checkpoint maps and shuffled method order. Full body + PI passed 5/5;
each control passed 0/5. Full terminal depth was 15.54777-15.56700 mm, lateral
error 0.03398-0.09794 mm, and angle 0.11595-0.23997 deg. Independent peaks were
1.4212-2.5440 N; median completion 17.167 sim /163.214 wall s, and maximum
whole-trial penetration 0.0003251 mm. No-body failures included 17.5668/11.1248 N
spikes despite the 8 N pre-command abort check, which is not a physical force
cap. Some offline GPU training ran concurrently; wall times are descriptive,
not controlled throughput comparisons. See `component_screen_component30v1.json`,
`component30_outcomes.svg` and `component30_execution_context.json` in evidence.
These results support the component interaction under the declared alignment
condition; they do not establish broad reliability or repair angular perception.

Ten fresh development cases (`translation30v1`) completed without
concurrent training: 0.5-1.5 mm starting lateral offset, 5-15 mm gap, and paired
unknown rigid-grasp perturbations. The initial physical tip is still aligned
upstream. Grasp jitter is actually passed to the scene generator; its provenance
and rendered images must be checked, not inferred from case names. Harness v16
uses a force-only own-robot load calibration file; geometry and controller
settings are unchanged. Final 100000-series cases remain untouched.

A paired image/truth audit found actual rigid-grasp motion <=0.000104 mm while
original estimates changed 0.229-0.503 mm and 1.275-2.074 deg. Images show the
moving orange cable occluding the centre view. Original point scores remain
0.9956 or higher there; these scores are not calibrated pose uncertainty.
Foreground-only refinement lowered median static angle to 0.641 deg, but failed
all four rigid-consistency pairs. Side-view refinement with fixed pixel-phase
averaging reached 0.306 deg median recorded-image error, yet passed only one
of four consistency pairs. Both remain offline; no threshold was relaxed or
candidate inserted into the physical policy. Current selected regressions: 183 passed; separate
training-loss mechanism checks: 6 passed.

Actual target-4 ablation footage: `video/target4_angular_ablation30/attempt.mp4`.
It visibly labels the ablation and upstream prior, uses actual headless target/
wrist/board captures at 1x simulation playback, and displays truth only as
independent diagnostic overlays. It was fully decoded and terminal images
inspected. Guarded target-0 footage: `video/target0_transfer30/attempt.mp4`.
Camera images, clocks, executable source and checkpoint hashes remain archived.
The old `seeded_pilot27` is stopped. Current status:
`.artifacts/active_original_perception29.json`; entries below are historical.

Completed wider screen: 9/10 verified, of 10 planned; 0 environment/execution failures. This is a development result, not a held-out reliability estimate.

The wider screen finished 9/10: default grasps 5/5, perturbed rigid grasps 4/5. The last case reached the shoulder with good physical alignment but a 34.7699 N independent wrist transient at about 5.669 mm/s axial speed. It failed the force guard. Its terminal lateral/angle ranges remained 0.05461-0.14172 mm /0.04449-0.05183 degrees. The failed run remains in the denominator.

Terminal braking v17 repaired that one development case (1/1): verified depth 15.566032-15.566088 mm, lateral 0.043176-0.049501 mm, angle 0.047340-0.047428 degrees, maximum whole-trial penetration 0.0001373 mm. Recorded 50 Hz peak was 1.15726 N and pre-seat diagnostic speed 1.7613 mm/s, versus 34.7699 N and 5.6686 mm/s in the preserved baseline. Durations were 205.958 wall /24.5225 sim seconds. This is a repaired-case regression, not a randomized reliability result. Both forces are SAMPLED 50 Hz peaks: short impulses can be missed. V18 adds optional per-physics-step raw wrist capture and requires full action coverage plus <=8 N measured peak for high-rate audited trials; historical geometry/force data are preserved. The first high-rate braked attempt (forceaudit31v3) passed geometry/contact and recorded a 2.21306 N raw peak. Its original strict force audit rejected duplicate timestamps: two sensor systems each publish every 0.5 ms step. Distinct timestamp coverage is complete at 2000 Hz; a single-publisher rerun is required before qualification. Four earlier high-rate attempts failed bootstrap before policy commands; all are preserved as environment failures.

The separate 50-case /150-trial conditional final study DID NOT START: its all-ten-pass development gate failed. No 200000-series final manifest was generated or opened; older 100000-series cases also remain untouched. The waiting worker exited, releasing its lock and awake request. See `.artifacts/final_aligned31/development_gate_failed.json`. A repaired, source-frozen development validation is required before reconsidering final execution.

V20 persistent braking: 4/4 completed development cases pass both geometry/contact AND complete 2000 Hz raw force coverage, of eleven planned (two known regressions plus nine previously unexecuted cases). Pending cases are not counted. The first repaired target-0 case held 15.56578-15.56585 mm depth, 0.08027-0.08028 mm lateral and 0.13479-0.13485 degrees angle, with 4.38478 N raw peak, zero samples over 8 N and no missing force step. Pre-seat speed was 0.62534 mm/s. V20 retains the scheduled axial target when the robot runs ahead, slows fully 1 mm before the candidate boundary, and adjusts only the operational stall timeout to the planned slow motion. Physical success/force thresholds are unchanged.

A separate final32 worker is queued for 50 new conditional SC cases x four matched methods (200 trials): full body+PI+brake, no body, no PI, no brake. It generates no final case unless all eleven V20 qualification cases pass exact source/checkpoint, jitter/load provenance and full force audit. It stops the queue on a failed qualification; no failed final case is tuned or replaced. Wilson intervals and Holm-corrected paired tests are generated only after actual completion. Status: `.artifacts/final_aligned32/waiting.json`, then `progress.json`; no final result is assumed.
<!-- /current-checkpoint -->


Work from this repository root (`.../Intrinsic/aic/aic`), branch
`research/gazebo-rap-evaluation`. Research entry points are separate from the
deployed robot policy. See EXPERIMENT_PLAN.md for the four-page paper and final
experimental design, PAPER_RESULTS.md for measured evidence and limitations.

## Current last-inch insertion workflow

The completed numerical study passed seed-3003 terminal geometric seating in
all three contexts, but whole-trial penetration remained 1.8577 / 1.8916 /
1.8923 mm (default/1 ms, 200/1 ms, 200/0.5 ms). The 200/1 ms run additionally
failed the pre-command contact screen despite passing its earlier hold, and
is excluded from policy comparisons. The first 0.5 ms startup failure and
successful retry are separately retained under
`.artifacts/physics_convergence/{num26v2,num26v3dt05}/`.

The matched five-seed batch `matched26v1` was intentionally stopped after three
executed stages because an isolated Bullet float32 collision-query failure
was reproduced. Its executed subset and pre-command screens are in
`.artifacts/matched_registration/matched26v1/partial_03_screened.json`.
The fourth setup was preserved before policy execution; do not count it as a
policy outcome. See `stop_request.json`; the wrapper's nonzero exit records
the researcher stop. Method-order seed was 934; final cases remain untouched.

The active sequence is now `double26v1`: qualify the same-source float64
backend with an SFP miss control, then nominal SC and a shared-prior SFP3003
insertion. Read `.artifacts/active_precision_qualification.json` before starting
another world or editing frozen SOURCES. The backend is selected explicitly
by `--physics-backend source_float64`; default remains `pinned_float32`.
The runner verifies actual server /proc mappings and hashes for the selected
plugin and Bullet dependencies before the hold. Contact proofs must match
the exact backend hash, geometry, timestep and solver configuration.

The alternative backend uses unmodified GzPhysics 8.3.0 sources with
BT_USE_DOUBLE_PRECISION and the installed Bullet float64 libraries; no source
asset, model scale, inertia or numerical seating gate changed. Build sources,
provenance and build log are `.artifacts/gz-physics8.3-double-src/` and
`.artifacts/build_bullet_double.log`; the CMake file is
`experiments/physics_probe/double_backend_CMakeLists.txt`.

After that job finishes, reproduce a fresh fixed study with a new label:

```powershell
python experiments/run_physics_convergence.py --label fresh_numerical_study
# Retry a prescribed subset without repeating completed contexts:
python experiments/run_physics_convergence.py --label fresh_dt05 --contexts iters200_dt05
```

The standalone engine capability probe read back default 50 iterations and
200 after setting 200. Requested iteration counts are also observed in the
live Gazebo component; setup and contact qualification fail on a mismatch.
New SFP clearance checks match the collision proof by timestep **and** solver
configuration. `--positive-control-proof` selects an explicit proof without
overwriting the global default proof. Source snapshots are now taken before
server launch, so failed handoffs retain their exact source files. Controller
activation uses the existing 70 s outer startup budget also for its ROS switch
request; the former 10 s request expired while software-rendered sensors were
initializing. This changes startup only, not insertion gates. Bootstrap errors
are saved explicitly and cause the runner to fail immediately.
The parameter path follows the version-pinned
[SDF component creation](https://github.com/gazebosim/gz-sim/blob/gz-sim9_9.5.0/src/SdfEntityCreator.cc)
and [physics dispatch](https://github.com/gazebosim/gz-sim/blob/gz-sim9_9.5.0/src/systems/physics/Physics.cc),
with runtime getter and workcell-component checks rather than configuration
text alone.

Nominal held SC and SFP geometric seats have passed independent verification.
These are an explicit prescribed-grasp benchmark; the deployed SC visual-grasp
policy still rejects its estimate under unchanged 6 px / 3.5 mm gates. Its
camera TF lookup was corrected to use each exposure timestamp.

The independent recorder observes actual cable pose, wrist FT, fixture state,
all active cable/robot collision bodies and empty-contact physics heartbeats.
The pinned Gazebo Contact implementation publishes only nonempty events:
https://raw.githubusercontent.com/gazebosim/gz-sim/gz-sim9_9.5.0/src/systems/contact/Contact.cc

The complete source active cable is unconstrained at the opposite end. Remote
fixed-support attempts were rejected after measuring about 117 mm drift.
Source Phase 1 uses 2 ms physics; current development uses 1 ms with the same
0.1 mm terminal penetration limit. Wrist cameras run at 5 Hz in software EGL.
Transient distal-cable penetration is reported separately, not erased by a
successful terminal seating check. Pickup, grasp stability and release
retention are outside this benchmark.

Contact interpretation was corrected on 2026-10-25. The exact pinned
GzPhysics 8.3.0 Bullet binary reports signed distance: negative means
penetration; positive cached manifold separation is not touch. Runtime metadata
now includes the binary SHA-256 and convention; unknown versions fail closed.
All old completed trials were reverified without changing numeric gates, and
both nominal seats remain verified. Wrist forces use the independent FT
sensor, not the plugin contact-wrench field. Use link-level contact identity.
Legacy raw streams and verdicts remain available in the export.

```powershell
python experiments/reverify_contact_semantics.py --indexes .artifacts/evaluation_sc.json .artifacts/evaluation_sfp.json .artifacts/evaluation_sfp_positive_control.json .artifacts/evaluation_seeded_pilot_v5.json .artifacts/evaluation_seeded_baseline_v6.json .artifacts/evaluation_seeded_tactile_v1.json .artifacts/evaluation_seeded_ftactile_v3.json .artifacts/evaluation_seeded_ftactile_translation_v3.json --runtime-audit .artifacts/runtime_contact_20261025.json --positive-control-proof .artifacts/sfp_contact_positive_control_signed_distance.json --out .artifacts/contact_reverification_20261025
python experiments/export_insertion_results.py --indexes .artifacts/contact_reverification_20261025/index.json
```

The runner stops only the owned research container, archives each scene and
retains policy/profile source snapshots, hashes, raw streams and camera frames.
The current handoff gate also requires a continuous 0.5 s of live full-scene
contact observation with maximum penetration <=0.1 mm. A stable robot alone
cannot validate a reset with a remote connector inside the board. Failed
handoffs never launch the policy. Legacy pre-command contact screens are
post-hoc and are explicitly weaker than this new gate.
Use a fresh label on every invocation:

```powershell
python experiments/run_seeded_evaluation.py --nominal sc --label nominal_sc --methods initial_pose_guarded --physics-step 0.002 --remote-support 0
python experiments/run_seeded_evaluation.py --nominal sfp_positive_control --label contact_control --methods initial_pose_guarded --physics-step 0.001 --remote-support 0
python experiments/qualify_sfp_contact.py --scene-dir .artifacts/lastinch_runs/sfp_positive_control_supported_contact_control_initial_pose_guarded --run sfp_positive_control_supported_contact_control_initial_pose_guarded
python experiments/run_seeded_evaluation.py --nominal sfp --label nominal_sfp --methods initial_pose_guarded --physics-step 0.001 --remote-support 0
python experiments/run_seeded_evaluation.py --limit-cases 10 --label development --physics-step 0.001 --remote-support 0
```

SFP source geometry has clearance at its nominal 45.8 mm seat. Its verifier
requires a live collision heartbeat, source-box clearance, and a qualified
lateral-contact control with the same collision contract and timestep. It does
not invent sustained contact at a clearance seat. SC requires terminal
ferrule/port contact. Numeric pose, penetration and dwell gates are unchanged.

Development seeds 3000–3009 cover all five target indices, both SFP slots,
5–15 mm gaps and lateral/angular perturbations. Setup/recording failures remain
separate from policy outcomes. Final 100000-series cases have not been used.

```powershell
python experiments/export_insertion_results.py --indexes .artifacts/contact_reverification_20261025/index.json
```

The export includes per-seed outcomes, Wilson 95% intervals for completed
seeded trials, times, forces, failure causes, before/during/after images and
checksums. Nominal demonstrations are not included in randomized success rates.
It also records force P95, measured initial gap/lateral/full-angle errors,
setup camera frames, explicit post-hoc initial-contact exclusions, unfiltered
execution counts and separate debugging revisions. Terminal geometric seating
and the additional recorded whole-trial 0.1 mm penetration screen are distinct
endpoints. Repeated debugging seeds are not pooled as independent trials.

## Completed historical-record reanalysis

Use Python with NumPy, Matplotlib, PyYAML and pytest; exact analysis versions are
in `experiments/requirements-analysis.txt`. Existing Windows Python 3.14.2 was
used; no system packages were installed.

```powershell
python -m experiments.analyze_perception
python -m pytest -q experiments/tests
```

Outputs: `experiments/evidence/perception_reanalysis/` contains all 405 exported
group records, source SHA-256 hashes, recomputed statistics, paired source-trial
bootstrap intervals, a Markdown table, and ECDFs in SVG/PNG/PDF. PDF is generated
locally and ignored by the repository's existing PDF rule. These commands do
not run the detector. Original TACC images and split manifests are still needed
for a genuine inference replication.

## Completed headless camera and actual-model checks

**The original `headless_assets` scene below is historical asset QA.** It is not
the experiment board. Its drop used DART; the impact penetration measurement
does not characterize official Bullet physics. The corrected five-cable scene,
robot and camera evidence are in `experiments/evidence/lastinch/`.

For the corrected native camera checks, obtain ur_description tag 4.3.0 in
`.artifacts/ur_description`, then run from the mounted repository in Ubuntu
24.04. `PYTHONNOUSERSITE=1` avoids this host's incompatible user NumPy/SciPy:

```bash
PYTHONNOUSERSITE=1 python3 -m experiments.prepare_robot
PYTHONNOUSERSITE=1 python3 -m experiments.lastinch_scene --connector sc --target-index 2 --out .artifacts/new_sc
python3 -m experiments.lastinch_capture --scene .artifacts/new_sc
python3 -m experiments.validate_lastinch --scene .artifacts/new_sc
PYTHONNOUSERSITE=1 python3 -m experiments.check_lastinch_matrix --out .artifacts/new_matrix
```

The source asset, camera and reset checks passed for SC and SFP; 31 matrix
geometry checks passed. All six cameras produced actual image messages.
The native joint PID is diagnostic only: its 4.07 mm hold drift failed the
acceptance gate. Do not run paper ablations through that controller.

Docker Desktop's Linux backend was restored by launching its desktop process.
No WSL reinstall was needed: Ubuntu 24.04 WSL2 already exists. Both cached
participant and official evaluator containers run ROS Kilted. The evaluator
is pinned as `ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923`
(Gazebo 9.5.0), with the pinned Phase 1 assets/description/bringup source overlay.
Its compiled controller is from this image; source equivalence to Phase 1 has
not been established, so retain this provenance in any resulting study.

`experiments/create_research_container.ps1` creates or starts the pinned
`research-aic-eval` container, with shared IPC, 2 GiB shared memory and this
repository mounted at `/research`. `start_kilted_handoff.sh` builds the small
research grasp plugin and starts the actual AIC controller, bridge and adapter
with `gz sim -s --headless-rendering`, software EGL and no GUI or task engine.
The script also starts Xvfb; the final server requests EGL explicitly.
It currently prepares one SC2 diagnostic, rather than executing the full matrix.

Startup is paused while the real controllers load. A temporary world fixture
holds the cable until the actual robot reset is applied, then is removed before
the rigid grasp attaches. An observed-clock gate confirms that simulation
advances after unpause; cold camera initialization takes tens of wall seconds.
Bootstrap and hold acceptance records are tied to the exact world SHA-256.
The hold and independent reset checks run inside the evaluator container:

```bash
bash /research/experiments/check_kilted_hold.sh --calibrate-load --calibrate-reset
bash /research/experiments/read_cable_pose.sh
cd /research
python3 -m experiments.summarize_kilted_handoff
python3 /research/experiments/capture_kilted_scene.py
bash /research/experiments/check_port_contact.sh
```

The final eight simulated second hold passed: terminal half-second maximum TCP
position error 0.000304 mm, with about 0.564 N upward load compensation. The
actual tip, measured independently after the contact probe and return, was
0.00415 mm and 0.0113 degrees from its specified handoff. These are one-reset
simulation measurements, not hardware accuracy claims or validation of all cases.

Setup uses evaluator truth to measure the actual attached grasp and establish
the initial condition. The policy receives images, camera calibration, robot
state, wrench and task metadata; it independently estimates its own grasp.
Only the shared load-support force is added by the research command adapter.
The prescribed grasp excludes gripper/ATI collision geometry while preserving
their visuals and inertia. The earlier mask experiment still reported finger
contacts, so masks alone were rejected. Connector, port and board collision
geometry remain intact. This benchmark does not evaluate grasping or collisions
between the gripper housing and the board.

The deliberate 2 mm lateral miss produced 126 contact pairs with the requested
SC port, zero gripper contact pairs, peak tared force 0.775 N and a completed
retract command. The shell and program exited successfully. This is a scripted
collision positive control, not insertion success; seating-depth/contact
thresholds and timestep sensitivity still need validation.

Run the real-policy pilot with the participant image sharing both network and
IPC with the evaluator, and GPU access enabled:

```powershell
docker run --rm --gpus all --network container:research-aic-eval --ipc container:research-aic-eval -e AIC_RESEARCH_RUN_ID=sc_policy_pilot_final --entrypoint bash --mount "type=bind,source=$PWD,target=/research" flowstate:aic_model /research/experiments/run_kilted_sc_pilot.sh
```

The pilot preserves the deployed vision gates and does not use target truth.
The prepared three-method ablation adapters and full case automation remain
unvalidated; do not turn a diagnostic pilot into a reliability result.

To run the source regressions in the participant image, mount the repository
at `/research` and execute `experiments/run_source_tests_container.sh`: 270 passed.
Direct activation of the baked environment avoids `pixi run` rebuilding local
packages and resolving newer dependencies. New research checks: 23 passed.

The existing Ubuntu-24.04 WSL distribution supplies Gazebo 8.15.0, Ogre2/EGL,
Gazebo Transport 13 Python bindings, Gazebo Msgs 10 Python bindings, and Pillow.
Rendering succeeded with Mesa software rendering; no display server or Gazebo
GUI was opened. The recipe follows [Gazebo's EGL headless rendering
documentation](https://gazebosim.org/api/sim/7/headless_rendering.html).

First obtain the exact upstream assets (already done in this workspace):

```powershell
git fetch --depth 1 https://github.com/intrinsic-dev/aic.git phase_1
git worktree add --detach .artifacts/upstream-phase1 7148df9ab62b4b550b101189098f05bc0dba8444
```

If the worktree already exists, verify its HEAD instead of adding it again:

```powershell
git -C .artifacts/upstream-phase1 rev-parse HEAD
```

Run the actual model scene, with a dynamic SC plug drop probe. Choose a fresh
output directory for each run. Substitute your absolute mounted repository path
on another machine:

```powershell
wsl -d Ubuntu-24.04 -- python3 /mnt/c/Users/anshu/College/Intrinsic/aic/aic/experiments/headless_assets.py --drop-probe --out /mnt/c/Users/anshu/College/Intrinsic/aic/aic/.artifacts/headless_assets_drop_new
python -m experiments.analyze_assets --run .artifacts/headless_assets_drop_new --out .artifacts/headless_assets_drop_new_analysis
```

Native Linux can run `python3 -m experiments.headless_assets --drop-probe --out
.artifacts/headless_assets_drop_new`. The renderer requests `LIBGL_ALWAYS_SOFTWARE=1`,
`EGL_PLATFORM=surfaceless`, and uses `gz sim -s -r --headless-rendering`.
It subscribes to actual RGB transport messages before starting the simulator,
saves 12 frames/camera with simulated timestamps and image hashes, records the
Gazebo scene graph, and saves the dynamic plug's 100 Hz pose stream. Merely
starting a server or advertising a topic is not treated as camera success.

The supplied evidence in `experiments/evidence/headless_assets/` includes:

- `overview.png`, `sc_closeup.png`, `sfp_closeup.png`: unmodified received RGB
  camera frames, manually inspected after capture; `overview_initial.png` shows
  the moving plug before impact;
- `asset_manifest.json` and `observed_scene.json`: pinned SDF/mesh hashes and
  the model/visual names and mesh filenames actually observed in Gazebo;
- `integrity.json`: comparisons of source meshes, unit scale, visual counts,
  and unchanged collision definitions, plus dynamic probe measurements;
- `drop_trajectory.csv`, `drop_probe.svg`/`.png`: measured SC drop trajectory
  and analytic collision-to-board clearance;
- `drop_poses.json`, `asset_qa.sdf`: raw pose messages and the exact staged world;
- `run.json`: executed headless command and transport-frame timestamps/hashes.

The source SDF and GLB files are used directly. The QA scene records its explicit
adaptations: absolute local mesh paths, static staging/placement, removal of
model behavior plugins, and one added dynamic plug with a pose observer.
Geometry, scale, materials and collision definitions are preserved. SC0/NIC0
placements come from the pinned task-board xacro; standalone plugs are staged
for inspection. This is **asset and basic dynamics QA**, not the complete UR5e
workcell, wrist-camera calibration, controller, cable mechanics, insertion,
or Kilted evaluator. No insertion-success claim follows from these images.

The drop settled at the board, but reached approximately **1.394 mm sampled
geometric penetration** during impact. This is a material contact-fidelity
finding: final visual plausibility is insufficient to validate tight insertion.
Repeat a low-speed insertion/contact test on the final simulator and compare
timestep/solver settings before publishing contact-force conclusions.

The simpler native server smoke test is also executable:

```powershell
python -m experiments.gazebo_smoke --wsl Ubuntu-24.04
python -m experiments.audit_environment
```

## Prepared Kilted interface baseline (not yet executed)

```powershell
python -m experiments.run_gazebo prepare
```

This prepares three immutable SC scenes from the official pinned sample config
under `.artifacts/gazebo_interface_smoke`. Each has a 20 s task limit and 180 s
wall timeout. The manifest hashes the spec, template and generated configs.
Scene perturbations are seeded; the upstream launch does not expose simulator
RNG control, so `simulator_seed` is explicitly null. This runner only supports
WaveArm interface smoke trials. It does not pretend that last-inch handoff,
recovery ablations, independent contact recording or full-system resets have
already been ported.

The official evaluator uses ROS Kilted and source dependencies pinned by its
`aic.repos`. Native Jazzy/Harmonic is a separate diagnostic backend. Docker's
Linux engine has now been restored; see the executed container setup above.

For a full source rebuild, this optional command is prepared and has **not**
run successfully on this host. Current diagnostics use the pinned binary image
plus source geometry overlay described above:

```powershell
docker build --file .artifacts/upstream-phase1/docker/aic_eval/Dockerfile --tag tar2-aic-eval:phase1-7148df9 .artifacts/upstream-phase1
```

Run inside a Linux environment with the resulting `/ws_aic/install/setup.bash`
sourced, mount this repository at `/research`, and use a new Linux output path:

```bash
source /ws_aic/install/setup.bash
cd /research
python3 -m experiments.run_gazebo run --out .artifacts/gazebo_interface_smoke_linux
```

The runner disables Gazebo/RViz GUIs and leaves `ground_truth:=false`; it checks
Kilted, packages and installed asset hashes before launching. It owns one router,
evaluator and model process group per trial, bounds execution, saves logs and
official result artifacts, and preserves interrupted/failed attempts. A process
exit is logged as `completed_unverified`, never as successful insertion.
Validate EGL configuration and actual wrist-camera frames in this Kilted
environment before proceeding: disabling a GUI alone does not establish that
the composition-based GzServer is configured for headless sensor rendering.

## Evaluator-only seating verification

```bash
python3 -m experiments.verification --profile calibrated_profile.json --trace evaluator_trace.jsonl --out verified.json
```

No production profile is supplied: source-revision geometry must first determine
`seated_depth_m`, depth/lateral/angular tolerances and allowed penetration.
Required profile fields also include `asset_revision`, `geometry_evidence`,
`dwell_s`, `max_sample_gap_s`. Every JSONL sample needs `sim_time_s`, `depth_m`,
`lateral_error_m`, `angular_error_deg`, `max_penetration_m`,
`contact_consistent` and `fixture_active`. The angle includes twist. The contact
and fixture booleans must be measured by the evaluator; missing fields fail
closed. Requested-target identity must be established by the recorder before
these relative measurements are passed to the verifier. It does not infer
electrical connectivity or latch engagement from geometry.

Trace export from the final Kilted simulator, anti-latch research mode and
calibrated thresholds remain implementation work. The synthetic regression
tests exercise the gate and are not robotic experiments.

## Source policy regression status

The eight-file source suite documented in `docs/CURRENT_SYSTEM.md` was executed
with `PYTHONPATH=aic_model;aic_example_policies` and pytest plugin autoload
disabled. Result: 232 passed; 15 failed importing missing generated ROS messages.
The frozen SFP suite additionally cannot collect one test for that dependency.
Run the complete suite in the Kilted workspace after setup; no fake ROS stubs
were introduced to make the test report green.

## 2026-10-27: SC reset calibration defect and corrective trial

The SC outboard run `sc_supported_sc_outboard26v1_initial_pose_guarded`
passed its TCP hold (0.08446 mm maximum terminal error) but was already
6.79944 mm laterally displaced at the first policy command, with +1.12310 mm
axial depth instead of the prescribed -10 mm approach gap. The setup-only
startup grasp measurement differed from the prescribed grasp translation by
13.18459 mm and had been used to change the reset target. The prescribed grasp
predicts the first recorded tip within 0.19559 mm using a nearest timestamp
(4 ms skew); this is diagnostic evidence, not exact-time certification.
The earlier force abort therefore does not isolate insertion-policy failure
from an invalid initial handoff. Original results remain preserved.

The reset now uses the prescribed rigid-grasp design. The early grasp sample
is retained only as a diagnostic. An additional independent actual-tip gate
requires 0.5 s coverage, position error <=0.25 mm and orientation error <=0.25
degrees relative to the requested reset pose, alongside existing TCP and
contact gates. No production policy receives simulator truth. Research tests:
40 passed. Deterministic float64 SC outboard trial `design27v1` has been
launched; no outcome is claimed until its logs and independent verification
are complete. Diagnostic: `.artifacts/sc_design27v1_diagnosis.json`.

### Completed design27v1 corrective run

The corrected prescribed-grasp SC reset completed its 20 s setup observation
but correctly failed validation before any insertion policy ran. Final 0.5 s
actual-tip position error maximum: **0.44738 mm** (limit 0.25 mm); orientation
error maximum: **0.01783 degrees**. TCP error maximum: **0.37648 mm**.
The full-scene contact gate also failed. For context, the 20.0-20.5 s window
reached **0.20848 mm** manifold penetration; the final single contact sample
was only 0.05968 mm, demonstrating why a continuous window is required.
These are setup measurements, not an insertion success/failure trial.

The large reset bias is substantially reduced; remaining cable-induced
motion/contact must be resolved before another insertion attempt. Next:
localize the offending cable/board contacts across the full hold, assess free
chain settling and load compensation, and rerun the same deterministic reset
with unchanged position/contact gates. The later SFP source-box overlap
question and final held-out evaluation remain open. Evidence:
`.artifacts/lastinch_runs/sc_supported_design27v1_initial_pose_guarded/`.

## 2026-10-27: corrected-reset SC dynamics and sharp-box diagnostic

With the prescribed reset, float64, 0.5 ms steps and 200 solver iterations,
`sc_supported_design27v2dt05_initial_pose_guarded` completed insertion commands
and dwell. Terminal depth was 15.52193-15.56632 mm, lateral error <=0.22538 mm,
angle <=0.02652 degrees, and engine whole-trial penetration <=0.05692 mm.
It **failed** the existing contact-consistency gate: no requested SC contact
in the terminal dwell. Source box SAT places initial axial interference near
15.58992 mm for a centered plug; the observed depth was still short of that.
Independent capsule/source-box auditing found 0.22940 mm overlap against the
board at 14.9805 s, so engine-reported penetration alone is insufficient.

The same numerical configuration with existing compensated shared-prior policy
`sc_supported_design27v3prior_force_tactile_no_registration` aborted before
port entry (7.284 simulated seconds; independent peak force 23.45154 N).
Near the abort, contacts included free SFP/board and cable/NIC bodies.
This does not establish a contact-registration benefit or failure, since that
ablation has no lateral probes. All attempts remain development only.

An isolated exact-pose float64 Bullet query reproduced the source-box
mismatch: default box margin 0.60000 mm gave -0.0009483 mm signed distance;
zero box margin gave -0.2294016 mm, agreeing with the independent sharp-SDF-box
calculation. Evidence: `.artifacts/box_margin27/`. A separate backend with
zero margin on SDF box construction is being built; prior binaries remain
unchanged and no result is yet qualified under the new backend.

A canonical horizontal outboard free-chain initialization is now under test
(`horizontal27v1`), avoiding the original elevated loop's fall. This changes
only initial free-link poses around the source SC ball joint. Source geometry,
inertias, constraints, held tip, and all joint anchor positions are preserved;
41 research tests pass, including both initializer variants' invariants.

### Horizontal SC initialization: terminal seat achieved, geometry caveat retained

`sc_supported_horizontal27v1_force_tactile_no_registration` passed the
requested-contact terminal seating verifier. Terminal depth 15.5670016-
15.5670092 mm; maximum lateral error 0.0166756 mm; maximum full angle
0.0032048 degrees; independent tared force peak 0.938779 N. Duration:
36.734 simulated / 380.150 wall seconds. Setup actual-tip error was <=0.020119
mm. Headless overview and final target camera images were inspected.

The whole-trial engine penetration maximum was only 0.0001807 mm, but the
independent sharp source-box audit found **0.177259 mm** cable/NIC overlap
(3739 exact timestamp matches, no missing packets). This remains above the
0.1 mm geometry limit; do not report it as fully physically validated.
The separately built `source_float64_sharp` backend preserves SDF box edges
by setting their Bullet margin to zero. Binary audit and source provenance
are in `experiments/configs/bullet_sharp_*`; plugin SHA256 is
`fea108d05308fe4adb67a7e64ee6b45e607a3d2164d103e201adedb19194843e`.
The same deterministic case is running as `sharp27v1`; this is a new backend
qualification, not a pooled repeatability result. No thresholds were relaxed.

### Sharp geometry qualified on one SC trajectory; cable interference remains

`sc_supported_sharp27v1_force_tactile_no_registration` passed reset checks
(actual-tip maximum 0.01834 mm), reached nominal seating depth, then aborted
during dwell at 28.760 simulated seconds (policy peak 8.05476 N). Contacts near
the abort included the free SFP module against NIC card/mount 4. This is a
failed insertion attempt and is retained. Crucially, independent sharp-box
geometry now agrees with engine contacts: maximum capsule/static-box overlap
0.00002436 mm versus same-scope engine maximum 0.00002458 mm, across 2939
exactly matched timestamps with no missing packets. This supports the box
margin correction within that explicitly audited scope, not whole-scene fidelity.

An unfolded cable reset preserving every source joint anchor was implemented
and tested (42 research tests passed). `straight27v1` failed scene generation
from an editing syntax error, before simulation; this was fixed and compilation
checked. `straight27v2` then executed but failed its 20 s setup hold: actual-tip
maximum error 0.62541 mm, orientation 0.04260 degrees, contacts passed. The chain
crossed staging fixtures and did not settle within the position gate. No policy
ran. A revised, case-independent open-corner initial direction [1,2,0]/sqrt(5)
is now running as `corner27v1`, preserving meshes, lengths, inertias, joint
anchors, fixed connector transforms, and held connector reset. Controller and
numeric acceptance gates remain unchanged. Repeats are conditional on a fully
validated first pass; none has yet run on the final corrected backend/layout.

## 2026-10-27: source ball-joint damping omission reproduced

Pinned gz-physics 8.3.0 assigns JointInfo damping from SDF only for revolute
and prismatic joints. Cable ball joints retain the default zero, even though
all 22 source ball joints declare damping 0.2. Its existing forward-step
AddJointDampingTorque already applies -D*qdot to every joint DOF. A separate
backend now also loads the SDF ball-joint value into that existing update.
This is an explicit backend correction, not an unmodified-upstream claim.

An isolated actual-plugin probe verified all three ball-joint axes: with zero
gravity, inertia 0.001, initial speed 1 rad/s, declared damping 0.2, 200 steps
of 0.5 ms, the sharp-only backend stayed at 1 rad/s. The corrected backend
reached 7.0550791e-10 rad/s on each axis, matching (1-D*dt/I)^200. All six
checks passed. Source, build files, logs and results:
`.artifacts/ball_damping27/`. Damped/sharp float64 plugin SHA256:
`8f7aff37ec99f324c637c73a6c96509e95ccbc54d512b62785d894fa5c8aacaf`.

The open-corner, undamped `corner27v1` reset passed after 16.062 simulated
seconds, but insertion failed stable dwell (policy peak 4.53678 N), with no
force abort. Its index is preserved in `.artifacts/continuation27/`.
The matched `damped27v1` trial is now running with identical reset/layout,
controller, timestep, solver iterations, and numerical acceptance limits.
42 research tests pass; affected Python modules compile. No causal insertion
or repeatability improvement is claimed until the matched run and independent
audits finish. Final held-out seeds remain untouched.

## First corrected SC run passes recorded checks; repeats underway

The `damped27v1` controller completed 36.738 simulated / 334.1957 wall seconds,
with independent tared force peak **1.09728 N**. Terminal depth was
15.56698861-15.56698886 mm; lateral error <=0.00551253 mm; full angle
<=0.00308475 degrees. Initial actual-tip error near the first command was
0.05738 mm (4 ms timestamp skew), within the declared handoff gate.

Its strict signed-distance-only contact verifier originally returned
`contact_inconsistent`. Raw contacts show the requested SC/adapter constraints
carrying positive normal impulses at tiny positive separations (1.10e-11 to
5.84e-10 m): numerically non-penetrating active contacts. Verifier semantics
revision 3 now also recognizes the audited gz-physics 8.3 duplicated/opposite
scalar normal-impulse encoding. Positive-gap cached points with zero impulse,
unrecognized encoding/version, and nonfinite values remain rejected. Depth,
lateral, angular, force, penetration and dwell limits are unchanged. All 44
research tests pass, including active-vs-cached contact regressions. The old
verdict is preserved in the raw run, and reanalysis is stored separately in
`.artifacts/sc_damped27v1_reverified/`.

The independent source audit checked all 20 cable capsules against 187 static
NIC/board boxes: max overlap 0.00003595 mm, versus engine same-scope 0.00003602
mm. There were 3750 exact timestamp matches and one leading unmatched health
packet at 8.9205 s, before the first contact packet (8.9305 s) and policy start
(9.244 s). No unmatched packet occurred during insertion or its recorded tail.
This establishes the declared recorded checks for one prescribed-grasp trial,
not hardware fidelity or general reliability. Decision record:
`.artifacts/continuation27/damped27v1_validated.json`.

Two identical prospective repeats now run serially under frozen source,
backend and controller settings. Progress:
`.artifacts/sharp_sc_repeats27/progress.json`. Each attempt is retained; a failed
setup, seating, source-geometry, initial-pose or motion-coverage check stops the
batch for diagnosis. No repeated-until-success or held-out tuning is permitted.

### First prospective corrected SC repeat passed

`sc_supported_damped27repeat2_force_tactile_no_registration` passed seating,
initial-pose, whole-trial engine and independent source-geometry checks, with
no missing matched geometry packets during motion/tail. Independent force
peak 1.175463 N. This is one prospective repeat after freezing verifier revision
3, in addition to the initial development run reanalysed under the corrected
contact semantics. A second prospective repeat is running as `damped27repeat3`.
Do not treat these identical-target trials as a randomized reliability estimate.

The undamped open-corner comparison was also reanalysed under revision 3 and
still failed stable dwell: lateral error reached 0.256828 mm against 0.25 mm,
with six terminal samples over the gate. Its independent source-box overlap
was 0.0269633 mm, agreeing with engine contacts. The matched diagnostic figure
in `.artifacts/sc_matched_damping27/` shows depth, lateral error and independent
wrist force for both runs. It is a single-case mechanism comparison, not a
statistical ablation or a general controller-performance claim.

### Three nominal SC runs passed; SFP qualification started

Both prospective repeats (`damped27repeat2`, `damped27repeat3`) passed every
recorded check. Together with the initial development run reanalysed under
verifier revision 3, nominal SC is now **3/3 at the same prescribed-grasp target**.
Independent tared force peaks: 1.09728, 1.17546, 1.07827 N. Both prospective
repeats have zero unmatched geometry packets during motion/tail. All initial,
terminal pose, active contact, engine whole-trial penetration, and independent
source-box checks passed. These are repeated nominal demonstrations, not a
randomized success estimate across targets; no general reliability claim or
randomized confidence interval follows from this count.

The same-backend SFP intentional 2 mm lateral miss is now running as
`sfp_positive_control_supported_damped27control_initial_pose_guarded`. Its
qualification uses the same audited active-impulse contact interpretation as
the seating verifier. All 44 research tests pass. No acceptance thresholds or
held-out trials were changed. After qualifying this contact control, run SFP
nominal insertion/repeats, then the frozen seeded development comparison.

Upstream corroborates the ball-joint damping omission and first-axis-to-all-DOF
semantics: https://github.com/gazebosim/gz-physics/pull/797 and
https://github.com/gazebosim/gz-physics/pull/753. This is a compatibility fix,
not an original algorithmic contribution.

### Corrected-backend SFP contact control qualified

`sfp_positive_control_supported_damped27control_initial_pose_guarded`
completed its intentional 2 mm offset test with force abort. It produced
1236 physical requested module/cage pairs out of 1243 cached manifold pairs,
under the audited active-impulse/signed-touch interpretation. Independent
capsule/static-box audit: zero overlap, 2306 exact timestamp matches and zero
missing packets. Contact proof (same binary, collision digest, 0.5 ms timestep,
200 solver iterations): `.artifacts/sfp_damped27/contact_proof.json`.
This is an environment control, not an insertion success or a reliability trial.

A prospective three-run SFP nominal insertion/repeat batch is now running with
shared-prior/no-registration policy and unchanged gates. Each attempt is
retained and the batch stops for diagnosis at the first failed independent
check. Completed repeat records: `.artifacts/sfp_damped27/repeats/progress.json`.

### First nominal corrected-backend SFP insertion passed

`sfp_supported_sfpdamped27v1_force_tactile_no_registration` passed all recorded
checks. Terminal depth 45.801733 mm, maximum lateral error 0.00965081 mm,
full angle 0.00100952 degrees. Independent tared force peak 1.346705 N;
66.900 simulated / 652.925 wall seconds. Maximum whole-trial engine penetration
was 0.000002982 mm; independent all-20-capsule/static-box overlap was zero.
No unmatched geometry packet occurred during motion/tail. Initial tip error
was 0.010579 mm with 4 ms nearest-stamp skew. This is a prescribed-grasp
clearance seat, supported by the matching qualified deliberate-miss control;
it does not establish electrical or latch engagement. Two prospective repeats
remain in progress. Nominal SC is 3/3; nominal SFP is 1/1 completed so far.

### Nominal SC and SFP repeatability completed; seeded pilot executing

All three prospective SFP runs passed all recorded checks. Independent force
peaks: 1.34671, 1.39103, 1.40028 N. Each had zero independent capsule/static-box
overlap and zero unmatched geometry packets during motion/tail. Nominal SC is
3/3 (initial reanalysed development run plus two prospective repeats); nominal
SFP is 3/3 prospective runs. These demonstrate same-target repeatability, not
randomized reliability. The portable export now contains all six nominal runs,
the failed undamped comparison, and the deliberate SFP contact control:
`experiments/evidence/qualified_insertion/` (8 executed records).

The next prospective study is frozen before execution in
`.artifacts/seeded_pilot27/spec.json`: ten existing development cases (five SC,
five SFP) across target locations and pose perturbations, paired shared-prior
force policy with lateral registration enabled/disabled, randomized method
order seed 270027. Source hashes, backend, timestep, solver, and per-connector
cable initialization are frozen. It is running serially, retaining all
attempts and stopping at a boundary for invalid setup/execution. Policy
failures remain data, not automatic reruns. Actual success rates, Wilson CIs,
success completion times, force peaks and causes will be logged in
`.artifacts/seeded_pilot27/progress.json`. No held-out final seed was used.

### First perturbed paired SC case completed; longer pilot remains active

For development case 3000, both shared-prior methods passed every recorded
check. Lateral registration: maximum terminal lateral error 0.045211 mm,
53.100 simulated / 476.870 wall seconds, independent force peak 1.66488 N.
No registration: 0.095525 mm, 36.098 simulated / 306.869 wall seconds,
independent force peak 1.52834 N. The registration trace recorded opposed
physical contact plateaus on both lateral axes. This one pair describes a
precision/time/force tradeoff; it does not establish general superiority.

The frozen 20-attempt pilot is still running serially, now on development
case 3001. Runner PID at launch: 31368. A separate completion worker (launch
PID 64196) waits for its recorded completion/diagnosis status, then exports
actual traces, camera images, source snapshots, confidence intervals and
failure causes to `experiments/evidence/seeded_damped_pilot/` and appends measured
summaries to PAPER_RESULTS.md, EXPERIMENTS.md and RESEARCH_PROGRESS.md. It does
not claim unexecuted outcomes. Current state is always
`.artifacts/seeded_pilot27/progress.json`; completion marker is
`.artifacts/seeded_pilot27/finalization.json`.

Resume by inspecting that state and the live processes. Do not launch a second
world or rerun the same driver/labels while this batch is active. Do not change
its frozen source during a trial; use the boundary stop file
`.artifacts/seeded_pilot27/STOP_REQUEST.json` if a new defect requires diagnosis,
then retain the old cohort and define a new revision for any controller fix.
The six-run nominal export has 835 checksummed files, verified with zero
mismatches. No held-out final trial has run.

## 2026-10-30 matched comparison and fresh translation screen

The completed 15-trial matched comparison is stored in
`experiments/evidence/original_repair/component_screen_component30v1.json`.
Its three methods use the same original plug weights, longer-adapted mouth,
source geometry, native controller/load support and independent verifier. All
three assume upstream connector angular alignment and omit the learned angular
veto as explicit diagnostic ablations. Results: full body + PI 5/5, no body 0/5,
no PI 0/5. Do not pool these with guarded-policy results or treat them as final
reliability. `component30_execution_context.json` records concurrent offline
training windows; wall times are descriptive, not controlled throughput evidence.

Reproduce under a NEW label; the runner preserves existing archived runs:

```powershell
python experiments/run_seeded_evaluation.py --case-ids dev_3000 dev_3002 dev_3004 dev_3006 dev_3008 --methods original_perception_rim_position_body_track_noangular_handoff original_perception_rim_position_body_noangular_handoff original_perception_rim_position_track_noangular_handoff --method-order-seed 20261030 --label component_reproduce --grasp-profile challenge_sc --sc-mouth-weights .artifacts/perception_data30/training/sc_mouth_standard30_v1/run/weights/best.pt --physics-backend source_float64_damped --physics-step 0.0005 --solver-iterations 200 --cable-initialization sc_straight_outboard
python experiments/summarize_component_ablation30.py --label component_reproduce
```

The actual `translation30v1` batch is running the pre-frozen
`experiments/configs/translation_development30.json`: five fresh layouts/targets,
0.5-1.5 mm lateral norms, 5-15 mm gaps, each paired with/without a prescribed
rigid-grasp perturbation. The initial physical connector angle remains aligned;
this is not an arbitrary-angle recovery test. Source grasp translation and xyz
Euler perturbations are sampled within +/-2 mm / +/-0.04 rad. Harness v16 really
passes the jitter seed to the scene generator. Verify the recorded provenance,
changed grasp transform and actual cameras; case names alone are insufficient.
No new plug/refiner model is deployed. Final 100000-series cases are untouched.

```powershell
python experiments/run_seeded_evaluation.py --case-manifest experiments/configs/translation_development30.json --limit-cases 10 --methods original_perception_rim_position_body_track_noangular_handoff --label translation_reproduce --grasp-profile challenge_sc --sc-mouth-weights .artifacts/perception_data30/training/sc_mouth_standard30_v1/run/weights/best.pt --physics-backend source_float64_damped --physics-step 0.0005 --solver-iterations 200 --cable-initialization sc_straight_outboard
```

Do not launch a second simulation runner while the owned evaluator is running.
Current raw status is `.artifacts/seeded_evaluation_progress.json`; `seeded_pilot27`
is an older stopped study. V16 reads only `static_load_calibration.json` for
own-robot force support. The larger hold/scene diagnostic files retain evaluator
poses but are not opened by the original-policy harness. The force-only vector
is identical to the independently retained own-robot hold calibration.

Completed checkpoint snapshots were consolidated into byte-verified NTFS hard
links to immutable `.artifacts/frozen_checkpoint_objects`; original checkpoint
files and all measurements remain intact. Source paths and SHA256 bytes are
unchanged. Subsequent snapshots use the same hash-verified objects to avoid
copying identical binaries for every trial. No live model is hard-linked to
this archival cache. The deduplication report records preserved paths/hashes.

The new foreground/anchor/stereo-phase perception prototypes are OFFLINE.
Their qualification scripts write image-only fits before diagnostic truth.
The final foreground model improved static median angle to 0.641 degrees, but
none of the tested variants passed all four rigid-grasp consistency checks.
They must not be silently substituted into physical runs. For a ROS-dependent
archived-live replay, use `experiments/replay_sc_live_screen30.sh` inside the
built evaluator container; the GPU training container lacks that built ROS
workspace. Model-only audits do not require ROS. Original rejection and strict
physical scoring thresholds remain unchanged.

## Queued source-locked conditional final study

A single owned worker is waiting for ALL ten `translation30v1` cases to pass
independent verification, correct grasp provenance and the force-only calibration
audit. Only then does it generate 50 new 200000-series conditional SC cases and
execute all three matched methods (150 trials). This is the nominally aligned
last-inch contract, not the older full-angle/SFP 100000-series protocol. The
latter stays unused. Protocol and scripts are frozen before final execution;
results cannot be used to tune or replace a failed final case.

Monitor `.artifacts/final_aligned31/waiting.json` before start, then `progress.json`.
The active command is `python experiments/run_frozen_aligned_final31.py
--wait-for-development`; do not start another worker. A failed development gate
stops the queue with an explicit diagnostic. A completed final study automatically
creates measured JSON/CSV, Wilson 95% plots, paired exact tests, and
`paper/FINAL_ALIGNED31_RESULTS.md`. Those outputs do not yet exist at this entry.

The final worker has an exclusive PID lock and a process-scoped system-awake
request (no power-plan change); both are released on normal exit. It stops
before disk drops below 1.5 GiB. For a clean stopped run, `--resume` verifies all
frozen sources, retains every previous failed attempt and skips recorded cases.
An unindexed archive/unarchived evaluator workspace or stale lock requires
inspection and preservation before restart; do not delete evidence to force a
resume. Do not edit any frozen source/checkpoint during final execution, and do
not run offline training/inference audits concurrently with final measurements.

Time definitions: raw `wall_s` includes harness/model configuration, while
`sim_s` is the action window. The final analyzer also reports
`configuration_wall_s` and `action_wall_s = wall_s - configuration_wall_s` for
matching action-clock comparisons. These are simulation/computational timings,
not established real-robot cycle times. A held geometric seat does not validate
post-release retention, mechanical latch engagement or electrical continuity.


## 2026-11-01: repaired shoulder approach and force-sampling audit

The complete wider development screen is 9/10 (default rigid grasps 5/5;
perturbed grasps 4/5), with zero environment failures in that batch. Do not
replace its failed dev_41004 result with a later repair. The failure was a
shoulder impact: independent geometry remained aligned, but the connector
approached first 15.5 mm depth at approximately 5.669 mm/s and the 50 Hz wrist
stream recorded 34.7699 N. The public seat was reached only momentarily before
the force abort. The separate final readiness gate correctly stopped; no final
manifest/case was opened. Earlier queue descriptions are historical, not an
active overnight study.

V17 adds an explicit, selectable terminal braking method. Beginning 2 mm before
the estimated candidate depth, it tapers speed to 0.5 mm/s and elastic lead to
0.3 mm; it caps the command at public CAD seating depth and retains the same
governor through the dwell transition. Perception, calibration and independent
geometric/contact limits are unchanged. The original unselected controller
retains its earlier depth command behavior.

The preserved failed development case passed its V17 regression (1/1): terminal
depth 15.566032-15.566088 mm, lateral 0.043176-0.049501 mm, full angle
0.047340-0.047428 degrees. Maximum observed whole-trial penetration was
0.0001373 mm. The sampled 50 Hz force peak was 1.15726 N, pre-seat diagnostic
speed 1.76128 mm/s, and duration 205.9576 wall /24.5225 simulation seconds.
This single repaired case is not randomized robustness. Video was fully decoded
and an actual contact frame inspected: video/terminal_brake31v1/attempt.mp4.
Both runs use the angular-screen ablation and assume upstream angular alignment;
they do not establish repaired general angular perception.

A necessary measurement caveat: fifty samples per simulated second can miss a
short contact impulse. Low historical sampled peaks cannot certify a continuous
force ceiling. V18 retains that stream and adds a separate raw sensor recording.
High-rate trials additionally require complete timestamp coverage and <=8 N raw
tared peak; missing samples fail the new force audit. This tightens validation;
older raw outcomes and unchanged geometry profiles remain preserved. Contact
observations remain sampled at 100 Hz, so reported maximum penetration is the
maximum observed across the recorded trial, not an every-physics-step bound.

The first four high-rate attempts (forceaudit31v1/v2) failed controller bootstrap,
before policy commands. A stale X lock was repaired with a live-server check,
but the substantive remaining stall was located by debugger inside
Sensor::Update. Installed gz-sensors 9.2.0 truncates a 2000 Hz period to zero
milliseconds and loops forever. The versioned upstream source and installed
binary disassembly agree. SDF rate zero bypasses that scheduler and requests
one update per physics step; actual delivered timestamps must still pass the
audit. Neither the installed sensor library nor the contact solver was patched.
See force_audit31/scheduler_diagnosis.json, stack/disassembly and runtime package
record. Installing the debugger also updated container libc6, explicitly recorded;
no valid policy trial ran during that package operation.

Reference: https://github.com/gazebosim/gz-sensors/blob/gz-sensors9_9.2.0/src/Sensor.cc

A matched braked/unbraked development rerun, forceaudit31v3, is in progress with
0.5 ms physics and unthrottled raw sensor capture. A separate ten-case fresh
DEVELOPMENT manifest terminal_braking31_development.json was frozen before its
execution. No result is filled in in advance. Current regressions: 157 ROS policy/
geometry tests plus 26 independent contact/verification/raw-sensor tests passed.


## 2026-11-01: per-step force failures and persistent-reference repair

The paired V18 raw-force diagnostic (forceaudit31v3) recorded 2.21306 N for
braked and 43.76820 N for unbraked on dev_41004. The unbraked 50 Hz record showed
only 2.28168 N: sparse sampling missed the impact. Both original V18 verdicts
remain failed because two ForceTorque systems produced duplicate timestamps.
Every distinct 0.5 ms tick was present; all samples, duplicates included, are
retained in the raw diagnostic plot. Neither is relabeled as a qualified trial.

V19 removed the redundant world ForceTorque system, preserving the source ROS
robot's system. Its qualification was stopped after two complete policy trials:
known dev_41004 seated but failed strict coverage after one missing action sample;
fresh dev_42000 seated but produced 24.12851 N despite the first braking repair.
The latter recording had complete 2000 Hz coverage and is a real force failure.
The parent runner was stopped only after that policy completed; its recorder
finished draining, the unchanged V19 analyzers/verifier ran manually, and the
complete archive/result was retained. The remaining nine cases were not executed
in V19. See .artifacts/qualification31v1_partial.json. These outcomes are 0/2
under the stronger force-aware verification, not two insertion successes.

The drop investigation found a RELIABLE bridge publisher and a BEST_EFFORT raw
recorder. V20 requests RELIABLE delivery with queue depth 10000. Actual ROS endpoint
QoS was inspected; delivered full-action timestamps still must independently pass
the strict no-missing-step audit. Coverage requirements and the 8 N limit were
not relaxed.

The new physical defect was in the axial controller: max(commanded_depth,
observed_depth) advanced the setpoint to an overshooting robot, removing the
restoring position error needed for braking. V20 keeps a persistent scheduled
reference, permits it to lag the robot, and reaches its 0.5 mm/s terminal speed
1 mm before the candidate boundary. The taper occupies the preceding 2 mm.
The same governor continues through the terminal dwell. A bounded operational
stall timeout accommodates the planned slow motion; the 0.5 mm progress distance,
force checks, independent seating limits and one-second dwell are unchanged.

Source-frozen persistent32v1 is running two known regressions followed by nine
previously unexecuted development cases. Both known regressions passed actual
geometry/contact AND full 2000 Hz force verification:

| Case | Raw peak N | Terminal depth mm | Terminal lateral mm | Terminal full angle deg |
|---|---:|---:|---:|---:|
| dev_42000 | 4.384778 | 15.565782-15.565854 | 0.080271-0.080277 | 0.134794-0.134846 |
| dev_41004 | 4.551700 | 15.566169-15.566224 | 0.064075-0.064090 | 0.049827-0.049896 |

Both have complete 0.5 ms force spacing and zero samples over 8 N. Target-0
pre-seat speed fell to 0.62534 mm/s; its earlier V19 impact approached at roughly
2-3 mm/s. Actual V20 video was fully decoded; contact and terminal frames were
visually inspected. These two repaired cases are not a fresh reliability estimate.
No new plug weights, object/target truth inputs or physical success thresholds
were introduced. General angular recovery and SFP perception remain unqualified.

A source-locked final32 worker is queued for 50 new conditional SC cases x four
matched methods (200 trials): body+PI+brake, no body, no PI, no brake. It opens
no final case unless all eleven V20 qualification attempts pass source/weight,
force coverage, load/grasp provenance and unchanged geometry/contact checks.
A qualification failure stops that queue. All selected final cases, including
failures, will be retained without tuning/replacement. Reports are generated only
after actual completion, using Wilson intervals and Holm-corrected paired exact
tests. Runtime settings include raw rate-zero (every physics step), source
float64/damped backend, 0.5 ms step, 200 solver iterations and no concurrent owned
ML work. The old final31 gate failure is preserved; no final31 cases were opened.

Live development records: .artifacts/seeded_evaluation_progress.json and
experiments/evidence/original_repair/persistent_screen32v1.json. Live aggregate:
.artifacts/active_original_perception29.json. The read-only progress observer
updates measured counts; final status is .artifacts/final_aligned32/waiting.json,
then progress.json. A process-scoped awake request permits the display to sleep
without changing the power plan. Final disk floor is 1.5 GiB; a stopped study
requires source-checked resume, never replacement of failed cases.
