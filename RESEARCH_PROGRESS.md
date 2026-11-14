# Research progress

November 14 live checkpoint: 92/100 primary attempts, SC44/46 and SFP31/46.
The unchanged final44 coordinator is running; all 80 ablations remain pending.
SFP's fifteen failures comprise nine visual rejections, three sensor-freshness
or gap failures and three waypoint timeouts. Both-connector robustness is not
established. The bounded faster SFP lip development test independently passed
at 47.0710 sim s and 8.881041 N peak with full force coverage, compared with
51.9735 sim s before that change. This is one reused case. All 31 exported
evidence hashes pass and all 110 final-source hashes remain unchanged.
Detailed accounting: `PAPER_RESULTS.md`, `experiments/checkpoints_20261114/`.

Latest SC cache pair independently passed: 26.1890 versus 27.6735 sim s, port
measurement 45.833 versus 53.654 wall s, raw peak 3.060299 N and complete
2000 Hz coverage. SC and SFP speed changes now each have one measured paired
development insertion, not randomized validation. Compact evidence is in
`experiments/evidence/cache49/` and `motion48/`. Next bounded SFP lip search
is queued; its actual outcome is still pending. Frozen final44 is separate
and at 30 accounted attempts (SC13/15, SFP12/15), with failures retained.

November 13 measured physical improvement: same-case SFP side-wall registration
16.8725 -> 11.8295 sim s (29.9% less), action 56.1585 -> 51.9735 sim s (7.45% less).
Both insertions independently verified. New raw peak 4.684120 N, complete 2000 Hz
coverage; wall time 660.3565 vs 645.2428 s, so no wall-speed claim. Evidence is
preserved outside `.artifacts` in `experiments/evidence/motion48/`. Next finite
development tests are SC cache runtime47v4 and, after it, SFP lip-search
runtime47v5. Lip search has synthetic checks only. Final44 policy/source/model
hashes remain unchanged; no held-out case is used to tune candidates.
Broader angular recovery still fails qualification. URDF and pixel-error
diagnostics explain why more wrist frames/high detector scores alone do not
resolve it. Details: `experiments/efficiency_followup48/README.md`.

Later November 13: actual SC and SFP GPU runtime insertions both verified
(3.508088 /3.256220 N peaks; complete 2000 Hz coverage). Compact camera and raw
evidence is preserved outside `.artifacts` in `experiments/evidence/runtime47`.
SFP measured-start registration is under physical comparison next; SC cache
qualification follows conditionally. SC cache outputs matched exactly on
12/12 development image sets; SFP association pruning preserved all 45 accepted
fits across four old scenes. Neither component result is called a complete
insertion speedup. Angular variants remain rejected. Current continuation and
scheduling safety: `experiments/efficiency_followup48/README.md`.

November 13 parallel development completed six paired CPU/GPU cropped-perception
replays and a real four-detector integration check of the bounded-thread runtime
repair. Candidate changes are confined to `.artifacts/dev46_workspace`; all 110
final source/checkpoint hashes match. Angular fitting remains unqualified, and
the naive two-depth SFP angular formula was rejected by actual CAD geometry.
Concurrent development rendering was stopped to prioritize final sensor coverage;
only offline diagnostics completed. Five setup/cancelled probes and final
SC700045's raw-force gap remain failures. Measurements, candidate patch and
resume scope: `experiments/parallel_dev46/README.md`. Frozen final44 continues.

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

Next active work: fixed unchanged-source V33 SFP regressions on `dev_45003`,
`dev_45006`, `dev_45005` under label `pulse36regv1`, after the verified
`pulse36v1` seat. Source map: `.artifacts/repair36/pulse36regv1_source_freeze.json`.
The serial runner preserves every outcome in
`.artifacts/seeded_evaluation_progress.json`; it may show the prior completed
batch until the first new case completes. Host execution session 53785.
No final cases are opened. Capture/decoded QA of the new successful footage is
complete: `experiments/evidence/original_repair/video/sfp_pulse36v1_verified/attempt.mp4`.

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

V29 `dev_45002_clearance36v1`: 0/1, both lips accepted but side-wall contrast
0.272984 N was below its unchanged 0.287588 N 3-SEM margin; complete raw peak
2.862218 N. V30 `sampling36v1` is now physically running a fixed 0.4 s
stationary force window with actual sensor-stamp deduplication and bounded
faster axial waypoint compensation. 46 selected tests pass. Keep its sources
frozen; the separate one-unload/slower-reload candidate remains undeployed.

November 3: completed V27 SC 1/1, SFP 1/3; V28 SFP 0/2 complete insertion.
The axial waypoint repair worked as a component in both V28 cases, but
`dev_45002` later hit the unchanged action deadline and `dev_45003` missed a
real lip because its force transient decayed during the detector window.
The host worker exited before archiving the second case; completed recorded
policy/sensor evidence was recovered with matching frozen hashes and full
2000 Hz coverage, explicitly labelled and retained as failure. No OOM event
was reported; host exit cause remains unknown. The orphan owned container
was stopped before launching any new trial.

V29 now tests the measured 0.75 mm inter-probe clearance path on `dev_45002`.
43 selected tests pass; ten-slot public CAD audit has zero collisions in 54
sampled traverses. A separate one-unload/slower-reload candidate retains
all contact thresholds and passes five ideal positive/negative tests. It
stays undeployed during the V29 trial. Final seeds unopened; no paper-ready flag.

New independent V27 SFP success: `dev_45001` held 45.739129-45.755050 mm
depth, 0.009460-0.010172 mm lateral error and complete 2000 Hz peak 3.320212 N.
V27 is now SC 1/1, SFP 1/2, with `dev_45003` running. The finite V28 follow-up
is active but waits for V27 to finish before any source mutation. It will
test the reviewed bounded axial free-waypoint compensation on the two axial
regressions, retaining all prior failures. Twelve ideal-fixture candidate tests
passed; no V28 physical success is yet claimed. Progress:
`.artifacts/repair35/axial35v1/progress.json`.

Latest late-November-2 checkpoint: V27 SC 1/1 verified; SFP `dev_45002`
0/1, still failing the initial waypoint. Command/hold diagnosis gives mostly
axial bias (60.387 um axial, 7.571 um lateral; 25 ms-separated proxy), while
V25 `dev_45000` and `dev_45001` had predominantly lateral error. V26's lateral
repair is therefore only partial. V27 `dev_45001` has now passed both lip
probes and is measuring side walls; `dev_45003` remains queued. Final cases
remain unopened. Do not mutate this active source freeze.

Actual V26 SFP video has been encoded, decoded without errors, and its
mid-insertion and terminal frames viewed:
`experiments/evidence/original_repair/video/sfp_waypoint35v1_verified/attempt.mp4`.
Independent diagnostic overlays never feed the policy. Offline candidate
checks for bounded axial compensation and shorter inter-probe clearance are
saved under `.artifacts/repair35`; neither has been deployed or physically
qualified. Longer force-window replay remains diagnostic only.

V27 `dev_42000_regression35v1` now physically verified: 15.565579-15.565594 mm
depth, complete 2000 Hz raw peak 2.463022 N. Confirmation passed 1.003 s,
145 samples, max gap 22 ms, zero resets under the unchanged 30 ms /one-second
gate. The four-case regression queue has moved to SFP `dev_45002`.
Keep its frozen sources unchanged. V25 historical counts remain SC 10/11,
SFP 4/10. No final cases opened and no broad robustness claim yet.

Late November 2 local continuation: the completed V25 screen is SC 10/11,
SFP 4/10, with complete 2000 Hz records. V26's free-waypoint PI repair passed
known SFP `dev_45000` (depth 45.746811-45.762552 mm, raw peak 3.418766 N),
separate 1/1. Its camera terminal was viewed. V27 source-frozen regressions
now run SC `dev_42000` plus three remaining SFP waypoint failures. SC terminal
poll sleep is 5 ms; original 30 ms continuity and one-second dwell gates remain.
30 selected tests pass. Current queue:
`.artifacts/repair35/regression35v1/progress.json`.
The historical V25 chart is exported/visually checked, and no final seed is open.

19:13 UTC, 2026-11-02: **first verified learned SFP insertion**, V25
`datum34v5`, nominal 1/1. Full actual depth 45.801216-45.802584 mm with
0.004699-0.006650 mm lateral error and complete 2000 Hz peak 3.253280 N.
Video saved/decoded/terminal visually inspected. This is a conditional source
clearance seat, not latch or release validation and not broad robustness.
`experiments/run_contact34_validation.py` is actively running 11 SC +10 SFP
development cases under frozen source/checkpoint hashes, no final seeds.
Status `.artifacts/repair34/validation34v1/progress.json`. Preserve the sources
while this study runs; changed files cause it to stop. Details:
[`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md).

2026-11-02 continuation: SC persistent32v1 finished 8/11, V21 known-failure
screen 2/3, separate V22 regression 1/1. Do not pool revisions. SFP adapted
models recovered detections but failed metric accuracy. New directional CAD
rim/stereo association plus active entrance/side-wall datum controller is
implemented under explicit level-fixture/upstream-angle conditions.
SFP datum34v1 refused incompatible configuration; V2 rejected age counted
after computation; V3 reached real lip contacts but rejected the second due
to a wrist-force sign bug (complete 2000 Hz, peak 3.086902 N). V24 corrects
positive compression to negative port-Z wrist reaction in SC and SFP;
226 selected tests pass. Corrected SFP datum34v4 is a fresh physical attempt.
SC needs new physical regressions after this change. No final cases opened.
Actual diagnosis, candidate boundaries, videos and prior failures:
[`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md). Current live status:
`.artifacts/active_original_perception29.json`.

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


## 2026-10-24

- Read the full attached autonomous evaluation brief, team article and adjacent
  article source, active policy/perception source, existing results, engine and
  scoring code, author instructions, special collection page and primary
  assembly literature. Created `research/gazebo-rap-evaluation` from clean
  `bd6d743c5afccada4240c588ed645476255418b6`; no commits or external publication.
- Fetched upstream phase_1, pinned `7148df9ab62b4b550b101189098f05bc0dba8444`
  (SC collision update), created isolated `.artifacts/upstream-phase1` worktree.
  It differs from this fork's qualification-style models and orchestration;
  preserved both trees rather than merging incompatible assets.
- Audited environment: Windows Python 3.14.2; RTX 4070 Laptop GPU with 8 GiB;
  WSL Ubuntu 24.04 has Jazzy and Gazebo 8.15.0; WSL memory ~15 GiB. No Kilted
  AIC installation found. Docker Linux engine unavailable; `docker desktop start`
  reported already running while engine queries failed; CLI restart timed out.
  Direct Docker launch shell command was rejected by automatic approval review
  with `blocked by policy`. No forced WSL shutdown or privileged install.
- Authored experiment plan, measured-results ledger, reproduction instructions,
  primary-source related-work notes, resumable smoke runner and evaluator-only
  seating gate. Corrected stale SC action-success documentation after reading
  actual `finish(outcome, ...)` behavior. No deployed controller changes.
- Reanalysed original SC reports: 0.4562 to 0.2633 mm median, 42.29% reduction;
  difference 95% trial-cluster bootstrap CI [−0.2516, −0.1443] mm. Each report
  has 135 groups / 45 source trial IDs. Held-out selected-crop median 0.2733 mm.
  Exported raw group CSV, source hashes, summaries, tables and inspected ECDF.
  Original images/training split unavailable; no fresh inference was claimed.
- Existing source regressions: 232 passed, 15 missing-ROS-interface import
  failures; frozen SFP test collection blocked by the same missing dependency.
  New verifier/analysis tests: 20 passed. Prepared three WaveArm scenes;
  execution preflight correctly rejected the unavailable Kilted environment.
- Native Gazebo shapes server ran 100 iterations successfully (exit 0).
- User added mandatory headless camera/model QA. First finite-iteration model
  run advertised cameras but saved no frames; retained its failed integrity
  outcome at `.artifacts/headless_assets`. Fixed the harness to subscribe to
  actual transport images before launch and wait for frames rather than assuming
  camera initialization completed before finite physics iterations.
- Headless Ogre2/EGL camera run `.artifacts/headless_assets_v2` succeeded:
  three 960×720 RGB cameras with timestamped transport frames. Manually inspected
  board/logo, NIC, SC ports, SFP module and SC plug; loader logs show embedded
  original GLB textures. This is a staged asset scene, not the complete robot.
- Ran `.artifacts/headless_assets_drop`: 12 RGB frames/camera, 246 observed
  dynamic SC plug poses, exit 0. Verified nine observed models (eight visual
  scene objects plus dynamic probe), original mesh hashes, unit scale and exact
  collision definitions. Corrected pose timestamp extraction to read the
  per-pose header (the vector envelope stamp is empty); raw per-pose timestamps
  were preserved, and analysis uses those original stamps.
- Drop probe: SC model height fell from 149.46 to 17.049 mm; final collision
  clearance to the 12 mm board surface was −0.000855 mm; terminal 0.3 s height
  span 0 mm at recorded precision. Peak sampled geometric penetration was
  **1.394 mm** after impact. This rules out assuming visual plausibility proves
  precision-contact fidelity. No contact-force/insertion accuracy claim follows.
  Selected unmodified images, scene/hash manifests, trajectory and inspected
  physics plot are preserved in `experiments/evidence/headless_assets/`.

## Next executable milestones

1. Carry validated handoff/load support into each actual policy trial.
2. Validate live policy perception, sensor timing and calibration across targets.
3. Port close-handoff orchestration; add evaluator-only requested-target,
   depth/contact/fixture logging and no-auto-latch research mode. Validate
   low-speed contact/timestep sensitivity before measuring insertion success.
4. Pilot SC paired visual-only/full-recovery experiments, freeze parameters,
   then execute held-out scenes. Add SFP only once it passes the same gates.

## Corrected environment and Docker restoration

- The initial asset-only scene was rejected as an incomplete testbed. It is
  preserved as historical QA, explicitly superseded by `evidence/lastinch`.
  Its DART drop result is not evidence against official Bullet Featherstone.
- Rebuilt the last-inch scene using five dual-port NIC cards, five SC adapters,
  five full original cable assemblies, both five-seat staging types and the
  official UR5e/tool/three wrist cameras. Kept pickup/transport outside scope.
  Actual headless SC/SFP frames and observed scene graphs passed inspection.
  Thirty pilot reset geometries plus one perturbed final reset passed checks.
- Prepared 290 matched cases / 870 planned method trials. They remain marked
  not run; freezing a configuration is not an experimental result.
- Native Bullet joint PID hold failed with 4.07 mm TCP drift. No insertion
  result is inferred from it. Corrected disconnected static cable trees so
  Bullet no longer disabled their physics links.
- User authorized Docker/WSL recreation as needed. Launching Docker Desktop
  restored Linux engine 29.1.5; Ubuntu 24.04 WSL2 was already installed. No
  distro deletion/reinstall or forced WSL shutdown was needed.
- Pulled and pinned official evaluator digest `9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923`.
  It provides Kilted/Gazebo 9.5/Xvfb. Used pinned Phase 1 source geometry and
  bringup overlay, with the image's compiled controller provenance recorded.
- Participant ROS imports work; the formerly import-blocked source suite now
  passes **270 tests**, including frozen SFP validation. Research suite: 23 pass.
- Real AIC Cartesian impedance controller activated headlessly; six actual
  cameras worked. First hold error 1.175 mm. Setup-only bounded load calibration
  estimated 0.563 N upward support and reduced the terminal half-second maximum
  TCP position error to 0.000775 mm over one 8 simulated second diagnostic.
  This does not establish cable-tip/orientation/contact accuracy or generalize
  across resets. It is not an insertion trial.
- Corrected a research documentation error: the SC controller returns its bool,
  but the outer action wrapper DOES convert SC misses to action success by
  default. Research must set `RL_INSERT_SC_MISS_ABORTS=1` and
  `RL_INSERT_REPORT_MISS_AS_SUCCESS=0` and use independent seating verification.

## Executed final last-inch environment checks

- Recreated the owned evaluator container with shareable IPC and 2 GiB shared
  memory. Verified RTX 4070 access in the participant container. Prewarmed SC
  outside the timed action and retained the original image-age gate.
- Added a small compiled research grasp plugin. Startup holds the cable while
  paused, applies the real robot reset, removes that startup fixture, then
  attaches a prescribed rigid grasp. Controller activation took seven physics
  steps. The bootstrap verifies actual clock advancement after cold rendering,
  rather than trusting an unpause acknowledgment. Runtime acceptance records
  are checked against the exact world hash, preventing stale-pass reuse.
- Collision masks alone still reported finger contacts. Final last-inch scope
  explicitly excludes gripper/ATI collision geometry while retaining their
  visuals/inertia and every connector/port collision. This approximates a
  prescribed grasp and excludes gripper-board collision or pickup claims.
- Setup-only calibration measures the actual attached grasp and load support.
  Final world `16184d8d2f6e4b8f272f288de6bc4bb7b9c891887d35aa4fd47ac7f441c2dfb7`:
  eight simulated second hold passed, terminal TCP error maximum 0.000304 mm.
  Independent tip error after contact/retract was 0.00415 mm and 0.0113 degrees.
  These are single-reset simulator readings, not hardware accuracy or a sweep.
- The actual Scene service reported all five NICs, five SC adapters, five
  cables, ten staging fixtures, the UR5e and three external QA cameras. All six
  actual cameras delivered images. Final overview and close-up were inspected.
- Deliberate 2 mm lateral miss at 2 mm/s produced 126 requested-port contact
  pairs, zero gripper contact pairs, peak tared force 0.775 N, and completed
  its retract command. Program and shell exited zero. This validates a collision
  positive control, not seating success or force/contact-model accuracy.
- Final real-policy SC2 pilot ran with GPU, shared IPC and current reset gates.
  It returned false at visual-grasp priming (10.07 px reprojection > 6 px),
  issuing zero motion commands. No scoring event or ground-truth pose was
  substituted. Same-frame crop and cuboid-correspondence diagnostics did not
  clear the vision gate. The cause still needs calibrated image-label analysis.
- Exported raw world, scene observation, hold/contact streams (losslessly
  compressed), camera frames, world hashes, checkpoints and outcome summaries
  under `experiments/evidence/lastinch/`. Historical failures remain in
  `.artifacts/`; the old asset-only screenshot is explicitly superseded.

Remaining: validate all target resets and the final seating recorder; diagnose
SC image/keypoint transfer before running paired insertion ablations. The 870
trial manifest is still a prepared protocol, not executed reliability evidence.

### Insertion continuation, 2026-10-24

- Synchronized image/actual-cable/actual-camera audit implemented in
  `experiments/capture_projection_audit.py`. CameraInfo focal length agrees
  with the source 0.8718 rad horizontal field of view. Timestamped optical TF
  agrees with actual Gazebo camera poses within 0.025 mm / 0.002 degrees
  during reset and within 0.001 mm / 0.0001 degrees at rest. RGB conversion,
  zero distortion, pixel dimensions and optical-axis conventions were checked.
- Fixed a production defect in `InsertionPolicy._build_views`: wrist TF is
  now looked up at each image stamp. Latest-TF mixing displaced the camera
  by about 2 mm during reset. Two regression tests and the selected existing
  controller/perception/research suites passed: 203 tests. The stationary SC
  network still fails unchanged reprojection/shape gates; checkpoint and asset
  identity match. Its keypoints disagree with source geometry substantially.
  Do not claim that the network transfer problem is solved by the timing fix.
- Implemented an explicitly prescribed-grasp last-inch policy. It uses the
  fixture design, controller TCP/wrench and protocol handoff gap; it does not
  read scene/cable/port truth or the setup's measured GT grasp. This is a
  separate benchmark condition, not a fallback in the deployed visual policy.
- Executed several real SC and SFP motions with raw truth/contact/wrench data
  and camera frames. SC reached a stable 15.567 mm geometric seat (nominal
  15.64 mm), about 0.017 mm lateral and 0.022 degrees angular error, 0.95 N peak
  tared wrist force. SFP reached a stable 45.80 mm geometric seat, about
  0.028 mm lateral and 0.020 degrees angular error, 1.27 N peak force. These
  early runs are **not counted as verified successes**: contact transport tails
  and distal cable contacts violate or leave gaps in the strict verifier.
- Expanded collision recording to all active cable and robot collision bodies.
  This exposed distal connectors/cable segments hitting board/NIC clutter;
  held-plug contact alone cannot explain every wrist impulse. Source SFP
  geometry is clear at its nominal seat; a terminal target contact cannot be
  assumed. A deliberate SFP contact positive control remains necessary.
- Source Gazebo 9.5 Contact publishes only nonempty contact events. Added a
  read-only physics-component observer (`experiments/gazebo/ContactAudit.cc`)
  that publishes empty-contact heartbeats and actual support-position errors.
  It does not command the robot or change contact geometry.
- Remote fixed-support experiments were rejected: although two detachable
  joint components existed, the supposed remote support drifted about 117 mm.
  The new hold gate catches this. Do not use that boundary condition or count
  its trials as policy failures. Unconstrained source-cable trials are being
  repeated with the new observer; all failed versions remain archived.
- A separate development manifest (seeds 3000–3009) now spans all five SC/NIC
  indices, both SFP slots, 5–15 mm gaps, lateral and angular perturbations.
  The runner archives each world, hashes policy/profile sources and supports
  initial-pose, physical-mouth vision, and bounded vision/force variants.
  No seeded comparison is claimed until its recorded trials complete. The
  original final 100000-series cases remain untouched.

### Verified seats and development continuation, 2026-10-24

- Independent read-only contact-component heartbeats resolved the recording
  ambiguity. Accepted nominal SC seed 1001: final-second depth
  15.5668?15.5674 mm, lateral ?0.0168 mm, full angle ?0.02276?, terminal
  penetration 0.001624 mm after signed-distance correction; independent FT peak 0.9654 N; 27.642 sim /
  119.558 wall s. Accepted nominal SFP: 45.7959?45.7993 mm, lateral
  ?0.0331 mm, angle ?0.02063?, corrected penetration 0.001116 mm; 1.9218 N;
  57.802 sim / 307.477 wall s. These are held geometric seats only.
- Qualified an actual SFP 2 mm lateral-miss collision control: 1,326 physical
  tip/cage pairs (2,135 raw cached pairs) and an 8.036 N policy-sampled force abort. Nominal SFP has
  source-geometry clearance, so its terminal check uses a matching collision
  contract, source-box clearance and live physics observation.
- Pilot v5 completed six matched trials on seeds 3000/3001. SC guarded
  baseline passed; SFP guarded baseline failed lateral alignment. Four analytic
  physical-mouth visual variants rejected before motion. Their recovery branch
  was never reached; this is not a recovery-benefit measurement. SC accepted
  views disagreed by 2.15?2.38 mm against the unchanged 0.6 mm gate. SFP had
  only one usable view. Original neural-grasp transfer remains a separate issue.
- A stationary epipolar diagnostic found 7/8 SC pad-6 correspondences with
  8.32?26.98 px Sampson distances. Direct rigid pixel fitting still leaves
  13.48 px mean reprojection / 14.85 mm diagnostic tip error. Neither camera
  repair, cuboid symmetry changes nor a different rigid-fit objective rescues
  the recorded predictions. Original training images/labels are unavailable.
- Two seed-3001 tactile v1 prototypes aborted on a 0.2 Nm sensor-origin torque
  norm guard before probing. A distinct force-guarded revision is prepared,
  not yet executed: translation keeps the common 8 N guard, commands are capped
  at 0.5 mm forward lead, opposed clipping/force evidence is required, and
  angular probes guard shifted tip-axis torque. No success is claimed for it.
- The extended baseline matrix is executing seeds 3002?3009. It has exposed
  stalls, lateral failures and a pre-entry force abort. Two invalid SC slot-1
  entries fail before simulation. Manifest revision 3 corrects only these slot
  indices; revision 2 is preserved in .artifacts/development_cases_revision2.json.
  New target-validation regression tests prevent this error.
- Fixed a measurement-memory defect: full-resolution segmentation masks were
  retained and expanded into JSON. They are now eagerly replaced by dimensions,
  pixel count and SHA-256; raw PNGs remain. Docker's backend was restarted after
  the old logging failure, preserving images/volumes and all raw evidence. The
  eligible v5 batch completed without this failure. Research tests: 28 passed.
- Exported nominal and v5 traces, source snapshots, world/profiles/proof, images,
  CSV/JSON outcomes and Wilson intervals to experiments/evidence/insertion_development/.
  Calibration/prediction/epipolar evidence is in perception_development/.
  Added recorded depth/force and headless-image figures. The preliminary
  manuscript is paper/MANUSCRIPT_DRAFT.md; final reliability and control novelty
  are explicitly unestablished. No final 100000-series case has been used.

### Signed-distance audit and control development, 2026-10-26

- Corrected the contact evaluator after auditing the exact GzPhysics 8.3.0
  plugin: negative signed distance means penetration; positive cached gaps
  cannot prove touch. All completed archived trials were reverified under
  unchanged numeric gates. Nominal SC/SFP remain verified. Raw records and
  legacy verdicts remain alongside corrected evidence. Corrected nominal
  whole-trial penetration is 0.001762 / 0.06793 mm, respectively.
- SFP contact-positive control has 1,326 physical touching pairs among 2,135
  cached pairs. Plugin contact-wrench values are not used as forces; wrist FT
  is recorded and analysed independently. Contact identity is link-level.
- Force-tactile revision 3 failed: full-pose stopped on tip-axis torque guard;
  translation completed travel but missed lateral/angular gates. Its largest
  penetrations are distal cable-link/NIC-mount contacts, not the held plug.
- Revision 4 repeated SFP seed 3001: terminal lateral error 0.05924 mm and
  own lateral tracking error below 0.00223 mm, but angular error 0.101183 deg
  still fails 0.100000 deg. Depth 45.72955-45.73034 mm; independent peak
  3.0530 N; 80.592 simulated / 458.816 wall s. Whole-scene penetration
  reached 2.1566 mm. No successful recovery claim; no threshold relaxation.
- Fixed setup-only calibration to pair TCP/cable samples at identical sim
  timestamps. The new SFP hold passed at common timestamp 1.21 s. Production
  policy receives no per-trial GT grasp, target pose or cable pose.
- Export now separates debugging revisions so repeats cannot inflate pooled
  independent trial counts. Force P95 is included with peak force. Current
  research suite: 31 passed. Two legal-index SC baseline reruns are executing;
  final 100000-series cases remain untouched.

- Legal-index SC reruns: seed 3006 achieved another held geometric seat at
  target 3 (15.5421-15.5697 mm depth, 0.19351 mm max lateral, 0.14417 deg max
  full angle, 0.02356 mm terminal penetration, 4.9882 N independent peak).
  Distal cable penetration during motion reached 5.7918 mm. Seed 3002 had
  8.0014 mm remote-end/board overlap in every pre-command packet; its force
  abort and 28.8658 N independent peak remain recorded but excluded from
  policy comparisons. The screened baseline is provisionally SC 2/4 eligible,
  Wilson 15.00-85.00%, and SFP 0/5, Wilson 0-43.45%.
- New setup gate checks all live contact components and <=0.1 mm penetration
  for 0.5 s, in addition to the existing TCP/camera gate. It rejected SC3004
  with 8.09 mm TCP error and 8.92 mm penetration before policy execution.
- CAD source boxes show no centered SFP twist contact through +/-3 deg at
  5/15/35 mm depth; the old +/-1.5 deg sweep was not a contact measurement.
  Full-pose v5 used +/-4.5 deg, slower motion and plateau/moment checks, but
  still stopped on the raw 0.08 Nm guard before target contact.
- Source sensor noise explains those raw torque stops: predicted tip-axis
  sigma 0.051456 Nm, measured detrended stationary sigma 0.049460 Nm. Eight
  of 100 post-tare samples exceeded 0.08 Nm in the recorded diagnostic. A
  conservative 50-sample estimate plus 100-sample tare uncertainty is prepared;
  0.08 Nm remains the bound value, not a raw-sample ceiling. Force stays at the
  instantaneous 8 N abort threshold. Research regressions: 33 passed.
- SFP3003 v5 translation passed lateral/angular checks but stopped short in
  depth, at 45.41835-45.41916 mm. Own axial tracking lag was 0.38044 mm.
  Revision 6 completed that repeat with terminal axial compensation capped
  at 0.25 N: verified held clearance seat, depth 45.71906-45.72022 mm,
  lateral <=0.05758 mm, angle <=0.05570 deg, independent force peak/P95
  6.3950/0.8588 N, 83.924 sim / 452.726 wall s. Whole-scene penetration
  was 1.8890 mm, so it fails the additional whole-trial 0.1 mm screen.

### Current continuation state, 2026-10-26

- Research regressions: **34 passed**. All earlier raw evidence is preserved.
- Full-pose v6 reached actual cage contact near 3.8 deg and correctly stopped
  at a conservative 0.08310 Nm bound, against 0.08000. A fixed forward sweep
  was loading its first wall; v7 reverses on statistically supported settled
  contact and checks opposed signs. No force/seat/moment value was increased.
- V7 measured +3.67460 / -3.87716 deg twist limits and recentered to about
  0.00401 deg true full angle, but stopped at 20.49 mm depth on stale wrench
  delivery in the policy path. Independent FT publication stayed at 50 Hz.
  It is a failed insertion, not a full-pose success.
- V8 adds callback draining and fixed-pose input recovery with the same 0.1 s
  freshness limit. Its repeat failed earlier at a 0.08054 Nm moment bound;
  neither telemetry recovery nor reliable full-pose insertion is established.
- Fixed abort load support: retain existing feedforward while targeting
  measured TCP pose. V6 depth shifted from 4.8952 to 7.1718 mm in 0.5 s after
  abort; v8 shifted from 4.8944 to 4.8751 mm. Raw traces remain available.
- A setup-only SC free-chain reorientation preserves all source meshes,
  collision definitions, inertias, joints, joint anchors and held-tip pose.
  Its live seed-3002 test still had 10.8637 mm remote-SFP/board penetration.
  No policy ran. The option is an explicitly failed diagnostic, disabled by
  default; do not treat its geometry unit test as physical validation.
- Terminal seats and whole-trial contact fidelity are separate exported
  endpoints. Nominal SC/SFP pass the recorded whole-trial 0.1 mm screen;
  all development geometric successes so far fail it from distal contact.

**Latest status: float64 qualification and SC outboard jobs are complete.**
Read the final outcome note at the end of this file before starting more work.
The following ownership instructions describe the previous active run.
The hidden background Python wrapper (PID 14208 at launch) owns
`research-aic-eval`. Read `.artifacts/active_precision_qualification.json` and
`.artifacts/precision_qualification/double26v1/progress.json` before doing anything
with the evaluator. Do not launch another world or edit its frozen SOURCES
until that progress status is complete/failed or the user explicitly cancels.
PID alone can become stale; the marker and process state govern continuation.

The completed `num26v2` study verified SFP3003 terminal geometric seating at
default / 1 ms and 200 iterations / 1 ms. Whole-trial penetration remained
1.8577 and 1.8916 mm respectively; both fail the additional 0.1 mm screen.
Its 0.5 ms control failed before policy execution from controller bootstrap.
The retry aligns the ROS switch request with the existing 70 s outer startup
budget, saves explicit bootstrap errors and snapshots sources before launch.
It qualified its 0.5 ms control (1,242 physical / 1,256 raw pairs, independent
peak 8.24935 N); the matching insertion completed with a terminal geometric
seat at 45.64766-45.64989 mm, lateral <=0.06367 mm, angle <=0.05556 deg,
independent peak 2.45720 N, 83.924 sim / 726.171 wall s. Whole-trial penetration
was 1.89232 mm, so smaller steps did not resolve contact fidelity. The short
pre-command screen passed. No insertion threshold
changed. Repeated seeds are distinct numerical contexts, not pooled n.
The exact installed engine returned 50 default iterations and read back 200;
the live workcell component and matching collision controls also report 200.
No held-out final case has run. No source threshold has been relaxed.

Post-hoc pre-command screening of the new numerical runs found the 200 / 1 ms
insertion invalid before motion: cable_1::link_3 / NIC-mount-0 overlap peaked
at 0.89005 mm; nine of ten final pre-command packets exceeded 0.1 mm. The
earlier setup hold had passed, exposing a hold-to-policy-start coverage gap.
First bad recorded packet was 9.911 s; first policy command was 9.992 s.
Keep its measured terminal seat but exclude it from policy comparisons. The
default / 1 ms repeat passed the short screen. Screened numerical indexes are
under `.artifacts/evidence_export_20261026_continuation/`.

Matched batch: existing SFP development seeds 3001/3003/3005/3007/3009,
`force_tactile_no_registration` versus `force_tactile_translation`, default
solver / 1 ms, source inertias unchanged, method-order shuffle seed 934.
Both share CAD prior, load support, tracking compensation, freshness handling,
approach waypoint and dwell. The ablation omits opposed lateral probes only.
The fixed config is `experiments/configs/matched_registration_development.json`.
Its wrapper archives results and runs pre-command screens and geometry audits
after all stages. Do not edit frozen SOURCES while it runs. The incremental
`.artifacts/seeded_evaluation_progress.json` can initially contain the previous
batch: check result labels before treating it as current.

ContactAudit now captures 21 chain/end-link world poses (all 20 capsules) at the SAME
PostUpdate timestamp as contact data, without changing physics or publishing
policy inputs. A late read-only capture from num26v3dt05 (49.390-94.140 s)
found the separate PosePublisher stream half a timestep out of phase; the
exact-stamp audit refused comparison. It is not a geometry validation.
Independent capsule/static-board-and-NIC-box distance math adds three tests;
the complete current research suite passed **37 tests**, and the observer
plugin rebuilt successfully. SDFormat-15 readback retained source zero mass
and 0.001 kg m² diagonal inertia on 21 segments; parser-valid is not physical
validation, and no causal inertia claim or parameter change was made.

First matched outcome: SFP3001 no-registration failed lateral alignment;
depth 45.78461-45.78549 mm, max lateral 0.25510 mm, angle 0.10033 deg,
independent peak/P95 4.49982/0.69449 N, 68.096 sim / 373.320 wall s,
whole manifold penetration 2.20090 mm. Short pre-command screen passed.
Its separate geometry audit matched 7,049 timestamps, all 20 capsules versus
187 static NIC/board boxes. At 33.281 s, cable_0/link_3 centerline entered
NIC-mount-0's collision box: >=2 mm solid overlap, also confirmed by an
independent slab-intersection point witness. The per-instant manifold field
was much smaller; full-trial max later reached 2.20090 mm. Geometry audit,
slab witness and screened partial index are under the active matched job.
The paired translation run completed: max lateral 0.06078 mm but full angle
0.10070 deg fails the unchanged 0.1 deg gate. Peak independent force 1.63914 N,
80.558 sim / 389.279 wall s, whole-trial manifold penetration 2.13832 mm.
An 8,322-match independent geometry audit also found source capsule overlap.
A read-only late PosePublisher/ECM
frame cross-check is also capturing; do not confuse nearby-stamp diagnostics
with exact-stamp geometry validation.

**Precision pivot:** `matched26v1` was intentionally stopped after three
completed trials, before its fourth policy. The wrapper reports nonzero exit;
`stop_request.json` records the researcher stop, not a policy failure. The
interrupted setup is preserved as
`.artifacts/lastinch_runs/unexecuted_precision_pivot_matched26v1/`.
Its complete executed subset is `partial_03_screened.json`. Third outcome,
SFP3003 without registration, passed terminal seating at 45.72640-45.72841 mm,
lateral <=0.05396 mm, angle <=0.05511 deg, independent peak 3.84198 N,
71.424 sim / 397.492 wall s; whole penetration 1.89095 mm. The shared prior
and compensation can solve that case without registration. No benefit claim.

Actual recorded capsule/box pose isolated in Bullet 3.24: float32 returns
+1.94224 mm separation despite source centerline/box intersection. Float64
returns -1.9761848 mm at original meter scale; scaling the query by 10/100/1000
with proportional margins gives the same normalized overlap. Sphere and
centered-capsule controls overlap correctly. Probe code, transform and outputs
are in `.artifacts/precision_diagnostic_20261026/`. This localizes a genuine
query scale/precision defect, not every whole-cable failure or a particular
numeric constant. Installed float64 libraries already exist; no upgrade needed.

Built `/opt/research-bullet-double/libResearchBulletDouble.so` from 21
unmodified 8.3.0 backend source files, with BT_USE_DOUBLE_PRECISION and Bullet
float64 libraries. New plugin SHA:
`94f9a5932091b262205d8dce2d15882e6314032f61f643c1579bb9c8db97a1ec`.
Sources/provenance: `.artifacts/gz-physics8.3-double-src/`; CMake:
`experiments/physics_probe/double_backend_CMakeLists.txt`. Original backend is
unchanged. Isolated feature/solver readback passed. The real server's /proc
maps and library hashes passed the new selected-backend audit. Sources and
new binary audit config are frozen while `double26v1` runs. It executes:
matching SFP miss control -> qualify proof -> nominal SC -> SFP3003 shared-prior
baseline. No threshold or asset inertia changed. Exact plugin hash is now part
of contact-proof matching, preventing use of float32 proof for float64 seats.
Research tests: **38 passed**. No new insertion result is assumed yet.

The earlier numerical launcher `numerical_20261026_v1` failed before trials
because the standalone capability probe lacked the installed library path.
The corrected launcher sources the pinned ROS environment. Preserve that
startup-failure log; it is not a policy outcome. Global default collision
proof is unchanged; each changed context gets its own archived proof.

Latest completed outcomes: double26v1 SFP3003 passed terminal seating at
45.798811 mm, lateral <=0.008020 mm, angle <=0.054344 deg, independent peak
1.19720 N. Whole manifold penetration was 0.035119 mm, but the independent
7,369-match sharp-source-box audit found 0.162896 mm overlap (>0.1 mm).
Investigate Bullet rounded margins versus source geometry before claiming
full-scene physical validation. Interior chain anchors stayed within
0.001678 mm across 20 observed joints. Geometry and chain audits are saved
under `.artifacts/precision_qualification/double26v1/`.
SC nominal reset failed hold (TCP 5.49261 mm, peak contact 9.27950 mm).
Queued sc_outboard26v1 completed: policy force_abort at 8.02112 N independent,
depth 2.13813-2.18718 mm, max lateral 6.10057 mm; whole manifold 0.037633 mm.
Both markers report complete; no evaluator job remains active. Current
portable export has 43 attempts and needs the latest three indexes added,
new diagnostic copies, figures and checksum refresh. No held-out trial ran.

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


## November 14: under-30-second speed/reliability work

User target is complete-action speed plus preserved SC and improved SFP reliability. See `experiments/speed_reliability54.md`. Exact association-pruning physical test runtime47v6 is running in the isolated development slot after final44 attempt 97. Faster lateral travel passed 35/35 synthetic cases and all five negative controls (11.58 to 8.02 synthetic seconds); physical v7 waits for v6 success. An offline crop-scale audit follows in a serialized slot. Joint-fit diagnosis on old dev_45009 still fails 6 px (16.19 to 13.44 px), with up to 9.18 mm view disagreement; no rejected pose or relaxed threshold was installed. All 110 frozen final44 hashes match. The under-30-second/both-connector reliability target remains unachieved; queued tests are not results.
