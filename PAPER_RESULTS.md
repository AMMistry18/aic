# Measured results and claim boundaries

November 14 checkpoint: 92/100 primary frozen final44 attempts are accounted for;
the coordinator is still running. SC is 44/46 (95.65%, all-attempt Wilson 95%
interval 85.47-98.80%); SFP is 31/46 (67.39%, 52.97-79.13%). Successful action
medians are 30.5335/59.4290 sim s and 286.9781/582.3424 wall s, SC/SFP. Maximum
raw peaks are 3.487740/7.889495 N. The two SC failures are the retained setup
interruption and raw-force coverage gap. SFP has eight grasp-reprojection
rejections, one rim-residual rejection, three sensor-freshness/gap failures and
three waypoint-tracking timeouts. The seven generic policy failures were classified
from their retained policy diagnostics; none was rerun or removed. All 80 paired
ablations are still pending. Snapshot: `experiments/checkpoints_20261114/`.
This partial study supports neither a final success rate nor a robust-both-
connectors claim. All 110 frozen source/checkpoint hashes still match.

Separately, the faster bounded SFP lip-search development case completed and
independently verified. Lip phase 21.7560 -> 17.2475 sim s; complete action
51.9735 -> 47.0710 sim s; recorded wall 660.3565 -> 569.9130 s. Raw peak rose
4.684120 -> 8.881041 N, with complete 2000 Hz coverage and no sample over the
unchanged 20 N ceiling. Stable depth 45.805496 mm, lateral 0.003725 mm and angle
0.020214 deg. Relative to the earlier runtime47v2 development baseline, the
combined changes reduce action 56.1585 -> 47.0710 sim s (16.18%) on this one
reused case. This is not a randomized improvement estimate or part of final44.
31 checksummed camera/raw/source files are preserved in
`experiments/evidence/lip51/`; all hashes were verified. The final image remains
heavily arm-occluded and does not by itself establish internal insertion depth.
Broader angular correction and multi-case optimized-policy robustness remain
unqualified. Earlier chronological development checkpoints follow.

Latest November 13 SC cache qualification: dev_42000 independently passed all
original seating/contact/dwell/raw-force checks. Action 27.6735 -> 26.1890 sim s;
port measurement 53.654 -> 45.833 wall s; recorded episode 300.0661 -> 282.1937
wall s. Raw peak 3.060299 N, complete 2000 Hz coverage. Stable depth
15.565530-15.565568 mm; maximum lateral 0.061038 mm and angle 0.134707 deg.
This is one reused paired case, not a reliability or population timing estimate.
Evidence: `experiments/evidence/cache49/` (29 checksummed files). Along with the
SFP motion pair below, both connectors now have a measured physical development
speed result. Broader angular correction remains unqualified. Faster SFP lip
search is queued separately and has no physical result yet. Final44 remains
the unchanged frozen policy and its own study, not a pooled optimized cohort.

November 13 measured SFP speed pair (known dev_45005, not final trials): the
measured-start contact paths independently seated with complete 2000 Hz force
coverage. Side-wall registration decreased 16.8725 to 11.8295 simulated seconds
(29.9%); complete action decreased 56.1585 to 51.9735 (7.45%). Raw peak increased
3.256220 to 4.684120 N, remaining below the unchanged 20 N ceiling. New stable
depth 45.804919 mm, lateral error 0.005392 mm, angle 0.020210 deg. Recorded wall
time increased 645.2428 to 660.3565 seconds; no wall-time improvement is claimed.
This is one paired development case, not a reliability estimate. Full compact
evidence: `experiments/evidence/motion48/`, baseline in `evidence/runtime47/`.
SC image/CAD reuse is queued for physical qualification; a faster bounded SFP
lip search has passed synthetic controls only and follows conditionally. Neither
is a physical result yet. All final44 sources/checkpoints remain frozen.
Angular fitting remains unqualified; repeated wrist motion cannot change
gripper self-occlusion because the three cameras share the rigid wrist mount.
Full component measurements and claim limits:
`experiments/efficiency_followup48/README.md`.

November 13 later speed-development checkpoint: complete GPU/bounded-thread
runtime qualification independently passed SC42000 and SFP45005. Stable depth
15.565526-15.565562 /45.805618 mm; raw peaks 3.508088 /3.256220 N, both with
complete 2000 Hz coverage. Action time 27.6735 /56.1585 simulated seconds;
recorded wall time 300.0661 /645.2428 seconds. These are reused development
cases and are not added to final44 or treated as a robustness estimate. Full
checksummed force/contact/camera evidence is preserved outside `.artifacts`:
`experiments/evidence/runtime47/`. The runtime improvement alone did not
substantially shorten physical insertion. The faster SFP measured-start
registration is now under separate physical test; SC image/CAD reuse follows
only after that succeeds. Offline cache/pruning and rejected angular fits are
reported separately in `experiments/efficiency_followup48/README.md`.

November 13 short-development update: the unchanged cropped plug detector and
weighted multiview pipeline made matching CPU/GPU decisions on 6/6 historical
development image triplets (three per connector, five accepted poses and one
rejection). Median CPU/GPU wall times were 5.7431/0.1133 s SC and
5.3209/0.1230 s SFP under a one-core CPU quota on a shared laptop. These exclude
port/rim/body fitting, camera waits and motion; they are not insertion speed
results. A CPU initialization thread-budget override was identified and a
runtime callback repair passed actual inference in all four learned detectors.
It is installed only in a copied development workspace, pending physical
qualification. All 110 frozen final source/checkpoint hashes remain unchanged.

Angular diagnostics on four old static image sets per connector improved
SC median orientation error from 1.9188 to 0.5141 deg with joint image fitting,
but only 1/4 met 0.25 deg. SFP improved on only 1/4 cases; none met 0.10 deg.
All eight sets retained their production synchronization rejection; this is
static geometry analysis only. Four six-DOF SC body fits all failed the
existing view-consistency check. Broader angular insertion is not verified.
Full measurements and the rejected naive two-depth contact-angle idea:
`experiments/parallel_dev46/README.md` and `measured_summary.json`.

The 17-attempt final accounting checkpoint for this update was SC 7/9 all attempts,
SFP 6/8. SC includes one setup interruption and one raw-force coverage failure;
SFP includes two policy/perception failures. The force gap was 0.3605 s during
overlap with development rendering. Contention is a possible contributor,
not an established cause. Concurrent development rendering was stopped and
that final failure is retained. Shared-laptop wall times must disclose the
overlap. Frozen final44 continues; these partial counts supersede the older
eight-success checkpoint immediately below, without pooling cohorts or
updating the manuscript's older snapshot.

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

November 12 recovery checkpoint: original adapted checkpoints and latest raw
qualification41v2 archives were absent. New `recovery20261112` weights were
trained on the original development split recipes and are a separate revision.
Fresh independent verification passed SC dev_42000 and dev_43001 (2/2), and
SFP dev_45005 (1/1); the 17-case historical replay is continuing. This is not
yet a reproduction of the complete historical SC 8/9 and SFP 8/8 counts.
SC dev_42000: 15.565555–15.565579 mm terminal depth, maximum lateral error
0.120401 mm, raw peak 3.322033 N, 28.522 simulated /286.514 wall seconds.
SFP dev_45005: 45.805251–45.816004 mm terminal depth, maximum lateral error
0.004608 mm, raw peak 3.106516 N, 57.458 simulated /715.483 wall seconds.
Both have complete 2000 Hz coverage and unchanged independent seat/contact
gates. Exported evidence and checksums are under
`experiments/recovery_20261112/verified_evidence/`; later replay outcomes are
exported under `experiments/recovery_20261112/historical_batch_replay/`.

The SFP final stroke took 6.96 simulated seconds; its 50 Hz policy trace had
no 3 N travel-pause samples and a 1.411957 N travel peak. These are separate
from the whole-action 2000 Hz raw-force peak. Most latency occurred before
the stroke; see `experiments/recovery_20261112/timing_audit.json`.
An unqualified speed43 candidate and six matched physical development trials
are queued after the unchanged baseline. Eight motion/guard regression checks
passed; no speed43 physical result is claimed. The 100-primary-trial combined
protocol is `experiments/configs/recovered_robustness43_protocol.json`; final
cases have not been generated or executed, and matched ablations remain pending.
Historical checkpoints below retain their original revision and evidence scope.

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

Timing definition correction for the checkpoints below: `policy.wall_s`
includes Python policy configuration, while `policy.sim_s` is measured from
the overwritten action-start simulation timestamp. Both exclude preceding
Docker reset/bootstrap/hold calibration. For V31, wall time was 686.127787 s,
configuration 15.688700 s, hence 670.439087 wall s after configuration;
recorded action simulation time was 79.466 s. Earlier phrases saying both
durations include configuration should be read with this correction. Compare
the directly logged insertion segment (V31 18.4435 sim s) separately when
testing travel speed; whole-attempt durations also vary with perception and
contact stages. No archive values are changed.

Offline mechanism check on the independently verified V31 SFP case: visual
rim residual was 1.017133 px, but estimated relative insertion depth exceeded
independent physical depth by medians 0.908293 /0.908034 mm at the opposed
lips. The policy's contact correction was 0.905281 mm. After correction,
terminal median relative-depth discrepancy was 0.002065 mm (45 paired own
observations; range -0.004863 to 0.002943 mm). This is one-case diagnostic
evidence, not global mouth/grasp pose accuracy or a population estimate.
Truth was read only afterward; it did not determine the applied correction.
Timestamp join tolerance was 20 ms. Reproduce with
`python experiments/diagnose_relative_depth36.py`; source artifact:
`.artifacts/repair36/relative_depth_diagnostic.json`.

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

Latest November 3 development: V29 `dev_45002_clearance36v1` completed **0/1**.
Both lip measurements passed, differing by 3.767 um. The modified crossing
reached aligned entry at sim 50.7495 s, versus 55.9285 s in the preceding
V28 run of this case (different camera acquisitions; descriptive comparison).
It then rejected side-wall axis 0: force contrast **0.272984 N** versus
**0.287588 N** 3-SEM margin. Complete 2000 Hz peak was **2.862218 N**;
duration 492.449128 wall /55.075 sim s. A correct noise rejection is not
insertion success. Archive: `.artifacts/repair36/clearance36v1_completed.json`.

V30 `sampling36v1` now tests two bounded implementation repairs: a fixed
0.4 s stationary side-wall window with **actual wrench timestamp deduplication**,
and free-waypoint axial integral gain 500 rather than 200 N/(m s), retaining
the 0.15 N saturation and 0.5 mm total pose lead. The gain lies below the
625 N/(m s) critical value of the ideal 500 N/m, 100 Ns/m axial model;
this model calculation does not establish stability of the full robot.
The same stationary-velocity, geometry, reaction/3-SEM, force and deadline
criteria remain. Forty-six selected tests pass, including stale-force and
high-noise rejection; V30 physical outcome is pending. The lip retry remains
a separate undeployed candidate. No held-out final trial has been opened.

## November 3 continuation: completed regressions and remaining defects

V27 `regression35v1` finished: SC **1/1**, SFP **1/3**. Preserve both SFP
waypoint failures. V28 `axial35v1` finished with SFP **0/2 complete insertions**;
both initial waypoints passed after bounded axial compensation, which is a
component repair rather than insertion success. Forty-two selected tests passed
after V28 installation.

| V28 case | Actual outcome | Complete 2000 Hz peak |
|---|---|---:|
| `dev_45002` | Both lips and both side walls measured; 720 s action deadline at estimated 23.237 mm depth (independent final window 21.620-23.455 mm), below the 45.8 mm seat | 3.267764 N |
| `dev_45003` | First waypoint passed; first lip could not satisfy the reaction gate despite physical lip contact; no accepted datum | 3.147061 N |

The second V28 trial was recovered after the host worker exited and left the
owned Docker simulator running. Original policy failure and recorded streams
were preserved. Frozen source/checkpoint hashes match; post-hoc verification
found complete 2000 Hz action coverage and zero samples over 8 N. Container
memory events reported no OOM kill. The exact host-process exit cause remains
unknown. This is explicitly marked `recovered_after_host_worker_exit`, not a
clean uninterrupted automation result, and it remains a failed insertion.

Policy-observation diagnosis of `dev_45003`: seven windows met the command
advance, <=25 um motion, >=150 um lead and <=50 ms gap checks, but their
compression changes were **-0.208 to -0.110 N**, below the unchanged +0.025 N
reaction-rise gate. A declining/noisy reaction window coincided with saturation
of the 0.3 mm lead; later stationary windows had too little commanded advance.
Impact decay is a plausible fixture model, not an independently identified
unique physical cause. The measured limitation is the one-shot opportunity to
meet all gates without subsequent controlled re-excitation. No threshold was
relaxed. Exact windows: `.artifacts/repair36/lip_diagnosis.json`.

V29 `clearance36v1` now tests **only** the shorter inter-probe path on
`dev_45002`: 0.75 mm measured clearance and a second start referenced to the
first lip, with both independent lip measurements still mandatory before
accepting the axial coordinate. Public collision sweeps cover all ten slots
(two distinct canonical box geometries), 54 sampled traverses, zero collisions
at the sampled 0.75 mm clearance / +/-0.1 degree angular components. This is
geometry screening, not physical qualification. Forty-three selected tests
pass; V29 physical outcome is pending. No held-out final case is opened.

A separate candidate uses one bounded unload and slower reload after a
kinematic blocking trigger. It retains the full original contact acceptance
test and requires >=120 um measured release. Five ideal-fixture tests pass,
including impact decay, free air, missing reaction and failed release. It is
not deployed while the V29 source freeze runs and is not a Gazebo result.

The raw policy-observed force/depth trace is exported to
`experiments/evidence/original_repair/lip36/missed_contact.png` (also SVG/PDF),
with source hashes and no smoothing. The figure was viewed; substantial force
noise is visible and must not be misrepresented as a clean deterministic
impact signature. The independent 2000 Hz vector-force ceiling remains a
separate audit from this lower-rate axial observation.

Earlier dated checkpoints below are retained history and are superseded by
the completed counts above. Revisions are never pooled for a reliability claim.

Subsequent V27 SFP `dev_45001_regression35v1` passed independent clearance-seat
verification: depth **45.739129-45.755050 mm**, lateral
**0.009460-0.010172 mm**, angular **0.004004-0.004092 degrees**; complete
2000 Hz raw peak **3.320212 N**, zero samples over 8 N. Duration
654.496564 wall /77.8845 sim s. V27 completed totals are now SC **1/1**,
SFP **1/2**, with `dev_45003` still running. Source/checkpoint hashes are
unchanged. Local analysis/video encoding overlapped this development run;
these are logged durations, not controlled throughput benchmarks.

A finite follow-up waits for V27 completion, checks all frozen hashes and
the stopped evaluator, then installs only the reviewed bounded free-waypoint
axial compensation candidate, reruns unit tests and executes `dev_45002` and
`dev_45003` as V28 `axial35v1`. It preserves V27 source backups and creates
new archives. Twelve candidate unit tests passed before deployment; physical
V28 outcomes are pending. No threshold, checkpoint, inter-probe path or force
window change is included. Progress:
`.artifacts/repair35/axial35v1/progress.json`.

Latest V27 SFP regression: `dev_45002_regression35v1` **0/1**, initial
waypoint timeout before any lip measurement; 192.892735 wall /21.296 sim s,
complete 2000 Hz raw peak **1.019469 N**. Its final approach command was stable;
subtracting the subsequent failure-hold command gives a diagnostic proxy of
60.387 um axial /7.571 um lateral error, separated by 25 ms. This supports
investigating bounded axial load-bias compensation, not relaxing the 40 um
waypoint tolerance. At that checkpoint V27 completed totals were SC 1/1 and SFP 0/1;
two SFP trials remained pending. These selected regressions are not reliability
estimates. Sources remain frozen while they execute.

Offline candidate checks are kept separate: ten ideal damped-load waypoint
fixtures passed with bounded axial integral compensation, and 81 canonical
public-CAD inter-probe sweeps had no plug/NIC box intersections. Neither
candidate has been deployed or physically qualified. Recorded-force replay
on the noisy V25 `dev_45006` axis 1 gives 0.314540 N contrast versus 0.193868 N
3-SEM over .4 s (20 distinct sensor samples per side). Paired own-robot
lateral observations were stationary over that window; full tip-pose
stationarity is not archived. This replay does not change its failed verdict
or establish a calibrated statistical significance claim.

## Subsequent repair regression — V26, separate from V25

V27 first physical result: SC `dev_42000_regression35v1` passed **1/1**,
depth 15.565579-15.565594 mm, lateral 0.120683-0.120720 mm, angle
0.134543-0.134582 degrees, complete 2000 Hz peak **2.463022 N**. Observable
confirmation held 1.003 s across 145 samples, maximum gap 0.022 s, zero gap
resets. This prospectively repairs the confirmation failure without changing
the accepted gap, dwell, geometry or force thresholds. The remaining three
SFP regressions were pending at this checkpoint; the newer outcome is above.
Do not pool this result into V25's 10/11.

`dev_45000_waypoint35v1` passed **1/1** after enabling lateral PI during free
waypoints and restoring the probe freeze only for actual measurements. This
is the same target-0 development case that failed its initial waypoint in V25,
not an independent reliability sample. Stable verified SFP depth was
45.746811-45.762552 mm, lateral 0.006669-0.007756 mm, angle
0.023680-0.023732 degrees. Complete 2000 Hz raw peak was **3.418766 N**,
with zero samples over 8 N. Total harness duration was 740.653069 wall /
79.5275 sim s including configuration; the unchanged 720-second action timer
started after configuration and did not abort. Actual terminal camera geometry
was visually inspected. Timing margin remains a concern.

New actual V26 footage:
`experiments/evidence/original_repair/video/sfp_waypoint35v1_verified/attempt.mp4`.
It decoded without errors; middle and terminal frames were viewed. Source
camera stamps and a causal held-frame provenance map are recorded alongside.
The independent depth overlay is diagnostic only and never a policy input.

V27 `regression35v1` is a new source-frozen four-case study: SC `dev_42000`
and SFP `dev_45002`, `dev_45001`, `dev_45003`. It retains the V26 waypoint
change and polls SC terminal dwell with 5 ms requested sleeps, leaving the
30 ms continuity and one-second dwell thresholds unchanged. The failed V25
SC command trace had 14 gaps over 30 ms and a longest uninterrupted span of
0.9475 s. Thirty relevant unit tests pass; later physical outcomes are above.
Status: `.artifacts/repair35/regression35v1/progress.json`.

V25 development plot and machine-readable evidence:
`experiments/evidence/original_repair/validation34v1/outcomes.svg` and
`outcomes.png`, visually checked. No subsequent repair cases are included.

## Completed V25 robustness-development screen — November 2, 2026

`validation34v1` completed all 21 cases under frozen sources/checkpoints.
No final seeds were opened. These are development results; the SC cases are
selected regressions, not a random population sample.

| Connector | Verified complete trials | Descriptive Wilson 95% interval | Maximum raw force | Median successful time, sim /wall |
|---|---:|---:|---:|---:|
| SC | 10/11 (90.9%) | 62.3-98.4% | 3.5624 N | 31.7395 /271.7379 s |
| SFP | 4/10 (40.0%) | 16.8-68.7% | 3.3291 N | 72.2090 /696.2629 s |

All 21 raw wrist records have complete 2000 Hz coverage. The SC failed trial
passed independent physical seating geometry/contact but the policy timed out
in observable confirmation; retain it as a failed complete trial. SFP failures:
four initial waypoint-tracking timeouts before probes, one 720-second action
deadline during travel (about 42 mm reached), and one side-wall force difference
below its unchanged 3-SEM noise margin. Root-cause analysis must distinguish
physical success from policy completion and a valid noisy-contact abstention
from a controller defect. No broader robustness claim is supported.

Analysis: `.artifacts/repair34/validation34v1/analysis.json` and `trials.csv`.
The next V26 development regression (`waypoint35v1`, `dev_45000`) enables
lateral tracking during free waypoint motion, restoring the original probe
freeze for actual force measurements. It keeps all geometry, force, timeout
and noise thresholds. Twenty-five relevant controller/probe tests pass; the
later verified result is above. Never pool V26 with this completed V25 batch.
Paper/test plan: [`paper/ROBUSTNESS_AND_PAPER_PLAN.md`](paper/ROBUSTNESS_AND_PAPER_PLAN.md).

Earlier dated partial counts below are historical checkpoints.

19:18 UTC: corrected-sign SC `validation34v1` first regression passed,
**1/11 completed**, depth 15.563789-15.563800 mm and complete 2000 Hz peak
2.254981 N. The separate SFP nominal qualification is **1/1**; none of the ten
varied SFP cases has completed yet. The finite queue continues. Both have
verified simulated insertion under declared alignment conditions, but neither
has a completed final reliability study or new matched ablations.

**Latest, 2026-11-02:** V25 learned SFP +RGB rim/contact-datum pipeline passed
its first nominal case (**1/1**, not a robustness estimate). Stable independent
depth 45.801216-45.802584 mm, lateral 0.004699-0.006650 mm, angle
0.018342-0.018564 degrees; complete 2000 Hz peak 3.253280 N. The seat is a
source-model clearance region with a separately qualified collision control;
no latch/release/electrical claim is supported. A frozen development queue
now runs 11 SC regressions and ten varied SFP cases. Final seeds remain closed.
See [`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md) for exact provenance,
video, timing and limitations. Earlier pending states below are historical.

Latest physical checkpoint: corrected SFP `datum34v4` registered entrance and
both side-wall axes, reaching **35.51659/45.8 mm** before the unchanged 720 s
action deadline. It remains **0/1**, with complete 2000 Hz peak 3.139588 N.
V25 `datum34v5` is testing a bounded free-travel lead correction for impedance
damping, keeping terminal and independent verification limits unchanged.
No new learned SFP insertion has yet been independently verified.

Subsequent force audit: SFP `datum34v3` completed **0/1**, stopping on its
second entrance-lip probe; 2000 Hz force coverage was complete, peak 3.086902 N.
The wrist reaction sign was wrong in the new axial detector. Correct-sign
replay detects both physical lip stops with unchanged thresholds; V24
`datum34v4` is undergoing fresh physical validation. The SC axial component
shared that sign assumption and now also requires new regressions. Historical
SC independent seating verdicts remain valid outcomes; their force-based
mechanism and robustness claims must not be carried over to the new revision.
All 226 selected regressions pass. **Both connectors remain unqualified for a
combined robustness claim; final insertion trials remain unopened.**

2026-11-02 subsequent checkpoint: SC `axial33v1` repaired **2/3** known
failures, and the distinct V22 `partial33v1` regression passed **1/1**. These
versions are separate denominators, not a pooled robustness rate. Both axial
repairs and the V22 regression have complete 2000 Hz force coverage, peaks
2.630116 /2.347025 /3.102888 N, and independent stable seating contact.
The V22 first partial visual candidate was followed by an original full-gate
pass, so its causal benefit is not isolated. Adapted SFP perception improves
detection but remains metrically inaccurate. `datum34v1` failed configuration
before movement because its inherited pull-lead demand exceeded the 6 N guard;
the reduced-lead `datum34v2` physical development attempt is pending. **Neither
connector is declared robust or paper-ready; final trials remain unopened.**
Details and videos: [`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md).

Latest measured update (2026-11-02): SC `persistent32v1` finished **8/11**
(known repairs 2/2, new development cases 6/9), all with complete 2000 Hz force
coverage; final trials stayed unopened. The original SFP learned-policy trial
failed port perception after valid setup; a separate clamp setup failed before
policy execution. Both connectors remain required. See
[`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md) for actual failures and
the unqualified repairs now under test. Earlier running counts below are
historical snapshots, not current totals.

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


2026-10-26. This file contains actual reanalysis and executed simulator trials.
Nominal SC and SFP insertions now pass independent seating verification under an
explicit prescribed-grasp condition. The original visual-grasp policy still
fails its unchanged gate. Seeded method comparisons and final reliability are
not yet established. Prepared cases are not experimental results.

## Contact-report correction, 2026-10-25

The pinned GzPhysics 8.3.0 Bullet plugin reports **signed distance**, with
negative values denoting penetration. The earlier evaluator incorrectly
treated positive cached manifold separation as penetration. All archived
completed trials were reverified with the audited binary hash and corrected
sign, without changing any seating threshold. Both nominal seats and SC seed
3000 remain verified; the other policy outcomes remain failures. Raw records
and original verdicts are preserved alongside corrected verdicts in the export.
The corrected SFP control has 1,326 physical touching pairs among 2,135 raw
cached pairs. Force measurements come from the independent wrist FT stream,
not the plugin's contact-wrench field. Collision identities are used at link
level, because the upstream 8.3.0 collision-shape naming is unreliable.
See `.artifacts/contact_reverification_20261025/index.json` and the portable
`experiments/evidence/insertion_development/` export. The reported values below
supersede earlier penetration and contact-count interpretations.

## Seeded development pilot (completed, not a final comparison)

Batch `pilot_v5` executed six matched trials on development seeds 3000 (SC0)
and 3001 (NIC0, SFP slot 0), at a 1 ms physics step. Both scenes had randomized
handoff gap, lateral offset and full angular error. One case per connector and
method is far too small for a reliability claim. Nominal demonstrations and
repeated earlier versions are excluded from these denominators.

| Connector / method | Verified / valid | Wilson 95% interval | Sim / wall s | Independent force peak N |
|---|---:|---|---:|---:|
| SC / initial-pose guarded | 1 / 1 | 20.7–100% | 27.006 / 129.818 | 1.3925 |
| SC / physical-mouth vision | 0 / 1 | 0–79.3% | 5.282 / 26.722 | 0.7501 |
| SC / physical-mouth vision + recovery | 0 / 1 | 0–79.3% | 5.274 / 26.345 | 0.7337 |
| SFP / initial-pose guarded | 0 / 1 | 0–79.3% | 58.950 / 269.291 | 1.1307 |
| SFP / physical-mouth vision | 0 / 1 | 0–79.3% | 5.498 / 27.966 | 0.8630 |
| SFP / physical-mouth vision + recovery | 0 / 1 | 0–79.3% | 5.556 / 26.280 | 0.7420 |

The SFP baseline failed lateral alignment. All four visual variants rejected
physical-mouth perception before commanding insertion. These variants use
the existing analytic dark-opening detector around the nominal handoff plane;
they are not a rerun of the complete challenge policy's neural port estimator.
The recovery branch was never reached, so this pilot does not measure recovery
benefit or establish that recovery is ineffective.

SC accepted hits from left/center views disagreed by median 2.3785 mm and
2.1509 mm in the two runs (unchanged limit 0.6 mm). The right view failed blue
association. SFP accepted only the left view; center/right were low contrast.
Initial TF-buffer warmup failures did not account for the later sustained
rejections. Raw image and compact diagnostic exports accompany the summary.

Two additional SFP tactile prototypes on seed 3001 stopped on a newly added
0.2 Nm sensor-origin torque guard before collecting a centering probe. Their
policy peaks were 0.7762/0.8178 N and 0.2058/0.2798 Nm. They are negative
development trials, not evidence for successful tactile centering. A separate
force-guarded revision was executed. Full-pose probing stopped on the unchanged
0.08 Nm tip-axis moment guard (31.824 simulated / 149.330 wall s). Translation
probing completed travel but failed lateral and angular gates: terminal lateral
error 0.1662-0.1670 mm and full angle 0.10093-0.10104 degrees, against 0.10 mm
and 0.10 degrees. It took 74.600 simulated / 340.985 wall s. Corrected maximum
plug/target penetrations were below 0.003 mm; whole-scene maxima of 1.0198 and
1.9674 mm occurred at distal cable-link/NIC-mount contact. These prototypes
do not demonstrate reliable recovery. A subsequent SC attempt failed the setup
hold gate and is not a policy outcome.

Revision 4 then repeated development SFP seed 3001 with bounded lateral and
rotational controller-error compensation outside probing, a 0.5 mm lateral
probe command-lead cap and an eight-second terminal dwell. It passed depth,
lateral, penetration, contact-observation and dwell checks but **failed** full
angular alignment: maximum 0.101183 degrees against 0.100000 degrees. Terminal
depth was 45.72955-45.73034 mm and lateral error 0.05890-0.05924 mm; independent
force peak was 3.0530 N, elapsed time 80.592 simulated / 458.816 wall s.
Whole-scene penetration still reached 2.1566 mm at distal cable/NIC-mount
contact; requested plug/target penetration stayed at or below 0.001165 mm.
An evaluator-only tracking audit found the terminal lateral tracking error
changed from about 0.224 mm in revision 3 to at most 0.00223 mm in revision 4.
Exact-stamp own-tip versus simulated-tip comparisons gave a maximum positional
residual below 0.01065 mm in both runs. The rotation prior itself retained
approximately 0.10117 degrees of yaw error in revision 3; translation-only
registration does not observe that error. This is one repeated development
seed with several simultaneous control changes, not an isolated ablation or
new independent reliability sample. The angular gate was not relaxed.

The extended baseline matrix completed seeds 3002–3009. Combined with the
nonrepeated v5 baselines, SC passed 1/3 valid trials (Wilson 95% CI 6.15–79.23%)
and SFP passed 0/5 (0–43.45%). Two additional SC attempts were setup failures.
SC policy failures were one tracking stall and one force abort; SFP failures
were four lateral misalignments and one tracking stall. Maximum independent
force peaks were 11.2501 N for SC and 5.1596 N for SFP. The 8 N guard is an
abort threshold, not a demonstrated hard physical force bound: the SC impulse
crossed from about 0.35 N to 11.4 N in the policy stream at 15.368 simulated s,
and the independent FT peak occurred at that same stamp. The trace does not
support attributing this particular peak to subscriber delay or a post-abort
tail. This limits force-safety claims.

Two SC entries in manifest
revision 2 incorrectly requested a nonexistent second SC slot; those fail
before simulation and are not policy outcomes. Revision 3 corrects those two
indices only, retaining all seeds/gaps/perturbations. Final 100000-series cases
have not been executed.

Legal-index reruns subsequently executed SC seeds 3002 and 3006. Seed 3006
passed at target 3: terminal depth 15.5421-15.5697 mm, maximum lateral error
0.19351 mm, full angle 0.14417 degrees and terminal penetration 0.02356 mm.
Independent force peak was 4.9882 N; elapsed time 24.193 simulated / 139.610
wall s. Its 5.7918 mm distal-cable/NIC-mount penetration during motion remains
a substantial full-cable fidelity limitation despite a verified terminal seat.
Seed 3002 aborted on force, but all ten recorded pre-command contact packets
already showed excessive overlap (maximum 8.0014 mm at the remote SFP/board).
It is excluded from policy-comparison denominators by an explicit post-hoc
initial-contact screen, while its execution and 28.8658 N independent force
peak remain exported. The latter peak occurred after the policy's force stop;
it is not a demonstration of an 8 N physical bound.

The resulting **provisional screened development baseline** is SC 2/4 eligible
trials (Wilson 95% 15.00-85.00%) and SFP 0/5 (0-43.45%). SC also has two
pre-simulation invalid-slot failures and the one invalid-contact execution.
Screening the short legacy pre-command window is weaker than the new 0.5 s
full-contact hold gate; these counts are not final reliability. Seed 3002 still
requires a physically valid reset. Distal contact limitations remain explicit.

## Torsion observability and sensor-noise diagnosis

Source-box SAT checks at centered 5, 15 and 35 mm SFP depths found no collision
at 0, 0.5, 1.5 or 3 degrees of twist; overlap begins by 5 degrees. The original
1.5-degree twist probe therefore cannot establish opposed cage contact from
a centered pose. CAD is used as fixed design information, not per-trial truth.
Revision 5 used a slower 4.5-degree sweep, 3 Nms/rad twist damping and additional
settled-speed/opposed-moment checks. It stopped after 31.726 simulated /
171.764 wall s on the unchanged 0.08 Nm raw tip-axis moment guard, at -0.09664
Nm and 0.6021 N tared force. No requested plug/target touch was observed during
the short twist phase. This remains a failed development trial.

The source URDF specifies independent Gaussian force noise of 0.2 N and torque
noise of 0.05 Nm per axis. Propagating this through the sensor-to-tip wrench
shift predicts 0.051456 Nm tip-axis standard deviation. A zero-signal,
known-tare independent Gaussian model predicts 12.0% raw-sample exceedance of
0.08 Nm, or 99.83% probability of at least one exceedance in 50 samples.
An independent audit of 200 samples from four seconds of the revision-4
stationary-command dwell measured detrended sigma 0.049460 Nm and lag-one
correlation -0.2217. Eight of the 100 post-tare samples exceeded 0.08 Nm.
Thus a raw one-sample moment gate is inconsistent with this noisy sensor
model. These are analytic/source and recorded-sensor diagnostics, not hardware
calibration or proof of zero physical torque.

The noise-aware revision retains 0.08 Nm for a conservative bound on estimated
physical moment: 50 unique current samples, 100 stationary tare samples and a
99.9% Gaussian uncertainty margin, with variability never below the source
noise. The expected margin is 0.02933 Nm. This introduces a one-second window,
not a raw-sample bound; the instantaneous 8 N force abort and seating gates
remain unchanged. On the recorded diagnostic window it gives 0/51 stops.
That retrospective check is not an executed control or insertion result.

The revision-6 full-pose trial on seed 3001 subsequently reached real requested
cage contact near 3.8 degrees. It stopped at a conservative moment bound of
0.08310 Nm (estimated mean -0.05255 Nm, 50 unique samples spanning 0.98 s),
against 0.08000 Nm. It took 50.390 simulated / 251.069 wall s. During the twist
phase, 369 physical requested contact pairs were recorded with maximum
penetration 0.003227 mm. Thus the noise-aware observation progressed beyond
the earlier raw-noise stops, but no full-pose seat was achieved.

The prototype continued loading the first wall instead of terminating that
probe. A retrospective contact cue from the same recorded controller/wrench
stream would trigger at 60.012 s, 0.218 s before the actual bound abort: angular
lag 0.2051 degrees, settled speed 0.00003186 rad/s, moment lower/upper bounds
0.005548 / 0.06420 Nm. This is a replay diagnostic, not an executed revision-7
outcome. Revision 7 reverses at an observed contact endpoint and requires
opposed moment signs and settled clipping before using their midpoint.

A second defect was observed after the revision-6 abort. The fresh hold command
dropped existing load feedforward: actual depth changed from 4.8952 mm at the
stop to 7.1718 mm after 0.5 s. Revision 7 preserves existing load support while
replacing the pose target with measured TCP pose; it does not retain the
pose-error probe preload. The moment bound, instantaneous force abort and
independent seat gates are not increased.

Revision 7 measured opposed twist limits of +3.67460 / -3.87716 degrees relative
to its orientation prior, with settled plateau speeds below 0.000026 rad/s and
opposed raw moment difference -0.06575 Nm. It recentered to approximately
0.00401 degrees true full angular error, but stopped at about 20.49 mm depth
on the unchanged 0.1 s wrench-freshness gate. It is a failed insertion, not a
full-pose success. Elapsed time was 115.064 simulated / 578.763 wall s;
independent force peak was 1.2312 N. The independent recorder received FT
samples every 0.02 s around this stop, indicating a delivery/callback problem
in the policy path rather than an observed sensor publication outage.

Revision 8 adds bounded callback draining and a fixed-pose telemetry hold.
Trajectory progress pauses during that hold and motion resumes only with
wrench age <=0.1 s; the age limit was not raised. Its repeat stopped earlier
on the conservative moment bound: 0.08054 Nm versus 0.08000 Nm, at 49.936
simulated / 265.817 wall s. Independent force peak was 0.8889 N. This repeat
does not validate telemetry recovery or robust torsional registration.
The load-support abort correction did prevent the previous large depth drop:
at the moment stop the measured depth was 4.8944 mm, and after 0.5 s it was
4.8751 mm, compared with 4.8952 to 7.1718 mm for revision 6. These are repeated
development diagnostics with different controller startup histories.

The revision-5 translation matrix then executed SFP seed 3003 and attempted
SC seed 3004. SFP passed lateral (maximum 0.06148 mm), full angle (0.05686
degrees), penetration and contact-observation checks, but failed depth at
45.41835-45.41916 mm against 45.8 +/- 0.25 mm. Own-controller axial tracking
error was approximately 0.38044 mm, while exact-stamp prescribed-grasp position
residual stayed below 0.01070 mm. Independent force peak was 6.1917 N; elapsed
time 83.928 simulated / 686.735 wall s. SC failed the setup hold with 8.0899 mm
TCP error and 8.9246 mm terminal full-scene penetration, so no policy executed.
These are a failed policy trial and a separate setup failure, respectively.
Revision 6 added terminal-only axial load compensation bounded
to 0.25 N (500 N/m times the existing 0.5 mm lead bound), enabled below 1 N.

Revision 6 translation on the same SFP development seed 3003 **passed** the
independent held-clearance seat check. Terminal depth was 45.71906-45.72022 mm,
maximum lateral error 0.05758 mm, full angle 0.05570 degrees and global
penetration 0.000418 mm. Requested plug/target penetration over the trial was
0.001674 mm. Independent force peak / P95 were 6.3950 / 0.8588 N (policy peak
6.6410 N); elapsed time was 83.924 simulated / 452.726 wall s. This establishes
one successful perturbed held geometric seat under the declared observation
contract, not a randomized reliability or isolated causal effect. Revision 5
and 6 had the identical source-world hash but different controller bootstrap
step counts; the faster wall time cannot be attributed solely to control.

Whole-scene penetration still reached 1.8890 mm at distal cable/context contact.
Consequently this trial **fails** the additional recorded whole-trial <=0.1 mm
penetration screen despite passing terminal seating. The two accepted nominal
demonstrations pass that screen; the SC development geometric successes have
4.2583 and 5.7918 mm maxima and do not. These distinctions preclude a claim of
physically faithful full-cable randomized insertion. The export records both
terminal success and this additional sampled full-trial screen.

A setup-only SC free-chain reorientation was also tested on seed 3002. Offline
checks preserved all source joints, joint-anchor positions, inertias, meshes,
collision definitions and held-tip pose. Live simulation nevertheless failed
the contact hold: remote SFP/board penetration reached 10.8637 mm while TCP
error was only 0.000753 mm. No insertion policy ran. This initialization is
an explicitly failed diagnostic option, not a validated default or a physics
fix. The robot-only hold would have missed this defect.

## Numerical development diagnostic

The fixed development study in `experiments/configs/numerical_development_study.json`
uses seed 3003 and the same frozen translation policy in three contexts:
engine default / 1 ms, 200 solver iterations / 1 ms, and 200 iterations /
0.5 ms. Changed contexts require separately executed, matching SFP collision
controls. No held-out case is selected and repeats are not pooled as new n.
An isolated probe of the exact installed engine returned default 50 iterations
and read back 200 after setting 200. The first live 200-iteration control also
passed the full handoff gate with the Gazebo component reporting 200.
That deliberate lateral-miss control qualified with 1,192 physical contacts
among 1,213 raw cached pairs, and independent force peak 8.0949 N. It is not
insertion success. This verifies configuration propagation.

The three completed seed-3003 translation runs passed the unchanged terminal
geometric-seat gates (the invalid pre-command state in one run is disclosed below):

| Solver / timestep | Terminal depth mm | Max lateral mm / angle deg | Whole-trial penetration mm | Independent peak force N | Sim / wall s |
|---|---|---|---:|---:|---|
| Default 50 / 1 ms | 45.71903–45.72004 | 0.05755 / 0.05563 | 1.85771 | 6.18916 | 83.920 / 436.203 |
| 200 / 1 ms | 45.76238–45.76427 | 0.05804 / 0.05479 | 1.89160 | 4.91179 | 83.924 / 466.568 |
| 200 / 0.5 ms | 45.64766–45.64989 | 0.06367 / 0.05556 | 1.89232 | 2.45720 | 83.924 / 726.171 |

All three fail the separate whole-trial 0.1 mm penetration screen. Increasing
iterations or reducing the timestep did not resolve distal-cable contact. These are single
repeated-seed development runs; activation differs between runs, so force and
wall-time differences do not establish a causal improvement.

The same post-hoc pre-command contact screen used for older runs also found
an invalid initial state in the 200 / 1 ms insertion: nine of ten final
pre-command packets exceeded 0.1 mm, with a maximum 0.89005 mm overlap between
`cable_1::link_3` and `nic_card_mount_0::nic_card_mount_link`. Its 0.5 s setup
contact hold had passed earlier. Recorded contacts were small at 9.521 s but
first exceeded the screen at 9.911 s, before the first policy command at
9.992 s. This run is excluded from policy comparisons; its terminal seat and
force measurements remain recorded. The default / 1 ms repeat passed the
short pre-command screen. A prior hold alone does not certify contact
integrity through the later policy-start interval.

The first 0.5 ms collision control failed at controller startup; its policy
trial was not executed. Logs showed a 10 s ROS controller-switch request
expiring during sensor initialization. The retry uses the existing outer
70 s startup budget for that request as well; insertion thresholds are
unchanged. Retry `num26v3dt05` qualified its control with 1,242 physical / 1,256
raw cached pairs and independent force peak 8.24935 N, then started the
matching insertion. That insertion completed and passed the short pre-command
screen (maximum 0.001555 mm); its terminal seat and 1.89232 mm whole-trial
penetration are in the table. The numerical study is complete, with failed
startup attempts preserved separately.

An additional read-only SDFormat-15 load of the actual generated world
confirmed that all 21 cable chain/end segments retain source mass 0 kg and
diagonal rotational inertia 0.001 kg m². The parser reports these mass
matrices valid; it does not replace them with positive masses. This is a
resolved-file diagnostic, not a runtime engine mass measurement or proof
that these parameters cause the observed penetration. No inertia was changed.
The readback, compiler source and stderr are retained under
`.artifacts/sdf_inertial_readback*`.

A read-only late observer captured cable poses and contacts from 49.390 to
94.140 simulated seconds of the 0.5 ms run. The separate PosePublisher stream
is offset from contact timestamps by half a physics step, so an exact-stamp
geometry audit correctly refused to compare them. This capture does not
independently validate geometric penetration. Subsequent trials record 21
cable chain/end-link world poses, covering all 20 source capsules, within the contact observer's same PostUpdate
callback; the observer remains read-only. The independent capsule/box distance
calculation has three mathematical regression tests, including degenerate,
oblique and rotated cases. Total research regressions: **37 passed**.

## Matched registration ablation and precision investigation

`matched26v1` prescribed five existing SFP development seeds (3001, 3003, 3005,
3007, 3009) across all five NICs, comparing lateral registration with its
omission. Both variants share the CAD orientation prior, load support, own-TCP
tracking compensation, input guards, approach waypoint and terminal dwell.
Only opposed lateral probing/recentering is omitted. This compares the
registration procedure, including its motion and duration; it does not
separate the information benefit from the probing motion. Method order is
shuffled with fixed seed 934. The frozen configuration is
`experiments/configs/matched_registration_development.json`.

Three stages completed before an isolated collision-query precision failure
was reproduced. The batch was then stopped at the next trial boundary;
the fourth setup was preserved before policy execution and is not a policy
outcome. Pre-command contact integrity and
terminal seating remain separate from whole-trial penetration. Reused
development seeds and differing controller activation histories preclude a
held-out reliability claim. Progress is in
`.artifacts/matched_registration/matched26v1/progress.json`; the three executed
results and screens are in `partial_03{,_screened}.json`. The wrapper's
nonzero exit records the intentional researcher stop, explained in
`stop_request.json`. No held-out final trial has run.

The first completed matched run (SFP3001, registration omitted) passed the
short pre-command screen but failed terminal lateral alignment: depth
45.78461–45.78549 mm, lateral 0.25455–0.25510 mm and full angle
0.10030–0.10033 deg. Independent force peak/P95 was 4.49982/0.69449 N;
elapsed time was 68.096 simulated / 373.320 wall seconds. Whole-trial
manifold penetration reached 2.20090 mm.

The independent geometry audit matched 7,049 contact-observer timestamps,
checking every source capsule against 187 static NIC/board source boxes,
including pairs absent from contact output. At 33.281 s, the centerline of
`cable_0::link_3` entered a NIC-mount-0 collision box, proving at least a
2 mm solid overlap from the capsule radius. A separate analytic slab
intersection calculation found an explicit interior centerline point, without
using manifold depths. The engine manifold maximum within that body scope
was only 0.000896 mm at that instant, though it later reached 2.20090 mm.
The geometry and manifold depths are not interchangeable. This is an actual
source-collision intersection diagnostic, not evidence of an effective
registration method.

Its paired registration run also failed: lateral error fell to a maximum
0.06078 mm, but full angle remained 0.10070 deg against the unchanged
0.10000 deg limit. Terminal depth was 45.78851–45.78934 mm; independent force
peak was 1.63914 N, with 80.558 simulated / 389.279 wall seconds. Whole-trial
manifold penetration reached 2.13832 mm; its independent 8,322-match geometry
audit likewise found a source capsule centerline entering a NIC-mount box.
Both runs passed the short pre-command screen. One paired development case
shows reduced lateral error, not successful recovery or a general effect.

SFP3003 without registration passed terminal seating at 45.72640–45.72841 mm,
maximum lateral 0.05396 mm and full angle 0.05511 deg; independent peak force
was 3.84198 N and elapsed time 71.424 simulated / 397.492 wall seconds.
Whole-trial penetration was 1.89095 mm. Thus this near-aligned development
case can seat with the shared prior and compensation alone; its earlier
registration success does not demonstrate a registration benefit.

### Isolated narrowphase precision counterexample

The actual first-run capsule/box pose at 33.281 s was reconstructed in a
standalone Bullet 3.24 query, using the same capsule dimensions, source box
half-sizes and relative transform. A separate interior sphere control reports
overlap. The default float32 capsule query instead reports **+1.94224 mm
separation**. With uniform scale factors 10/100/1000 and proportionally scaled
collision margins, normalized distances converge near **−1.97618 mm**. The
installed float64 Bullet libraries report **−1.9761848 mm at the original
meter scale** and agree across those scale factors. The probe changes no
Gazebo asset, policy input or seating threshold.

This is a reproducible collision-query scale/precision failure for this actual
geometry, not proof that every distal contact failure has that cause. The
version-pinned [GJK implementation](https://raw.githubusercontent.com/bulletphysics/bullet3/3.24/src/BulletCollision/NarrowPhaseCollision/btGjkPairDetector.cpp)
and [EPA implementation](https://raw.githubusercontent.com/bulletphysics/bullet3/3.24/src/BulletCollision/NarrowPhaseCollision/btGjkEpa2.cpp)
use precision-dependent numeric tolerances. No particular tolerance is yet
isolated as the cause. Raw probe code, transforms, outputs and hashes are in
`.artifacts/precision_diagnostic_20261026/`.

A separate GzPhysics 8.3 backend was built from the unmodified version-pinned
sources with `BT_USE_DOUBLE_PRECISION` and the installed float64 Bullet 3.24
libraries. Original binaries remain unchanged. The new backend passed engine
feature/solver readback, and `/proc` mapping plus binary hashes verified the
actual headless server loaded the selected backend and only matching Bullet
dependencies. A matching collision-control / nominal-SC / SFP3003-prior
qualification sequence completed under
`.artifacts/precision_qualification/double26v1/progress.json`. Its actual
outcomes are recorded below. New
collision proofs are matched by exact backend hash as well as geometry,
timestep and solver configuration. Research regressions: **38 passed**.

### Latest completed float64 insertion outcomes

The SFP collision control qualified with 878 physical / 1,289 raw pairs,
independent force peak 8.16347 N and whole-trial manifold penetration
0.005054 mm. The independent 2,347-sample capsule/static-box audit found zero
overlap; maximum observed interior chain-anchor separation was 0.001283 mm.

SFP3003 with the shared CAD prior and compensation, without registration,
passed terminal seating: depth 45.79881065–45.79881334 mm, lateral <=0.008020 mm,
full angle <=0.054344 deg, independent peak 1.19720 N, and 71.302 simulated /
416.228 wall seconds. Whole-trial manifold penetration was 0.035119 mm and
the pre-command contact screen passed. However, an independent 7,369-match
source-geometry audit found 0.162896 mm capsule overlap with a declared source
NIC collision box, above 0.1 mm. The audit uses sharp source boxes; Bullet's
rounded collision margins still require a separate comparison. Do not call
this full-scene physical validation. Twenty observed interior chain joints
had maximum anchor separation 0.001678 mm.

Nominal SC failed setup before policy execution: terminal TCP error 5.49261 mm
and peak setup penetration 9.27950 mm. The canonical outboard initialization
then allowed a policy trial, but insertion stopped at the unchanged force
guard: independent peak 8.02112 N, depth 2.13813–2.18718 mm versus required
15.64 mm, maximum lateral error 6.10057 mm, and whole-trial manifold penetration
0.037633 mm. Elapsed time was 14.364 simulated / 92.482 wall seconds. The
alignment/tracking cause still needs investigation. Result index:
`.artifacts/precision_qualification/sc_outboard26v1/index.json`.

Both jobs are complete. An accurately held SFP seat is demonstrated; SC on
this backend, whole-scene physical fidelity, held-out randomized reliability
and registration advantage remain unestablished. Resolve the source-box/margin
discrepancy and SC failure before final evaluation.

## Verified nominal SC insertion and contact controls

The prescribed-grasp initial-pose baseline executed SC insertion using controller
TCP/wrench, a fixed fixture-design transform and the protocol handoff gap. No
simulator cable/port pose or per-trial GT grasp was passed to the policy. The
independent recorder used actual simulated poses and physics contact components.
Automatic port latching was disabled; the connector remained held by the robot.

| Executed condition | Measured outcome |
|---|---|
| SC2, seed 1001, 10 mm gap, 2 ms physics step | Verified terminal geometric seat with requested-target contact |
| Final 1 s depth range | 15.5668–15.5674 mm; nominal source seat 15.6400 mm |
| Final 1 s lateral error | 0.0154–0.0168 mm |
| Final 1 s full angular error | 0.02265–0.02276 degrees |
| Final 1 s maximum global contact penetration | 0.001624 mm |
| Maximum requested plug/port penetration over trial | 0.0005234 mm |
| Maximum global penetration over trial | 0.001762 mm |
| Peak policy tared wrist force | 0.8917 N; abort limit 8 N |
| Independent recorded tared wrist-force peak | 0.9654 N |
| Motion plus dwell / policy wall time | 27.642 simulated s / 119.558 wall s |
| Contact observation | 2,865 heartbeat samples; all 87 monitored collision components observed; one active grasp joint |

This is one deterministic demonstration, not a randomized success-rate estimate.
The verifier establishes a held geometric seat; it does not establish electrical
mating, latch engagement or retention after releasing the gripper.

An intentional SFP 2 mm lateral-miss control at a 1 ms step produced 1,326
physical requested SFP-tip/NIC-cage contact pairs (2,135 raw cached pairs) and stopped on the force guard at
8.036 N. This is a collision positive control, **not insertion success**.
Its observed tip stayed at the entrance with approximately 1.97 mm lateral error.
The aligned SFP repeat at a 1 ms step also passed a held geometric clearance
seat check. Its final 1 s depth was 45.7959–45.7993 mm (source seat 45.8 mm),
lateral error 0.0316–0.0331 mm, full angular error 0.02037–0.02063 degrees and
maximum terminal global penetration 0.001116 mm. Independent tared wrist-force
peak was 1.9218 N (policy-sampled peak 1.7782 N). Motion plus dwell took 57.802
simulated seconds / 307.477 wall seconds. Maximum global penetration over the
whole trial was 0.06793 mm, outside the terminal dwell.

There was no terminal SFP target contact, as expected from the source geometry
at the nominal seat. The check required zero source-box overlap at the observed
terminal pose, live physics-contact observation and the matching deliberate
SFP lateral-contact control. It therefore establishes a held geometric
clearance seat, not electrical mating or mechanical latch retention.
An earlier 2 ms run was rejected under the superseded positive-depth
interpretation. That rejection does not establish excessive penetration or a
timestep effect. No isolated timestep-convergence result is claimed.

Raw traces, selected before/during/after camera frames, source-world and policy
hashes are exported under `experiments/evidence/insertion_development/`.

## Perception transfer diagnosis

The deployed camera-view builder mixed image-time pixels with latest wrist TF.
The synchronized audit measured about 2 mm camera displacement during reset.
It now queries TF at each exposure timestamp. Source field of view and camera
intrinsics agree; timestamped ROS optical TF and Gazebo camera poses agree
within 0.025 mm / 0.002 degrees during reset, and within 0.001 mm / 0.0001 degrees
at rest. Image dimensions, zero distortion, encoding and optical conventions
were checked. The selected regression suites passed 203 tests.

Stationary SC predictions still fail the unchanged 6 px reprojection and 3.5 mm
shape gates. The timing fix does not solve that detector-transfer problem.
An additional stationary epipolar audit found seven of eight recorded pad-6
keypoint correspondences with Sampson distances of 8.32–26.98 px (mean across
eight points 14.29 px). This check uses calibration and pixels, independently
of the plug cuboid. Direct rigid-pose optimization in pixel space reduced the
same detections' mean reprojection error from 20.2739 to 13.4789 px; the
diagnostic true tip error remained 14.85 mm and rotation error about 100.98
degrees. Neither swapping cuboid symmetries nor changing the rigid-fit
objective rescues this frame. The evidence localizes an inconsistent image
prediction/correspondence problem; the unavailable original training images
and labels prevent a proven attribution to a specific training defect.
The prescribed-grasp benchmark removes a newly estimated grasp as an input
requirement because the grasp is fixed by construction; the deployed visual
policy retains its strict checks and has no new fallback.

## Paper-ready perception text

We reanalysed stored SC plug-pose evaluation records for a fixed full-frame
detector and its native-resolution crop-refinement variant. Each condition
contained 135 synchronized multiview image groups from 45 source trial IDs
(three snapshots per trial). On the split used to select crop parameters,
median 3D tip-position error decreased from 0.4562 mm to 0.2633 mm, a 42.3%
relative reduction. The difference between medians was −0.1929 mm (paired
trial-cluster bootstrap 95% CI [−0.2516, −0.1443] mm; 10,000 resamples, seed 731).
The corresponding relative reduction interval was [34.0%, 51.4%]. These
intervals describe sampling variation conditional on the chosen method; they
do not correct for crop-parameter selection on this split.

For the selected crop setting, a separate stored held-out evaluation contained
135 groups from 45 different trial IDs, with median position error 0.2733 mm
and lateral P95 error 0.3291 mm. Every group in the three audited reports
produced a pose. A matched full-frame baseline on the held-out split was not
located, so the 42.3% reduction is not a held-out paired improvement claim.

These values were recomputed from per-group records and checked against their
stored summaries. This is a historical-record reanalysis, not a new detector
inference run. Disjoint evaluation trial IDs were checked; the original image
dataset and training split manifest were not available to independently verify
training leakage, scene independence, or checkpoint-to-report identity.

| Condition | Groups / trial clusters | Position median (mm) | Lateral P95 (mm) | Axis P95 (deg) | Group misses |
|---|---:|---:|---:|---:|---:|
| Full frame, tuning split | 135 / 45 | 0.4562 | 0.4121 | 1.993 | 0 / 135 |
| Crop 6, tuning split | 135 / 45 | 0.2633 | 0.3434 | 1.809 | 0 / 135 |
| Crop 6, held out | 135 / 45 | 0.2733 | 0.3291 | 1.883 | 0 / 135 |

Figure: `experiments/evidence/perception_reanalysis/perception_ecdf.svg` (also
PNG and a locally generated PDF). Reproduction exports hashes, all 405 group
records as CSV, audited summaries, and bootstrap settings. ECDFs are descriptive;
the resampling unit for the comparison is the source trial, not a snapshot.

## Runtime interpretation

Historical median inference time per image group was 14.17 ms full frame,
25.66 ms crop on the tuning split, and 26.10 ms crop held out. Stored maxima
were 24.25 s, 0.571 s, and 1.957 s respectively. This indicates startup/outlier
effects; it does not establish production cycle time or comparable cold-start
throughput. Per-group timings are not in the exported per-group records, so
the timing summaries cannot be independently recomputed or bootstrapped.
Measure warm/cold inference, observation synchronization, motion, simulated
time and wall time separately on the eventual evaluation host.

## Executed validation and simulator status

- Native WSL Ubuntu 24.04 Gazebo 8.15.0 completed a 100-iteration, server-only
  `shapes.sdf` smoke test with exit code 0. Logs include DART's cone/ellipsoid
  mesh-substitution warnings. This is not the AIC workcell or an insertion trial.
- New analysis/verifier regression suite: 20 tests passed. Synthetic traces
  reject shallow placement, excess depth, lateral/angular error, missing contact,
  timestamp resets/gaps, excessive penetration and automatic-fixture assistance.
  These tests validate software decisions, not the simulator's physical fidelity.
- The earlier Windows source suite had ROS-interface import failures. Rerunning
  the full selected suite in the real Kilted participant environment, including
  frozen SFP validation, produced **270 passed**. Research tests: **23 passed**.
- Three upstream WaveArm interface-smoke configurations were prepared. Actual
  these particular WaveArm engine configurations have not executed. Kilted is
  now working, and the real controller has run on the corrected last-inch scene.
  WaveArm is not an insertion baseline.
- Headless model-camera QA is implemented in `experiments/headless_assets.py`;
  the executed drop run saved 12 RGB frames per camera at 960×720 and 246 dynamic
  SC plug poses. The scene graph, source hashes, unit mesh scale, visual counts
  and unchanged collision definitions were checked for nine model instances.
  Camera images were manually inspected. The plug fell from 149.46 mm model
  height to 17.049 mm and settled at the board surface. The sampled impact
  penetration reached **1.394 mm** before resolving, so visually correct assets
  do not establish precision-contact fidelity. This single native Harmonic
  probe is not the Kilted evaluator, controller, or insertion validation.

## Findings that affect the paper

The corrected last-inch Gazebo environment now has executed evidence in
`experiments/evidence/lastinch/kilted/`. One SC reset passed its actual-controller
hold and independent tip-pose checks. A scripted 2 mm lateral miss generated
requested-port contact (126 pairs, peak tared force 0.775 N) and returned to its
handoff; zero gripper contact pairs were reported. The prescribed grasp excludes
gripper/ATI collision geometry, while connector and port collisions remain.
These are environment controls, not successful autonomous insertions.

The real policy then failed visual grasp priming: 10.07 px reprojection error
against the unchanged 6 px gate, with zero motion commands. This is a single
negative transfer diagnostic. Neither crop changes on one stored frame nor
simple cuboid correspondence permutations resolved it. Do not infer that the
historical 0.27 mm held-out error transfers to this new simulator/viewpoint.
The cause requires image/label and camera-geometry analysis; it has not been
isolated experimentally.

The SC controller returns its outcome, but the outer `AicModel` action wrapper
reports an unconfirmed SC attempt as action success by default so Flowstate can
continue. An earlier research note missed this outer wrapper; this paragraph
corrects that note. Research explicitly sets `RL_INSERT_SC_MISS_ABORTS=1` and
`RL_INSERT_REPORT_MISS_AS_SUCCESS=0`. Both controllers still accept fresh scoring
events under the deployment contract; events may be for a different task target.
The pinned simulator's CablePlugin can make cable connections static after a
touch trigger. Independent requested-target seating validation is therefore
required, and post-latch stability cannot substantiate a physical-seating claim.

Source history changed cable-model collisions; it does not establish that the
SC Plug mesh or keypoint geometry changed. Camera inspection, mesh hashes and
collision/insertion-depth checks are tied to
`7148df9ab62b4b550b101189098f05bc0dba8444`.

## Claims presently supported

The stored SC records support improved geometric pose accuracy for crop
refinement on the tuning split and low held-out error for the chosen setting.
They support a practical lesson about judging the full pose pipeline rather
than detector scores alone. The historical records do not establish insertion
success. Separate current nominal SC/SFP demonstrations establish held
geometric seating under the declared prescribed-grasp condition. These
results do not establish final randomized reliability, recovery benefit,
hardware transfer, marker independence, cable pickup, electrical connectivity
or release retention.

## Missing validation, in priority order

1. Complete the seeded development matrix and implement a correction that
   improves verified seating under perturbations, using the same numeric gates.
2. Localize remaining neural transfer errors using original image/label data;
   calibration, exposure-time TF, correspondence and pixel-fit diagnostics are
   recorded, but the training cause is not proven.
3. Establish cable-contact fidelity across target locations and repeat matched
   timestep checks. The current verifier and latch-disable condition have
   executed; distal-cable penetration remains a measured limitation.
4. Freeze validated visual-only/full-recovery ablations; execute held-out
   paired insertion trials before claiming robustness or causal recovery gains.
5. Recover original perception images/split manifests and rerun inference;
   collect a held-out full-frame baseline and warm/cold timing on the same host.
6. Retain the collection's reported simulation-eligibility correspondence;
   complete SFP/generalization only if SC evidence and runtime justify it.

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

### Prospective seeded pilot checkpoint

Status: stopped_for_diagnosis. Results are development only; final held-out trials remain untouched.

- sc / force_tactile_no_registration: 2/3 valid full recorded successes; 3 attempts.
- sc / force_tactile_translation: 3/3 valid full recorded successes; 3 attempts.
- sfp / force_tactile_no_registration: 1/2 valid full recorded successes; 2 attempts.
- sfp / force_tactile_translation: 2/2 valid full recorded successes; 2 attempts.

Full measurements, confidence intervals, failure causes and source hashes: `.artifacts/seeded_pilot27/progress.json`.

Earlier measured V20 snapshot: 3/3 completed cases passed; this includes two known repairs and one new unknown-grasp case, with eight more development cases pending at this snapshot. The new grasp case held 15.565818-15.565950 mm depth, 0.163368-0.165001 mm lateral error and 0.135021-0.135181 degrees full angle. Raw peak 6.190797 N; complete 2000 Hz coverage, no sample over 8 N. Independent setup diagnostics measured a 2.939885 mm /2.546762 degree grasp difference while the authored initial physical tip pose remained identical. These setup diagnostics are never policy inputs. Actual video was decoded and terminal image inspected. No broad success rate or final confidence interval is inferred from these development cases.
