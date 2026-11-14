# RA-P experiment plan

Latest November 13: SC image/CAD reuse physically verified on dev_42000. Its
action is 26.1890 versus 27.6735 sim s, raw peak 3.060299 N with full coverage.
The SC and SFP speed pairs are complete but each contains only one reused case.
The finite next test is runtime47v5 bounded SFP lip search, which has synthetic
controls only so far. Repetition and a new freeze/cohort are required before
claiming the selected optimizations are robust. Final44 remains unchanged.

November 13 measured-speed update: SFP runtime47v3 physically verified on old
dev_45005, with complete raw-force coverage. Side-wall stage 16.8725 -> 11.8295
sim s; complete action 56.1585 -> 51.9735 sim s; raw peak 4.684120 N. This is one
development pair, with no randomized or wall-time improvement claim. The finite
next sequence is SC cache qualification runtime47v4, then bounded faster lip
search runtime47v5 if its prior checks pass. Actual source, results, negative
controls and evidence: `experiments/efficiency_followup48/README.md`.
Final44 remains unchanged and runs between serialized development slots.

Later November 13: both complete GPU runtime qualification cases passed strict
physical verification. Serialized development slots preserve final44 policy,
source hashes, manifest order and all outcomes, with a watchdog that resumes
the owned coordinator after cleanup. The active finite sequence is SFP
measured-start registration (`runtime47v3`), then SC immutable-CAD/image reuse
(`runtime47v4`) only if prior checks pass. Neither candidate has a physical
success assumed in advance. Broader angular recovery remains unqualified.
Measured data, cohort boundaries and live continuation paths:
`experiments/efficiency_followup48/README.md`.

November 13 parallel development: six paired CPU/GPU image replays and a
four-detector runtime integration check completed without changing final44.
A bounded-thread runtime candidate is installed only in the copied workspace.
Short angular tests remain below qualification: SC joint-fit median error
0.5141 deg, SFP 1.6494 deg; do not command these rotations. The uniform-channel
two-depth contact-angle formula is biased on the actual SFP CAD and requires
geometry and uncertainty modeling. Actual physical speed/angular qualification
is deferred while final44 owns the recording resources: concurrent full-scene
rendering was stopped after a final force-coverage gap. Next physical work must
use new development labels and the unchanged independent checks, then a new
freeze/cohort for any selected revision. Details and measurements:
`experiments/parallel_dev46/README.md`. The older speed43 profile stays stopped.

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

Full study is launched through `experiments/run_full_evaluation44.py`, with
status in `experiments/full_evaluation44/progress.json`. Before opening final
cases, the job must pass a coverage-only recorder regression, complete the
six-case speed selection, requalify contact controls and exercise both ablation
interfaces. Current negative-control and verification checks: 78 passed, one
missing historical raw-replay artifact deliberately deselected. Planned final
allocation remains 50 SC +50 SFP full-policy trials and 80 paired ablations;
no outcome-dependent stopping, retuning or case replacement is permitted.
Earlier checkpoint notes below remain historical.

November 12 active plan: finish the frozen 17-case historical replay using
new recovery20261112 checkpoints, then run the queued matched speed43 screen
(three SC and three SFP cases). Require unchanged independent verification,
complete 2000 Hz force coverage and <=20 N measured resultant peak. Raising
controller authority does not establish a speed benefit or a guaranteed force
bound. The recovered SFP stroke itself took only 6.96 of 57.458 simulated
action seconds and had no observed 3 N travel pauses; approach/probe motion
and simulation throughput are the main optimization targets.

After development selection, freeze code/weights and qualify controls before
opening the proposed 50 SC +50 SFP final cases. Balanced mount/slot coverage,
perturbations, denominators, timings and force statistics are prespecified in
`experiments/configs/recovered_robustness43_protocol.json`. A further 80 paired
ablation trials require implementation and development qualification first.
No held-out tuning or claims of full-task/hardware robustness are permitted.
The baseline replay and speed screen run serially in the owned evaluator;
unrelated WSL work stays untouched. Older plan snapshots follow.

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

November 3 current state: V27 completed SC 1/1, SFP 1/3. V28 axial waypoint
repair cleared both waypoints but completed 0/2 SFP insertions: one action
deadline, one unobservable first lip. Its second result was recovered with
matching frozen hashes and complete raw force coverage after the host worker
exited. V29 now physically tests the shorter contact-referenced clearance
path on `dev_45002`; keep its sources frozen. A separate bounded unload/reload
candidate has five passing ideal tests and remains undeployed. Both connectors
remain required, no final cases are opened, and paper readiness is false.
Detailed actual measurements and recovery limitations are in `PAPER_RESULTS.md`.

The older dated checkpoints below are historical, not active queue instructions.

Current development status (November 2, late evening America/Chicago): V25
finished SC 10/11 and SFP 4/10. Separate V26 SFP `dev_45000` passed; V27 SC
`dev_42000` passed, but SFP `dev_45002` still timed out at its initial waypoint.
The frozen V27 queue continues on `dev_45001`, then `dev_45003`; do not edit
its sources or launch another simulator. The remaining error in `dev_45002`
is predominantly axial, beyond the scope of the lateral-only repair.
Current measurements: `PAPER_RESULTS.md`; progress:
`.artifacts/repair35/regression35v1/progress.json`. Final cases remain closed.

Earlier 2026-11-02 repair checkpoint: see
[`paper/REPAIR_20261102.md`](paper/REPAIR_20261102.md). Both connectors remain
required. New SFP camera/contact datum control is physically under test; a
wrist compression sign correction shared with SC requires fresh regression
before any new source-frozen development/final comparison. No older final
worker may resume against changed sources or failed qualification. Final
seeds remain unopened. The SFP ten-case development manifest in
`experiments/configs/sfp_contact34_development.json` was subsequently executed
as V25; retain its failures and original source freeze.

## Required paper scope - user clarification, 2026-11-01

The intended paper covers **both SC and SFP last-inch insertion** using the
original learned perception pipeline and evaluated improvements. The current
SC-focused repair campaign is one workstream, not a reduction of the paper's
scope. Completing SC alone does not complete the requested combined evaluation.

SFP must receive its own source-model/pose/contact audit, original-checkpoint
perception diagnosis, deterministic insertion qualification, and frozen seeded
evaluation with relevant baselines and ablations. Reuse shared components where
appropriate, but derive and verify connector-specific geometry, insertion axes,
seat criteria and force behavior rather than copying SC settings. Earlier
prescribed-grasp/shared-prior SFP demonstrations remain separate evidence; they
do not validate the repaired learned pipeline. Report connector-specific rates,
forces, times and failure causes. A claim of transfer across SC and SFP remains
a hypothesis until both evaluations support it. Preserve the active frozen SC
study; do not change its sources or relabel its results to add SFP.

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


Status: 2026-10-26. Nominal held SC and SFP geometric seats have passed independent
verification under a prescribed grasp, including corrected signed-distance
contact reanalysis. Small seeded development batches have completed; final
insertion experiments have **not** run. See EXPERIMENTS.md and PAPER_RESULTS.md
for executed evidence and claim boundaries.

Scope is now explicitly **last-inch insertion only**: a held connector begins
5–15 mm from its requested port. The scene contains the real UR5e/tool/cameras,
five dual-port NIC cards, five SC adapters, five cables and both staging-fixture
types. Pickup and long-range transport are excluded. Staging context is not a
validated pickup benchmark. See `experiments/evidence/lastinch/README.md`.

The frozen manifest has 30 pilot cases (15 targets at two gaps), 200 final cases
and 60 lateral robustness cases, paired across three methods: 870 planned
trials. Method adapters are not yet validated; these counts are not results.
All methods must share the same calibrated load support, force protection,
timeout, initial conditions and independent requested-target verifier.
The last-inch grasp is a prescribed rigid attachment: gripper/ATI visuals and
inertia are retained, while their collision geometry is excluded. All connector
and target collisions remain. Inactive cable rest shapes are fixed context.
Disclose these approximations; neither pickup, grasp stability nor gripper-board
collision avoidance is evaluated. Nominal SC/SFP and a deliberate contact miss
have passed environment checks. The initial valid randomized baselines passed
SC 2/4 eligible trials and SFP 0/5 after legal-index reruns and a disclosed
post-hoc initial-contact screen. One SC reset remains invalid from remote-end
penetration; the new hold gate checks full-scene contact before policy motion.
Contact recovery remains under development; no held-out final trial has run.

One perturbed SFP development case now passes terminal geometric seating with
bounded axial load compensation. All development geometric successes still
fail the additional recorded whole-trial 0.1 mm penetration screen from distal
cable/context contact. A fixed solver/timestep diagnostic with matching
collision controls completed: 200 iterations and 0.5 ms steps still produced
about 1.89 mm whole-trial penetration. A five-seed SFP registration ablation
was stopped after three trials when source-geometry witnesses and an isolated
Bullet query reproduced a float32 scale/precision failure. A same-source
float64 backend is now being qualified with unchanged geometry and gates.
The matched subset shares CAD prior, compensation and guards. It remains development work;
no final reliability claim follows from a successful terminal seat alone.
Full-pose tactile registration has measured and corrected yaw, but its trials
have stopped on moment/input gates before a verified seat. No general recovery
advantage is established. Future causal ablations must share the same CAD
orientation prior, load compensation and input handling; comparing to the old
initial-axis baseline alone cannot isolate contact registration's benefit.

## Paper question and contribution

Working title: **From Pixel Bias to Contact Recovery: Practical Lessons in
Precision Cable-Connector Insertion**.

Test whether insertion-oriented geometric perception, followed by bounded
visual/force recovery, improves verified seating rather than just detector
accuracy or challenge score. The most defensible contribution is the measured
failure-to-fix chain: common image-coordinate bias survives multiview fusion;
native-image crop refinement improves the insertion-frame estimate; contact
feedback addresses the remaining residual error. This is a candidate practical
contribution, not a claim that crops, PnP, triangulation, or force control are new.

Primary causal question: does recovery improve success at the **same** camera
calibration, initial errors, force ceiling, timeout, and grasp distribution?
Do not claim novelty from integrating vision and force alone; ContactDP,
Insert-One, FORGE, and earlier assembly literature already do related work.

## Four-page story

| Body budget | Content | Evidence |
|---|---|---|
| 0.65 page | Problem, bounded scope, related work, contributions | Precision assembly setting; no general-purpose cable handling claim |
| 1.10 pages | Shared tip/mouth frames, confidence-weighted geometry, crop pass, bounded recovery | One pipeline figure; concise state machine |
| 0.65 page | Simulator, observation boundary, task scope, independent verifier, protocol | Explicit distinction between held-plug insertion and Flowstate pickup/orchestration |
| 1.30 pages | Failure mechanisms and paired comparisons | One perception figure, one success/robustness figure, one compact time/force table |
| 0.30 page | Limitations and lessons | Simulation contact/latching, logo dependence, missing hardware |

RA-P allows 4 pages body + optional 1 page supplementary descriptions + 1 page
references. Deadline is December 1, 2026. Its public author page requires
real-world demonstration, whereas the supplied brief reports a September 30
collection-specific simulation allowance. Retain that correspondence and resolve
the scope of the exception before submission. No editorial acceptance is assumed.

## Evidence and negative results

1. Reanalyse original baseline, pad-6, and held-out SC JSON reports. Match by
   metadata identity, recompute summaries, hash inputs, report missing detections.
   Multiple snapshots of one simulated trial are clustered, not independent trials.
2. Historical pad-2/3 dropout and pad-12/16 no-op values appear in a narrative
   document; raw reports for those conditions are not currently located. Label them
   historical observations until rerun. Do not invent error bars.
3. Residual RL was retired because contact parity/transfer were unresolved. This
   is an engineering decision, not evidence that RL is worse. No RL comparison
   unless an equivalent checkpoint, training budget, and held-out protocol exist.
4. Action-return success, touch/proximity events, automatic static attachment,
   and physical seating are separate outcomes. Never pool them.

## Smallest strong experiment set

**E1 — perception mechanism (available records + later inference rerun).**
Full-frame versus native-resolution crop, same weights and matched images.
Primary endpoint: median 3D tip-position error; also lateral P95, axis/rotation
error, group miss rate, and inference latency. Use a paired bootstrap resampling
whole source trials, 10,000 replicates. Test split was used for crop tuning;
held-out `val` report is confirmatory only for the selected crop setting, with
no matched held-out baseline presently available. Original images, split
manifest, environment, and checkpoint provenance are required for an inference
replication. Reanalysis alone does not verify absence of training leakage.

**E2 — insertion reliability and recovery (primary new experiment).**
SC first; SFP only after geometry and perception validation pass. Compare:
(a) initial perception then nominal guarded straight insertion, (b) visual
feedback with recovery disabled, (c) full visual/force recovery. All conditions
retain the same hard force abort; removing recovery must not remove protection.
Do not implement ablations by renaming identical configurations.
10 independent pilot scenes/method/connector, then 100 held-out scenes per
principal condition if runtime permits (minimum 50, chosen before final runs).
Paired scene seeds and method order randomization; pilots and tuning cannot use
final seeds. 100 Bernoulli trials give roughly ±10 percentage points precision
near 50% success; smaller effects may need more trials.
Primary endpoint: independently verified seating within fixed budget.
Secondary: official success, disagreement rate, completion time including
timeouts, peak/95th-percentile force, force-ceiling violations, retries,
recovery frequency, and failure categories. Wilson 95% CIs per proportion;
paired success difference CIs and exact McNemar test, with scene as the unit.
Report successful-only cycle times alongside all-trial budgeted times.

**E3 — bounded robustness (conditional on E2).**
Matched lateral offsets 0/0.5/1/2 mm and angular errors 0/1/2/4 degrees, initially
one axis at a time; confirm against measured clearances before freezing ranges.
Select three severities in pilots, 50 scenes/severity for visual-only and full.
Separate camera calibration bias from injected handoff error. Add 0/50/100 ms
delay only if timestamps and camera synchronization are validated. Logo masking
is a distinct survey experiment: distinguish failure to find a board from
insertion failure after a valid handoff. Do not call the full system marker-free.

## Simulator and port acceptance gates

1. Freeze upstream phase_1 at `7148df9ab62b4b550b101189098f05bc0dba8444`.
   It contains updated SC collisions and differs materially from this checkout.
   Keep a separate worktree; no blind merge or mutable `latest` image in final runs.
2. Run Kilted evaluator with the upstream WaveArm policy, one short SC task.
   This validates interfaces/reset/logging; WaveArm is not an insertion baseline.
3. Use only participant cameras/calibration, joint/controller state, robot TF,
   and wrist wrench in the proposed policy. Ground-truth recorder is evaluator
   only, isolated from model communications. Keep ground_truth:=false.
4. Verify scene names, tip/mouth transform conventions, collision clearance,
   grasp pose, force sign/tare, timing, and contact force scaling. Geometry in
   old controller comments is not authoritative for revised phase_1 assets.
5. Existing policy is a last-inch skill requiring a close upstream handoff.
   Qualification starts with a gripper-attached cable; it is not a complete
   reproduction of Flowstate pickup/survey/pose storage. Port survey and approach
   explicitly or label experiments as insertion-only with a controlled handoff.
6. Inspect automatic latching: CablePlugin makes connectors static after its
   trigger. Stability after that operation cannot demonstrate physical seating.
   Record before latch; add an explicit research mode disabling latch, preserve
   the unmodified evaluator as a separate comparison. Validate wrong pose,
   near-port, shallow, jammed, excess-penetration, and stable-seated controls.
7. Success requires calibrated insertion depth and full-frame alignment, a
   contact-consistency check, bounded interpenetration, a continuous terminal
   dwell, and no automatic fixture during that dwell. Thresholds must come from
   the chosen asset revision. Missing contact/fixture observability is unverified.
8. Restart scene/processes per seed; persist immutable configs, logs, bags,
   simulator seed, model hashes, versions, hardware and commit/dirty status.
   Retries after infrastructure errors are new attempts and remain in the ledger.

## Runtime and stop rules

Pilot wall time determines batch size; record simulated time separately. At
30–120 s wall/trial, 3 methods × 2 connectors × 100 scenes needs 5–20 h plus
startup; with the existing 720 s deadline the worst case exceeds 120 h. Prefer
SC plus a strong causal comparison over an underpowered two-connector sweep.
Reject final runs until physical-verifier controls and reset reproducibility
pass. Freeze policy/config before held-out execution. Infrastructure errors are
reported separately, never silently dropped from denominators. Stop and inspect
force violations, invalid physics, missing observations, or repeated resets.

## Immediate priority

The executable subset is historical perception reanalysis, source-level policy
regressions, a native Gazebo smoke run, and preparation of a pinned Kilted
evaluation runner. Full contact experiments require a functioning Linux
container engine/Kilted environment, a verified handoff adapter, and instrumentation
for contact and automatic fixture state. Do not present prepared runs as results.
