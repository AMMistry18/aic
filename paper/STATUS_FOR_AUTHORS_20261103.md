# What works and what the paper can claim

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

Current outcome: V33 completed **one verified SFP insertion out of one selected
development case**, with the new reversible lip checks and faster travel.
Stable depth 45.829-45.867 mm, maximum lateral error 0.006745 mm, complete
2000 Hz peak 3.284503 N. Travel took 11.8535 simulation seconds versus V31's
18.4435 on the same selected case, a descriptive 35.7% reduction across
different revisions and camera captures. Entire attempt remained 75.2925
action sim /681.7796 wall seconds (wall includes configuration). Thus travel
improved; total cycle time remains a major limitation. Full measurement record:
`.artifacts/repair36/pulse36v1_summary.json`.

The unchanged V33 policy is now running a fixed regression queue on
`dev_45003`, `dev_45006`, `dev_45005`, chosen for previous lip, side-wall
noise and travel-deadline failures. These are development repairs, not held-out
robustness. The broad V25 screen remains SC 10/11 and SFP 4/10; do not pool
its rates with the newer selected successes. Draft methods and failure analysis
now; final reliability and improvement claims require frozen varied evaluation.

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

This is a development checkpoint, not a final reliability claim.

| Evidence | SC | SFP |
|---|---:|---:|
| Completed V25 development screen, one frozen implementation | 10/11 | 4/10 |
| V27 selected repair regressions, separate implementation | 1/1 | 1/3 |
| Other selected successful repair regression | — | V26: 1/1 |

Do not add these denominators: implementations and selected cases differ.
After V27, V28-V30 SFP repairs improved intermediate stages but did not complete
insertion. V31 remains under test at this checkpoint. Final held-out trials: zero.

## Plain description of the method

1. Three RGB cameras detect the connector and the requested port with the
   existing learned keypoint pipeline. Confidence weights combine observations;
   adapted SFP models address the Gazebo appearance change. Public CAD edges
   refine the visual estimate. Camera calibration is ordinary sensor calibration.
2. The robot starts with a rigidly held, approximately angularly aligned
   connector. Recovering arbitrary orientation or planning the whole grasp and
   arm motion is outside this last-inch study.
3. Contact measurements correct what vision cannot measure accurately enough.
   SC has a terminal shoulder that provides a repeatable depth reference. SFP
   uses two entrance-lip measurements and opposed side-wall measurements;
   the nominal SFP seat is a clearance region, not an invented hard backstop.
4. Compliant motion and bounded own-TCP tracking correction advance and hold
   the connector. A separate recorder checks actual geometry, contact health,
   force coverage and dwell. Simulator object/port truth is evaluation-only.

Confidence weighting is retained, but its causal benefit has not been isolated.
An earlier selected SC component experiment found 5/5 with body refinement and
tracking versus 0/5 when either was omitted. This is small, development-only,
SC-only evidence, not the required matched final SC/SFP comparison.

## What still fails

SFP can spend too long reaching precise free-space waypoints, reject a weak
or noisy contact measurement, or slow near the seat under changing cable load.
V30 reached an independently observed 44.078 mm maximum in its final window
against a 45.8 mm target, then timed out. V31 addresses a diagnosed terminal
controller deadlock; its outcome is not yet known.

The old lip reaction-rise gate is not sufficiently noise-qualified: an ideal
sensor-negative test with no physical reaction accepted 31/32 noise seeds.
This is not a simulator insertion failure rate. It prevents a strong claim that
this gate alone reliably identifies contact. Controlled excitation and actual
stationary sensor sampling must be qualified before the final evaluation.

## Speed and the new force allowance

The user permits a 20 N ceiling for new trials. Prior studies retain their
8 N criterion. A ceiling does not prescribe a 20 N push. The next protocol
should use bounded model-based damping compensation to move faster under
ordinary sub-newton loads, while retaining braking near the seat. Contact
measurement needs sufficient signal and unique sensor samples, not long
unproductive waiting. Independent full-rate measurement must still check peaks.

V25 successful SFP median duration was 72.209 simulation seconds and 696.263
wall seconds including setup. Simulation motion time and computer throughput
must be reported separately. No industrial cycle-time claim is supported.

## Four-page paper shape

- Problem and scope: existing learned perception leaves small but consequential
  errors during last-inch SC/SFP insertion.
- Method: assign corrections to visual or connector-specific contact evidence;
  bound tracking compensation; explicitly preserve uncertain components.
- Experiments: actual public geometry, calibrated RGB, real arm/controller
  physics, all target locations, seeded variation, frozen matched ablations,
  independently checked seating and full-rate forces.
- Results and limits: per-connector success intervals, simulation completion
  time, force peaks and failure categories. Explain failed strategies with
  measurements. Disclose upstream angular alignment and lack of hardware tests.

The candidate contribution is this tested allocation of measurements and
connector-specific relative datums. Confidence weighting, PI, CAD fitting and
vision/force fusion individually are established techniques. Novelty and
improvement remain contingent on prior-work comparison and final ablations.
Drafting the method and failure analysis can proceed now; combined robustness
claims require the new qualified, frozen evaluation.
