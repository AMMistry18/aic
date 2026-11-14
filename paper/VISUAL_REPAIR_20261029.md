# Visual correction development, 2026-10-29

No final 100000-series trial has been used. Current measured screens are
**2/5 for the longer-mouth guarded version and 5/5 for an explicitly separate
angular-screen ablation**, conditional on upstream alignment. Three earlier
same-setup nominal repeats also passed. Do not pool versions or count the
ablation as repaired angular perception. See PAPER_RESULTS.md for current
status; entries below are chronological development records. Arbitrary visual
rotation and the original SFP pipeline remain unqualified.

## What the successful control baseline assumes

`run_force_tactile_registration.py` uses a prescribed rigid grasp, controller
TCP and wrist wrench, a specified initial gap, and a fixture orientation prior.
It computes an approximate mouth from the initial tip plus that gap. It does
not read live simulator object/port poses, but these shared reset assumptions
provide substantial localization information. The six nominal successes and
the stopped small paired pilot must remain separate from learned-perception
results. Calibrated camera intrinsics are ordinary sensor inputs; robustness
to imperfect calibration has not been established.

## Newly executed physical development trials

| Run suffix | Change | Actual outcome |
|---|---|---|
| repair30v1 / reacquire | Transferred mouth model, original plug weights; simulation-clock, exposure-time grasp and SC wrist-frame fixes | Real contact; terminal depth 2.078–2.092 mm of 15.64 mm, lateral error 2.583–2.586 mm; outer 8 N pre-command guard fired. Policy peak 8.380 N, independent peak 8.322 N; 214.427 wall / 17.755 sim s. Failed. |
| repair30v2 / reacquire | Lower force demand, 1 mm axial lead, earlier stall recovery and denser 3 mm search | Policy force peak 2.686 N. Recovery drifted away and exceeded the wall deadline; 744.412 wall / 77.943 sim s including cleanup. Failed. A forward-depth-gain message outside the mouth falsely called the opening captured; external verification rejected it. |
| rim30v1 / rim | CAD front-rim refinement with three-view and leave-one-view-out checks; one seating attempt maximum | Two initial accepted fits, then disagreement/fit failures. Only 2/7 samples passed consensus; bounded retreat and remeasurement also failed. 348.686 wall / 34.228 sim s; 0.698 N policy peak. No insertion advance. Failed. |
| rim30v2 / rim | Recompute edge visibility after pose correction | Visual consensus passed. Actual terminal depth 3.138–3.149 mm, lateral error 0.392–0.474 mm, angle 0.449–0.466 deg. 238.045 wall / 24.904 sim s; policy peak 2.851 N. Failed after one seating attempt. |
| contact30v1 / rim + contact | Add bounded opposed contact registration | Visual consensus obtained only two valid samples and failed before registration; no contact correction was executed. |
| contact30v2 / rim + contact | Neural association once, then fresh-frame CAD tracking | 6/7 visual samples accepted in 69.318 wall s, versus 137.560 s for contact30v1. Opposed probes ran but one axis was unobservable, so no grasp correction was applied. Failed. |

Each archive under `.artifacts/lastinch_runs/` retains actual RGB, source
snapshots, setup gates, wrench/contact traces and independent verification.
These distinct development versions are not pooled into a success rate.

## Why the original visual recovery did not run

The last SC run before rim refinement obtained reference samples but rejected
all cameras as `weak_blue_side_reference`. Actual image overlays show both
misplaced measurement regions from the learned port bias and occlusion by the
held connector at a 10 mm initial gap. Removing the gripper mask did not repair
the gate. An evaluator-only exact-port counterfactual still failed two cameras;
after a real retreat the exact-port counterfactual passed two cameras. These
counterfactuals diagnose visibility and are never live control inputs.

The legacy dark-bore correction was also replayed: it failed multiview fusion
with the learned plane. Even with the evaluator-only true plane it retained
about 0.710 mm center error in one replay, so it was not enabled as a solution.

## Image-based front-rim repair

The new opt-in `original_perception_rim` retains learned target selection and
the original grasp estimator. It extracts 522 boundary samples on 22 segments
of the foremost blue CAD rim from the public asset, then fits these edges to
actual calibrated RGB images. No scene placement is read by the fitter. The
original board-normal prior is retained; position and yaw are refined. A free
six-axis fit was explored and did not consistently improve orientation.

The fit requires three views, sufficient visible edge support, mean edge
distance <=2 px, bounded correction, and <=0.25 mm maximum modeled-rim
disagreement for all leave-one-view-out fits. These agreement tests do not
constitute a calibrated uncertainty bound or independent insertion proof.

A fixed -0.5 px raster-to-array sampling correction is explicit and confined
to this rendered-edge experiment. An offline truth-based calibration audit
found approximately half-pixel shifts across the training/development scenes.
The OpenGL pixel-center convention supports investigating this conversion;
it does not by itself certify every Gazebo camera implementation. CameraInfo
and neural-pipeline intrinsics were not silently modified, and no per-camera
or per-trial truth-fitted offsets are deployed.

On six static development scenes, after iterating visibility at the corrected
pose, all six passed the fit checks. Evaluator-only mouth-position errors were
0.0441, 0.0489, 0.0944, 0.0456, 0.0532 and 0.1358 mm; maximum lateral error
0.0851 mm and maximum angle 0.1984 degrees. These scenes were used during
development and cannot support a held-out generalization or insertion claim.
The transferred mouth model's previous median position error was 1.636 mm.

The first live rim trial exposed a visibility-selection defect: masks were
selected at the coarse neural pose and retained after refinement. The next
revision recomputes visibility after fitting while retaining the same quality
and agreement limits. It also archives raw images, exposure stamps, K and
camera transforms for every attempted fit. Its physical trial is pending.

Selected controller/camera/wrench/reacquisition/rim rejection regression:
140 passed before launching that next revision.

Subsequent tracking keeps the original neural association and seed reprojection
gate, requires new exposures, and rejects >0.25 mm temporal modeled-rim motion
for the static fixture. It avoids repeating the expensive detector for each
consensus sample. After the first probe trial, registration was strengthened
with stationary plateaus, unique wrench timestamps, a three-standard-error
force contrast check, lower translational damping, and a bounded 0.5 mm axial
preload. A delayed free-tracking regression explicitly rejects apparent wall
contact. Selected tests now pass 147 cases. The next physical probe revision
is still pending; no contact-correction benefit has been established.

The six-scene static replay's median mouth error changes from 1.636 to 0.0511 mm,
while median relative lateral error changes from 1.893 to 0.640 mm. One relative
error worsens because the former plug and mouth biases partly canceled. This
is evidence that independent object accuracy is insufficient; the remaining
grasp estimate matters. Actual measurements and a standalone PNG/SVG plot are
in `experiments/evidence/original_repair/static_rim/`.

Two further offline plug prototypes were rejected. Front ferrule-edge fitting
gave 1.62–2.43 mm position and 4.38–8.77 deg angle errors; projected ferrule
boundaries are not reliably visible in the actual images. A gripper-masked
white silhouette improved position in most scenes but retained 0.357–5.257 deg
orientation errors. Neither is enabled in a physical policy. Low image error
alone did not establish an accurate grasp.

## Novelty and next evidence

Visual-force control, active probing and pose-belief fusion already have close
precedent in [Insert-One](https://www.merl.com/publications/TR2024-137),
[FORGE](https://arxiv.org/abs/2408.04587), and
[BayesContact](https://arxiv.org/abs/2607.16123). No first-of-kind claim follows.
The candidate practical contribution remains a measured mechanism that repairs
biased learned relative geometry using connector clearance and observable
visual/contact evidence. It needs a working complete insertion and matched
ablations before any effectiveness claim. The next physical trial isolates
the rim repair with one attempt; plug-grasp error and contact behavior must be
measured independently if port localization passes.


## Dual-rate force and contact diagnosis, later 2026-10-29

Combined image observations supplied unique wrist samples at only about 5 Hz.
A separate `/fts_broadcaster/wrench` subscription now merges fresh raw force into
the policy observation without replacing camera pixels, headers or transforms.
`contact30v5` measured 49.978 Hz (2244 unique messages over 44.88 simulation s),
with maximum policy sample age 0.0205 s. Stale, future, frame-mismatched and
out-of-order data are rejected. This is an implementation correction, not a
new general sensor-fusion algorithm.

| Physical development run | Wall / sim s | Peak policy N | Actual terminal depth mm | Actual lateral mm | Actual angle deg | Outcome |
|---|---:|---:|---:|---:|---:|---|
| contact30v3 | 270.790 / 28.117 | 2.042 | 2.0754-2.0811 | 3.2551-3.2876 | 0.5530-0.5537 | Failed |
| contact30v4 | 258.074 / 27.826 | 2.715 | 2.0738-2.0770 | 1.4411-1.4418 | 0.5550-0.5734 | Failed |
| contact30v5 | 433.881 / 43.366 | 3.269 | 15.4697-15.4698 | 0.2569-0.2571 | 0.5205-0.5211 | Failed |
| handoff30v1 | 330.582 / 34.692 | 2.027 | 1.7561-2.1684 | 2.3939-2.8711 | 0.0761-0.0964 | Failed |

The first two runs lacked enough fresh force observations for calibration.
`contact30v5` passed then-current plateau/force tests and reached the depth
acceptance band, but the independent lateral and angular gates failed. No
thresholds were changed. Its 0.213 mm axis-0 span is insufficient under the
subsequently tightened observability screen; acceptance should not be treated
as proof of correct two-wall registration.

`handoff30v1` is a distinct condition preserving the initial robot wrist
orientation rather than using uncertain learned-grasp rotation to align it.
It assumes upstream angular alignment, retains learned grasp translation and
visual port localization, and does not use prescribed grasp or live object
truth. Angle stayed within tolerance, but entry failed. Axis-0 probe endpoints
were 1.170 and 1.238 mm from their reference (only 0.068 mm separation), yet the
old screen accepted them and changed grasp translation by 2.827 mm. Force
contrast can occur under entrapment and does not establish bore centering.
The Gazebo contact adapter resolves arbitrary collision names within a link;
these logs establish contact with the requested port link, not which internal
wall was touched. Do not claim identical named collision means identical wall.

The revised necessary screen rejects probe spans <=0.25 mm, equal to the final
lateral acceptance tolerance. This is a conservative observability requirement,
not a geometric theorem or a relaxed accuracy threshold. A new bounded spiral
seeks 2 mm of robot-observed forward entry under <=0.5 mm tip lead, <=3 mm
radius and 6 N internal force guard, before attempting two-axis calibration.
It fixes another defect: the legacy fixed 0.8 mm press could not itself prove
a required 1.5 mm entry gain. Search uses fresh robot kinematics and force;
its progress message remains a candidate, never an insertion verdict.

Revision 8 tests: 157 passed, including entrapped-span rejection, hidden-opening
search, no-opening exhaustion, force abort and <=0.5 mm command-lead checks.
`capture30v1` is prospective development; no success is assumed before its
independent verifier finishes. Final seeds remain unused.


## Bounded search and native feedback ablation

`capture30v1` completed with a safe failure: 501.845 wall /56.6565 simulation s,
3.760 N peak whole-policy force and 1.126 micrometres maximum global penetration.
The entry search itself took 40.0215 simulation s, peaked at 1.360 N, and gained
-0.0042 mm axially (required +2 mm). No grasp correction was applied.
Actual terminal depth was 2.1596-2.1600 mm, lateral 1.1252-1.2882 mm and angle
0.0923-0.0942 deg. Angle passed; depth and lateral failed.

This was not a completed sweep of the full 3 mm search disk. The tracking-aware
path advance stalled: measured X excursion was only -0.184 to 0 mm and Y only
-0.035 to +0.003 mm. The robot could not track even the early path closely
enough to continue. Do not claim a 3 mm recovery basin was physically searched.

The independent rigid-attachment tracking audit reconstructs actual TCP from
recorded tip truth and the reset attachment strictly after the trial. It never
feeds that information into control. For actual depth >0, before registration,
median/max lateral command-to-TCP gaps were 0.842/1.404 mm (`rim30v2`),
0.438/0.690 mm (`contact30v5`) and 1.500/1.613 mm (`handoff30v1`). These include
transients and contact-limited motion, so they identify tracking as a contributor,
not proof of a single causal mechanism. Data: `tracking_audit.json` in the
portable evidence directory; script `experiments/diagnose_original_tracking.py`.

`position30v1` prospectively disables the native controller's extra wrist-force
feedback (gain 0 instead of 0.5) while retaining original learned perception,
CAD rim refinement, force-aware SC seating/alignment and raw force guards.
The upstream-aligned orientation assumption is explicit. It has one seating
attempt, no contact calibration or search; its first attempt can be compared
with earlier aligned-handoff attempts but is not a randomized paired estimate.
No result is assumed while it is running. Revision 9: 163 targeted tests passed.
The original default gain remains 0.5 outside the opt-in ablation.


`position30v1` is now complete. The policy's estimated geometry passed its
1.0495 simulation-second candidate dwell, but independent verification correctly
rejected lateral misalignment. Actual terminal depth was 15.4427-15.4755 mm,
lateral 0.3252-0.3274 mm and angle 0.0793-0.0840 deg. Depth and angle passed;
lateral failed. Policy peak force was 1.231 N; global penetration 1.324
micrometres; 190.460 wall /19.224 simulation s. Its terminal lateral vector in
the true port frame was approximately [-0.3266,+0.0236] mm, calculated only
post hoc. No truth was supplied to control.

Post-hoc median/max lateral command-to-TCP gap was 0.222/0.519 mm, compared with
1.500/1.613 mm in the earlier aligned-handoff first attempt. This development
comparison supports investigating native gain, but is not a paired causal
estimate or an insertion success rate. Actual footage (explicit aligned-wrist
prior label): `experiments/evidence/original_repair/video/position30v1/attempt.mp4`.

`center30v1` prospectively adds opposed contact centering at a deep estimated
candidate (>=15.2 mm), with **zero additional axial preload**. This avoids assuming
that an early front-face jam is an internal bore bracket. It retains the
>=0.25 mm span screen, stationary plateaus, fresh wrench samples and force
confidence tests; rejection stops the attempt. The calibration adjusts only
visual-grasp translation. The final independent 0.25 mm/0.25 deg and depth
limits are unchanged. This condition still assumes an upstream-aligned wrist;
it does not solve arbitrary learned-grasp rotation. Revision 10: 166 targeted
tests passed. No improvement is assumed while the trial is running.


## Conditional image-based body translation repair

`center30v1` failed before the proposed deep-contact refinement: actual depth
3.1017-3.1616 mm, lateral 0.3813-0.4870 mm, angle 0.0737-0.0934 deg;
164.814 wall /16.940 simulation s, 1.463 N peak policy force. No contact
registration ran. Thus zero native gain alone is not a consistent insertion
solution; the repeated same setup remains sensitive to visual-grasp bias.

The earlier unconstrained white-body silhouette fit had unacceptable angular
errors and was rejected. A new **translation-only** fit uses the declared
upstream angular alignment and image-derived port rotation while preserving
physical starting wrist orientation and the learned grasp rotation. It uses
the public white SC body convex hull and tool/ATI self-masks rasterized from
public robot visual triangles, current K, and exposure-time robot/camera TF.
No known connector grasp, object segmentation or live object truth is used.
Fixed 6.55 mm finger-joint aperture is part of the prescribed-grasp scope.

On first-rim image sets from six completed **nominal** archives (same scene and
grasp; not six independent randomized environments), image-only fits were
persisted before truth scoring. Initial neural position error 0.512-0.546 mm
fell to 0.043-0.060 mm. Mean support residual was 0.164-0.190 px, and maximum
leave-one-view-out position disagreement 0.0303 mm. These are diagnostic
replays used during development. Runtime gates require three views, >=12
unoccluded support directions per view, <=1 px mean support residual,
<=0.25 mm leave-one-view-out disagreement, and <=3 mm position correction.
There is no fixed-grasp fallback after a rejected body fit.

A separate replay intentionally violates the angular prior using six existing
static development scenes with 2.01-3.99 deg true mismatch. The body fit accepted
four despite 0.328-0.501 mm position error; two were rejected by view agreement.
This demonstrates that image-fit agreement **does not certify the angular
handoff assumption**. Do not claim arbitrary angular-error recovery or calibrated
uncertainty from these gates. No simulator truth entered the fit; truth was
read afterward to expose this limitation. This experiment is a negative
control, not an insertion evaluation.

Production `body30v1` retains original neural plug weights and transferred mouth
weights, uses these detections as proposals, applies CAD mouth/body refinement,
keeps native wrench gain zero, and preserves the upstream-aligned wrist.
It has one seating attempt and no contact calibration/search. Raw body input
RGB, masks, K, camera TF, exposure-time own TCP TF and fit diagnostics are saved.
The public geometry template is packaged with provenance, not generated from a
trial's object pose. Revision 11: 171 targeted tests passed, including partial
occlusion/absent-body rejection, synthetic subpixel translation recovery, robot
mask frame invariance, and rejection of silent use outside the handoff mode.
The live outcome is pending; no success or robustness is assumed.


## Separating visual bias from controller tracking

`body30v1` passed all runtime image checks: 0.226 px mean support error,
24-33 accepted directions per view, <=0.0609 mm leave-one-view-out disagreement.
Actual body input images, current K, camera/base TF, own TCP TF at each exposure,
and public-tool masks are archived. Its fitted tip position at the recorded
19.8 s exposure had **0.04649 mm actual error**, evaluated after fitting.
This is a measured initial position result conditional on angular handoff,
not proof of full six-DoF accuracy or generalization.

The physical attempt still failed: depth 3.0818-3.1587 mm, lateral
0.3383-0.4739 mm, angle 0.0888-0.1126 deg; peak policy force 1.664 N,
169.786 wall /17.104 simulation s. Median/max lateral command-to-TCP gap was
0.375/0.621 mm. Accurate initial visual translation therefore did not guarantee
entry under this controller's residual tracking/load error.

Revision 12 prospectively adds a bounded lateral integral compensation term,
using only commanded/own measured TCP position and raw wrist-force guard.
Gain 200 N/(m s), force-vector norm <=0.5 N, no axial integration, dt <=0.05 s;
integration freezes at >=1 N contact force and during registration probes.
Existing calibrated load support, original SC force-aware logic and final
verification limits remain. Native wrench feedback is zero to avoid a second
uncontrolled force-feedback contribution. This borrows a conventional tracking
mechanism from the separately qualified diagnostic controller; it is **not a
novel PI algorithm**. Its effect on the learned pipeline still requires live
measurement and matched ablations.

The conditional body fit also gains a necessary orientation-consistency screen:
choose the port's closest 0/180-deg symmetry branch to learned plug orientation;
reject measured mismatch greater than atan(0.25/15.64)=0.9158 deg. The ratio uses
the lateral/depth budget; it does not certify true <=0.25 deg alignment. This
would reject all six earlier 2-4 deg prior-violation development examples based
on their learned orientation, without reading truth. It remains an empirically
chosen development safeguard and must be evaluated prospectively.

176 targeted tests passed, including unknown constant-load compensation,
contact/probe freeze, vector-force clipping, clock/input rejection and angular
symmetry handling. `track30v1` is the first prospective physical run with this
combination. No physical success is assumed before independent verification.


## Continued physical control repair, 2026-10-30

`track30v1` (integral-only lateral compensation) reached actual terminal depth
10.4638-10.4781 mm with 0.2134-0.2819 mm lateral error and 0.0685-0.0898 deg
angle error. It failed; 188.473 wall /19.387 simulation s, 2.421 N peak policy
force, 0.603 micrometres maximum global penetration. Peak added tracking force
was 0.0947 N. Only 4/328 tracking updates froze on contact, so the data do not
support blaming contact freeze for this failure.

The trace shows a lateral tracking-error reversal during the rapid advance
beyond the mouth region. At 28.25 s own TCP lateral error was approximately
[-0.236,+0.247] mm; at 29.52 s it was [+0.244,-0.001] mm. Integral X force had
to unwind from negative before correcting the new positive error. Axial stall
ended the attempt after the unchanged 1 s stall budget. This motivates testing
a prompt bounded proportional term, not changing acceptance tolerances.

A separate endpoint audit checks actual tip truth against own robot TF before
and after motion. The inferred rigid attachment changed only 0.520 micrometres
in `body30v1` and 0.0617 micrometres in `track30v1` (rotation changes <0.00003 deg).
These are two-exposure checks, not continuous slip certification, and never
feed control. The actual live finger joint state was 0.00655 m, matching the
public self-mask geometry. Script: `experiments/audit_rigid_attachment.py`.

Revision 13 adds proportional gain 400 N/m to integral gain 200 N/(m s), with
**combined added force norm still <=0.5 N**. Together with native 90 N/m seating
compliance this approximates the earlier diagnostic controller's 500 N/m
lateral response near zero error. Integral still freezes at >=1 N contact and
in probes; explicit probes omit the extra proportional term. No axial integral
or relaxation of the 1 s stall timeout, 6/8 N guards, or final geometry/contact
criteria was introduced. A load-reversal plant regression was added; 177
targeted tests passed. `track30v2` is prospective and its physical verdict is
pending. The tracking-only method is now available for the recorded four-way
body/tracking ablation; that protocol has not yet executed.


## First qualified conditional visual insertion, 2026-10-30

`track30v2`, revision 13, independently passed a stable requested-target seat.
Actual terminal depth 15.559144-15.559220 mm, lateral 0.095018-0.097273 mm,
angle 0.077391-0.078155 deg. All 101 final samples passed depth, lateral, angle,
penetration and contact-observation checks. The requested pair had 101 signed
contacts and 101 active solver-impulse points (maximum normal impulse
4.38185e-5 N s). Global maximum recorded penetration was 4.59690e-8 m.

Policy peak tared force was 2.27872 N, independently tared raw-wrench peak
2.30894 N and independent p95 0.81065 N. Added tracking force peaked at
0.11255 N, with P=400 N/m, I=200 N/(m s), norm bound 0.5 N. Native wrench gain
was zero. Total recorded duration was 181.5876 wall /19.9120 simulation s;
configuration and motion timing fields remain separately available. The
estimated candidate dwell took 1.9115 simulation s to obtain a continuous pass.
No scoring events or simulator object/target poses were policy inputs.

The body fit mean support error was 0.2280 px, maximum leave-one-view-out
position disagreement 0.0637 mm, learned-orientation/prior discrepancy 0.5924
deg. These consistency measures are not calibrated accuracy bounds. The method
requires an upstream-aligned wrist, fixed known finger aperture and rigid
attachment. It preserves learned grasp rotation and refines its translation
from pixels. The transferred mouth checkpoint is used; all original weight
files remain preserved. This is not a pass of the original weights alone or
of arbitrary six-DoF grasp-error recovery.

Footage: `experiments/evidence/original_repair/video/track30v2/attempt.mp4`.
Three actual cameras are shown, at 1x simulation playback using causal holds
between sparse captures. It was fully decoded and a terminal frame visually
inspected. Evaluator truth appears only in the video overlay and verifier.

At this checkpoint this was **1/1 nominal qualification**, not a statistical
reliability estimate. Later completed repeats and varied cases follow below.

## 2026-10-30: three unchanged nominal repeats; varied target exposes cable occlusion

Both preselected repeats passed the independent verifier. Across the three
nominal trials, actual terminal depth was 15.55874-15.55922 mm, lateral error
0.08318-0.11393 mm, and angular error 0.07739-0.08223 deg. Independent force
peaks were 2.30894, 1.54596 and 2.12351 N; wall times were 181.5876, 188.3498
and 182.2819 s. All 94 source/checkpoint hashes matched exactly.
Machine-readable details: `experiments/evidence/original_repair/nominal_repeatability.json`.
This is repeatability on seed 1001, not three independently randomized setups.

The prospective five-target, four-method factorial (`factor30v1`, method order
seed 20261029) then completed two target-0 trials. Both the tracking-only and
body-only variants rejected port perception, before insertion. Their recorded
wall times were 314.546 and 312.153 s. The batch was deliberately stopped
between completed trials; the other 18 planned trials are not observations.
All raw images, frozen code and failures remain archived. It must not be pooled
with later policy versions.

Actual centre-camera images show the orange cable crossing the blue SC port.
The rim edge extractor masked the white plug but retained the orange/blue
occlusion boundary. On the first two saved captures, the old mask produced
maximum leave-one-view-out disagreements of 1.869 and 2.073 mm, correctly
rejected by the unchanged 0.25 mm gate. Masking the existing orange cable HSV
class as well produced 0.108 and 0.091 mm disagreement and 0.373 px mean edge
residual in both captures. Truth was read only after fits/acceptance were saved:
diagnostic position errors were 0.0453 and 0.0409 mm. The replay script now
defines the historical white-only baseline explicitly, so later production
changes cannot silently replace the negative control.

Harness revision 14 applies this cable mask. All neural/CAD acceptance and
physical verification thresholds remain unchanged; 179 targeted tests passed,
including exclusion of a cable-created false edge and fail-closed rejection
when the cable hides the entire target. The prospective physical trial
`dev_3000_cable30v1_original_perception_rim_position_body_track_handoff` is
pending at this entry. Replay agreement is not an insertion result. Final
100000-series trials remain untouched.


## 2026-10-30: completed v14 five-target screen

The same 94 frozen source/checkpoint hashes were used on all five fixed seeded
development cases: **1/5 passed**. The successful case is a new target and
starting pose, separate from the three earlier nominal v13 repeats.

| Case / target | Verified | Failure stage | Wall / sim s | Independent peak N |
|---|---|---|---:|---:|
| dev_3000 / 0 | fail | learned angular-prior veto | 277.023 / 23.828 | 0.815 |
| dev_3002 / 1 | fail | port_perception_failed | 333.304 / 25.370 | 0.931 |
| dev_3004 / 2 | fail | port_perception_failed | 298.413 / 24.422 | 0.811 |
| dev_3006 / 3 | pass | stable seat + contact | 386.037 / 41.816 | 2.365 |
| dev_3008 / 4 | fail | port_perception_failed | 293.095 / 23.995 | 0.906 |

Target 3 held depth 15.53882-15.56600 mm, lateral error 0.07829-0.12896 mm,
and angle 0.11926-0.12310 deg. All independent terminal contact/geometry and
whole-trial penetration checks passed; maximum recorded penetration was
6.5190e-8 m. Policy peak force was 2.3338 N, independent peak 2.3650 N.
The image-only body translation fit's post-hoc initial error was 0.05083 mm;
its angular prior differed from actual tip orientation by 0.20709 deg.
The learned/prior mismatch was 0.90352 deg, below the unchanged 0.91578 deg
screen. No simulator target/object poses entered the policy.

Actual footage: `experiments/evidence/original_repair/video/target3_v14/attempt.mp4`.
All frames decoded; the terminal frame was visually inspected. The actual
connector, adapter, robot and cable are shown in recorded cameras. The ASCII
companion and provenance file document the aligned-handoff condition and hashes.

The other four cases stopped before insertion. The port fit improvement on
case 3000 therefore repairs one stage, not the whole varied-case policy.
This target-stratified development screen is not a final reliability estimate.
CPU diagnostic jobs and later GPU environment/training work overlapped parts
of the screen; wall times are observed durations, not a controlled speed
comparison. Simulation times and all raw traces are retained. All final
100000-series cases remain untouched.

Additional offline rotation candidates were rejected. Direct weighted
reprojection optimization did not improve the target-0 learned rotation;
visibility exclusion left fewer than six triangulated keypoints. Free body
rotation and straight-edge fits were not consistent enough across omitted
views. A public mesh-crease correspondence prototype regressed orientation
and sometimes hit its search bound. Training-only keypoint bias calibration
improved static median translation from 0.744 to 0.269 mm but left rotation
at 1.304 deg; variance weighting gave 1.083 deg on six static scenes but
regressed to 2.331 deg on the new target-0 capture (original 1.414 deg).
An affine pose adapter selected by training-scene cross-validation also
regressed on development and attenuated angular response. None is deployed.

A separate GPU adaptation experiment now trains the original models for
60 epochs on the existing 20 training scenes, using six development scenes
for model selection and no final cases. Original weights are unchanged.
The plug run completed; its standard pose metric is not an insertion result.
The mouth run exhausted memory during checkpoint writing with two training
jobs and Gazebo concurrent. Both files and logs were preserved: best.pt passed
ZIP integrity, partial last.pt failed. Recovery restores the validated epoch-10
checkpoint and optimizer/schedule; identical RNG continuation is not claimed.
Subsequent training runs sequentially. Candidate weights must pass the actual
production gates on recorded images, then prospective physical trials, before
any improved success rate is claimed.

## Completed mouth transfer and angular-screen ablation, 2026-10-30

Longer mouth adaptation plus the original plug checkpoint completed **2/5**
guarded insertions (`transfer30v1`). All five passed target perception; targets
1, 2 and 4 failed learned/prior angular consistency (1.185, 0.985 and 2.670 deg
respectively). Their actual terminal angular errors were approximately 0.176,
0.213 and 0.170 deg; simulator truth was opened for diagnosis only.

The predeclared `angular_ablation30v1` then completed **5/5** independently
verified physical insertions on the same fixed cases. It removed only that
learned angular veto, preserved upstream wrist orientation and retained all
independent geometry/contact/penetration thresholds. These are explicitly
excluded from guarded-production successes. Across terminal windows: depth
15.54785-15.56677 mm; lateral 0.02961-0.13204 mm; angle 0.11650-0.23988 deg.
Independent peak forces were 1.4970-3.7933 N; wall/sim durations ranged
146.883-385.425 /14.751-37.323 s. No broad reliability is inferred from these
reused development cases. Per-case metrics and source/checkpoint identity
checks are in the portable `varied_screen_*` summaries.

Original-weight crop audit: scales full-image, 1.5, 2, 3, 4, 6 and 8 were fixed
before inference; training-only median angular error selected pad 4. On six
static development scenes, pad 4 improved median angle to 0.810 deg versus
1.307 at pad 6, but on nine recorded live captures regressed to 1.733 versus
1.327 deg. Pads 1.5/2 caused large errors; pad 3 failed the original geometric
gates on all 20 training scenes. No crop setting was deployed or gate changed.
Raw image-only predictions precede diagnostic truth in
`.artifacts/reference30/crop_resolution30`.

The projected-edge training loss completed with best/final static angular
medians 1.960/1.625 deg; affine-rotation loss 1.986/1.825 deg. Neither qualified
against the original 1.310 deg baseline, much less the task's 0.25 deg limit.
Equal-camera-weight, supersampled full-CAD silhouette fitting also failed
cross-view checks (nominal maximum disagreement 1.042 mm). Those prototypes
remain offline. No negative experiment is described as a repaired rotation model.

A frozen 15-trial component ablation (`component30v1`) now tests the full
conditional path, no body correction and no lateral PI, with shuffled method
order. Its first no-body target-0 control failed lateral alignment and aborted
on force; independent peak 17.5668 N. This is retained as a failure. The 8 N
outer check is pre-command only and must not be described as a physical cap.
Remaining outcomes are pending and are not filled in prospectively.

## Rigid-grasp invariance diagnosis and targeted refinement, 2026-10-30

The original learned grasp estimates changed by 0.229-0.503 mm and 1.275-2.074
 degrees between before/after-retreat captures on four development cases.
A separate truth audit, performed only after those image-only estimates were
saved, found actual grasp translation change <=0.000104 mm and rotation change
<=0.00000209 deg. All four image/independent-pose timestamp joins were exact.
The robot camera-to-tool transforms also agree across all three views.
Wrist cameras and held plug move together: retreat changes the port view, but
cannot provide a new viewpoint of the rigidly held plug. This rules out actual
grasp motion as the explanation for these large estimate changes. It does not
isolate background, illumination or inference effects as the sole cause.
Portable audit: `rigid_grasp_invariance_diagnostic.json`.

Native-crop-matched training used 120 training /36 development images from the
same 20/6 scene split, original weights, and a non-saturating coordinate loss.
Best/final static medians were 0.2300/0.1958 mm translation and 1.0855/1.4875 deg
rotation. Archived live replay still rejected four of five angular checks for
the best checkpoint (mismatches 1.216, 1.328, 1.581, 0.823 and 0.948 deg).
It is not deployed and does not repair angular perception.

The next offline candidate retains the frozen original first-stage detector
and trains a separate refinement copy on observed white plug support, with
background suppressed using original-image proposal boxes and the robot-owned
tool mask. No object-pose/CAD mask or simulator segmentation defines this input.
The existing 20/6 split supplies 60/18 native-crop images. Its 60-epoch training
completed; final-epoch weights are the declared primary checkpoint, ordinary
fitness-best is secondary. Qualification requires unchanged geometry gates,
better median/no-worse worst static angular error and <=0.25 mm/deg grasp drift
on every recorded pair. Those are development qualification checks, not task
accuracy certificates or new physical scoring limits. Results are pending.

## Completed matched study and fresh wider screen, 2026-10-30

`component30v1` completed all 15 frozen trials with identical source/checkpoint
maps and shuffled method order: body + PI **5/5**, no body **0/5**, no PI **0/5**.
Full terminal bounds: depth 15.54777-15.56700 mm, lateral 0.03398-0.09794 mm,
angle 0.11595-0.23997 degrees. Independent force peaks 1.4212-2.5440 N, median
completion 17.167 simulated seconds, maximum whole-trial penetration 0.0003251 mm.
Four no-PI controls reached seat depth but remained 0.300-0.342 mm laterally off;
one stalled shallow. No-body controls had three shallow stalls and two force
aborts (17.5668 and 11.1248 N peaks). The 8 N check is pre-command, not a hard
physical force ceiling. This is evidence for the body/tracking interaction in
these five reused development cases, not a generalized reliability estimate.

Concurrent offline GPU work affected host load during some matched trials;
`component30_execution_context.json` records those windows. Wall time includes
harness/model configuration; simulated action time excludes it. Final reporting
will separately retain configuration wall time and action wall time, and must
not infer real-hardware cycle time from Gazebo real-time factor.

The first eight fresh `translation30v1` cases have independently passed; two
remain unmeasured at this entry. The test uses 0.5-1.5 mm initial lateral norms
and 5-15 mm gaps, paired with unknown rigid grasp jitter. Independent setup
records confirm grasp changes 1.196, 2.089, 1.941 and 2.385 mm for targets 0-3,
with source initial tip poses identical within each pair. The initial connector
is still nominally angularly aligned upstream. This is not correction of those
in-hand angular perturbations as port angular errors. No new plug model was
substituted. The v16 force-only support file exactly matches the retained
own-robot calibration vector without exposing object/target pose diagnostics.

New actual footage, decoded and terminal-frame inspected:
`video/wider_grasp41000/attempt.mp4` in portable evidence. It visibly labels the
angular-screen ablation and aligned connector handoff; truth is an evaluator
overlay only. RGB frames, executable sources, checkpoint hashes and raw records
are preserved. Stable insertion is evaluated while held; release retention,
latch engagement and electrical continuity are not tested.

The centre camera's moving cable occlusion causes 8-11 px changes in the new
foreground model's corners while side-view changes stay below 0.45 px. Original
centre point scores stay >=0.9956 despite occlusion. Fixed crops alone do not
resolve this. Three-view foreground refinement attained 0.641 deg static median
angle but 0/4 rigid-consistency passes. Left/right alone attained 0.326 deg median
recorded-image angle and 2/4 consistency passes. Side-view pixel-phase averaging
attained 0.306 deg and 1/4. The unchanged consistency limits remain 0.25 mm/deg;
none qualified, including results narrowly above the limit. All remain offline.
See `perception_repair30/` for actual diagnostics and versioned source snapshots.

An independent, frozen conditional study (50 new cases, three methods, 150
trials) is queued only if all ten wider development cases pass. The explicit
protocol is `experiments/configs/final_aligned31_protocol.json`. It samples new
200000-series cases after the gate; the older broader/full-angle 100000-series
is not opened or relabeled. Failed final cases are retained, no final tuning or
success-based early stopping is allowed, and source changes halt execution.
No final results or confidence intervals are assumed. Automatic analysis will
write `paper/FINAL_ALIGNED31_RESULTS.md` only after all 150 actual trials finish.


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

Earlier measured V20 snapshot: 3/3 completed cases passed; this includes two known repairs and one new unknown-grasp case, with eight more development cases pending at this snapshot. The new grasp case held 15.565818-15.565950 mm depth, 0.163368-0.165001 mm lateral error and 0.135021-0.135181 degrees full angle. Raw peak 6.190797 N; complete 2000 Hz coverage, no sample over 8 N. Independent setup diagnostics measured a 2.939885 mm /2.546762 degree grasp difference while the authored initial physical tip pose remained identical. These setup diagnostics are never policy inputs. Actual video was decoded and terminal image inspected. No broad success rate or final confidence interval is inferred from these development cases.
