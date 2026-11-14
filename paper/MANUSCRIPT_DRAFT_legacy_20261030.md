# From Perception Accuracy to Verified Seating: A Last-Inch Evaluation of SC and SFP Insertion

<!-- current-evidence -->
Current evidence, 2026-10-30: the repaired visual SC method has three independently
verified nominal runs with identical frozen source/checkpoint maps. It assumes
an upstream-aligned wrist, estimates grasp translation from images, and combines
learned proposals with CAD mouth/body fitting and bounded lateral PI tracking.
It does not yet establish randomized robustness, arbitrary angular-error
recovery, or repaired SFP perception. The first varied-target batch stopped
after two port-perception failures; a cable-occlusion repair is undergoing a
new physical trial. Its result is not assumed or pooled with the three repeats.

The first qualified run held 15.559 mm actual depth, 0.095-0.097 mm lateral error
and 0.077-0.078 deg angular error, with 2.309 N independently measured peak force.
All final contact/geometry checks passed. Earlier prototype results and failed
versions below are historical evidence, not pooled with this method. The old
force-only pilot is stopped. See [current measurements](VISUAL_REPAIR_20261029.md).

The four-page body should centre on conditional image-to-seating refinement and
the measured interaction between pose accuracy and controller tracking. Final
result tables require the frozen body/tracking ablation and fresh final trials.
No generic novelty in PI, CAD fitting or vision/force fusion is claimed.
<!-- /current-evidence -->


Primary-method decision, 2026-10-28: use the team's original learned
perception-to-controller pipeline as the paper's main system. The prescribed-
grasp force-only controller below is a diagnostic ablation. See
[NOVELTY_DECISION.md](NOVELTY_DECISION.md) for the focused literature review,
selected hypothesis and matched experiment requirements. The opt-in geometric
refinement extension has passed mechanism/regression tests and identified an
inconsistent SFP hypothesis on recorded images; it has not yet improved live
insertion success.

Fresh nominal original-policy trials failed before motion: SC at visual grasp
reprojection (14.91 px against 6 px) and SFP at multiview port perception after
plug priming passed. On a separately replayed before-priming SFP frame, a
geometry-accepted estimate nevertheless had evaluator-only tip/orientation
errors of 0.904 mm / 3.834 degrees. A full/crop hypothesis disagreement of
0.684 mm was detectable without truth. These observations motivate the proposed
quality-to-action mechanism; they are not evidence that the mechanism enables
reliable insertion. Final held-out trials remain unexecuted.

Working manuscript, 2026-10-26. Intended body budget: four IEEE two-column
pages including two figures and one compact results table. This is a research
draft, not a submission-ready claim of reliable insertion. Update the seeded
results from the exported trial index after each completed development batch;
do not substitute nominal demonstrations for randomized reliability.

### Interim evidence update, 2026-10-28 (replace after the frozen pilot completes)

The corrected float64/sharp-box/ball-damping backend has three successful nominal
SC repeats and three successful nominal SFP repeats. These videos and verifier
records demonstrate prescribed-grasp seating, not end-to-end neural-policy
transfer. The current paired experiment executes a separate deterministic
force-controller prototype. Its runtime observations are own TCP state and
wrist wrench; its design inputs include the prescribed grasp, fixture
orientation family and protocol handoff gap. Initial near-port placement is a
simulator reset obtained from source target frames and robot inverse kinematics.
Autonomous pickup and coarse camera-guided placement are excluded.

At the interim eight-trial checkpoint, four distinct perturbed development
configurations have each been executed with and without lateral contact
registration. SC passes 2/2 with registration and 1/2 without it; SFP passes
2/2 with registration and 1/2 without it. Each 2/2 Wilson 95% interval is
34.2-100%; each 1/2 interval is 9.5-90.5%. The unregistered failures are an SC
depth/alignment failure and an SFP lateral misalignment. These small samples
cannot establish general robustness or superiority. The planned ten-case,
twenty-trial development pilot remains active; final held-out cases are
untouched. The machine-readable interim snapshot is
`results/pilot27_interim_2026-10-28.json`.

The original YOLO pose weights and multiview/crop-refinement pipeline remain
separate evidence. Their accuracy reports must not be combined with this
prototype's insertion success as though a complete learned perception-to-control
pipeline had been evaluated. The unchanged SC visual grasp gate still rejects
the observed 10.07 px reprojection residual against its 6 px threshold.

## Abstract

Precision connector assembly requires more than accurate detections or a
simulator success event. We examine the transfer of a multiview vision and
force-based cable-connector pipeline from a proprietary execution environment
to a reproducible Gazebo last-inch testbed. The testbed retains source SC/SFP
connector and target geometry, the UR5e/tool/cameras, five dual-port NICs, five
SC adapters, five cables and staging fixtures. An independent evaluator checks
actual tip depth, lateral and full angular alignment, terminal dwell and
physics-contact observations with automatic latching disabled. We identify
image/TF timing inconsistency and correct exposure-time pose lookup, while
showing that a residual SC detector failure persists even with verified camera
calibration. Under an explicit prescribed-grasp condition, nominal SC and SFP
motions reach independently verified held geometric seats. Initial randomized
development trials expose both perception rejection and contact/tracking
failures. These results support an auditable distinction between perception
accuracy, a command-completion event and a physically consistent geometric
seat; they do not establish hardware transfer or final randomized reliability.

## I. Introduction

Electronics manufacturing combines narrow connector clearances with occlusion,
uncertain grasp geometry and contact-rich motion. Our challenge system used
YOLO11-Pose, CAD geometry, multiview triangulation, native-image crop refinement
and guarded insertion with recovery. Moving this system from Flowstate to
Gazebo raises two practical questions: which observations and assumptions
actually transfer, and what evidence establishes insertion independently of
the orchestration or scoring system?

We study the last inch: a held connector starts 5–15 mm from its requested
target. Pickup, transport and release are excluded. We distinguish an arbitrary
grasp, which requires visual grasp estimation, from a prescribed rigid fixture,
whose tip-to-TCP transform is a declared design input. This controlled condition
isolates insertion from a grasp-estimation failure; it must not be interpreted
as solving uncertain grasp transfer. Our practical contribution is an explicit
observation contract, calibrated frame/timing diagnostics and independently
checked insertion evidence with negative results retained. A new control
contribution remains contingent on the seeded recovery experiments.

Related assembly systems already combine visual estimation and compliance.
Factory and IndustReal address contact simulation and transfer; Insert-One
studies visual tracking and impedance under in-hand uncertainty; FORGE and
ContactDP use force-aware or multimodal policies. We therefore do not claim
novelty for combining cameras and force sensing. Our emphasis is the measured
failure-to-verification chain for this connector system. See RELATED_WORK.md
for primary references and the limits of the present comparison.

## II. Observation contract and execution

The policy receives camera images and intrinsics, exposure-time optical TF,
controller TCP state and wrist wrench. Simulator cable/port poses and contact
identities are evaluator-only. For prescribed-grasp trials, a single fixed
fixture transform is used across SC and SFP resets; the protocol supplies the
handoff gap. Per-trial ground-truth grasp estimates are not read by the policy.
All compared methods share the controller's measured load compensation.

For camera i, image coordinates and the transform into the camera must refer
to the same exposure. The deployed view builder previously combined an image
with latest wrist TF. We changed lookup to the image timestamp and drop a view
if that transform is unavailable. We retain the SC 6 px reprojection and
3.5 mm rigid-shape rejection limits. A fixed-grasp benchmark does not silently
replace the deployed visual-grasp estimator: it is a separate declared
condition. The deployed policy continues to reject invalid grasp estimates.

The initial-pose baseline advances along the measured handoff insertion axis
at 1 mm/s, with Cartesian stiffness 500 N/m, rotational stiffness 50 Nm/rad,
and an 8 N tared wrist-force abort. It dwells for two simulated seconds after
nominal travel. Research visual variants attempt bounded physical-mouth
refinement using analytic opening detections and timestamped camera geometry;
they require agreeing multiview evidence. A force-recovery variant additionally
provides bounded search after a force-associated tracking stall. The current
pilot's visual rejections occurred before recovery, so it cannot measure the
recovery branch's effect. These adapters are reimplementations around reused
components, not demonstrated replicas of the entire challenge controller.

Development prototypes register the two lateral axes from opposed, settled
contact limits under the prescribed grasp. Axial command lead is capped at
0.5 mm. Bounded tracking compensation uses only controller TCP error; terminal
axial support is limited to 0.25 N, derived from 500 N/m times the lead bound,
and enabled below 1 N. Twist registration requires opposing moment signs and
settled clipping; merely failing to track a commanded angle is insufficient.
Source sensor torque noise is 0.05 Nm per axis, so a raw 0.08 Nm check produces
frequent noise stops. A separate prototype uses a 50-sample moment estimate,
100-sample stationary tare and conservative Gaussian uncertainty margin against
the same numerical value. This is a windowed observation contract with finite
delay, not a raw-sample torque bound or a demonstrated hardware safety limit.

## III. Testbed and independent evaluation

We pin Phase 1 assets to commit 7148df9 and the evaluator container by digest.
The runtime is ROS 2 Kilted with Gazebo 9.5.0 and BulletFeatherstone physics.
Headless cameras render the actual source meshes. Development uses a 1 ms
physics step and 5 Hz wrist images. The first accepted SC nominal trial used
the source 2 ms step; it is reported separately from the 1 ms SFP trial.
Gripper/ATI visuals and inertia remain, but their collision shapes are excluded
for the prescribed attachment. Connector and target collisions remain active.
One source cable is articulated; four inactive cables are fixed-rest context.
The opposite endpoint is unconstrained. Attempts to fix it were rejected after
measured support drift rather than accepted from joint-component existence.

Automatic port touch/latch behavior is disabled. A startup fixture is removed
before the trial; one robot-to-connector grasp constraint remains. The evaluator
records actual simulated tip poses, wrist wrench, fixture state and all 87
monitored collision components. A read-only physics observer publishes empty
contact heartbeats as well as contact events, making silence distinguishable
from missing observation. Streams are joined by simulation timestamp after
transport has drained.

A success requires a continuous terminal second within source-geometry depth
and alignment bounds: SC 15.64 ± 0.25 mm depth, ≤0.25 mm lateral and ≤0.25° full
angular error; SFP 45.8 ± 0.25 mm, ≤0.10 mm and ≤0.10°. Both require ≤0.1 mm
terminal penetration, bounded sample gaps and valid contact observation. SC
requires requested-target contact in the terminal window. The nominal SFP seat
has source-geometry clearance; its consistency check instead requires zero
source-box overlap at observed terminal poses and a matching contact-positive
control, in addition to live physics observation. No result establishes
electrical mating, latch engagement or retention after release.

Development seeds are separated from the untouched final 100000-series
manifest. The development design varies all five target indices, both SFP
slots, handoff gap, lateral error and three angular components. We report
per-seed outcomes, independent force peaks, simulated and wall time, all failure
categories and Wilson 95% intervals. Setup failures are excluded from policy
denominators and retained separately. Repeated debugging runs of one seed are
not independent observations.

## IV. Results

Historical SC records contain 135 image groups from 45 source trials per
condition. On the crop-selection split, median tip error decreased from
0.4562 to 0.2633 mm; the paired trial-cluster bootstrap difference was
−0.1929 mm, 95% CI [−0.2516, −0.1443]. The selected crop variant's separate
stored held-out median was 0.2733 mm. A matched held-out full-frame baseline
and original training/image manifests were unavailable. These are recomputed
records, not a new inference replication or an insertion-success result.

The transfer audit measured about 2 mm camera displacement from latest/image
TF mixing during reset. Exposure-time TF agreed with actual camera poses
within 0.025 mm / 0.002° while moving and 0.001 mm / 0.0001° at rest; source
field of view and reported intrinsics agree. The stationary SC network still
fails the unchanged gates. Seven of eight recorded pad-6 correspondences have
epipolar Sampson distances of 8.32–26.98 px. Direct pixel-space rigid fitting
leaves 13.48 px mean reprojection error and 14.85 mm diagnostic tip error.
Calibration/timing repair alone does not solve this inconsistent prediction.

| Nominal held-seat demonstration | SC | SFP |
|---|---:|---:|
| Terminal depth range, mm | 15.5668–15.5674 | 45.7959–45.7993 |
| Maximum terminal lateral error, mm | 0.0168 | 0.0331 |
| Maximum terminal full angular error, ° | 0.02276 | 0.02063 |
| Maximum terminal penetration, mm | 0.001624 | 0.001116 |
| Independent tared force peak, N | 0.9654 | 1.9218 |
| Motion plus dwell, simulated / wall s | 27.642 / 119.558 | 57.802 / 307.477 |

The SFP 2 mm lateral-miss positive control produced 1,326 physical requested
tip/cage contact pairs among 2,135 raw cached pairs and stopped at about 8.04 N
policy-sampled force. Nominal seats are demonstrations, not a randomized
reliability estimate. Whole-trial penetration was 0.001762 mm for SC and
0.06793 mm for SFP. These values follow a correction to the evaluator: the
pinned Bullet plugin reports negative penetration and positive separation.
Reanalysis preserved raw data and thresholds; the nominal verdicts survived.

The first matched development pilot had one SC and one SFP case per method.
Initial-pose insertion passed SC and failed SFP lateral alignment; all four
analytic visual variants rejected perception. SC view estimates disagreed by
more than 2 mm, while SFP had only one usable view. Thus recovery was not
exercised. The resulting one-trial Wilson intervals are extremely broad:
20.7–100% for the SC baseline and 0–79.3% for each zero-success condition.
The extended initial-pose baseline produced two SC terminal geometric seats among four
eligible cases (Wilson 95% 15.0-85.0%) and no SFP seats among five (0-43.4%).
One additional completed SC execution already had 8 mm remote-connector/board
overlap before motion and is reported separately; raw completed SC verdicts
were 2/5. A new full-scene setup contact gate rejects such handoffs.

On repeated development SFP seed 3003, lateral registration passed alignment
but initially stopped 0.381 mm short in depth. Bounded terminal axial support
reduced own tracking error to about 0.089 mm and achieved a held clearance
seat: depth 45.7191-45.7202 mm, lateral <=0.05758 mm, full angle <=0.05570
degrees, independent force peak 6.395 N and 83.924 simulated seconds. This is
one development repeat, not final reliability or an isolated causal effect.
The shared-prior baseline subsequently seated the same case without lateral
registration, so that repeat does not establish registration's benefit.
An opposed-twist prototype recentered another case to about 0.004 degrees,
but that trial stopped on stale policy-path wrench data before seating; a
repeat stopped on its moment bound. No reliable full-pose recovery is claimed.

Terminal geometric seating does not erase earlier contact errors. The two
SC development seats and the successful SFP repeat had whole-trial distal
penetrations of 4.258, 5.792 and 1.889 mm, respectively, and fail the additional
recorded whole-trial 0.1 mm penetration screen. The nominal demonstrations
pass it. Repeating SFP3003 at 200 solver iterations and 0.5 ms steps retained
about 1.89 mm whole-trial penetration. A passed setup hold also failed to
guarantee valid pre-command contact in one repeat. Contact fidelity remains
unresolved; no full-cable robustness or final effect-size claim is supported.
An isolated query at a recorded capsule/box pose returned a false positive
gap in float32 Bullet, while float64 reported overlap. A separate same-source
backend is under qualification; this is not yet an insertion improvement.

Figure 1: headless source-mesh images before and after nominal insertion,
with explicit gripper-occlusion and collision-scope caveats.
Figure 2: recorded depth and independently tared wrist force versus simulation
time; source seat tolerances and the unchanged force-abort threshold are shown.
Artifacts: experiments/evidence/insertion_development/nominal_camera_frames.png
and nominal_depth_force.png. Prefer a recovery comparison over a second
nominal figure only if a completed controlled study supports it.

## V. Limitations and practical lessons

A physically consistent held geometric seat is a narrower claim than reliable
industrial assembly. Prescribed grasp, omitted gripper collisions, fixed-rest
inactive cables, software rendering, uncertain distal contact fidelity and
missing hardware limit transfer. The simulation exception reported for this
special collection does not remove those limitations. No marker-independence,
friction-generalization or final reliability claim is supported yet.

The evidence favors checking frame/time contracts before retuning perception,
matching grasp checks to the declared evaluation condition, testing contact
observability with a deliberate miss and verifying seating independently of
action or scoring events. The next scientific requirement is a correction that
improves matched randomized outcomes under frozen gates, followed by a held-out
evaluation. Until then, practical failure analysis is supported; a robust new
insertion method is not established.
