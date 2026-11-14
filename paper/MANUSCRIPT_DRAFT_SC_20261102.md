# From Weighted Keypoints to Verified SC Seating: A Task-Conditioned Last-Inch Pipeline

Latest measured checkpoint: `persistent32v1` completed 8/11 SC development
trials (2/2 known repairs, 6/9 new cases); final evaluation remains unopened.
SFP original-perception repair is now active. See `REPAIR_20261102.md`.
Older partial counts below are historical, not current reliability claims.

**Scope clarification, 2026-11-01:** The intended final paper covers both SC and
SFP last-inch insertion. The title, abstract and results below are a preserved
SC evidence draft, not the final agreed scope. Add the original-perception SFP
method, verified experiments and connector-specific comparisons before rewriting
the final claims. Earlier shared-prior SFP seats cannot fill that evidence gap.

Working research manuscript, updated 2026-10-30. Intended four-page body; not
submission-ready or typeset/page-verified. The previous SC/SFP draft is preserved
in `MANUSCRIPT_DRAFT_legacy_20261030.md`. Current claims concern SC only. No
held-out final trial has run. Results below are development measurements.

## Abstract

Transferring a connector-insertion policy requires validating the path from
perception to physical seating. We study a learned multiview pipeline in a
source-asset Gazebo testbed for last-inch SC cable insertion. Confidence-weighted
keypoints propose plug and port poses; public-CAD image fits refine translation,
while bounded lateral tracking compensates compliant robot motion. An independent
recorder verifies seating geometry, continuous dwell, contact and penetration.
Across five fixed development targets, longer mouth adaptation improves the
guarded policy from one to two verified insertions. A separately declared
angular-screen ablation achieves five, conditional on upstream angular alignment.
In a separate matched component study, body correction plus lateral tracking
passes five of five cases while either component removed passes none.
The failures distinguish mouth occlusion, inaccurate learned rotation and
translation/control error. These results establish a conditional mechanism,
not unrestricted visual rotation recovery or broad insertion reliability.

## I. Practical problem and contribution

The team's AI for Industry Challenge system combined learned connector/port
keypoints, confidence-weighted multiview geometry and guarded force-aware
insertion. Flowstate supplied execution and transport. To examine the strategy
outside that environment, we isolate the final approach of a held SC connector
and port the original learned proposals and controller into Gazebo. The robot
starts near its requested port; pickup and long-range transport are excluded.

Two errors must be separated. A geometrically self-consistent estimate can be
biased relative to a submillimetre assembly tolerance. Conversely, a correct
image estimate need not produce the requested physical motion under a compliant
controller. We therefore evaluate the complete chain, retaining rejection and
force-abort failures in the denominator. A successful simulator event or an
estimated insertion depth is insufficient evidence of seating.

The proposed engineering formulation separates angular readiness supplied by
an upstream process from translation observable in the near-port RGB images.
It combines learned proposals with occlusion-aware public-CAD fits and bounded
tracking, and records the assumptions and independent physical outcome. Our
present contribution is this task-conditioned integration and its measured
failure analysis. CAD fitting, confidence weighting, PI control and vision/force
fusion are established methods; their individual use is not a novelty claim.
A matched component comparison supports their interaction on these development
cases; generalized performance still requires a frozen final study.

## II. Related methods

Insert-One combines 6-DoF visual tracking and impedance control for insertion
with uncertain object attachment [1]. FORGE learns force-aware manipulation
under pose uncertainty [2]. BayesContact maintains a pose belief using visual
and contact evidence and active probing [3]. Multiview keypoint probabilities
and geometric alignment have also supported pose estimation and false-positive
rejection [4]. Thus, neither multiple sensors nor a confidence gate establishes
novelty. Our narrower question is which observable errors must be corrected
for a specified connector tolerance, and whether the corrected estimate survives
physical tracking and contact. We do not claim stronger generality than these
prior methods, or compare their published success rates across different tasks.

## III. System and task-conditioned correction

### A. Observations and original weighted proposal

The policy consumes three actual RGB wrist-camera streams, their calibrated
intrinsics, exposure-time camera transforms from robot TF, own TCP state and
wrist wrench. It receives task identity but no simulator object/target poses,
segmentation IDs or contact truth. Such truth is available only to dataset-label
generation and independent diagnostic scoring. Intrinsics and public CAD are
ordinary model inputs, not privileged live scene state.

The original plug network estimates eight projected cuboid keypoints in each
view. Confidence-weighted DLT triangulates them, followed by a weighted rigid
fit to the known local geometry. The original six-pixel reprojection and 3.5 mm
keypoint-shape rejection limits remain unchanged. These limits check consistency;
they do not certify 0.25 mm or 0.25 degree physical accuracy. Native pad-six crop
refinement and the original plug checkpoint remain in the reported trials.
The mouth proposal uses a separately adapted copy of the original mouth model.

Image/TF timing and camera conventions are checked before geometry: camera
optical coordinates, ROS clock domain, calibration matrix, exposure-time robot
pose and raster-centre conversion must agree. The public-CAD projection uses
the verified minus-half-pixel raster convention without altering calibration.
Retreat and re-priming require fresh observations; old camera frames must not
be paired with current wrist extrinsics.

### B. Public-CAD translation refinement

A blue mouth proposal is refined against projected source rim geometry. White
plug and orange cable occlusion masks exclude image edges that do not belong to
the rim. On two recorded captures, omitting the cable mask gave 1.869/2.073 mm
maximum leave-one-view-out disagreement. Including it reduced these to
0.108/0.091 mm under unchanged gates. This is a diagnosed image-association
failure, not justification for increasing the gate.

For the held white plug, the projected public body hull and robot-owned tool
mask support subpixel boundary measurements. Conditional on the port-derived
orientation prior, translation minimizes a robust support residual across three
views. For image direction d and public model points X, the model support is
max_X d^T pi(P_c(RX+t)); it is compared with measured, unoccluded body support.
The fit retains a one-pixel residual bound, three-millimetre correction bound
and 0.25 mm leave-one-view-out agreement limit. None of these consistency tests
is an uncertainty-coverage guarantee.

The crucial condition is that the upstream process supplies angular alignment
under a prescribed rigid grasp. The policy preserves the physical wrist
orientation rather than commanding uncertain learned grasp rotation. It still
estimates grasp translation from pixels. This evaluation does not solve arbitrary
grasp-rotation errors. The guarded version additionally rejects learned/port
rotation disagreement above atan(0.25/15.64), or 0.916 degrees. That is an
estimator-consistency veto, not certification of the true angle. An explicit
ablation removes only this veto; its successes are reported separately.

### C. Tracking and force-aware approach

The visually estimated target and grasp feed the original force-aware axial
SC approach. A lateral own-robot tracking term uses proportional gain 400 N/m
and integral gain 200 N/(m s), bounded together to 0.5 N. Its integral freezes
at observed force of at least 1 N. The native extra wrench-feedback gain is
zero for these compared methods, avoiding an unintended second force loop.
A calibrated common load-support/tare procedure precedes each trial.

The outer eight-newton check runs before motion commands; it is not a guaranteed
physical force ceiling. Any observed overshoot is retained in independent peak
force measurements and failure analysis. A low nominal insertion-force target
must not be reported as the peak experienced load.

## IV. Testbed and independent verification

The corrected source-asset scene contains the UR5e, tool and cameras, five
dual-port NIC cards, five SC duplex adapters, five cables and staging context.
The active cable is dynamic; inactive cables provide fixed context. Rigid grasp
attachment excludes pickup/slip behavior. Gripper/ATI visuals and inertia remain,
while their collision shapes are excluded; connector and target collisions
remain active. No automatic seating weld or success latch assists insertion.
These are disclosed simulation approximations, not hardware validation.
Seating is verified while held; release retention, latch engagement and electrical
continuity are outside this evaluation.

![Actual headless scene, fixture and held SC seat](../experiments/evidence/original_repair/actual_headless_setup30.svg)

Trials use the audited float64 Bullet backend with zero source-box margins,
source ball-joint damping, 0.5 ms steps and 200 solver iterations. Binary/source
hashes, controller bootstrap/hold records and actual headless images are archived.
Signed negative contact distance or nonzero solver impulse establishes active
contact; positive cached manifold gaps alone do not. Independent wrist FT,
not the plugin's scalar-impulse contact-wrench field, supplies reported force.

A successful SC trial must continuously hold 15.64 +/-0.25 mm insertion depth,
<=0.25 mm lateral error and <=0.25 degree full angular error for one simulated
second, with requested-target contact and bounded observation gaps. Whole-trial
penetration must remain <=0.1 mm. The verifier retains the post-command tail.
Ground truth enters this scoring process only, after policy execution.

## V. Development results and failures

Table I lists separate frozen versions on the same five development targets.
Initial gaps were 5.18-9.37 mm, lateral offsets <=0.307 mm in norm and small
angular perturbations. These cases were reused during repair; they are not
fresh final trials or evidence of broad robustness. All five cases, including
perception rejection, remain in each completed batch denominator.

| Version | Verified / attempted | Failure stage |
|---|---:|---|
| Earlier mouth + cable mask, guarded | 1/5 | Mouth: 3; learned angular veto: 1 |
| Longer mouth adaptation, guarded | 2/5 | Learned angular veto: 3 |
| Same models, angular-screen ablation | 5/5 | None in these fixed cases; alignment assumed |

Longer mouth adaptation removed the observed mouth-stage failures, but did not
repair learned plug rotation. Removing its angular veto allowed the conditional
translation/control path to seat all five. Across their terminal windows,
measured depth was 15.54785-15.56677 mm, lateral error 0.02961-0.13204 mm and
angular error 0.11650-0.23988 degrees. Sampled independent 50 Hz force peaks ranged
1.4970-3.7933 N; maximum observed whole-trial penetration ranged
0.000064-0.002060 mm. These rates can miss sub-sample impacts and cannot
certify a continuous force or penetration ceiling.
Durations were 146.883-385.425 wall seconds or 14.751-37.323 simulated seconds.
One terminal angle approached the unchanged limit; no larger angular capture
range is inferred. Earlier three-of-three nominal repeats demonstrate repeatability
of one setup only and are not pooled with these versions.

Offline experiments also show why better pose-detection metrics are insufficient.
On six static development scenes, original plug median angular error was
1.310 degrees. Standard adaptation selected by its usual validation fitness
gave 2.709 degrees for the best checkpoint and 1.086 for the final epoch.
Projected-edge and weak-perspective affine-rotation training objectives gave
best/final medians 1.960/1.625 and 1.986/1.825 degrees. These variants were
rejected for deployment. Free rotation fits, line/silhouette fitting and
augmentation ensembles likewise did not qualify a general rotation repair.

The separate source-frozen, shuffled-order component study completed all fifteen
trials with identical source/checkpoint maps. Body correction plus lateral PI
passed 5/5; no body correction passed 0/5; no lateral PI passed 0/5. The full
method held 15.54777-15.56700 mm depth, 0.03398-0.09794 mm lateral error and
0.11595-0.23997 degrees angular error. Independent peak forces were
1.4212-2.5440 N, median completion was 17.167 simulated seconds, and maximum
whole-trial penetration was 0.0003251 mm. Four no-PI trials reached seating
depth but retained approximately 0.300-0.342 mm lateral error; one stalled
shallow. Without body correction, three trials stalled near 3.14-3.17 mm and
two aborted with 17.57/11.12 N peaks. Thus, force-aware motion and high detector
confidence alone did not compensate the remaining position error.

![Matched component outcomes and measured forces](../experiments/evidence/original_repair/component30_outcomes.svg)

These reused development cases support the proposed component interaction under
the stated contract. They do not establish general superiority. Some concurrent
offline GPU work changed host load; recorded wall durations are descriptive,
not a controlled throughput comparison. A separate ten-case wider translation/
grasp screen without concurrent training passed 9/10: default grasps 5/5 and
perturbed rigid grasps 4/5. The failed case reached the shoulder with less than
0.15 mm lateral and 0.06 degrees angular error, but approached at approximately
5.67 mm/s and produced a 34.77 N independent wrist transient. Its force abort
remains a failure; good alignment alone did not ensure a gentle seat. The final
readiness gate stopped, and no final cases were opened. A selectable terminal braking repair passed that failed case once, holding
15.5660 mm depth, less than 0.050 mm lateral error and 0.048 degrees angular
error, with a 1.16 N sampled 50 Hz peak. This regression is provisional pending
per-step sensor validation and fresh development trials. A held-out frozen-policy
study remains required before reliability
confidence intervals or robustness superiority are reported.
A diagnostic later showed why force bandwidth matters: an unbraked attempt
recorded a 2.28 N peak at 50 Hz while its raw force stream captured 43.77 N.
The first braking repair still produced 24.13 N on a new development case.
Inspection exposed an axial setpoint that advanced to a robot overshoot,
removing the restoring error needed to brake. Retaining a persistent reference
and adding a settling reserve repaired two known cases under full 2000 Hz
coverage, with 4.38/4.55 N peaks. This remains a development regression result;
The first two new development cases also pass (4/4 completed overall, seven
pending), with a maximum raw peak of 6.19 N across these four cases. The
source-gated frozen comparison remains pending. Sparse low force
records and visual seating alone are insufficient evidence of gentle insertion.

The final 100000-series remains unused. Repaired SFP perception, arbitrary
angular recovery and hardware generalization are not demonstrated.

A further diagnostic explains why more training alone was insufficient. Across
four exact-stamp before/after-retreat pairs, actual grasp translation changed
by at most 0.000104 mm, while original estimates changed by 0.229-0.503 mm and
1.275-2.074 degrees. The moving orange cable occluded the centre camera even
as original point scores remained above 0.9956. Since the held connector and
wrist cameras move together, retreat changes the port view but does not create
a new connector-relative viewpoint. Context-suppressed refinement reduced
static median angular error to 0.641 degrees; side-view phase averaging reached
0.306 degrees median on recorded captures. Neither passed the declared rigid-
consistency qualification, so neither replaced the physical policy's weights.

## VI. Practical implications

The results support reporting separate interfaces for estimated pose consistency,
declared upstream alignment, robot tracking and verified physical seating.
An inaccurate learned angular veto can reject an otherwise executable conditional
last-inch task; omitting that veto does not solve angular perception. Likewise,
low image residual and low requested insertion force cannot substitute for actual
alignment and measured peak load. The component study supports the translation/
tracking mechanism. Fresh randomized coverage and an independent final study
are still needed; broader grasp variation and reliable angular correction
remain open.

## References

[1] H. Chang, A. Boularias, and S. Jain, "Insert-One: One-Shot Robust Visual-Force
Servoing for Novel Object Insertion with 6-DoF Tracking," IROS, 2024.
https://www.merl.com/publications/TR2024-137

[2] M. Noseworthy et al., "FORGE: Force-Guided Exploration for Robust Contact-Rich
Manipulation under Uncertainty," arXiv:2408.04587, revised 2025.
https://arxiv.org/abs/2408.04587

[3] A. Kamireddypalli et al., "BayesContact: Uncertain Pose Estimation via
Visuo-Tactile Proposals and Simulation-based Inference," arXiv:2607.16123, 2026.
https://arxiv.org/abs/2607.16123

[4] A. Li and A. P. Schoellig, "Multi-View Keypoints for Reliable 6D Object Pose
Estimation," ICRA, 2023. https://arxiv.org/abs/2303.16833

Evidence: `../experiments/evidence/original_repair/`; chronological diagnostics:
`VISUAL_REPAIR_20261029.md`. Intended figures: actual headless scene/seat and
paired body/tracking outcomes with measured force traces. Do not fabricate
uncompleted rows, confidence intervals, physical measurements or novelty claims.
