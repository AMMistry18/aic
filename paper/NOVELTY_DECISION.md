# Research direction: improve the original visual insertion pipeline

## Required scope clarification - 2026-11-01

The requested paper remains about **SC and SFP last-inch insertion**. SC-only
claims in later development entries describe the available evidence, not the
agreed final scope. The candidate contribution is a shared perception-to-control
formulation with explicitly connector-specific geometry and seating behavior,
supported by matched component and failure analysis for both connector types.
Cross-connector effectiveness is not established by SC results or by the earlier
shared-prior SFP demonstrations. Retain the original learned SFP pipeline as an
executable baseline and qualify its improvements prospectively before claiming
robust SFP performance. This scope extension supplies no automatic novelty claim.

Decision date: 2026-10-28. This is a testable research choice, not an established
novelty or performance claim.

## Primary system and evidence boundary

The primary method is the team's existing `InsertionPolicy`: original learned
SC/SFP plug and port checkpoints, native-image crop refinement, confidence-
weighted multiview geometry, target selection and the original guarded
controllers/recovery. Preserve an executable baseline with checkpoint and source
hashes before making changes. The prescribed-grasp force-only prototype is a
diagnostic/control ablation; its successful videos do not validate the learned
pipeline. The analytic `lastinch_visual.py` development adapter is also not the
original learned port pipeline.

Simulator target poses may set and evaluate a trial, but must not enter runtime
visual candidate selection, pose fitting, reacquisition, or motion decisions.
Rigid grasping can remain a declared environment approximation while the
original system estimates its grasp visually. A fixed-grasp pose supplied to
control is a separate ablation, never a silent fallback.

## Relevant prior work and excluded novelty claims

Primary sources checked 2026-10-28; this is a focused comparison, not an exhaustive
search or a claim of being the first. Full-text sources were consulted where
available; the Insert-One comparison below is based on its publisher abstract.

| Work | Existing capability | Consequence for our contribution |
|---|---|---|
| [Insert-One, IROS 2024](https://www.merl.com/publications/TR2024-137) | Visual 6-DoF tracking coupled to impedance control for insertion, with uncertain object attachment | Vision plus compliance and visual grasp tracking are not novel alone |
| [FORGE](https://arxiv.org/abs/2408.04587) | Force-guided exploration under pose uncertainty | Bounded force-aware recovery alone is insufficient novelty |
| [ContactDP, 2026](https://arxiv.org/html/2609.23800v1) | RGB, wrist wrench and fingertip tactile observations with diffusion actions and hybrid low-level control | Do not claim novelty for multimodal insertion or switching sensor roles |
| [SyPAS, author project](https://sypas2026.github.io/) | Synthetic socket perception, visual servoing and compliant connector search | Synthetic retraining, crop refinement and servoing are not new in isolation |
| [Keypoint-based EKF visual servoing, 2026](https://arxiv.org/html/2602.06834v2) | Keypoint measurement fusion, pose covariance and probabilistic visual control | Adding an EKF or covariance to keypoints is not a new contribution by itself |
| [Active Extrinsic Contact Sensing](https://arxiv.org/html/2110.03555v2) | Active contact estimation and contact-space insertion policy | Active probing and estimating contact geometry already have strong precedent |
| [Touch2Insert, ICRA 2026](https://arxiv.org/abs/2603.03627) | Relative shape registration from high-resolution tactile contact | Our wrist FT sensor must not be described as providing equivalent tactile images |
| [Dual control for assembly, TASE 2026](https://research.polyu.edu.hk/en/publications/dual-control-for-active-estimation-and-path-planning-in-the-autom/) | Belief-based active estimation and planning from contact forces on a UR10e; publisher abstract checked | Information-seeking assembly actions have precedent; no generic active-sensing novelty claim |

## Selected hypothesis

**Use measured multiview disagreement and connector geometry to decide when to
reacquire visual evidence and when a bounded contact correction is justified.**

The specific problem is that detector confidence can remain high for geometrically
inconsistent keypoints, and a second crop pass can make a pose worse. The
original implementation independently keeps the highest-scoring detection in
each view before fitting a shared rigid body. That is a candidate association
failure mechanism, not yet a diagnosed cause in every failed scene.

The proposed extension retains the learned checkpoints and existing strict
reprojection/shape gates. It jointly evaluates candidate identities and crop
conditions using exposure-time camera geometry, checks held-out-view or temporal
agreement when available, and reports abstention when visual evidence is
inconsistent or ambiguous. Passing a geometric consistency gate does not prove
the pose is accurate or the correct object was selected.

After valid plug and mouth estimates exist, transform uncertainty into the
relative insertion frame. Compare a development-calibrated error envelope to
the geometry's directional clearance over insertion depth. Choose a bounded
visual reacquisition or a force correction according to which residual degrees
of freedom are observable. Do not invent narrow Gaussian confidence from YOLO
scores or from an inverse Hessian alone: systematic bias and correlated views
need explicit calibration and coverage checks. Translation probes cannot
certify yaw. Do not add camera motion until collision-free bounded movements
and observability predictions are validated.

The candidate paper contribution is this concrete perception-quality-to-action
mechanism and its controlled evidence on SC and SFP, including cases where it
correctly refuses insertion. We will narrow or abandon that novelty claim if
prior art already provides the same mechanism or matched experiments show no
benefit. Physics compatibility fixes remain supporting engineering evidence.

## Execution and ablations

1. Replay the actual original checkpoints, not detector substitutes. Record all
   detections, confidence, crop geometry, epipolar residuals, rigid-fit residuals,
   miss reasons, inference times and source hashes. Score GT only afterwards.
   Deduplicate archived images: several old archive directories contain copies
   of the same capture and are not independent samples.
2. Run the original SC and SFP policy on corrected-physics nominal scenes. Record
   the first failed stage separately: detection, association, grasp estimation,
   port identity, alignment, contact recovery, terminal seating. An accepted
   pose and a completed command are not insertion success.
3. Improve the measured bottleneck. First test geometric candidate association
   and selective crop reacquisition with frozen weights. If all views remain
   inconsistent on the correct object, collect a separate development dataset
   to diagnose semantic keypoint labels/domain shift before considering training.
   Do not fix a detector by fitting per-trial simulator poses or raising gates.
4. Compare original pipeline, original plus association/refinement, and original
   plus refinement and uncertainty-directed recovery. Share initial cases,
   controller support, force guards and time budget. Include fixed crop and
   always-probe ablations to test whether adaptive decisions add value.
5. Freeze the method after development. Run independent target/pose/layout seeds,
   with final 100000-series untouched during diagnosis. Report verified seat
   rate with confidence intervals, failure/abstention rate, sim and wall time,
   force peaks, calibration coverage, and recovery use. More accepted estimates
   with worse actual alignment are a regression.

## Four-page paper structure

Problem and narrow last-inch scope; original pipeline plus a compact new
quality-to-action module; one failure-analysis figure; one matched-ablation
table/robustness figure; limitations. Historical crop accuracy, corrected-
physics force-only repeats and fresh end-to-end learned results are separate
datasets. The existing manuscript can be written now, but its abstract must not
promise end-to-end reliability or novel recovery before those trials exist.

## First executed checkpoint replay

Original `best_sc_plug_pose.pt`, SHA256
`acdae2c1896d0d8d262b6c07fba4da1af6a80d501d7c23dc38aef33dffd2b665`,
was rerun on one deduplicated archived three-camera development capture.
Full-frame inference yielded 0/4/1 candidates in left/center/right views. All
four center/right pairings failed unchanged 6 px / 3.5 mm geometry gates. The
lowest-reprojection pairing measured 7.957 px, 2.421 mm shape RMSE; post-hoc
evaluator tip error remained 9.003 mm and rotation error 103.802 degrees. It
would be incorrect to accept that pairing or claim association solved the scene.

The original independent-argmax full-frame result measured 9.746 px, 4.467 mm
shape RMSE and 15.038 mm evaluator tip error. The pad-6 refinement measured
20.249 px, 6.206 mm and 18.014 mm, respectively. Pad-3 reproduced the full-frame
result. Warm inference wall times were 1.647 / 2.125 / 3.264 seconds for full /
pad-3 / pad-6 in this CPU runtime; these single calls are diagnostics, not a
latency benchmark. No tested condition passed.

Raw detections, every candidate pairing, checkpoint/image hashes and annotated
images: `.artifacts/original_perception28/sc_archive/`. This replay supports
selective refinement and detector/correspondence diagnosis, but provides zero
new insertion successes. Runtime packages are frozen separately from the
simulator under `.artifacts/original_perception28/runtime_freeze.txt`.


## Corrected-grasp diagnosis and implemented action, 2026-10-28

The SC failure must be reinterpreted after finding that the earlier reset used
an SFP grasp. The actual canonical SC collection uses the reversed cable model
and a world-axis spawn offset at robot HOME. Composing those source frames gives
`T_tcp_tip = inverse(T_world_tcp_HOME) @ T_world_cable_HOME @ T_cable_sc_tip`.
The derivation agrees with independently expanded libsdformat frames to 1e-12.
This is evaluator setup geometry; runtime perception never reads the transform.
Original weights then prime SC in three cameras at 1.29 px, so retraining is no
longer the next justified response to the old 14.91 px failure.

A new capture still shows 0.558 mm / 0.649 degree evaluator error after the
original pad-6 plug refinement. The SC mouth is partially occluded: a local
candidate has 2.072 mm evaluator center error, 7.863 mm size residual, and
1.559 px center reprojection. Fitting its known rectangle gives 4.090 mm center
error and 8.960 px corner reprojection, so using the rigid fit as an automatic
replacement would make this sample worse. Pads 4, 6, 12 and 24 do not repair it.
This supports an action on visibility, rather than accepting the center alone.

`visual_reacquisition.py` now implements one 10 mm retreat at 2 mm/s along the
negative visually measured insertion axis, with force-change and tracking
bounds, a measured endpoint check, and re-priming after motion. The opt-in
method promotes existing SC dimension/reprojection warnings to abstention and
requires strict consensus. This is an implemented component of the selected
quality-to-action hypothesis, not a claim that retreat itself is novel.
Matched trials must compare it against the same tared controller baseline and
an always-retreat ablation. A successful retreat is not an insertion outcome.

The native controller tare omission and swallowed motion-rejection exceptions
were also repaired. These are execution defects, not algorithmic novelty. New
trials identify harness revision 2 and preserve the original feedback gains.


## 2026-10-29 experimental narrowing

The current candidate is conditional last-inch SC insertion under an upstream
angular-alignment contract, using learned pose proposals, public-CAD mouth/body
refinement with image-time robot self-masks, and bounded tracking compensation.
It does not yet solve arbitrary visual grasp rotation. The original learned
models remain proposals; replacing their grasp translation with a design
constant is not part of this method. Calibration intrinsics, robot kinematics
and public local geometry are normal model inputs; live object/port truth is
not a policy input.

The useful research question has narrowed to an interaction: good image fit
alone does not ensure physical readiness, and compliant force-aware control
alone does not remove image bias. The live body fit achieved 0.0465 mm initial
translation error yet insertion failed with 0.375 mm median lateral tracking
gap. The proposed four-way development comparison isolates body refinement and
tracking compensation under shared native gain, grasp condition and force limits.
These are engineering components with substantial prior art. Any novelty claim
must be about the demonstrated task-specific formulation and measured interaction,
not first use of vision/force fusion, CAD fitting, masks, gating or PI control.
There is still no qualified learned-pipeline insertion or final robustness result
at the moment of this entry. Replace this status only with actual later evidence.

## 2026-10-30 measured update

The body-refined visual method with bounded lateral PI subsequently qualified
three identical nominal repeats, with independently verified geometry/contact.
The first varied target exposed a cable-occlusion defect in the CAD rim image
fit; two factorial attempts rejected perception and the batch was stopped.
An image-only cable-mask repair passes recorded-image diagnostics and 179
regressions; its physical qualification is pending. This supports reporting
visibility and tracking as separate failure mechanisms, but does not establish
a causal body/tracking interaction or randomized robustness. Those require the
same frozen-version matched comparison. Cable colour masking itself adds no
generic novelty claim. Final seeds remain unused.

## 2026-10-30: loss experiment and claim limits

The completed v14 screen was 1/5, with three port-perception rejections and one
learned/prior angular-consistency rejection. A longer mouth adaptation passed
the original neural geometry and CAD gates on all five archived live captures;
a prospective physical batch is testing it with the original plug model.
Static replay does not establish fresh consensus or insertion reliability.

Standard pose metrics were a poor checkpoint selector for this task. The new
plug run's standard-score best checkpoint had 2.709 deg median 3D angular error
on six existing static development scenes; the final-epoch checkpoint had
1.086 deg, compared with 1.310 deg for the original model. Neither establishes
0.25 deg accuracy or robust physical insertion. Ground truth scores these
offline predictions only after their fits are saved.

`task_edge_keypoint_loss.py` now tests a training-only objective combining
non-saturating coordinate error with the signed projected direction error of
the twelve cuboid edges. Its angular scale is the unchanged 0.25 deg task
tolerance; the coordinate scale is approximately half that angle times apparent
object extent. This scaling is a training objective, not a certification of
3D error. Three mechanism/gradient tests passed. The original head/checkpoint
format and runtime acceptance/physical verification thresholds are preserved.
The experiment is running; no improvement or originality result is claimed.

Primary literature already addresses the mismatch between correspondence losses
and final 6D pose: [Hu et al., CVPR 2020](https://arxiv.org/abs/1911.08324).
[Zappel et al., 2021](https://arxiv.org/abs/2107.02057) use keypoints and part
affinity fields for pose estimation. [Li and Schoellig, ICRA 2023](https://arxiv.org/abs/2303.16833)
combine multiview keypoint information and geometric confidence for rejection.
Consequently, geometry-aware supervision, edge relations, multiview confidence
and rejection are established ideas. Any eventual contribution must be the
specific tolerance-scaled formulation and demonstrated task-level effect under
matched controls, together with the audited visual-to-physical error budget.
It must not be framed as first use of these generic components.

## Completed diagnostic results, 2026-10-30

Longer mouth adaptation completed 2/5 guarded insertions; all three remaining
failures were learned angular-prior vetoes. A separately predeclared veto ablation
completed 5/5 physical seats under the upstream-alignment assumption. It does
not repair unrestricted visual rotation or certify that assumption. This
supports an observability split between conditional translation correction
and upstream angular readiness; it does not prove a novel general estimator.

Projected-edge and affine-rotation training losses failed to improve static
rotation reliably (best/final medians 1.960/1.625 and 1.986/1.825 deg). Tighter
native crops selected on training data improved static but worsened recorded
live rotation. Supersampled full-CAD fitting still failed cross-view checks.
None is a positive novelty result. The matched conditional body/tracking
comparison is now running and must supply the causal component evidence.

The paper draft has been rewritten around current conditional SC evidence;
the legacy SC/SFP draft is preserved. The defensible possible contribution is
an audited task-conditioned perception-to-seating formulation with measured
failure mechanisms and component interaction, rather than a new fusion, PI,
loss or gating method. Broader randomized coverage and final frozen trials
remain required. Do not manufacture novelty by renaming established components.

## Rigid-grasp observability, 2026-10-30

Exact-stamp independent diagnostics now show that a physically invariant grasp
can acquire 1.275-2.074 deg estimated rotation change after retreat. Since all
three cameras are rigid to the held connector, retreat can improve the port
view but cannot diversify the connector-relative viewpoints. A targeted offline
candidate suppresses context in the native refinement pass using RGB proposals,
observed white pixels and robot self-masks, retaining the original first detector.
It must pass absolute 3D and rigid-invariance qualification before any live trial.
Colour masking, separate refinement networks and rigid-motion consistency are
not individually novel. A positive task-level effect, if measured, could support
the engineering contribution about observable correction and physical tracking;
there is no positive outcome or first-use claim at this point.

## Measured component interaction and final evidence boundary

The matched component study is complete: full conditional SC body + PI 5/5,
no body 0/5, no PI 0/5, with identical frozen source/checkpoint maps. The
independent trajectories support distinct position-estimation and compliant-
tracking errors; controller force feedback alone does not repair either.
The fresh wider/unknown-grasp screen finished 9/10, including one shoulder-impact
force abort despite good physical alignment. These outcomes support a practical integration
contribution within the explicit nominal-alignment contract. They do not
establish a new generic PI, multiview, masking or uncertainty algorithm.

The foreground/side-view/phase-averaged rotation candidates remain rejected:
median recorded-image error down to 0.306 degrees is not a 0.25-degree accuracy
certificate, and not all rigid-consistency checks pass. No new angular recovery
claim is warranted. A broader motion or unstructured angular task requires
additional perception work after the frozen conditional study; do not tune on
its final trials. The all-ten-pass readiness gate failed and the final worker
exited without opening any final case. A source-frozen braking repair and new
development validation are required. Hardware transfer,
SFP perception, release retention and latch/electrical validation remain open.

## Persistent-reference repair and force observability

A complete raw stream exposed impacts missed by the 50 Hz guard. V20 fixes a
setpoint-chasing defect and adds settling distance; two known failures now pass
full geometry/contact and 2000 Hz force audits. This supports a practical
end-to-end error/verification contribution, not a claim that persistent
trajectories, braking, PI or high-rate sensors are individually novel. Nine
unexecuted development cases and the source-gated final comparison remain
pending. The original guarded angular policy still has its earlier 2/5 result;
no unrestricted rotation/SFP or hardware claim follows from these repairs.
