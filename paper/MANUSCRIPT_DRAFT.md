# Task-Conditioned Visual and Contact Registration for Last-Inch SC and SFP Insertion

Working combined manuscript, November 5, 2026 local. Intended four-page main
body; layout has not been typeset or page-verified. Development results are
separated by revision. Held-out results and matched current-method ablations
remain pending. Previous draft preserved in
`MANUSCRIPT_DRAFT_before_finish41_20261105.md`.

## Abstract

Last-inch connector insertion requires errors to be corrected in coordinates
that available sensors can actually measure. We adapt an existing learned,
confidence-weighted multiview pipeline to SC and SFP insertion in a headless
Gazebo workcell using public challenge assets. Learned keypoints initialize
image-based geometric refinement; connector-specific contact probes register
relative coordinates that remain uncertain under occlusion. SC uses a terminal
shoulder, whereas SFP uses an entrance lip because its nominal seated region has
no rigid axial stop. Bounded robot tracking compensation then executes the
insertion. An independent recorder evaluates stable geometry, contact validity,
penetration and full-rate wrist forces. Development testing exposed visual
bias, missing contact evidence and exposure-time transform expiry. Retaining
transforms at acquisition repaired all three selected transform failures and
preserved one prior success: four independently verified SFP seats, with raw
force peaks of 3.00–3.18 N. Earlier SC development achieved 10/11 seats under a
different revision. These results demonstrate conditional feasibility; they do
not yet establish combined-method robustness or a causal improvement over
matched baselines.

## I. Application and scope

Electronics assembly combines small clearances, occluded mating surfaces and
compliant robot motion. A low reprojection residual alone does not guarantee
that a held connector will enter its requested port. Our AI for Industry
Challenge system used learned plug/port keypoints, confidence-weighted
multiview geometry and guarded insertion, with Flowstate supplying upstream
execution [1]. We isolate its final insertion skill and study failures during
transfer to an inspectable simulation environment.

The task starts with an already grasped connector near its requested port.
Upstream angular alignment and a rigid grasp are prescribed; the actual grasp
transform is estimated from RGB rather than supplied to the policy. We study
translation correction and physical insertion, excluding pickup, long-range
transport and arbitrary rotational recovery. The public level fixture family
is an additional SFP prior. Success is therefore conditional on the declared
handoff rather than a complete reproduction of challenge automation.

Active external-contact estimation already supports insertion [2], and
synthetic perception combined with visual servoing and compliant connector
insertion is established [3]. Our candidate engineering contribution is a
measurement allocation strategy: visual observations support observable
components, connector-specific contacts resolve remaining relative coordinates,
and unsupported estimates cause abstention. Matched component removals must
still establish its benefit. Confidence weighting and generic vision/force
fusion are established techniques.

## II. Perception and contact-guided execution

**Learned proposals and geometric refinement.** We retain the original learned
keypoint models and confidence-weighted multiview geometry. SC retains the
original plug checkpoint and adapts the mouth model; SFP adapts the original
NIC and plug checkpoints on separate rendered training scenes. Original and
adapted checkpoint hashes and training lineage are recorded. Confidence weights
weight geometric residuals; they are not treated as calibrated uncertainty.
Simulator labels are used offline for adaptation, never as live object-pose
inputs.

For SC, learned proposals initialize occlusion-aware CAD mouth/body image fits.
The original full-translation validation gate remains active. A lateral-only
candidate can support a restricted control update only after agreement from
another exposure and subsequent axial contact registration. It cannot certify
full visual pose or insertion depth. For SFP, the public two-port NIC geometry
provides eight rim corners and a 23.2 mm pitch. Stereo-supported proposals are
associated with the held plug inside a bounded approach envelope. Directional
image edges refine the two entrances while excluding the held plug footprint.
The level-fixture prior constrains tilt; the controller preserves the aligned
wrist orientation instead of commanding a noisy learned grasp rotation.

**Exposure-time consistency.** Intrinsics and camera/robot transforms correspond
to actual image exposures. A development failure arose when slow inference and
CAD fitting outlasted TF history retention: an initially available exposure-time
TCP transform was requested only after it had expired. The repaired pipeline
copies that transform at acquisition, before expensive computation, and later
matches it to the selected image stamp. Missing transforms still reject the
measurement. A separate unchanged stationarity check limits motion during
processing to 0.10 mm and 0.05 degrees. Neither latest-pose substitution nor
timestamp relabeling is used.

**Connector-specific relative datums.** Contact is interpreted in the relative
plug-to-port coordinate system. SC provides a terminal shoulder near 15.64 mm
depth. Slow load/unload/reload looks for continued command advance, stationary
measured robot motion, increasing positive compression, measured release and
repeatable stops. SFP has no equivalent hard stop at its nominal 45.8 mm
clearance seat. Opposed small lateral offsets instead touch its entrance lip
near a public 53.5 µm axial datum; consistent opposed measurements register
depth. At shallow engagement, opposed side-wall probes register lateral
coordinates.

SFP search intent is retained separately from the current stationary measurement
window, so an isolated sampling gap cannot erase an already executed bounded
search. Search readiness does not accept contact: one fixed reversible force
excitation must still produce a distinguishable wrist response and release.
Free motion, insufficient evidence or inconsistent opposed probes cause
abstention. Policy contact decisions use robot motion and tared wrist force,
never simulator contact flags. SC force-noise discrimination remains an explicit
validation item; current evidence does not establish equivalent rejection
properties for both connector-specific detectors.

**Execution.** Positive compression is the negative insertion-axis projection
of tared wrist force. Persistent bounded setpoints drive the AIC Cartesian
impedance controller. Own-TCP lateral PI compensates tracking bias, freezes
during contact probes and has bounded authority. Axial lead compensates
damping, with early braking before the seat. SFP's current travel profile is
10 mm/s away from the terminal region and 2 mm/s near the seat. The prospective
20 N outer ceiling does not replace the existing 3 N travel pause or 6 N inner
abort. All proposed ablations retain protection and independent verification.

## III. Testbed and independent verification

The workcell uses public phase-1 asset revision
`7148df9ab62b4b550b101189098f05bc0dba8444`: five NICs with two SFP entrances
each, five SC duplex mounts, staging fixtures, cables, UR5e and ATI sensor.
Visual and collision assets are retained. Gazebo runs headlessly with recorded
camera images. A source-audited double-precision Bullet backend uses a 0.5 ms
step and 200 solver iterations, including declared cable-joint damping. The
grasp attachment is explicit; automatic port attachment is disabled during
verification. These simulation assumptions accompany every physical claim.

Production observations are RGB, calibrated intrinsics, exposure-time camera
extrinsics, own-robot kinematics, wrist wrench and public CAD. Independent
simulator object poses are available only to the evaluator and diagnostic
comparisons. They do not initialize or correct the production policy. Current
handoffs are narrowly aligned: angular components within ±0.1 degrees for SC
and ±0.03 degrees for SFP; the declared unknown-grasp translation variation is
within ±2 mm per component. These bounds do not test general angular recovery.

Verification requires a continuous one-second terminal dwell. SC depth must
remain within 15.64 ±0.25 mm, lateral error within 0.25 mm and full angular
error within 0.25 degrees, with independently observed requested-port seating
contact. SFP requires 45.8 ±0.25 mm depth, 0.10 mm lateral and 0.10 degree
angular error. Its nominal seat is a clearance region, so contact validity is
supported by a deliberate lateral-miss positive control with the identical
collision contract. Both require healthy contact observations, sampled
penetration no greater than 0.10 mm and complete every-step 2000 Hz raw wrist
coverage. Policy completion alone does not count as success. These tests
demonstrate a held geometric seat, not electrical engagement, retention after
release or latch operation.

## IV. Development measurements and failure analysis

The following are development cohorts with distinct source revisions and
protocols. Selected repairs are not pooled with earlier successes to construct
a larger success rate.

| Cohort | Connector | Verified / attempted | Interpretation |
|---|---|---:|---|
| V25 broad development | SC | 10/11 | Earlier 8 N protocol; not current-version qualification |
| V37 qualification39v1 | SFP | 7/10 | Three exposure-TF failures before insertion motion |
| V38 exposure40v1 | SFP | 4/4 | Three selected failed cases repaired, one prior-pass regression |
| V38 qualification41v2 | SC + SFP | Pending | Fixed 11 SC /10 SFP development cases; no held-out cases |

The V37 failures requested transforms at 13.2, 12.8 and 13.0 s; the earliest
remaining TF samples were 13.31, 14.2 and 13.002 s respectively. All three
failed before insertion commands. This identifies acquisition/processing
coordination as the observed failure, rather than poor physical alignment or
insufficient commanded force. After the acquisition-time transform repair,
all four selected V38 runs passed unchanged independent seat criteria and
complete 2000 Hz force audits. Peak tared raw force ranged from 2.9993 to
3.1833 N. Median action time was 58.9248 simulated seconds; median recorded
wall time was 565.4294 s. The descriptive Wilson 95% interval for 4/4 is
51.0–100%; selection on development failures prevents interpreting this as
population reliability.

Other development failures explain the method design. Mouth occlusion and
inaccurate learned rotation made full-pose visual corrections unreliable.
Geometry-only completion could pass depth tolerances without actual SC seating
contact. An incorrect compression sign rejected a real SFP lip probe. Later,
a timing gap consumed the only command-advance window and stalled lip-search
readiness; separating executed search intent from the current measurement
window repaired that mechanism without accepting gaps as valid contact data.
Failed trials and source snapshots are retained rather than overwritten.

A selected same-seed speed comparison increased measured interior SFP mesh
speed from 5.0068 to 10.0393 mm/s and shortened the travel segment from 11.6505
to 6.8570 simulated seconds. Whole-action time fell only from 62.4860 to
58.3065 s. No 3 N travel pause occurred in those completed comparisons;
perception and contact-probe overhead remain substantial. This is a development
comparison across revisions, not a throughput or superiority claim.

The fixed same-version SC/SFP screen must finish before broad consistency can
be assessed. Planned matched controls retain probes but discard axial
registration, or remove lateral tracking compensation. These comparisons are
not yet executed for the current combined method. Final seeds remain unopened;
final tables must report per-connector success intervals, all-attempt and
successful-only times, raw forces and failure causes.

## V. Discussion

The current evidence supports genuine conditional insertion and identifies
specific failures in the path from an image to a physical seat. The method
assigns uncertain coordinates to visual or contact measurements according to
connector geometry. Its practical value and component effects still require
matched evaluation. Simulation contact modeling, upstream angular alignment,
rigid grasp and SFP's level-fixture prior limit generality. This draft supports
writing and review; a final robustness conclusion awaits the frozen experiments.

## References

[1] Team TAR-2-SOL, [AI for Industry Challenge solution](https://satyaa27.github.io/intrinsic-ai-for-industry-challenge-team-tar-2-sol/).
Method source and checkpoint provenance: `ORIGINAL_PIPELINE_TRACEABILITY.md`.

[2] S. Kim and A. Rodriguez, [Active Extrinsic Contact Sensing: Application to General Peg-in-Hole Insertion](https://arxiv.org/abs/2110.03555), ICRA, 2022.

[3] [SyPAS: Synthetic Socket Perception and Visual Servoing for Robotic Connector Insertion](https://sypas2026.github.io/), authors' project page, accessed November 5, 2026.

Internal measurement provenance: `experiments/evidence/exposure40/summary.json`,
`experiments/evidence/exposure40/trials.csv`,
`.artifacts/repair39/qualification39v1/progress.json`, and
`.artifacts/finish41/qualification41v2/progress.json`. These are research records,
to be replaced by a frozen supplementary archive in the submission.
