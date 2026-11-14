# Combined SC/SFP robustness study and four-page paper

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

Decision date: November 2, 2026 (America/Chicago). This is a prospective plan;
planned sample counts are not results. Current measurements are in
`../PAPER_RESULTS.md` and `.artifacts/repair34/validation34v1/progress.json`.
Preserve the running V25 source/checkpoint freeze.

**Evening update, November 2:** the V25 batch has finished: SC 10/11 and SFP
4/10 verified, all with complete 2000 Hz force coverage. Its original frozen
archives are preserved. The qualification gate below failed, so final cases
remain closed. V26 `waypoint35v1` is a separate development repair regression,
not an extension or relabelling of V25. Detailed measured failures and
intervals are at the top of `../PAPER_RESULTS.md`. Earlier partial counts in
this plan are dated planning snapshots.

## Central question and candidate contribution

Can an existing confidence-weighted, multiview learned perception pipeline
perform reliable last-inch insertion when the controller corrects only the
components supported by its measurements, and obtains the remaining relative
translation from connector-specific contact datums?

The candidate contribution is the explicit measurement contract and tested
allocation of visual/contact correction: SC uses its terminal shoulder; SFP
uses opposed entrance-lip and side-wall measurements because its nominal seat
is a clearance region. Known upstream angular alignment is assumed. A visual
lateral candidate never certifies unmeasured axial depth or full 6D pose.
Own-TCP tracking and an independent physics/force verifier distinguish an
estimation success from actual insertion. This must outperform matched controls
before it becomes an improvement claim.

Confidence weighting, CAD fitting, PI control and active contact sensing are
established techniques. [Active Extrinsic Contact Sensing](https://arxiv.org/abs/2110.03555)
estimates extrinsic contact for insertion; [SyPAS](https://sypas2026.github.io/)
combines synthetic visual learning, servoing and compliant connector insertion.
Our narrower task-conditioned formulation and engineering evidence must be
distinguished from those methods. No state-of-the-art or generic vision/force
novelty claim is justified by the current nominal successes.

## Evaluation sequence

1. **Complete the active development batch.** Eleven SC regression cases and
   ten varied SFP cases. Current snapshot: 3/3 completed SC pass; varied SFP
   has not started. The known nominal SFP pass remains a separate 1/1.
   Preserve all failures and complete the frozen batch before changing code.
   Do not describe selected SC regressions as an independent population test.
2. **Repair development defects if any.** A changed policy is a new revision;
   keep its denominator separate. Repeat known regressions and add fresh
   development cases before returning to qualification. Check any SFP timeout
   against perception/probe/translation/terminal stage times: its nominal
   duration is already 717.89 wall /72.34 simulation seconds including startup.
   Improve the implementation rather than increasing success tolerances.
3. **Qualify controls and contact-negative cases.** Require free-air/no-lip,
   shallow wrong/early blockage, and contact-enabled positive controls. A stop
   without the required wrist reaction, repeatability or release must abstain.
   Preserve actual meshes, collision instrumentation and unchanged thresholds.
4. **Implement matched controls after the active freeze ends.** Use the same
   adapted weights, calibrated images, proposal/refinement, retry logic,
   handoff prior, nominal trajectory, force guards and action budget.
5. **Freeze the final protocol and execute once.** Minimum 50 independent
   scenes per connector, paired across three methods: 300 final trials.
   No final case is generated or opened before qualification. Never tune,
   replace failed cases, or enlarge the sample count based on interim results.

Development readiness for opening final cases: all 11 current SC and all ten
current SFP qualification cases pass; force coverage is complete and every
raw peak <=20 N under the newly labelled prospective protocol; all five qualified SC target mounts and all ten SFP slot
identities have a verified insertion; contact controls pass; matched ablation
implementations are reviewed and piloted; sources/checkpoints are frozen.
If a qualification fails, keep final cases closed. This gate does not turn
development results into a reliability estimate.

## Three principal methods

| Method | Controlled difference | Interpretation |
|---|---|---|
| Full | Apply RGB refinement, measured contact datums and bounded own-TCP tracking | Proposed conditional system |
| Discard axial correction | Execute the same axial probes and validation, but do not apply their estimated axial coordinate correction | Isolates use of the axial datum; does not claim a faster vision-only baseline |
| No lateral PI | Apply the same visual and contact corrections, but omit only the additional own-TCP lateral PI term | Isolates tracking compensation |

These matched variants are **planned, not implemented/executed**. Existing
method names often also change fresh-image retries, partial-pose eligibility
or terminal logic; they cannot simply be relabelled as these controls. In the
axial-discard variant, keep probe measurements/validation and partial-candidate
rules identical, record that the coordinate update was deliberately omitted,
and let the unchanged external verifier judge the outcome. Keep all force
protections. A measurement-discard control isolates incorporation of evidence;
it is not a claim about a production controller that knowingly ignores data.

The faithful original policy/checkpoints remain an additional diagnostic
baseline. Compare original versus adapted perception on the same newly
recorded development images, then use a small physical development screen if
needed. Domain adaptation must not be confounded with the main contact/control
comparison. Native-crop or confidence ablations are included only if their
implementations actually change outputs; saturated keypoint confidences are
not calibrated uncertainty estimates.

## Primary variation and sampling

The proposed final envelope stays inside the declared aligned-handoff task.
Its levels require prospective development validation before final freeze.

| Variable | SC | SFP |
|---|---|---|
| Target coverage | Five qualified target mounts; two repeats per target/severity | Five NICs x both slots; one case per slot/severity |
| Initial lateral radius | 0, 0.375, 0.75, 1.125, 1.5 mm | 0, 0.2, 0.4, 0.6, 0.8 mm |
| Count | Ten independent cases per radius, 50 total | Ten independent cases per radius, 50 total |
| Gap | 9-14 mm | 7-14 mm |
| Initial angular components | Within +/-0.10 degrees | Within +/-0.03 degrees |
| Unknown rigid grasp jitter | Source generator: each translation +/-2 mm; each rotation +/-0.04 rad | Same jitter distribution, connector-specific nominal grasp |
| Fixture/layout variation | Seeded public board yaw and source target-position variation | Same scene generator, both slot identities |

Sample lateral direction and gap independently within each target/severity
stratum. Use the same scene, grasp jitter and sensor/simulator seed for paired
methods; randomize method order within each case. Report the actual achieved
reset distributions and any hold-gate failures. Unknown grasp rotation does
not imply unknown physical insertion angle: the reset supplies the declared
upstream angular alignment, while the policy estimates the grasp from RGB.
Only one currently qualified channel per SC target is covered; do not claim
coverage of every channel in a duplex model. SFP's level fixture family is an
explicit prior, not measured unrestricted angular recovery.

## Additional robustness boundaries

The primary lateral-severity strata already provide a controlled robustness
curve. Add these development diagnostics before making any corresponding
claim; their injection hooks and physical validation are still pending:

| Stress | Prospective pilot levels | Required separation |
|---|---|---|
| Intrinsics error | Focal-scale bias 0 /0.1% /0.25% | Perturb policy calibration only, preserve rendered ground truth |
| Extrinsics error | Translation 0 /0.25 /0.5 mm; separately rotation 0 /0.05 /0.10 degrees | Perturb estimated exposure transforms, not physical object pose |
| Camera delay | 0 /50 /100 ms | Preserve original stamps and age/synchronization gates; count safe abstentions as task failures |
| Missing view | Three versus two cameras | Keep visibility gates; do not assume SC remains observable with two views |
| Partial occlusion | Fixed masks on connector/port ROI, pilot area fractions | Do not change target geometry or use a live truth mask in policy |
| Contact/friction | Optional 0.5x /1x /1.5x source friction | New collision-contract positive control required for each changed physics condition |

Do not run a large unpaired Cartesian product. First locate useful failure
boundaries with development cases. Then, if compute permits, choose one
meaningful calibration or delay axis and predeclare ten independent paired
scenes per severity per connector for full versus axial-discard. This is
120 additional trials for three severities. Wider rotations, arbitrary fixture
tilt, grasp release, electrical engagement and sim-to-real transfer remain
outside the primary claim. A two-camera abstention can be correct behavior;
it is not evidence of successful insertion under camera loss.

## Fixed outcomes and analysis

Primary: independently verified stable insertion within the existing budget.
SC: 15.64 +/-0.25 mm depth, <=0.25 mm lateral, <=0.25 degree angle and actual
seating contact. SFP: 45.8 +/-0.25 mm depth, <=0.10 mm lateral, <=0.10 degree
angle; source clearance and matching collision positive-control evidence.
Both: continuous one-simulation-second dwell, <=0.1 mm observed penetration,
valid contact instrumentation, complete 2000 Hz raw force and peak <=20 N.
This prospective ceiling follows the user's November 3 clarification and
supersedes the earlier 8 N planning limit; historical 8 N archives retain
their original criteria. Freeze the common protocol for every matched method
before opening final cases. Nominal commanded forces remain substantially lower.
No scoring-event-only success. The 6 N policy abort is a sampled pre-command
guard, not a guarantee that physical force cannot exceed it.

Report connector-specific verified counts/denominators, Wilson 95% intervals,
paired success differences and paired confidence intervals, exact McNemar
tests with Holm correction for two controls x two connectors, and effect sizes.
No pooling SC/SFP or policy revisions. Show severity-stratified rates with
intervals; ten cases per level give wide intervals, not a precise boundary.
Report all-attempt budgeted times and successful-only completion times,
simulation and wall time separately, raw-force peak/p95, coverage failures,
retries/abstentions and predefined failure causes. Distinguish environment/hold
failure, stale/occluded perception, association/fit rejection, unobservable
contact, entry jam, deep jam, terminal geometry/contact failure and timeout.
Derive diagnostics from archived observations; never feed evaluation truth
back into a production decision. Preserve all trials and exclusions.

A development aspiration is >=95% observed final success for each connector,
with no force/coverage violations. This is not a guarantee of >=95% population
reliability: for example, 48/50 gives 96% observed success but roughly 86.5-98.9%
Wilson limits. Claim wording follows actual intervals and paired effects.
If the controls perform equally well, do not claim the added component improves
success. A paper may discuss failures honestly; publication readiness is not
equivalent to a universal robustness certification.

## Compute and publication schedule

Current completed SC trials take about 4.4 minutes of recorded policy wall
time; nominal SFP about 12.0, plus reset/bootstrap overhead. At the current
single-world speed, 300 final trials require approximately 41 hours of policy
execution, plausibly 50-60 hours including setup. This is a planning estimate,
not measured final throughput. A 120-trial stress comparison adds roughly
16-24 hours. Keep GPU adaptation and unrelated heavy work stopped for controlled
final timing; log actual machine load and interruptions. Do not reduce physics
accuracy or run conflicting workers merely to hit a schedule.

Working milestones: finish/diagnose the active development run; qualify the
matched controls and negative cases; freeze the final study; execute/analyze;
then assemble figures and revise claims. Leave time before December 1 for
rerendering, paper review and the submission package. There is no promise that
all stages pass on the first version.

## Four-page paper layout

Working title: **Task-Conditioned Perception and Contact Registration for
Last-Inch SC and SFP Insertion**.

| Main page | Content and purpose |
|---|---|
| 1 | Abstract, industrial problem, concise related work, scoped contribution; diagram separates camera policy inputs from independent evaluator |
| 2 | Original weighted multiview pipeline, geometric refinement, component-level evidence contract, SC/SFP datum differences, bounded control and essential simulator setup |
| 3 | Paired evaluation and main quantitative results: success/95% CI, lateral-severity curve, compact time/force table for both connectors |
| 4 | Causal ablations and representative failures, what the old strategies got wrong, practical limitations and short conclusion |

Use two main figures and one compact table. Place detailed meshes/physics,
calibration audits, checkpoint hashes, full per-case logs, extended stress
results and both success/failure videos in reproducible supplementary material.
Historical development counts explain failure mechanisms; the primary results
table must use one frozen final revision. Do not fill final cells with nominal
or development successes. Draft motivation/methods now; finish abstract/result
claims after the paired measurements exist.

Official format: four main pages including figures/tables, up to one page
describing supplementary materials and up to one page of references; double
column IEEE format. Attachments require plain-text descriptions. Sources:
[RA-P author instructions](https://www.ieee-ras.org/publications/ra-p/information-for-authors/),
[collection CFP](https://www.ieee-ras.org/wp-content/uploads/2026/07/CFP-RA-P-Special-Collection-on-Dexterous-Manipulation-and-AI-Advancements-in-Electronics-Manufacturing.pdf).
The public CFP confirms December 1, 2026. The generic author page requests
real-world demonstrations; the user reports collection-specific September 30
correspondence allowing simulation-only validation. Keep that correspondence
with the submission records and describe the simulation scope explicitly;
do not claim the public CFP itself grants that exception. This does not pause
the authorized simulation work or create a new hardware requirement here.

## Readiness decision

Both connectors must have a completed, frozen reliability evaluation, credible
contact controls, measured component comparisons, camera evidence, reproducible
artifacts and claim wording supported by their actual results. Current nominal
and regression successes meet deterministic qualification only. No automatic
paper-ready flag is set by finishing the current 21 development cases.
