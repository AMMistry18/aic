# Original-pipeline repair evidence, 2026-10-28/29

This record concerns development. No final 100000-series trial was used.
The complete original learned pipeline has no verified insertion success yet.
The current checkpoint and subsequent physical trials are recorded in
[visual correction development](VISUAL_REPAIR_20261029.md); pending statements
below describe the earlier repair stage.

## Executed corrected-grasp SC trials

These versions have different execution conditions; do not pool them as a
success-rate estimate. Exact records are in
`experiments/evidence/original_repair/executed_trials.json`.

| Run suffix / method | What happened | Verified insertion |
|---|---|---|
| grasp29v2 / original | Correct SC source grasp yielded three-view 1.29 px plug priming; degraded port geometry and untared-controller drift then caused failure | No |
| tare29v1 / original | Controller tare and shared load support restored; plug primed; no multiview port consensus; zero commands | No |
| view29v1 / reacquire | Bounded retreat moved 9.94 mm with 0.765 N peak tared force; 0.292 mm endpoint error failed unchanged 0.25 mm gate | No |
| view29v2 / reacquire | Bounded tracking correction passed endpoint gate at 0.1763 mm, 0.6976 N peak tared force; re-primed plug at 1.93 px; rejected inconsistent rigid mouth geometry | No |

The fourth trial's independently recorded maximum global penetration was
0.0000437 mm. This qualifies its noncontact retreat component, not insertion.
Actual camera footage is `experiments/evidence/original_repair/video/sc_visual_retreat.mp4`.
It has causal timestamps and repeated sparse captures, with no interpolation.

## Diagnosed causes and code changes

1. The former SC reset used an SFP grasp. `challenge_grasp.py` derives the
   connector-specific SC setup from the source reversed cable, documented
   world-axis spawn offset and robot HOME. This setup-only transform is never a
   policy input. Its source-frame composition agrees with independently
   expanded libsdformat frames to 1e-12. Original SC weights then pass the
   unchanged reprojection gate.
2. The challenge engine's controller tare was omitted by the last-inch harness.
   The original pose helper's 0.5 force-feedback gain reacted to static payload.
   Harness revision 2 restores tare and shared load support using the native
   `ff + gain*(ff-sensed_tared)` equation. Rejected commands now propagate and
   TCP travel is bounded to 80 mm.
3. The original calibrated gripper masks were present under
   `flowstate/aic_perception` but missing from the research Python path. The
   plain Python helper is now exposed without requiring the private platform.
4. The new optional quality method requires consistent mouth dimensions and
   rigid-corner reprojection, strict temporal consensus, and one bounded 10 mm
   visual retreat. It remeasures the grasp and port. A 0.5 mm bounded tracking
   offset corrects changing cable load; the endpoint and force gates are intact.
5. Harness revision 3 switches physical controller intervals to ROS simulation
   time while retaining the overall wall-clock deadline and communication
   watchdogs. It also ignores scoring events and uses a one-second policy-
   observable seating candidate dwell. SFP event-pad overtravel is removed for
   this mode. Estimated geometry is never independent proof: external source-
   geometry, contact, penetration and force verification still decides success.
   These revision-3 changes passed tests but have not had a new physical trial.

The controller/clock/terminal/command tests passed 200 cases; the separate
native geometry/evaluation/perception selection passed 79 tests before the
latest controller-only changes.

## Failed diagnostic alternatives retained

SC mouth crop pads 4, 6, 12 and 24 did not repair a 2.072 mm local center error.
After retreat, a candidate with 2.59 px center reprojection still had 6.933 mm
center error and 9.65 px rigid-corner reprojection. This is why a center-only
check must not authorize contact. Ground truth was attached only after fitting.

Public-mesh outline fitting was tested offline. Accounting for an inferred rear
occlusion plane reduced mouth center errors to about 0.55-0.58 mm, but image
residuals remained 2.31-2.39 px against the prototype's 2 px gate. The plug
outline prototype was worse (1.043 mm / 5.657 degrees, 10.37 px support error).
Neither prototype is enabled in the policy. These are failed development
hypotheses, not insertion improvements or calibrated uncertainty estimates.

## Fresh transfer data and compute

A frozen collection schedules 20 training and six validation scenes, split by
entire randomized scene/grasp, with all five targets, 7-24 mm gaps, lateral and
angular perturbations, and prescribed-grasp perturbations. It uses actual
headless source meshes and calibrated wrist images. Static captures are
training data, not physical trials. The early URI/rest-shape/floating-link
capture failures were preserved and excluded; the qualified collection uses
fixed source geometry, camera color checks and occluded-mouth screening.
The isolated renderer is deliberately terminated after pixels/calibration are
copied to avoid an observed Ogre teardown heap error; this protocol is recorded.

Collection progress: `.artifacts/perception_data29/qualified/progress.json`.
Training/evaluation progress: `.artifacts/perception_data29/transfer_progress.json`.
The collection completed with 20 training and six development scenes. Four
separate 12-epoch checkpoints completed: standard transfer and precision-loss
ablations for mouth and plug. Original files are untouched. Mouth detection
recall rose from 41.2% to 97.6% with standard transfer; center pixel medians
remained 7.58 versus 7.06 px. Standard plug transfer worsened median keypoint
error from 2.23 to 3.31 px. Static multiview mouth error improved from 6.241 to
1.636 mm; using both standard transferred models reduced median relative
lateral error from 4.433 to 1.315 mm. Precision-loss variants were mixed, not
a demonstrated improvement. These six scenes were subsequently used during
repair and must now be described as development data.
Pixel validation alone will not establish 3D accuracy or insertion robustness.
The next required evidence is matched multiview accuracy and actual corrected-
physics insertions with the new models, followed by frozen randomized trials.

The RTX 4070 8 GB was successfully enumerated through Docker. Automatic review
blocked the proposed workspace-mounted training container and subsequently the
GPU framework installer; no CUDA training environment was installed. The
queued transfer uses the established isolated CPU PyTorch runtime. An empty
isolated container `research-aic-train` exists but is not running work.
