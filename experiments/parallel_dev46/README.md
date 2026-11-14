# Short speed and angular diagnostics, November 13, 2026

This is development work conducted while frozen final44 runs. It adds **no
verified insertions**. All 110 final source/checkpoint hashes still match.
Final cases were not used to select, fit or tune these candidates.

## Measured speed and concrete runtime repair

`cropped_device_benchmark.json` replays three historical development exposures
per connector through the original full-frame proposal, padded crop inference,
confidence-weighted multiview geometry and unchanged acceptance gates. It uses
the original SC plug checkpoint and recovered20261112 SFP plug checkpoint.
No simulator truth is loaded. Two warmups precede three timed repetitions.

| Connector | Cases | Median CPU seconds | Median GPU seconds | Matching decisions |
|---|---:|---:|---:|---:|
| SC | 3 | 5.7431 | 0.1133 | 3/3 |
| SFP | 3 | 5.3209 | 0.1230 | 3/3 |

CPU had a one-core container quota and one Torch thread, sharing the laptop
with final44. GPU was an RTX4070 Laptop, FP32. These are runtime observations,
not isolated hardware throughput or robot insertion-time improvements. This
step excludes port detection, CAD rim/body fitting, image acquisition and
motion. Five poses were accepted and one rejected on both devices. Maximum
CPU/GPU pose differences among accepted pairs were 2.090 micrometres and
0.008277 degrees; this small sample cannot guarantee equivalence near a gate.

A real library interaction was found: Ultralytics `select_device('cpu')`
sets Torch to eight threads on first initialization, overriding the harness's
earlier setting. Under the one-core quota, the first SC replay measured
17.5580 seconds before correction versus 5.7431 seconds after restoring one
thread following warmup. The initial observation is retained separately in
`cropped_device_benchmark_initial_thread_observation.json`; its inflated ratio
must not be treated as the corrected CPU/GPU benchmark.

The runtime candidate uses the prediction-start callback, which runs after
backend initialization, to enforce the explicit thread budget for full-frame
and crop calls. Real CPU initialization/inference passed in
`thread_callback_verification.json`. The integrated copied modules then passed
all four detectors (SC plug/mouth and SFP plug/NIC), each retaining one thread;
mouth and NIC detections each found five instances in the original saved
full scene. See `thread_candidate_integration.json`.

The candidate is installed **only** in `.artifacts/dev46_workspace`.
Reviewable helper, patch and hashes are in
`experiments/candidates/efficiency46/`. Checkpoints, correspondence weights,
geometry, thresholds, motion and force settings were not changed. Physical
SC/SFP qualification of this runtime revision remains unexecuted.

## Angular tests: improvement is not yet sufficient

The first four old training/development image sets per connector are used;
these are not new robustness trials. Ground truth is accessed only after
fitting, for diagnostic error measurement. The learned crop predictions feed
a confidence-weighted joint SE(3) reprojection fit, compared with original
weighted triangulation/Kabsch.

| Connector | Cases improved | Median original error | Median joint error | Within final angle budget |
|---|---:|---:|---:|---:|
| SC | 4/4 | 1.9188 deg | 0.5141 deg | 1/4 at 0.25 deg |
| SFP | 1/4 | 2.3658 deg | 1.6494 deg | 0/4 at 0.10 deg |

The static collector saved different first exposures for the three cameras,
1.5-7.5 seconds apart. The production synchronization guard rejected **all
eight** sets. Their timestamp rejection remains intact; pure geometry is
diagnosed only because the collector froze bodies and cameras. No timestamp
was relabeled and no accepted runtime angular pose is claimed.

Six-DOF SC CAD body-outline fitting was also attempted on all four SC sets.
All four failed the existing leave-one-view-out modeled displacement check,
despite 0.120-0.163 pixel support residuals. Low image residual alone does not
certify angular accuracy. These fits will not command robot rotations.

Data: `angular_inputs.json`, `angular_reprojection_results.json`,
`body_pose_results.json`.

## Direct CAD feasibility check for another angular datum

`angular_contact_feasibility.json` evaluates pinned source collision boxes at
2 mm and 10 mm penetration, two tilt axes and signed angles from 0 to 2 deg
(26 configurations). This is analytic collision geometry, **not physics or
contact validation**. The proposed uniform-channel two-depth wall-center
formula is biased on one SFP axis: even the zero-tilt case produces a
-56.857 micrometre center shift, equivalent to a spurious -0.81436 deg tilt.
The source channel is not uniform with depth. The other tilt axis follows the
uniform formula in these single-axis cases; combined tilt/roll is untested.

The analytical intervals passed 24 independent checks using the existing
scalar separating-axis implementation: bore midpoint clear and points
1 micrometre outside each boundary colliding, at zero and +2 deg on both axes
and both depths. See `angular_contact_scalar_validation.json`.

Noise analysis is hypothetical, not measured: two independent 10 micrometre
center standard deviations over an 8 mm depth separation imply roughly
0.397 deg angular 95% half-width under the uniform model. A usable contact
angular estimator needs the actual CAD geometry and measured uncertainty;
the naive formula is not being installed.

## Parallel simulation resource limit and final-study accounting

Five SC short-probe setup attempts were retained; none reached a qualified
policy probe. V1 had shell CRLF, V2 missing copied CMakeLists, V3 controller
activation timeout, V4 a world-control reply timeout while loading meshes.
V5 was deliberately stopped before policy to prioritize final44 recording.
The last development scene limited background visuals only, retaining actual
active meshes and all collision geometry; it is not the full robustness scene.

During overlap, final SC700045 failed raw force coverage with a 0.3605 second
gap. Contention may have contributed; causation is not established. It remains
a failed final attempt, with no rerun or replacement. Concurrent rendering
also makes wall timings shared-laptop observations. The second Gazebo process
was stopped; only bounded offline diagnostics continued. The short physical
driver now refuses to start while `research-aic-eval` is running. The frozen
main study, full five-position scene, policy and verifier were not changed.

Resume actual short physical qualification only when it will not compete with
the final recorder. Use a new development label and unchanged physical checks.
Any selected new policy/runtime needs a new frozen cohort; do not pool it with
final44 or tune it using final outcomes. See `measured_summary.json` and the
separate final44 ledger for the current partial accounting.
