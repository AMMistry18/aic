# Efficiency and robustness revision, November 13

The active 100-primary/80-ablation final44 study remains frozen. No final outcome
was used to choose a speed setting. Analysis here uses the historical DEVELOPMENT
cohort only. The failed higher-force speed43 profile remains stopped.

The evaluator container has no GPU device assignment. The separate perception
container exposes the existing GPU. Headless rendering currently explicitly
uses software rendering. This explains substantial laptop wall-time overhead;
it does not explain away the policy's simulated probing time.

Measured successful-development medians (9 SC, 6 SFP):

| Phase | SC simulated s | SFP simulated s |
|---|---:|---:|
| Complete recorded action after configuration |30.166|58.54175|
| Visual initialization |8.306|not separately delimited|
| SC seating, including shoulder registration |21.039|not applicable|
| Shoulder registration subset |8.029|not applicable|
| Opposed entrance-lip registration |not applicable|22.28775|
| Aligned entry |not separately delimited|5.08325|
| Sidewall registration |not applicable|16.96575|
| Final stroke to first estimated depth/lateral goal |not separately delimited|5.0855|
| Final insertion phase, including controller dwell |not separately delimited|6.9185|

Medians of phases need not add to the median whole action. The SFP stroke's
wall duration is not logged directly and has not been invented by multiplying
by a real-time factor. `wall_s` includes configuration and post-action evidence
capture: median configuration alone is 22.3543 s SC and 20.2780 s SFP. The
existing manuscript's wall-time terminology needs correction before submission.
The 1.2-second controller dwell remains distinct from the independent evaluator's
one-second continuous terminal dwell.

A single-CPU-thread FP32 full-frame detector benchmark, five timed repeated
triplets after two warmups, measured SC 0.95329 -> 0.05811 wall seconds and
SFP 3.51046 -> 0.10698 seconds on GPU. Same development images and weight hashes;
detection counts matched in every timed repeat. This is NOT the complete crop /
weighted-triangulation pipeline, does not certify subpixel pose equivalence,
and is NOT an insertion speedup or hardware throughput result. Full raw timing
samples and device details are in `gpu_benchmark.json`.

An uninstalled SC prototype reuses three distance maps across the primary and
leave-one-camera-out fits, reducing nine pixel-map computations to three. Six
replays preserved the primary pose and acceptance reason, but exact diagnostic
parity failed on one. The saved repeat of the original also varies in the last
leave-one-out fit. Both observations are retained; no acceptance threshold was
relaxed and this prototype has not been promoted to the active policy.

## Next executable studies

- Finish the current frozen 50 SC +50 SFP and 80 component-removal trials.
- Qualify a new GPU-backed execution revision on old development cases before
  any new final evaluation. Keep the original weights, views, synchronization,
  quality gates and motion/force limits. Warm one-time models outside the
  per-task timer, but report the configuration cost separately.
- The SC/SFP boundary manifests in this directory each contain 36 unique
  development seeds: four lateral radii x four directions, three tilt levels
  x two axes x two signs, plus eight unknown-grasp/random-handoff cases. They
  are prepared, not executed. Larger angles characterize failure boundaries;
  the current upstream-aligned controller cannot be called general angular
  recovery merely by increasing its initial perturbations. Scoring is unchanged.
- New matched final comparisons must use the SAME learned initialization for
  full, vision-only and generic force/spiral conditions. The old known-grasp
  diagnostic controller is not a fair perception baseline. Remove shoulder
  registration for SC and separately remove lip and sidewall registration for
  SFP. Disabling lateral actuator tracking is a different ablation.
- `next_study_protocol.json` reserves 20 fresh cases per connector for those
  matched comparisons (180 trials). Baseline implementation/qualification is
  still required; no comparison result is claimed. New final seeds are NOT
  generated yet. No tuning or case replacement from final44 outcomes.
- Characterize numerical sensitivity with fixed representative configurations
  and one-factor-at-a-time step/iteration changes. Every numerical regime must
  have its own matching contact control and every-physics-step force coverage.

The first speed target is <=5 simulated seconds for the registered axial stroke;
it is a target, not a result. Complete cycle time will remain a separate main
metric. Probe-path optimization should preserve reversible-contact excitation,
sensor freshness and contact-near slow motion. Raising force/speed everywhere
is excluded after the speed43 failure.

Primary technical references for the revision:

- ROS time semantics: https://design.ros2.org/articles/clock_and_time.html
- Detector inference/batching interface: https://docs.ultralytics.com/modes/predict/
- Force/search context: Chhatpar and Branicky, Search Strategies for Peg-in-Hole
  Assemblies with Position Uncertainty (2001), author manuscript.
- Vision/touch insertion: https://arxiv.org/abs/1810.10191
- Residual robot control: https://arxiv.org/abs/1812.03201

The general ingredients and computational caching are not novelty claims. The
research claim requires the new matched comparisons to show when assigning
coordinates to connector-specific physical datums improves outcomes or cost.
