# Measured speed development, November 13, 2026

November 14 follow-up: the bounded lip-search test below completed and verified
at 47.0710 sim s with 8.881041 N raw peak and complete 2000 Hz coverage.
Its 31 exported evidence checksums pass. This remains a single development
case. Separately, frozen final44 is at 92/100 primary attempts, SC44/46 and
SFP31/46; all 80 ablations are pending. All 110 frozen hashes still match.
SFP robustness remains the main gap. See `PAPER_RESULTS.md` for the retained
failure breakdown and `experiments/checkpoints_20261114/` for the snapshot.

The frozen final44 policy and its 110 source/checkpoint hashes remain unchanged.
New candidates are developed on known historical cases in a copied workspace.
No final-case outcome is used for tuning. The stopped speed43 force profile
has not been restarted. Physical thresholds, full 2000 Hz force coverage and
the 20 N measured ceiling remain mandatory.

## Complete physical runtime qualification

The GPU/bounded-thread runtime completed both known development cases with
independent mesh-pose, contact, penetration, dwell and raw-force verification:

| Connector/case | Stable depth mm | Maximum lateral mm | Maximum angle deg | Raw peak N | Action sim s | Recorded wall s |
|---|---:|---:|---:|---:|---:|---:|
| SC 42000 |15.565526-15.565562|0.054253|0.134644|3.508088|27.6735|300.0661|
| SFP 45005 |45.805618|0.003954|0.020214|3.256220|56.1585|645.2428|

Both audits cover every 0.5 ms action tick. SC has physical shoulder contact;
SFP is a stable clearance seat with the matching deliberate-collision proof.
These are one reused case per connector, **not a randomized success rate**,
electrical/latch verification, or proof of broad angular recovery.

Full five-position assets were retained: five NICs, ten SFP targets, five SC
adapters, five staging fixtures of each type, five cables, UR5e and three wrist
cameras. Headless initial/final target images were inspected; the terminal
view is partly obscured by the physical gripper. Images supplement the
independent geometry/force/contact checks rather than replace them.

Compact evidence is preserved outside `.artifacts` in
`experiments/evidence/runtime47/`: 54 checksummed files, including full raw
wrench, truth/contact traces, policies, calibration, scene/robot descriptions
and camera images. `summary.json` contains actual measurements and source
provenance; `SHA256SUMS.json` authenticates exported files.

GPU alone did not substantially shorten physical action time. The same old
SC development case previously took 28.5220 sim /286.5137 recorded wall seconds;
the SFP case took 57.4580 sim /715.4829 wall seconds. These historical wall
comparisons differ in resource conditions. Current SC configuration alone took
32.5523 wall seconds. The policy still performs seven complete rim fits during
port consensus (53.653 wall seconds for the new SC run) and slow contact
registration. Do not equate a GPU detector benchmark with insertion speed.

## Measured physical SFP motion comparison

The former lateral sweep starts its return from an unreachable commanded
endpoint after a wall has blocked the robot. The new candidate starts each
trajectory at the actual measured robot position, using bounded quintic paths
(2 mm/s peak, 10 mm/s^2 acceleration), while preserving the 0.5 mm pose-lead
bound, 1.5 mm probe amplitude, 3 mm workspace, 6 N guard and every stationary
force/visibility check. It changes neither registration acceptance nor the
seating/force evaluator.

Controller-logic tests used seven old development plateau-span pairs with five
noise seeds each. All 35 synthetic cases remained observable. Median synthetic
registration time changed from 16.66 to 11.58 seconds. Five negative controls
(free motion, one-sided contact, entrapped plug, stale force, overforce) all
rejected. These are **synthetic logic results, not Gazebo insertion results**.
See `experiments/registration_development47/synthetic_replay.json`.

The candidate is installed only in the copied development policy. Both complete
insertions on the same old case 45005 independently verified:

| Condition | Side-wall stage sim s | Complete action sim s | Recorded wall s | Raw peak N |
|---|---:|---:|---:|---:|
| GPU baseline, runtime47v2 |16.8725|56.1585|645.2428|3.256220|
| Measured-start paths, runtime47v3 |11.8295|51.9735|660.3565|4.684120|

This is a 29.9% stage and 7.45% complete-action reduction on **one reused paired
development case**, not randomized reliability evidence. Both cover every
0.5 ms force tick and meet the original independent seating/contact criteria.
The new stable seat is 45.804919 mm deep, 0.005392 mm lateral and 0.020210 deg
angular error. Recorded wall time did not improve; simulator throughput varied.
The improved motion therefore has a physical-time result, not a wall-time claim.
`progress.json` here records the pair. The new camera, full raw-force/contact
and source evidence is preserved in `experiments/evidence/motion48/`, with
29 checksummed files; the baseline remains in `experiments/evidence/runtime47/`.

The SFP final-stroke log reaches its first estimated goal after 5.1130 sim s,
with 6.8710 sim s for that phase including controller dwell. This is estimated
controller timing, not a separate independent seating verdict. The complete
action still takes 51.9735 sim s because perception and contact registration
precede that stroke. Report both phase and complete-action timing; do not
discard contact-measurement time as preprocessing. The original target-camera
frames were inspected: the final SFP view is heavily occluded by the arm and
does not independently resolve the internal seat depth. Geometry, collision
control, dwell and full-rate wrist data remain necessary evidence.

## Perception optimizations tested independently

SC image/CAD reuse preserves all seven temporal observations and all original
fits/gates. It reuses each image's unchanged distance map across its own
leave-one-view-out fits and caches only immutable source CAD across frames.
Twelve historical image triplets produced **bit-identical full diagnostics and
poses**, with the same gate decisions. Distance-transform calls changed from
9 to 3 per measurement; median local CPU fitting time changed from 1.46554 to
1.12224 seconds. Its complete SC insertion `runtime47v4` subsequently passed
all original independent seat/contact/dwell/2000 Hz force checks:

| Condition | Port measurement wall s | Action sim s | Recorded wall s | Raw peak N |
|---|---:|---:|---:|---:|
| GPU baseline, runtime47v2 |53.654|27.6735|300.0661|3.508088|
| Image/CAD reuse, runtime47v4 |45.833|26.1890|282.1937|3.060299|

This is a 14.6% port-measurement wall reduction, 5.36% complete-action sim
reduction and 5.96% recorded-wall reduction on one reused case. Shared-laptop
conditions and one pair do not isolate a population timing effect. Stable
depth was 15.565530-15.565568 mm, maximum lateral 0.061038 mm and angle
0.134707 deg. Headless images show the white plug entering the actual blue
adapter; the arm obscures much of the final view. The strict depth/contact
conclusion comes from the independent trace, not visual inference alone.
Compact source/camera/raw evidence: `experiments/evidence/cache49/` (29 files).
See `experiments/perception_efficiency49/benchmark.json` and
`experiments/rim_followup49_v2/progress.json`. The first continuation stopped
before installing the cache because it compared the original file's UTF-8 BOM
with a clone lacking that marker. The retry verified strictly identical decoded
source before installation, preserving the first failure and both byte hashes.

The original SFP fixture estimator tries every combination of detected NICs
across views. A new epipolar lower bound screens out only associations whose
minimum possible weighted-L1 error already exceeds the unchanged 5 px gate.
The original nonlinear fit remains authoritative. On four actual old learned
NIC detection sets, all **45 originally accepted candidate fits were identical**:

| Scene | Original fit calls | Screened fit calls | Original CPU s | Screened CPU s |
|---|---:|---:|---:|---:|
|45005|200|20|4.13192|0.34038|
|45009|136|11|2.75312|0.22598|
|45002|136|14|2.95159|0.23849|
|45001|88|14|1.85952|0.26086|

The bound also stayed below a known feasible reprojection cost in 1000
synthetic camera/noise checks. This component is **not installed** and does not
fix an originally rejected estimate or establish an end-to-end speedup. Data:
`experiments/perception_efficiency48/`.

An uninstalled camera-workload proposal lowers only three diagnostic camera
rates from 5 to 1 Hz, retaining all three policy RGB cameras at 5 Hz. It changes
no geometry, intrinsics, physics or force sampling. The calculated render
pixel workload falls 40.8%; no latency improvement has been measured. It is
excluded from the motion/cache comparisons above to avoid mixing changes.

## Angular recovery remains unqualified

Recomputing SC contour visibility after pose updates still rejected all four
development examples. SFP silhouette attempts and subpixel contour fitting
also failed the required view consistency or visible-support checks. No such
pose is installed or used for rotation commands. Data are in
`experiments/angular_development47/` and `angular_development48/`.
The previous SC joint-image fit improves some angular errors but does not meet
the 0.25 deg budget consistently; SFP does not meet 0.10 deg. Static data retain
their original production timestamp veto. No broader angular insertion is
claimed. New sensing/visibility evidence is needed before widening the scope.

The actual URDF confirms all three cameras share the rigid wrist mount: their
tool-relative transforms stay constant across 21 joint configurations (maximum
translation change 2.33e-16 m). Moving this wrist cannot change gripper/plug
self-occlusion for a rigid grasp, although fixture/background occlusion can
change. Repeating wrist poses alone is not a way to reveal a hidden plug corner.

An image-only robustness diagnosis on the same eight old static sets found
near-unity learned scores despite several-pixel label errors. Among scores
at least 0.99, SC median/p95 projected error was 2.36/6.36 px (93 points), and
SFP 2.31/9.50 px (96 points). Truth was used only after inference for this
diagnostic; scores are not assumed to be calibrated subpixel probabilities.
Robust six-keypoint consensus still failed independent view agreement in all
eight sets. Conservative robot-mask removal left too few SC keypoints and
changed none of the SFP points. No rejected fit is used for robot rotation.
See `experiments/angular_development50/`. Better visible features or a qualified
additional sensing/contact constraint is required before widening angular scope.

The follow-up ray/triangle depth check uses actual robot CAD, own FK and the
learned proposal depth, rather than a flat image mask. It still excludes enough
SC corners to leave only 4, 4, 3 and 2 triangulated points, below the unchanged
six-point minimum. It excludes no SFP corners. These eight static development
cases retain the synchronization veto and cannot certify an online pose.
`experiments/angular_development52/depth_ordered_results.json` records the result.

A separate two-depth contact Jacobian audit uses the actual nonuniform source
collision geometry and treats tip translation as uncertain. Its six-parameter
inverse is rank-deficient or poorly conditioned in several configurations;
unmodeled axial/roll ambiguity cannot be replaced by a uniform-channel formula.
Hypothetical linearized noise propagation is not measured uncertainty or a
physical recovery result. The existing lip measurement supplies an additional
axial constraint; its real uncertainty would need calibration in any extension.
See `experiments/angular_development52/contact_angle_identifiability.json`.

## Completed bounded lip-search development test

SFP runtime47v3 still spends 21.756 simulated seconds between the first lip-probe
and aligned entry. A prospective candidate changes only the qualified axial
search speed from 0.4 to 1.2 mm/s, retaining the 0.3 mm lead, 3 mm search endpoint,
fresh stationary trigger and one fixed reversible 0/1/0 N force measurement.
The same 6 N observed guard and independent 20 N raw audit remain in force.

Actual controller-logic replay on a synthetic unilateral actuator accepted all
32/32 stop/noise cases for each version and rejected all four negative controls
(absent lip, stale sensor, monotonic force drift, overforce). Maximum lead stayed
0.3 mm. Median axial-logic time was 7.95/4.66 synthetic seconds, excluding
waypoint travel. These are not Gazebo times, forces or insertion results.
`continue_lip51.py` completed dev_45005 against runtime47v3 after the SC slot
released. Runtime47v5 passed the unchanged independent seat, contact-control,
dwell, penetration and full-rate force checks:

| Condition | Lip phase sim s | Action sim s | Recorded wall s | Raw peak N |
|---|---:|---:|---:|---:|
| Measured-start baseline, runtime47v3 |21.7560|51.9735|660.3565|4.684120|
| Faster bounded lip search, runtime47v5 |17.2475|47.0710|569.9130|8.881041|

The phase/action reduction is 20.72%/9.43% on this reused case. Combined with
the earlier side-wall change, action is 16.18% shorter than runtime47v2's
56.1585 sim s. The larger raw peak remains below the unchanged 20 N ceiling;
two 0.5 ms raw samples exceeded 8 N, so the high-rate audit matters. This is
one development observation, not optimized-policy robustness or a randomized
timing effect. Stable depth is 45.805496 mm, lateral error 0.003725 mm and
angle 0.020214 deg. The final headless target image was inspected but is heavily
arm-occluded. Internal seating is established by independent traces rather
than that image alone. All 31 camera/raw/source files in
`experiments/evidence/lip51/` pass their exported checksums. Both continuation
and exporter finished, the development container is stopped and the unchanged
main coordinator resumed. State: `experiments/lip_followup51/progress.json`.

SC's runtime47v2 shoulder registration alone took 8.0335 sim s. A separate
uninstalled proposal (`experiments/candidates/unload53/`) retracts faster while
keeping the original slow press, two-stop repeatability test and force gates.
It has not passed controller or physical qualification and is not in the queue.
Simply speeding the press can saturate its 0.3 mm lead before the required
0.8 s loading window; this would remove the force-rise evidence rather than
make a qualified registration faster.

## Serialization, provenance and storage

`development_slot47.py` observes a completed final-trial boundary, suspends only
the owned final coordinator with its creation-time identity checked, verifies
its container is stopped, then runs development. A separate heartbeat watchdog
stops development and resumes that coordinator if the slot owner disappears.
The final queue resumes in its unchanged order after each slot; no selected
final case is retried or replaced. Current main count at the second slot was
28 completed attempts. Slot leases record the scheduling separately from
scientific outcomes. The first slot stopped before motion for a duplicated
development startup path; its failed attempt was retained and corrected before
the successful v2 tests.

Finite continuation workers:

- `continue_efficiency48.py`: completed the actual detection replay and physical
  SFP measured-start pair; both insertions verified.
- `continue_rim49_v2.py`: checked the original source after UTF-8 BOM decoding
  and queues physical SC cache qualification after all twelve exact replays.
- `continue_lip51.py`: completed one full SFP bounded-lip-search qualification;
  the independent verifier passed and compact evidence export completed.

A failed check stops its dependent continuation. No policy changes are applied
to final44. Subsequent new-policy robustness must be frozen and evaluated
separately. Future development snapshots reuse seven immutable checkpoint
objects by hardlink, avoiding about 205.7 MB duplication per full snapshot;
no prior evidence or checkpoint was deleted.
