# Under-30-second development, November 14, 2026

User target: preserve SC reliability and bring SFP to comparable reliability,
with complete insertion under 30 seconds. Primary timing includes perception,
registration, travel and terminal dwell after action start. Record configuration
time and wall time separately. A simulated 30 seconds is not 30 laptop seconds.
Report success-within-30-seconds over ALL attempts, not only conditional median
time among successful runs. Keep the 20 N raw ceiling and existing independent
depth, lateral, angle, contact, dwell and complete 2000 Hz coverage checks.

## Established starting point

SC cache49: one reused development case verified at 26.189 sim /282.194 wall s.
SFP lip51: one reused development case verified at 47.071 sim /569.913 wall s.
Neither is a randomized optimized-policy success rate.
At inspection, frozen final44 had 96 primary attempts: SC46/48, SFP32/48.
These are baseline results, not results for the new candidates. Live ledger
supersedes this snapshot. All final44 source and weight hashes remain frozen.

The SFP v5 action breaks down approximately as follows, from actual stage logs:
6.0715 s before lip registration; 17.2475 s for entrance lip measurement and
clearance; 5.013 s to align and enter; 11.8085 s for lateral registration and
recentering; 6.9305 s for final travel and completion. Thus a faster final push
alone cannot achieve 30 s. Exact phase intervals do not independently certify
seat time; the separate trace verifier remains authoritative.

## Implemented development work and serialized queue

1. `continue_pruning54.py`: exact NIC association pruning, runtime47v6, old
   dev_45005. The previous four-scene replay preserved all 45 accepted fits;
   never accept directly from the bound. Physical qualification is pending.
2. `replay_fast_registration55.py`: 4 mm/s, 20 mm/s^2 lateral quintic scheduling
   versus 2 mm/s, 10 mm/s^2. Preserves 0.5 mm lead, 3 mm workspace, 6 N observed
   guard, 0.4 s fresh stationary measurement and all contrast/geometry gates.
   Synthetic replay: 35/35 accepted for each; median 11.58 to 8.02 seconds;
   all five negative controls reject. These are not physical times or forces.
   `continue_registration55.py` waits for v6 success/release, then qualifies v7
   on dev_45005. Stop on failure; no automatic parameter tuning or case retry.
3. `diagnose_sfp_reprojection56.py`: three cached old development image sets.
   Joint image fitting does not fix the rejected dev_45009 example: 16.1867 to
   13.4367 px, still over 6 px; leave-one-camera-out shifts reach 9.18 mm.
   dev_45002 passes the original pixel gate but has up to 3.47 mm inter-fit
   translation disagreement. Pixel acceptance alone is not precision proof.
   No truth read, no pose installed, no view discarded by the production policy.
4. `audit_sfp_crop57.py`: prospective full-frame versus pad4/pad6/pad8 inference
   comparison on all completed old historical SFP development cases, unchanged
   checkpoint, original stamps and gates. `continue_crop57.py` waits until v7
   releases its lease, then runs this offline diagnostic in its own serialized
   slot. This tests a possible training/deployment scale mismatch; it does not
   establish that mismatch beforehand. No favourable-crop selection rule is
   installed. No training or new checkpoint has started.

Only the isolated development clone changes. Owned coordinator identity,
stopped-main-container boundary, watchdog and frozen hashes are checked. No
two Gazebo worlds run together. Development scheduling supports both primary
and paired-ablation boundaries; selected final cases are not retried/replaced.
Progress is in pruning_followup54, registration_followup55, crop_followup57,
and runtime_qualification47_v6/v7/v57. Dependencies stop when checks fail.

## Remaining work before claiming the target

- Review actual v6/v7 force, seating, timing and camera evidence. Export compact
  checksummed artifacts before cleanup. In particular, faster contact motion
  may increase raw peaks even when low-rate forces look small.
- SFP currently primes plug vision in InsertionPolicy and independently repeats
  it in sfp_fixture_refinement.perceive. A future single-pass change can dispatch
  the datum method before generic priming, retaining the complete datum plug
  gate, exposure-time FK, stationarity and fixture checks. This is identified,
  not installed or physically qualified. Other SFP methods must keep priming.
- Inspect crop diagnostics; correct actual learned localization/association
  defects with development-only data. Do not solve them by increasing the
  6 px limit, discarding inconvenient views without evidence, or supplying the
  prescribed simulation grasp. Retain the original weighted model pipeline.
- Qualify shorter entrance/side-wall trajectories without shortening fixed
  force evidence windows. The current improvements alone are not demonstrated
  sufficient for 30 s. Keep conservative braking, measurement and dwell.
- Repeat across development mounts/slots and allowed offsets before freezing
  a new policy. Include failures and regressions. Then predeclare fresh seeds
  and run both connectors; do not use final44 examples to tune the new policy.
  Compare all-attempt success and success-within-30 s, Wilson intervals, timing
  distributions, raw peaks, complete coverage and failure causes. Aim for at
  least 95% observed success for both; report uncertainty rather than claiming
  a guaranteed 95% underlying reliability from a small sample.
- Broad angular recovery remains unqualified; under-30-second work does not
  enlarge the existing conditional angular/rigid-grasp scope.
