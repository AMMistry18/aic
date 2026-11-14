"""Refresh measured status only; does not modify frozen policy sources."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT='''## Current checkpoint - 2026-11-01, shoulder braking and force measurement

The original-perception SC pipeline retains the original confidence-weighted
plug network and uses an adapted mouth network, public-CAD rim/body image fits,
and bounded own-robot lateral PI tracking. It estimates grasp translation from
RGB and receives no live object/target truth. The held connector begins aligned upstream
under a prescribed rigid grasp. General visual angular recovery and repaired
SFP perception remain unqualified.

| Frozen development batch | Verified insertion | Remaining failures |
|---|---:|---|
| v13, one nominal setup repeated | 3/3 | Same seed/setup; not randomized reliability |
| v14, cable mask + earlier mouth weights | 1/5 | Three mouth rejections; one learned angular-prior rejection |
| v14, longer mouth adaptation (`transfer30v1`) | 2/5 | Three learned angular-prior rejections |
| v15, angular-screen diagnostic ablation | 5/5 | No physical failures in these five cases; angular alignment assumed |
| Matched conditional body + PI, `component30v1` | 5/5 | Same five development cases; distinct frozen batch |
| Matched control: no body correction | 0/5 | Three shallow stalls, two force aborts |
| Matched control: no lateral PI | 0/5 | Four lateral misses and one shallow stall |
| V19 high-rate qualification, stopped after two | 0/2 | One missing-sample audit failure; one 24.1285 N shoulder impact |

Each five-case batch uses the fixed development targets 0-4 (seeds 3000, 3002,
3004, 3006, 3008), not fresh held-out cases. Do not pool versions. The conditional
ablation and component comparisons omit the learned angular veto; they are
**not** repaired guarded-policy results and do not certify upstream alignment.
Guarded production results remain 2/5 for the latest mouth checkpoint. No final
100000-series case has been used; these screens do not establish broad reliability.
Older rows use their frozen geometry/contact protocol and 50 Hz sampled force
peaks. They do not establish compliance with the newer every-step force audit.

All five ablation insertions passed the unchanged independent criteria:
15.64 +/-0.25 mm depth, <=0.25 mm lateral and <=0.25 deg full angular error,
continuous one-simulation-second dwell with valid contact evidence, and <=0.1 mm
whole-trial penetration. Across terminal windows they held 15.54785-15.56677 mm
depth, 0.02961-0.13204 mm lateral error and 0.11650-0.23988 deg angular error.
Sampled 50 Hz wrist peaks were 1.4970-3.7933 N, maximum observed penetrations
0.000064-0.002060 mm, and durations 146.883-385.425 wall /14.751-37.323 sim s.
The largest terminal angle is close to the unchanged limit and warrants broader
testing. Per-case records: `varied_screen_v14.json`,
`varied_screen_transfer30v1.json`, `varied_screen_angular_ablation30v1.json`
under `experiments/evidence/original_repair`.

The longer mouth model passes all five target-perception stages. The original
plug angular estimate rejects three physically aligned cases; removing only
that veto permits all five conditional insertions. This supports the translation/
control path under the declared angular condition, not arbitrary 6D perception.

The orange-cable occlusion defect was repaired in the rim fitter; recorded
cross-view disagreement fell from 1.869/2.073 to 0.108/0.091 mm without changing
calibration or acceptance thresholds. Direct rotation fitting, keypoint
calibration, visibility filtering, line/silhouette fits, augmentation ensembles
and longer plug training have not qualified a rotation repair. On six static
development scenes, original median angular error was 1.310 deg; standard plug
adaptation best/final were 2.709/1.086 deg, projected-edge loss best/final
1.960/1.625 deg, and affine-rotation loss best/final 1.986/1.825 deg. None was
substituted into these physical results.

The source-frozen 15-trial component comparison is complete, with identical
source/checkpoint maps and shuffled method order. Full body + PI passed 5/5;
each control passed 0/5. Full terminal depth was 15.54777-15.56700 mm, lateral
error 0.03398-0.09794 mm, and angle 0.11595-0.23997 deg. Independent peaks were
1.4212-2.5440 N; median completion 17.167 sim /163.214 wall s, and maximum
whole-trial penetration 0.0003251 mm. No-body failures included 17.5668/11.1248 N
spikes despite the 8 N pre-command abort check, which is not a physical force
cap. Some offline GPU training ran concurrently; wall times are descriptive,
not controlled throughput comparisons. See `component_screen_component30v1.json`,
`component30_outcomes.svg` and `component30_execution_context.json` in evidence.
These results support the component interaction under the declared alignment
condition; they do not establish broad reliability or repair angular perception.

Ten fresh development cases (`translation30v1`) completed without
concurrent training: 0.5-1.5 mm starting lateral offset, 5-15 mm gap, and paired
unknown rigid-grasp perturbations. The initial physical tip is still aligned
upstream. Grasp jitter is actually passed to the scene generator; its provenance
and rendered images must be checked, not inferred from case names. Harness v16
uses a force-only own-robot load calibration file; geometry and controller
settings are unchanged. Final 100000-series cases remain untouched.

A paired image/truth audit found actual rigid-grasp motion <=0.000104 mm while
original estimates changed 0.229-0.503 mm and 1.275-2.074 deg. Images show the
moving orange cable occluding the centre view. Original point scores remain
0.9956 or higher there; these scores are not calibrated pose uncertainty.
Foreground-only refinement lowered median static angle to 0.641 deg, but failed
all four rigid-consistency pairs. Side-view refinement with fixed pixel-phase
averaging reached 0.306 deg median recorded-image error, yet passed only one
of four consistency pairs. Both remain offline; no threshold was relaxed or
candidate inserted into the physical policy. Current selected regressions: 183 passed; separate
training-loss mechanism checks: 6 passed.

Actual target-4 ablation footage: `video/target4_angular_ablation30/attempt.mp4`.
It visibly labels the ablation and upstream prior, uses actual headless target/
wrist/board captures at 1x simulation playback, and displays truth only as
independent diagnostic overlays. It was fully decoded and terminal images
inspected. Guarded target-0 footage: `video/target0_transfer30/attempt.mp4`.
Camera images, clocks, executable source and checkpoint hashes remain archived.
The old `seeded_pilot27` is stopped. Current status:
`.artifacts/active_original_perception29.json`; entries below are historical.
'''
def main():
    wide_path=ROOT/'experiments/evidence/original_repair/wider_screen_translation30v1.json'
    wide=json.loads(wide_path.read_text()) if wide_path.exists() else None
    checkpoint=CHECKPOINT
    if wide:
        checkpoint+=f"\nCompleted wider screen: {wide['verified']}/{wide['completed']} verified, of {wide['planned']} planned; {wide['environment_failures']} environment/execution failures. This is a development result, not a held-out reliability estimate.\n"
    checkpoint+='\nThe wider screen finished 9/10: default grasps 5/5, perturbed rigid grasps 4/5. The last case reached the shoulder with good physical alignment but a 34.7699 N independent wrist transient at about 5.669 mm/s axial speed. It failed the force guard. Its terminal lateral/angle ranges remained 0.05461-0.14172 mm /0.04449-0.05183 degrees. The failed run remains in the denominator.\n'
    checkpoint+='\nTerminal braking v17 repaired that one development case (1/1): verified depth 15.566032-15.566088 mm, lateral 0.043176-0.049501 mm, angle 0.047340-0.047428 degrees, maximum whole-trial penetration 0.0001373 mm. Recorded 50 Hz peak was 1.15726 N and pre-seat diagnostic speed 1.7613 mm/s, versus 34.7699 N and 5.6686 mm/s in the preserved baseline. Durations were 205.958 wall /24.5225 sim seconds. This is a repaired-case regression, not a randomized reliability result. Both forces are SAMPLED 50 Hz peaks: short impulses can be missed. V18 adds optional per-physics-step raw wrist capture and requires full action coverage plus <=8 N measured peak for high-rate audited trials; historical geometry/force data are preserved. The first high-rate braked attempt (forceaudit31v3) passed geometry/contact and recorded a 2.21306 N raw peak. Its original strict force audit rejected duplicate timestamps: two sensor systems each publish every 0.5 ms step. Distinct timestamp coverage is complete at 2000 Hz; a single-publisher rerun is required before qualification. Four earlier high-rate attempts failed bootstrap before policy commands; all are preserved as environment failures.\n'
    checkpoint+='\nThe separate 50-case /150-trial conditional final study DID NOT START: its all-ten-pass development gate failed. No 200000-series final manifest was generated or opened; older 100000-series cases also remain untouched. The waiting worker exited, releasing its lock and awake request. See `.artifacts/final_aligned31/development_gate_failed.json`. A repaired, source-frozen development validation is required before reconsidering final execution.\n'
    persistent_path=ROOT/'experiments/evidence/original_repair/persistent_screen32v1.json'
    if persistent_path.exists():
        current=json.loads(persistent_path.read_text(encoding='utf-8'))
        checkpoint+=f"\nV20 persistent braking: {current['verified']}/{current['completed']} completed development cases pass both geometry/contact AND complete 2000 Hz raw force coverage, of eleven planned (two known regressions plus nine previously unexecuted cases). Pending cases are not counted. The first repaired target-0 case held 15.56578-15.56585 mm depth, 0.08027-0.08028 mm lateral and 0.13479-0.13485 degrees angle, with 4.38478 N raw peak, zero samples over 8 N and no missing force step. Pre-seat speed was 0.62534 mm/s. V20 retains the scheduled axial target when the robot runs ahead, slows fully 1 mm before the candidate boundary, and adjusts only the operational stall timeout to the planned slow motion. Physical success/force thresholds are unchanged.\n"
        checkpoint+='\nA separate final32 worker is queued for 50 new conditional SC cases x four matched methods (200 trials): full body+PI+brake, no body, no PI, no brake. It generates no final case unless all eleven V20 qualification cases pass exact source/checkpoint, jitter/load provenance and full force audit. It stops the queue on a failed qualification; no failed final case is tuned or replaced. Wilson intervals and Holm-corrected paired tests are generated only after actual completion. Status: `.artifacts/final_aligned32/waiting.json`, then `progress.json`; no final result is assumed.\n'
    for name in ['EXPERIMENT_PLAN.md','EXPERIMENTS.md','PAPER_RESULTS.md','RESEARCH_PROGRESS.md']:
        p=ROOT/name;s=p.read_text(encoding='utf-8')
        start=s.index('<!-- current-checkpoint -->')+len('<!-- current-checkpoint -->')
        end=s.index('<!-- /current-checkpoint -->',start)
        p.write_text(s[:start]+'\n'+checkpoint+s[end:],encoding='utf-8')
    path=ROOT/'.artifacts/active_original_perception29.json';status=json.loads(path.read_text())
    status.update(        regression_tests_passed=183,training_loss_tests_passed=6,
        changes_under_test=['v20 persistent axial trajectory, early settling reserve and reliable per-step raw sensor recording; same models and geometric limits'],
        physical_v14_outcome='Old mouth 1/5; longer mouth 2/5; distinct versions',
        v14_new_mouth_screen={'planned':5,'completed':5,'verified':2},
        v15_angular_screen_ablation={'planned':5,'completed':5,'verified':5,'counts_as_guarded_production_policy':False},
        verified_conditional_sc_insertions=16,verified_guarded_conditional_sc_insertions=6,
        verified_unrestricted_visual_rotation_insertions=0,
        qualified_by_condition={'v13_nominal_same_setup':3,'v14_cable_mask_varied':1,'v14_new_mouth_varied':2,'v15_explicit_angular_ablation':5,'component30v1_body_PI':5,'component30v1_no_body':0,'component30v1_no_PI':0},
        aggregate_count_is_not_a_reliability_rate=True,
        component30v1={'completed':15,'full':{'verified':5,'attempted':5},'no_body':{'verified':0,'attempted':5},'no_PI':{'verified':0,'attempted':5}},
        latest_attempt_video='experiments/evidence/original_repair/video/wider_grasp41000/attempt.mp4',
        failed_final31_study={'planned_cases':50,'planned_trials':150,'status':'not_started_development_gate_failed','worker_active':False,'status_file':'.artifacts/final_aligned31/waiting.json','scope':'nominally aligned connector handoff; not unrestricted rotation or SFP'})
    if wide:
        status['translation30v1']=dict(planned=wide['planned'],completed=wide['completed'],verified=wide['verified'],environment_failures=wide['environment_failures'])
        if wide['runs']:status['last_completed_wider_run']=wide['runs'][-1]['run']
    for k in ['verified_original_pipeline_insertions','verified_conditional_sc_insertions','verified_guarded_conditional_sc_insertions','planned_varied_trials','completed_varied_trials','varied_trials_note']:
        status.pop(k,None)
    path.write_text(json.dumps(status,indent=2)+'\n')
if __name__=='__main__':main()
