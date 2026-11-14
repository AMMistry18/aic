"""Preserve the complete three-case V33 queue with explicit timing interruption."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair36/pulse36regv1'
batch=json.loads((OUT/'progress.json').read_text())
assert [r['case']['seed'] for r in batch['results']]==[45003,45006,45005]
original=json.loads((ROOT/'.artifacts/repair36/pulse36v1_completed.json').read_text())
assert batch['source_sha256']==original['source_sha256']
rows=[]
for result in batch['results']:
    run=result['run'];folder=ROOT/'.artifacts/lastinch_runs'/run/run
    p=json.loads((folder/'policy.json').read_text());d=p['sfp_contact_datum'];s=d['insertion_samples']
    timeout=any('timed out' in f['message'] for f in p['feedback'])
    rows.append(dict(run=run,verified_success=result['verified_success'],
        failure_cause='action_deadline_during_insertion' if timeout else result['reason'],
        raw_peak_N=result['high_rate_wrist_audit']['peak_tared_force_N'],
        full_rate_coverage=result['high_rate_wrist_audit']['complete_action_coverage'],
        action_sim_s=p['sim_s'],recorded_process_wall_s=p['wall_s'],
        waypoint_total_sim_s=sum(w['samples'][-1]['sim_s']-w['start_sim_s'] for w in d['waypoints']),
        travel_segment_sim_s=s[-1]['sim_s']-s[0]['sim_s'],
        last_estimated_depth_mm=s[-1]['estimated_depth_m']*1000,
        last_extra_axial_force_N=s[-1]['axial_tracking_force_N'],
        confirmed_host_sleep_during_run=result['case']['seed']==45006))
report=dict(status='development_complete_analyzed',verified=sum(r['verified_success'] for r in rows),attempted=len(rows),
    identical_source_and_checkpoint_maps_to_pulse36v1=True,rows=rows,
    timing_note='45006 experienced confirmed Windows low-power suspension. Its process wall duration is not uninterrupted calendar time. Keep its failure and report interruption.',
    held_out_final_trials_run=0,paper_ready=False)
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'trials.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
note="""Latest complete V33 fixed regression queue: **1/3** verified SFP
insertions. `dev_45003` passed (raw peak 3.250855 N); `dev_45006` and
`dev_45005` passed contact registration but exceeded the unchanged 720 s action
budget during travel (peaks 3.128375 and 2.906452 N). All three have complete
2000 Hz coverage. Keep both failures. Their source/checkpoint maps match the
earlier `pulse36v1` selected success exactly; they remain development cases,
not final reliability. Analysis/CSV: `.artifacts/repair36/pulse36regv1/`.
`dev_45006` contains a confirmed host sleep interval; its recorded process wall
duration cannot stand for uninterrupted calendar time. The interruption is
preserved separately and is not grounds to erase the failed insertion.

Offline truth/command diagnosis found post-deadline forward travel of 0.489695
mm over 0.3125 sim s (45006), and 2.913399 mm over 0.3805 sim s (45005).
The prior send hook could inject its last 0.124375 /0.605089 N axial motion
feedforward into the generic 90 N/m hold. The source-audited mechanism is
consistent with the observed drift; a counterfactual physical abort test has
not yet been run. Diagnostic artifact:
`.artifacts/repair37/timeout_hold_diagnosis.json`.

V34 `waypoint37v1` now tests bounded smooth positioning plus cleared motion
compensation before deadline/failure holds, on 45006 then 45005. Default free
waypoints use a quintic path bounded at 5 mm/s and 20 mm/s^2, native base-frame
damping feedforward, and the same 0.15 N axial integral /0.5 mm position-lead
bounds. Explicit 1 mm/s entry moves retain their earlier controller. The
40 um arrival tolerance, 0.3 s dwell, 12 sim s waypoint budget, force guards,
720 s action budget and independent seat criteria remain. 142 checks pass;
physical outcome pending. Source freeze:
`.artifacts/repair37/waypoint37v1_source_freeze.json`. No model was retrained.

The checkpoint/article audit is complete: original SC plug weights unchanged;
SC mouth, SFP plug and NIC weights are fine-tuned descendants of the tracked
originals, with matching training and archived runtime hashes. See
`paper/ORIGINAL_PIPELINE_TRACEABILITY.md`. An undeployed support-priority
search optimization returned exactly identical selected pose dictionaries on
three saved RGB captures, with 19 selector tests passing and 0.36-0.70 s
selection savings in quiet-container replay. It is outside V34, which isolates
the larger positioning repair. No final trial has been opened.

"""
for name in ['EXPERIMENT_PLAN.md','EXPERIMENTS.md','PAPER_RESULTS.md','RESEARCH_PROGRESS.md']:
    path=ROOT/name;first,rest=path.read_text(encoding='utf-8').split('\n',1)
    path.write_text(first+'\n\n'+note+rest.lstrip('\n'),encoding='utf-8')
print(json.dumps(report,indent=2))
