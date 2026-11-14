"""Snapshot completed V34 pilot measurements without changing the live policy."""
import json
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--batch',default='.artifacts/seeded_evaluation_progress.json')
args=parser.parse_args()
batch=json.loads((ROOT/args.batch).read_text())
assert batch['results'] and all('_waypoint37v1_' in r['run'] for r in batch['results'])
OUT=ROOT/'.artifacts/repair37/waypoint37v1';OUT.mkdir(exist_ok=True)
rows=[]
for result in batch['results']:
    run=result['run'];f=ROOT/'.artifacts/lastinch_runs'/run/run
    p=json.loads((f/'policy.json').read_text());d=p['sfp_contact_datum'];samples=d['insertion_samples']
    old=f"dev_{result['case']['seed']}_pulse36regv1_original_perception_sfp_contact_datum_handoff"
    previous=json.loads((ROOT/'.artifacts/lastinch_runs'/old/old/'policy.json').read_text())
    waypoints=lambda data:sum(w['samples'][-1]['sim_s']-w['start_sim_s'] for w in data['waypoints'])
    before=waypoints(previous['sfp_contact_datum']);after=waypoints(d)
    rows.append(dict(run=run,verified_success=result['verified_success'],reason=result['reason'],
        terminal_metrics=result['terminal_metrics'],force_audit=result['high_rate_wrist_audit'],
        recorded_wall_s=p['wall_s'],configuration_wall_s=p['configuration_wall_s'],action_sim_s=p['sim_s'],
        waypoint_total_sim_s=after,previous_waypoint_total_sim_s=before,
        descriptive_waypoint_reduction_percent=100*(1-after/before),
        travel_segment_sim_s=samples[-1]['sim_s']-samples[0]['sim_s'],
        previous_same_case_outcome='action deadline failure',
        previous_45006_host_sleep_not_a_clean_wall_comparison=result['case']['seed']==45006,
        abort_compensation_fix_exercised=bool(d.get('motion_compensation_stopped')),
        success_scope='stable clearance seat while held; no latch/electrical/release claim'))
report=dict(completed=len(rows),verified=sum(r['verified_success'] for r in rows),planned=2,rows=rows,
    held_out_final_trials_run=0,paper_ready=False,
    timing_scope='Selected development comparisons with different image captures; not final matched speed estimates')
(OUT/'progress.json').write_text(json.dumps(batch,indent=2)+'\n')
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
