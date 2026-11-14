"""Separate configuration, vision, registration and insertion from saved DEV logs.

No final outcomes are read, no simulated trial is launched, and missing stage
timestamps stay missing. A controller dwell is included in its reported phase.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/efficiency45'


def main():
    OUT.mkdir(exist_ok=True)
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    records=[]
    for r in ledger['results']:
        if r.get('status')!='completed':continue
        path=ROOT/r['evidence_dir'].replace('\\','/')/r['run']/'policy.json'
        policy=json.loads(path.read_text());kind=r['case']['connector']
        log=ROOT/'.artifacts/container'/(r['run']+'_policy.log');text=log.read_text(errors='replace')
        row=dict(run=r['run'],connector=kind,verified_success=bool(r.get('verified_success')),
                 source_policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 record_wall_s=policy['wall_s'],configuration_wall_s=policy.get('configuration_wall_s'),
                 policy_and_capture_wall_s=policy['wall_s']-policy.get('configuration_wall_s',0.),
                 action_sim_s=policy['sim_s'],stage_wall_s={},stage_sim_s={})
        if kind=='sc':
            points={}
            for line in text.splitlines():
                m=re.search(r'SC_TIMING phase=(\S+).*total_wall_s=([\d.]+) sim_elapsed_s=([\d.]+)',line)
                if m:points[m[1]]=(float(m[2]),float(m[3]))
            for name,a,b in [('visual_initialization','sc_start','handoff_validated'),
                             ('prealignment','visual_prealign_start','alignment_converged'),
                             ('seating_including_shoulder_registration','seating_start','seating_end')]:
                if a in points and b in points:
                    row['stage_wall_s'][name]=points[b][0]-points[a][0]
                    row['stage_sim_s'][name]=points[b][1]-points[a][1]
            probe=policy.get('axial_contact_registration') or {}
            if 'elapsed_sim_s' in probe:row['stage_sim_s']['shoulder_registration_subset']=probe['elapsed_sim_s']
        else:
            points={}
            for line in text.splitlines():
                m=re.search(r'\[([\d.]+)\].*DATUM_PHASE (\S+) sim_s=([\d.]+)',line)
                if m:points[m[2]]=(float(m[1]),float(m[3]))
            for name,a,b in [('opposed_lip_registration','entrance_lip_probe_+0.002m','aligned_entry'),
                             ('aligned_entry','aligned_entry','opposed_side_wall_registration'),
                             ('sidewall_registration','opposed_side_wall_registration','full_depth_insertion')]:
                if a in points and b in points:
                    row['stage_wall_s'][name]=points[b][0]-points[a][0]
                    row['stage_sim_s'][name]=points[b][1]-points[a][1]
            samples=(policy.get('sfp_contact_datum') or {}).get('insertion_samples',[])
            if samples:
                row['stage_sim_s']['final_insertion_phase_including_dwell']=samples[-1]['sim_s']-samples[0]['sim_s']
                # Match the deployed controller's unchanged estimated depth /
                # lateral convergence condition; this timing is not scoring.
                goal=[x for x in samples if abs(x['estimated_depth_m']-.0458)<=.0001
                      and sum(y*y for y in x['estimated_lateral_m'])<=.0001**2]
                if goal:
                    row['stage_sim_s']['final_stroke_to_first_estimated_goal']=goal[0]['sim_s']-samples[0]['sim_s']
                row['stage_wall_s']['final_insertion_phase_including_dwell']=None
        records.append(row)
    summary={}
    for kind in ['sc','sfp']:
        rows=[r for r in records if r['connector']==kind and r['verified_success']]
        keys=set(k for r in rows for k in r['stage_sim_s'])
        summary[kind]=dict(successful_development_trials=len(rows),
            median_record_wall_s=statistics.median(r['record_wall_s'] for r in rows),
            median_configuration_wall_s=statistics.median(r['configuration_wall_s'] for r in rows),
            median_action_sim_s=statistics.median(r['action_sim_s'] for r in rows),
            median_stage_sim_s={k:statistics.median(r['stage_sim_s'][k] for r in rows if k in r['stage_sim_s']) for k in sorted(keys)},
            median_stage_wall_s={k:statistics.median(v) for k in sorted(keys)
                if (v:=[r['stage_wall_s'][k] for r in rows if r['stage_wall_s'].get(k) is not None])})
    result=dict(scope='Historical development cohort only; successful-only timing summaries with every completed policy retained',
                final_trials_read=0,configuration_note='record_wall_s includes configuration and post-action evidence capture; action_sim_s starts after configuration',
                stroke_note='SFP phase includes 1.2s controller dwell; stroke wall time is not logged and is not inferred by a scale factor',
                summary=summary,records=records)
    (OUT/'development_timing.json').write_text(json.dumps(result,indent=2))
    with (OUT/'development_timing.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['run','connector','verified','metric','sim_s','wall_s'])
        for r in records:
            for k in sorted(set(r['stage_sim_s'])|set(r['stage_wall_s'])):
                writer.writerow([r['run'],r['connector'],r['verified_success'],k,r['stage_sim_s'].get(k),r['stage_wall_s'].get(k)])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
