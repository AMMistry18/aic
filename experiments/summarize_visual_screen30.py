"""Summarize completed v14 development screens, preserving stage failures."""
import hashlib
import gzip
import argparse
import json
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
NAMES=['dev_3000_cable30v1_original_perception_rim_position_body_track_handoff']+[
    f'dev_{i}_cable30screen_original_perception_rim_position_body_track_handoff' for i in [3002,3004,3006,3008]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label');parser.add_argument('--method',default='original_perception_rim_position_body_track_handoff');parser.add_argument('--out',type=Path);args=parser.parse_args()
    names=NAMES if args.label is None else [f'dev_{i}_{args.label}_{args.method}' for i in [3000,3002,3004,3006,3008]]
    rows=[];maps=[]
    for n in names:
        archive=ROOT/'.artifacts/lastinch_runs'/n;r=archive/n
        if not (r/'verified.json').exists():continue
        policy=json.loads((r/'policy.json').read_text());verified=json.loads((r/'verified.json').read_text());wrench=json.loads((r/'wrench_analysis.json').read_text())
        log=(ROOT/'.artifacts/container'/(n+'_policy.log')).read_text()
        summary=re.findall(r'SC_TIMING_SUMMARY outcome=(\S+) reason=(\S+)',log)
        reason=summary[-1][1] if summary else policy['reason']
        source={str(p.relative_to(archive/'source_snapshot')).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (archive/'source_snapshot').rglob('*') if p.is_file()}
        maps.append(source)
        rows.append(dict(run=n,harness_revision=policy['harness_revision'],verified_success=verified['verified_success'],
            angular_screen_diagnostic_ablation=policy.get('angular_screen_diagnostic_ablation',False),
            counts_as_guarded_production_policy=policy.get('counts_as_guarded_production_policy',True),
            failure_stage=reason if not verified['verified_success'] else None,verifier_reason=verified['reason'],
            body_diagnostic=policy.get('sc_body_diagnostic'),wall_s=policy['wall_s'],sim_s=policy['sim_s'],
            peak_policy_force_N=policy['peak_tared_force_N'],peak_independent_force_N=wrench['peak_independent_tared_force_N'],
            max_global_penetration_m=verified['max_global_penetration_m'],terminal_metrics=verified['terminal_metrics']))
        diagnostic=policy.get('sc_body_diagnostic') or {}
        if diagnostic.get('accepted'):
            trace=json.load(gzip.open(r/'verification_trace.json.gz','rt'))
            actual=min(trace,key=lambda x:abs(x['sim_time_s']-max(diagnostic['camera_stamp_s'])))
            B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
            tip=np.linalg.inv(B)@actual['tip_world']
            rows[-1]['posthoc_body_position_error_mm']=float(np.linalg.norm(tip[:3,3]-diagnostic['position'])*1000)
            rows[-1]['posthoc_body_prior_angle_error_deg']=float(np.rad2deg(Rotation.from_matrix(tip[:3,:3].T@diagnostic['prior_rotation']).magnitude()))
    result=dict(purpose='Five fixed seeded development cases, one per target; not final or nominal repeats',
        planned=5,completed=len(rows),verified=sum(r['verified_success'] for r in rows),
        identical_source_and_checkpoint_maps=bool(maps) and all(m==maps[0] for m in maps),
        condition='upstream_aligned_handoff_assumed, original plug proposals, transferred mouth, v14 cable mask',
        final_100000_series_used=False,confidence_interval=None,
        interval_note='Do not infer broad reliability from five target-stratified development cases selected during repair',
        runs=rows)
    result['verified_guarded_production']=sum(r['verified_success'] and r['counts_as_guarded_production_policy'] for r in rows)
    result['purpose']='Five fixed development cases; diagnostic ablations explicitly excluded from guarded-policy successes'
    result['condition']='upstream_aligned_handoff_assumed; checkpoint/source hashes distinguish batches'
    path=args.out or ROOT/'experiments/evidence/original_repair/varied_screen_v14.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}));print([(r['run'],r['verified_success'],r['failure_stage']) for r in rows])


if __name__=='__main__':main()
