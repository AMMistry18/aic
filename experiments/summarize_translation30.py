"""Summarize executed wider cases, verifying actual grasp-jitter provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
METHOD='original_perception_rim_position_body_track_noangular_handoff'


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='translation30v1');a=p.parse_args()
    manifest_path=ROOT/'experiments/configs/translation_development30.json';manifest=json.loads(manifest_path.read_text());manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    progress=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text());execution={r['run']:r for r in progress['results']} if progress['manifest_sha256']==manifest_sha else {}
    rows=[];scenes={};holds={};maps=[]
    for c in manifest['cases']:
        name=f"{c['case_id']}_{a.label}_{METHOD}";archive=ROOT/'.artifacts/lastinch_runs'/name;r=archive/name
        if not (r/'verified.json').exists():
            if name in execution and execution[name]['status']=='environment_or_execution_failure':
                rows.append(dict(case_id=c['case_id'],run=name,status='environment_or_execution_failure',error=execution[name]['error'],verified_success=False,eligible_policy_measurement=False,expected_jitter_seed=c['grasp_jitter_seed']))
            continue
        v=json.loads((r/'verified.json').read_text());q=json.loads((r/'policy.json').read_text());w=json.loads((r/'wrench_analysis.json').read_text());scene=json.loads((archive/'scene.json').read_text());hold=json.loads((archive/'controller_hold_calibrated.json').read_text());load_path=archive/'static_load_calibration.json';load=json.loads(load_path.read_text())
        actual_jitter=scene['reset']['grasp_provenance'].get('jitter_seed');valid_jitter=actual_jitter==c['grasp_jitter_seed']
        valid_load=load['calibrated_force_base_N']==hold['calibrated_force_base_N'] and q['load_calibration_sha256']==hashlib.sha256(load_path.read_bytes()).hexdigest()
        if not q['angular_screen_diagnostic_ablation'] or q['counts_as_guarded_production_policy']:raise ValueError('Unexpected production/ablation labeling')
        rows.append(dict(case_id=c['case_id'],run=name,status='completed',target_index=c['target_index'],seed=c['seed'],gap_mm=c['gap_mm'],initial_lateral_norm_mm=float(np.linalg.norm(c['lateral_mm'])),
            expected_jitter_seed=c['grasp_jitter_seed'],actual_jitter_seed=actual_jitter,jitter_provenance_matches=valid_jitter,
            force_only_calibration_matches=valid_load,eligible_policy_measurement=valid_jitter and valid_load,
            verified_success=bool(v['verified_success'] and valid_jitter and valid_load),raw_verifier_success=v['verified_success'],
            reason=v['reason'],geometry_reason=v.get('geometry_verifier',v)['reason'],harness_revision=q['harness_revision'],
            terminal_metrics=v['terminal_metrics'],peak_independent_force_N=w['peak_independent_tared_force_N'],p95_independent_force_N=w['p95_independent_tared_force_N'],
            max_global_penetration_m=v['max_global_penetration_m'],wall_s=q['wall_s'],sim_s=q['sim_s'],counts_as_guarded_production_policy=False))
        scenes[c['case_id']]=scene;holds[c['case_id']]=hold
        maps.append({str(f.relative_to(archive/'source_snapshot')).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in (archive/'source_snapshot').rglob('*') if f.is_file()})
    pairs=[]
    for target in range(5):
        n0=f'dev_{40000+target}';n1=f'dev_{41000+target}'
        if n0 not in scenes or n1 not in scenes:continue
        s0,s1=scenes[n0],scenes[n1];G0,G1=[np.array(s['reset']['tip_in_tcp_reset']) for s in [s0,s1]];H0,H1=[np.array(holds[n]['setup_tip_in_tcp_measured']) for n in [n0,n1]]
        delta=lambda a,b:dict(translation_mm=float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),rotation_deg=float(np.rad2deg(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())))
        perturb=np.array(s1['reset']['grasp_provenance']['tip_frame_perturbation'])
        pairs.append(dict(target_index=target,source_initial_tip_poses_identical=bool(np.allclose(s0['reset']['tip_world'],s1['reset']['tip_world'],rtol=0,atol=1e-9)),
            jitter_was_applied_to_actual_source_grasp=bool(np.allclose(G0@perturb,G1,rtol=0,atol=1e-9)),
            authored_grasp_delta=delta(G0,G1),independent_setup_grasp_delta_diagnostic=delta(H0,H1),
            setup_diagnostic_not_used_by_policy=True))
    summary=dict(purpose='Fresh wider DEVELOPMENT screen; aligned connector handoff; not final/unrestricted rotation recovery',planned=10,completed=len(rows),
        policy_measurements=sum(r['eligible_policy_measurement'] for r in rows),environment_failures=sum(r['status']=='environment_or_execution_failure' for r in rows),
        verified=sum(r['verified_success'] for r in rows),complete=len(rows)==10,manifest_sha256=manifest_sha,
        identical_source_checkpoint_maps=bool(maps) and all(m==maps[0] for m in maps),
        per_grasp_condition={condition:dict(planned=5,completed=sum((r['expected_jitter_seed'] is not None)==jitter for r in rows),verified=sum(r['verified_success'] and (r['expected_jitter_seed'] is not None)==jitter for r in rows)) for condition,jitter in [('prescribed_default',False),('perturbed_rigid_grasp',True)]},
        paired_geometry_audit=pairs,final_100000_series_used=False,confidence_interval=None,
        interval_note='Small development/repair screen; do not claim held-out population reliability',runs=rows)
    out=ROOT/'experiments/evidence/original_repair';(out/f'wider_screen_{a.label}.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/f'wider_screen_{a.label}.csv').open('w',newline='') as f:
        fields=['case_id','status','target_index','gap_mm','initial_lateral_norm_mm','expected_jitter_seed','actual_jitter_seed','jitter_provenance_matches','force_only_calibration_matches','verified_success','reason','peak_independent_force_N','sim_s','wall_s'];writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
