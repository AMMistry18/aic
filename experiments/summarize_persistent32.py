"""Measured development progress, retaining separate repair/fresh denominators."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METHOD='original_perception_rim_position_body_track_brake_noangular_handoff'


def main():
    manifest_path=ROOT/'experiments/configs/terminal_braking32_qualification.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));rows=[];maps=[]
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    for c in manifest['cases']:
        run=f"{c['case_id']}_persistent32v1_{METHOD}";archive=ROOT/'.artifacts/lastinch_runs'/run;p=archive/run
        if not (p/'verified.json').exists():continue
        v=read(p/'verified.json');q=read(p/'policy.json');scene=read(archive/'scene.json');w=read(p/'wrench_analysis.json')
        audit=w.get('high_rate_wrist_audit',{});load=read(archive/'static_load_calibration.json');hold=read(archive/'controller_hold_calibrated.json')
        provenance=(scene['reset']['grasp_provenance'].get('jitter_seed')==c['grasp_jitter_seed']
            and scene['force_torque_system_instances']==1 and q['harness_revision']==20
            and not q['ground_truth_policy_inputs'] and not q['prescribed_grasp_policy_input']
            and load['calibrated_force_base_N']==hold['calibrated_force_base_N']
            and q['load_calibration_sha256']==hashlib.sha256((archive/'static_load_calibration.json').read_bytes()).hexdigest())
        rows.append(dict(run=run,case_id=c['case_id'],case_origin=c['case_origin'],target_index=c['target_index'],grasp_jitter_seed=c['grasp_jitter_seed'],
            verified_success=bool(v['verified_success'] and provenance),raw_verifier_success=v['verified_success'],provenance_matches=provenance,
            reason=v['reason'],raw_peak_N=audit.get('peak_tared_force_N'),raw_p95_N=audit.get('p95_tared_force_N'),
            complete_force_coverage=audit.get('complete_action_coverage',False),raw_observed_hz=audit.get('observed_mean_hz'),
            raw_maximum_gap_s=audit.get('maximum_action_gap_s'),sampled_50hz_peak_N=w['peak_independent_tared_force_N'],
            terminal_metrics=v['terminal_metrics'],maximum_observed_penetration_m=v['max_global_penetration_m'],
            wall_s=q['wall_s'],configuration_wall_s=q['configuration_wall_s'],action_wall_s=q['wall_s']-q['configuration_wall_s'],sim_s=q['sim_s'],
            counts_as_guarded_production_policy=q['counts_as_guarded_production_policy']))
        maps.append({str(f.relative_to(archive/'source_snapshot')).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (archive/'source_snapshot').rglob('*') if f.is_file()})
    groups={origin:dict(planned=sum(c['case_origin']==origin for c in manifest['cases']),
        completed=sum(r['case_origin']==origin for r in rows),verified=sum(r['case_origin']==origin and r['verified_success'] for r in rows))
        for origin in ['known_failure_regression','unexecuted_seeded_development']}
    result=dict(purpose='Development only: upstream aligned SC handoff; not unrestricted angular recovery',
        planned=11,completed=len(rows),verified=sum(r['verified_success'] for r in rows),complete=len(rows)==11,
        separate_denominators=groups,identical_source_checkpoint_maps=bool(maps) and all(m==maps[0] for m in maps),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),confidence_interval=None,
        inference_note='Development repair/reused cases; do not interpret as final population reliability',runs=rows)
    out=ROOT/'experiments/evidence/original_repair'
    (out/'persistent_screen32v1.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    fields=['run','case_id','case_origin','target_index','grasp_jitter_seed','verified_success','provenance_matches','reason','raw_peak_N','raw_p95_N','complete_force_coverage','raw_observed_hz','raw_maximum_gap_s','sampled_50hz_peak_N','wall_s','action_wall_s','sim_s']
    with (out/'persistent_screen32v1.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
