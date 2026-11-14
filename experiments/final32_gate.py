"""Readiness for an unopened final set; no final case generation in this module."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METHOD='original_perception_rim_position_body_track_brake_noangular_handoff'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check(out):
    development=ROOT/'experiments/configs/terminal_braking32_qualification.json'
    expected=json.loads(development.read_text(encoding='utf-8'))['cases']
    path=ROOT/'.artifacts/evaluation_seeded.json'
    try:prior=json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return None
    if prior['manifest_sha256']!=sha(development) or len(prior['results'])!=11:return None
    expected_names={f"{c['case_id']}_persistent32v1_{METHOD}" for c in expected}
    if {r['run'] for r in prior['results']}!=expected_names:raise ValueError('Qualification identities changed')
    failures=[dict(run=r['run'],status=r['status'],reason=r.get('reason',r.get('error')))
        for r in prior['results'] if not r.get('verified_success',False)]
    if failures:
        out.mkdir(exist_ok=True)
        result=dict(status='final_not_started_qualification_failed',failures=failures,final_cases_opened=0)
        for name in ['development_gate_failed.json','waiting.json']:
            (out/name).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result),flush=True);return False
    if prior.get('wrist_rate')!=0 or prior['physics_step']!=.0005:raise ValueError('Expected per-step raw force protocol')
    for c in expected:
        name=f"{c['case_id']}_persistent32v1_{METHOD}"
        archive=ROOT/'.artifacts/lastinch_runs'/name
        read=lambda p:json.loads(p.read_text(encoding='utf-8'))
        scene=read(archive/'scene.json');load=read(archive/'static_load_calibration.json')
        hold=read(archive/'controller_hold_calibrated.json');q=read(archive/name/'policy.json')
        v=read(archive/name/'verified.json');audit=v.get('high_rate_wrist_audit',{})
        if scene['reset']['grasp_provenance'].get('jitter_seed')!=c['grasp_jitter_seed']:raise ValueError('Grasp jitter not applied')
        if scene.get('force_torque_system_instances')!=1:raise ValueError('Duplicate force system')
        if not audit.get('qualified') or not audit['complete_action_coverage'] or audit['peak_tared_force_N']>8:raise ValueError('Unqualified force evidence')
        if abs(audit['observed_mean_hz']-2000)>1 or audit['maximum_action_gap_s']>.000750001:raise ValueError('Missing physics-step force samples')
        if load['calibrated_force_base_N']!=hold['calibrated_force_base_N'] or q['load_calibration_sha256']!=sha(archive/'static_load_calibration.json'):raise ValueError('Force-only calibration mismatch')
        if q['ground_truth_policy_inputs'] or q['prescribed_grasp_policy_input'] or q['harness_revision']!=20:raise ValueError('Wrong policy input/version contract')
        if not q['angular_screen_diagnostic_ablation'] or q['counts_as_guarded_production_policy']:raise ValueError('Wrong angular-contract labeling')
        if q['research_controller_parameters'].get('RL_INSERT_SC_TERMINAL_BRAKE')!='1':raise ValueError('Braking not enabled')
    for source,digest in prior['source_sha256'].items():
        if sha(ROOT/source)!=digest:raise ValueError(f'Validated development source changed: {source}')
    return prior
