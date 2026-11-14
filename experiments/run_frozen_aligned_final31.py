"""Wait for the exact development gate, then run a source-locked final study.

Separate explicit route for new conditional-task final seeds. The existing
development runner still rejects final manifests. No old 100000-series case is
opened. A failed development gate stops execution, not a reason to select easier
final seeds. Failed final trials are retained and never tuned/replaced.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import run_seeded_evaluation as runner
OUT=ROOT/'.artifacts/final_aligned31'
PROTOCOL=ROOT/'experiments/configs/final_aligned31_protocol.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def gate():
    development=ROOT/'experiments/configs/translation_development30.json';expected_sha=sha(development)
    expected=json.loads(development.read_text())['cases'];path=ROOT/'.artifacts/evaluation_seeded.json'
    try:prior=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
    if prior['manifest_sha256']!=expected_sha or len(prior['results'])!=10:return None
    expected_names={f"{c['case_id']}_translation30v1_original_perception_rim_position_body_track_noangular_handoff" for c in expected}
    if {r['run'] for r in prior['results']}!=expected_names:raise ValueError('Development identities do not match frozen gate')
    failures=[dict(run=r['run'],status=r['status'],reason=r.get('reason',r.get('error'))) for r in prior['results'] if not r.get('verified_success',False)]
    if failures:
        OUT.mkdir(exist_ok=True);(OUT/'development_gate_failed.json').write_text(json.dumps(dict(status='final_not_started_development_repair_required',failures=failures),indent=2)+'\n')
        (OUT/'waiting.json').write_text(json.dumps(dict(status='final_not_started_development_gate_failed',failures=failures),indent=2)+'\n')
        active=ROOT/'.artifacts/active_original_perception29.json';state=json.loads(active.read_text());state.update(status='wider_development_requires_repair_final_not_started',live_insertion_world=False,final_gate_failures=failures);active.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(dict(status='final_not_started_development_gate_failed',failures=failures)),flush=True);return False
    for c in expected:
        name=f"{c['case_id']}_translation30v1_original_perception_rim_position_body_track_noangular_handoff";archive=ROOT/'.artifacts/lastinch_runs'/name
        scene=json.loads((archive/'scene.json').read_text());load=json.loads((archive/'static_load_calibration.json').read_text());hold=json.loads((archive/'controller_hold_calibrated.json').read_text());policy=json.loads((archive/name/'policy.json').read_text())
        if scene['reset']['grasp_provenance'].get('jitter_seed')!=c['grasp_jitter_seed']:raise ValueError('Development jitter was not applied')
        if load['calibrated_force_base_N']!=hold['calibrated_force_base_N'] or policy['load_calibration_sha256']!=sha(archive/'static_load_calibration.json'):raise ValueError('Force-only calibration provenance mismatch')
    for source,digest in prior['source_sha256'].items():
        if sha(ROOT/source)!=digest:raise ValueError(f'Validated development source changed: {source}')
    return prior


def make_cases():
    cases=[]
    for i in range(50):
        seed=200000+i;rng=np.random.default_rng(seed);target=int(rng.integers(5));gap=float(rng.uniform(5,15));radius=float(rng.uniform(.5,1.5));theta=float(rng.uniform(-np.pi,np.pi))
        cases.append(dict(case_id=f'final_sc_{seed}',seed=seed,connector='sc',target_index=target,sfp_port=0,gap_mm=gap,
            lateral_mm=(radius*np.array([np.cos(theta),np.sin(theta)])).tolist(),angular_deg=rng.uniform(-.1,.1,3).tolist(),
            grasp_jitter_seed=300000+i if rng.integers(2) else None))
    return runner.validate_cases(cases)


def run():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-for-development',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True);protocol=json.loads(PROTOCOL.read_text());progress=OUT/'progress.json';manifest_path=OUT/'manifest.json'
    if a.resume:
        if (ROOT/'.artifacts/kilted_handoff').exists():
            raise RuntimeError('Unarchived evaluator workspace exists; preserve/recover it before resuming')
        batch=json.loads(progress.read_text());manifest=json.loads(manifest_path.read_text())
        if batch['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Final protocol changed')
        for source,digest in batch['source_sha256'].items():
            if sha(ROOT/source)!=digest:raise ValueError(f'Frozen final source changed: {source}')
        runner.SOURCES=list(batch['source_sha256'])
    else:
        if progress.exists() or manifest_path.exists():raise FileExistsError('Preserve existing final study; use --resume')
        if not a.wait_for_development:raise ValueError('Explicit --wait-for-development required')
        (OUT/'waiting.json').write_text(json.dumps(dict(status='waiting_for_all_ten_development_successes',protocol_sha256=sha(PROTOCOL),planned_cases=50,planned_trials=150,old_100000_series_used=False),indent=2)+'\n')
        print('Waiting for ten correctly configured, verified wider development cases; no final case opened.',flush=True)
        while True:
            prior=gate()
            if prior is False:return
            if prior is not None:break
            time.sleep(5)
        cases=make_cases();manifest=dict(purpose='FINAL_frozen_conditional_SC_no_tuning',protocol_sha256=sha(PROTOCOL),cases=cases,old_100000_series_used=False)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');runner.SOURCES=list(prior['source_sha256'])
        for path in [Path(__file__),PROTOCOL,manifest_path,ROOT/'experiments/analyze_frozen_aligned_final31.py']:
            name=str(path.relative_to(ROOT)).replace('\\','/')
            if name not in runner.SOURCES:runner.SOURCES.append(name)
        batch={k:prior[k] for k in ['remote_support','physics_step','cable_initialization','grasp_profile','solver_iterations_requested','physics_backend','weight_overrides','positive_control_proof_file','positive_control_proof_sha256']}
        order_rng=np.random.default_rng(protocol['method_order_seed'])
        batch.update(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='FINAL_frozen_conditional_SC_no_tuning',status='running',
            protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(manifest_path),validated_development_manifest_sha256=prior['manifest_sha256'],
            source_sha256={s:sha(ROOT/s) for s in runner.SOURCES},method_order_seed=protocol['method_order_seed'],
            method_order_by_case={c['case_id']:order_rng.permutation(protocol['methods']).tolist() for c in cases},results=[],
            planned_cases=50,planned_trials=150,old_100000_series_used=False,unrestricted_rotation_or_SFP_evaluated=False)
        progress.write_text(json.dumps(batch,indent=2)+'\n')
    completed={r['run'] for r in batch['results']};(OUT/'waiting.json').write_text(json.dumps(dict(status='final_started_or_resumed',progress='progress.json'),indent=2)+'\n')
    for case in manifest['cases']:
        for method in batch['method_order_by_case'][case['case_id']]:
            name=case['case_id']+'_'+method
            if name in completed:continue # Includes failed attempts; never replace failures.
            if (ROOT/'.artifacts/lastinch_runs'/name).exists():raise RuntimeError('Unindexed final archive exists; inspect before any resume')
            if shutil.disk_usage(ROOT).free<protocol['disk_stop_floor_GiB']*2**30:
                batch['status']='stopped_disk_floor';progress.write_text(json.dumps(batch,indent=2)+'\n');print('Final stopped at disk floor; all existing evidence preserved.',flush=True);return
            runner.trial(case,method,batch);progress.write_text(json.dumps(batch,indent=2)+'\n');completed.add(name)
    batch['status']='complete';progress.write_text(json.dumps(batch,indent=2)+'\n')
    print(json.dumps(dict(status='final_complete',trials=len(batch['results']),old_100000_series_used=False)),flush=True)
    from experiments.analyze_frozen_aligned_final31 import main as analyze
    analyze()


def main():
    OUT.mkdir(exist_ok=True);lock=OUT/'runner.lock'
    try:
        with lock.open('x') as f:f.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError('Final runner lock exists; verify its owner before any restart') from None
    # Process-scoped request only: permit the display to turn off, keep local
    # compute awake while queued/running, and restore on exit. No power-plan edit.
    power=None
    if os.name=='nt':
        import ctypes
        power=ctypes.windll.kernel32
        requested=bool(power.SetThreadExecutionState(0x80000001))
        (OUT/'execution_power.json').write_text(json.dumps(dict(system_awake_request=requested,display_required=False,power_plan_modified=False,cleared_on_process_exit=True),indent=2)+'\n')
    try:run()
    finally:
        if power is not None:power.SetThreadExecutionState(0x80000000)
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
