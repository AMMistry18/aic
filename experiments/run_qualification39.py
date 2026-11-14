"""Finite serial SFP development screen, gated on both frozen V37 repairs.

Uses existing scene/recorder/verifier runner. No final seeds, tuning, parallel
worlds, case replacement or cleanup. Stops on execution/storage/source errors.
"""
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_seeded_evaluation import SOURCES
OUT=ROOT/'.artifacts/repair39/qualification39v1'
METHOD='original_perception_sfp_contact_datum_handoff'
MANIFEST='experiments/configs/sfp_contact34_development.json'
WEIGHTS=['.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
         '.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt']
REQUIRED=[f'dev_{seed}_lip39v1_{METHOD}' for seed in [45001,45000]]


def host_awake_seconds():
    """Resource-wait clock excluding OS sleep; never used by the policy.

    Microsoft QueryUnbiasedInterruptTime measures working-state time in 100 ns.
    See https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryunbiasedinterrupttime
    """
    ticks=ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(ticks)):
        raise RuntimeError('Could not read host working-state clock')
    return ticks.value*1e-7


def main():
    OUT.mkdir(exist_ok=False)
    state=dict(status='awaiting_both_pilot_results',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               prerequisite_runs=REQUIRED,results=[],held_out_final_trials_run=0,paper_ready=False)
    def write(active=False):
        (OUT/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
        if active:
            path=ROOT/'.artifacts/active_original_perception29.json';d=json.loads(path.read_text())
            d.update(status='qualification39v1_'+state['status'],current_run=state.get('current_run'),
                live_insertion_world=state['status']=='executing_development_case',
                qualification_progress='.artifacts/repair37/qualification39v1/progress.json',
                qualification_completed=len(state['results']),paper_ready=False,held_out_final_trials_run=0)
            path.write_text(json.dumps(d,indent=2)+'\n')
    power=ctypes.windll.kernel32
    if not power.SetThreadExecutionState(0x80000001):raise RuntimeError('Idle-sleep request failed')
    try:
        state['prerequisite_wait_timebase']='Windows working-state time excluding OS sleep; policy budgets unchanged'
        write();deadline=host_awake_seconds()+3600
        while True:
            try:
                batch=json.loads((ROOT/'.artifacts/evaluation_seeded.json').read_text())
                by_run={r['run']:r for r in batch['results']}
                if all(name in by_run for name in REQUIRED):break
            except (FileNotFoundError,json.JSONDecodeError):pass
            if host_awake_seconds()>deadline:raise TimeoutError('Both pilot results were not completed within one hour of host working time')
            time.sleep(5)
        state['prerequisite_results']=[by_run[name] for name in REQUIRED]
        if not all(by_run[name].get('verified_success') for name in REQUIRED):
            state.update(status='not_started_pilot_gate_failed',current_run=None);write();return
        assert batch['force_ceiling_N']==20. and batch['sfp_motion_profile']=='fast38'
        assert batch['physics_step']==.0005 and batch['wrist_rate']==0 and batch['solver_iterations_requested']==200
        # The pilot source/checkpoint map must still match, not just its verdict.
        for name,digest in batch['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Pilot source changed: '+name)
        proof='.artifacts/sfp_damped27/contact_proof.json'
        assert hashlib.sha256((ROOT/proof).read_bytes()).hexdigest()==batch['positive_control_proof_sha256']
        sources=set(SOURCES)|set(WEIGHTS)|{MANIFEST,proof,'experiments/run_qualification39.py'}
        hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)}
        state['source_and_checkpoint_sha256']=hashes
        state['planned_cases']=json.loads((ROOT/MANIFEST).read_text())['cases']
        for case in state['planned_cases']:
            if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Less than 5 GiB free; preserve existing evidence and stop')
            for name,digest in hashes.items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+name)
            run=case['case_id']+'_qualification39v1_'+METHOD
            state.update(status='executing_development_case',current_run=run);write(True)
            args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest',MANIFEST,
                '--case-ids',case['case_id'],'--methods',METHOD,'--label','qualification39v1',
                '--physics-backend','source_float64_damped','--physics-step','0.0005','--wrist-rate','0',
                '--solver-iterations','200','--positive-control-proof','.artifacts/sfp_damped27/contact_proof.json',
                '--nic-weights',WEIGHTS[0],'--sfp-plug-weights',WEIGHTS[1],
                '--force-ceiling-N','20','--sfp-motion-profile','fast38']
            with (OUT/(case['case_id']+'.log')).open('wb') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            measured=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text())
            assert len(measured['results'])==1 and measured['results'][0]['run']==run
            (OUT/(case['case_id']+'_batch.json')).write_text(json.dumps(measured,indent=2)+'\n')
            result=measured['results'][0];state['results'].append(result);write(True)
            if result['status']!='completed':raise RuntimeError('Execution failed; remaining cases not run')
        state.update(status='development_complete_needs_analysis',current_run=None);write(True)
    except Exception as exc:
        state.update(status='stopped_on_execution_error',error=f'{type(exc).__name__}: {exc}',current_run=None)
        write('source_and_checkpoint_sha256' in state)
        raise
    finally:power.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
