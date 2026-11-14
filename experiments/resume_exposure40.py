"""Finish frozen V37, then install/test one exposure-history repair and pilot it.

Finite development work only. Preserve all prior results; never open final seeds.
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
from experiments.run_qualification39 import host_awake_seconds
OUT=ROOT/'.artifacts/repair40'
METHOD='original_perception_sfp_contact_datum_handoff'
WEIGHTS=['.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
         '.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt']


def main():
    state=dict(status='waiting_for_frozen_qualification39_completion',held_out_final_trials_run=0,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paper_ready=False)
    path=OUT/'resume_progress.json'
    if path.exists():raise RuntimeError('Repair boundary worker already has a record; do not overwrite it')
    def write():path.write_text(json.dumps(state,indent=2)+'\n')
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):raise RuntimeError('Idle-sleep request failed')
    write()
    try:
        deadline=host_awake_seconds()+7200
        while True:
            q=json.loads((ROOT/'.artifacts/repair39/qualification39v1/progress.json').read_text())
            if q['status']=='development_complete_needs_analysis':break
            if q['status']!='executing_development_case':raise RuntimeError('Prior screen stopped unexpectedly: '+q['status'])
            if host_awake_seconds()>deadline:raise TimeoutError('Prior screen did not finish in two working hours')
            time.sleep(5)
        assert len(q['results'])==10 and all(r['status']=='completed' for r in q['results'])
        state['prior_completed']=10;state['prior_verified']=sum(r['verified_success'] for r in q['results'])
        (OUT/'qualification39_completed.json').write_text(json.dumps(q,indent=2)+'\n')
        for name,digest in q['source_and_checkpoint_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+name)
        if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Less than 5 GiB free')
        subprocess.run(['docker','stop','research-aic-eval'],check=True,stdout=subprocess.DEVNULL,timeout=60)
        meta=json.loads((OUT/'exposure_candidate/candidate.json').read_text())
        target=ROOT/meta['source'];candidate=OUT/'exposure_candidate/sfp_fixture_refinement.py'
        assert hashlib.sha256(target.read_bytes()).hexdigest()==meta['before_sha256']
        assert hashlib.sha256(candidate.read_bytes()).hexdigest()==meta['candidate_sha256']
        harness=ROOT/'experiments/run_original_insertion.py'
        before={target:target.read_bytes(),harness:harness.read_bytes()}
        for source,data in before.items():
            backup=OUT/'source_v37'/source.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True)
            if backup.exists():raise RuntimeError('Backup exists; stop instead of overwriting')
            backup.write_bytes(data)
        assert b"result['harness_revision']=37" in before[harness]
        state['status']='installing_and_checking_exposure_repair';write()
        target.write_bytes(candidate.read_bytes())
        harness.write_bytes(before[harness].replace(b"result['harness_revision']=37",b"result['harness_revision']=38"))
        checks=['experiments/test_exposure40.py','experiments/test_lip_search39.py',
                'experiments/test_speed38.py','experiments/test_wrist_causal38.py',
                'aic_model/test/test_sfp_fixture_refinement.py','aic_model/test/test_sfp_contact_datum.py',
                'aic_model/test/test_tracking_compensation.py','aic_model/test/test_contact_registration.py']
        import os
        env=dict(os.environ,PYTHONPATH=str(ROOT)+';'+str(ROOT/'aic_model')+';'+str(ROOT/'aic_example_policies'))
        with (OUT/'installation_checks.log').open('wb') as log:
            check=subprocess.run([sys.executable,'-m','pytest','-q',*checks,'-k','not real_ros_observation'],
                                 cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if check.returncode:
            for source,data in before.items():source.write_bytes(data)
            raise RuntimeError('Checks failed; prior production bytes restored')
        files=set(SOURCES)|set(WEIGHTS)|{'experiments/resume_exposure40.py','experiments/test_exposure40.py',
            'experiments/configs/sfp_contact34_development.json','.artifacts/sfp_damped27/contact_proof.json'}
        hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(files)}
        freeze=dict(source_sha256=hashes,harness_revision=38,profile='fast38',case_order=[45001,45002,45003,45000],
                    held_out_final_trials_run=0,measurement_and_seating_criteria='unchanged')
        (OUT/'exposure40v1_source_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
        state.update(status='executing_four_case_development_pilot',current_run='exposure40v1');write()
        active=ROOT/'.artifacts/active_original_perception29.json';d=json.loads(active.read_text())
        d.update(status=state['status'],harness_revision=38,current_run='dev_45001_exposure40v1_'+METHOD,
                 live_insertion_world=True,source_freeze='.artifacts/repair40/exposure40v1_source_freeze.json',paper_ready=False)
        active.write_text(json.dumps(d,indent=2)+'\n')
        args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest','experiments/configs/sfp_contact34_development.json',
            '--case-ids','dev_45001','dev_45002','dev_45003','dev_45000','--methods',METHOD,'--label','exposure40v1',
            '--physics-backend','source_float64_damped','--physics-step','0.0005','--wrist-rate','0','--solver-iterations','200',
            '--positive-control-proof','.artifacts/sfp_damped27/contact_proof.json','--nic-weights',WEIGHTS[0],
            '--sfp-plug-weights',WEIGHTS[1],'--force-ceiling-N','20','--sfp-motion-profile','fast38']
        with (OUT/'pilot.log').open('wb') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        batch=json.loads((ROOT/'.artifacts/evaluation_seeded.json').read_text())
        assert len(batch['results'])==4 and all('_exposure40v1_' in r['run'] for r in batch['results'])
        (OUT/'pilot_completed.json').write_text(json.dumps(batch,indent=2)+'\n')
        state.update(status='pilot_complete_needs_analysis',results=batch['results']);write()
    except Exception as exc:
        state.update(status='stopped_on_execution_error',error=f'{type(exc).__name__}: {exc}');write();raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
