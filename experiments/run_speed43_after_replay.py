"""At the completed baseline boundary, run six matched speed development cases.

No final cases are generated or opened. Every completed outcome is exported.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/recovery_20261112/speed43_development'
CANDIDATE=ROOT/'experiments/recovery_20261112/speed43_candidate'
PLAN=[('sc',42000),('sfp',45005),('sc',43001),('sfp',45009),('sc',43003),('sfp',45002)]
METHODS=dict(sc='original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff',
             sfp='original_perception_sfp_contact_datum_handoff')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    resume='--resume' in sys.argv
    OUT.mkdir(exist_ok=resume)
    if resume:
        previous=json.loads((OUT/'progress.json').read_text())
        if previous['status']!='stopped' or previous.get('installed') or previous['results']:
            raise RuntimeError('Resume only before candidate installation and before trials')
        (OUT/('progress_before_resume_'+str(int(time.time()))+'.json')).write_text(json.dumps(previous,indent=2))
    metadata=json.loads((CANDIDATE/'changes.json').read_text())
    state=dict(status='waiting_for_completed_baseline',plan=PLAN,results=[],candidate=metadata,
               held_out_trials_run=0,raw_force_ceiling_N=20,qualification_claimed=False)
    def save():
        state['updated_unix']=time.time();(OUT/'progress.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        deadline=time.monotonic()+6*3600
        baseline_path=ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json'
        while True:
            baseline=json.loads(baseline_path.read_text())
            if baseline['status']=='complete_requires_analysis':break
            if baseline['status']=='stopped':raise RuntimeError('Baseline stopped; inspect failure before speed trials')
            if time.monotonic()>deadline:raise TimeoutError('Baseline did not finish within six hours')
            time.sleep(10)
        sys.path.insert(0,str(ROOT))
        from experiments.verify_recovery_batch_20261112 import completed_case_ids
        completed_cases=completed_case_ids(baseline['results'])
        expected={'dev_'+str(seed) for seed in [42000,45005,43001,45009,43003,45002,43004,45001,42001,
                  45008,42004,45007,42003,45004,43000,45006,42002]}
        if completed_cases!=expected:raise RuntimeError('Expected all 17 historical cases; preserve extra setup attempts')
        (OUT/'matched_baseline.json').write_text(json.dumps(baseline,indent=2))
        for path,d in metadata['files'].items():
            if sha(ROOT/path)!=d['before_sha256'] or sha(CANDIDATE/path)!=d['candidate_sha256']:
                raise RuntimeError('Source/candidate changed: '+path)
        # Complete all checks before the first mutation; preserve every original.
        for path in metadata['files']:
            dest=OUT/'baseline_sources'/path;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/path,dest)
        for path in metadata['files']:shutil.copy2(CANDIDATE/path,ROOT/path)
        state.update(status='candidate_installed_for_development',installed=True);save()
        with (OUT/'unit_checks.log').open('wb') as log:
            subprocess.run([sys.executable,'-m','pytest','experiments/test_speed43.py','-q'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        sys.path.insert(0,str(ROOT))
        from experiments.run_seeded_evaluation import SOURCES
        frozen={p:sha(ROOT/p) for p in SOURCES}
        state['source_sha256']=frozen;save()
        for kind,seed in PLAN:
            if any(sha(ROOT/p)!=h for p,h in frozen.items()):raise RuntimeError('Candidate source changed')
            case=f'dev_{seed}';method=METHODS[kind]
            state.update(status='running',current_case=case);save()
            args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest',
                'experiments/configs/'+('sc_finish41_qualification.json' if kind=='sc' else 'sfp_contact34_development.json'),
                '--case-ids',case,'--methods',method,'--label','speed43v1',
                '--physics-backend','source_float64_damped','--physics-step','0.0005','--wrist-rate','0',
                '--solver-iterations','200','--force-ceiling-N','20','--positive-control-proof',
                'experiments/recovery_20261112/revalidated_historical_contact_proof.json']
            weights='experiments/recovery_20261112/checkpoints/'
            if kind=='sc':args+=['--grasp-profile','challenge_sc','--cable-initialization','sc_straight_outboard',
                               '--sc-mouth-weights',weights+'sc_mouth_recovery20261112/best.pt']
            else:args+=['--sfp-motion-profile','fast43','--nic-weights',weights+'nic_recovery20261112/best.pt',
                        '--sfp-plug-weights',weights+'plug_recovery20261112/best.pt']
            with (OUT/(case+'.log')).open('wb') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2000)
            batch=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text())
            if len(batch['results'])!=1 or batch['results'][0]['run']!=case+'_speed43v1_'+method:
                raise RuntimeError('Result identity mismatch')
            (OUT/(case+'_batch.json')).write_text(json.dumps(batch,indent=2))
            result=batch['results'][0];state['results'].append(result);save()
            if result['status']!='completed':raise RuntimeError('Environment failure; inspect preserved evidence')
            with (OUT/(case+'_export.log')).open('wb') as log:
                subprocess.run([sys.executable,'experiments/export_original_repair_evidence.py',
                    '--runs',result['run'],'--out',str(OUT/'evidence'/case)],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        state.update(status='complete_requires_speed_and_failure_analysis',current_case=None);save()
    except Exception as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
