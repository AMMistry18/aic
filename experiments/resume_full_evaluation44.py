"""Resume the frozen final manifest without retrying or replacing any attempt.

This coordinator leaves all frozen source files untouched. It only reconciles
interrupted bookkeeping and recovers container-shutdown transport failures.
Policy, setup, action deadlines, verification and case order are unchanged.
"""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import run_seeded_evaluation as executor
from experiments import run_full_evaluation44 as original
from experiments.full_study44 import summarize,paired_exact_mcnemar

OUT=ROOT/'experiments/full_evaluation44'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs_for(manifest):
    cases=manifest['cases'];known={c['case_id']:c for c in cases}
    if len(known)!=100:raise ValueError('Expected 100 distinct frozen primary cases')
    jobs=[(c,'full') for c in cases]
    jobs.extend((known[name],mode) for name in manifest['paired_subset']
                for mode in ('no_axial_correction','no_lateral_tracking'))
    if len(jobs)!=180:raise ValueError('Expected 80 frozen paired ablations')
    return jobs


def run_name(case,mode):
    method=original.SC if case['connector']=='sc' else original.SFP
    return case['case_id']+'_final44v1_'+mode+'_'+method


def validate_prefix(rows,jobs):
    if len(rows)>len(jobs):raise ValueError('Too many results')
    for row,(case,mode) in zip(rows,jobs):
        expected=copy.deepcopy(case);expected['case_id']+='_final44v1_'+mode
        if (row['run']!=run_name(case,mode) or row['base_case_id']!=case['case_id']
            or row['ablation']!=mode or row['case']!=expected or not row['final']):
            raise ValueError('Result is not the exact frozen manifest prefix')


def stopped_container():
    result=subprocess.run(['docker','inspect',executor.CONTAINER,'--format','{{.State.Running}}'],
                          capture_output=True,text=True,check=True,timeout=30)
    if result.stdout.strip()=='true':
        # Same owned container only; this transport cleanup does not extend a
        # policy action or alter physics/verification thresholds.
        subprocess.run(['docker','stop',executor.CONTAINER],capture_output=True,
                       check=True,timeout=120)


def recover_uncommitted(case,mode,error):
    run=run_name(case,mode);root=ROOT/'.artifacts/kilted_handoff'
    archived=ROOT/'.artifacts/lastinch_runs'/run
    stopped_container()
    if root.exists() and archived.exists():raise RuntimeError('Ambiguous live/archive scene; inspect before resuming')
    scene=root if root.exists() else archived;folder=scene/run
    if not scene.exists():raise RuntimeError('Missing uncommitted scene; inspect before resuming')
    if not (folder/'verified.json').exists() and all((folder/n).exists() for n in
            ['policy.json','evaluation.json','status.json','wrench_raw.json.gz','truth_trace.json.gz']):
        if not json.loads((folder/'status.json').read_text())['active']:
            for script,args,kind in [('experiments/analyze_recorded_wrench.py',['--run-dir',str(folder)],'wrench'),
                                     ('experiments/verify_recorded_trial.py',['--scene-dir',str(scene),'--run',run],'verify')]:
                original.run([sys.executable,script,*args],OUT/(run+'_recovered_'+kind+'.log'),180)
    if (folder/'verified.json').exists():
        result=json.loads((folder/'verified.json').read_text());result['status']='completed'
        result['peak_independent_tared_force_N']=json.loads((folder/'wrench_analysis.json').read_text())['peak_independent_tared_force_N']
    else:
        # A setup/policy interruption remains a failed attempt. It is never
        # rerun as a replacement for this selected final case.
        result=dict(run=run,status='environment_or_execution_failure',
                    reason='setup_interruption_before_policy' if not (folder/'policy.json').exists() else 'execution_interruption',
                    error=error,policy_started=(folder/'policy.json').exists())
    archive=executor.archive(run) if root.exists() else archived
    selected=copy.deepcopy(case);selected['case_id']+='_final44v1_'+mode
    result.update(case=selected,method=original.SC if case['connector']=='sc' else original.SFP,
                  evidence_dir=str(archive.relative_to(ROOT)),base_case_id=case['case_id'],ablation=mode,final=True)
    return result


def main():
    state=json.loads((OUT/'progress.json').read_text())
    freeze=json.loads((OUT/'source_freeze.json').read_text());hashes=freeze['source_sha256']
    manifest_path=OUT/'final_manifest.json';manifest=json.loads(manifest_path.read_text())
    if sha(manifest_path)!=state['manifest_sha256']:raise RuntimeError('Final manifest changed')
    if sha(OUT/'source_freeze.json')!=manifest['source_freeze_sha256']:raise RuntimeError('Source freeze changed')
    if freeze['selection']['selected']!='recovered_baseline':raise RuntimeError('Expected the paper baseline')
    def check_sources():
        if sha(manifest_path)!=state['manifest_sha256']:raise RuntimeError('Final manifest changed')
        for path,digest in hashes.items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    check_sources();jobs=jobs_for(manifest);validate_prefix(state['results'],jobs)
    if state['status']!='stopped_requires_inspection':raise RuntimeError('Resume requires an inspected stopped job')
    stamp=str(int(time.time()));shutil.copy2(OUT/'progress.json',OUT/f'progress_before_resume_{stamp}.json')
    resume_file=OUT/f'resume_coordinator_{stamp}.py';shutil.copy2(Path(__file__),resume_file)
    metadata=dict(coordinator_sha256=sha(resume_file),coordinator_file=str(resume_file.relative_to(ROOT)),
                  unchanged_source_freeze_sha256=sha(OUT/'source_freeze.json'),manifest_sha256=sha(manifest_path),
                  policy_or_verification_changes=False,no_retries_or_replacements=True,started_unix=time.time(),
                  original_error=state.get('error'),events=[])
    state.setdefault('resume_history',[]).append(metadata)
    def save():
        state['updated_unix']=time.time()
        state['summary']=summarize([r for r in state['results'] if r['ablation']=='full'])
        state['ablation_summary']={mode:summarize([r for r in state['results'] if r['ablation']==mode])
            for mode in ('no_axial_correction','no_lateral_tracking')}
        original.write(OUT/'progress.json',state)
    def retain(result,batch):
        dest=OUT/'trials'/result['run'];dest.mkdir(parents=True,exist_ok=False)
        batch['results']=[result];original.write(dest/'batch.json',batch)
        # Commit the outcome before export; an export failure must never cause
        # this selected case to be executed again.
        state['results'].append(result);state['current_run']=None;save()
        if result['status']=='completed':
            original.run([sys.executable,'experiments/export_original_repair_evidence.py','--runs',result['run'],
                          '--out',str(dest/'evidence')],dest/'export.log',180)
            policy=ROOT/result['evidence_dir']/result['run']/'policy.json'
            if json.loads(policy.read_text()).get('research_ablation','full')!=result['ablation']:
                raise RuntimeError('Runtime ablation mismatch')
        else:
            logs=dest/'setup_logs';logs.mkdir()
            for path in (ROOT/'.artifacts/container').glob(result['run']+'*'):
                if path.is_file():shutil.copy2(path,logs/path.name)
    def batch_for(case,mode):
        kind=case['connector'];weights=original.WEIGHTS
        overrides=({'AIC_SC_POSE_WEIGHTS':str(weights[0].relative_to(ROOT)).replace('\\','/')} if kind=='sc' else
                   {'AIC_NIC_WEIGHTS':str(weights[1].relative_to(ROOT)).replace('\\','/'),
                    'AIC_SFP_PLUG_POSE_WEIGHTS':str(weights[2].relative_to(ROOT)).replace('\\','/')})
        proof=OUT/'contact_positive_control.json'
        return dict(source_sha256=hashes,purpose='held_out_final44',remote_support='0',physics_step=.0005,
                    cable_initialization='sc_straight_outboard' if kind=='sc' else 'source',
                    grasp_profile='challenge_sc' if kind=='sc' else 'legacy_sfp',wrist_rate=0,force_ceiling_N=20.,
                    sfp_motion_profile='fast38' if kind=='sfp' else 'legacy36',solver_iterations_requested=200,
                    physics_backend='source_float64_damped',positive_control_proof_file=str(proof.resolve()),
                    positive_control_proof_sha256=sha(proof),weight_overrides=overrides,ablation=mode,
                    resume_coordinator_sha256=metadata['coordinator_sha256'],results=[])
    executor.SOURCES[:]=sorted(hashes)
    try:
        # The prior coordinator can lose its local result if Docker stop raises
        # in finally. Reconcile that exact attempt before opening another case.
        if state.get('current_run'):
            case,mode=jobs[len(state['results'])]
            if run_name(case,mode)!=state['current_run']:raise RuntimeError('Uncommitted run identity mismatch')
            result=recover_uncommitted(case,mode,state['error'])
            metadata['events'].append(dict(run=result['run'],operation='recover_uncommitted_attempt',reason=result.get('reason')))
            retain(result,batch_for(case,mode))
        state.pop('error',None)
        consecutive=0
        for result in reversed(state['results']):
            if result['status']=='completed':break
            consecutive+=1
        for case,mode in jobs[len(state['results']):]:
            if consecutive>=3:raise RuntimeError('Three consecutive environment failures; remaining cases unattempted')
            check_sources();validate_prefix(state['results'],jobs)
            if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Safe storage boundary: below 8 GiB free')
            state.update(status='primary_final_trials' if mode=='full' else 'paired_final_ablations',current_run=run_name(case,mode));save()
            batch=batch_for(case,mode);selected=copy.deepcopy(case);selected['case_id']+='_final44v1_'+mode
            begin=time.monotonic()
            try:
                executor.trial(selected,original.SC if case['connector']=='sc' else original.SFP,batch)
                result=batch['results'][0]
                result.update(base_case_id=case['case_id'],ablation=mode,final=True)
            except Exception as exc:
                result=recover_uncommitted(case,mode,f'{type(exc).__name__}: {exc}')
                metadata['events'].append(dict(run=result['run'],operation='recover_shutdown_transport_error',error=str(exc)))
            result['coordinator_elapsed_wall_s']=time.monotonic()-begin
            retain(result,batch)
            consecutive=consecutive+1 if result['status']!='completed' else 0
        comparisons=[]
        for kind in ('sc','sfp'):
            rows=[r for r in state['results'] if r['case']['connector']==kind]
            for mode in ('no_axial_correction','no_lateral_tracking'):
                comparisons.append(dict(connector=kind,ablation=mode,**paired_exact_mcnemar(rows,mode)))
        running=0.
        for rank,test in enumerate(sorted(comparisons,key=lambda r:r['exact_two_sided_p'])):
            running=max(running,min(1.,(len(comparisons)-rank)*test['exact_two_sided_p']));test['holm_adjusted_p']=running
        original.write(OUT/'final_summary.json',dict(primary=summarize([r for r in state['results'] if r['ablation']=='full']),
            paired_comparisons=comparisons,all_trials=len(state['results']),sources_frozen=True,failures_retained=True))
        state.update(status='complete_requires_scientific_review',current_run=None);save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':
    power=ctypes.windll.kernel32;power.SetThreadExecutionState(0x80000001)
    try:main()
    finally:power.SetThreadExecutionState(0x80000000)
