"""Finite overnight study: development gates, freeze, 100 fresh +80 paired trials.

Final cases are generated only after development selection. Final outcomes never
select or modify a policy. Source changes stop the job, and all attempts survive.
"""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.full_study44 import connector_cases,summarize,paired_exact_mcnemar

BASE=ROOT/'experiments/recovery_20261112'
OUT=ROOT/'experiments/full_evaluation44'
SC='original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff'
SFP='original_perception_sfp_contact_datum_handoff'
WEIGHTS=[BASE/'checkpoints'/name/'best.pt' for name in
         ['sc_mouth_recovery20261112','nic_recovery20261112','plug_recovery20261112']]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def run(args,log,timeout=2200):
    with log.open('wb') as stream:
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join(str(p) for p in
            [ROOT/'aic_model',ROOT/'aic_example_policies',ROOT/'flowstate/aic_perception',ROOT])
        subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)


def seed_checkpoint_objects():
    objects=ROOT/'.artifacts/frozen_checkpoint_objects';objects.mkdir(exist_ok=True)
    originals=ROOT/'aic_example_policies/aic_example_policies/ros/weights'
    for path in [*WEIGHTS,*[originals/n for n in ['best.pt','best_sc_mouth_pose.pt','best_sc_plug_pose.pt','best_sfp_plug_pose.pt']]]:
        digest=sha(path);dest=objects/(digest+'.pt')
        if not dest.exists():shutil.copy2(path,dest)
        if sha(dest)!=digest:raise RuntimeError('Checkpoint object corrupt')


def select_speed(baseline,speed):
    passed=len(speed['results'])==6 and all(r.get('verified_success') for r in speed['results'])
    comparison={}
    for connector in ['sc','sfp']:
        new=[r for r in speed['results'] if r['case']['connector']==connector]
        identities={r['run'].split('_speed')[0] for r in new}
        old=[r for r in baseline['results'] if r['run'].split('_recovery')[0] in identities and r.get('verified_success')]
        old_median=statistics.median(r['sim_s'] for r in old) if old else None
        new_median=statistics.median(r['sim_s'] for r in new if r.get('verified_success')) if any(r.get('verified_success') for r in new) else None
        improved=bool(len(old)==3 and len(new)==3 and new_median is not None and new_median<=.9*old_median)
        comparison[connector]=dict(baseline_median_sim_s=old_median,candidate_median_sim_s=new_median,passes_10percent=improved)
        passed=passed and improved
    return dict(selected='speed43' if passed else 'recovered_baseline',comparison=comparison,
                rule='All six verified and >=10% smaller median action simulation time for each connector; otherwise baseline')


def main():
    resume_setup='--resume-setup' in sys.argv
    resume_speed_stop='--resume-after-speed-stop' in sys.argv
    direct_final='--start-final-now' in sys.argv
    if sum([resume_setup,resume_speed_stop,direct_final])>1:raise ValueError('Select one entry boundary')
    OUT.mkdir(exist_ok=resume_setup or resume_speed_stop or direct_final)
    if direct_final:
        state=json.loads((OUT/'progress.json').read_text())
        if state['final_trials_started'] or state['results'] or (OUT/'source_freeze.json').exists():
            raise RuntimeError('Direct final entry cannot replace an opened or frozen final study')
        stopped=json.loads((BASE/'speed43_development/progress.json').read_text())
        if stopped.get('installed') or not stopped.get('baseline_restored'):
            raise RuntimeError('The faster profile must remain stopped and the baseline restored')
        for path,meta in stopped['candidate']['files'].items():
            if sha(ROOT/path)!=meta['before_sha256']:raise RuntimeError('Restored baseline changed: '+path)
    elif resume_speed_stop:
        state=json.loads((OUT/'progress.json').read_text())
        if state['final_trials_started'] or state['results'] or state['status']!='speed_development':
            raise RuntimeError('Speed-stop resume requires the unopened-final speed boundary')
        if len(state['development'])!=1 or not state['development'][0].get('verified_success'):
            raise RuntimeError('Expected the completed recording regression only')
        stopped=json.loads((BASE/'speed43_development/progress.json').read_text())
        if stopped['status']!='stopped_by_user' or stopped.get('installed') or not stopped.get('baseline_restored'):
            raise RuntimeError('User-stopped speed profile must already be restored')
        for path,meta in stopped['candidate']['files'].items():
            if sha(ROOT/path)!=meta['before_sha256']:raise RuntimeError('Restored baseline changed: '+path)
    elif resume_setup:
        state=json.loads((OUT/'progress.json').read_text())
        if state['final_trials_started'] or state['results'] or state['development']:
            raise RuntimeError('Setup resume requires no trial outcomes and no final cases opened')
        if state['status']!='waiting_for_recording_regression':raise RuntimeError('Unexpected setup state')
    else:
        state=dict(status='waiting_for_recording_regression',results=[],development=[],final_trials_started=False,
                   planned_primary=100,planned_ablation=80,started_unix=time.time())
    def save():
        state['updated_unix']=time.time()
        state['summary']=summarize([r for r in state['results'] if r['ablation']=='full'])
        state['ablation_summary']={mode:summarize([r for r in state['results'] if r['ablation']==mode])
            for mode in ['no_axial_correction','no_lateral_tracking']}
        write(OUT/'progress.json',state)
    def phase(name,**extra):state.update(status=name,**extra);save()
    save();seed_checkpoint_objects()
    try:
        if direct_final:
            choice=dict(selected='recovered_baseline',rule='User explicitly requested final study now; faster profile remains stopped',
                        remaining_development_checks_skipped=True)
            profile='fast38'
            proof=OUT/'contact_positive_control.json'
            if not json.loads(proof.read_text()).get('qualified'):
                raise RuntimeError('Existing same-contract contact proof is unavailable')
            state['preliminary_checks_override']=dict(
                instruction='forget the current checks, run the proper final test for the paper',
                skipped=['further preliminary-check repetitions','four development ablation interface trials'],
                contact_control='Already finished; existing recording scored without another simulation',
                measurement_checks_already_passed=78,
                retained=['per-trial controller/load/camera setup','independent seating/contact verification',
                          'complete 2000 Hz raw force audit','20N raw resultant ceiling'],
                contact_proof=str(proof.relative_to(ROOT)),contact_proof_sha256=sha(proof))
            write(OUT/'development_selection.json',choice)
            phase('freezing_final_protocol',selected_policy='recovered_baseline')
        else:
            # An already-running owned regression must finish before another world starts.
            identity='dev_45006_coverage44v1_'+SFP
            deadline=time.monotonic()+2200
            while True:
                p=OUT/'recording_regression.json' if resume_speed_stop else ROOT/'.artifacts/evaluation_seeded.json'
                if p.exists():
                    batch=json.loads(p.read_text());matches=[r for r in batch.get('results',[]) if r['run']==identity]
                    if matches:
                        result=matches[0];break
                if time.monotonic()>deadline:raise TimeoutError('Recording regression did not finish')
                save();time.sleep(10)
            write(OUT/'recording_regression.json',batch)
            if not resume_speed_stop:
                state['development'].append(result);save()
            elif state['development'][0]['run']!=result['run']:
                raise RuntimeError('Recording regression identity changed')
            if not result.get('verified_success'):raise RuntimeError('Recording regression failed; inspect before final evaluation')
            archive=ROOT/result['evidence_dir'];evaluation=json.loads((archive/identity/'evaluation.json').read_text())
            if not evaluation['post_policy_recording_coverage']['completed']:raise RuntimeError('Fixed coverage-only tail incomplete')
            if not resume_speed_stop:
                run([sys.executable,'experiments/export_original_repair_evidence.py','--runs',result['run'],
                     '--out',str(OUT/'development_evidence/recording_regression')],OUT/'recording_regression_export.log',180)
            phase('speed_development')
            if not resume_speed_stop:
                run([sys.executable,'experiments/run_speed43_after_replay.py','--resume'],OUT/'speed_queue.log',12000)
            speed=json.loads((BASE/'speed43_development/progress.json').read_text())
            baseline=json.loads((BASE/'historical_batch_replay/progress.json').read_text())
            if resume_speed_stop:
                choice=dict(selected='recovered_baseline',rule='User stopped the faster profile; preserve five outcomes and cancel the sixth',
                            completed_speed_trials=len(speed['results']),verified_speed_trials=sum(bool(r.get('verified_success')) for r in speed['results']),
                            speed_qualification_claimed=False)
            else:
                if speed['status']!='complete_requires_speed_and_failure_analysis':raise RuntimeError('Speed stage unfinished')
                choice=select_speed(baseline,speed)
            write(OUT/'development_selection.json',choice)
            if choice['selected']=='recovered_baseline':
                # Restore only files installed by this owned candidate, after checking
                # that no human or other process changed them during development.
                for path,meta in speed['candidate']['files'].items():
                    expected=meta['before_sha256'] if resume_speed_stop else meta['candidate_sha256']
                    if sha(ROOT/path)!=expected:raise RuntimeError('Candidate/baseline changed: '+path)
                if not resume_speed_stop:
                    for path in speed['candidate']['files']:
                        shutil.copy2(BASE/'speed43_development/baseline_sources'/path,ROOT/path)
            profile='fast43' if choice['selected']=='speed43' else 'fast38'
            phase('measurement_negative_controls')
            run([sys.executable,'-m','pytest','experiments/test_recording_tail.py',
                 'experiments/test_research_component_ablation.py','aic_model/test/test_axial_contact_registration.py',
                 'aic_model/test/test_contact_registration.py','experiments/test_lip_search39.py',
                 'experiments/tests/test_verification.py','experiments/tests/test_raw_wrist_audit.py',
                 '-k','not actual_failed_search','-q'],OUT/'measurement_negative_controls.log',180)
            phase('contact_controls',selected_policy=choice['selected'])
            run([sys.executable,'experiments/run_seeded_evaluation.py','--nominal','sfp_positive_control',
                '--methods','initial_pose_guarded','--label','control44v1','--physics-backend','source_float64_damped',
                '--physics-step','0.0005','--wrist-rate','0','--solver-iterations','200','--force-ceiling-N','20'],
                OUT/'positive_control.log')
            control=json.loads((ROOT/'.artifacts/evaluation_sfp_positive_control.json').read_text())
            write(OUT/'positive_control_batch.json',control);r=control['results'][0]
            if r['status']!='completed':raise RuntimeError('Contact control environment failure')
            proof=OUT/'contact_positive_control.json'
            run([sys.executable,'experiments/qualify_sfp_contact.py','--scene-dir',r['evidence_dir'],
                 '--run',r['run'],'--out',str(proof)],OUT/'positive_control_qualification.log',120)
            if not json.loads(proof.read_text())['qualified']:raise RuntimeError('Contact control not qualified')
            run([sys.executable,'experiments/export_original_repair_evidence.py','--runs',r['run'],
                 '--out',str(OUT/'development_evidence/contact_control')],OUT/'contact_control_export.log',180)
            # Qualify each component-removal interface on known development cases.
            phase('ablation_development')
        from experiments.run_seeded_evaluation import SOURCES,trial,validate_cases
        sources=set(SOURCES)|{'experiments/full_study44.py','experiments/run_full_evaluation44.py',
            'experiments/configs/recovered_robustness43_protocol.json'}
        sources.update(str(p.relative_to(ROOT)).replace('\\','/') for p in WEIGHTS)
        sources.add(str(proof.relative_to(ROOT)).replace('\\','/'))
        hashes={p:sha(ROOT/p) for p in sorted(sources)}
        # trial() snapshots SOURCES; include all freeze dependencies and models.
        SOURCES[:]=sorted(sources)
        def execute(case,ablation,label,final):
            if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Safe storage boundary: less than 8 GiB free')
            for path,digest in hashes.items():
                if sha(ROOT/path)!=digest:raise RuntimeError('Frozen source changed: '+path)
            c=copy.deepcopy(case);base_id=c['case_id'];c['case_id']+='_'+label+'_'+ablation
            kind=c['connector'];method=SC if kind=='sc' else SFP
            batch=dict(source_sha256=hashes,purpose='held_out_final44' if final else 'development_component_qualification',
                remote_support='0',physics_step=.0005,cable_initialization='sc_straight_outboard' if kind=='sc' else 'source',
                grasp_profile='challenge_sc' if kind=='sc' else 'legacy_sfp',wrist_rate=0,force_ceiling_N=20.,
                sfp_motion_profile=profile if kind=='sfp' else 'legacy36',solver_iterations_requested=200,
                physics_backend='source_float64_damped',positive_control_proof_file=str(proof.resolve()),
                positive_control_proof_sha256=sha(proof),weight_overrides={},ablation=ablation,results=[])
            if kind=='sc':batch['weight_overrides']['AIC_SC_POSE_WEIGHTS']=str(WEIGHTS[0].relative_to(ROOT)).replace('\\','/')
            else:
                batch['weight_overrides']['AIC_NIC_WEIGHTS']=str(WEIGHTS[1].relative_to(ROOT)).replace('\\','/')
                batch['weight_overrides']['AIC_SFP_PLUG_POSE_WEIGHTS']=str(WEIGHTS[2].relative_to(ROOT)).replace('\\','/')
            state['current_run']=c['case_id']+'_'+method;save()
            trial(c,method,batch);result=batch['results'][0]
            result.update(base_case_id=base_id,ablation=ablation,final=final)
            dest=OUT/'trials'/result['run'];dest.mkdir(parents=True,exist_ok=False);write(dest/'batch.json',batch)
            if result['status']=='completed':
                run([sys.executable,'experiments/export_original_repair_evidence.py','--runs',result['run'],
                     '--out',str(dest/'evidence')],dest/'export.log',180)
                policy_path=ROOT/result['evidence_dir']/result['run']/'policy.json'
                observed=json.loads(policy_path.read_text()).get('research_ablation','full')
                if observed!=ablation:raise RuntimeError('Ablation did not reach the runtime policy')
            return result
        if not direct_final:
            for kind,seed in [('sc',42000),('sfp',45005)]:
                path=ROOT/'experiments/configs'/('sc_finish41_qualification.json' if kind=='sc' else 'sfp_contact34_development.json')
                case=next(c for c in json.loads(path.read_text())['cases'] if c['case_id']==f'dev_{seed}')
                for mode in ['no_axial_correction','no_lateral_tracking']:
                    result=execute(case,mode,'ablation44dev',False);state['development'].append(result);save()
                    if result['status']!='completed':raise RuntimeError('Ablation development environment failed')
        # Do not open these seeds until the selected policy and all variants are locked.
        phase('freezing_final_protocol')
        write(OUT/'source_freeze.json',dict(source_sha256=hashes,selection=choice,
            protocol_sha256=sha(ROOT/'experiments/configs/recovered_robustness43_protocol.json'),
            no_tuning_on_final_trials=True,checkpoint_sha256={str(p.relative_to(ROOT)):sha(p) for p in WEIGHTS}))
        groups={k:connector_cases(k,first) for k,first in [('sc',700000),('sfp',710000)]}
        cases=[groups[k][i] for i in range(50) for k in ['sc','sfp']];validate_cases(cases)
        # Balanced subset fixed before outcomes: four per SC mount, two per SFP slot.
        subset=[]
        for kind in ['sc','sfp']:
            counts={}
            for c in groups[kind]:
                key=(c['target_index'],c['sfp_port']);limit=4 if kind=='sc' else 2
                if counts.get(key,0)<limit:subset.append(c);counts[key]=counts.get(key,0)+1
        manifest=dict(purpose='held_out_final44',cases=cases,paired_subset=[c['case_id'] for c in subset],
            allocation='50 SC, 50 SFP; balanced mounts/slots',source_freeze_sha256=sha(OUT/'source_freeze.json'))
        write(OUT/'final_manifest.json',manifest);manifest_hash=sha(OUT/'final_manifest.json')
        state.update(final_trials_started=True,manifest_sha256=manifest_hash);phase('primary_final_trials')
        consecutive_environment_failures=0
        for case in cases:
            if sha(OUT/'final_manifest.json')!=manifest_hash:raise RuntimeError('Final manifest changed')
            result=execute(case,'full','final44v1',True);state['results'].append(result);save()
            consecutive_environment_failures=consecutive_environment_failures+1 if result['status']!='completed' else 0
            if consecutive_environment_failures>=3:raise RuntimeError('Three consecutive environment failures; remaining final cases unopened')
        phase('paired_final_ablations')
        for case in subset:
            for mode in ['no_axial_correction','no_lateral_tracking']:
                result=execute(case,mode,'final44v1',True);state['results'].append(result);save()
                consecutive_environment_failures=consecutive_environment_failures+1 if result['status']!='completed' else 0
                if consecutive_environment_failures>=3:raise RuntimeError('Three consecutive ablation environment failures')
        primary=[r for r in state['results'] if r['ablation']=='full']
        comparisons=[]
        for kind in ['sc','sfp']:
            for mode in ['no_axial_correction','no_lateral_tracking']:
                rows=[r for r in state['results'] if r['case']['connector']==kind]
                comparisons.append(dict(connector=kind,ablation=mode,**paired_exact_mcnemar(rows,mode)))
        running=0.
        for rank,test in enumerate(sorted(comparisons,key=lambda x:x['exact_two_sided_p'])):
            running=max(running,min(1.,(len(comparisons)-rank)*test['exact_two_sided_p']));test['holm_adjusted_p']=running
        write(OUT/'final_summary.json',dict(primary=summarize(primary),paired_comparisons=comparisons,
              all_trials=len(state['results']),sources_frozen=True,failures_retained=True))
        phase('complete_requires_scientific_review',current_run=None)
    except Exception as exc:
        phase('stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    power=ctypes.windll.kernel32
    power.SetThreadExecutionState(0x80000001)
    try:main()
    finally:power.SetThreadExecutionState(0x80000000)
