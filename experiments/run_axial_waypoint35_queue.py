"""Finite development follow-up after the V27 source freeze completes.

Installs only the reviewed axial free-waypoint candidate, reruns its tests,
then physically evaluates two preselected known failures. Never opens finals.
"""
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/repair35/axial35v1'
PRIOR=ROOT/'.artifacts/repair35/regression35v1/progress.json'
CANDIDATE=ROOT/'.artifacts/repair35/axial_candidate'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    state=dict(status='waiting_for_v27_completion',results=[],planned=['dev_45002','dev_45003'],
        held_out_final_trials_run=0,paper_ready=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    reviewed={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in
        [CANDIDATE/'candidate.json',CANDIDATE/'sfp_contact_datum_v28.py',
         ROOT/'experiments/test_waypoint_axial35.py',Path(__file__).resolve()]}
    state['reviewed_candidate_sha256']=reviewed

    def save(active=False):
        (OUT/'progress.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        p=ROOT/'.artifacts/active_original_perception29.json'
        a=json.loads(p.read_text(encoding='utf-8-sig'))
        a['axial35_followup_progress']='.artifacts/repair35/axial35v1/progress.json'
        if active:
            a.update(status='axial35_'+state['status'],current_run=state.get('current_run'),
                live_insertion_world=state['status']=='executing_development_case',
                held_out_final_trials_run=0,paper_ready=False,axial35_completed=len(state['results']))
        p.write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8')

    def run(args,log,timeout=1800):
        with (OUT/log).open('wb') as f:
            subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)

    def unchanged(hashes):
        for source,h in hashes.items():
            if digest(ROOT/source)!=h:raise RuntimeError('Source/checkpoint changed: '+source)

    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    installed=False
    try:
        save();deadline=time.monotonic()+3600
        while True:
            prior=json.loads(PRIOR.read_text())
            if prior['status']=='development_complete_needs_analysis':break
            if prior['status']=='stopped_on_execution_error':raise RuntimeError('Prior development queue execution error')
            if time.monotonic()>deadline:raise RuntimeError('Prior queue did not complete; policy untouched')
            time.sleep(10)
        if len(prior['results'])!=4:raise RuntimeError('Incomplete prior queue; policy untouched')
        unchanged(prior['source_and_checkpoint_sha256']);unchanged(reviewed)
        running=subprocess.check_output(['docker','inspect','--format','{{.State.Running}}','research-aic-eval'],text=True).strip()
        if running!='false':raise RuntimeError('Evaluator container still running; policy untouched')
        metadata=json.loads((CANDIDATE/'candidate.json').read_text())
        target=ROOT/metadata['source'];replacement=CANDIDATE/'sfp_contact_datum_v28.py'
        if digest(target)!=metadata['original_sha256'] or digest(replacement)!=metadata['candidate_sha256']:
            raise RuntimeError('Candidate identity mismatch')
        harness=ROOT/'experiments/run_original_insertion.py';old=harness.read_bytes()
        if old.count(b"result['harness_revision']=27")!=1:raise RuntimeError('Harness revision mismatch')
        (OUT/'harness_v27.py').write_bytes(old)
        (OUT/'sfp_contact_datum_v27.py').write_bytes(target.read_bytes())
        state.update(status='installing_reviewed_development_candidate',prior_results=prior['results']);save(active=True)
        temporary=target.with_suffix('.candidate.tmp');temporary.write_bytes(replacement.read_bytes());temporary.replace(target)
        harness.write_bytes(old.replace(b"result['harness_revision']=27",b"result['harness_revision']=28"))
        installed=True
        state['source_and_checkpoint_sha256']={p:digest(ROOT/p) for p in prior['source_and_checkpoint_sha256']}
        state['source_and_checkpoint_sha256'].update(reviewed)
        state.update(status='unit_testing_installed_candidate');save(active=True)
        run(['docker','start','research-aic-eval'],'unit_container_start.log',60)
        try:
            shell=('source /ws_aic/install/setup.bash && '
                'export PYTHONPATH=/research/aic_model:/research/aic_example_policies:/research/flowstate/aic_perception:/research && '
                '/opt/research-perception/bin/python -m pytest -q experiments/test_waypoint_axial35.py '
                'aic_model/test/test_sfp_contact_datum.py aic_model/test/test_observable_terminal.py '
                'aic_model/test/test_axial_contact_registration.py aic_model/test/test_tracking_compensation.py '
                'aic_model/test/test_contact_registration.py')
            run(['docker','exec','-w','/research','research-aic-eval','bash','-lc',shell],'unit_tests.log',180)
        finally:run(['docker','stop','research-aic-eval'],'unit_container_stop.log',60)
        for case in state['planned']:
            unchanged(state['source_and_checkpoint_sha256'])
            method='original_perception_sfp_contact_datum_handoff'
            state.update(status='executing_development_case',current_run=case+'_axial35v1_'+method);save(active=True)
            run([sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest',
                'experiments/configs/sfp_contact34_development.json','--case-ids',case,
                '--methods',method,'--label','axial35v1','--physics-backend','source_float64_damped',
                '--physics-step','0.0005','--wrist-rate','0','--solver-iterations','200',
                '--positive-control-proof','.artifacts/sfp_damped27/contact_proof.json',
                '--nic-weights','.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
                '--sfp-plug-weights','.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt'],case+'.log')
            batch=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text())
            (OUT/(case+'_batch.json')).write_text(json.dumps(batch,indent=2)+'\n',encoding='utf-8')
            if len(batch['results'])!=1 or batch['results'][0]['run']!=state['current_run']:
                raise RuntimeError('Result identity mismatch')
            state['results'].append(batch['results'][0]);save(active=True)
            if batch['results'][0]['status']!='completed':raise RuntimeError('Infrastructure failed; remaining cases unopened')
        state.update(status='development_complete_needs_analysis',current_run=None);save(active=True)
    except Exception as exc:
        state.update(status='stopped_on_execution_error',current_run=None,error=f'{type(exc).__name__}: {exc}',
                     candidate_installed=installed);save(active=installed);raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
