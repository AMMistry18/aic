"""Finite short GPU development tests concurrent with unchanged final44.

Uses only the separate copied workspace, ROS domain 49 and perception container.
SC stops after alignment; SFP stops after perception. No seating is claimed.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/'.artifacts/dev46_workspace'
OUT=ROOT/'experiments/parallel_dev46'


def main():
    OUT.mkdir(exist_ok=True)
    state=dict(status='waiting_for_offline_GPU_capture',results=[],final_policy_modified=False,
               container='research-aic-perception30',cpu_limit=4,ros_domain_id=49,held_out_cases_used=0,
               renderer_threads=2,memory_limit_GiB=5)
    label='short46v1'
    previous=OUT/'progress.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old['status']!='stopped_requires_inspection':raise RuntimeError('Development queue already started; inspect first')
        (OUT/('progress_before_retry_'+str(int(time.time()))+'.json')).write_text(json.dumps(old,indent=2))
        state['results']=old['results'];label='short46v'+str(len(old['results'])+1)
        state['environment_repair']='Active fixture/arm/cable visuals only, all physics retained; LLVM threads 2, CPU quota 4; previous failed setups retained'
    def save():
        state['updated_unix']=time.time();(OUT/'progress.json').write_text(json.dumps(state,indent=2))
    save()
    freeze=json.loads((ROOT/'experiments/full_evaluation44/source_freeze.json').read_text())['source_sha256']
    def unchanged():
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in freeze.items()):
            raise RuntimeError('A frozen final source changed')
    try:
        main_running=subprocess.run(['docker','inspect','research-aic-eval','--format','{{.State.Running}}'],
                                    capture_output=True,text=True,check=True).stdout.strip()
        if main_running=='true':
            state.update(status='deferred_for_final_recording_priority',current=None,
                         reason='Two full renderers exceed practical laptop resource budget; run offline diagnostics only during final44')
            save();return
        deadline=time.monotonic()+600
        while True:
            path=OUT/'angular_inputs.json'
            if path.exists() and len(json.loads(path.read_text())['rows'])==8:break
            if time.monotonic()>deadline:raise TimeoutError('GPU development capture did not finish')
            time.sleep(2)
        unchanged()
        for kind,identity,manifest,method in [
            ('sc','dev_42000','sc_finish41_qualification.json','original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff'),
            ('sfp','dev_45005','sfp_contact34_development.json','original_perception_sfp_contact_datum_handoff')]:
            unchanged();state.update(status='short_development_trial',current=kind+'_'+identity);save()
            weights='experiments/recovery_20261112/checkpoints/'
            args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest','experiments/configs/'+manifest,
                  '--case-ids',identity,'--methods',method,'--label',label,'--physics-backend','source_float64_damped',
                  '--physics-step','.0005','--wrist-rate','0','--solver-iterations','200','--force-ceiling-N','20',
                  '--positive-control-proof','experiments/full_evaluation44/contact_positive_control.json']
            if kind=='sc':args+=['--grasp-profile','challenge_sc','--cable-initialization','sc_straight_outboard',
                                '--sc-mouth-weights',weights+'sc_mouth_recovery20261112/best.pt']
            else:args+=['--sfp-motion-profile','fast38','--nic-weights',weights+'nic_recovery20261112/best.pt',
                        '--sfp-plug-weights',weights+'plug_recovery20261112/best.pt']
            with (OUT/(kind+'_'+label+'.log')).open('wb') as log:
                subprocess.run(args,cwd=DEV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            batch=json.loads((DEV/'.artifacts/evaluation_seeded.json').read_text());r=batch['results'][0]
            result=dict(connector=kind,run=r['run'],execution_status=r['status'],
                        expected_seating_verdict=False,verified_insertion_claimed=False,
                        evidence_dir=str(DEV/Path(r['evidence_dir'])),geometry_and_force_verdict=r)
            if r['status']=='completed':
                folder=DEV/r['evidence_dir']/r['run'];policy=json.loads((folder/'policy.json').read_text())
                result.update(short_result=policy.get('development_short_result'),
                              wall_s=policy['wall_s'],sim_s=policy['sim_s'],configuration_wall_s=policy['configuration_wall_s'])
            state['results'].append(result);save()
            if r['status']!='completed':raise RuntimeError('Development environment failed; inspect retained evidence')
        unchanged();state.update(status='complete_short_tests_require_analysis',current=None);save()
    except Exception as exc:
        state.update(status='stopped_requires_inspection',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':
    power=ctypes.windll.kernel32;power.SetThreadExecutionState(0x80000001)
    try:main()
    finally:power.SetThreadExecutionState(0x80000000)
