"""Owned serial queue: finish RGB collection, adapt models, then SC regression.

No final evaluation cases are generated or read. Stops on infrastructure errors;
every completed policy failure remains in its original archived denominator.
"""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.artifacts/repair33';OUT.mkdir(exist_ok=True)


def phase(name,**extra):
    (OUT/'progress.json').write_text(json.dumps(dict(status=name,**extra),indent=2)+'\n',encoding='utf-8')
    active=ROOT/'.artifacts/active_original_perception29.json';data=json.loads(active.read_text(encoding='utf-8-sig'))
    data.update(status='repair33_'+name,current_run=name,repair33_progress='.artifacts/repair33/progress.json',
                held_out_final_trials_run=0,paper_scope=['sc','sfp'])
    active.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def command(args,name,timeout):
    with (OUT/(name+'.log')).open('wb') as log:
        subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def main():
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        phase('waiting_for_sfp_collection')
        deadline=time.monotonic()+2700
        while True:
            p=ROOT/'.artifacts/sfp_visual33/data/progress.json'
            if p.exists() and json.loads(p.read_text())['status']=='collection_complete':break
            if time.monotonic()>deadline:raise TimeoutError('SFP collection did not complete')
            time.sleep(15)
        command(['docker','start','research-aic-perception30'],'gpu_start',60)
        try:
            for kind,name in [('sfp_nic','nic33_v1'),('sfp_plug','plug33_v1')]:
                phase('training_'+kind)
                command(['docker','exec','-w','/research','research-aic-perception30','/opt/research-perception/bin/python',
                         'experiments/train_sfp_transfer33.py','--kind',kind,'--name',name],name,3600)
        finally:command(['docker','stop','research-aic-perception30'],'gpu_stop',60)
        for name,nic,plug in [
            ('original33v1','aic_example_policies/aic_example_policies/ros/weights/best.pt',
             'aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt'),
            ('adapted33v1','.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
             '.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt')]:
            phase('sfp_static_development_evaluation_'+name)
            command(['docker','exec','-w','/research','research-aic-eval','bash','experiments/evaluate_sfp_visual33.sh',
                     '--nic-weights','/research/'+nic,'--plug-weights','/research/'+plug,'--name',name],name,600)
        phase('sc_three_failure_regressions_running')
        command([sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest','experiments/configs/terminal_braking32_qualification.json',
                 '--case-ids','dev_43001','dev_43003','dev_43004','--methods',
                 'original_perception_rim_position_body_track_brake_axial_noangular_handoff','--label','axial33v1',
                 '--grasp-profile','challenge_sc','--sc-mouth-weights','.artifacts/perception_data30/training/sc_mouth_standard30_v1/run/weights/best.pt',
                 '--physics-backend','source_float64_damped','--physics-step','0.0005','--wrist-rate','0',
                 '--solver-iterations','200','--cable-initialization','sc_straight_outboard'], 'sc_axial33v1',2400)
        source=ROOT/'.artifacts/seeded_evaluation_progress.json'
        batch=json.loads(source.read_text());(OUT/'sc_axial33v1_completed.json').write_text(json.dumps(batch,indent=2)+'\n',encoding='utf-8')
        phase('development_complete_needs_inspection',sc_verified=sum(r.get('verified_success',False) for r in batch['results']),
              sc_attempts=len(batch['results']),sfp_insertion_trials_with_new_weights=0)
    except Exception as e:
        phase('queue_failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
