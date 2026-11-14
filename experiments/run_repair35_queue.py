"""Finite frozen development regressions; no held-out cases or tuning."""
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_seeded_evaluation import SOURCES
OUT=ROOT/'.artifacts/repair35/regression35v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    plan=[('sc','dev_42000'),('sfp','dev_45002'),('sfp','dev_45001'),('sfp','dev_45003')]
    state=dict(status='starting',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               held_out_final_trials_run=0,paper_ready=False,results=[],planned=plan,current_run=None)
    sources=set(SOURCES)|{'experiments/run_repair35_queue.py',
        'experiments/configs/terminal_braking32_qualification.json','experiments/configs/sfp_contact34_development.json',
        '.artifacts/perception_data30/training/sc_mouth_standard30_v1/run/weights/best.pt',
        '.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
        '.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt'}
    state['source_and_checkpoint_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)}
    def save():
        (OUT/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
        p=ROOT/'.artifacts/active_original_perception29.json';a=json.loads(p.read_text(encoding='utf-8-sig'))
        a.update(status='repair35_'+state['status'],current_run=state['current_run'],
            live_insertion_world=state['status']=='executing_development_case',paper_ready=False,
            held_out_final_trials_run=0,repair35_progress='.artifacts/repair35/regression35v1/progress.json',
            repair35_completed=len(state['results']))
        p.write_text(json.dumps(a,indent=2)+'\n')
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for connector,case in plan:
            for source,h in state['source_and_checkpoint_sha256'].items():
                if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+source)
            method=('original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff' if connector=='sc'
                    else 'original_perception_sfp_contact_datum_handoff')
            state.update(status='executing_development_case',current_run=case+'_regression35v1_'+method);save()
            args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest',
                  'experiments/configs/'+('terminal_braking32_qualification.json' if connector=='sc' else 'sfp_contact34_development.json'),
                  '--case-ids',case,'--methods',method,'--label','regression35v1',
                  '--physics-backend','source_float64_damped','--physics-step','0.0005','--wrist-rate','0',
                  '--solver-iterations','200','--positive-control-proof','.artifacts/sfp_damped27/contact_proof.json']
            if connector=='sc':args+=['--grasp-profile','challenge_sc','--cable-initialization','sc_straight_outboard',
                '--sc-mouth-weights','.artifacts/perception_data30/training/sc_mouth_standard30_v1/run/weights/best.pt']
            else:args+=['--nic-weights','.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
                '--sfp-plug-weights','.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt']
            with (OUT/(case+'.log')).open('wb') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            batch=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text())
            (OUT/(case+'_batch.json')).write_text(json.dumps(batch,indent=2)+'\n')
            if len(batch['results'])!=1 or batch['results'][0]['run']!=state['current_run']:raise RuntimeError('Result identity mismatch')
            state['results'].append(batch['results'][0]);save()
            if batch['results'][0]['status']!='completed':raise RuntimeError('Infrastructure failed; remaining cases unopened')
        state.update(status='development_complete_needs_analysis',current_run=None);save()
    except Exception as exc:
        state.update(status='stopped_on_execution_error',error=f'{type(exc).__name__}: {exc}',current_run=None);save();raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
