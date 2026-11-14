"""Finite serial development evaluation for both connectors, never final seeds.

Requires a physically qualified SFP nominal trial. Preserves every policy
failure, freezes source/checkpoint files, and stops on infrastructure errors.
"""
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_seeded_evaluation import SOURCES
OUT=ROOT/'.artifacts/repair34/validation34v1'
NOMINAL='dev_44000_datum34v5_original_perception_sfp_contact_datum_handoff'
SC_METHOD='original_perception_rim_position_body_track_brake_axial_partial_noangular_handoff'
SFP_METHOD='original_perception_sfp_contact_datum_handoff'


def write(state):
    (OUT/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    p=ROOT/'.artifacts/active_original_perception29.json';a=json.loads(p.read_text(encoding='utf-8-sig'))
    a.update(status='validation34v1_'+state['status'],current_run=state.get('current_run'),
             live_insertion_world=state['status']=='executing_development_case',
             held_out_final_trials_run=0,paper_ready=False,paper_scope=['sc','sfp'],
             validation_progress='.artifacts/repair34/validation34v1/progress.json',
             validation_completed=len(state['results']))
    p.write_text(json.dumps(a,indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    state=dict(status='awaiting_nominal_qualification',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               paper_ready=False,held_out_final_trials_run=0,results=[],current_run=NOMINAL)
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        write(state);root=ROOT/'.artifacts/lastinch_runs'/NOMINAL
        deadline=time.monotonic()+1800
        while not (root/NOMINAL/'verified.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Nominal archive not ready')
            time.sleep(5)
        # Wait for the owning runner to stop its world and finish archiving.
        while True:
            if time.monotonic()>deadline:raise TimeoutError('Nominal runner did not finish')
            try:
                completed=json.loads((ROOT/'.artifacts/evaluation_seeded.json').read_text())
                if any(r.get('run')==NOMINAL for r in completed.get('results',[])):break
            except (FileNotFoundError,json.JSONDecodeError):pass
            time.sleep(2)
        nominal=json.loads((root/NOMINAL/'verified.json').read_text())
        (OUT/'nominal_qualification.json').write_text(json.dumps(nominal,indent=2)+'\n')
        if not nominal.get('verified_success'):
            state.update(status='nominal_qualification_failed',current_run=None);write(state);return
        manifests={'sc':'experiments/configs/terminal_braking32_qualification.json',
                   'sfp':'experiments/configs/sfp_contact34_development.json'}
        sources=set(SOURCES)|set(manifests.values())|{'experiments/run_contact34_validation.py',
            '.artifacts/perception_data30/training/sc_mouth_standard30_v1/run/weights/best.pt',
            '.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
            '.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt'}
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)}
        state['source_and_checkpoint_sha256']=hashes
        cases={k:json.loads((ROOT/v).read_text())['cases'] for k,v in manifests.items()}
        known=['dev_43001','dev_43003','dev_43004']
        sc=[next(c for c in cases['sc'] if c['case_id']==n) for n in known]
        remaining=[c for c in cases['sc'] if c['case_id'] not in known]
        rng=random.Random(20261102);rng.shuffle(remaining);rng.shuffle(cases['sfp'])
        plan=[('sc',c) for c in sc+remaining]+[('sfp',c) for c in cases['sfp']]
        state['planned_cases']=[dict(connector=k,case=c) for k,c in plan]
        for connector,case in plan:
            for p,h in hashes.items():
                if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+p)
            method=SC_METHOD if connector=='sc' else SFP_METHOD
            run=case['case_id']+'_validation34v1_'+method
            state.update(status='executing_development_case',current_run=run);write(state)
            args=[sys.executable,'experiments/run_seeded_evaluation.py','--case-manifest',manifests[connector],
                  '--case-ids',case['case_id'],'--methods',method,'--label','validation34v1',
                  '--physics-backend','source_float64_damped','--physics-step','0.0005','--wrist-rate','0',
                  '--solver-iterations','200','--positive-control-proof','.artifacts/sfp_damped27/contact_proof.json']
            if connector=='sc':
                args+=['--grasp-profile','challenge_sc','--cable-initialization','sc_straight_outboard',
                       '--sc-mouth-weights','.artifacts/perception_data30/training/sc_mouth_standard30_v1/run/weights/best.pt']
            else:
                args+=['--nic-weights','.artifacts/sfp_visual33/training/nic33_v1/run/weights/best.pt',
                       '--sfp-plug-weights','.artifacts/sfp_visual33/training/plug33_v1/run/weights/best.pt']
            with (OUT/(case['case_id']+'.log')).open('wb') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            batch=json.loads((ROOT/'.artifacts/seeded_evaluation_progress.json').read_text())
            (OUT/(case['case_id']+'_batch.json')).write_text(json.dumps(batch,indent=2)+'\n')
            if len(batch['results'])!=1 or batch['results'][0]['run']!=run:raise RuntimeError('Runner result identity mismatch')
            result=batch['results'][0];state['results'].append(result);write(state)
            if result['status']!='completed':raise RuntimeError('Infrastructure failed; remaining cases unopened')
        state.update(status='development_complete_needs_analysis',current_run=None);write(state)
    except Exception as exc:
        state.update(status='stopped_on_execution_error',error=f'{type(exc).__name__}: {exc}',current_run=None);write(state)
        raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
