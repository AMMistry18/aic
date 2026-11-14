"""Reanalyse immutable trial records with an explicitly audited depth convention."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--indexes',nargs='+',type=Path,required=True)
    p.add_argument('--runtime-audit',type=Path,required=True);p.add_argument('--positive-control-proof',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    results=[];batches=[]
    for index_path in a.indexes:
        batch=json.loads(index_path.read_text());batches.append({'index':str(index_path),'sha256':hashlib.sha256(index_path.read_bytes()).hexdigest()})
        for old in batch['results']:
            if old['status']!='completed':results.append(old);continue
            scene=ROOT/old['evidence_dir'];run=old['run'];out=a.out/run;out.mkdir(exist_ok=True)
            command=[sys.executable,str(ROOT/'experiments/verify_recorded_trial.py'),'--scene-dir',str(scene),'--run',run,
                '--runtime-audit',str(a.runtime_audit),'--positive-control-proof',str(a.positive_control_proof),'--output-dir',str(out)]
            with (out/'verification.log').open('wb') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            verified=json.loads((out/'verified.json').read_text())
            merged={**old,**verified,'legacy_verdict':{'verified_success':old.get('verified_success'),'reason':old.get('reason')},
                'verification_file':str((out/'verified.json').resolve().relative_to(ROOT)),
                'verification_trace_file':str((out/'verification_trace.json.gz').resolve().relative_to(ROOT))}
            results.append(merged);print(json.dumps({'run':run,'success':verified['verified_success'],'reason':verified['reason'],
                'terminal_penetration_mm':verified['terminal_metrics'].get('max_penetration_m',{}).get('max',0)*1000}),flush=True)
    index={'purpose':'immutable_archive_reverification_no_new_policy_trials','verification_semantics_revision':2,
        'runtime_contact_contract':json.loads(a.runtime_audit.read_text()),'source_indexes':batches,
        'positive_control_proof':str(a.positive_control_proof),'results':results}
    (a.out/'index.json').write_text(json.dumps(index,indent=2)+'\n')

if __name__=='__main__':main()
