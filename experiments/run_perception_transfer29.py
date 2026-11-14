"""Finish frozen development collection, then train/evaluate original models.

Runs only in the existing CPU environment. Explicitly excludes final physics
cases. Does not start a second simulator or overwrite the original weights.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'.artifacts/perception_data29/qualified'
STATE=ROOT/'.artifacts/perception_data29/transfer_progress.json'


def save(status,**extra):
    STATE.parent.mkdir(parents=True,exist_ok=True)
    STATE.write_text(json.dumps({'status':status,'final_heldout_trials_used':0,**extra},indent=2)+'\n')
    print(status,flush=True)


def main():
    save('waiting_for_scene_collection')
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        progress=json.loads((DATA/'progress.json').read_text())
        if progress['status']=='collection_complete':break
        if time.time()-(DATA/'progress.json').stat().st_mtime>180:raise RuntimeError('Collection progress stale; inspect preserved case logs')
        time.sleep(5)
    else:raise TimeoutError('Collection did not complete')
    files=[f for f in (DATA/'dataset').rglob('*') if f.is_file()]
    checksums={str(f.relative_to(DATA)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (DATA/'dataset_SHA256SUMS.json').write_text(json.dumps(checksums,indent=2)+'\n')
    completed=[]
    for kind in ['sc_mouth','sc_plug']:
        save('training',kind=kind,completed=completed,dataset_files=len(files))
        name=f'{kind}_transfer29_v1'
        log=ROOT/f'.artifacts/container/{name}_train.log'
        with log.open('wb') as stream:
            subprocess.run(['docker','exec','research-aic-eval','bash','/research/experiments/train_perception_transfer.sh',
                            '--kind',kind,'--name',name],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
        original=('best_sc_mouth_pose.pt' if kind=='sc_mouth' else 'best_sc_plug_pose.pt')
        for method,weights in [('original',f'/research/aic_example_policies/aic_example_policies/ros/weights/{original}'),
            ('transfer',f'/research/.artifacts/perception_data29/training/{name}/run/weights/best.pt')]:
            result=f'/research/.artifacts/perception_data29/evaluation/{kind}_{method}_val.json'
            with (ROOT/f'.artifacts/container/{kind}_{method}_val.log').open('wb') as stream:
                subprocess.run(['docker','exec','research-aic-eval','/opt/research-perception/bin/python',
                    '/research/experiments/evaluate_transferred_keypoints.py','--dataset',f'/research/.artifacts/perception_data29/qualified/dataset/{kind}',
                    '--weights',weights,'--split','val','--out',result],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        completed.append(kind)
    save('development_training_and_image_evaluation_complete',completed=completed,
         warning='2D validation is not a physical insertion result; run multiview and independently verified physics trials next')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('failed',error=f'{type(exc).__name__}: {exc}');raise
