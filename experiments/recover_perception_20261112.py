"""Finite recovery queue. New weights are NOT the missing historical checkpoints.

Run on Windows after collect_sfp_development33.py has started. Uses only the
existing training/development splits; never launches held-out trials.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/recovery_20261112'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(status, **extra):
    OUT.mkdir(exist_ok=True)
    (OUT / 'progress.json').write_text(json.dumps(dict(status=status,
        updated_unix=time.time(), fresh_verified_insertions=0,
        historical_checkpoint_identity_claimed=False, **extra), indent=2))


def run(command, name, timeout):
    state(name)
    with (OUT / (name + '.log')).open('wb') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def preserve(source, name):
    dest = OUT / 'checkpoints' / name
    dest.mkdir(parents=True, exist_ok=False)
    for filename in ['training_plan.json', 'training_result.json']:
        shutil.copy2(source / filename, dest / filename)
    shutil.copy2(source / 'run/weights/best.pt', dest / 'best.pt')
    (dest / 'SHA256SUMS.json').write_text(json.dumps(
        {p.name: digest(p) for p in dest.iterdir() if p.is_file()}, indent=2))


def train_sc():
    # New recovery training recipe, explicitly versioned; no fabricated old plan.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import cv2
    from ultralytics import YOLO
    data = ROOT / '.artifacts/perception_data29/qualified'
    if json.loads((data / 'progress.json').read_text())['status'] != 'collection_complete':
        raise RuntimeError('Incomplete SC data')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    out = ROOT / '.artifacts/perception_data30/training/sc_mouth_recovery20261112'
    out.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'aic_example_policies/aic_example_policies/ros/weights/best_sc_mouth_pose.pt'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    args = dict(data=str(data/'dataset/sc_mouth/dataset.yaml'), epochs=60,
        imgsz=960, batch=4, workers=0, device=0, amp=False, seed=20261030,
        deterministic=True, freeze=10, cache=False, optimizer='AdamW', lr0=.0002,
        lrf=.1, cos_lr=True, patience=60, mosaic=0., mixup=0., degrees=2.,
        translate=.03, scale=.1, fliplr=0., flipud=0., hsv_h=.015, hsv_s=.3,
        hsv_v=.3, plots=False, project=str(out), name='run', exist_ok=False,
        save=True, close_mosaic=0)
    plan = dict(status='running', revision='recovery20261112',
        original_weight_sha256=digest(original), training_parameters=args,
        manifest_sha256=digest(data/'manifest.json'),
        dataset_sha256={str(p.relative_to(data)): digest(p)
            for p in (data/'dataset').rglob('*') if p.is_file()},
        trainer_sha256=digest(Path(__file__)), held_out_final_cases_used=0,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0))
    (out/'training_plan.json').write_text(json.dumps(plan, indent=2))
    start = time.monotonic()
    result = YOLO(str(original)).train(**args)
    plan.update(status='complete', wall_s=time.monotonic()-start,
        checkpoint_sha256=digest(out/'run/weights/best.pt'),
        metrics={k:float(v) for k,v in result.results_dict.items()})
    (out/'training_result.json').write_text(json.dumps(plan, indent=2))


def main():
    try:
        state('waiting_for_sfp_collection')
        deadline = time.monotonic()+5400
        while True:
            path = ROOT/'.artifacts/sfp_visual33/data/progress.json'
            if path.exists() and json.loads(path.read_text())['status']=='collection_complete':
                break
            if time.monotonic()>deadline:
                raise TimeoutError('SFP collection incomplete; inspect collector logs')
            time.sleep(10)
        for kind, name in [('sfp_nic','nic_recovery20261112'),('sfp_plug','plug_recovery20261112')]:
            run(['docker','exec','-w','/research','research-aic-perception30',
                 '/opt/research-perception/bin/python','experiments/train_sfp_transfer33.py',
                 '--kind',kind,'--name',name], 'train_'+kind, 7200)
            preserve(ROOT/'.artifacts/sfp_visual33/training'/name, name)
        run([sys.executable,'experiments/collect_perception_development.py'], 'collect_sc', 5400)
        run(['docker','exec','-w','/research','research-aic-perception30',
             '/opt/research-perception/bin/python','experiments/recover_perception_20261112.py',
             '--train-sc'], 'train_sc_mouth', 7200)
        preserve(ROOT/'.artifacts/perception_data30/training/sc_mouth_recovery20261112',
                 'sc_mouth_recovery20261112')
        state('training_complete_requires_static_and_physical_validation')
    except Exception as exc:
        state('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    if '--train-sc' in sys.argv:
        train_sc()
    else:
        main()
