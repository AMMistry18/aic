"""Recover the interrupted GPU run from its preserved, validated best file."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'.artifacts/perception_data30/training/sc_mouth_standard30_v1'
    checkpoint=out/'interrupted_checkpoint_snapshot/best.pt'
    record=out/'recovery.json'
    if record.exists():raise RuntimeError('Preserve prior recovery; do not launch twice')
    # Load only this locally trained, checksum-audited checkpoint.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    shutil.copy2(out/'run/results.csv',out/'interrupted_checkpoint_snapshot/results.csv')
    recovery=dict(status='running',source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_epoch_zero_based=saved['epoch'],
        interrupted_last_checkpoint_preserved=True,resume_semantics='restore EMA/model/optimizer/schedule; RNG replay is not claimed identical',
        gpu=torch.cuda.get_device_name(0),final_100000_series_used=False)
    record.write_text(json.dumps(recovery,indent=2)+'\n');del saved
    torch.set_num_threads(2);started=time.monotonic();model=YOLO(str(checkpoint));result=model.train(resume=True,device=0,workers=0)
    recovery.update(status='complete',recovery_wall_s=time.monotonic()-started,metrics={k:float(v) for k,v in result.results_dict.items()})
    for name in ['best.pt','last.pt']:
        recovery[name+'_sha256']=hashlib.sha256((out/'run/weights'/name).read_bytes()).hexdigest()
    record.write_text(json.dumps(recovery,indent=2)+'\n')
    plan=json.loads((out/'training_plan.json').read_text());plan.update(status='complete_after_checkpoint_recovery',recovery=recovery)
    (out/'training_result.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(recovery))


if __name__=='__main__':main()
