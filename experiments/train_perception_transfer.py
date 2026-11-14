"""Fine-tune an original frozen checkpoint on separate development scenes.

Never overwrites deployed weights. The validation split is for development;
held-out final physics trials are not loaded by this program.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'.artifacts/perception_data29/qualified'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['sc_mouth','sc_plug'],required=True)
    p.add_argument('--name',required=True)
    a=p.parse_args()
    progress=json.loads((DATA/'progress.json').read_text())
    if progress['status']!='collection_complete':raise RuntimeError('Wait for complete, frozen scene-separated collection')
    weights=ROOT/'aic_example_policies/aic_example_policies/ros/weights'/('best_sc_mouth_pose.pt' if a.kind=='sc_mouth' else 'best_sc_plug_pose.pt')
    output=ROOT/'.artifacts/perception_data29/training'/a.name
    if output.exists():raise RuntimeError('Preserve previous training run; choose a new name')
    output.mkdir(parents=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    args={'data':str(DATA/'dataset'/a.kind/'dataset.yaml'),'epochs':12,'imgsz':960,'batch':2,'workers':0,
          'device':'cpu','amp':False,'seed':20261029,'deterministic':True,'freeze':10,'cache':False,
          'optimizer':'AdamW','lr0':.0002,'lrf':.1,'cos_lr':True,'patience':12,'mosaic':0.,'mixup':0.,
          'degrees':2.,'translate':.03,'scale':.1,'fliplr':0.,'flipud':0.,'hsv_h':.015,'hsv_s':.3,'hsv_v':.3,
          'plots':False,'project':str(output),'name':'run','exist_ok':False,'save':True,'close_mosaic':0}
    plan={'purpose':'development transfer of original learned perception; final 100000-series untouched',
          'kind':a.kind,'source_weights':str(weights),'source_weights_sha256':sha(weights),
          'dataset_manifest_sha256':sha(DATA/'manifest.json'),'torch_version':torch.__version__,
          'training_parameters':args,'status':'running'}
    (output/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    started=time.monotonic();model=YOLO(str(weights))
    result=model.train(**args)
    plan.update(status='complete',wall_s=time.monotonic()-started,
                metrics={k:float(v) for k,v in result.results_dict.items()})
    for name in ['best.pt','last.pt']:
        path=output/'run/weights'/name
        if path.exists():plan[name+'_sha256']=sha(path)
    (output/'training_result.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:v for k,v in plan.items() if k not in ['training_parameters','metrics']},indent=2))


if __name__=='__main__':main()
