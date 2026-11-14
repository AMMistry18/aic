"""Record checkpoint identity and training metadata without running inference."""
import hashlib
import json
from pathlib import Path
from ultralytics import YOLO

root=Path('/research/aic_example_policies/aic_example_policies/ros/weights')
baked=Path('/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/weights')
rows=[]
for name in ('best.pt','best_sc_mouth_pose.pt','best_sc_plug_pose.pt','best_sfp_plug_pose.pt'):
    paths={label:directory/name for label,directory in (('source',root),('baked',baked))}
    row={'name':name,'sha256':{label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in paths.items()}}
    model=YOLO(str(paths['source']))
    checkpoint=model.ckpt
    row['training']={key:checkpoint.get(key) for key in ('date','epoch','version')}
    row['training']['data']=str(checkpoint.get('train_args',{}).get('data'))
    row['keypoint_shape']=list(getattr(model.model,'kpt_shape',[]))
    rows.append(row)
out=Path('/research/.artifacts/container/checkpoints.json')
out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
