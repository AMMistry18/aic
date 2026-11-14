"""Real CPU inference check for the runtime-only thread guard on old RGB."""
import json
from pathlib import Path
import sys
import time
import cv2
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.candidates.efficiency46.perception_runtime import bind_inference_threads


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1)
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    run=next(r for r in ledger['results'] if r['case']['connector']=='sc' and r.get('verified_success'))
    image=cv2.imread(str(ROOT/run['evidence_dir'].replace('\\','/')/run['run']/'perception_before/center_camera_raw.png'))
    if image is None:raise RuntimeError('Missing development image')
    model=bind_inference_threads(YOLO(str(ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_plug_pose.pt')))
    calls=[]
    def record(predictor):calls.append(dict(threads=torch.get_num_threads(),device=str(predictor.device)))
    model.add_callback('on_predict_start',record)
    began=time.perf_counter();prediction=model(image,device='cpu',imgsz=960,half=False,verbose=False)
    result=dict(scope='Real CPU inference runtime guard check only, not insertion qualification',
                final_cases_read=0,robot_commands_sent=0,calls=calls,wall_s=time.perf_counter()-began,
                threads_after_inference=torch.get_num_threads(),detections=len(prediction[0].boxes))
    result['passed']=len(calls)==1 and calls[0]['threads']==1 and result['threads_after_inference']==1
    (ROOT/'experiments/parallel_dev46/thread_callback_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Backend initialization overrode explicit runtime budget')


if __name__=='__main__':main()
