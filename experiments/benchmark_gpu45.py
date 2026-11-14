"""CPU/GPU microbenchmark on saved development RGB with the same learned weights.

FP32 inference only. Reports outputs as well as time; this does not establish
end-to-end insertion equivalence or speed. Never reads the held-out ledger.
"""
import hashlib
import json
from pathlib import Path
import statistics
import time
import cv2
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('No CUDA device; do not invent a GPU benchmark')
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    rows=[]
    for kind,checkpoint in [('sc','sc_mouth_recovery20261112'),('sfp','nic_recovery20261112')]:
        run=next(r for r in ledger['results'] if r['case']['connector']==kind and r.get('verified_success'))
        folder=ROOT/run['evidence_dir'].replace('\\','/')/run['run']/'perception_before'
        images=[cv2.imread(str(folder/(camera+'_raw.png'))) for camera in ['left_camera','center_camera','right_camera']]
        if any(i is None for i in images):raise RuntimeError('Missing saved development triplet')
        weight=ROOT/'experiments/recovery_20261112/checkpoints'/checkpoint/'best.pt'
        measurements={}
        for device in ['cpu','0']:
            model=YOLO(str(weight))
            kwargs=dict(source=images,device=device,imgsz=960,verbose=False,half=False,conf=.25,save=False)
            for _ in range(2):model.predict(**kwargs)
            times=[];speeds=[];counts=[]
            for _ in range(5):
                if device=='0':torch.cuda.synchronize()
                start=time.perf_counter();results=model.predict(**kwargs)
                if device=='0':torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                speeds.append([r.speed for r in results]);counts.append([len(r.boxes) for r in results])
            measurements[device]=dict(wall_s=times,median_triplet_wall_s=statistics.median(times),
                detections_per_view=counts,instrumented_per_image_ms=speeds)
            del model
            if device=='0':torch.cuda.empty_cache()
        rows.append(dict(connector=kind,development_run=run['run'],weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
            input_sha256=[hashlib.sha256((folder/(c+'_raw.png')).read_bytes()).hexdigest() for c in ['left_camera','center_camera','right_camera']],
            measurements=measurements,speedup_cpu_over_gpu=measurements['cpu']['median_triplet_wall_s']/measurements['0']['median_triplet_wall_s']))
        result=dict(scope='FP32 full-frame YOLO microbenchmark; original weighted multi-view/crop pipeline and physical insertion not benchmarked',
            final_trials_read=0,cpu_threads=1,gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,results=rows,
            does_not_certify_pose_parity=True,installed=False)
        (ROOT/'experiments/efficiency45/gpu_benchmark.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(connector=kind,speedup=rows[-1]['speedup_cpu_over_gpu'],cpu_s=measurements['cpu']['median_triplet_wall_s'],
                              gpu_s=measurements['0']['median_triplet_wall_s'])),flush=True)


if __name__=='__main__':main()
