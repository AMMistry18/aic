"""Paired CPU replay of exact image-local reuse on historical DEVELOPMENT RGB.

No simulator truth or final-study samples are consumed. Numerical parity is
checked for pose and every diagnostic field, including abstention decisions.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=6);args=parser.parse_args()
    cv2.setNumThreads(1)
    out=ROOT/'experiments/candidates/efficiency45'
    baseline=module('baseline_rim45',ROOT/'aic_model/aic_model/insertion/sc_rim_refinement.py')
    candidate=module('cached_rim45',out/'sc_rim_refinement.py')
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    records=[]
    for row in ledger['results']:
        if row.get('status')!='completed' or row['case']['connector']!='sc':continue
        run=ROOT/row['evidence_dir'].replace('\\','/')/row['run']
        for stamp in ['000','003','006']:
            path=run/'rim_inputs'/stamp/'inputs.json'
            if path.exists():records.append(path)
        if len(records)>=args.limit:break
    rows=[]
    for index,path in enumerate(records[:args.limit]):
        data=json.loads(path.read_text());views=[]
        for v in data['views']:
            views.append(dict(v,P=np.asarray(v['P']),image=cv2.imread(str(path.parent/(v['camera']+'.png')))))
        position=np.asarray(data['initial_position']);rotation=np.asarray(data['initial_rotation'])
        values={};timings={};calls={}
        # Alternate execution order to reduce systematic warm-cache bias.
        for name,impl in ([('baseline',baseline),('candidate',candidate)] if index%2==0 else
                          [('candidate',candidate),('baseline',baseline)]):
            original=impl.edge_distance;counter=[0]
            def measured(image):counter[0]+=1;return original(image)
            impl.edge_distance=measured
            begin=time.perf_counter()
            pose,diagnostic=impl.refine_with_agreement(views,position,rotation,raster_offset_px=-.5)
            timings[name]=time.perf_counter()-begin;calls[name]=counter[0];impl.edge_distance=original
            values[name]={'pose':None if pose is None else [x.tolist() for x in pose],'diagnostic':diagnostic}
        identical=json.dumps(values['baseline'],sort_keys=True)==json.dumps(values['candidate'],sort_keys=True)
        if not identical:
            repeated_pose,repeated_diagnostic=baseline.refine_with_agreement(views,position,rotation,raster_offset_px=-.5)
            values['baseline_repeat']={'pose':None if repeated_pose is None else [x.tolist() for x in repeated_pose],
                                       'diagnostic':repeated_diagnostic}
            (out/'parity_diagnostic.json').write_text(json.dumps(values,indent=2))
        result=dict(input=str(path.relative_to(ROOT)),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    baseline_wall_s=timings['baseline'],candidate_wall_s=timings['candidate'],
                    speedup=timings['baseline']/timings['candidate'],edge_distance_calls=calls,
                    identical_pose_and_all_diagnostics=identical,reason=values['baseline']['diagnostic']['reason'])
        rows.append(result)
        progress=dict(scope='Historical development RGB only; image-cache microbenchmark, not insertion speed',
                      cpu_threads=1,samples=rows,all_outputs_identical=all(x['identical_pose_and_all_diagnostics'] for x in rows),
                      median_baseline_wall_s=statistics.median(x['baseline_wall_s'] for x in rows),
                      median_candidate_wall_s=statistics.median(x['candidate_wall_s'] for x in rows),
                      median_paired_speedup=statistics.median(x['speedup'] for x in rows),
                      physics_trials=0,installed=False)
        result['pose_identical']=values['baseline']['pose']==values['candidate']['pose']
        result['same_accept_reject_reason']=values['baseline']['diagnostic']['reason']==values['candidate']['diagnostic']['reason']
        progress['qualification']='Uninstalled microbenchmark prototype; no physical speed or robustness qualification'
        (out/'benchmark.json').write_text(json.dumps(progress,indent=2))
        print(json.dumps(result),flush=True)
        if not identical:raise RuntimeError('Exact diagnostic parity failed; inspect saved baseline-repeat control; candidate remains uninstalled')


if __name__=='__main__':main()
