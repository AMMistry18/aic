"""Measure exact image/CAD reuse; preserve policy gate decisions and report drift.

Numerical deviations are compared with an original self-repeat control. No
physical/perception threshold is changed. This is offline development only.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def difference(a,b,points):
    if (a[0] is None)!=(b[0] is None):return dict(same_decision=False)
    result=dict(same_decision=a[1]['reason']==b[1]['reason'],bit_identical=json.dumps(a[1],sort_keys=True)==json.dumps(b[1],sort_keys=True))
    if a[0] is not None:
        wa=(a[0][1]@points.T).T+a[0][0];wb=(b[0][1]@points.T).T+b[0][0]
        result['maximum_modeled_displacement_m']=float(np.linalg.norm(wa-wb,axis=1).max())
    values=[]
    def compare(x,y):
        if isinstance(x,dict):
            for k in x:
                if k in y:compare(x[k],y[k])
        elif isinstance(x,list):
            for u,v in zip(x,y):compare(u,v)
        elif isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)):
            values.append(abs(x-y))
    compare(a[1],b[1]);result['maximum_raw_diagnostic_difference_mixed_units']=max(values,default=0.)
    return result


def main():
    cv2.setNumThreads(1);out=ROOT/'experiments/perception_efficiency49';out.mkdir(exist_ok=True)
    if (out/'benchmark.json').exists():raise FileExistsError('Preserve first experiment')
    baseline=module('rim_original49',ROOT/'aic_model/aic_model/insertion/sc_rim_refinement.py')
    candidate=module('rim_reuse49',ROOT/'experiments/candidates/efficiency49/sc_rim_refinement.py')
    points=candidate.template_points();rows=[];paths=[]
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text())
    for run in ledger['results']:
        if run['case']['connector']!='sc' or run.get('status')!='completed':continue
        folder=ROOT/run['evidence_dir'].replace('\\','/')/run['run']/'rim_inputs'
        for stamp in ['000','003','006']:
            path=folder/stamp/'inputs.json'
            if path.exists():paths.append(path)
        if len(paths)>=12:break
    with threadpool_limits(limits=1):
        for index,path in enumerate(paths[:12]):
            data=json.loads(path.read_text());views=[dict(v,P=np.array(v['P']),image=cv2.imread(str(path.parent/(v['camera']+'.png')))) for v in data['views']]
            position=np.array(data['initial_position']);rotation=np.array(data['initial_rotation']);outputs={};times={};calls={}
            for name,impl in [('baseline',baseline),('candidate',candidate),('baseline_repeat',baseline)]:
                original=impl.edge_distance;counter=[0]
                def measured(image):counter[0]+=1;return original(image)
                impl.edge_distance=measured;begin=time.perf_counter()
                outputs[name]=impl.refine_with_agreement(views,position,rotation,raster_offset_px=-.5)
                times[name]=time.perf_counter()-begin;calls[name]=counter[0];impl.edge_distance=original
            row=dict(input=str(path.relative_to(ROOT)),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),times_s=times,
                edge_distance_calls=calls,candidate_vs_original=difference(outputs['baseline'],outputs['candidate'],points),
                original_repeat=difference(outputs['baseline'],outputs['baseline_repeat'],points))
            rows.append(row)
            result=dict(scope='Offline historical SC development RGB/CAD reuse, no insertion claim',final_cases_used=0,
                original_thresholds_unchanged=True,installed=False,rows=rows,
                all_gate_decisions_match=all(r['candidate_vs_original']['same_decision'] for r in rows),
                maximum_modeled_displacement_m=max(r['candidate_vs_original'].get('maximum_modeled_displacement_m',0.) for r in rows),
                median_original_s=statistics.median(r['times_s']['baseline'] for r in rows),
                median_candidate_s=statistics.median(r['times_s']['candidate'] for r in rows),
                source_sha256=hashlib.sha256((ROOT/'experiments/candidates/efficiency49/sc_rim_refinement.py').read_bytes()).hexdigest())
            (out/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
            if not row['candidate_vs_original']['same_decision']:raise RuntimeError('Gate decision changed, candidate remains uninstalled')


if __name__=='__main__':main()
