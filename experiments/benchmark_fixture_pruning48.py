"""Paired original fixture fits with an admissible association prefilter."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import time
import cv2
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT)]
from aic_model.insertion.sfp_fixture_refinement import fit_fixture
from experiments.candidates.efficiency48.epipolar_bound import mean_residual_lower_bound


def main():
    cv2.setNumThreads(1);out=ROOT/'experiments/perception_efficiency48';data=json.loads((out/'detections.json').read_text());rows=[]
    if (out/'fixture_benchmark.json').exists():raise FileExistsError('Preserve experiment')
    with threadpool_limits(limits=1):
        for row in data['rows']:
            cams={name:[{k:np.asarray(v) for k,v in d.items()} for d in item['detections']] for name,item in row['cameras'].items()}
            combinations=[(subset,picks) for count in range(2,len(cams)+1) for subset in itertools.combinations(cams,count)
                          for picks in itertools.product(*(cams[c] for c in subset))]
            results={}
            for mode in ['baseline','candidate']:
                began=time.perf_counter();accepted=[];pruned=0;fit_calls=0
                for index,(subset,picks) in enumerate(combinations):
                    if mode=='candidate' and mean_residual_lower_bound(picks)>5.+1e-8:pruned+=1;continue
                    fit_calls+=1;fit=fit_fixture(picks)
                    if fit is not None:accepted.append(dict(index=index,cameras=subset,fit=fit))
                results[mode]=dict(wall_s=time.perf_counter()-began,accepted=accepted,fit_calls=fit_calls,pruned=pruned)
            identical=results['baseline']['accepted']==results['candidate']['accepted']
            rows.append(dict(run=row['run'],combinations=len(combinations),results=results,
                all_accepted_candidates_identical=identical,speedup=results['baseline']['wall_s']/results['candidate']['wall_s']))
            (out/'fixture_benchmark.json').write_text(json.dumps(dict(scope='Offline original DEVELOPMENT NIC fits, no motion/simulator truth/final cases',
                prefilter='Admissible weighted L1 epipolar lower bound; original nonlinear fitting and 5 px gate retained',rows=rows,
                source_sha256=hashlib.sha256((ROOT/'experiments/candidates/efficiency48/epipolar_bound.py').read_bytes()).hexdigest(),
                all_accepted_candidates_identical=all(r['all_accepted_candidates_identical'] for r in rows)),indent=2))
            print(json.dumps(dict(run=row['run'],identical=identical,speedup=rows[-1]['speedup'],
                baseline_fits=results['baseline']['fit_calls'],candidate_fits=results['candidate']['fit_calls'])),flush=True)
            if not identical:raise RuntimeError('An original accepted candidate changed; do not install')


if __name__=='__main__':main()
