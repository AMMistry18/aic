"""Summarize executed visual rejections without feeding truth to any policy."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();results=[]
    for trial in json.loads(a.index.read_text())['results']:
        if 'visual' not in trial['method'] or trial['status']!='completed':continue
        folder=ROOT/trial['evidence_dir']/trial['run']
        policy=json.loads((folder/'policy.json').read_text());reasons=Counter();pairs=[];views=Counter();fused=0
        for frame in policy.get('vision',[]):
            fused+=frame.get('estimate') is not None;hits=[]
            for diag in frame.get('diagnostics',[]):
                reasons[diag.get('reason',diag.get('error','unknown').split(':')[0])]+=1
                if 'plane_point' in diag:
                    hits.append(np.array(diag['plane_point']));views[diag['camera']]+=1
            if len(hits)>=2:pairs.append(min(float(np.linalg.norm(x-y))*1000 for x,y in itertools.combinations(hits,2)))
        results.append({'run':trial['run'],'reason':policy['reason'],'frames':len(policy.get('vision',[])),
            'fused_frames':fused,'view_reasons':dict(reasons),'accepted_hits_per_camera':dict(views),
            'frames_with_multiple_hits':len(pairs),'closest_pair_distance_mm_median':float(np.median(pairs)) if pairs else None,
            'closest_pair_distance_mm_min':min(pairs,default=None),'unchanged_view_disagreement_limit_mm':.6})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
