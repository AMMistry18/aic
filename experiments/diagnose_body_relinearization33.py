"""Rebuild occlusion/support correspondences after each bounded image fit.

Offline development diagnostic. No truth is read until all estimates are saved.
"""
import gzip
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_body_refinement import fit_translation


def main():
    rows=[]
    summary=json.loads((ROOT/'experiments/evidence/original_repair/persistent_screen32v1.json').read_text())
    for run in summary['runs']:
        folder=ROOT/'.artifacts/lastinch_runs'/run['run']/run['run']/'body_inputs/000';data=json.loads((folder/'inputs.json').read_text())
        views=[dict(camera=v['camera'],image=cv2.imread(str(folder/(v['camera']+'.png'))),P=np.array(v['P']),
                    tool_mask=cv2.imread(str(folder/(v['camera']+'_tool_mask.png')),0)) for v in data['views']]
        position=np.array(data['initial_position']);rotation=np.array(data['initial_rotation']);iterations=[]
        for _ in range(4):
            result=fit_translation(views,position,rotation);iterations.append(result)
            if 'position' not in result:break
            position=np.array(result['position'])
        rows.append(dict(run=run['run'],stamp=max(v['stamp_s'] for v in data['views']),iterations=iterations))
        print(run['case_id'],[dict(accepted=x['accepted'],reason=x['reason'],loo_mm=(1000*np.array(x.get('leave_one_view_out_disagreement_m',[]))).tolist()) for x in iterations],flush=True)
    out=ROOT/'.artifacts/body_relinearization33';out.mkdir(exist_ok=True)
    (out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    base=np.eye(4);base[:3,:3]=np.diag([-1.,-1.,1.]);base[:3,3]=[-.2,.2,0]
    for row in rows:
        folder=ROOT/'.artifacts/lastinch_runs'/row['run']/row['run'];trace=json.load(gzip.open(folder/'truth_trace.json.gz','rt'))
        actual=min(trace,key=lambda x:abs(x['sim_time_s']-row['stamp']));truth=np.linalg.inv(base)@np.array(actual['tip_world'])
        for fit in row['iterations']:
            if 'position' in fit:fit['diagnostic_error_base_mm']=(1000*(np.array(fit['position'])-truth[:3,3])).tolist()
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
