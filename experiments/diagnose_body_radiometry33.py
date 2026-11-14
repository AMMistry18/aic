"""Offline SC subpixel fits in linear light; immutable source remains unchanged."""
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion import sc_body_refinement as body


def main():
    source=inspect.getsource(body.observations)
    assert source.count("image[:,:,c].astype(float)")==1
    source=source.replace("image[:,:,c].astype(float)","linear[:,:,c]")
    source=source.replace("image=view['image'];hsv=", "image=view['image'];srgb=image.astype(float)/255.;linear=255.*np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4);hsv=")
    namespace=body.__dict__.copy();exec(source,namespace);body.observations=namespace['observations']
    rows=[];summary=json.loads((ROOT/'experiments/evidence/original_repair/persistent_screen32v1.json').read_text())
    for run in summary['runs']:
        folder=ROOT/'.artifacts/lastinch_runs'/run['run']/run['run']/'body_inputs/000';d=json.loads((folder/'inputs.json').read_text())
        views=[dict(camera=v['camera'],image=cv2.imread(str(folder/(v['camera']+'.png'))),P=np.array(v['P']),
                    tool_mask=cv2.imread(str(folder/(v['camera']+'_tool_mask.png')),0)) for v in d['views']]
        result=body.fit_translation(views,np.array(d['initial_position']),np.array(d['initial_rotation']))
        rows.append(dict(run=run['run'],stamp=max(v['stamp_s'] for v in d['views']),**result))
    out=ROOT/'.artifacts/body_linear33';out.mkdir(exist_ok=True)
    (out/'fits_without_truth.json').write_text(json.dumps(dict(transformed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),fits=rows),indent=2)+'\n',encoding='utf-8')
    base=np.eye(4);base[:3,:3]=np.diag([-1.,-1.,1.]);base[:3,3]=[-.2,.2,0]
    for row in rows:
        if 'position' not in row:continue
        folder=ROOT/'.artifacts/lastinch_runs'/row['run']/row['run'];trace=json.load(gzip.open(folder/'truth_trace.json.gz','rt'))
        actual=min(trace,key=lambda x:abs(x['sim_time_s']-row['stamp']));truth=np.linalg.inv(base)@np.array(actual['tip_world'])
        row['diagnostic_error_base_mm']=(1000*(np.array(row['position'])-truth[:3,3])).tolist()
        print(row['run'].split('_persistent')[0],row['accepted'],row['diagnostic_error_base_mm'],[round(x*1000,3) for x in row.get('leave_one_view_out_disagreement_m',[])])
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
