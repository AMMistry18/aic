"""Offline image-only translation fit under a declared aligned-handoff prior.

All poses are fitted before reading evaluator truth for diagnostic scoring.
Uses actual first-rim images and their image-derived port rotation. No grasp
constant, per-trial object truth, or calibrated residual offset enters fitting.
"""
import argparse
import gzip
import json
from pathlib import Path
import re
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.diagnose_sc_plug_masked_outline import fit,template


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    points=template();rows=[]
    for name in a.runs:
        path=ROOT/'.artifacts/lastinch_runs'/name/name
        policy=json.loads((path/'policy.json').read_text());folder=path/'rim_inputs/000'
        inputs=json.loads((folder/'inputs.json').read_text())
        log=(ROOT/f'.artifacts/container/{name}_policy.log').read_text()
        match=re.search(r'plug-pose primed:.*?tip=\[([^\]]+)\]',log)
        initial=np.array([float(v) for v in match.group(1).split(',')])
        rotation=np.array(policy['sc_rim_diagnostics'][0]['rotation'])
        views=[{'camera':v['camera'],'P':np.array(v['P']),
                'image':cv2.imread(str(folder/f'{v["camera"]}.png'))} for v in inputs['views']]
        result=fit(views,points,initial,rotation,subpixel=True,robot_mask=True,translation_only=True)
        result.update(run=name,initial_position=initial.tolist(),prior='image-derived port rotation; upstream angular alignment assumed',
                      camera_stamps_s=[v['stamp_s'] for v in inputs['views']],GT_used_for_fit=False)
        if 'position' in result:
            pairs=[]
            for omit in range(3):
                pair=fit([v for i,v in enumerate(views) if i!=omit],points,np.array(result['position']),rotation,
                         subpixel=True,robot_mask=True,translation_only=True)
                pairs.append(pair)
            result['leave_one_view_out']=pairs
            result['pair_position_disagreement_m']=[float(np.linalg.norm(np.array(x['position'])-result['position'])) if 'position' in x else None for x in pairs]
        rows.append(result)
    (a.out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    base_R=Rotation.from_euler('z',-np.pi).as_matrix();base_t=np.array([-.2,.2,0.])
    for row in rows:
        if 'position' not in row:continue
        path=ROOT/'.artifacts/lastinch_runs'/row['run']/row['run']
        truth=json.load(gzip.open(path/'verification_trace.json.gz','rt'))
        t=max(row['camera_stamps_s']);actual=min(truth,key=lambda r:abs(r['sim_time_s']-t))
        tip=np.array(actual['tip_world']);pb=base_R.T@(tip[:3,3]-base_t)
        error=np.array(row['position'])-pb
        row.update(diagnostic_position_error_mm=float(np.linalg.norm(error)*1000),
                   diagnostic_initial_error_mm=float(np.linalg.norm(np.array(row['initial_position'])-pb)*1000),
                   diagnostic_error_base_mm=(error*1000).tolist(),
                   diagnostic_prior_angle_deg=float(Rotation.from_matrix((base_R.T@tip[:3,:3]).T@np.array(row['rotation'])).magnitude()*180/np.pi))
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k in ['run','failure','mean_support_error_px','directions_per_view','pair_position_disagreement_m','diagnostic_position_error_mm','diagnostic_initial_error_mm','diagnostic_prior_angle_deg']} for r in rows],indent=2))


if __name__=='__main__':main()
