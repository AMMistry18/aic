"""Frozen SC mouth model crop audit, with GT attached only after fitting."""
import argparse
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT/'aic_example_policies')]
from aic_example_policies.ros.perception_core import PerceptionCore
from aic_model.insertion.InsertionPolicy import InsertionPolicy
from aic_model.insertion.sc_controller import sc_multiview_candidates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    archive=json.loads(a.audit.read_text())
    pc=PerceptionCore(sc_weights=str(ROOT/'aic_example_policies/aic_example_policies/ros/weights/best_sc_mouth_pose.pt'))
    logger=SimpleNamespace(info=lambda *x:None,warn=lambda *x:None,error=lambda *x:None)
    policy=InsertionPolicy.__new__(InsertionPolicy);policy._pc=pc
    policy.get_logger=lambda:logger
    def no_tcp():raise RuntimeError('Offline all-target diagnostic, no robot TCP supplied')
    policy._tcp=no_tcp
    rows=[]
    for crop,pad in [(False,24),(True,4),(True,6),(True,12),(True,24)]:
        per_cam={};counts={}
        for view in archive['views']:
            name=view['camera'];image=cv2.imread(str(a.audit.parent/f'{name}_raw.png'))
            dets=pc.detect_sc_pose(image,conf_thresh=.2,crop_refine=crop,crop_pad_scale=pad)
            K=np.array(view['K']);T=np.array(view['T_camera_from_base_exact'])
            counts[name]=len(dets)
            if dets:per_cam[name]=[{'kps':np.asarray(d['kps']),'conf':d['conf'],'K':K,'T':T,'P':pc.build_projection_matrix(K,T)} for d in dets]
        # All camera pairs make absence of a third view explicit; no target pose
        # or ground-truth nearest-neighbor selection participates in these fits.
        candidates=[]
        for names in itertools.combinations(per_cam,2):
            candidates.extend(sc_multiview_candidates(policy,{n:per_cam[n] for n in names}))
        rows.append({'crop':crop,'pad':pad,'detection_counts':counts,'candidates':candidates})
    encode=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)
    (a.out/'candidates_without_truth.json').write_text(json.dumps(rows,indent=2,default=encode))
    scene=json.loads(a.scene.read_text())
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    mouth=np.linalg.inv(base)@np.array(scene['reset']['mouth_world'])
    for row in rows:
        for c in row['candidates']:
            c['diagnostic_GT_center_error_mm']=float(np.linalg.norm(c['X']-mouth[:3,3])*1000)
            if c['fit_X'] is not None:c['diagnostic_GT_fitted_center_error_mm']=float(np.linalg.norm(c['fit_X']-mouth[:3,3])*1000)
        # Diagnostic reporting only, after all candidates have been persisted.
        row['diagnostic_requested_neighborhood']=[c for c in row['candidates'] if c['diagnostic_GT_center_error_mm']<10]
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2,default=encode))
    print(json.dumps([{'crop':r['crop'],'pad':r['pad'],'counts':r['detection_counts'],
        'candidates':len(r['candidates']),'requested_neighborhood':[{k:c[k] for k in ['centre_reproj_px','reproj_px','opening_residual_m','diagnostic_GT_center_error_mm','fit_reproj_px','diagnostic_GT_fitted_center_error_mm'] if k in c} for c in r['diagnostic_requested_neighborhood']]} for r in rows],indent=2))


if __name__=='__main__':main()
