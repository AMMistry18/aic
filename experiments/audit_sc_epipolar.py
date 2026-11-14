"""Read-only pixel-objective / epipolar diagnostic on the synchronized SC audit.

Optimization uses camera calibration and detected pixels only. Ground truth is
read afterwards for diagnostic comparison; no fitted pose is sent to a policy.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as POINTS

def project(points,T,K):
    xyz=(T[:3,:3]@points.T).T+T[:3,3]
    uv=(K@xyz.T).T
    return uv[:,:2]/uv[:,2:]

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=json.loads(a.audit.read_text());rows=[r for r in raw['views'] if 'predicted_px' in r];pairs=[]
    for left,right in itertools.combinations(rows,2):
        T=np.array(right['T_camera_from_base_exact'])@np.linalg.inv(left['T_camera_from_base_exact']);t=T[:3,3]
        skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
        F=np.linalg.inv(right['K']).T@skew@T[:3,:3]@np.linalg.inv(left['K'])
        x=np.column_stack([left['predicted_px'],np.ones(8)]);y=np.column_stack([right['predicted_px'],np.ones(8)])
        Fx=(F@x.T).T;Fty=(F.T@y.T).T;n=np.sum(y*Fx,axis=1)
        dist=np.abs(n)/np.sqrt(np.sum(Fx[:,:2]**2,axis=1)+np.sum(Fty[:,:2]**2,axis=1))
        pairs.append({'cameras':[left['camera'],right['camera']],'Sampson_distance_px':dist.tolist(),'mean_px':float(dist.mean())})
    # audit.json's predicted_px were captured with the configured pad-6 pass;
    # the subsequent fusion sweep also includes DIFFERENT full-frame detections.
    initial=next(x for x in raw['fusion'] if x['crop'] and x['pad']==6)
    x0=np.r_[Rotation.from_matrix(initial['rotation']).as_rotvec(),initial['position']]
    def residual(x,weighted=False):
        points=(Rotation.from_rotvec(x[:3]).as_matrix()@POINTS.T).T+x[3:];errors=[]
        for row in rows:
            e=project(points,np.array(row['T_camera_from_base_exact']),np.array(row['K']))-row['predicted_px']
            valid=np.array(row['confidence'])>=.15
            errors.extend((e[valid]*(np.sqrt(np.array(row['confidence'])[valid])[:,None] if weighted else 1)).ravel())
        return np.array(errors)
    fits=[]
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    truth=np.linalg.inv(base)@rows[0]['tip_world']
    for weighted in (False,True):
        fit=least_squares(lambda x:residual(x,weighted),x0,max_nfev=500,xtol=1e-12,ftol=1e-12,gtol=1e-12)
        errors=np.linalg.norm(residual(fit.x).reshape(-1,2),axis=1)
        fits.append({'confidence_weighted':weighted,'mean_reprojection_px':float(errors.mean()),'max_reprojection_px':float(errors.max()),
            'GT_tip_error_mm_diagnostic_only':float(np.linalg.norm(fit.x[3:]-truth[:3,3])*1000),
            'GT_rotation_error_deg_diagnostic_only':float(Rotation.from_matrix(truth[:3,:3].T@Rotation.from_rotvec(fit.x[:3]).as_matrix()).magnitude()*180/np.pi),
            'position_base':fit.x[3:].tolist(),'converged':bool(fit.success)})
    output={'purpose':'diagnostic_only_no_policy_commands','input_detection_mode':'recorded_configured_pad_6',
        'initial_pose_reprojection_recomputed_px':float(np.linalg.norm(residual(x0).reshape(-1,2),axis=1).mean()),
        'original_mean_reprojection_px':initial['reprojection_px'],'unchanged_gate_px':6.,'epipolar':pairs,'direct_pixel_rigid_fits':fits}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
