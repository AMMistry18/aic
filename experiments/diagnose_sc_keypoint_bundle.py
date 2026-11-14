"""Compare original DLT/Kabsch and direct weighted reprojection, offline only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as POINTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    replay=json.loads(a.replay.read_text());audit=json.loads(a.audit.read_text());views={v['camera']:v for v in audit['views']};rows=[]
    for row in replay['conditions']:
        if 'position' not in row:continue
        p0=np.array(row['position']);R0=np.array(row['rotation']);obs=[]
        for d in row['detections']:
            v=views[d['camera']];P=np.array(v['K'])@np.array(v['T_camera_from_base_exact'])[:3]
            confidence=np.array(d['keypoint_confidences']);keep=confidence>=.15
            obs.append((P,POINTS[keep],np.array(d['keypoints_px'])[keep],confidence[keep]))
        def pose(x):return p0+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@R0
        def residual(x,weighted=True):
            pos,R=pose(x);res=[]
            for P,points,uv,c in obs:
                world=(R@points.T).T+pos;q=(P@np.c_[world,np.ones(len(world))].T).T
                errors=q[:,:2]/q[:,2:]-uv
                res.append((errors*(np.sqrt(c[:,None]) if weighted else 1)).ravel())
            return np.concatenate(res)
        for loss in ['linear','soft_l1']:
            fit=least_squares(residual,np.zeros(6),loss=loss,f_scale=1,bounds=([-3]*3+[-5]*3,[3]*3+[5]*3),diff_step=1e-4,max_nfev=200)
            pos,R=pose(fit.x)
            rows.append(dict(crop_refine=row['crop_refine'],pad=row['pad'],loss=loss,position=pos.tolist(),rotation=R.tolist(),delta_mm_deg=fit.x.tolist(),mean_reprojection_px=float(np.linalg.norm(residual(fit.x,False).reshape(-1,2),axis=1).mean()),original_mean_reprojection_px=row['reprojection_px'],GT_used_for_fit=False))
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    B=np.eye(4);B[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();B[:3,3]=[-.2,.2,0]
    truth=np.linalg.inv(B)@np.array(audit['views'][0]['tip_world'])
    for r in rows:r.update(diagnostic_position_mm=float(np.linalg.norm(np.array(r['position'])-truth[:3,3])*1000),diagnostic_angle_deg=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@np.array(r['rotation'])).magnitude())))
    (a.out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['position','rotation','delta_mm_deg']} for r in rows],indent=2))


if __name__=='__main__':main()
