"""Offline directional edge fit after a declared level-fixture proposal.

Learned candidates supply identity/coarse pose. Public dual-port geometry and
RGB edges supply refinement. Estimates are persisted before diagnostic truth.
"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from experiments.sfp_joint_geometry33 import CORNERS,SLOT1
from aic_model.insertion.sc_rim_refinement import project
from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M


def main():
    run='dev_44000_sfpvisual33v2_original_perception';folder=ROOT/'.artifacts/lastinch_runs'/run/run
    proposal=json.loads((ROOT/'.artifacts/sfp_visual33/fixture_proposal/fits_without_truth.json').read_text())
    plug=np.array(proposal['plug_translation_fit']['position'])
    # Require multi-camera support before attempting precision refinement.
    candidates=[c for c in proposal['candidates'] if len(c['cameras'])>=2]
    def lateral(c):return float(np.linalg.norm((np.array(c['rotation']).T@(plug-np.array(c['position'])))[:2]))
    seed=min(candidates,key=lambda c:(-len(c['cameras']),lateral(c)))
    position=np.array(seed['position']);rotation=np.array(seed['rotation'])
    geometry=json.loads((folder/'perception_before/camera_geometry.json').read_text());prepared=[]
    edges=[np.linspace(c[j],c[(j+1)%4],35) for c in [CORNERS,CORNERS+SLOT1] for j in range(4)]
    for v in geometry:
        image=cv2.imread(str(folder/'perception_before'/(v['camera']+'_raw.png')));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        K=np.array(v['K']);T=np.linalg.inv(v['T_base_from_camera']);P=K@T[:3]
        all_edges=cv2.Canny(gray,12,35);gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        magnitude=np.hypot(gx,gy);mask=np.zeros(gray.shape,np.uint8)
        uv=project(P,(rotation@SFP_PLUG_LOCAL_KEYPOINTS_M.T).T+plug)-.5
        cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype(np.int32)),255);mask=cv2.dilate(mask,np.ones((9,9),np.uint8))
        view=[]
        for points in edges:
            uv=project(P,(rotation@points.T).T+position)-.5;direction=uv[-1]-uv[0];normal=np.array([direction[1],-direction[0]])/np.linalg.norm(direction)
            aligned=abs(gx*normal[0]+gy*normal[1])>=magnitude*np.cos(np.deg2rad(20))
            edge=all_edges.copy();edge[~aligned]=0;edge[mask!=0]=0
            field=cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float)
            view.append((points,field))
        prepared.append((P,mask,view))
    def pose(x):return position+x[:3]*.001,Rotation.from_euler('z',x[3],degrees=True).as_matrix()@rotation
    def fit(selected):
        def residual(x):
            p,R=pose(x);rows=[]
            for i in selected:
                P,mask,view=prepared[i]
                for points,field in view:
                    uv=project(P,(R@points.T).T+p)-.5
                    visible=map_coordinates(mask,uv[:,::-1].T,order=0,mode='constant',cval=255)==0
                    # Keep a fixed-size residual vector as visibility changes.
                    values=map_coordinates(field,uv[:,::-1].T,order=1,mode='constant',cval=20)
                    rows.extend(np.where(visible,np.minimum(values,12),0.))
            return np.array(rows)
        import itertools
        options=[least_squares(residual,[x,y,z,0.],bounds=([-4,-4,-4,-5],[4,4,4,5]),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=100)
                 for x,y,z in itertools.product([0.,-1.5,1.5],repeat=3)]
        best=min(options,key=lambda x:x.cost);return best,float(np.mean(residual(best.x)))
    best,error=fit(range(3));p,R=pose(best.x);result=dict(seed=seed,position=p.tolist(),rotation=R.tolist(),residual_px=error,
        correction_mm_deg=best.x.tolist(),GT_used_for_fitting=False,physics_qualified=False)
    out=ROOT/'.artifacts/sfp_visual33/rim_family';out.mkdir(parents=True,exist_ok=True)
    (out/'fit_without_truth.json').write_text(json.dumps(result,indent=2)+'\n')
    spec=json.loads((folder.parent/'scene.json').read_text());B=np.eye(4);B[:3,:3]=np.diag([-1.,-1.,1.]);B[:3,3]=[-.2,.2,0]
    truth=np.linalg.inv(B)@spec['reset']['mouth_world'];result['diagnostic_error_mm']=(1000*(p-truth[:3,3])).tolist()
    result['diagnostic_angle_deg']=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@R).magnitude()))
    (out/'diagnostic_scores.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
