"""Offline public-CAD outline refinement of original learned port proposals.

The shared normal is the original policy's level-board prior. Only XYZ and yaw
are fitted. Truth is read only after inference/refinement is written to disk.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'flowstate/aic_perception'))
from aic_perception.gripper_masks import GripperMaskBank


def project(P,points):
    q=(P@np.column_stack([points,np.ones(len(points))]).T).T
    return q[:,:2]/q[:,2,None]


def fit_outline(views,points,position,rotation,blue_lower=(90,80,60)):
    # The back of a panel-mounted adapter is occluded. Infer a common local
    # depth clip as a nuisance parameter from pixels, never from board GT.
    facets=ConvexHull(points).simplices
    edges=np.unique(np.sort(np.concatenate([facets[:,[0,1]],facets[:,[1,2]],facets[:,[2,0]]]),axis=1),axis=0)
    def clipped(depth):
        a,b=points[edges[:,0]],points[edges[:,1]]
        crossing=((a[:,2]<depth)&(b[:,2]>depth))|((a[:,2]>depth)&(b[:,2]<depth))
        a,b=a[crossing],b[crossing]
        intersections=a+(b-a)*((depth-a[:,2])/(b[:,2]-a[:,2]))[:,None]
        return np.concatenate([points[points[:,2]<=depth],intersections])
    directions=np.column_stack([np.cos(np.linspace(0,2*np.pi,48,endpoint=False)),np.sin(np.linspace(0,2*np.pi,48,endpoint=False))])
    observations=[];masks=GripperMaskBank()
    for view in views:
        img=view['image'];h,w=img.shape[:2];P=view['P']
        predicted=project(P,(rotation@points.T).T+position)
        center=predicted.mean(0)
        ignored=masks.ignored_pixels(view['camera'],img.shape)
        silhouette=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(silhouette,cv2.convexHull(np.rint(predicted).astype(np.int32)),255)
        if np.count_nonzero((silhouette!=0)&ignored)>np.count_nonzero(silhouette)*.01:continue
        blue=cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),np.array(blue_lower),np.array([130,255,255]))
        blue[ignored]=0
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(blue)
        choices=[]
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_AREA]<100:continue
            delta=float(np.linalg.norm(centroids[i]-center))
            if delta>max(50.,np.linalg.norm(np.ptp(predicted,axis=0))*.6):continue
            choices.append((delta,i))
        if not choices:continue
        _,i=min(choices)
        ys,xs=np.where(labels==i)
        hull=cv2.convexHull(np.column_stack([xs,ys]).astype(np.float32)).reshape(-1,2)
        # A clipped/structurally masked outline cannot certify its full extent.
        if np.any((hull[:,0]<=1)|(hull[:,0]>=w-2)|(hull[:,1]<=1)|(hull[:,1]>=h-2)):continue
        observations.append({'view':view,'hull':hull,'support':np.max(hull@directions.T,axis=0)})
    if len(observations)<2:return {'accepted':False,'reason':'insufficient_complete_outlines','views':len(observations)}
    def transformed(x):
        R=Rotation.from_rotvec([0,0,x[3]]).as_matrix()@rotation
        return (R@clipped(x[4]).T).T+position+x[:3],R
    def residual(x,subset=None):
        world,_=transformed(x)
        return np.concatenate([np.max(project(o['view']['P'],world)@directions.T,axis=0)-o['support']
                               for o in (observations if subset is None else subset)])
    fit=least_squares(residual,np.array([0.,0.,0.,0.,.0175]),bounds=([-.010]*3+[-.175,.005],[.010]*3+[.175,float(points[:,2].max())]),
                      x_scale=[.002,.002,.002,.05,.005],loss='soft_l1',f_scale=2.,diff_step=1e-4,max_nfev=100)
    errors=residual(fit.x)
    world,R=transformed(fit.x)
    result={'accepted':bool(fit.success and np.mean(np.abs(errors))<=2.),'optimizer_success':bool(fit.success),
            'reason':'joint_outline_fit','position':(position+fit.x[:3]).tolist(),'rotation':R.tolist(),
            'mean_abs_support_error_px':float(np.mean(np.abs(errors))),'max_abs_support_error_px':float(np.max(np.abs(errors))),
            'views':[o['view']['camera'] for o in observations],
            'observed_hulls':{o['view']['camera']:o['hull'].tolist() for o in observations},'delta':fit.x.tolist()}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--min-saturation',type=int,default=80);ap.add_argument('--min-value',type=int,default=60)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    audit=json.loads(a.audit.read_text());rows=json.loads(a.candidates.read_text())
    points=np.load(ROOT/'experiments/configs/sc_blue_outline.npz')['points_m']
    views=[{'camera':r['camera'],'image':cv2.imread(str(a.audit.parent/f'{r["camera"]}_raw.png')),
            'P':np.array(r['K'])@np.array(r['T_camera_from_base_exact'])[:3]} for r in audit['views']]
    results=[]
    # Fit every full-frame learned candidate, not a GT-selected neighborhood.
    for i,c in enumerate(rows[0]['candidates']):
        w,x,y,z=c['q_wxyz'];R=Rotation.from_quat([x,y,z,w]).as_matrix()
        result=fit_outline(views,points,np.array(c['X']),R,(90,a.min_saturation,a.min_value))
        result.update(candidate_index=i,learned_position=c['X'])
        results.append(result)
    (a.out/'fits_without_truth.json').write_text(json.dumps(results,indent=2))
    scene=json.loads(a.scene.read_text());base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    truth=np.linalg.inv(base)@np.array(scene['reset']['mouth_world'])
    for result in results:
        if 'position' not in result:continue
        result['diagnostic_GT_center_error_mm']=float(np.linalg.norm(np.array(result['position'])-truth[:3,3])*1000)
        result['diagnostic_learned_GT_error_mm']=float(np.linalg.norm(np.array(result['learned_position'])-truth[:3,3])*1000)
        R=np.array(result['rotation']);world=(R@points.T).T+result['position']
        if result['diagnostic_learned_GT_error_mm']<10:
            for v in views:
                image=v['image'].copy();hull=cv2.convexHull(project(v['P'],world).astype(np.float32)).astype(np.int32)
                cv2.polylines(image,[hull],True,(0,255,255),2)
                cv2.imwrite(str(a.out/f'candidate{result["candidate_index"]}_{v["camera"]}.jpg'),image)
    (a.out/'diagnostic_scores.json').write_text(json.dumps(results,indent=2))
    print(json.dumps([{k:r[k] for k in ['candidate_index','accepted','reason','mean_abs_support_error_px','diagnostic_learned_GT_error_mm','diagnostic_GT_center_error_mm','delta','views'] if k in r} for r in results if r.get('diagnostic_learned_GT_error_mm',999)<10],indent=2))


if __name__=='__main__':main()
