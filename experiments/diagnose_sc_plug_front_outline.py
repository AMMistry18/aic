"""Development CAD outline audit initialized by the original SC plug network."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
from experiments.lastinch_geometry import ROOT,MODELS,sdf_frames,source_model
sys.path.insert(0,str(ROOT/'flowstate/aic_perception'))
from aic_perception.gripper_masks import GripperMaskBank
from experiments.diagnose_sc_outline import project


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--replay',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    path=MODELS/'SC Plug/sc_plug_visual.glb';scene=trimesh.load_scene(path);vertices=[]
    for node in scene.graph.nodes_geometry:
        T,name=scene.graph[node];mesh=scene.geometry[name];color=mesh.visual.to_color().vertex_colors[:,:3].astype(float)
        white=(color.max(1)>120)&(np.ptp(color,axis=1)<color.max(1)*.3)
        vertices.append(trimesh.transform_points(mesh.vertices[white],T))
    body=np.concatenate(vertices);frame=sdf_frames(source_model('SC Plug'))['sc_tip_link']
    points=(np.linalg.inv(frame)@np.column_stack([body,np.ones(len(body))]).T).T[:,:3]
    points=points[ConvexHull(points).vertices]
    facets=ConvexHull(points).simplices;edges=np.unique(np.sort(np.concatenate([facets[:,[0,1]],facets[:,[1,2]],facets[:,[2,0]]]),axis=1),axis=0)
    def clipped(z):
        l,r=points[edges[:,0]],points[edges[:,1]];cross=((l[:,2]<z)&(r[:,2]>z))|((l[:,2]>z)&(r[:,2]<z));l,r=l[cross],r[cross]
        return np.concatenate([points[points[:,2]>=z],l+(r-l)*((z-l[:,2])/(r[:,2]-l[:,2]))[:,None]])
    audit=json.loads(a.audit.read_text());replay=json.loads(a.replay.read_text());estimate=replay['conditions'][-1]
    position=np.array(estimate['position']);rotation=np.array(estimate['rotation']);masks=GripperMaskBank()
    directions=np.column_stack([np.cos(np.linspace(0,2*np.pi,48,endpoint=False)),np.sin(np.linspace(0,2*np.pi,48,endpoint=False))]);observations=[]
    for view in audit['views']:
        name=view['camera'];image=cv2.imread(str(a.audit.parent/f'{name}_raw.png'));P=np.array(view['K'])@np.array(view['T_camera_from_base_exact'])[:3]
        center=project(P,position.reshape(1,3))[0]
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);white=cv2.inRange(hsv,np.array([0,0,65]),np.array([179,90,255]));white[masks.ignored_pixels(name,image.shape)]=0
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(white)
        predicted=project(P,(rotation@clipped(-.030).T).T+position)
        lower=predicted.min(0)-10;upper=predicted.max(0)+10
        choices=[(np.linalg.norm(centroids[i]-center),i) for i in range(1,count)
                 if 15<stats[i,cv2.CC_STAT_AREA]<20000 and np.all(centroids[i]>=lower) and np.all(centroids[i]<=upper)]
        if not choices:continue
        selected=[i for distance,i in choices if distance<120]
        if not selected:continue
        ys,xs=np.where(np.isin(labels,selected));hull=cv2.convexHull(np.column_stack([xs,ys]).astype(np.float32)).reshape(-1,2)
        axis=project(P,(position+rotation[:,2]*.005).reshape(1,3))[0]-center;axis/=np.linalg.norm(axis)
        front=directions[directions@axis>.3]
        observations.append({'name':name,'image':image,'P':P,'hull':hull,'directions':front,'support':np.max(hull@front.T,axis=0)})
    def world(x):
        R=Rotation.from_rotvec(x[3:6]).as_matrix()@rotation
        return (R@clipped(-.030).T).T+position+x[:3],R
    def residual(x):
        pts,R=world(x)
        return np.concatenate([np.max(project(o['P'],pts)@o['directions'].T,axis=0)-o['support'] for o in observations])
    fit=least_squares(residual,np.zeros(6),bounds=([-.003]*3+[-.0873]*3,[.003]*3+[.0873]*3),
                      x_scale=[.001]*3+[.02]*3,loss='soft_l1',f_scale=2.,diff_step=1e-4,max_nfev=200)
    pts,R=world(fit.x);pos=position+fit.x[:3]
    result={'purpose':'offline development prototype, not a production policy','GT_used_for_fit':False,'source_mesh_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'position':pos.tolist(),'rotation':R.tolist(),'mean_abs_support_error_px':float(np.mean(np.abs(residual(fit.x)))),
            'delta':fit.x.tolist(),'success':bool(fit.success),'views':[o['name'] for o in observations]}
    (a.out/'fit_without_truth.json').write_text(json.dumps(result,indent=2))
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0];truth=np.linalg.inv(base)@audit['views'][0]['tip_world']
    result['diagnostic_GT_position_error_mm']=float(np.linalg.norm(pos-truth[:3,3])*1000)
    result['diagnostic_GT_rotation_error_deg']=float(np.rad2deg(Rotation.from_matrix(truth[:3,:3].T@R).magnitude()))
    for o in observations:
        image=o['image'].copy();hull=cv2.convexHull(project(o['P'],pts).astype(np.float32)).astype(np.int32);cv2.polylines(image,[hull],True,(0,255,255),2)
        cv2.imwrite(str(a.out/f'{o["name"]}.jpg'),image)
    (a.out/'diagnostic_scores.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
