"""Static-development SFP silhouette fitting from pinned visual mesh geometry.

Learned multiview pose initializes the fit; robot FK supplies its own occlusion
mask. Object truth is a scoring-only read after fitting. No runtime gate changes.
"""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'aic_model'),str(ROOT)]
from experiments.lastinch_geometry import MODELS,source_model,sdf_frames,robot_fk
from aic_model.insertion.sc_body_refinement import robot_tool_mask
from aic_model.insertion.sc_rim_refinement import project
from experiments.angular_pose_development46 import errors


def template(material='dark'):
    path=MODELS/'SFP Module/sfp_module_visual.glb';scene=trimesh.load_scene(path);vertices=[];materials=[]
    inv=np.linalg.inv(sdf_frames(source_model('SFP Module'))['sfp_tip_link'])
    for node in scene.graph.nodes_geometry:
        T,key=scene.graph[node];mesh=scene.geometry[key];color=mesh.visual.to_color().vertex_colors[:,:3].astype(float)
        points=trimesh.transform_points(mesh.vertices,inv@T)
        # Use the dark rigid housing, excluding rear receptacle/latch geometry.
        good=((color.max(1)>120)&(np.ptp(color,axis=1)<color.max(1)*.3) if material=='metallic' else
              (color.max(1)<120)&(points[:,2]>=-.043))&(points[:,2]<=.0001)
        materials.append(dict(node=str(node),vertices=len(points),kept=int(good.sum()),
            median_color=np.median(color,axis=0).tolist(),bounds=np.array([points.min(0),points.max(0)]).tolist()))
        if good.any():vertices.append(points[good])
    if not vertices:raise ValueError('No dark housing visual vertices')
    points=np.concatenate(vertices);points=points[ConvexHull(points).vertices]
    return points,dict(mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),materials=materials,hull_points=len(points),bounds=[points.min(0).tolist(),points.max(0).tolist()])


def observations(views,points,position,rotation):
    directions=np.c_[np.cos(np.linspace(0,2*np.pi,96,endpoint=False)),np.sin(np.linspace(0,2*np.pi,96,endpoint=False))];prepared=[]
    for view in views:
        image=view['image'];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);uv=project(view['P'],(rotation@points.T).T+position)-.5
        ignored=cv2.dilate(view['tool_mask'],np.ones((5,5),np.uint8))!=0
        roi=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
        roi=cv2.dilate(roi,np.ones((15,15),np.uint8))!=0
        # Contrast class only; no simulator segmentation/geometry truth.
        dark=(gray<75)&roi&~ignored;y,x=np.nonzero(dark)
        if len(x)<50:raise ValueError('insufficient_dark_shell_pixels')
        observed=np.c_[x,y].astype(float);model_ids=np.argmax(uv@directions.T,axis=0);obs_ids=np.argmax(observed@directions.T,axis=0)
        def visible(coords):
            xy=np.rint(coords).astype(int);h,w=gray.shape;inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            valid=inside.copy();valid[inside]&=~ignored[xy[inside,1],xy[inside,0]];return valid
        keep=visible(uv[model_ids])&visible(observed[obs_ids]);D=directions[keep];boundary=observed[obs_ids[keep]]
        s=np.linspace(-2.5,2.5,51);query=boundary[:,None,:]+D[:,None,:]*s[None,:,None]
        colors=np.stack([map_coordinates(image[:,:,c].astype(float),query[:,:,::-1].reshape(-1,2).T,order=1,mode='nearest').reshape(len(D),len(s)) for c in range(3)],axis=-1)
        foreground=np.median(colors[:,:10],axis=1);background=np.median(colors[:,-10:],axis=1);contrast=foreground-background;norm=np.sum(contrast**2,axis=1)
        alpha=np.sum((colors-background[:,None,:])*contrast[:,None,:],axis=2)/np.maximum(norm[:,None],1.)
        valid=[];offsets=[]
        for i in range(len(D)):
            transitions=np.flatnonzero((alpha[i,:-1]>=.5)&(alpha[i,1:]<.5))
            if norm[i]<30**2 or not len(transitions):continue
            j=min(transitions,key=lambda k:abs(s[k]));offset=s[j]+(.5-alpha[i,j])*(s[j+1]-s[j])/(alpha[i,j+1]-alpha[i,j])
            if abs(offset)<=1.5:valid.append(i);offsets.append(offset)
        if len(valid)<12:raise ValueError('insufficient_unoccluded_subpixel_shell_boundary')
        support=np.max(observed@D.T,axis=0)
        prepared.append((view['P'],D[valid],support[valid]+offsets))
    return prepared


def fit(views,points,position,rotation):
    prepared=observations(views,points,position,rotation)
    def pose(x):return position+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@rotation
    def residual(x,selected):
        p,R=pose(x);world=(R@points.T).T+p
        return np.concatenate([np.max((project(P,world)-.5)@D.T,axis=0)-support for P,D,support in selected])
    starts=[np.zeros(6)]
    for axis in range(3):
        for sign in [-1,1]:
            v=np.zeros(6);v[3+axis]=sign;starts.append(v)
    bounds=([-3.]*3+[-6.]*3,[3.]*3+[6.]*3)
    fits=[least_squares(residual,x,args=(prepared,),bounds=bounds,loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=120) for x in starts]
    best=min(fits,key=lambda x:x.cost);p,R=pose(best.x);world=(R@points.T).T+p;disagreements=[]
    for omit in range(3):
        other=least_squares(residual,best.x,args=([v for i,v in enumerate(prepared) if i!=omit],),bounds=bounds,loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=120)
        op,oR=pose(other.x);model=(oR@points.T).T+op
        disagreements.append(float(np.linalg.norm(model-world,axis=1).max()))
    error=float(np.mean(abs(residual(best.x,prepared))))
    return dict(position=p.tolist(),rotation=R.tolist(),mean_support_error_px=error,
        directions_per_view=[len(D) for _,D,_ in prepared],leave_one_view_out_displacement_m=disagreements,
        accepted=bool(best.success and error<=1. and max(disagreements)<=.0001),ground_truth_used_for_fitting=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--material',choices=['dark','metallic'],default='dark');args=parser.parse_args()
    cv2.setNumThreads(1);out=ROOT/('experiments/angular_development47' if args.material=='dark' else 'experiments/angular_development47/metallic_v2');out.mkdir(parents=True,exist_ok=True)
    if (out/'sfp_shell_results.json').exists():raise FileExistsError('Preserve first candidate experiment')
    points,provenance=template(args.material);(out/'sfp_shell_template.json').write_text(json.dumps(provenance,indent=2))
    cache=json.loads((ROOT/'experiments/parallel_dev46/angular_inputs.json').read_text())['rows']
    seeds={r['case_id']:r.get('joint_fit') for r in json.loads((ROOT/'experiments/parallel_dev46/angular_reprojection_results.json').read_text())['rows']}
    robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot();rows=[]
    with threadpool_limits(limits=1):
        for row in cache:
            if row['connector']!='sfp':continue
            seed=seeds[row['case_id']];tcp=robot_fk(robot,row['own_robot_q'])['gripper/tcp'];views=[];result=dict(case_id=row['case_id'])
            for v in row['inputs']:
                image=cv2.imread(str(ROOT/v['image']));K=np.array(v['K']);T=np.linalg.inv(v['T_world_from_camera'])
                views.append(dict(camera=v['camera'],image=image,P=K@T[:3],tool_mask=robot_tool_mask(K,T,tcp,image.shape)))
            began=time.perf_counter()
            try:
                candidate=fit(views,points,np.array(seed['position']),np.array(seed['rotation']));result['fit']=candidate
                result['scoring_only']=errors(np.array(candidate['position']),np.array(candidate['rotation']),np.array(row['scoring_only']['tip_world']))
            except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
            result['wall_s']=time.perf_counter()-began;rows.append(result)
            (out/'sfp_shell_results.json').write_text(json.dumps(dict(scope='Static development image fit only; original synchronization rejections preserved',
                final_cases_used=0,rows=rows),indent=2));print(json.dumps({k:v for k,v in result.items() if k!='fit'}),flush=True)


if __name__=='__main__':main()
