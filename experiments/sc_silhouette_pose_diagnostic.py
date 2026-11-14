"""Full public SC mesh silhouette fit; offline prototype, never commands motion."""
from functools import lru_cache
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from experiments.lastinch_geometry import MODELS,sdf_frames,source_model
from aic_model.insertion.sc_rim_refinement import project


@lru_cache(maxsize=1)
def template():
    scene=trimesh.load_scene(MODELS/'SC Plug/sc_plug_visual.glb');inv=np.linalg.inv(sdf_frames(source_model('SC Plug'))['sc_tip_link']);triangles=[];points=[]
    for node in scene.graph.nodes_geometry:
        T,name=scene.graph[node];mesh=scene.geometry[name].copy();color=mesh.visual.to_color().vertex_colors[:,:3].astype(float)
        white=(color.max(1)>120)&(np.ptp(color,axis=1)<color.max(1)*.3);faces=mesh.faces[np.all(white[mesh.faces],axis=1)]
        if not len(faces):continue
        mesh=trimesh.Trimesh(mesh.vertices,faces,process=False);mesh.merge_vertices(merge_tex=True,merge_norm=True);mesh.apply_transform(inv@T)
        triangles.append(mesh.triangles)
        crease=mesh.face_adjacency_edges[mesh.face_adjacency_angles>np.deg2rad(25)];edges=mesh.vertices[crease]
        for a,b in edges:
            length=np.linalg.norm(b-a)
            if length<.002:continue
            points.extend(a+np.linspace(0,1,max(3,int(np.ceil(length/.00025))))[:,None]*(b-a))
    return np.concatenate(triangles),np.array(points)


def fit_one(views,p,R,starts=True):
    triangles,points=template();prepared=[]
    for v in views:
        hsv=cv2.cvtColor(v['image'],cv2.COLOR_BGR2HSV);white=((hsv[:,:,1]<60)&(hsv[:,:,2]>80)).astype(np.uint8)*255
        blocked=(cv2.dilate(v['tool_mask'],np.ones((3,3),np.uint8))>0)|(cv2.dilate(cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255])),np.ones((5,5),np.uint8))>0)
        edge=cv2.Canny(white,50,100);edge[blocked]=0;field=cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float)
        prepared.append(dict(v,blocked=blocked,field=field,visible=None))
    def pose(x):return p+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@R
    def update(pos,rot):
        world=(rot@triangles.reshape(-1,3).T).T+pos;worldpoints=(rot@points.T).T+pos
        for v in prepared:
            uv=project(v['P'],world).reshape(-1,3,2)-.5;h,w=v['image'].shape[:2]
            origin=np.maximum(0,np.floor(uv.reshape(-1,2).min(0)-4)).astype(int)
            end=np.minimum([w,h],np.ceil(uv.reshape(-1,2).max(0)+5)).astype(int)
            if np.any(end<=origin):v['visible']=np.zeros(len(points),dtype=bool);continue
            supersample=4;mask=np.zeros(tuple(((end-origin)*supersample)[::-1]),np.uint8)
            valid=(uv[:,:,0].max(1)>=0)&(uv[:,:,0].min(1)<w)&(uv[:,:,1].max(1)>=0)&(uv[:,:,1].min(1)<h)
            # Fill triangles independently: a multi-contour even/odd fill can
            # erase overlapping faces instead of forming their silhouette.
            high=(uv[valid]-origin+.5)*supersample-.5
            for triangle in np.rint(high).astype(np.int32):cv2.fillConvexPoly(mask,triangle,255)
            boundary=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));distance=cv2.distanceTransform(255-boundary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            q=project(v['P'],worldpoints)-.5
            highq=(q-origin+.5)*supersample-.5
            visible=(map_coordinates(distance,highq[:,::-1].T,order=1,mode='constant',cval=255)/supersample<np.sqrt(2)/supersample)&(map_coordinates(v['blocked'].astype(np.uint8),q[:,::-1].T,order=0,mode='constant',cval=1)==0)
            v['visible']=visible
        return all(np.count_nonzero(v['visible'])>=30 for v in prepared)
    if not update(p,R):return {'reason':'insufficient_visible_silhouette_points'}
    def residual(x,weighted=True):
        pos,rot=pose(x);world=(rot@points.T).T+pos
        count=sum(int(v['visible'].sum()) for v in prepared)
        return np.concatenate([np.minimum(map_coordinates(v['field'],(project(v['P'],world[v['visible']])-.5)[:,::-1].T,order=1,mode='constant',cval=20),12)*(np.sqrt(count/(len(prepared)*v['visible'].sum())) if weighted else 1) for v in prepared])
    bounds=([-3]*3+[-5]*3,[3]*3+[5]*3);fits=[]
    for x,y in ([(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)] if starts else [(0,0)]):
        fits.append(least_squares(residual,[x,y,0,0,0,0],bounds=bounds,loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=100))
    best=min(fits,key=lambda x:x.cost)
    for _ in range(3):
        pos,rot=pose(best.x)
        if not update(pos,rot):return {'reason':'insufficient_refined_visible_points'}
        best=least_squares(residual,np.clip(best.x,-np.array([3]*3+[5]*3)+1e-8,np.array([3]*3+[5]*3)-1e-8),bounds=bounds,loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=100)
    pos,rot=pose(best.x)
    return dict(position=pos.tolist(),rotation=rot.tolist(),mean_silhouette_distance_px=float(np.mean(residual(best.x,False))),delta_mm_deg=best.x.tolist(),points_per_view=[int(v['visible'].sum()) for v in prepared],optimizer_success=bool(best.success),bound_hit=bool(np.any(abs(best.x)>=np.array([2.99]*3+[4.99]*3))),view_weighting='equal total squared residual weight per camera',model_visibility_supersample=4)


def fit(views,p,R):
    result=fit_one(views,p,R)
    if 'position' not in result:return result
    pos=np.array(result['position']);rot=np.array(result['rotation']);_,points=template();main=(rot@points.T).T+pos;pairs=[]
    for omit in range(3):
        alt=fit_one([v for i,v in enumerate(views) if i!=omit],pos,rot,starts=False)
        if 'position' not in alt:pairs.append(alt);continue
        RR=np.array(alt['rotation']);pp=np.array(alt['position']);other=(RR@points.T).T+pp
        pairs.append(dict(model_displacement_mm=float(np.max(np.linalg.norm(other-main,axis=1))*1000),rotation_disagreement_deg=float(np.rad2deg(Rotation.from_matrix(rot.T@RR).magnitude())),mean_distance_px=alt['mean_silhouette_distance_px']))
    result.update(leave_one_view_out=pairs,production_enabled=False)
    result['diagnostic_quality_passed']=bool(result['optimizer_success'] and not result['bound_hit'] and result['mean_silhouette_distance_px']<=2 and all(x.get('model_displacement_mm',float('inf'))<=.25 for x in pairs))
    return result
