"""Public-mesh crease/LSD registration prototype; offline diagnostic only."""
from functools import lru_cache
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from experiments.lastinch_geometry import MODELS,sdf_frames,source_model
from aic_model.insertion.sc_rim_refinement import project


@lru_cache(maxsize=1)
def template():
    scene=trimesh.load_scene(MODELS/'SC Plug/sc_plug_visual.glb');inv=np.linalg.inv(sdf_frames(source_model('SC Plug'))['sc_tip_link']);edges=[];normals=[]
    for node in scene.graph.nodes_geometry:
        T,name=scene.graph[node]
        if not name.startswith(('HOUSING','ferula')):continue
        mesh=scene.geometry[name].copy();mesh.merge_vertices(merge_tex=True,merge_norm=True);mesh.apply_transform(inv@T)
        adjacency=mesh.face_adjacency;creases=mesh.face_adjacency_angles>np.deg2rad(25)
        vertices=mesh.vertices[mesh.face_adjacency_edges[creases]];N=mesh.face_normals[adjacency[creases]]
        keep=np.linalg.norm(vertices[:,1]-vertices[:,0],axis=1)>.002
        edges.extend(vertices[keep]);normals.extend(N[keep])
    return np.array(edges),np.array(normals)


def fit(views,p,R):
    edges,normals=template();prepared=[]
    for v in views:
        image=v['image'];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);white=((hsv[:,:,1]<60)&(hsv[:,:,2]>80)).astype(np.uint8)
        blocked=(cv2.dilate(v['tool_mask'],np.ones((3,3),np.uint8))>0)| (cv2.dilate(cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255])),np.ones((5,5),np.uint8))>0)
        nearwhite=cv2.dilate(white,np.ones((3,3),np.uint8))>0;lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))[0];kept=[]
        for uv in ([] if lines is None else lines.reshape(-1,2,2)):
            if np.linalg.norm(uv[1]-uv[0])<8:continue
            xy=np.rint(uv[0]+np.linspace(.1,.9,9)[:,None]*(uv[1]-uv[0])).astype(int);h,w=white.shape
            if np.any(xy<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]>=h):continue
            yy,xx=xy[:,1],xy[:,0]
            if np.all(nearwhite[yy,xx]&~blocked[yy,xx]):kept.append(uv)
        prepared.append(dict(v,lines=np.array(kept),blocked=blocked))
    def correspondences(pos,rot,selected):
        world=(rot@edges.reshape(-1,3).T).T.reshape(-1,2,3)+pos;wn=(rot@normals.reshape(-1,3).T).T.reshape(-1,2,3);pairs=[];counts={}
        for v in selected:
            lines=v['lines'];counts[v['camera']]=0
            if not len(lines):continue
            T=np.linalg.solve(v['K'],v['P']);camera=np.linalg.inv(np.vstack([T,[0,0,0,1]]))[:3,3]
            front=np.any(np.sum(wn*(camera-world.mean(1))[:,None,:],axis=2)>0,axis=1)
            projected=project(v['P'],world.reshape(-1,3)).reshape(-1,2,2)-.5
            dirs=lines[:,1]-lines[:,0];lengths=np.linalg.norm(dirs,axis=1);dirs/=lengths[:,None];line_normal=np.c_[-dirs[:,1],dirs[:,0]]
            for i,uv in enumerate(projected):
                if not front[i]:continue
                d=uv[1]-uv[0];length=np.linalg.norm(d)
                if length<8:continue
                xy=np.rint(uv[0]+np.linspace(.1,.9,5)[:,None]*d).astype(int);h,w=v['blocked'].shape
                if np.any(xy<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]>=h) or np.any(v['blocked'][xy[:,1],xy[:,0]]):continue
                angle=np.rad2deg(np.arccos(np.clip(abs(dirs@(d/length)),0,1)))
                perp=np.abs(np.sum((uv[None,:,:]-lines[:,0,None,:])*line_normal[:,None,:],axis=2)).mean(1)
                along=np.sum((uv[None,:,:]-lines[:,0,None,:])*dirs[:,None,:],axis=2);overlap=np.maximum(0,np.minimum(along.max(1),lengths)-np.maximum(along.min(1),0))
                valid=(angle<8)&(perp<4)&(overlap>=np.minimum(lengths,length)*.5)
                if not np.any(valid):continue
                cost=perp+angle*.1;j=int(np.argmin(np.where(valid,cost,np.inf)));pairs.append((v['P'],edges[i],lines[j,0],line_normal[j]));counts[v['camera']]+=1
        return pairs,counts
    def optimize(selected):
        current_p=p.copy();current_R=R.copy();last_count={};cost=None;delta=None
        for _ in range(5):
            pairs,last_count=correspondences(current_p,current_R,selected)
            if len(pairs)<8:return None,{'reason':'insufficient_cad_line_matches','matches':last_count}
            def pose(x):return p+x[:3]*.001,Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@R
            def residual(x):
                pp,RR=pose(x);return np.concatenate([np.sum((project(P,(RR@e.T).T+pp)-.5-origin)*normal,axis=1) for P,e,origin,normal in pairs])
            initial=np.r_[(current_p-p)*1000,np.rad2deg(Rotation.from_matrix(current_R@R.T).as_rotvec())]
            initial=np.clip(initial,-np.array([3]*3+[5]*3)+1e-8,np.array([3]*3+[5]*3)-1e-8)
            fit=least_squares(residual,initial,bounds=([-3]*3+[-5]*3,[3]*3+[5]*3),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=150)
            current_p,current_R=pose(fit.x);cost=float(np.mean(abs(residual(fit.x))));delta=fit.x.tolist()
        return (current_p,current_R),dict(matches=last_count,mean_line_distance_px=cost,delta_mm_deg=delta)
    pose,d=optimize(prepared)
    if pose is None:return d
    pos,rot=pose;loo=[]
    for omit in range(3):
        pair,dd=optimize([v for i,v in enumerate(prepared) if i!=omit]);loo.append(None if pair is None else dict(angle_deg=float(np.rad2deg(Rotation.from_matrix(rot.T@pair[1]).magnitude())),position_mm=float(np.linalg.norm(pos-pair[0])*1000)))
    return dict(d,rotation=rot.tolist(),position=pos.tolist(),leave_one_view_out=loo,production_enabled=False)
