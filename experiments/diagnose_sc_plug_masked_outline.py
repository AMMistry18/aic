"""Offline white CAD silhouette fit using only unoccluded support directions."""
import json
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'flowstate/aic_perception')]
from experiments.lastinch_geometry import MODELS,source_model,sdf_frames
from experiments.diagnose_sc_rim import project
from aic_perception.gripper_masks import GripperMaskBank


def template():
    scene=trimesh.load_scene(MODELS/'SC Plug/sc_plug_visual.glb');vertices=[]
    inv=np.linalg.inv(sdf_frames(source_model('SC Plug'))['sc_tip_link'])
    for node in scene.graph.nodes_geometry:
        T,name=scene.graph[node];mesh=scene.geometry[name];color=mesh.visual.to_color().vertex_colors[:,:3].astype(float)
        white=(color.max(1)>120)&(np.ptp(color,axis=1)<color.max(1)*.3)
        vertices.append(trimesh.transform_points(mesh.vertices[white],inv@T))
    points=np.concatenate(vertices);return points[ConvexHull(points).vertices]


def fit(views,points,position,rotation,*,subpixel=False,robot_mask=False,translation_only=False):
    bank=GripperMaskBank();observations=[]
    directions=np.c_[np.cos(np.linspace(0,2*np.pi,96,endpoint=False)),np.sin(np.linspace(0,2*np.pi,96,endpoint=False))]
    for v in views:
        image=v['image'];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        uv=project(v['P'],(rotation@points.T).T+position)-.5
        if robot_mask:
            mask=cv2.imread(str(ROOT/'.artifacts/reference30/robot_masks'/f'{v["camera"]}.png'),cv2.IMREAD_GRAYSCALE)
            ignored=cv2.dilate(mask,np.ones((3,3),np.uint8))!=0
        else:
            ignored=bank.ignored_pixels(v['camera'],image.shape)
        orange=cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255]))!=0
        occluded=ignored|orange
        expanded=cv2.dilate(occluded.astype(np.uint8),np.ones((5,5),np.uint8))!=0
        roi=np.zeros(image.shape[:2],np.uint8);cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
        roi=cv2.dilate(roi,np.ones((15,15),np.uint8))!=0
        white=(hsv[:,:,1]<60)&(hsv[:,:,2]>80)&~occluded&roi
        y,x=np.nonzero(white)
        if len(x)<50:return {'failure':'insufficient_white_pixels'}
        observed=np.c_[x,y].astype(float)
        model_ids=np.argmax(uv@directions.T,axis=0);obs_ids=np.argmax(observed@directions.T,axis=0)
        def visible(coords):
            xy=np.rint(coords).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<image.shape[1])&(xy[:,1]>=0)&(xy[:,1]<image.shape[0])
            valid=inside.copy();valid[inside]&=~expanded[xy[inside,1],xy[inside,0]];return valid
        keep=visible(uv[model_ids])&visible(observed[obs_ids])
        D=directions[keep]
        if len(D)<12:return {'failure':'insufficient_unoccluded_directions'}
        support=np.max(observed@D.T,axis=0)
        if subpixel:
            boundary=observed[obs_ids[keep]];s=np.linspace(-2.5,2.5,51)
            query=boundary[:,None,:]+D[:,None,:]*s[None,:,None]
            colors=np.stack([map_coordinates(image[:,:,c].astype(float),query[:,:,::-1].reshape(-1,2).T,order=1,mode='nearest').reshape(len(D),len(s)) for c in range(3)],axis=-1)
            foreground=np.median(colors[:,:10],axis=1);background=np.median(colors[:,-10:],axis=1)
            contrast=foreground-background;norm=np.sum(contrast**2,axis=1)
            alpha=np.sum((colors-background[:,None,:])*contrast[:,None,:],axis=2)/np.maximum(norm[:,None],1.)
            offsets=[];valid=[]
            for i in range(len(D)):
                transitions=np.flatnonzero((alpha[i,:-1]>=.5)&(alpha[i,1:]<.5))
                if norm[i]<30**2 or not len(transitions):continue
                j=min(transitions,key=lambda k:abs(s[k]))
                offset=s[j]+(.5-alpha[i,j])*(s[j+1]-s[j])/(alpha[i,j+1]-alpha[i,j])
                if abs(offset)>1.5:continue
                valid.append(i);offsets.append(offset)
            if len(valid)<12:return {'failure':'insufficient_subpixel_boundary_support'}
            D=D[valid];support=support[valid]+offsets
        observations.append((v['P'],D,support))
    def pose(p):return position+p[:3]*.001,(rotation if translation_only else Rotation.from_rotvec(p[3:]*np.pi/180).as_matrix()@rotation)
    def residual(p):
        xyz,R=pose(p);world=(R@points.T).T+xyz
        return np.concatenate([np.max((project(P,world)-.5)@D.T,axis=0)-support for P,D,support in observations])
    fits=[least_squares(residual,([x,y,0] if translation_only else [x,y,0,0,0,0]),bounds=([-3]*3,[3]*3) if translation_only else ([-3]*3+[-5]*3,[3]*3+[5]*3),loss='soft_l1',f_scale=1,diff_step=1e-4,max_nfev=150)
          for x,y in [(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)]]
    best=min(fits,key=lambda f:f.cost);xyz,R=pose(best.x)
    return {'position':xyz.tolist(),'rotation':R.tolist(),'mean_support_error_px':float(np.mean(np.abs(residual(best.x)))),
            'delta_mm_deg':best.x.tolist(),'directions_per_view':[len(d) for _,d,_ in observations],'production_enabled':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--subpixel',action='store_true');parser.add_argument('--robot-mask',action='store_true');parser.add_argument('--label',default='plug_masked_outline');args=parser.parse_args()
    points=template();data=ROOT/'.artifacts/perception_data29/qualified';rows=[]
    predictions=json.loads((ROOT/'.artifacts/perception_data29/evaluation/mouth_transfer_original_plug_multiview_val_without_truth.json').read_text())
    out=ROOT/'.artifacts/reference30'/args.label;out.mkdir(parents=True,exist_ok=True)
    for prediction in predictions:
        case=prediction['case_id'];capture=json.loads((data/'cases'/case/'training_capture.json').read_text())
        views=[{'camera':name,'image':cv2.imread(str(data/'dataset/sc_plug/images/val'/f'{case}_{name}.png')),
                'P':np.array(v['K'])@np.linalg.inv(v['T_world_from_camera'])[:3]} for name,v in capture['cameras'].items()]
        row=fit(views,points,np.array(prediction['plug_position']),np.array(prediction['plug_rotation']),subpixel=args.subpixel,robot_mask=args.robot_mask);row['case_id']=case;rows.append(row)
    (out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        if 'position' not in row:continue
        truth=np.array(json.loads((data/'cases'/row['case_id']/'scene.json').read_text())['reset']['tip_world'])
        row['diagnostic_position_mm']=float(np.linalg.norm(row['position']-truth[:3,3])*1000)
        row['diagnostic_angle_deg']=float(Rotation.from_matrix(truth[:3,:3].T@row['rotation']).magnitude()*180/np.pi)
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['case_id','failure','mean_support_error_px','diagnostic_position_mm','diagnostic_angle_deg'] if k in r} for r in rows],indent=2))


if __name__=='__main__':main()
