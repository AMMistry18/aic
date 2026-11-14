"""Offline SC ferrule-edge pose refinement, initialized only by learned poses."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import MODELS,source_model,sdf_frames
from experiments.diagnose_sc_rim import project


def template():
    scene=trimesh.load_scene(MODELS/'SC Plug/sc_plug_visual.glb')
    inv=np.linalg.inv(sdf_frames(source_model('SC Plug'))['sc_tip_link']);segments=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        if not name.lower().startswith('ferula'):continue
        vertices=trimesh.transform_points(scene.geometry[name].vertices,transform)
        front=vertices[vertices[:,0]>vertices[:,0].max()-.0000001]
        front=np.unique(np.round(front,8),axis=0)
        hull=ConvexHull(front[:,1:]);ring=front[hull.vertices]
        ring=trimesh.transform_points(ring,inv)
        segments.extend(np.stack([ring,np.roll(ring,-1,axis=0)],axis=1))
    segments=np.array(segments)
    return np.concatenate([np.linspace(a,b,4) for a,b in segments]),segments


def fit(views,points,position,rotation):
    fields=[]
    for view in views:
        gray=cv2.cvtColor(view['image'],cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(gray,50,100)
        fields.append(cv2.distanceTransform(255-edges,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float))
    def pose(p):return position+p[:3]*.001,Rotation.from_rotvec(p[3:]*np.pi/180).as_matrix()@rotation
    def residual(p):
        xyz,R=pose(p);world=(R@points.T).T+xyz
        return np.concatenate([np.minimum(map_coordinates(field,(project(v['P'],world)-.5)[:,::-1].T,
                                                          order=1,mode='constant',cval=10),6) for v,field in zip(views,fields)])
    fits=[]
    for x,y in [(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)]:
        fit=least_squares(residual,[x,y,0,0,0,0],bounds=([-2]*3+[-5]*3,[2]*3+[5]*3),loss='soft_l1',f_scale=.5,diff_step=1e-4,max_nfev=150)
        fits.append(fit)
    best=min(fits,key=lambda r:r.cost);xyz,R=pose(best.x)
    return {'position':xyz.tolist(),'rotation':R.tolist(),'delta_mm_deg':best.x.tolist(),
            'mean_edge_error_px':float(np.mean(residual(best.x))),'production_enabled':False}


def main():
    points,segments=template();data=ROOT/'.artifacts/perception_data29/qualified'
    predictions=json.loads((ROOT/'.artifacts/perception_data29/evaluation/mouth_transfer_original_plug_multiview_val_without_truth.json').read_text())
    out=ROOT/'.artifacts/reference30/ferrules';out.mkdir(parents=True,exist_ok=True);rows=[]
    for prediction in predictions:
        if 'plug_position' not in prediction:continue
        case=prediction['case_id'];capture=json.loads((data/'cases'/case/'training_capture.json').read_text())
        views=[{'camera':name,'image':cv2.imread(str(data/'dataset/sc_plug/images/val'/f'{case}_{name}.png')),
                'P':np.array(v['K'])@np.linalg.inv(v['T_world_from_camera'])[:3]} for name,v in capture['cameras'].items()]
        result=fit(views,points,np.array(prediction['plug_position']),np.array(prediction['plug_rotation']));result['case_id']=case
        rows.append(result)
        for v in views:
            world=(np.array(result['rotation'])@segments.reshape(-1,3).T).T+result['position']
            image=v['image'].copy()
            for edge in project(v['P'],world).reshape(-1,2,2):cv2.polylines(image,[np.rint(edge-.5).astype(np.int32)],False,(0,255,255),1)
            cv2.imwrite(str(out/f'{case}_{v["camera"]}.png'),image)
    (out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        truth=np.array(json.loads((data/'cases'/row['case_id']/'scene.json').read_text())['reset']['tip_world'])
        row['diagnostic_position_mm']=float(np.linalg.norm(row['position']-truth[:3,3])*1000)
        row['diagnostic_angle_deg']=float(Rotation.from_matrix(truth[:3,:3].T@row['rotation']).magnitude()*180/np.pi)
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['case_id','mean_edge_error_px','diagnostic_position_mm','diagnostic_angle_deg']} for r in rows],indent=2))


if __name__=='__main__':main()
