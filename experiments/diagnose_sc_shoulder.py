"""Offline refinement of the visible SC rear shoulder, initialized by YOLO."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'aic_model')]
from experiments.lastinch_geometry import MODELS,source_model,sdf_frames
from aic_model.insertion import sc_rim_refinement as rim


def template(out):
    scene=trimesh.load_scene(MODELS/'SC Plug/sc_plug_visual.glb');triangles=[]
    inv=np.linalg.inv(sdf_frames(source_model('SC Plug'))['sc_tip_link'])
    for node in scene.graph.nodes_geometry:
        T,name=scene.graph[node]
        if name!='Cube.002':continue
        mesh=scene.geometry[name];vertices=trimesh.transform_points(mesh.vertices,T)
        front=vertices[:,0].max();faces=mesh.faces[np.min(vertices[mesh.faces,0],axis=1)>front-.00001]
        triangles.extend(trimesh.transform_points(vertices[faces].reshape(-1,3),inv).reshape(-1,3,3))
    vertices,ids=np.unique(np.round(np.array(triangles).reshape(-1,3),7),axis=0,return_inverse=True);faces=ids.reshape(-1,3)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    edges,count=np.unique(edges,axis=0,return_counts=True);segments=vertices[edges[count==1]]
    segments=segments[np.linalg.norm(segments[:,1]-segments[:,0],axis=1)>.0001]
    points=np.concatenate([np.linspace(a,b,max(3,int(np.linalg.norm(a-b)/.00025)+1)) for a,b in segments])
    path=out/'shoulder_template.npz';np.savez(path,points_m=points,segments_m=segments)
    print('shoulder segments',len(segments),'points',len(points),'bounds',points.min(0),points.max(0),flush=True)
    return path


def main():
    out=ROOT/'.artifacts/reference30/shoulder';out.mkdir(parents=True,exist_ok=True);rim.TEMPLATE=template(out)
    data=ROOT/'.artifacts/perception_data29/qualified';rows=[]
    predictions=json.loads((ROOT/'.artifacts/perception_data29/evaluation/mouth_transfer_original_plug_multiview_val_without_truth.json').read_text())
    for prediction in predictions:
        case=prediction['case_id'];capture=json.loads((data/'cases'/case/'training_capture.json').read_text());views=[];maps={}
        for name,v in capture['cameras'].items():
            image=cv2.imread(str(data/'dataset/sc_plug/images/val'/f'{case}_{name}.png'));hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            white=((hsv[:,:,1]<60)&(hsv[:,:,2]>80)).astype(np.uint8)*255
            edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),50,100)&cv2.dilate(white,np.ones((3,3),np.uint8))
            mask=cv2.imread(str(ROOT/'.artifacts/reference30/robot_masks'/f'{name}.png'),cv2.IMREAD_GRAYSCALE)
            occluded=cv2.dilate(mask,np.ones((3,3),np.uint8))|cv2.inRange(hsv,np.array([5,100,60]),np.array([35,255,255]))
            edges[occluded!=0]=0
            maps[id(image)]=(cv2.distanceTransform(255-edges,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float),occluded)
            views.append({'camera':name,'image':image,'P':np.array(v['K'])@np.linalg.inv(v['T_world_from_camera'])[:3]})
        rim.edge_distance=lambda image:maps[id(image)]
        result=rim.fit_rim(views,np.array(prediction['plug_position']),np.array(prediction['plug_rotation']),raster_offset_px=-.5,full_rotation=True)
        result.update(case_id=case,production_enabled=False);rows.append(result)
    (out/'fits_without_truth.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        if 'position' not in row:continue
        truth=np.array(json.loads((data/'cases'/row['case_id']/'scene.json').read_text())['reset']['tip_world'])
        row['diagnostic_position_mm']=float(np.linalg.norm(row['position']-truth[:3,3])*1000)
        row['diagnostic_angle_deg']=float(Rotation.from_matrix(truth[:3,:3].T@row['rotation']).magnitude()*180/np.pi)
    (out/'diagnostic_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['case_id','optimizer_success','mean_edge_distance_px','diagnostic_position_mm','diagnostic_angle_deg'] if k in r} for r in rows],indent=2))


if __name__=='__main__':main()
