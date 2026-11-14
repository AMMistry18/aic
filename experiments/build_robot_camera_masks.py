"""Project public robot tool meshes into its rigid wrist cameras.

Only robot URDF, known finger aperture and camera intrinsics are used. No
connector/port placement or simulator segmentation enters these masks.
"""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import HOME,MODELS,robot_fk,transform


def main():
    robot_path=ROOT/'.artifacts/lastinch_resources/ur5e.urdf';robot=ET.parse(robot_path).getroot();frames=robot_fk(robot,HOME)
    capture=ROOT/'.artifacts/perception_data29/qualified/cases/val_90010/training_capture.json'
    cameras=json.loads(capture.read_text())['cameras'];out=ROOT/'.artifacts/reference30/robot_masks';out.mkdir(parents=True,exist_ok=True)
    geometry=[];sources={str(robot_path.relative_to(ROOT)):hashlib.sha256(robot_path.read_bytes()).hexdigest()}
    for link in robot.findall('link'):
        if not link.get('name').startswith(('gripper/','ati/')):continue
        for visual in link.findall('visual'):
            node=visual.find('geometry/mesh')
            if node is None:continue
            file=MODELS.parent/str(node.get('filename')).replace('package://aic_assets/','')
            sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            origin=visual.find('origin');values=None if origin is None else [*map(float,origin.get('xyz','0 0 0').split()),*map(float,origin.get('rpy','0 0 0').split())]
            base=frames[link.get('name')]@transform(values)
            scene=trimesh.load_scene(file)
            for name in scene.graph.nodes_geometry:
                T,key=scene.graph[name];mesh=scene.geometry[key]
                vertices=trimesh.transform_points(mesh.vertices,base@T)
                geometry.append(vertices[mesh.faces])
    for camera,v in cameras.items():
        K=np.array(v['K']);inv=np.linalg.inv(frames[camera+'/optical']);mask=np.zeros((1024,1152),np.uint8)
        for triangles in geometry:
            points=(inv[:3,:3]@triangles.reshape(-1,3).T).T+inv[:3,3];triangles_cam=points.reshape(-1,3,3)
            for triangle in triangles_cam:
                if np.any(triangle[:,2]<=.001):continue
                uv=(K@triangle.T).T;uv=uv[:,:2]/uv[:,2:]-.5
                if np.all(uv[:,0]<0) or np.all(uv[:,0]>1152) or np.all(uv[:,1]<0) or np.all(uv[:,1]>1024):continue
                cv2.fillConvexPoly(mask,np.rint(uv).astype(np.int32),255)
        cv2.imwrite(str(out/f'{camera}.png'),mask)
        image=cv2.imread(str(ROOT/f'.artifacts/perception_data29/qualified/dataset/sc_plug/images/val/val_90010_{camera}.png'))
        overlay=image.copy();overlay[mask!=0]=(overlay[mask!=0]*.5+np.array([0,0,127])).astype(np.uint8)
        cv2.imwrite(str(out/f'{camera}_overlay.jpg'),overlay)
    (out/'provenance.json').write_text(json.dumps({'purpose':__doc__,'sources':sources,'finger_joint_positions_m':[.00655,.00655],
        'camera_tool_rigidity':'robot HOME is arbitrary; arm joint motion cancels between camera and tool',
        'calibration_source':'recorded camera K only; object/world transforms ignored','production_enabled':False},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
