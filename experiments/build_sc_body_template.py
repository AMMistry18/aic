"""Extract public white SC body hull and robot tool triangles in TCP axes."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.diagnose_sc_plug_masked_outline import template
from experiments.lastinch_geometry import HOME,MODELS,robot_fk,transform


def main():
    robot_path=ROOT/'.artifacts/lastinch_resources/ur5e.urdf'
    robot=ET.parse(robot_path).getroot();frames=robot_fk(robot,HOME)
    tcp_from_base=np.linalg.inv(frames['gripper/tcp']);geometry=[]
    sources={str(robot_path.relative_to(ROOT)):hashlib.sha256(robot_path.read_bytes()).hexdigest()}
    for link in robot.findall('link'):
        if not link.get('name').startswith(('gripper/','ati/')):continue
        for visual in link.findall('visual'):
            node=visual.find('geometry/mesh')
            if node is None:continue
            file=MODELS.parent/str(node.get('filename')).replace('package://aic_assets/','')
            sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            origin=visual.find('origin');values=None if origin is None else [*map(float,origin.get('xyz','0 0 0').split()),*map(float,origin.get('rpy','0 0 0').split())]
            base=tcp_from_base@frames[link.get('name')]@transform(values)
            scene=trimesh.load_scene(file)
            for name in scene.graph.nodes_geometry:
                T,key=scene.graph[name];mesh=scene.geometry[key]
                vertices=trimesh.transform_points(mesh.vertices,base@T)
                geometry.append(vertices[mesh.faces])
    points=template();triangles=np.concatenate(geometry)
    out=ROOT/'aic_model/aic_model/insertion/sc_body_template.npz'
    np.savez_compressed(out,body_hull_m=points,tool_triangles_tcp_m=triangles)
    for name in ['SC Plug/model.sdf','SC Plug/sc_plug_visual.glb']:
        file=MODELS/name;sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    info={'purpose':__doc__,'source_sha256':sources,'body_vertices':len(points),'tool_triangles':len(triangles),
          'finger_aperture_joint_m':[.00655,.00655],'contains_object_pose_or_grasp_constant':False,
          'robot_configuration':'arbitrary HOME cancels in TCP-relative tool geometry',
          'template_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    (ROOT/'experiments/configs/sc_body_template.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps({'body_vertices':len(points),'tool_triangles':len(triangles),'bytes':out.stat().st_size}))


if __name__=='__main__':main()
