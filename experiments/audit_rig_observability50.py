"""Audit camera-to-tool invariance from the actual robot URDF, no object truth."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import robot_fk,HOME


def main():
    path=ROOT/'.artifacts/lastinch_resources/ur5e.urdf';robot=ET.parse(path).getroot()
    rng=np.random.default_rng(50001);qs=[HOME]+[HOME+rng.uniform(-.3,.3,6) for _ in range(20)]
    references={};rows=[]
    for camera in ['left_camera','center_camera','right_camera']:
        transforms=[]
        for q in qs:
            frames=robot_fk(robot,q);transforms.append(np.linalg.inv(frames['gripper/tcp'])@frames[camera+'/optical'])
        references[camera]=transforms[0].tolist()
        rows.append(dict(camera=camera,joint_configurations=len(qs),
            maximum_tcp_relative_position_change_m=max(float(np.linalg.norm(T[:3,3]-transforms[0][:3,3])) for T in transforms),
            maximum_tcp_relative_rotation_matrix_change=max(float(np.max(abs(T[:3,:3]-transforms[0][:3,:3]))) for T in transforms)))
    result=dict(scope='Actual robot kinematics/observability audit; no object pose or grasp truth read',
        robot_urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,tcp_from_camera=references,
        inference='For a rigid grasp G, camera_from_plug = inverse(tcp_from_camera) @ G. Moving the common wrist cannot change gripper self-occlusion of the plug; fixture/background occlusion and lighting may change.',
        implication='Angular recovery needs additional measurement information, such as a genuinely different fixed view or qualified contact constraints; repeating wrist frames alone cannot remove systematic hidden-corner bias.')
    out=ROOT/'experiments/angular_development50';out.mkdir(exist_ok=True)
    (out/'rig_observability.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(rows=rows,inference=result['inference'])))


if __name__=='__main__':main()
