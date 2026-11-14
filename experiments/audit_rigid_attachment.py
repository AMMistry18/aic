"""Post-hoc endpoint check of the prescribed rigid grasp; never policy input."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import robot_fk,HOME


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    robot=ROOT/'.artifacts/lastinch_resources/ur5e.urdf'
    frames=robot_fk(ET.parse(robot).getroot(),HOME)
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
    rows=[]
    for run in a.runs:
        path=ROOT/'.artifacts/lastinch_runs'/run/run
        first=json.loads((path/'body_inputs/000/inputs.json').read_text())['views']
        last=json.loads((path/'perception_after/camera_geometry.json').read_text())
        truth=json.load(gzip.open(path/'verification_trace.json.gz','rt'));endpoints=[]
        for phase,views in [('before',first),('after',last)]:
            view=next(v for v in views if v['camera']=='left_camera');stamp=view['stamp_s']
            sample=min(truth,key=lambda x:abs(x['sim_time_s']-stamp))
            if abs(sample['sim_time_s']-stamp)>.015:raise RuntimeError('Missing synchronized truth for endpoint audit')
            base_tip=np.linalg.inv(base)@np.array(sample['tip_world'])
            tcp=(np.array(view['T_base_from_tcp']) if phase=='before' else
                 np.array(view['T_base_from_camera'])@np.linalg.inv(frames['left_camera/optical'])@frames['gripper/tcp'])
            endpoints.append({'phase':phase,'camera_stamp_s':stamp,'truth_stamp_s':sample['sim_time_s'],
                              'tcp_from_tip':(np.linalg.inv(tcp)@base_tip).tolist()})
        change=np.linalg.inv(endpoints[0]['tcp_from_tip'])@endpoints[1]['tcp_from_tip']
        rows.append({'run':run,'endpoint_translation_change_m':float(np.linalg.norm(change[:3,3])),
                     'endpoint_rotation_change_deg':float(Rotation.from_matrix(change[:3,:3]).magnitude()*180/np.pi),
                     'endpoints':endpoints})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'purpose':__doc__,'continuous_grasp_validation':False,
        'production_policy_input':False,'robot_geometry_sha256':hashlib.sha256(robot.read_bytes()).hexdigest(),'runs':rows},indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='endpoints'} for row in rows]))


if __name__=='__main__':main()
