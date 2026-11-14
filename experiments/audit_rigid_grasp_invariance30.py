"""Post-hoc actual-vs-estimated grasp drift; no policy or inference changes."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import HOME,robot_fk


def main():
    directory=ROOT/'.artifacts/reference30'
    before=directory/'live_screen_before_original30/replay_without_truth.json'
    after=directory/'live_screen_new_mouth30/replay_without_truth.json'
    b=json.loads(before.read_text());a=json.loads(after.read_text())
    if b['plug_sha256']!=a['plug_sha256']:raise ValueError('Require same frozen plug weights')
    fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),HOME)
    rows=[];world_base=np.eye(4);world_base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();world_base[:3,3]=[-.2,.2,0]
    for r0 in b['results']:
        name=r0['run']
        if 'dev_3006_' in name:continue # Its later capture is a different body-input protocol.
        r1=next(r for r in a['results'] if r['run']==name);root=ROOT/'.artifacts/lastinch_runs'/name/name
        trace=json.load(gzip.open(root/'verification_trace.json.gz','rt'));actual=[];estimate=[];skews=[];stamps=[]
        for capture,row in [('perception_before',r0),('perception_after',r1)]:
            metadata=json.loads((root/capture/'camera_geometry.json').read_text());times=[v['stamp_s'] for v in metadata]
            if max(times)-min(times)>1e-6:raise ValueError('Unsynchronized diagnostic views')
            t=max(times);near=min(trace,key=lambda x:abs(x['sim_time_s']-t));skews.append(abs(near['sim_time_s']-t));stamps.append(t)
            if skews[-1]>.01:raise ValueError('Cannot reliably join camera and independent geometry')
            reconstructed=[np.array(v['T_base_from_camera'])@np.linalg.inv(fk[v['camera']+'/optical'])@fk['gripper/tcp'] for v in metadata]
            if max(np.linalg.norm(B-reconstructed[0]) for B in reconstructed)>1e-6:raise ValueError('Inconsistent rigid camera-to-tool transforms')
            own_tcp=reconstructed[0];actual.append(np.linalg.inv(own_tcp)@np.linalg.inv(world_base)@near['tip_world'])
            measured=np.eye(4);measured[:3,:3]=row['plug_rotation'];measured[:3,3]=row['plug_position'];estimate.append(np.linalg.inv(own_tcp)@measured)
        distance=lambda x:dict(translation_mm=float(np.linalg.norm(x[0][:3,3]-x[1][:3,3])*1000),rotation_deg=float(np.rad2deg(Rotation.from_matrix(x[0][:3,:3].T@x[1][:3,:3]).magnitude())))
        rows.append(dict(run=name,image_stamps_s=stamps,independent_trace_join_skew_s=skews,estimated_grasp_change=distance(estimate),actual_grasp_change_diagnostic=distance(actual),
            interpretation='Wrist cameras and held connector are rigid to the tool; viewpoint relative to plug should remain fixed. Background, illumination, rasterization and inference can change.',
            limitation='Nearest independent trace sample; not exact runtime prime frames; this diagnostic does not identify background as the sole cause.'))
    result=dict(purpose=__doc__,inference_files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [before,after]},prediction_files_saved_before_this_truth_audit=True,simulator_truth_used_only_for_diagnostics=True,final_cases_used=False,rows=rows)
    out=ROOT/'experiments/evidence/original_repair/rigid_grasp_invariance_diagnostic.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
