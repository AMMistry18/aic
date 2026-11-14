"""Evaluator-only interior cable-joint integrity from observed world poses."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.lastinch_geometry import parse_pose,sdf_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-dir',type=Path,required=True)
    p.add_argument('--run',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    scene=json.loads((a.scene_dir/'scene.json').read_text());name=f"cable_{scene['reset']['target_index']}"
    model=ET.parse(a.scene_dir/'world.sdf').getroot().find(f"world/model[@name='{name}']");frames=sdf_frames(model)
    rows=json.load(gzip.open(a.scene_dir/a.run/'contact_audit_status.json.gz','rt'))
    names=sorted(rows[0]['cable_link_world_poses']);indices={n.rsplit('/',1)[1]:i for i,n in enumerate(names)}
    values=np.array([[row['cable_link_world_poses'][n] for n in names] for row in rows])
    if not np.isfinite(values).all():raise ValueError('Nonfinite observed link pose')
    rotations=Rotation.from_quat(values[:,:,3:].reshape(-1,4)).as_matrix().reshape(len(rows),len(names),3,3)
    parent_indices=[];child_indices=[];parent_local=[];child_local=[];joints=[]
    for joint in model.findall('joint'):
        parent=joint.findtext('parent');child=joint.findtext('child')
        if parent not in indices or child not in indices:continue
        pose=joint.find('pose');relative=pose.get('relative_to') if pose is not None else None
        # SDF joint pose defaults to the child frame, unlike link poses.
        anchor=frames[relative or child]@parse_pose(joint)
        parent_indices.append(indices[parent]);child_indices.append(indices[child]);joints.append(joint.get('name'))
        parent_local.append((np.linalg.inv(frames[parent])@anchor)[:3,3]);child_local.append((np.linalg.inv(frames[child])@anchor)[:3,3])
    if not joints:raise ValueError('No observed source joint endpoints')
    pa=np.einsum('nkij,kj->nki',rotations[:,parent_indices],parent_local)+values[:,parent_indices,:3]
    ca=np.einsum('nkij,kj->nki',rotations[:,child_indices],child_local)+values[:,child_indices,:3]
    error=np.linalg.norm(pa-ca,axis=2)*1000;sample,j=np.unravel_index(error.argmax(),error.shape)
    result={'purpose':'evaluator_only_interior_chain_anchor_integrity','run':a.run,'samples':len(rows),
        'observed_joint_count':len(joints),'observed_joints':joints,
        'anchor_error_mm_median_p95_max':np.percentile(error,[50,95,100]).tolist(),
        'worst':{'sim_s':rows[sample]['sim_time_s'],'joint':joints[j],'anchor_separation_mm':float(error[sample,j])},
        'maximum_observed_quaternion_norm_error':float(np.max(np.abs(np.linalg.norm(values[:,:,3:],axis=2)-1))),
        'scope':'Only joints with both endpoints in the recorded 21 chain/end-link poses; connector attachment joints not covered.',
        'joint_pose_convention_source':'https://sdformat.org/spec/1.12/joint/',
        'world_sha256':hashlib.sha256((a.scene_dir/'world.sdf').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
