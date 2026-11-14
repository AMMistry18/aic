"""Configure a source ball joint to park the remote connector clear of clutter.

Rigidly rotate the cable side of the endpoint ball joint, leaving the grasped
connector and all source geometry unchanged. This preserves joint anchors;
it is an initial cable configuration, not a deformation or a motion command.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.lastinch_geometry import parse_pose,pose_string

def park_remote(model,connector):
    held='cable_connection_1' if connector=='sc' else 'cable_connection_0'
    held_names={held,'sc_plug_link','sc_tip_link'} if connector=='sc' else {held,'lc_plug_link','sfp_module_link','sfp_tip_link'}
    remote='cable_connection_0' if connector=='sc' else 'cable_connection_1'
    world=parse_pose(model);frames={l.get('name'):world@parse_pose(l) for l in model.findall('link')}
    pivot=frames[held][:3,3];vector=frames[remote][:3,3]-pivot
    desired=np.array([1.,0.,1.]);desired/=np.linalg.norm(desired)
    direction=vector/np.linalg.norm(vector)
    cross=np.cross(direction,desired);angle=np.arctan2(np.linalg.norm(cross),direction@desired)
    align=Rotation.from_rotvec(cross/max(np.linalg.norm(cross),1e-12)*angle).as_matrix()
    selected=[name for name in frames if name not in held_names]
    points=np.array([frames[name][:3,3] for name in selected]);best=None
    for twist in np.linspace(0,2*np.pi,72,endpoint=False):
        R=Rotation.from_rotvec(desired*twist).as_matrix()@align
        changed=(R@(points-pivot).T).T+pivot
        # Keep the loop above the table and prefer its bulk outside the NIC bank.
        score=5*float(changed[:,2].min())+.2*float(changed[:,0].mean())
        if best is None or score>best[0]:best=(score,R,changed)
    R=best[1];W=np.eye(4);W[:3,:3]=R;W[:3,3]=pivot-R@pivot
    for link in model.findall('link'):
        if link.get('name') in held_names:continue
        link.find('pose').text=pose_string(np.linalg.inv(world)@W@frames[link.get('name')])
    new_remote=W@frames[remote]
    return {'joint':'joint_connection_end_1' if connector=='sc' else 'joint_connection_end_0',
        'held_links_unchanged':sorted(held_names),'world_rotation':R.tolist(),
        'remote_connection_world':new_remote.tolist(),'minimum_reconfigured_link_center_z_m':float(best[2][:,2].min()),
        'endpoint_ball_joint_anchor_error_m':float(np.linalg.norm((W@np.r_[pivot,1])[:3]-pivot))}
