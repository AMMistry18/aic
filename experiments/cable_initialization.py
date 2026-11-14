"""Setup-only reorientation at the source's unconstrained SC cable ball joint.

All meshes, collision shapes, inertias and joints are preserved. This changes
the initial free-chain configuration, never the held connector or policy input.
Headless contact/hold validation is still required after gravity settling.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.lastinch_geometry import sdf_frames,pose_string,parse_pose


def straighten_sc_free_chain(model,cable_world):
    """Unfold the source ball-joint chain outboard, preserving every anchor.

    This is a reset configuration, not a cable constraint or policy observation.
    Fixed connector assemblies retain their exact internal source transforms.
    """
    old=sdf_frames(model);edges={name:[] for name in old}
    for joint in model.findall('joint'):
        parent=joint.findtext('parent');child=joint.findtext('child')
        child_anchor=parse_pose(joint)
        parent_anchor=np.linalg.inv(old[parent])@old[child]@child_anchor
        edges[parent].append((child,joint.get('type'),parent_anchor,child_anchor))
        edges[child].append((parent,joint.get('type'),child_anchor,parent_anchor))
    root='cable_connection_1';new={root:cable_world@old[root]};queue=[root]
    direction=np.array([1.,2.,0.])/np.sqrt(5)
    while queue:
        parent=queue.pop(0)
        for child,kind,pa,ca in edges[parent]:
            if child in new:continue
            pose=new[parent]@np.linalg.inv(old[parent])@old[child]
            if kind=='ball':
                anchor=(new[parent]@pa)[:3,3]
                exits=[a[:3,3] for name,k,a,b in edges[child] if name!=parent and k=='ball']
                if exits:
                    local=exits[0]-ca[:3,3]
                    if np.linalg.norm(local)>1e-9:
                        vector=pose[:3,:3]@local
                        pose[:3,:3]=Rotation.align_vectors([direction],[vector/np.linalg.norm(vector)])[0].as_matrix()@pose[:3,:3]
                pose[:3,3]=anchor-pose[:3,:3]@ca[:3,3]
            new[child]=pose;queue.append(child)
    for link in model.findall('link'):
        previous=link.find('pose')
        if previous is not None:link.remove(previous)
        ET.SubElement(link,'pose').text=pose_string(np.linalg.inv(cable_world)@new[link.get('name')])
    return {'revision':'sc_straight_outboard_v2_open_corner','direction_world':direction.tolist(),
            'changes_geometry_or_constraints':False,'requires_runtime_hold_contact_gate':True,
            'free_endpoint_world_m':new['sfp_module_link'][:3,3].tolist()}

def orient_sc_free_chain_outboard(model,cable_world,horizontal=False):
    frames=sdf_frames(model);held={'cable_connection_1','sc_plug_link','sc_tip_link'}
    free=[link for link in model.findall('link') if link.get('name') not in held]
    anchor=(cable_world@frames['cable_connection_1'])[:3,3]
    mean=np.mean([(cable_world@frames[link.get('name')])[:3,3]-anchor for link in free],axis=0)
    # A single canonical, case-independent direction: away from the NIC rail
    # and above the board. This is an initialization, not an optimized policy.
    direction=np.array([1.,0.,0.]) if horizontal else np.array([1.,0.,1.])/np.sqrt(2)
    R=Rotation.align_vectors([direction],[mean/np.linalg.norm(mean)])[0].as_matrix()
    world_motion=np.eye(4);world_motion[:3,:3]=R;world_motion[:3,3]=anchor-R@anchor
    local_motion=np.linalg.inv(cable_world)@world_motion@cable_world
    for link in free:
        old=link.find('pose')
        if old is not None:link.remove(old)
        ET.SubElement(link,'pose').text=pose_string(local_motion@frames[link.get('name')])
    return {'revision':'sc_horizontal_free_chain_v1' if horizontal else 'sc_outboard_free_chain_v1','anchor_world_m':anchor.tolist(),
        'free_chain_world_transform':world_motion.tolist(),'changed_link_count':len(free),
        'source_ball_joint':'joint_connection_end_1','desired_mean_direction_world':direction.tolist(),
        'changes_geometry_or_constraints':False,'requires_runtime_hold_contact_gate':True}
