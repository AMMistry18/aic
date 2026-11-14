"""Produce an explicit source-box interior point from a recorded capsule overlap.

The slab intersection witness is independent of the distance minimizer used
by analyze_contact_witness.py. This does not use engine depth values.
"""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir',type=Path,required=True);p.add_argument('--run',required=True)
    p.add_argument('--geometry-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit=json.loads(a.geometry_audit.read_text());worst=audit['worst_geometry_sample']
    sample=worst['worst_geometry'];cable,link=sample['capsule_link'].split('::')
    box_model,box_link,box_name=sample['source_box'].split('::')
    health=json.load(gzip.open(a.scene_dir/a.run/'contact_audit_status.json.gz','rt'))
    row=next(r for r in health if abs(r['sim_time_s']-worst['sim_s'])<1e-9)
    world=ET.parse(a.scene_dir/'world.sdf').getroot().find('world')
    cm=world.find(f"model[@name='{cable}']/link[@name='{link}']/collision")
    cap=cm.find('geometry/capsule');length=float(cap.findtext('length'));radius=float(cap.findtext('radius'))
    value=row['cable_link_world_poses'][cable+'/'+link];T=np.eye(4)
    T[:3,3]=value[:3];T[:3,:3]=Rotation.from_quat(value[3:]).as_matrix();T=T@parse_pose(cm)
    bm=world.find(f"model[@name='{box_model}']");bl=bm.find(f"link[@name='{box_link}']")
    bc=bl.find(f"collision[@name='{box_name}']");B=parse_pose(bm)@sdf_frames(bm)[box_link]@parse_pose(bc)
    half=np.array([float(x) for x in bc.findtext('geometry/box/size').split()])/2
    local=np.linalg.inv(B)@T;start=local[:3,3]-local[:3,2]*length/2;delta=local[:3,2]*length
    enter,exit=0.,1.
    for k in range(3):
        if abs(delta[k])<1e-15:
            if abs(start[k])>half[k]:raise ValueError('No source-box centerline intersection')
            continue
        low,high=sorted(((-half[k]-start[k])/delta[k],(half[k]-start[k])/delta[k]))
        enter=max(enter,low);exit=min(exit,high)
    if enter>exit:raise ValueError('No source-box centerline intersection')
    parameter=(enter+exit)/2;point=start+parameter*delta;world_point=(B@np.r_[point,1.])[:3]
    result={'purpose':'independent_slab_intersection_witness_not_engine_manifold_depth','run':a.run,
        'sim_s':row['sim_time_s'],'capsule_link':sample['capsule_link'],'source_box':sample['source_box'],
        'capsule_cylindrical_length_m':length,'capsule_radius_m':radius,
        'centerline_box_parameter_interval':[enter,exit],'interior_point_parameter':parameter,
        'interior_point_world_m':world_point.tolist(),'interior_point_box_m':point.tolist(),
        'box_half_size_m':half.tolist(),'interior_point_face_margins_m':(half-np.abs(point)).tolist(),
        'point_inside_box':bool(np.all(np.abs(point)<=half+1e-12)),
        'caveat':'A ball of capsule radius is centered on this interior point. This proves solid intersection, not the exact manifold depth or continuous contact behavior.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'world_sha256':hashlib.sha256((a.scene_dir/'world.sdf').read_bytes()).hexdigest(),
        'geometry_audit_sha256':hashlib.sha256(a.geometry_audit.read_bytes()).hexdigest()}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
