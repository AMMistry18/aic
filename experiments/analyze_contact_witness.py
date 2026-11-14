"""Independent source capsule / static NIC-and-board box overlap audit.

Checks every captured cable capsule against every source box on the static
NIC mounts and task board, including pairs absent from engine contact output.
Uses exact-stamp link poses; never reconstructs moving links from reset poses.
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
from experiments.contact_geometry import segment_box_distance


def matrix(transform):
    result=np.eye(4);p=transform['translation'];q=transform['rotation']
    result[:3,3]=[p[k] for k in ('x','y','z')]
    result[:3,:3]=Rotation.from_quat([q[k] for k in ('x','y','z','w')]).as_matrix()
    return result


def stamp(message):
    value=message['header']['stamp'];return value['sec']+value['nanosec']/1e9


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--capture',type=Path)
    source.add_argument('--recorded-run-dir',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--scene-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.recorded_run_dir:
        health=json.load(gzip.open(args.recorded_run_dir/'contact_audit_status.json.gz','rt'))
        if not health or not all('cable_link_world_poses' in row for row in health):
            raise ValueError('This trial lacks exact contact-observer link poses')
        metadata={'purpose':'read_only_evaluator_geometry_witness','run':args.recorded_run_dir.name,
            'coverage':'recorded contact-observer interval; exact PostUpdate timestamps',
            'scene_sha256':hashlib.sha256((args.scene_dir/'scene.json').read_bytes()).hexdigest(),
            'stop_reason':'completed_recorded_trial'}
        args.out=args.out or args.recorded_run_dir/'contact_geometry_witness';args.out.mkdir(parents=True,exist_ok=True)
        contact_messages=json.load(gzip.open(args.recorded_run_dir/'contacts.json.gz','rt'))
        def observed_poses():
            for row in health:
                ns=round(row['sim_time_s']*1e9);sec,ns=divmod(ns,10**9);transforms=[]
                for name,pose in row['cable_link_world_poses'].items():
                    transforms.append({'header':{'frame_id':'world','stamp':{'sec':sec,'nanosec':ns}},
                        'child_frame_id':name,'transform':{'translation':dict(zip(('x','y','z'),pose[:3])),
                        'rotation':dict(zip(('x','y','z','w'),pose[3:]))}})
                yield {'transforms':transforms}
        pose_messages=observed_poses()
    else:
        metadata=json.loads((args.capture/'capture.json').read_text());args.out=args.out or args.capture
        contact_messages=[json.loads(line) for line in gzip.open(args.capture/'contacts.jsonl.gz','rt')]
        pose_messages=(json.loads(line) for line in gzip.open(args.capture/'poses.jsonl.gz','rt'))
    if metadata.get('stop_reason') is None:raise ValueError('Capture must finish before analysis')
    if hashlib.sha256((args.scene_dir/'scene.json').read_bytes()).hexdigest()!=metadata['scene_sha256']:
        raise ValueError('Scene differs from witnessed scene')
    scene=json.loads((args.scene_dir/'scene.json').read_text());cable=f"cable_{scene['reset']['target_index']}"
    world=ET.parse(args.scene_dir/'world.sdf').getroot().find('world')
    model=world.find(f"model[@name='{cable}']");capsules=[];boxes=[]
    for link in model.findall('link'):
        for collision in link.findall('collision'):
            capsule=collision.find('geometry/capsule')
            if capsule is None:continue
            if collision.find('pose') is not None and collision.find('pose').get('relative_to'):
                raise ValueError('Explicit collision frame needs resolution')
            capsules.append((link.get('name'),parse_pose(collision),float(capsule.findtext('length')),float(capsule.findtext('radius'))))
    for model in world.findall('model'):
        if model.get('name')!='task_board' and not model.get('name').startswith('nic_card_mount_'):continue
        if model.findtext('static')!='true':raise ValueError('Reference box model is not static')
        frames=sdf_frames(model)
        for link in model.findall('link'):
            for collision in link.findall('collision'):
                box=collision.find('geometry/box')
                if box is None:continue
                if collision.find('pose') is not None and collision.find('pose').get('relative_to'):
                    raise ValueError('Explicit collision frame needs resolution')
                T=parse_pose(model)@frames[link.get('name')]@parse_pose(collision)
                half=np.array([float(x) for x in box.findtext('size').split()])/2
                boxes.append({'name':f"{model.get('name')}::{link.get('name')}::{collision.get('name')}",
                    'link':f"{model.get('name')}::{link.get('name')}",'inverse':np.linalg.inv(T),'center':T[:3,3],
                    'half':half,'bound':np.linalg.norm(half)})
    centers=np.array([b['center'] for b in boxes]);bounds=np.array([b['bound'] for b in boxes])
    reference_links={b['link'] for b in boxes};capsule_links={cable+'::'+c[0] for c in capsules}
    engine={}
    for message in contact_messages:
            max_depth=0.
            for contact in message['contacts']:
                links={contact[k]['name'].rsplit('::',1)[0] for k in ('collision1','collision2')}
                if links&capsule_links and links&reference_links:
                    max_depth=max(max_depth,max([0.]+[-float(x) for x in contact['depths']]))
            engine[round(stamp(message),6)]=max_depth
    rows=[];missing=0
    for message in pose_messages:
            t=stamp(message['transforms'][0])
            if round(t,6) not in engine:missing+=1;continue
            frames={v['child_frame_id']:matrix(v['transform']) for v in message['transforms']}
            flat_world=message['transforms'][0]['header']['frame_id']=='world'
            parent=np.eye(4) if flat_world else frames[cable];maximum=0.;worst=None
            for link,offset,length,radius in capsules:
                T=parent@frames[cable+'/'+link]@offset
                near=np.flatnonzero(np.linalg.norm(centers-T[:3,3],axis=1)<=bounds+length/2+radius)
                for i in near:
                    box=boxes[i];local=box['inverse']@T
                    p0=local[:3,3]-local[:3,2]*length/2;p1=local[:3,3]+local[:3,2]*length/2
                    distance=segment_box_distance(p0,p1,box['half']);overlap=max(0.,radius-distance)
                    if overlap>maximum:
                        maximum=overlap;worst={'capsule_link':cable+'::'+link,'source_box':box['name'],
                            'centerline_distance_m':distance,'centerline_intersects_box':distance<=1e-12}
            rows.append({'sim_s':t,'geometric_overlap_lower_bound_m':maximum,
                'engine_capsule_reference_max_penetration_m':engine[round(t,6)],'worst_geometry':worst})
    if not rows:raise ValueError('No exact-stamp pose/contact matches')
    worst=max(rows,key=lambda r:r['geometric_overlap_lower_bound_m'])
    summary={'purpose':'evaluator_only_independent_source_geometry_witness','run':metadata['run'],
        'coverage':metadata['coverage'],'first_sim_s':rows[0]['sim_s'],'last_sim_s':rows[-1]['sim_s'],
        'matched_packets':len(rows),'unmatched_pose_packets':missing,'capsule_count':len(capsules),'static_box_count':len(boxes),
        'maximum_geometric_overlap_lower_bound_m':worst['geometric_overlap_lower_bound_m'],'worst_geometry_sample':worst,
        'maximum_engine_penetration_in_same_body_scope_m':max(r['engine_capsule_reference_max_penetration_m'] for r in rows),
        'caveat':'Capsule/static NIC-and-board boxes only; overlap lower bound is not the engine manifold depth or a whole-scene validation.',
        'capsule_convention_source':'https://sdformat.org/spec/1.12/geometry/',
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'geometry_source_sha256':hashlib.sha256((Path(__file__).parent/'contact_geometry.py').read_bytes()).hexdigest()}
    with gzip.open(args.out/'geometry_trace.json.gz','wt') as file:json.dump(rows,file)
    (args.out/'geometry_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
