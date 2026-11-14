"""Static source-lighting diagnostic; no policy or physical-trial mutation.

The pinned upstream world enables sensor global illumination and enclosure
lighting. Compare those features against the simplified research illumination
without changing connector meshes/materials or their camera-relative poses.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_scene import load_asset
from experiments.lastinch_geometry import parse_pose,pose_string


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gi',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT/'.artifacts/sfp_visual33/lighting_aligned');a=p.parse_args()
    original=ROOT/'.artifacts/sfp_visual33/data/cases/train_81000'
    out=a.root/'cases'/a.name
    if out.exists():raise RuntimeError('Preserve earlier lighting diagnostic')
    out.mkdir(parents=True)
    tree=ET.parse(original/'training_world.sdf');world=tree.getroot().find('world')
    source_path=ROOT/'.artifacts/upstream-phase1/aic_description/world/aic.sdf'
    upstream=ET.parse(source_path).getroot().find('world')
    for light in list(world.findall('light')):world.remove(light)
    # The upstream task_board xacro documents its source origin (.25,0,1.14).
    # This research world lowers that board to (.15,-.2,0). Move the enclosure
    # and its lamps by the same offset; otherwise the source table hides it.
    environment=np.eye(4);environment[:3,3]=[-.1,-.2,-1.14]
    for light in upstream.findall('light'):
        moved=copy.deepcopy(light);pose=moved.find('pose')
        if pose is None:pose=ET.SubElement(moved,'pose')
        pose.text=pose_string(environment@parse_pose(light));world.append(moved)
    world.find('scene/ambient').text='0 0 0 1'
    for directory,name in [('Enclosure','source_enclosure'),('Enclosure Walls','source_walls'),('Floor','source_floor')]:
        world.append(load_asset(directory,name,environment))
    sensor=world.find("plugin[@name='gz::sim::systems::Sensors']")
    if a.gi:
        gi=copy.deepcopy(upstream.find("plugin[@name='gz::sim::systems::Sensors']/global_illumination"))
        gi.find('resolution').text='128 128 128'  # explicitly recorded reduced-cost diagnostic
        sensor.append(gi)
    tree.write(out/'lastinch.sdf',encoding='utf-8',xml_declaration=True)
    spec=json.loads((original/'scene.json').read_text())
    spec['purpose']='Static source lighting diagnostic, not training or insertion'
    spec['source_lighting_diagnostic']=dict(sensor_global_illumination=a.gi,resolution='128 128 128' if a.gi else None,
        upstream_resolution='256 256 256',source_world_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        baseline_capture=str(original.relative_to(ROOT)),connector_geometry_or_material_changed=False,
        environment_translation_m=environment[:3,3].tolist(),source_board_origin_m=[.25,0,1.14],
        research_nominal_board_origin_m=[.15,-.2,0],
        policy_or_collision_threshold_changed=False)
    (out/'scene.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(str(out),flush=True)


if __name__=='__main__':main()
