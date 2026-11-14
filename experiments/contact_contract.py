"""Evaluator-only source collision contract and clearance checks."""
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from experiments.lastinch_geometry import parse_pose,sdf_frames
from experiments.audit_sc_collision_fit import boxes,penetration

def contract(scene_dir):
    spec=json.loads((scene_dir/'scene.json').read_text());world=ET.parse(scene_dir/'world.sdf').getroot().find('world')
    active=next(m for m in world.findall('model') if m.get('name')==f"cable_{spec['reset']['target_index']}")
    target=next(m for m in world.findall('model') if m.get('name')==spec['reset']['target'].split('/')[0])
    frames=sdf_frames(active);tip=frames['sfp_tip_link'];rows=[]
    for model,links in ((active,['sfp_module_link','sfp_tip_link','lc_plug_link']),(target,['nic_card_link'])):
        for link in model.findall('link'):
            if link.get('name') not in links:continue
            prefix=np.linalg.inv(tip)@frames[link.get('name')] if model is active else np.eye(4)
            for collision in link.findall('collision'):
                rows.append([link.get('name'),collision.get('name'),np.round(prefix@parse_pose(collision),8).tolist(),
                    ET.tostring(collision.find('geometry'),encoding='unicode').strip()])
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return spec,active,target,digest

def terminal_box_clearance(scene_dir,truth_rows):
    spec,active,target,digest=contract(scene_dir);frames=sdf_frames(active);tf=sdf_frames(target)
    target_boxes=[]
    for link in target.findall('link'):target_boxes+=boxes(link,parse_pose(target)@tf[link.get('name')])
    maximum=0.
    for row in truth_rows:
        tip=np.array(row['tip_world']);active_boxes=[]
        for link in active.findall('link'):
            if link.get('name') not in ('sfp_module_link','sfp_tip_link'):continue
            active_boxes+=boxes(link,tip@np.linalg.inv(frames['sfp_tip_link'])@frames[link.get('name')])
        maximum=max([maximum]+[max(0.,penetration(a,b)) for a in active_boxes for b in target_boxes])
    return maximum,digest
