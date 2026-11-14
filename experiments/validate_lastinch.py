"""Validate actual generated scene structure, source frames, IK and camera output."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .lastinch_geometry import ROOT, transform, parse_pose, sdf_frames, robot_fk


def validate(directory):
    data=json.loads((directory/'scene.json').read_text())
    root=ET.parse(directory/'lastinch.sdf').getroot()
    models={m.get('name'):m for m in root.find('world').findall('model')}
    counts={
        'nic_cards':sum(bool(re.fullmatch(r'nic_card_mount_[0-4]',name)) for name in models),
        'sc_target_adapters':sum(bool(re.fullmatch(r'sc_port_[0-4]',name)) for name in models),
        'sfp_staging_fixtures':sum(bool(re.fullmatch(r'sfp_mount_[0-4]',name)) for name in models),
        'sc_staging_fixtures':sum(bool(re.fullmatch(r'sc_mount_[0-4]',name)) for name in models),
        'cables':sum(bool(re.fullmatch(r'cable_[0-4]',name)) for name in models),
        'ur5e_arms':int('ur5e' in models),
        'wrist_cameras':sum(s.get('name') in ('left_camera','center_camera','right_camera') for s in models['ur5e'].iter('sensor'))}
    assert all(counts[name]==data['counts'][name] for name in counts),counts
    assert len(data['targets'])==15
    assert len({r['name'] for r in data['targets']})==15
    reset=data['reset']
    if reset:
        cable=models[f"cable_{reset['target_index']}"]
        frames=sdf_frames(cable)
        tip_name='sc_tip_link' if reset['connector']=='sc' else 'sfp_tip_link'
        actual=parse_pose(cable)@frames[tip_name]
        requested=np.array(reset['tip_world'])
        assert np.allclose(actual,requested,atol=1e-9),'Cable endpoint transform mismatch'
        mouth=np.array(reset['mouth_world'])
        local=np.linalg.inv(mouth)@actual
        lateral=np.array(reset.get('lateral_error_port_mm',[0,0]))/1000
        assert np.allclose(local[:3,3],[*lateral,-reset['gap_m']],atol=1e-9),'Handoff offset differs from frozen case'
        robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot()
        fk=robot_fk(robot,data['robot_q'])
        assert np.allclose(fk['gripper/tcp']@np.array(reset['tip_in_tcp_reset']),actual,atol=1e-5),'IK reset mismatch'
    observed_path=directory/'observed_scene.json'
    if observed_path.exists():
        observed={m['name'] for m in json.loads(observed_path.read_text())['model']}
        assert models.keys()<=observed,'Missing models in running Gazebo scene'
    cameras={}
    if (directory/'capture.json').exists():
        capture=json.loads((directory/'capture.json').read_text())
        for name,frames in capture['camera_frames'].items():
            assert len(frames)>=3, f'Missing camera data: {name}'
            pixels=np.asarray(Image.open(directory/frames[-1]['file']))
            assert np.std(pixels)>10, f'Blank image: {name}'
            cameras[name]={'frames':len(frames),'shape':list(pixels.shape)}
    result={'counts_from_scene':counts,'all_15_targets_defined':True,'frame_and_ik_checks':'passed',
            'observed_scene_checked':observed_path.exists(),'cameras':cameras,
            'controller_evaluation_ready':False}
    log=(directory/'gazebo.log').read_text(errors='replace') if (directory/'gazebo.log').exists() else ''
    result['simulator_errors_present']='[Err]' in log
    result['disabled_physics_links']='Disabling link:' in log
    if data['backend']=='native_joint_position' and (directory/'robot_poses.json').exists():
        samples=json.loads((directory/'robot_poses.json').read_text())
        tcp=[p for v in samples for p in v.get('pose',[]) if p.get('name')=='ur5e::gripper/tcp']
        desired=fk['gripper/tcp'][:3,3]
        error=[np.linalg.norm(np.array([p['position'].get(k,0) for k in ('x','y','z')])-desired)*1000 for p in tcp]
        result['native_hold_max_tcp_error_mm']=max(error) if error else None
        result['native_hold_gate_passed']=bool(error and max(error)<.25)
    (directory/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(validate(a.scene.resolve()),indent=2))


if __name__=='__main__': main()
