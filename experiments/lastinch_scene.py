"""Five-cable Phase-1 layout and controlled last-inch resets from source frames.

Native Gazebo backend uses the official URDF geometry/inertia with native joint
position controllers. It is not the AIC Cartesian admittance controller.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from .headless_assets import put
from .lastinch_geometry import (ROOT, MODELS, HOME, JOINTS, transform, parse_pose,
    pose_string, sdf_frames, robot_fk, solve_robot_pose, source_model)


def resolve_uri(text, parent):
    if text.startswith('package://'):
        package, rel = text[10:].split('/', 1)
        bases = {'aic_assets': MODELS.parent, 'ur_description': ROOT / '.artifacts/ur_description'}
        return str((bases[package] / rel).resolve())
    if text.startswith('file://'):
        return text[7:]
    if text.startswith('model://'):
        return str((MODELS / text[8:]).resolve())
    return str((parent / text).resolve())


def load_asset(directory, name, pose, static=True):
    model = copy.deepcopy(source_model(directory))
    model.set('name', name)
    for old in model.findall('pose') + model.findall('static'):
        model.remove(old)
    put(model, 'pose', pose_string(pose))
    put(model, 'static', str(static).lower())
    for parent in model.iter():
        for plugin in list(parent.findall('plugin')):
            parent.remove(plugin)  # no event trigger or automatic latch in research scene
    for uri in model.iter('uri'):
        uri.text = resolve_uri(uri.text.strip(), MODELS / directory)
    return model


def flatten_cable(name, pose, static=True):
    """Use libsdformat-expanded official cable, preserving relative frame graph."""
    model = copy.deepcopy(ET.parse(ROOT / '.artifacts/lastinch_resources/cable.sdf').getroot().find('model'))
    frames = sdf_frames(model)
    for link in model.findall('link'):
        old = link.find('pose')
        if old is not None:
            link.remove(old)
        put(link, 'pose', pose_string(frames[link.get('name')]))
        # Expanded output already has resolved absolute mesh filenames.
    if static:
        # A rest-shape reference has no articulated dynamics. DART must not
        # construct the source's massless flexible joint chain for a static image.
        for joint in model.findall('joint'):
            model.remove(joint)
        for frame in model.findall('frame'):
            model.remove(frame)
        for link in model.findall('link'):
            inertia=link.find('inertial')
            if inertia is not None and float(inertia.findtext('mass','1'))<=0:
                link.remove(inertia)
        root_link=model.find('link').get('name')
        for index,link in enumerate(model.findall('link')[1:]):
            joint=put(model,'joint',name=f'rest_shape_{index}',type='fixed')
            put(joint,'parent',root_link)
            put(joint,'child',link.get('name'))
    for plugin in model.findall('plugin'):
        model.remove(plugin)
    for uri in model.iter('uri'):
        if uri.text in ('lc_plug_visual.glb','sc_plug_visual.glb','sfp_module_visual.glb'):
            directory={'lc_plug_visual.glb':'LC Plug','sc_plug_visual.glb':'SC Plug',
                       'sfp_module_visual.glb':'SFP Module'}[uri.text]
            uri.text=str(MODELS/directory/uri.text)
    model.set('name', name)
    for old in model.findall('pose') + model.findall('static'):
        model.remove(old)
    put(model, 'pose', pose_string(pose))
    put(model, 'static', str(static).lower())
    return model, frames


def add_robot(world, robot, q, native_dynamic=False):
    frames = robot_fk(robot, q)
    model = put(world, 'model', name='ur5e')
    put(model, 'static', str(not native_dynamic).lower())
    for original in robot.findall('link'):
        name = original.get('name')
        if name in ('world', 'tabletop'):
            continue
        link = put(model, 'link', name=name)
        put(link, 'pose', pose_string(frames[name]))
        inertial = original.find('inertial')
        if inertial is not None:
            inertia = put(link, 'inertial')
            o = inertial.find('origin')
            vals = [*map(float, o.get('xyz', '0 0 0').split()), *map(float, o.get('rpy', '0 0 0').split())] if o is not None else [0]*6
            put(inertia, 'pose', ' '.join(map(str, vals)))
            put(inertia, 'mass', inertial.find('mass').get('value'))
            tensor = put(inertia, 'inertia')
            for key, value in inertial.find('inertia').attrib.items():
                put(tensor, key, value)
        elif native_dynamic:
            # Frame-only links have no independent physical role; lock to neighbors.
            inertia = put(link, 'inertial')
            put(inertia, 'mass', '0.000001')
            tensor = put(inertia, 'inertia')
            for key in ('ixx', 'iyy', 'izz'):
                put(tensor, key, '1e-9')
        for kind in ('visual', 'collision'):
            for index, original_item in enumerate(original.findall(kind)):
                item = put(link, kind, name=original_item.get('name', f'{kind}_{index}'))
                o = original_item.find('origin')
                if o is not None:
                    put(item, 'pose', o.get('xyz', '0 0 0') + ' ' + o.get('rpy', '0 0 0'))
                geometry = put(item, 'geometry')
                old_geometry = original_item.find('geometry')
                for shape in old_geometry:
                    new = put(geometry, shape.tag)
                    if shape.tag == 'mesh':
                        put(new, 'uri', resolve_uri(shape.get('filename'), ROOT))
                        put(new, 'scale', shape.get('scale', '1 1 1'))
                    else:
                        for key, value in shape.attrib.items():
                            put(new, key, value)
        # Carry exact camera sensors from the xacro's Gazebo extensions.
        for extension in robot.findall(f"gazebo[@reference='{name}']"):
            for sensor in extension.findall('sensor'):
                link.append(copy.deepcopy(sensor))
    for original in robot.findall('joint'):
        parent = original.find('parent').get('link')
        child = original.find('child').get('link')
        if parent in ('world', 'tabletop'):
            continue
        joint = put(model, 'joint', name=original.get('name'), type=original.get('type') if native_dynamic else 'fixed')
        put(joint, 'parent', parent)
        put(joint, 'child', child)
        for extension in robot.findall(f"gazebo[@reference='{original.get('name')}']"):
            for sensor in extension.findall('sensor'):
                joint.append(copy.deepcopy(sensor))
        if native_dynamic and original.get('type') != 'fixed':
            axis = put(joint, 'axis')
            put(axis, 'xyz', original.find('axis').get('xyz'))
            limit = put(axis, 'limit')
            old_limit = original.find('limit')
            offset = dict(zip(JOINTS, q)).get(original.get('name'), .00655)
            for key in ('lower', 'upper'):
                put(limit, key, float(old_limit.get(key, '0')) - offset)
            put(limit, 'effort', old_limit.get('effort', '150'))
            controller = put(model, 'plugin', filename='gz-sim-joint-position-controller-system',
                             name='gz::sim::systems::JointPositionController')
            put(controller, 'joint_name', original.get('name'))
            put(controller, 'topic', '/lastinch/joints/' + original.get('name').replace('/', '_'))
            put(controller, 'p_gain', 8000 if original.get('name') in JOINTS else 1000)
            put(controller, 'd_gain', 120 if original.get('name') in JOINTS else 10)
            put(controller, 'cmd_max', old_limit.get('effort', '150'))
            put(controller, 'cmd_min', -float(old_limit.get('effort', '150')))
    if native_dynamic:
        anchor = put(model, 'joint', name='research_base_anchor', type='fixed')
        put(anchor, 'parent', 'world')
        put(anchor, 'child', 'base_link')
        state = put(model, 'plugin', filename='gz-sim-joint-state-publisher-system',
                    name='gz::sim::systems::JointStatePublisher')
        put(state, 'topic', '/lastinch/robot/joints')
    poses = put(model, 'plugin', filename='gz-sim-pose-publisher-system', name='gz::sim::systems::PosePublisher')
    for key, value in (('publish_link_pose','true'),('publish_model_pose','true'),
                       ('use_pose_vector_msg','true'),('update_frequency','100'),('topic','/lastinch/robot/poses')):
        put(poses, key, value)
    return model, frames


def add_camera(world, name, xyz, target, fov=1.0):
    delta = np.array(target)-xyz
    T = transform([*xyz, 0, -np.arctan2(delta[2], np.linalg.norm(delta[:2])), np.arctan2(delta[1],delta[0])])
    model = put(world, 'model', name='camera_'+name)
    put(model,'static','true')
    put(model,'pose',pose_string(T))
    sensor = put(put(model,'link',name='link'),'sensor',name=name,type='camera')
    for key, val in (('always_on','true'),('update_rate','5'),('topic','/lastinch/'+name+'/image')):
        put(sensor,key,val)
    cam = put(sensor,'camera')
    put(cam,'horizontal_fov',fov)
    image=put(cam,'image')
    for key,val in (('width',1280),('height',960),('format','R8G8B8')):
        put(image,key,val)
    clip=put(cam,'clip')
    put(clip,'near',.005)
    put(clip,'far',5)


def build(out, connector='sc', target_index=2, sfp_port=0, gap_m=.01, seed=1001,
          native_dynamic=False, lateral_mm=(0,0), angular_deg=(0,0,0),cable_initialization='source',grasp_profile='legacy_sfp',grasp_jitter_seed=None):
    if grasp_profile not in ('legacy_sfp','challenge_sc'):raise ValueError('Unknown grasp profile')
    if grasp_profile=='challenge_sc' and connector!='sc':raise ValueError('Challenge SC grasp requires SC connector')
    if cable_initialization not in ('source','sc_outboard','sc_outboard_horizontal','sc_straight_outboard'):raise ValueError('Unknown cable initialization')
    if cable_initialization!='source' and connector!='sc':raise ValueError('Outboard initialization applies only to SC')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    root=ET.Element('sdf',version='1.9')
    world=put(root,'world',name='aic_lastinch')
    physics=put(world,'physics',name='physics',type='bullet-featherstone')
    put(physics,'max_step_size',.002)
    put(physics,'real_time_factor',1)
    for lib, system in (('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster'),
                         ('sensors','Sensors'),('contact','Contact'),('forcetorque','ForceTorque')):
        plugin=put(world,'plugin',filename=f'gz-sim-{lib}-system',name=f'gz::sim::systems::{system}')
        if system=='Sensors': put(plugin,'render_engine','ogre2')
        if system=='Physics': put(put(plugin,'engine'),'filename','gz-physics-bullet-featherstone-plugin')
    scene=put(world,'scene')
    put(scene,'ambient','.6 .6 .6 1')
    put(scene,'background','.18 .20 .23 1')
    light=put(world,'light',name='sun',type='directional')
    put(light,'direction','-.3 .2 -1')
    put(light,'diffuse','.85 .85 .85 1')
    board_T=transform([.15,-.2,0,0,0,np.deg2rad(rng.uniform(-3,3))])
    world.append(load_asset('Task Board Base','task_board',board_T))
    targets=[]
    # Five NIC rails, two ports/card. Exact nominal rail coordinates from xacro.
    for i,y in enumerate((-.1745,-.1345,-.0945,-.0545,-.0145)):
        T=board_T@transform([-.081418+rng.uniform(-.008,.008), y,.012,0,0,rng.uniform(-.04,.04)])
        asset=load_asset(f'NIC Card Mount {i}',f'nic_card_mount_{i}',T)
        world.append(asset)
        frames=sdf_frames(asset)
        for p in (0,1):
            targets.append(dict(connector='sfp',index=i,port=p,
                name=f'nic_card_mount_{i}/sfp_port_{p}',
                mouth=T@frames[f'sfp_port_{p}_link_entrance'],
                seat=T@frames[f'sfp_port_{p}_link']))
    # Three SC targets on rail 0, two on rail 1; distinct translations prevent overlap.
    for i,(dx,y) in enumerate(((-.042,.0295),(0,.0295),(.042,.0295),(-.025,.0705),(.025,.0705))):
        T=board_T@transform([-.075+dx+rng.uniform(-.002,.002),y,.0165,1.57,0,1.57])
        asset=load_asset(f'SC Port {i}',f'sc_port_{i}',T)
        world.append(asset)
        frames=sdf_frames(asset)
        targets.append(dict(connector='sc',index=i,port=0,name=f'sc_port_{i}',
                            mouth=T@frames['sc_port_base_link_entrance'], seat=T@frames['sc_port_base_link']))
    # Five occupied fixture pairs across the six documented legal staging seats.
    seats=np.array([-.15625,-.10625,-.05625,.05625,.10625,.15625])
    missing=int(seed%6)
    occupied=np.delete(seats,missing)
    stage_rotation=np.array([[0,2**-.5,2**-.5],[1,0,0],[0,2**-.5,-2**-.5]])
    compiled=ET.parse(ROOT/'.artifacts/lastinch_resources/cable.sdf').getroot().find('model')
    cable_frames=sdf_frames(compiled)
    staging=[]
    for i,y in enumerate(occupied):
        world.append(load_asset('SFP Mount',f'sfp_mount_{i}',board_T@transform([.055,y,.01,0,0,0])))
        world.append(load_asset('SC Mount',f'sc_mount_{i}',board_T@transform([.1,y,.012,0,0,0])))
        module_T=transform([.075,y,.028,0,0,0])
        module_T[:3,:3]=stage_rotation
        cable_T=board_T@module_T@np.linalg.inv(cable_frames['sfp_module_link'])
        staging.append(dict(index=i,y=float(y),cable_pose=pose_string(cable_T)))
        if connector=='layout' or i!=target_index:
            cable,_=flatten_cable(f'cable_{i}',cable_T)
            world.append(cable)
    robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot()
    q=HOME.copy()
    reset=None;initialization={'revision':'source_authored_rigid_reset'}
    if connector!='layout':
        target=next(t for t in targets if t['connector']==connector and t['index']==target_index and t['port']==sfp_port)
        target_tip=target['mouth'].copy()
        error=transform([lateral_mm[0]/1000,lateral_mm[1]/1000,-gap_m,*np.deg2rad(angular_deg)])
        target_tip=target['mouth']@error
        sys_path=ROOT/'aic_model'
        import sys
        sys.path.insert(0,str(sys_path))
        from aic_model.insertion.contract import SFP_TIP_IN_TCP_POS, SFP_TIP_IN_TCP_QUAT
        grasp=np.eye(4)
        grasp[:3,3]=SFP_TIP_IN_TCP_POS
        w,x,y,z=SFP_TIP_IN_TCP_QUAT
        grasp[:3,:3]=Rotation.from_quat([x,y,z,w]).as_matrix()
        grasp_provenance={'profile':'legacy_sfp','runtime_policy_input':False}
        if grasp_profile=='challenge_sc':
            from experiments.challenge_grasp import sc_challenge_grasp
            grasp,grasp_provenance=sc_challenge_grasp(robot)
        if grasp_jitter_seed is not None:
            # Dataset/development setup variation, fixed throughout each grasp.
            # The runtime policy still estimates its transform from images.
            grasp_rng=np.random.default_rng(grasp_jitter_seed)
            perturb=transform([*grasp_rng.uniform(-.002,.002,3),*grasp_rng.uniform(-.04,.04,3)])
            grasp=grasp@perturb
            grasp_provenance.update(jitter_seed=int(grasp_jitter_seed),tip_frame_perturbation=perturb.tolist())
        # SFP calibration is a reset choice, not an SC measured-grasp claim.
        # The native test policy must reestimate the grasp before any final evaluation.
        q=solve_robot_pose(robot,target_tip@np.linalg.inv(grasp))
        cable_T=target_tip@np.linalg.inv(cable_frames['sfp_tip_link' if connector=='sfp' else 'sc_tip_link'])
        cable,_=flatten_cable(f'cable_{target_index}',cable_T,static=not native_dynamic)
        if cable_initialization=='sc_straight_outboard':
            from experiments.cable_initialization import straighten_sc_free_chain
            initialization=straighten_sc_free_chain(cable,cable_T)
        elif cable_initialization.startswith('sc_outboard'):
            from experiments.cable_initialization import orient_sc_free_chain_outboard
            initialization=orient_sc_free_chain_outboard(cable,cable_T,horizontal=cable_initialization=='sc_outboard_horizontal')
        if native_dynamic:
            observer=put(cable,'plugin',filename='gz-sim-pose-publisher-system',name='gz::sim::systems::PosePublisher')
            for key,val in (('publish_link_pose','true'),('publish_model_pose','true'),('use_pose_vector_msg','true'),
                            ('update_frequency','100'),('topic','/lastinch/cable/poses')):
                put(observer,key,val)
            held=cable.find("link[@name='sc_plug_link']" if connector=='sc' else "link[@name='sfp_module_link']")
            sensor=put(held,'sensor',name='research_contact',type='contact')
            put(sensor,'always_on','true')
            put(sensor,'update_rate',100)
            put(sensor,'topic','/lastinch/contacts')
            contact=put(sensor,'contact')
            for collision in held.findall('collision'):
                put(contact,'collision',collision.get('name'))
        world.append(cable)
        reset={'target':target['name'],'connector':connector,'target_index':target_index,
               'port':sfp_port,'mouth_world':target['mouth'].tolist(),'seat_world':target['seat'].tolist(),
               'tip_world':target_tip.tolist(),'tip_in_tcp_reset':grasp.tolist(),'grasp_provenance':grasp_provenance,'gap_m':gap_m,
               'lateral_error_port_mm':list(lateral_mm),'angular_error_port_deg':list(angular_deg),
               'cable_pose':pose_string(cable_T)}
    robot_model,robot_frames=add_robot(world,robot,q,native_dynamic)
    if native_dynamic and connector!='layout':
        attachment=put(robot_model,'plugin',filename='gz-sim-detachable-joint-system',name='gz::sim::systems::DetachableJoint')
        for key,val in (('parent_link','ati/tool_link'),('child_model',f'cable_{target_index}'),
                        ('child_link','cable_connection_0' if connector=='sfp' else 'cable_connection_1')):
            put(attachment,key,val)
    add_camera(world,'overview',[.95,-1.15,.95],[.05,-.12,.22],1.05)
    add_camera(world,'board',[.55,-.78,.70],[.14,-.2,.06],.8)
    if reset:
        tip=np.array(reset['tip_world'])[:3,3]
        add_camera(world,'target',tip+np.array([-.10,.09,.12]),tip,.75)
    else:
        add_camera(world,'target',[.02,-.08,.25],[.08,-.15,.03],.9)
    ET.indent(root)
    path=out/'lastinch.sdf'
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    data={'scope':'last_inch_only','backend':'native_joint_position' if native_dynamic else 'kinematic_reset_render',
          'native_backend_is_aic_admittance':False,'seed':seed,'connector':connector,
          'counts':{'nic_cards':5,'sfp_target_ports':10,'sc_target_adapters':5,'sfp_staging_fixtures':5,
                    'sc_staging_fixtures':5,'cables':5,'ur5e_arms':1,'wrist_cameras':3},
          'staging':staging,'reset':reset,'robot_q':q.tolist(),
          'cable_initialization':initialization,
          'targets':[{**t,'mouth':t['mouth'].tolist(),'seat':t['seat'].tolist()} for t in targets],
          'robot_source':'official AIC xacro + ur_description 4.3.0',
          'automatic_latch_enabled':False,'cable_static_for_camera_check':not native_dynamic,
          'physics_engine':'bullet-featherstone','timestep_s':.002,
          'scene_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'staging_seating_validation':'reference placement derived from cable/module frames; not a pickup or seating result'}
    (out/'scene.json').write_text(json.dumps(data,indent=2)+'\n')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--connector',choices=['sc','sfp','layout'],default='sc')
    p.add_argument('--target-index',type=int,choices=range(5),default=2)
    p.add_argument('--sfp-port',type=int,choices=[0,1],default=0)
    p.add_argument('--gap-mm',type=float,default=10)
    p.add_argument('--seed',type=int,default=1001)
    p.add_argument('--native-dynamic',action='store_true')
    p.add_argument('--grasp-profile',choices=['legacy_sfp','challenge_sc'],default='legacy_sfp')
    p.add_argument('--grasp-jitter-seed',type=int)
    p.add_argument('--cable-initialization',choices=['source','sc_outboard','sc_outboard_horizontal','sc_straight_outboard'],default='source')
    p.add_argument('--case-id',help='Load a frozen case from experiments/configs/lastinch_cases.json')
    p.add_argument('--lateral-mm',type=float,nargs=2,default=[0.,0.])
    p.add_argument('--angular-deg',type=float,nargs=3,default=[0.,0.,0.])
    a=p.parse_args()
    kwargs={'lateral_mm':a.lateral_mm,'angular_deg':a.angular_deg,'cable_initialization':a.cable_initialization}
    if a.case_id:
        spec=json.loads((ROOT/'experiments/configs/lastinch_cases.json').read_text())
        row=next(r for r in spec['cases'] if r['case_id']==a.case_id)
        a.connector,a.target_index,a.sfp_port,a.gap_mm,a.seed=(row['connector'],row['target_index'],row['sfp_port'],row['gap_mm'],row['scene_seed'])
        kwargs={'lateral_mm':row['lateral_error_port_mm'],'angular_deg':row['angular_error_port_deg']}
    data=build(a.out.resolve(),a.connector,a.target_index,a.sfp_port,a.gap_mm/1000,a.seed,a.native_dynamic,grasp_profile=a.grasp_profile,grasp_jitter_seed=a.grasp_jitter_seed,**kwargs)
    print(json.dumps({'counts':data['counts'],'reset':data['reset'],'robot_q':data['robot_q']},indent=2))


if __name__=='__main__': main()
