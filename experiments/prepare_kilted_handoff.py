"""Adapt a native dynamic QA scene to the actual ROS AIC robot/controller.

This is environment preparation, not a policy outcome. Run inside the mounted
evaluator container so absolute mesh paths resolve consistently.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import yaml
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--camera-rate',type=float,default=20)
    p.add_argument('--physics-step',type=float,default=.002)
    p.add_argument('--wrist-rate',type=float,default=100)
    p.add_argument('--solver-iterations',type=int,default=os.environ.get('AIC_RESEARCH_SOLVER_ITERATIONS'))
    a=p.parse_args()
    if not 0 <= a.wrist_rate <= 1/a.physics_step+1e-6:raise ValueError('Wrist rate must fit physics step; zero means every step')
    if a.wrist_rate>1000:raise ValueError('gz-sensors 9.2.0 truncates sub-ms periods; use zero for every physics step')
    if a.solver_iterations is not None and a.solver_iterations<=0:raise ValueError('Positive solver iteration count required')
    a.out.mkdir(parents=True,exist_ok=True)
    data=json.loads((a.scene/'scene.json').read_text())
    if data['backend']!='native_joint_position':
        raise ValueError('Input must have original dynamic active cable')
    tree=ET.parse(a.scene/'lastinch.sdf')
    world=tree.getroot().find('world')
    from experiments.runtime_contact_contract import runtime_contract
    physics_runtime=runtime_contract()
    world.find("plugin[@name='gz::sim::systems::Physics']/engine/filename").text=physics_runtime['plugin_path']
    world.remove(world.find("model[@name='ur5e']"))
    # The source ROS robot already includes ForceTorque. Each instance scans
    # every sensor globally; retaining the native world's copy creates two
    # noisy publications at every identical physics stamp.
    for duplicate in world.findall("plugin[@name='gz::sim::systems::ForceTorque']"):
        world.remove(duplicate)
    text=ET.tostring(tree.getroot(),encoding='unicode')
    text=text.replace('/mnt/c/Users/anshu/College/Intrinsic/aic/aic',str(ROOT))
    (a.out/'world.sdf').write_text(text)
    source=ROOT/'.artifacts/upstream-phase1/aic_description/urdf/ur_gz.urdf.xacro'
    text=source.read_text()
    for name,value in zip(('shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                           'wrist_1_joint','wrist_2_joint','wrist_3_joint'),data['robot_q']):
        text=re.sub(rf'{name}=[-0-9.]+',f'{name}={value:.14g}',text)
    child='cable_connection_1' if data['connector']=='sc' else 'cable_connection_0'
    plugin=f'''<gazebo>
    <plugin filename="/opt/research-grasp/libResearchGrasp.so" name="research::Grasp">
      <parent_link>ati/tool_link</parent_link><child_model>cable_{data['reset']['target_index']}</child_model>
      <child_link>{child}</child_link>
    </plugin>
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_link_pose>true</publish_link_pose><use_pose_vector_msg>true</use_pose_vector_msg>
      <update_frequency>100</update_frequency><topic>/lastinch/robot/poses</topic>
    </plugin>
    </gazebo>'''
    text=text.replace('</robot>',plugin+'\n</robot>')
    (a.out/'robot.xacro').write_text(text)
    # Compile the actual ROS-control robot before startup. A prescribed rigid
    # grasp must not also impose penetrating finger contacts as a second grasp.
    params=ROOT/'.artifacts/upstream-phase1/aic_bringup/config/aic_ros2_controllers.yaml'
    proc=subprocess.run(['xacro',str(a.out/'robot.xacro'),'name:=ur5e',
        'ur_type:=ur5e','x:=-0.2','y:=0.2','z:=0','yaw:=-3.141592653589793',
        f'simulation_controllers:={params}'],capture_output=True,text=True,check=True)
    (a.out/'robot.urdf').write_text(proc.stdout)
    compiled=subprocess.run(['gz','sdf','-p',str(a.out/'robot.urdf')],
        capture_output=True,text=True,check=True)
    (a.out/'robot_compile.log').write_text(compiled.stderr)
    robot=ET.fromstring(compiled.stdout).find('model');robot.set('name','ur5e')
    for sensor in robot.findall('.//sensor'):
        if sensor.get('type')=='camera':sensor.find('update_rate').text=str(a.camera_rate)
        if sensor.get('type')=='force_torque':sensor.find('update_rate').text=str(a.wrist_rate)
    from experiments.lastinch_geometry import robot_fk, pose_string
    fk=robot_fk(ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot(),data['robot_q'])
    plugin=robot.find("plugin[@name='research::Grasp']")
    ET.SubElement(plugin,'expected_parent_pose').text=pose_string(fk['ati/tool_link'])
    excluded=[]
    for link in robot.findall('link'):
        if link.get('name').startswith(('gripper/','ati/')):
            for collision in link.findall('collision'):
                excluded.append(f"{link.get('name')}::{collision.get('name')}")
                link.remove(collision)
    parsed=ET.fromstring((a.out/'world.sdf').read_text())
    world=parsed.find('world')
    world.find('physics/max_step_size').text=str(a.physics_step)
    if a.solver_iterations is not None:
        physics=world.find('physics');bullet=physics.find('bullet')
        if bullet is None:bullet=ET.SubElement(physics,'bullet')
        solver=bullet.find('solver')
        if solver is None:solver=ET.SubElement(bullet,'solver')
        iters=solver.find('iters')
        if iters is None:iters=ET.SubElement(solver,'iters')
        iters.text=str(a.solver_iterations)
    anchor=ET.SubElement(world,'model',name='research_startup_anchor')
    ET.SubElement(anchor,'static').text='true'
    startup_link=ET.SubElement(anchor,'link',name='link')
    def physical_anchor(link):
        # An empty visual frame is not sufficient evidence of a physics body.
        inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass').text='1'
        inertia=ET.SubElement(inertial,'inertia')
        for key in ('ixx','iyy','izz'):ET.SubElement(inertia,key).text='.001'
        collision=ET.SubElement(link,'collision',name='support_body')
        geometry=ET.SubElement(collision,'geometry');ET.SubElement(ET.SubElement(geometry,'sphere'),'radius').text='.001'
        surface=ET.SubElement(collision,'surface');contact=ET.SubElement(surface,'contact');ET.SubElement(contact,'collide_bitmask').text='0'
    physical_anchor(startup_link)
    world.append(robot)
    force_systems=world.findall(".//plugin[@name='gz::sim::systems::ForceTorque']")
    if len(force_systems)!=1:raise RuntimeError(f'Expected one ForceTorque system, got {len(force_systems)}')
    remote_support=os.environ.get('AIC_RESEARCH_REMOTE_SUPPORT','0')=='1'
    if remote_support:
        # Last-inch boundary condition: the opposite endpoint is held by an
        # external clamp. This never attaches the tested connector to a port.
        support=ET.SubElement(world,'model',name='research_remote_cable_support')
        ET.SubElement(support,'static').text='true'
        support_link=ET.SubElement(support,'link',name='link')
        active_model=world.find(f"model[@name='cable_{data['reset']['target_index']}']")
        from experiments.cable_boundary import park_remote
        data['remote_cable_initial_configuration']=park_remote(active_model,data['connector'])
        remote='cable_connection_0' if data['connector']=='sc' else 'cable_connection_1'
        from experiments.lastinch_geometry import parse_pose
        remote_pose=parse_pose(active_model)@parse_pose(active_model.find(f"link[@name='{remote}']"))
        ET.SubElement(support_link,'pose').text=pose_string(remote_pose)
        physical_anchor(support_link)
        visual=ET.SubElement(support_link,'visual',name='remote_clamp')
        geometry=ET.SubElement(visual,'geometry');ET.SubElement(ET.SubElement(geometry,'box'),'size').text='.016 .016 .016'
        material=ET.SubElement(visual,'material');ET.SubElement(material,'diffuse').text='.4 .6 .65 1'
        ET.SubElement(robot.find("plugin[@name='research::Grasp']"),'remote_child_link').text=remote
    active_model=world.find(f"model[@name='cable_{data['reset']['target_index']}']")
    remote='cable_connection_0' if data['connector']=='sc' else 'cable_connection_1'
    from experiments.lastinch_geometry import parse_pose
    observer=ET.SubElement(world,'plugin',filename='/opt/research-grasp/libResearchContactAudit.so',name='research::ContactAudit')
    ET.SubElement(observer,'cable_model').text=active_model.get('name')
    ET.SubElement(observer,'remote_link').text=remote
    ET.SubElement(observer,'remote_pose').text=pose_string(parse_pose(active_model)@parse_pose(active_model.find(f"link[@name='{remote}']")))
    contact_bridges=[]
    contact_topics=[]
    active=f"cable_{data['reset']['target_index']}"
    # A wrist impulse need not originate at the held plug. Observe all active
    # cable and robot collision bodies, including the remote connector.
    for model in (world.find(f"model[@name='{active}']"),robot):
        for link in model.findall('link'):
            collisions=link.findall('collision')
            if not collisions:continue
            sensor=ET.SubElement(link,'sensor',name='research_all_contact',type='contact')
            ET.SubElement(sensor,'always_on').text='true'
            ET.SubElement(sensor,'update_rate').text='100'
            contact=ET.SubElement(sensor,'contact')
            for collision in collisions:ET.SubElement(contact,'collision').text=collision.get('name')
            topic=f"/world/aic_lastinch/model/{model.get('name')}/link/{link.get('name')}/sensor/research_all_contact/contact"
            ros_topic=f"/research/contacts/{model.get('name')}/{link.get('name')}"
            contact_topics.append(ros_topic)
            contact_bridges.append({'ros_topic_name':ros_topic,'gz_topic_name':topic,
                'ros_type_name':'ros_gz_interfaces/msg/Contacts','gz_type_name':'gz.msgs.Contacts','direction':'GZ_TO_ROS'})
    ET.ElementTree(parsed).write(a.out/'world.sdf',encoding='utf-8',xml_declaration=True)
    data['backend']='aic_ros2_control_pending_hold_validation'
    data['grasp_model']='prescribed rigid attachment; gripper/ATI collision geometry excluded, visuals and inertia retained'
    data['excluded_gripper_ati_collision_geometry']=excluded
    data['connector_and_port_collision_geometry']='preserved from pinned sources'
    data['native_source_scene_sha256']=data.pop('scene_sha256')
    data['world_sha256']=hashlib.sha256((a.out/'world.sdf').read_bytes()).hexdigest()
    data['robot_urdf_sha256']=hashlib.sha256((a.out/'robot.urdf').read_bytes()).hexdigest()
    data['research_grasp_source_sha256']=hashlib.sha256((ROOT/'experiments/gazebo/ResearchGrasp.cc').read_bytes()).hexdigest()
    data['wrist_camera_rate_hz']=a.camera_rate
    data['wrist_force_sensor_rate_hz']=a.wrist_rate
    data['wrist_force_expected_rate_hz']=a.wrist_rate or 1/a.physics_step
    data['high_rate_wrist_audit_required']=a.wrist_rate==0 or a.wrist_rate>100
    data['force_torque_system_instances']=len(force_systems)
    sensor_lib=Path('/ws_aic/install/lib/libgz-sensors9.so.9')
    data['wrist_sensor_library_sha256']=hashlib.sha256(sensor_lib.read_bytes()).hexdigest()
    data['runtime_libc_package']=subprocess.run(['dpkg-query','-W','libc6'],capture_output=True,text=True,check=True).stdout.strip()
    data['simulator_seed']=data['seed']
    data['timestep_s']=a.physics_step
    data['solver_iterations_requested']=a.solver_iterations
    data['remote_endpoint_support']='prescribed static clamp on opposite cable connection; full articulated cable retained' if remote_support else 'unconstrained'
    data['contact_diagnostic_ros_topics']=contact_topics
    data['contact_audit_source']='read_only_physics_ecm_with_empty_contact_heartbeat'
    from experiments.runtime_contact_contract import runtime_contract
    data['physics_contact_contract']=physics_runtime
    data['minimum_expected_contact_components']=sum(len(link.findall('collision')) for model in (active_model,robot) for link in model.findall('link'))
    (a.out/'contact_bridge.yaml').write_text(yaml.safe_dump(contact_bridges))
    bridge=yaml.safe_load((ROOT/'.artifacts/upstream-phase1/aic_bringup/config/ros_gz_bridge_config.yaml').read_text())
    bridge.append({'ros_topic_name':'/lastinch/wrist_raw','gz_topic_name':'/axia80_m20/wrench',
        'ros_type_name':'geometry_msgs/msg/WrenchStamped','gz_type_name':'gz.msgs.Wrench',
        'direction':'GZ_TO_ROS','publisher_queue':1000,'subscriber_queue':1000})
    bridge.append({'ros_topic_name':'/lastinch/grasp_state','gz_topic_name':'/lastinch/grasp_state',
                   'ros_type_name':'std_msgs/msg/String','gz_type_name':'gz.msgs.StringMsg','direction':'GZ_TO_ROS'})
    for topic,ros_type,gz_type in (('/lastinch/evaluator_contacts','ros_gz_interfaces/msg/Contacts','gz.msgs.Contacts'),
            ('/lastinch/contact_audit_status','std_msgs/msg/String','gz.msgs.StringMsg')):
        bridge.append({'ros_topic_name':topic,'gz_topic_name':topic,'ros_type_name':ros_type,'gz_type_name':gz_type,'direction':'GZ_TO_ROS'})
    (a.out/'bridge.yaml').write_text(yaml.safe_dump(bridge))
    (a.out/'scene.json').write_text(json.dumps(data,indent=2)+'\n')
    for filename in ('controller_bootstrap.json','controller_hold_calibrated.json'):
        (a.out/filename).write_text(json.dumps({'ready':False,'gate_passed':False,
            'reason':'fresh world prepared; runtime acceptance pending','world_sha256':data['world_sha256']},indent=2)+'\n')
    print(a.out)


if __name__=='__main__': main()
