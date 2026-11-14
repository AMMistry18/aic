"""Bounded actual-controller hold and six-camera ROS integrity gate.

Ground-truth reset pose is allowed only for handoff/controller calibration, never
as a deployed insertion-policy observation. Outputs are not insertion results.
"""
import copy
import argparse
import json
import hashlib
import math
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from aic_control_interfaces.msg import ControllerState, MotionUpdate
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from ros_gz_interfaces.msg import Contacts
from scipy.spatial.transform import Rotation
from contact_depth import resolve_contract,penetration_m
from handoff_pose_gate import check_tip_window

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.artifacts/kilted_handoff'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibrate-load',action='store_true')
    parser.add_argument('--calibrate-reset',action='store_true',help='Record evaluator-only startup grasp diagnostic; never override prescribed grasp')
    args=parser.parse_args()
    spec=json.loads((OUT/'scene.json').read_text())
    target=np.array(spec['reset']['tip_world'])@np.linalg.inv(spec['reset']['tip_in_tcp_reset'])
    base=np.eye(4)
    base[:3,:3]=Rotation.from_euler('z',-math.pi).as_matrix()
    base[:3,3]=[-.2,.2,0]
    target=np.linalg.inv(base)@target
    rclpy.init()
    node=Node('research_controller_hold_gate')
    samples=[]; cameras={};last_cameras={}; forces=[]; cable_poses=[];contact_audit=[];contact_geometry=[]
    convention=resolve_contract(spec)['depth_convention']
    def contacts(msg):
        values=[(penetration_m(d,convention),c) for c in msg.contacts for d in c.depths]
        maximum,worst=max(values,key=lambda item:item[0]) if values else (0.,None)
        contact_geometry.append({'sim_time_s':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,
            'max_penetration_m':maximum,
            'worst_link_pair':[worst.collision1.name.rsplit('::',1)[0],worst.collision2.name.rsplit('::',1)[0]] if worst else None})
    node.create_subscription(Contacts,'/lastinch/evaluator_contacts',contacts,100)
    def physics_gate(end):
        geometry=[r for r in contact_geometry if end-.5<=r['sim_time_s']<=end+.025]
        health=[r for r in contact_audit if end-.5<=r['sim_time_s']<=end+.025]
        covered=lambda rows:bool(rows and rows[0]['sim_time_s']<=end-.47 and rows[-1]['sim_time_s']>=end-.025
            and all(b['sim_time_s']-a['sim_time_s']<=.03 for a,b in zip(rows,rows[1:])))
        joints=2 if spec['remote_endpoint_support'].startswith('prescribed') else 1
        return bool(covered(geometry) and covered(health)
            and max(r['max_penetration_m'] for r in geometry)<=.0001
            and all(r['observed_collision_components']>=spec['minimum_expected_contact_components']
                    and r['detachable_joint_components']==joints
                    and (spec.get('solver_iterations_requested') is None or r.get('solver_iterations_configured')==spec['solver_iterations_requested']) for r in health))
    node.create_subscription(String,'/lastinch/contact_audit_status',lambda m:contact_audit.append(json.loads(m.data)),100)
    def state(msg):
        p=msg.tcp_pose.position
        samples.append({'sim_time_s':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,
                        'tcp':[p.x,p.y,p.z],
                        'quaternion_xyzw':[msg.tcp_pose.orientation.x,msg.tcp_pose.orientation.y,msg.tcp_pose.orientation.z,msg.tcp_pose.orientation.w],
                        'error_mm':float(np.linalg.norm(np.array([p.x,p.y,p.z])-target[:3,3])*1000)})
    node.create_subscription(ControllerState,'/aic_controller/controller_state',state,10)
    def cable_pose(msg):
        transforms={}
        for item in msg.transforms:
            value=item.transform;matrix=np.eye(4)
            matrix[:3,3]=[value.translation.x,value.translation.y,value.translation.z]
            matrix[:3,:3]=Rotation.from_quat([value.rotation.x,value.rotation.y,value.rotation.z,value.rotation.w]).as_matrix()
            transforms[item.child_frame_id]=matrix
        name=f"cable_{spec['reset']['target_index']}"
        tip='sc_tip_link' if spec['connector']=='sc' else 'sfp_tip_link'
        if name in transforms and f'{name}/{tip}' in transforms:
            stamp=msg.transforms[0].header.stamp
            cable_poses.append((stamp.sec+stamp.nanosec/1e9,transforms[name]@transforms[f'{name}/{tip}']))
    node.create_subscription(TFMessage,f"/model/cable_{spec['reset']['target_index']}/pose",cable_pose,10)
    node.create_subscription(WrenchStamped,'/fts_broadcaster/wrench',
        lambda m:forces.append({'sim_time_s':m.header.stamp.sec+m.header.stamp.nanosec/1e9,
                               'force':[m.wrench.force.x,m.wrench.force.y,m.wrench.force.z]}),qos_profile_sensor_data)
    refs=[]
    for name in ('left_camera','center_camera','right_camera','overview','board','target'):
        def receive(msg,name=name):
            exposure=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            if name in last_cameras and exposure-last_cameras[name]['sim_time_s']<1:return
            raw=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)
            channels=4 if 'a8' in msg.encoding else 3
            pixels=raw[:,:msg.width*channels].reshape(msg.height,msg.width,channels)
            if msg.encoding.startswith('rgb'):pixels=cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR if channels==3 else cv2.COLOR_RGBA2BGRA)
            last_cameras[name]={'sim_time_s':exposure,'pixels':pixels.copy()}
            if name in cameras:return
            path=OUT/'cameras'/f'{name}.png';path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path),pixels)
            cameras[name]={'encoding':msg.encoding,'width':msg.width,'height':msg.height,
                'sim_time_s':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,'pixel_std':float(pixels.std())}
        refs.append(receive)
        topic=f'/{name}/image' if name.endswith('camera') else f'/lastinch/{name}/image'
        node.create_subscription(Image,topic,receive,qos_profile_sensor_data)
    pub=node.create_publisher(MotionUpdate,'/aic_controller/pose_commands',10)
    actual_grasp=None;calibration_stamp=None
    if args.calibrate_reset:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.05)
            # Both streams use the simulator clock. Match actual exposures,
            # not arrival order; 30 ms skew during startup can bias the grasp.
            matches={round(s['sim_time_s'],9):s for s in samples}
            matched=next(((t,p,matches[round(t,9)]) for t,p in reversed(cable_poses)
                          if round(t,9) in matches),None)
            if matched:
                calibration_stamp,tip_pose,sample=matched
                tcp=np.eye(4);tcp[:3,3]=sample['tcp']
                tcp[:3,:3]=Rotation.from_quat(sample['quaternion_xyzw']).as_matrix()
                actual_grasp=np.linalg.inv(base@tcp)@tip_pose
                # Startup pose streams can be transient even at matching stamps.
                # The rigid-grasp contract determines the reset target; truth only audits it.
                samples.clear();break
        if actual_grasp is None:raise TimeoutError('No synchronized setup-only cable/TCP sample')
    command=MotionUpdate();command.header.frame_id='base_link'
    command.pose.position.x,command.pose.position.y,command.pose.position.z=map(float,target[:3,3])
    q=Rotation.from_matrix(target[:3,:3]).as_quat()
    command.pose.orientation.x,command.pose.orientation.y,command.pose.orientation.z,command.pose.orientation.w=map(float,q)
    command.target_stiffness=np.diag([500.,500.,500.,50.,50.,50.]).ravel().tolist()
    command.target_damping=np.diag([45.,45.,45.,15.,15.,15.]).ravel().tolist()
    command.trajectory_generation_mode.mode=2
    bias=np.zeros(3); previous=None
    started=time.monotonic()
    duration=8 if args.calibrate_load else 2
    while time.monotonic()-started<240:
        if args.calibrate_load and samples and samples[-1]['sim_time_s']!=previous:
            sample=samples[-1]
            if previous is not None:
                dt=min(.02,max(0,sample['sim_time_s']-previous))
                bias=np.clip(bias+500*(target[:3,3]-np.array(sample['tcp']))*dt,-3,3)
            previous=sample['sim_time_s']
            tip_bias=target[:3,:3].T@bias
            command.feedforward_wrench_at_tip.force.x,command.feedforward_wrench_at_tip.force.y,command.feedforward_wrench_at_tip.force.z=map(float,tip_bias)
        command.header.stamp=node.get_clock().now().to_msg();pub.publish(command)
        rclpy.spin_once(node,timeout_sec=.02)
        if samples and samples[-1]['sim_time_s']-samples[0]['sim_time_s']>=duration and len(cameras)==6:
            window=[s for s in samples if s['sim_time_s']>=samples[-1]['sim_time_s']-.5]
            if max(s['error_mm'] for s in window)<.25 and physics_gate(samples[-1]['sim_time_s']) and check_tip_window(cable_poses,spec['reset']['tip_world'],samples[-1]['sim_time_s'])['passed']:break
            if samples[-1]['sim_time_s']-samples[0]['sim_time_s']>=20:break
    terminal=[s for s in samples if s['sim_time_s']>=samples[-1]['sim_time_s']-.5] if samples else []
    for name,frame in last_cameras.items():cv2.imwrite(str(OUT/'cameras'/f'{name}_latest.png'),frame['pixels'])
    maximum=max((s['error_mm'] for s in terminal),default=None)
    support_required=spec.get('remote_endpoint_support','').startswith('prescribed')
    audit_terminal=[s for s in contact_audit if samples and s['sim_time_s']>=samples[-1]['sim_time_s']-.5]
    support_ok=not support_required or bool(audit_terminal and all(s['remote_support_error_m']<.0001
        and s['detachable_joint_components']==2 and s['observed_collision_components']>=spec['minimum_expected_contact_components'] for s in audit_terminal))
    tip_gate=check_tip_window(cable_poses,spec['reset']['tip_world'],samples[-1]['sim_time_s'] if samples else 0)
    result={'tip_pose_gate':tip_gate,'reset_target_source':'prescribed_grasp_design','purpose':'environment_hold_not_insertion','samples':samples,'cameras':cameras,'forces':forces,
        'world_sha256':hashlib.sha256((OUT/'world.sdf').read_bytes()).hexdigest(),
        'last_camera_exposures_s':{name:frame['sim_time_s'] for name,frame in last_cameras.items()},
        'wall_time_s':time.monotonic()-started,'terminal_max_tcp_error_mm':maximum,
        'terminal_window_s':.5,'threshold_mm':.25,
        'gate_passed':maximum is not None and maximum<.25 and len(cameras)==6
            and samples[-1]['sim_time_s']-samples[0]['sim_time_s']>=duration and support_ok
            and physics_gate(samples[-1]['sim_time_s']) and tip_gate['passed'],
        'full_scene_contact_gate_passed':bool(samples and physics_gate(samples[-1]['sim_time_s'])),
        'setup_max_penetration_limit_m':.0001,'setup_contact_geometry':contact_geometry,
        'remote_support_gate_passed':support_ok,'contact_audit_terminal':audit_terminal,
        'target_base_link':target.tolist(),'reset_truth_used_for_hold_only':True,
        'load_bias_calibration':args.calibrate_load,'calibrated_force_base_N':bias.tolist(),
        'actual_grasp_reset_calibration':False,'startup_grasp_diagnostic_requested':args.calibrate_reset,
        'setup_tip_in_tcp_measured':actual_grasp.tolist() if actual_grasp is not None else None,
        'setup_calibration_common_sim_stamp_s':calibration_stamp,
        'duration_sim_s':samples[-1]['sim_time_s']-samples[0]['sim_time_s'] if samples else 0}
    filename='controller_hold_calibrated.json' if args.calibrate_load else 'controller_hold.json'
    (OUT/filename).write_text(json.dumps(result,indent=2)+'\n')
    if args.calibrate_load:
        # The runtime policy needs only own-robot payload support. Keep object
        # and target diagnostics in the independent setup/evaluation artifact.
        (OUT/'static_load_calibration.json').write_text(json.dumps({
            'schema':'aic_static_load_calibration_v1','frame_id':'base_link',
            'calibrated_force_base_N':bias.tolist(),
            'source':'own_TCP_hold_error_PI','object_or_target_pose_fields':False,
        },indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','forces','target_base_link')},indent=2))
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
