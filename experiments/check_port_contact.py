"""Evaluator-only deliberate lateral miss: validate real port contact, then retract.

This calibration uses source target axes and is not a perception-policy trial.
Never interpret a contact trigger or this scripted motion as insertion success.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from aic_control_interfaces.msg import ControllerState, MotionUpdate
from geometry_msgs.msg import WrenchStamped
from ros_gz_interfaces.msg import Contacts
from rosidl_runtime_py.convert import message_to_ordereddict

OUT=Path('/research/.artifacts/kilted_handoff')


def main():
    hold=json.loads((OUT/'controller_hold_calibrated.json').read_text())
    identity=hashlib.sha256((OUT/'world.sdf').read_bytes()).hexdigest()
    if not hold['gate_passed'] or hold.get('world_sha256')!=identity:raise RuntimeError('Current hold gate required')
    spec=json.loads((OUT/'scene.json').read_text())
    rclpy.init();node=Node('evaluator_deliberate_contact_probe')
    states=[];forces=[];contacts=[];trace=[]
    node.create_subscription(ControllerState,'/aic_controller/controller_state',
        lambda m:states.append((m.header.stamp.sec+m.header.stamp.nanosec/1e9,m)),10)
    node.create_subscription(WrenchStamped,'/fts_broadcaster/wrench',
        lambda m:forces.append(np.array([m.wrench.force.x,m.wrench.force.y,m.wrench.force.z])),qos_profile_sensor_data)
    node.create_subscription(Contacts,'/research/contact_probe',lambda m:contacts.append(message_to_ordereddict(m)),10)
    publisher=node.create_publisher(MotionUpdate,'/aic_controller/pose_commands',10)
    target=np.array(hold['target_base_link']);start_position=target[:3,3].copy()
    world_from_base=Rotation.from_euler('z',-np.pi).as_matrix()
    mouth=np.array(spec['reset']['mouth_world'])
    lateral=world_from_base.T@mouth[:3,0]
    axial=world_from_base.T@mouth[:3,2]
    command=MotionUpdate();command.header.frame_id='base_link'
    q=Rotation.from_matrix(target[:3,:3]).as_quat()
    command.pose.orientation.x,command.pose.orientation.y,command.pose.orientation.z,command.pose.orientation.w=map(float,q)
    command.target_stiffness=np.diag([500.,500.,500.,50.,50.,50.]).ravel().tolist()
    command.target_damping=np.diag([45.,45.,45.,15.,15.,15.]).ravel().tolist()
    tip_bias=target[:3,:3].T@hold['calibrated_force_base_N']
    command.feedforward_wrench_at_tip.force.x,command.feedforward_wrench_at_tip.force.y,command.feedforward_wrench_at_tip.force.z=map(float,tip_bias)
    command.trajectory_generation_mode.mode=2
    def send(position):
        command.pose.position.x,command.pose.position.y,command.pose.position.z=map(float,position)
        command.header.stamp=node.get_clock().now().to_msg();publisher.publish(command)
    started=time.monotonic()
    while (not states or len(forces)<10) and time.monotonic()-started<15:
        send(start_position);rclpy.spin_once(node,timeout_sec=.03)
    if not states or len(forces)<10:raise TimeoutError('Missing live state or wrench')
    baseline=np.median(forces[-10:],axis=0);t0=states[-1][0]
    reason='travel_limit';peak=0.;last_position=start_position.copy();port_contact=False
    while time.monotonic()-started<130:
        elapsed=states[-1][0]-t0
        shift=min(.002,elapsed*.002)
        advance=min(.013,max(0.,elapsed-1.5)*.002)
        position=start_position+lateral*shift+axial*advance
        send(position);last_position=position
        rclpy.spin_once(node,timeout_sec=.02)
        delta=float(np.linalg.norm(forces[-1]-baseline));peak=max(peak,delta)
        trace.append({'sim_s':states[-1][0],'advance_m':advance,'lateral_m':shift,'delta_force_N':delta})
        names=[str(c.get(k,{}).get('name','')) for msg in contacts[-3:]
               for c in msg.get('contacts',[]) for k in ('collision1','collision2')]
        port_contact=any('sc_port_2' in name for name in names)
        if port_contact:reason='requested_port_contact';break
        if delta>6:reason='force_abort';break
        if advance>=.013:break
    else:reason='wall_timeout'
    # Return using a bounded speed. The baseline force is an independent abort.
    retreat_start=states[-1][0];distance=float(np.linalg.norm(last_position-start_position))
    retract_complete=False
    while time.monotonic()-started<180:
        fraction=min(1.,(states[-1][0]-retreat_start)*.003/max(distance,1e-6))
        send(last_position*(1-fraction)+start_position*fraction)
        rclpy.spin_once(node,timeout_sec=.02)
        if fraction>=1:retract_complete=True;break
    result={'purpose':'GT_scripted_collision_positive_control_not_policy_or_seating',
        'world_sha256':identity,'deliberate_lateral_miss_mm':2,'speed_mm_s':2,
        'reason':reason,'requested_port_contact_observed':port_contact,'peak_tared_force_N':peak,
        'retract_command_completed':retract_complete,'trace':trace,'contacts':contacts,
        'wall_s':time.monotonic()-started,'seating_success_claim':False}
    (OUT/'port_contact_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('trace','contacts')},indent=2))
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
