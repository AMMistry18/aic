"""Last-inch guarded baseline with a declared, prescribed rigid grasp.

Policy inputs: controller TCP, wrist wrench, protocol gap, fixed fixture design.
No scene file, cable truth, port truth, scoring event, or measured GT grasp.
This deliberately tests the initial-pose baseline before visual correction.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'aic_model'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from aic_control_interfaces.msg import ControllerState, MotionUpdate
from geometry_msgs.msg import WrenchStamped
from aic_model_interfaces.msg import Observation
from tf2_ros import Buffer,TransformListener

OUT=Path('/research/.artifacts/kilted_handoff')
# Fixture design used to construct the reset, not fitted from trial truth.
P=np.array([-.0017771781,-.0188744563,.054722198])
Q_WXYZ=np.array([.9840750466,.1756266707,-.0115567892,-.0248599222])
GRASP_R=Rotation.from_quat(Q_WXYZ[[1,2,3,0]]).as_matrix()

def numpy_json(value):
    if isinstance(value,np.ndarray):
        if value.size>256:return {'shape':list(value.shape),'dtype':str(value.dtype),'omitted_large_array':True}
        return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')

def main():
    run=os.environ.get('AIC_RESEARCH_RUN_ID','sc_known_grasp_001')
    connector=os.environ.get('AIC_RESEARCH_CONNECTOR','sc')
    revision=int(os.environ.get('AIC_RESEARCH_CONTROL_REVISION','1'))
    method=os.environ.get('AIC_RESEARCH_METHOD','initial_pose_guarded')
    if method not in ('initial_pose_guarded','visual_without_recovery','full_visual_force_recovery'):raise ValueError(method)
    gap=float(os.environ.get('AIC_RESEARCH_GAP_MM','10'))/1000
    seat={'sc':.01564,'sfp':.0458}[connector]
    path=OUT/run;path.mkdir(exist_ok=True)
    (path/'status.json').write_text(json.dumps({'active':True,'run':run}))
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true']);node=Node('known_grasp_lastinch_policy')
    states=[];forces=[];trace=[];observations=[];vision=[];lateral_correction=np.zeros(3)
    buffer=Buffer();listener=TransformListener(buffer,node) if method!='initial_pose_guarded' else None
    if method!='initial_pose_guarded':
        node.create_subscription(Observation,'/observations',lambda m:observations.__setitem__(slice(None),[m]),1)
    node.create_subscription(ControllerState,'/aic_controller/controller_state',lambda m:states.append(m),10)
    node.create_subscription(WrenchStamped,'/fts_broadcaster/wrench',lambda m:forces.append(m),qos_profile_sensor_data)
    pub=node.create_publisher(MotionUpdate,'/aic_controller/pose_commands',10)
    start=time.monotonic();reason='missing_state';peak=0.
    try:
        while time.monotonic()-start<15 and (not states or len(forces)<10):rclpy.spin_once(node,timeout_sec=.02)
        if not states or len(forces)<10:raise TimeoutError('No state/wrench')
        sim=lambda m:m.header.stamp.sec+m.header.stamp.nanosec/1e9
        policy_sim_start=sim(states[-1])
        initial=states[-1].tcp_pose
        R=Rotation.from_quat([initial.orientation.x,initial.orientation.y,initial.orientation.z,initial.orientation.w]).as_matrix()
        position=np.array([initial.position.x,initial.position.y,initial.position.z])
        axis=(R@GRASP_R)[:,2]
        initial_position=position.copy()
        if method!='initial_pose_guarded':
            from lastinch_visual import refine
            origin=position+R@P+gap*axis
            seen=set();vision_deadline=time.monotonic()+25;estimates=[]
            while time.monotonic()<vision_deadline and len(estimates)<3:
                rclpy.spin_once(node,timeout_sec=.01)
                if not observations:continue
                obs=observations[-1];st=obs.center_image.header.stamp;key=(st.sec,st.nanosec)
                if key in seen:continue
                seen.add(key)
                stamps=[getattr(obs,f'{c}_image').header.stamp for c in ('left','center','right')]
                exposure=[s.sec+s.nanosec/1e9 for s in stamps]
                age=node.get_clock().now().nanoseconds/1e9-min(exposure)
                if not 0<=age<=.35 or max(exposure)-min(exposure)>.12:
                    vision.append({'stamp':key,'rejection':'age_or_synchronization','age_s':age});continue
                measured,diagnostics=refine(obs,buffer,origin,R@GRASP_R,connector,path)
                vision.append({'stamp':key,'diagnostics':diagnostics,'estimate':None if measured is None else measured.tolist()})
                if measured is not None:estimates.append(measured)
            if len(estimates)<3:
                reason='physical_mouth_perception_failed'
                result={'run':run,'connector':connector,'method':method,'reason':reason,'vision':vision,
                    'ground_truth_policy_inputs':False,'wall_s':time.monotonic()-start,'sim_s':sim(states[-1])-policy_sim_start,
                    'peak_tared_force_N':0,'verified_success':None}
                (path/'policy.json').write_text(json.dumps(result,indent=2,default=numpy_json)+'\n');print(reason);return
            lateral_correction=np.median(estimates,axis=0)-origin
            position+=lateral_correction
        force_array=lambda m:np.array([m.wrench.force.x,m.wrench.force.y,m.wrench.force.z])
        baseline=np.median([force_array(m) for m in forces[-10:]],axis=0)
        # Only the controller's shared setup gravity/load compensation is read.
        bias=np.array(json.loads((OUT/'controller_hold_calibrated.json').read_text())['calibrated_force_base_N'])
        cmd=MotionUpdate();cmd.header.frame_id='base_link';cmd.pose=initial
        cmd.target_stiffness=np.diag([800.,800.,800.,80.,80.,80.] if revision==2 else [500.,500.,500.,50.,50.,50.]).ravel().tolist()
        cmd.target_damping=np.diag([160.,160.,160.,30.,30.,30.] if revision==2 else [45.,45.,45.,15.,15.,15.]).ravel().tolist()
        cmd.trajectory_generation_mode.mode=2
        f=R.T@bias
        cmd.feedforward_wrench_at_tip.force.x,cmd.feedforward_wrench_at_tip.force.y,cmd.feedforward_wrench_at_tip.force.z=map(float,f)
        t0=sim(states[-1]);duration=(gap+seat)/.001
        reason='wall_timeout';last_stamp=None;previous_t=t0;correction=np.zeros(3);search=np.zeros(3);stalled_since=None
        while time.monotonic()-start<450:
            t=sim(states[-1]);elapsed=t-t0
            advance=min(gap+seat,max(0,elapsed)*.001)
            align_fraction=min(1.,max(0,elapsed)/2) if method!='initial_pose_guarded' else 1.
            advance=min(gap+seat,max(0,elapsed-(2 if method!='initial_pose_guarded' else 0))*.001)
            target=initial_position+lateral_correction*align_fraction+axis*advance+search
            if method=='full_visual_force_recovery':
                actual=states[-1].tcp_pose.position
                actual_p=np.array([actual.x,actual.y,actual.z])
                lag=float(axis@(target-actual_p))
                force=float(np.linalg.norm(force_array(forces[-1])-baseline))
                if force>2 and lag>.0005:
                    if stalled_since is None:stalled_since=t
                    if t-stalled_since>.5:
                        duration_search=t-stalled_since-.5
                        radius=min(.00075,.00005*duration_search)
                        theta=2*np.pi*duration_search/2
                        search=(R@GRASP_R)[:,:2]@np.array([radius*np.cos(theta),radius*np.sin(theta)])
                        target=target-axis*max(0,lag-.0005)
                else:stalled_since=None
            if revision==2 and advance<gap and np.linalg.norm(force_array(forces[-1])-baseline)<1:
                actual=states[-1].tcp_pose.position
                dt=min(.02,max(0,t-previous_t))
                correction=np.clip(correction+800*(target-np.array([actual.x,actual.y,actual.z]))*dt,-3,3)
            previous_t=t
            f=R.T@(bias+correction)
            cmd.feedforward_wrench_at_tip.force.x,cmd.feedforward_wrench_at_tip.force.y,cmd.feedforward_wrench_at_tip.force.z=map(float,f)
            cmd.pose.position.x,cmd.pose.position.y,cmd.pose.position.z=map(float,target)
            cmd.header.stamp=node.get_clock().now().to_msg();pub.publish(cmd)
            rclpy.spin_once(node,timeout_sec=.01)
            delta=float(np.linalg.norm(force_array(forces[-1])-baseline));peak=max(peak,delta)
            if t!=last_stamp:
                trace.append({'sim_s':t,'advance_command_m':advance,'tared_force_N':delta,
                    'tcp':[states[-1].tcp_pose.position.x,states[-1].tcp_pose.position.y,states[-1].tcp_pose.position.z]})
                last_stamp=t
            if delta>8:reason='force_abort';break
            if elapsed>=duration+2+(2 if method!='initial_pose_guarded' else 0):
                actual=states[-1].tcp_pose.position
                goal=initial_position+lateral_correction+axis*(gap+seat)+search
                error=float(np.linalg.norm(goal-np.array([actual.x,actual.y,actual.z])))
                if error<.00025:reason='command_and_dwell_completed';break
                if method!='full_visual_force_recovery' or elapsed>=duration+12:
                    reason='tracking_stall_at_travel_limit';break
        # Force abort holds actual measured pose instead of continuing preload.
        if reason=='force_abort':
            cmd.pose=states[-1].tcp_pose;pub.publish(cmd)
        result={'run':run,'connector':connector,'method':method,'grasp_condition':'prescribed_fixture_design',
            'initial_tcp_position':initial_position.tolist(),'initial_tcp_rotation':R.tolist(),
            'vision':vision,'lateral_visual_correction_m':lateral_correction.tolist(),
            'controller_revision':revision,'free_space_load_correction_N':correction.tolist(),
            'ground_truth_policy_inputs':False,'gap_protocol_m':gap,'nominal_seat_depth_m':seat,
            'speed_m_s':.001,'force_abort_N':8,'reason':reason,'peak_tared_force_N':peak,
            'wall_s':time.monotonic()-start,'sim_s':sim(states[-1])-policy_sim_start,
            'motion_sim_s':sim(states[-1])-t0,'trace':trace,
            'verified_success':None,'verification':'external_recorder_required'}
        (path/'policy.json').write_text(json.dumps(result,indent=2,default=numpy_json)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2,default=numpy_json))
    finally:
        (path/'status.json').write_text(json.dumps({'active':False,'run':run}))
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
