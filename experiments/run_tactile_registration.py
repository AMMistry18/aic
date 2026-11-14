"""Experimental, bounded tactile registration with a prescribed rigid grasp.

Runtime inputs are controller TCP and wrist wrench only. CAD supplies a level
fixture orientation family; its yaw and mouth position are NOT supplied by the
simulator. Opposed, clipped contact plateaus register the two lateral axes and
(for SFP full-pose mode) twist. This is a development prototype, not a result.
"""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from aic_control_interfaces.msg import ControllerState,MotionUpdate
from geometry_msgs.msg import WrenchStamped
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_known_grasp_trial import P,GRASP_R,numpy_json

OUT=Path('/research/.artifacts/kilted_handoff')

class StopTrial(RuntimeError):pass

def level_fixture_frame(initial,connector):
    if connector=='sfp':fixed=Rotation.from_euler('x',3.12895).as_matrix()
    else:fixed=Rotation.from_euler('xyz',[1.57,0,1.57]).as_matrix()@Rotation.from_euler('xyz',[1.5708,3.14159,0]).as_matrix()
    residual=initial@fixed.T
    yaw=np.arctan2(residual[1,0]-residual[0,1],residual[0,0]+residual[1,1])
    return Rotation.from_euler('z',yaw).as_matrix()@fixed

def main():
    run=os.environ['AIC_RESEARCH_RUN_ID'];connector=os.environ.get('AIC_RESEARCH_CONNECTOR','sfp')
    method=os.environ.get('AIC_RESEARCH_METHOD','tactile_full_pose')
    if method not in ('tactile_translation','tactile_full_pose'):raise ValueError(method)
    gap=float(os.environ.get('AIC_RESEARCH_GAP_MM','10'))/1000
    seat={'sc':.01564,'sfp':.0458}[connector]
    path=OUT/run;path.mkdir(exist_ok=True)
    (path/'status.json').write_text(json.dumps({'active':True,'run':run}))
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true']);node=Node('prescribed_grasp_tactile_registration')
    state=[];wrenches=[];trace=[];probes=[];started=time.monotonic();peak_force=0.;peak_torque=0.
    node.create_subscription(ControllerState,'/aic_controller/controller_state',lambda m:state.__setitem__(slice(None),[m]),10)
    node.create_subscription(WrenchStamped,'/fts_broadcaster/wrench',lambda m:wrenches.append(m),qos_profile_sensor_data)
    pub=node.create_publisher(MotionUpdate,'/aic_controller/pose_commands',10)
    stamp=lambda m:m.header.stamp.sec+m.header.stamp.nanosec/1e9
    def tcp():
        p=state[-1].tcp_pose
        R=Rotation.from_quat([p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]).as_matrix()
        return np.array([p.position.x,p.position.y,p.position.z]),R
    def tip():
        p,R=tcp();return p+R@P,R@GRASP_R
    result={'run':run,'method':method,'connector':connector,'ground_truth_policy_inputs':False,
        'grasp_condition':'prescribed_fixture_design','force_abort_N':8,'torque_abort_Nm':.2,
        'probe_torque_abort_Nm':.08,'translation_probe_amplitude_m':.00125,'angular_probe_amplitude_deg':1.5,
        'angular_lead_cap_rad':.0005,'nominal_seat_depth_m':seat,'gap_protocol_m':gap}
    reason='missing_input';sim_start=None;last_trace=None
    try:
        while time.monotonic()-started<15 and (not state or len(wrenches)<10):rclpy.spin_once(node,timeout_sec=.01)
        if not state or len(wrenches)<10:raise StopTrial('missing_input')
        sim_start=stamp(state[-1]);initial_tip,initial_R=tip();Rp=level_fixture_frame(initial_R,connector)
        force=lambda m:np.array([m.wrench.force.x,m.wrench.force.y,m.wrench.force.z])
        torque=lambda m:np.array([m.wrench.torque.x,m.wrench.torque.y,m.wrench.torque.z])
        f0=np.median([force(m) for m in wrenches[-10:]],axis=0)
        m0=np.median([torque(m) for m in wrenches[-10:]],axis=0)
        bias=np.array(json.loads((OUT/'controller_hold_calibrated.json').read_text())['calibrated_force_base_N'])
        command=MotionUpdate();command.header.frame_id='base_link';command.trajectory_generation_mode.mode=2
        command.target_stiffness=np.diag([500.,500.,500.,50.,50.,50.]).ravel().tolist()
        def send(point,Rgoal,phase,angular_probe=False):
            nonlocal peak_force,peak_torque,last_trace
            if time.monotonic()-started>600:raise StopTrial('wall_timeout')
            rclpy.spin_once(node,timeout_sec=.005)
            now=stamp(state[-1])
            if now-stamp(wrenches[-1])>.1:raise StopTrial('stale_wrench')
            delta_f=float(np.linalg.norm(force(wrenches[-1])-f0));delta_m=float(np.linalg.norm(torque(wrenches[-1])-m0))
            peak_force=max(peak_force,delta_f);peak_torque=max(peak_torque,delta_m)
            if delta_f>8:raise StopTrial('force_abort')
            if delta_m>(.08 if angular_probe else .2):raise StopTrial('probe_torque_abort' if angular_probe else 'torque_abort')
            _,Ractual=tcp();pactual,Rtip=tip()
            Rcmd=Rgoal
            if angular_probe:
                actual_yaw=Rotation.from_matrix(Rp.T@Rtip).as_rotvec()[2]
                target_yaw=Rotation.from_matrix(Rp.T@Rgoal).as_rotvec()[2]
                limited=actual_yaw+np.clip(target_yaw-actual_yaw,-.0005,.0005)
                Rcmd=Rp@Rotation.from_euler('z',limited).as_matrix()
            # Preserve the measured tip point while orientation is compliant.
            position=point-Ractual@P
            command.pose.position.x,command.pose.position.y,command.pose.position.z=map(float,position)
            q=Rotation.from_matrix(Rcmd@GRASP_R.T).as_quat()
            command.pose.orientation.x,command.pose.orientation.y,command.pose.orientation.z,command.pose.orientation.w=map(float,q)
            damping=np.diag([45.,45.,45.,15.,15.,15.])
            if angular_probe:damping[3:,3:]=Rp@np.diag([15.,15.,1.])@Rp.T
            command.target_damping=damping.ravel().tolist();f=Ractual.T@bias
            command.feedforward_wrench_at_tip.force.x,command.feedforward_wrench_at_tip.force.y,command.feedforward_wrench_at_tip.force.z=map(float,f)
            command.header.stamp=node.get_clock().now().to_msg();pub.publish(command)
            row={'sim_s':now,'phase':phase,'desired_tip':point.tolist(),'actual_tip':pactual.tolist(),
                'yaw_relative_rad':float(Rotation.from_matrix(Rp.T@Rtip).as_rotvec()[2]),
                'tared_force_N':delta_f,'tared_torque_Nm':delta_m}
            if now!=last_trace:trace.append(row);last_trace=now
            return row
        def segment(p0,p1,duration,phase,R0=None,R1=None,angular_probe=False):
            R0=Rp if R0 is None else R0;R1=R0 if R1 is None else R1
            relative=Rotation.from_matrix(R0.T@R1).as_rotvec();start=stamp(state[-1]);samples=[]
            while True:
                elapsed=stamp(state[-1])-start;u=min(1.,max(0,elapsed/duration))
                rotation=R0@Rotation.from_rotvec(relative*u).as_matrix()
                row=send(p0*(1-u)+p1*u,rotation,phase,angular_probe);samples.append(row)
                if elapsed>=duration:break
            return samples
        def plateau(point,phase,rotation=None,angular=False):
            samples=segment(point,point,.5,phase,rotation,rotation,angular)
            end=samples[-1]['sim_s'];return [r for r in samples if r['sim_s']>=end-.2]
        result.update(initial_tip=initial_tip.tolist(),initial_rotation=initial_R.tolist(),fixture_orientation_prior=Rp.tolist())
        segment(initial_tip,initial_tip,2,'level_fixture_orientation',initial_R,Rp)
        mouth=initial_tip+Rp[:,2]*gap;center=mouth+Rp[:,2]*.005
        segment(initial_tip,center,(gap+.005)/.001,'approach_registration_depth')
        for axis in (0,1):
            offset=Rp[:,axis]*.00125;plus=center+offset;minus=center-offset
            segment(center,plus,1.25,f'probe_{axis}_positive');positive=plateau(plus,f'plateau_{axis}_positive')
            segment(plus,minus,2.5,f'probe_{axis}_negative');negative=plateau(minus,f'plateau_{axis}_negative')
            hi=float(np.mean([(Rp.T@(np.array(r['actual_tip'])-center))[axis] for r in positive]))
            lo=float(np.mean([(Rp.T@(np.array(r['actual_tip'])-center))[axis] for r in negative]))
            clipped=(.00125-hi>.0001 and lo+.00125>.0001 and .00002<hi-lo<.0023)
            probes.append({'axis':axis,'positive_actual_m':hi,'negative_actual_m':lo,'opposed_contact_plateaus':clipped})
            if not clipped:raise StopTrial(f'tactile_axis_{axis}_unobservable')
            old=center.copy();center+=Rp[:,axis]*(hi+lo)*.5
            segment(minus,center,1.5,f'recenter_axis_{axis}')
        Rfinal=Rp
        if connector=='sfp' and method=='tactile_full_pose':
            angle=np.deg2rad(1.5);Rplus=Rp@Rotation.from_euler('z',angle).as_matrix();Rminus=Rp@Rotation.from_euler('z',-angle).as_matrix()
            segment(center,center,3,'twist_positive',Rp,Rplus,True);positive=plateau(center,'twist_positive_plateau',Rplus,True)
            segment(center,center,6,'twist_negative',Rplus,Rminus,True);negative=plateau(center,'twist_negative_plateau',Rminus,True)
            hi=float(np.mean([r['yaw_relative_rad'] for r in positive]));lo=float(np.mean([r['yaw_relative_rad'] for r in negative]))
            clipped=(angle-hi>np.deg2rad(.15) and lo+angle>np.deg2rad(.15) and np.deg2rad(.05)<hi-lo<2*angle-np.deg2rad(.3))
            probes.append({'axis':'twist','positive_actual_deg':np.rad2deg(hi),'negative_actual_deg':np.rad2deg(lo),'opposed_contact_plateaus':clipped})
            if not clipped:raise StopTrial('tactile_twist_unobservable')
            Rfinal=Rp@Rotation.from_euler('z',(hi+lo)/2).as_matrix()
            segment(center,center,3,'twist_recenter',Rminus,Rfinal,True);plateau(center,'twist_settle',Rfinal,True)
        finish=center+Rp[:,2]*(seat-.005)
        segment(center,finish,(seat-.005)/.001,'seat',Rfinal,Rfinal)
        segment(finish,finish,2,'terminal_dwell',Rfinal,Rfinal)
        reason='command_and_dwell_completed'
    except StopTrial as exc:reason=str(exc)
    except Exception as exc:
        reason=f'exception:{type(exc).__name__}:{exc}'
        import traceback;traceback.print_exc()
    finally:
        if state and reason!='command_and_dwell_completed':
            hold=MotionUpdate();hold.header.frame_id='base_link';hold.pose=state[-1].tcp_pose
            hold.target_stiffness=np.diag([500.,500.,500.,50.,50.,50.]).ravel().tolist()
            hold.target_damping=np.diag([45.,45.,45.,15.,15.,15.]).ravel().tolist();hold.trajectory_generation_mode.mode=2;pub.publish(hold)
        result.update(reason=reason,wall_s=time.monotonic()-started,sim_s=stamp(state[-1])-sim_start if state and sim_start is not None else 0,
            peak_tared_force_N=peak_force,peak_tared_torque_Nm=peak_torque,probes=probes,trace=trace,verified_success=None)
        with (path/'policy.json').open('w') as f:json.dump(result,f,indent=2,default=numpy_json)
        (path/'status.json').write_text(json.dumps({'active':False,'run':run}))
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2,default=numpy_json))
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
