"""Development force-guarded tactile registration under a prescribed grasp.

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
from tactile_geometry import moment_at_tip,bounded_forward_target
from wrench_noise import tip_axis_noise_sigma,moment_guard_estimate

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
    method=os.environ.get('AIC_RESEARCH_METHOD','force_tactile_full_pose')
    if method not in ('force_tactile_translation','force_tactile_full_pose','force_tactile_no_registration'):raise ValueError(method)
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
        'grasp_condition':'prescribed_fixture_design','force_abort_N':8,'global_torque_abort_Nm':None,'linear_motion_guard':'same instantaneous 8 N abort threshold as baseline',
        'probe_axial_torque_abort_Nm':.08,'translation_probe_amplitude_m':.002 if connector=='sc' else .00125,'angular_probe_amplitude_deg':4.5,
        'angular_lead_cap_rad':.0005,'nominal_seat_depth_m':seat,'gap_protocol_m':gap,
        'control_revision':9,'lateral_registration_enabled':method!='force_tactile_no_registration',
        'ablation_contract':'same CAD prior, load support, tracking compensation, input guards, approach waypoint and terminal dwell; omit lateral probes only',
        'tracking_feedback':'own TCP only; lateral PI outside probes, 0.5 N per base axis; rotation PI 0.02 Nm per base axis; terminal axial PI 0.25 N',
        'wrench_max_age_sim_s':.1,'telemetry_recovery_max_wall_s':1.,'input_recoveries':[],
        'twist_probe_stop':'settled tracking lag >0.15 deg and 99.9% moment lower bound >0.005 Nm; reverse before 0.08 Nm upper-bound abort',
        'moment_guard_observation':'50 unique samples and 100 stationary tare samples; 99.9% per-window Gaussian upper bound; not a raw-sample bound',
        'source_force_noise_sigma_N':.2,'source_torque_noise_sigma_Nm':.05,
        'terminal_axial_integral_gain_N_per_m_s':100,'terminal_axial_compensation_bound_N':.25,
        'twist_speed_deg_s':.25,'twist_damping_Nms_rad':3,'twist_opposed_moment_difference_min_Nm':.005,
        'twist_plateau_max_speed_rad_s':.0005,
        'lateral_probe_lead_cap_m':.0005,'terminal_command_dwell_s':8}
    reason='missing_input';sim_start=None;last_trace=None
    tracking_force=np.zeros(3);tracking_moment=np.zeros(3);last_control_stamp=None
    axial_compensation=0.;noise_tare=None;moment_cache={}
    try:
        while time.monotonic()-started<15 and (not state or len(wrenches)<10):rclpy.spin_once(node,timeout_sec=.01)
        if not state or len(wrenches)<10:raise StopTrial('missing_input')
        if wrenches[-1].header.frame_id!='ati/tool_link':raise StopTrial('unsupported_wrench_frame')
        sim_start=stamp(state[-1]);initial_tip,initial_R=tip();Rp=level_fixture_frame(initial_R,connector)
        force=lambda m:np.array([m.wrench.force.x,m.wrench.force.y,m.wrench.force.z])
        torque=lambda m:np.array([m.wrench.torque.x,m.wrench.torque.y,m.wrench.torque.z])
        f0=np.median([force(m) for m in wrenches[-10:]],axis=0)
        m0=np.median([torque(m) for m in wrenches[-10:]],axis=0)
        # Static URDF design: ati/tool_link -> gripper/tcp is +172 mm Z.
        # Shift moment to the prescribed connector tip before testing twist.
        sensor_to_tip=P+np.array([0.,0.,.172])
        tip_moment=lambda m:moment_at_tip(force(m),torque(m),sensor_to_tip)
        twist_tare=np.median([tip_moment(m) for m in wrenches[-10:]],axis=0)
        source_sigma=tip_axis_noise_sigma(GRASP_R[:,2],sensor_to_tip)
        def unique_messages(messages):return list({stamp(m):m for m in messages}.values())
        def moment_signal(messages):
            for m in messages:
                if stamp(m) not in moment_cache:moment_cache[stamp(m)]=float(tip_moment(m)@GRASP_R[:,2])
            return [moment_cache[stamp(m)] for m in messages]
        bias=np.array(json.loads((OUT/'controller_hold_calibrated.json').read_text())['calibrated_force_base_N'])
        command=MotionUpdate();command.header.frame_id='base_link';command.trajectory_generation_mode.mode=2
        command.target_stiffness=np.diag([500.,500.,500.,50.,50.,50.]).ravel().tolist()
        def current_hold():
            hold=MotionUpdate();hold.header.frame_id='base_link';hold.pose=state[-1].tcp_pose
            hold.target_stiffness=np.diag([500.,500.,500.,50.,50.,50.]).ravel().tolist()
            hold.target_damping=np.diag([45.,45.,45.,15.,15.,15.]).ravel().tolist();hold.trajectory_generation_mode.mode=2
            hold.feedforward_wrench_at_tip=command.feedforward_wrench_at_tip
            hold.header.stamp=node.get_clock().now().to_msg()
            return hold
        def send(point,Rgoal,phase,angular_probe=False):
            nonlocal peak_force,peak_torque,last_trace,tracking_force,tracking_moment,last_control_stamp,axial_compensation
            if time.monotonic()-started>1500:raise StopTrial('wall_timeout')
            rclpy.spin_once(node,timeout_sec=.005)
            now=stamp(state[-1])
            drained=0;drain_deadline=time.monotonic()+.02
            while now-stamp(wrenches[-1])>.04 and drained<32 and time.monotonic()<drain_deadline:
                rclpy.spin_once(node,timeout_sec=0);drained+=1;now=stamp(state[-1])
            hold_sim_s=0.
            if now-stamp(wrenches[-1])>.1:
                before=now;before_age=now-stamp(wrenches[-1]);wait_start=time.monotonic();hold=current_hold();pub.publish(hold)
                while now-stamp(wrenches[-1])>.04 and time.monotonic()-wait_start<1:
                    rclpy.spin_once(node,timeout_sec=.005);now=stamp(state[-1])
                    hold.header.stamp=node.get_clock().now().to_msg();pub.publish(hold)
                    observed=float(np.linalg.norm(force(wrenches[-1])-f0));peak_force=max(peak_force,observed)
                    if not np.isfinite(observed):raise StopTrial('nonfinite_wrench')
                    if observed>8:raise StopTrial('force_abort')
                hold_sim_s=max(0.,now-before)
                event={'sim_start_s':before,'sim_end_s':now,'initial_wrench_age_s':before_age,
                    'final_wrench_age_s':now-stamp(wrenches[-1]),'hold_wall_s':time.monotonic()-wait_start,'phase':phase}
                result['input_recoveries'].append(event)
                if now-stamp(wrenches[-1])>.1:
                    result['abort_observation']={**event,'state_stamp_s':now,'wrench_stamp_s':stamp(wrenches[-1])}
                    raise StopTrial('stale_wrench')
            delta_f=float(np.linalg.norm(force(wrenches[-1])-f0));delta_m=float(np.linalg.norm(torque(wrenches[-1])-m0))
            if not np.isfinite([delta_f,delta_m]).all():raise StopTrial('nonfinite_wrench')
            peak_force=max(peak_force,delta_f);peak_torque=max(peak_torque,delta_m)
            if delta_f>8:raise StopTrial('force_abort')
            # The rotational probe guards torque ABOUT the tip insertion axis.
            # Translation retains exactly the baseline's 8 N force threshold.
            # A v1 sensor-origin torque-norm gate confounded distal cable moments
            # with local twist contact. Its rejected trials remain archived.
            axial_moment=float((tip_moment(wrenches[-1])-twist_tare)@GRASP_R[:,2])
            guard=None
            if angular_probe:
                window=unique_messages(wrenches[-100:])[-50:]
                span=stamp(window[-1])-stamp(window[0]) if window else 0
                gaps=[stamp(b)-stamp(a) for a,b in zip(window,window[1:])]
                if noise_tare is None or len(window)<50 or not .9<=span<=1.1 or any(not 0<gap<=.05 for gap in gaps):
                    raise StopTrial('moment_guard_window_unavailable')
                guard=moment_guard_estimate(moment_signal(window),noise_tare,source_sigma)
                guard['window_span_s']=stamp(window[-1])-stamp(window[0])
                if guard['upper_abs_bound_Nm']>.08:
                    result['abort_observation']={'sim_s':now,'raw_tip_axial_moment_Nm':axial_moment,
                        'guard':guard,'tared_force_N':delta_f,'phase':phase}
                    raise StopTrial('probe_axial_torque_bound_abort')
            _,Ractual=tcp();pactual,Rtip=tip()
            Rcmd=Rgoal
            if angular_probe:
                actual_yaw=Rotation.from_matrix(Rp.T@Rtip).as_rotvec()[2]
                target_yaw=Rotation.from_matrix(Rp.T@Rgoal).as_rotvec()[2]
                limited=actual_yaw+np.clip(target_yaw-actual_yaw,-.0005,.0005)
                Rcmd=Rp@Rotation.from_euler('z',limited).as_matrix()
            # Preserve the measured tip point while orientation is compliant.
            # Bound preload when a connector or distal cable blocks progress.
            point=bounded_forward_target(point,pactual,Rp[:,2])
            probing=phase.startswith(('probe_','plateau_','twist_'))
            if phase.startswith(('probe_','plateau_')):
                lateral=Rp[:,:2].T@(point-pactual)
                point=point-Rp[:,:2]@(lateral-np.clip(lateral,-.0005,.0005))
            dt=0 if last_control_stamp is None else min(.02,max(0,now-last_control_stamp))
            last_control_stamp=now
            # Compensate slow load drift using the robot's tracking error.
            # Freeze during contact registration. Axial compensation is allowed
            # only at terminal dwell below 1 N; 500 N/m * 0.5 mm lead = 0.25 N.
            if not probing and delta_f<1:
                lateral_error=Rp[:,:2].T@(point-pactual)
                tracking_force=np.clip(tracking_force+200*Rp[:,:2]@lateral_error*dt,-.5,.5)
                rotation_error=Rotation.from_matrix(Rcmd@Rtip.T).as_rotvec()
                tracking_moment=np.clip(tracking_moment+2*rotation_error*dt,-.02,.02)
                if phase=='terminal_dwell':
                    axial_compensation=float(np.clip(axial_compensation+100*float((point-pactual)@Rp[:,2])*dt,-.25,.25))
            position=point-Ractual@P
            command.pose.position.x,command.pose.position.y,command.pose.position.z=map(float,position)
            q=Rotation.from_matrix(Rcmd@GRASP_R.T).as_quat()
            command.pose.orientation.x,command.pose.orientation.y,command.pose.orientation.z,command.pose.orientation.w=map(float,q)
            damping=np.diag([45.,45.,45.,15.,15.,15.])
            if angular_probe:damping[3:,3:]=Rp@np.diag([15.,15.,3.])@Rp.T
            command.target_damping=damping.ravel().tolist();f=Ractual.T@(bias+tracking_force+Rp[:,2]*axial_compensation)
            command.feedforward_wrench_at_tip.force.x,command.feedforward_wrench_at_tip.force.y,command.feedforward_wrench_at_tip.force.z=map(float,f)
            moment=Ractual.T@tracking_moment
            command.feedforward_wrench_at_tip.torque.x,command.feedforward_wrench_at_tip.torque.y,command.feedforward_wrench_at_tip.torque.z=map(float,moment)
            command.header.stamp=node.get_clock().now().to_msg();pub.publish(command)
            row={'sim_s':now,'phase':phase,'desired_tip':point.tolist(),'actual_tip':pactual.tolist(),
                'actual_tip_rotation':Rtip.tolist(),
                'yaw_relative_rad':float(Rotation.from_matrix(Rp.T@Rtip).as_rotvec()[2]),
                'tared_force_N':delta_f,'tared_sensor_torque_Nm':delta_m,
                'tared_tip_axial_torque_Nm':axial_moment,
                'tracking_force_base_N':tracking_force.tolist(),'tracking_moment_base_Nm':tracking_moment.tolist(),
                'terminal_axial_compensation_N':axial_compensation,'moment_guard':guard,
                'state_wrench_age_s':now-stamp(wrenches[-1]),'input_callbacks_drained':drained,'telemetry_hold_sim_s':hold_sim_s,
                'tared_force_tip_axes_N':(GRASP_R.T@(force(wrenches[-1])-f0)).tolist()}
            if now!=last_trace:trace.append(row);last_trace=now
            return row
        def segment(p0,p1,duration,phase,R0=None,R1=None,angular_probe=False,stop_on_torsion_contact=False):
            R0=Rp if R0 is None else R0;R1=R0 if R1 is None else R1
            relative=Rotation.from_matrix(R0.T@R1).as_rotvec();start=stamp(state[-1]);samples=[]
            while True:
                elapsed=stamp(state[-1])-start;u=min(1.,max(0,elapsed/duration))
                rotation=R0@Rotation.from_rotvec(relative*u).as_matrix()
                row=send(p0*(1-u)+p1*u,rotation,phase,angular_probe);samples.append(row)
                start+=row['telemetry_hold_sim_s']
                if stop_on_torsion_contact and elapsed>.3:
                    recent=[r for r in samples if r['sim_s']>=row['sim_s']-.2]
                    if len(recent)>10 and recent[-1]['sim_s']-recent[0]['sim_s']>=.18:
                        velocity=float(np.polyfit([r['sim_s']-recent[0]['sim_s'] for r in recent],
                                                  [r['yaw_relative_rad'] for r in recent],1)[0])
                        desired_yaw=Rotation.from_matrix(Rp.T@rotation).as_rotvec()[2]
                        guard=row['moment_guard'];margin=guard['upper_abs_bound_Nm']-abs(guard['estimate_Nm'])
                        lower=max(0.,abs(guard['estimate_Nm'])-margin)
                        if abs(desired_yaw-row['yaw_relative_rad'])>np.deg2rad(.15) and abs(velocity)<.0005 and lower>.005:
                            row['torsion_contact_condition']={'moment_abs_lower_Nm':lower,'plateau_speed_rad_s':velocity}
                            break
                if elapsed>=duration:break
            return samples
        def plateau(point,phase,rotation=None,angular=False):
            samples=segment(point,point,2 if angular else .5,phase,rotation,rotation,angular)
            end=samples[-1]['sim_s'];return [r for r in samples if r['sim_s']>=end-.2]
        result.update(initial_tip=initial_tip.tolist(),initial_rotation=initial_R.tolist(),fixture_orientation_prior=Rp.tolist())
        segment(initial_tip,initial_tip,2,'level_fixture_orientation',initial_R,Rp)
        mouth=initial_tip+Rp[:,2]*gap;center=mouth+Rp[:,2]*.005
        segment(initial_tip,center,(gap+.005)/.001,'approach_registration_depth')
        # Unload accumulated axial error before probing; no truth-depth input.
        measured,_=tip();center+=Rp[:,2]*float((measured-center)@Rp[:,2])
        segment(center,center,1,'unload_at_registration_depth')
        amplitude=.002 if connector=='sc' else .00125
        for axis in (() if method=='force_tactile_no_registration' else (0,1)):
            offset=Rp[:,axis]*amplitude;plus=center+offset;minus=center-offset
            segment(center,plus,amplitude/.001,f'probe_{axis}_positive');positive=plateau(plus,f'plateau_{axis}_positive')
            segment(plus,minus,2*amplitude/.001,f'probe_{axis}_negative');negative=plateau(minus,f'plateau_{axis}_negative')
            hi=float(np.mean([(Rp.T@(np.array(r['actual_tip'])-center))[axis] for r in positive]))
            lo=float(np.mean([(Rp.T@(np.array(r['actual_tip'])-center))[axis] for r in negative]))
            force_difference=float(np.mean([r['tared_force_tip_axes_N'][axis] for r in positive])-
                np.mean([r['tared_force_tip_axes_N'][axis] for r in negative]))
            clipped=(amplitude-hi>.0001 and lo+amplitude>.0001 and .00002<hi-lo<2*amplitude-.0002
                and abs(force_difference)>.05)
            probes.append({'axis':axis,'positive_actual_m':hi,'negative_actual_m':lo,
                'opposed_force_difference_N':force_difference,'opposed_contact_plateaus':clipped})
            if not clipped:raise StopTrial(f'tactile_axis_{axis}_unobservable')
            old=center.copy();center+=Rp[:,axis]*(hi+lo)*.5
            segment(minus,center,1.5,f'recenter_axis_{axis}')
        Rfinal=Rp
        if connector=='sfp' and method=='force_tactile_full_pose':
            tare_start=stamp(state[-1]);segment(center,center,3,'twist_noise_tare')
            tare_messages=unique_messages([m for m in wrenches if stamp(m)>=tare_start+1])
            while len(tare_messages)<100 and stamp(state[-1])-tare_start<5:
                send(center,Rp,'twist_noise_tare')
                tare_messages=unique_messages([m for m in wrenches if stamp(m)>=tare_start+1])
            if len(tare_messages)<100:raise StopTrial('stationary_moment_tare_unavailable')
            tare_messages=tare_messages[-100:]
            twist_tare=np.mean([tip_moment(m) for m in tare_messages],axis=0)
            noise_tare=moment_signal(tare_messages)
            result['moment_tare']={'samples':len(noise_tare),'span_s':stamp(tare_messages[-1])-stamp(tare_messages[0]),
                'mean_tip_axis_Nm':float(np.mean(noise_tare)),'source_sigma_Nm':source_sigma,
                'sample_std_Nm':float(np.std(noise_tare,ddof=1))}
            # Source CAD has clearance through +/-3 degrees at 5 mm depth.
            # The old 1.5-degree sweep could only measure tracking lag, not
            # opposing cage walls. Keep the same torque guard and lead bound.
            angle=np.deg2rad(4.5);Rplus=Rp@Rotation.from_euler('z',angle).as_matrix();Rminus=Rp@Rotation.from_euler('z',-angle).as_matrix()
            positive=segment(center,center,18,'twist_positive',Rp,Rplus,True,True)
            if not positive[-1].get('torsion_contact_condition'):raise StopTrial('torsion_positive_contact_unobserved')
            positive=[r for r in positive if r['sim_s']>=positive[-1]['sim_s']-.2]
            Rcontact=np.array(positive[-1]['actual_tip_rotation'])
            segment(center,center,.5,'twist_positive_unload',Rcontact,Rcontact,True)
            Rstart=tip()[1]
            duration=abs(Rotation.from_matrix(Rstart.T@Rminus).as_rotvec()[2])/np.deg2rad(.25)
            negative=segment(center,center,max(.5,duration),'twist_negative',Rstart,Rminus,True,True)
            if not negative[-1].get('torsion_contact_condition'):raise StopTrial('torsion_negative_contact_unobserved')
            negative=[r for r in negative if r['sim_s']>=negative[-1]['sim_s']-.2]
            hi=float(np.mean([r['yaw_relative_rad'] for r in positive]));lo=float(np.mean([r['yaw_relative_rad'] for r in negative]))
            slope=lambda rows:float(np.polyfit([r['sim_s']-rows[0]['sim_s'] for r in rows],[r['yaw_relative_rad'] for r in rows],1)[0])
            pos_speed=slope(positive);neg_speed=slope(negative)
            moment_difference=float(np.mean([r['tared_tip_axial_torque_Nm'] for r in positive])-
                                    np.mean([r['tared_tip_axial_torque_Nm'] for r in negative]))
            clipped=(angle-hi>np.deg2rad(.15) and lo+angle>np.deg2rad(.15) and np.deg2rad(.05)<hi-lo<2*angle-np.deg2rad(.3)
                and abs(pos_speed)<.0005 and abs(neg_speed)<.0005 and abs(moment_difference)>.005
                and np.mean([r['moment_guard']['estimate_Nm'] for r in positive])*np.mean([r['moment_guard']['estimate_Nm'] for r in negative])<0)
            probes.append({'axis':'twist','positive_actual_deg':np.rad2deg(hi),'negative_actual_deg':np.rad2deg(lo),
                'positive_plateau_speed_rad_s':pos_speed,'negative_plateau_speed_rad_s':neg_speed,
                'opposed_moment_difference_Nm':moment_difference,'opposed_contact_plateaus':clipped})
            if not clipped:raise StopTrial('tactile_twist_unobservable')
            Rfinal=Rp@Rotation.from_euler('z',(hi+lo)/2).as_matrix()
            Rcontact=np.array(negative[-1]['actual_tip_rotation'])
            segment(center,center,.5,'twist_negative_unload',Rcontact,Rcontact,True)
            Rstart=tip()[1];duration=abs(Rotation.from_matrix(Rstart.T@Rfinal).as_rotvec()[2])/np.deg2rad(.25)
            segment(center,center,max(3,duration),'twist_recenter',Rstart,Rfinal,True);plateau(center,'twist_settle',Rfinal,True)
        remaining=seat-float((center-mouth)@Rp[:,2]);finish=center+Rp[:,2]*remaining
        segment(center,finish,max(.1,remaining/.001),'seat',Rfinal,Rfinal)
        segment(finish,finish,8,'terminal_dwell',Rfinal,Rfinal)
        reason='command_and_dwell_completed'
    except StopTrial as exc:reason=str(exc)
    except Exception as exc:
        reason=f'exception:{type(exc).__name__}:{exc}'
        import traceback;traceback.print_exc()
    finally:
        if state and 'command' in locals() and reason!='command_and_dwell_completed':
            # Preserve existing load support while removing pose-error preload.
            # Dropping feedforward caused >2 mm post-abort depth motion in v6.
            pub.publish(current_hold())
        result.update(reason=reason,wall_s=time.monotonic()-started,sim_s=stamp(state[-1])-sim_start if state and sim_start is not None else 0,
            peak_tared_force_N=peak_force,peak_tared_torque_Nm=peak_torque,probes=probes,trace=trace,verified_success=None)
        with (path/'policy.json').open('w') as f:json.dump(result,f,indent=2,default=numpy_json)
        (path/'status.json').write_text(json.dumps({'active':False,'run':run}))
        print(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2,default=numpy_json))
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
