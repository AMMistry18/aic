"""Experimental camera-initialized, contact-registered last-inch SFP control.

Conditional on upstream angular alignment and the public level fixture family.
Contact registers relative coordinates; it does not certify global object pose
or insertion. Success is decided separately by the independent recorder.
"""
import os
import time
import numpy as np
from .sfp_controller import SfpInsertionController, matrix_to_quaternion
from .axial_contact_registration import blocked_window
from .contact_registration import register_lateral
from .tracking_compensation import install


def motion_parameters(profile):
    if profile=='legacy36':return .003,.0005,.25,False
    if profile in ('fast36','fast37'):return .005,.001,.5,True
    if profile=='fast38':return .010,.002,.5,True
    raise RuntimeError('Unknown SFP motion profile')


def next_axial_target(depth,command,dt,force,*,terminal=None,profile='legacy36'):
    """Persistent travel with authority for damping; brake before the seat.

    At 500 N/m and 100 Ns/m, a 0.3 mm lead alone permits at most 1.5 mm/s
    before drag. The 0.8 mm free lead permits the scheduled 3 mm/s, with
    nominal elastic demand <=0.4 N. Terminal lead remains 0.3 mm.
    """
    if not np.isfinite([depth,command,dt,force]).all():raise RuntimeError('Invalid SFP trajectory observation')
    if force>=3:return command
    terminal=depth>.044 if terminal is None else terminal
    free_speed,terminal_speed,_,_=motion_parameters(profile)
    return min(.0458,command+(terminal_speed if terminal else free_speed)*min(.04,max(0.,dt)),
               depth+(.0003 if terminal else .0008))


def next_axial_integral(integral_N,depth,command,dt,force,*,terminal,profile='legacy36'):
    """Reject drag from own tracking error, excluding the scheduled damping lag.

    This uses no object pose or contact state. Retain the existing 0.25 N
    axial authority as feedforward, without expanding the pose-lead bounds.
    """
    if not np.isfinite([integral_N,depth,command,dt,force]).all() or dt<0 or force<0:
        raise RuntimeError('Invalid axial tracking observation')
    if force>=3:return integral_N
    at_goal=command>=.0458-1e-9
    free_speed,terminal_speed,bound,damping_ff=motion_parameters(profile)
    speed=0. if at_goal else (terminal_speed if terminal else free_speed)
    error=command-depth-(0. if damping_ff else (100./500.)*speed)
    return float(np.clip(integral_N+500.*error*min(.04,dt),-bound if at_goal else 0.,bound))


def axial_tracking_force(integral_N,command,force,*,terminal,profile='legacy36'):
    free_speed,terminal_speed,_,damping_ff=motion_parameters(profile)
    if not damping_ff:return integral_N
    if force>=3:return 0.
    speed=0. if command>=.0458-1e-9 else (terminal_speed if terminal else free_speed)
    return integral_N+100.*speed


def waypoint_axial_integral(integral_N,error_m,dt_s,force_N):
    """Bounded own-robot load compensation after the free path reaches its goal."""
    if not np.isfinite([integral_N,error_m,dt_s,force_N]).all() or dt_s<0 or force_N<0:
        raise RuntimeError('Invalid free-waypoint axial observation')
    if force_N>=1.:return integral_N
    # For the public 500 N/m, 100 Ns/m damped model, the linear integral
    # critical gain is K^2/(4D)=625. Use 500 below that value; the same 0.15 N
    # saturation and total pose-lead bound remain active.
    return float(np.clip(integral_N+500.*error_m*min(.04,dt_s),-.15,.15))


def kinematic_lip_candidate(samples):
    from .lip_search import stationary_search_candidate
    return stationary_search_candidate(samples)


class DatumController(SfpInsertionController):
    def clock(self):
        return self.policy._parent_node.get_clock().now().nanoseconds*1e-9

    def read(self):
        obs=self.get_observation()
        stamp=obs.wrist_wrench.header.stamp
        self._last_wrist_stamp_s=stamp.sec+stamp.nanosec*1e-9
        f,_=self._wrench_plug_frame(obs);point,R=self._tip_pose()
        return point,R,f,self.clock()

    def _stop_motion_compensation(self):
        # A deadline hold must not inherit travel/probe feedforward or PI.
        # Calibrated payload support belongs to the outer harness and remains.
        self._tracking_motion_stopped=True
        self._axial_tracking_force_N=0.
        self._path_damping_force_base_N=np.zeros(3)
        self.diagnostic['motion_compensation_stopped']=True

    def guard(self):
        deadline=getattr(self.policy,'_action_deadline_wall',None)
        if deadline is not None and time.monotonic()>deadline:
            self._stop_motion_compensation()
            self.diagnostic['reason']='action_deadline'
        self.policy._enforce_action_deadline(self.move_robot)
        point,_,f,_=self.read()
        if not np.isfinite([*point,*f]).all() or np.linalg.norm(f)>=6:
            raise RuntimeError('SFP datum nonfinite/6N force guard')
        if np.linalg.norm(point-self.origin)>.080:raise RuntimeError('SFP datum 80mm workspace guard')
        if self.clock()-self.started>90:raise RuntimeError('SFP datum 90s simulation budget')

    def send(self,p,R):
        self.guard()
        self.policy.set_pose_target(self.move_robot,self.policy._tcp_target_for_tip(p,R),
                                    stiffness=self.STIFFNESS,damping=self.DAMPING)

    def line(self,goal,*,speed=.002,settle=.3,budget=12):
        # Waypoints are not the force-measurement sweeps. Freezing the tracker
        # here left posture-dependent robot bias uncompensated, so a valid
        # visually specified waypoint could never meet the unchanged 40um gate.
        previous=getattr(self,'_in_contact_probe',False)
        self._in_contact_probe=False
        try:
            if getattr(self,'_motion_profile','legacy36') in ('fast37','fast38') and speed==.002:
                return self._line_smooth(goal,settle=settle,budget=budget)
            return self._line_tracking(goal,speed=speed,settle=settle,budget=budget)
        finally:self._in_contact_probe=previous

    def _line_smooth(self,goal,*,settle=.3,budget=12):
        from .waypoint_motion import smooth_waypoint
        start=self.clock();last=start;initial,_,_,_=self.read();arrived=None;integral=0.
        # AIC Cartesian errors and impedance matrices are in base_link.
        # Only the resulting feedforward vector is rotated into TCP at send.
        axis=self.Rp[:,2];damping=np.diag(np.asarray(self.DAMPING,dtype=float)[:3])
        _,_,duration=smooth_waypoint(initial,goal,0.)
        record=dict(start_sim_s=start,goal_base_m=goal.tolist(),samples=[],
                    axial_integral_gain_N_per_m_s=500.,axial_integral_bound_N=.15,
                    trajectory='bounded quintic 5mm/s, 20mm/s^2; damping feedforward',
                    scheduled_duration_s=duration,compensation_scope='own robot tracking only')
        self.diagnostic.setdefault('waypoints',[]).append(record)
        try:
            while self.clock()-start<budget:
                point,_,force,stamp=self.read();dt=min(.04,max(0.,stamp-last));last=stamp
                command,velocity,_=smooth_waypoint(initial,goal,stamp-start)
                integral=waypoint_axial_integral(integral,float(axis@(command-point)),dt,float(np.linalg.norm(force)))
                self._axial_tracking_force_N=integral
                self._path_damping_force_base_N=damping@velocity
                delta=command-point;target=point+delta*min(1.,.0005/max(np.linalg.norm(delta),1e-12))
                record['samples'].append(dict(sim_s=stamp,error_port_m=(self.Rp.T@(goal-point)).tolist(),
                    axial_integral_N=integral,pose_lead_m=float(np.linalg.norm(target-point)),
                    scheduled_velocity_base_m_s=velocity.tolist(),damping_force_base_N=self._path_damping_force_base_N.tolist()))
                self.send(target,self.held_rotation)
                if (stamp-start>=duration or np.linalg.norm(goal-initial)<.00004) and np.linalg.norm(goal-point)<.00004:
                    if arrived is None:arrived=stamp
                    if stamp-arrived>=settle:return
                else:arrived=None
                self.policy.sleep_for(.02)
            raise RuntimeError('SFP datum waypoint tracking timeout')
        finally:
            self._path_damping_force_base_N=np.zeros(3);self._axial_tracking_force_N=0.

    def _line_tracking(self,goal,*,speed=.002,settle=.3,budget=12):
        start=self.clock();last=start;point,_,_,_=self.read();command=point.copy();arrived=None
        integral=0.;axis=self.Rp[:,2]
        record=dict(start_sim_s=start,goal_base_m=goal.tolist(),samples=[],
                    axial_integral_gain_N_per_m_s=500.,axial_integral_bound_N=.15,
                    compensation_scope='free waypoint only; own estimated tip error')
        self.diagnostic.setdefault('waypoints',[]).append(record)
        while self.clock()-start<budget:
            point,_,force,stamp=self.read();dt=min(.04,max(0.,stamp-last));last=stamp
            delta=goal-command;command+=delta*min(1.,speed*dt/max(np.linalg.norm(delta),1e-12))
            # Correct residual axial load bias only after the nominal free
            # path reaches its destination. Do not integrate travel damping
            # or use this compensation during either contact measurement.
            if np.linalg.norm(goal-command)<1e-9:
                integral=waypoint_axial_integral(integral,float(axis@(goal-point)),dt,float(np.linalg.norm(force)))
            # Preserve the total 0.5 mm pose-lead bound, including compensation.
            desired=command+axis*(integral/500.)
            delta=desired-point;target=point+delta*min(1.,.0005/max(np.linalg.norm(delta),1e-12))
            record['samples'].append(dict(sim_s=stamp,error_port_m=(self.Rp.T@(goal-point)).tolist(),
                axial_integral_N=integral,pose_lead_m=float(np.linalg.norm(target-point))))
            self.send(target,self.held_rotation)
            if np.linalg.norm(goal-point)<.00004:
                if arrived is None:arrived=stamp
                if stamp-arrived>=settle:return
            else:arrived=None
            self.policy.sleep_for(.02)
        raise RuntimeError('SFP datum waypoint tracking timeout')

    def _legacy_mouth_stop(self,offset,*,start_depth_m=-.0015):
        axis=self.Rp[:,2];startpoint=self.port_pos+self.Rp@np.array([offset,0.,start_depth_m])
        self.line(startpoint)
        point,_,_,last=self.read();command=float(axis@(point-self.port_pos));samples=[];start=last
        while self.clock()-start<28:
            point,_,f,stamp=self.read();depth=float(axis@(point-self.port_pos))
            # SFP's sensor wrench is the load transmitted at the wrist:
            # opposing insertion produces negative port-Z force. The inherited
            # helper returns wrist force, not a positive compression magnitude.
            # Feed positive compression to the detector.
            if not samples or stamp>samples[-1][0]:samples.append([stamp,depth,command,float(-f[2])])
            stop=blocked_window(samples)
            self.diagnostic['axial_probe_samples'].append(dict(sim_s=stamp,offset_m=offset,
                measured_depth_m=depth,applied_command_m=command,force_port_N=f.tolist()))
            if stop:
                stop['offset_m']=offset;return stop
            command=min(command+.0004*min(.04,max(0.,stamp-last)),depth+.0003,.003)
            last=stamp
            if command>=.003 and depth>=command-.00003:raise RuntimeError('No mouth lip in bounded axial capture range')
            self.send(self.port_pos+self.Rp@np.array([offset,0.,command]),self.held_rotation)
            self.policy.sleep_for(.02)
        raise RuntimeError('SFP mouth stop unobservable')

    def _lip_force_plateau(self,target,force_N):
        self._axial_tracking_force_N=force_N
        start=self.clock();rows=[];points=[];last_sensor=None
        self.send(target,self.held_rotation)
        while self.clock()-start<.65:
            point,_,force,stamp=self.read();sensor=self._last_wrist_stamp_s
            if stamp-start>=.2 and (last_sensor is None or sensor>last_sensor):
                rows.append([sensor,float(self.Rp[:,2]@(point-self.port_pos)),float(-force[2])])
                points.append((self.Rp.T@point).tolist());last_sensor=sensor
            self.send(target,self.held_rotation);self.policy.sleep_for(.02)
        return rows,self.clock(),points

    def mouth_stop(self,offset,*,start_depth_m=-.0015):
        from .qualified_lip_contact import fixed_triplet
        if getattr(self,'_axial_tracking_bound_N',.25)<1.:
            return self._legacy_mouth_stop(offset,start_depth_m=start_depth_m)
        axis=self.Rp[:,2];startpoint=self.port_pos+self.Rp@np.array([offset,0.,start_depth_m])
        self.line(startpoint)
        point,_,_,last=self.read();command=float(axis@(point-self.port_pos));samples=[];start=last
        while self.clock()-start<28:
            point,_,f,stamp=self.read();depth=float(axis@(point-self.port_pos))
            if not samples or stamp>samples[-1][0]:samples.append([stamp,depth,command,float(-f[2])])
            self.diagnostic['axial_probe_samples'].append(dict(sim_s=stamp,offset_m=offset,
                measured_depth_m=depth,applied_command_m=command,force_port_N=f.tolist()))
            trigger=kinematic_lip_candidate(samples)
            if trigger:
                # Blocking is a trigger, never an accepted contact measurement.
                # One fixed 0/1/0 N excitation with actual unique wrench stamps.
                target=self.port_pos+self.Rp@np.array([offset,0.,command])
                try:
                    low,t0,p0=self._lip_force_plateau(target,0.)
                    high,t1,p1=self._lip_force_plateau(target,1.)
                    released,t2,p2=self._lip_force_plateau(target,0.)
                finally:self._axial_tracking_force_N=0.
                proof=fixed_triplet(low,high,released,low_end=t0,high_end=t1,release_end=t2,kinematic_blocked=True)
                points=np.asarray(p0+p1+p2)
                stationary=bool(len(points) and np.all(np.ptp(points,axis=0)<=np.array([.00005,.00005,.000025])))
                record=dict(offset_m=offset,trigger=trigger,low=low,high=high,released=released,
                            proof=proof,own_tip_stationary_across_excitation=stationary,
                            object_or_target_truth_used=False)
                self.diagnostic.setdefault('qualified_lip_excitations',[]).append(record)
                if not proof['accepted'] or not stationary:
                    raise RuntimeError('SFP fixed lip excitation unobservable: '+proof['reason'])
                return dict(**trigger,offset_m=offset,stop_depth_m=proof['low']['depth_m'],
                            reaction_increase_N=proof['contrast_N'],qualified_reversible_excitation=True)
            command=min(command+.0012*min(.04,max(0.,stamp-last)),depth+.0003,.003)
            last=stamp
            if command>=.003 and depth>=command-.00003:raise RuntimeError('No mouth lip in bounded axial capture range')
            self.send(self.port_pos+self.Rp@np.array([offset,0.,command]),self.held_rotation)
            self.policy.sleep_for(.02)
        raise RuntimeError('SFP mouth stop unobservable')

    def run_datum(self):
        self.started=self.clock();self.origin,self.held_rotation=self._tip_pose()
        self.diagnostic=dict(ground_truth_or_scoring_inputs=False,scope='upstream angular alignment; level fixture family',
            axial_probe_samples=[],stops=[],physical_success_certified=False,reason='started',
            positive_compression_signal='negative tared wrist force in port Z')
        self.diagnostic.update(insertion_free_lead_m=.0008,insertion_terminal_lead_m=.0003,
                               insertion_terminal_braking_depth_m=.044,insertion_samples=[],
                               inter_probe_clearance_m=.00075,
                               lateral_plateau_window_s=.4,lateral_force_uses_actual_sensor_stamps=True,
                               first_lip_for_clearance_only_until_opposed_confirmation=True)
        self.policy._sfp_contact_datum=self.diagnostic
        profile=os.environ.get('AIC_SFP_MOTION_PROFILE','legacy36')
        parameters=motion_parameters(profile)
        self._motion_profile=profile
        if profile in ('fast36','fast37','fast38'):
            if float(os.environ.get('AIC_RESEARCH_FORCE_CEILING_N','8'))!=20.:
                raise RuntimeError('Fast SFP motion requires prospective 20 N protocol')
            self._axial_tracking_bound_N=1.;self._combined_tracking_bound_N=1.2
        if profile=='fast38':
            self._axial_tracking_bound_N=1.5;self._combined_tracking_bound_N=1.75
        self.diagnostic.update(motion_profile=profile,travel_speed_m_s=parameters[0],
            terminal_speed_m_s=parameters[1],static_drag_bound_N=parameters[2],
            scheduled_damping_feedforward=parameters[3])
        def stage(phase):
            self.diagnostic['phase']=phase
            self.policy.get_logger().info(f'[sfp] DATUM_PHASE {phase} sim_s={self.clock():.4f}')
        install(self)
        try:
            # Never rotate using an inaccurate learned angular estimate. The
            # measured grasp's own initial rotation keeps the physical wrist.
            relative=self.Rp.T@(self.origin-self.port_pos)
            if not -.025<=relative[2]<=-.003 or np.linalg.norm(relative[:2])>.004:
                raise RuntimeError('SFP visual handoff outside declared capture envelope')
            self._in_contact_probe=True
            for offset in (.002,-.002):
                stage(f'entrance_lip_probe_{offset:+.3f}m')
                # The first independently measured lip is a provisional free
                # traverse reference, not yet an accepted port-coordinate
                # correction. Avoid returning to the biased visual mouth.
                prior=self.diagnostic['stops']
                start_depth=-.0015 if not prior else prior[0]['stop_depth_m']-.00075
                stop=self.mouth_stop(offset,start_depth_m=start_depth);self.diagnostic['stops'].append(stop)
                tip,_,_,_=self.read();clear=tip-self.Rp[:,2]*.00075
                self.line(clear)
            stops=self.diagnostic['stops'];difference=abs(stops[0]['stop_depth_m']-stops[1]['stop_depth_m'])
            # Public source OBB audit: the held tip contacts the lip at roughly
            # 53.5 um depth for either signed 2 mm X probe. No final backstop is
            # assumed: the 45.8 mm seat itself is a source clearance region.
            correction=float(np.mean([s['stop_depth_m'] for s in stops])-.0000535)
            self.diagnostic.update(opposed_stop_difference_m=difference,axial_relative_datum_correction_m=correction,
                                   public_lip_datum_m=.0000535)
            if difference>.0001 or abs(correction)>.003:raise RuntimeError('Inconsistent SFP opposed mouth datum')
            from .research_ablation import axial_correction
            applied=axial_correction(self.policy,correction)
            self.diagnostic.update(measured_axial_correction_m=correction,applied_axial_correction_m=applied)
            self.port_pos+=self.Rp[:,2]*applied
            stage('aligned_entry')
            self.line(self.port_pos-self.Rp[:,2]*.00075)
            self.line(self.port_pos+self.Rp[:,2]*.002,speed=.001,budget=9)
            stage('opposed_side_wall_registration')
            center,records=register_lateral(read=self.read,send=self.send,sleep=self.policy.sleep_for,
                clock=self.clock,check_deadline=self.guard,axes=self.Rp,command_dt=.02,axial_preload_m=0,
                plateau_window_s=.4,wrench_stamp=lambda:self._last_wrist_stamp_s)
            self.diagnostic['lateral_registration']=records
            if center is None:raise RuntimeError('SFP opposed lateral contact unobservable')
            lateral=self.Rp[:,:2]@(self.Rp[:,:2].T@(center-self.port_pos))
            if np.linalg.norm(lateral)>.001:raise RuntimeError('SFP lateral datum outside 1mm refined capture range')
            self.port_pos+=lateral;self.diagnostic['lateral_relative_datum_correction_base_m']=lateral.tolist()
            self.diagnostic['registered_control_port_base_m']=self.port_pos.tolist()
            self._in_contact_probe=False
            self.line(self.port_pos+self.Rp[:,2]*.002,speed=.001)
            point,_,_,last=self.read();command=float(self.Rp[:,2]@(point-self.port_pos));start=last
            axial_integral=0.;settled=None;terminal=False
            stage('full_depth_insertion')
            # Full SFP travel is 45.8 mm, unlike the SC shoulder at 15.64 mm.
            while self.clock()-start<32:
                point,_,f,stamp=self.read();dt=min(.04,max(0.,stamp-last));last=stamp
                relative=self.Rp.T@(point-self.port_pos);depth=float(relative[2])
                terminal=terminal or depth>.044
                command=next_axial_target(depth,command,dt,float(np.linalg.norm(f)),terminal=terminal,profile=profile)
                axial_integral=next_axial_integral(axial_integral,depth,command,dt,float(np.linalg.norm(f)),terminal=terminal,profile=profile)
                self._axial_tracking_force_N=axial_tracking_force(axial_integral,command,float(np.linalg.norm(f)),terminal=terminal,profile=profile)
                self.diagnostic['insertion_samples'].append(dict(sim_s=stamp,estimated_depth_m=depth,
                    estimated_lateral_m=relative[:2].tolist(),command_depth_m=command,force_port_N=f.tolist(),
                    axial_tracking_force_N=self._axial_tracking_force_N,static_drag_integral_N=axial_integral,terminal_braking_latched=terminal))
                target=self.port_pos+self.Rp[:,2]*command
                self.send(target,self.held_rotation)
                if abs(depth-.0458)<=.0001 and np.linalg.norm(relative[:2])<=.0001:
                    if settled is None:settled=stamp
                    if stamp-settled>=1.2:
                        self.diagnostic.update(reason='estimated_datum_depth_dwell_completed',
                            elapsed_sim_s=stamp-self.started,estimated_depth_m=depth,terminal_axial_integral_N=axial_integral)
                        return True
                else:settled=None
                self.policy.sleep_for(.02)
            raise RuntimeError('SFP datum insertion timeout')
        except RuntimeError as exc:
            self._stop_motion_compensation()
            self.diagnostic.update(reason=str(exc),elapsed_sim_s=self.clock()-self.started)
            self.policy.get_logger().warn('[sfp] DATUM_REJECT '+str(exc))
            point,_,_,_=self.read()
            # Guard already checked every outgoing motion; use normal policy
            # hold on failure rather than issuing another probing trajectory.
            self.policy.set_pose_target(self.move_robot,self.policy._tcp_target_for_tip(point,self.held_rotation),
                                        stiffness=self.HOLD_STIFFNESS,damping=self.HOLD_DAMPING)
            return False
        finally:
            self._axial_tracking_force_N=0.


def run(policy,task,get_observation,move_robot,send_feedback):
    from .sfp_fixture_refinement import perceive
    try:p,R=perceive(policy,get_observation())
    except (ValueError,RuntimeError) as exc:
        policy._sfp_fixture_failure=str(exc);policy.get_logger().error('[sfp] FIXTURE_REJECT '+str(exc));return False
    if os.environ.get('AIC_DEV_SHORT_TEST','0')=='1':
        policy._development_short_result=dict(mode='SFP_perception_only',port_position=p.tolist(),port_rotation=R.tolist(),
            measured_grasp_translation=np.asarray(policy._sfp_grasp_transform[0]).tolist(),insertion_attempted=False)
        return False
    policy._wrench_baseline=policy._wrench_vector(get_observation())
    controller=DatumController(policy,task,get_observation,move_robot,send_feedback,
                               port_pos=p,port_quat=matrix_to_quaternion(R),Rp=R)
    return controller.run_datum()
