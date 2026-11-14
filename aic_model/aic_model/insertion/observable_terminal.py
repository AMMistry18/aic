"""Policy-observable seating candidate; independent evaluator decides success.

Uses estimated plug/port frames, robot kinematics, wrist force and simulation
time. Never reads scoring events, contact entities or simulator object poses.
"""
import os
import time
import numpy as np


def enabled():return os.environ.get('AIC_INSERT_OBSERVABLE_TERMINAL','0')=='1'


class SeatDwell:
    def __init__(self,depth_m,lateral_m,angle_deg,depth_tolerance_m=.00025,dwell_s=1.):
        self.depth=depth_m;self.lateral=lateral_m;self.angle=angle_deg
        self.depth_tolerance=depth_tolerance_m;self.dwell=dwell_s
        self.start=None;self.last=None;self.last_reason='no_samples'
        self.samples=0;self.maximum_gap_s=0.;self.gap_resets=0;self.longest_valid_span_s=0.

    def observe(self,stamp,relative_position,rotation_error_deg,force):
        p=np.asarray(relative_position,dtype=float)
        good=np.isfinite([stamp,*p,rotation_error_deg,force]).all() and 0<=force<=8.
        if not good:reason='invalid_or_excess_force'
        elif abs(p[2]-self.depth)>self.depth_tolerance:reason='estimated_depth'
        elif np.linalg.norm(p[:2])>self.lateral:reason='estimated_lateral'
        elif rotation_error_deg>self.angle:reason='estimated_angle'
        else:reason='within_estimated_geometry'
        contiguous=self.last is not None and 0<stamp-self.last<=.03
        self.samples+=1
        if self.last is not None:
            self.maximum_gap_s=max(self.maximum_gap_s,float(stamp-self.last))
            if not contiguous:self.gap_resets+=1
        if reason!='within_estimated_geometry':self.start=None
        elif self.start is None or not contiguous:self.start=stamp
        self.last=stamp;self.last_reason=reason
        if self.start is not None:self.longest_valid_span_s=max(self.longest_valid_span_s,stamp-self.start)
        return self.start is not None and stamp-self.start>=self.dwell


def hold_candidate(controller,fixed_tip,rotation,connector):
    policy=controller.policy
    depth=.01564 if connector=='sc' else .0458
    tolerance=.00025 if connector=='sc' else .0001
    angle=.25 if connector=='sc' else .1
    fixed=np.asarray(fixed_tip,dtype=float).copy()
    # Remove SFP overtravel used to reach a simulator scoring pad. Preserve
    # lateral recovery offsets and command the public CAD seating depth.
    offset=controller.Rp.T@(fixed-controller.port_pos);offset[2]=depth
    fixed=controller.port_pos+controller.Rp@offset
    registered=False
    if connector=='sc' and os.environ.get('AIC_SC_AXIAL_CONTACT_REGISTER','0')=='1':
        from .axial_contact_registration import register_sc_shoulder
        fixed=register_sc_shoulder(controller,rotation)
        if fixed is None:return 'stalled'
        registered=True
    checker=SeatDwell(depth,tolerance,angle)
    sim=lambda:policy._parent_node.get_clock().now().nanoseconds*1e-9
    start=sim();wall=time.monotonic();reason='terminal_timeout'
    previous_command=getattr(controller,'_last_seat_command_depth',None)
    previous_time=start
    while sim()-start<6 and time.monotonic()-wall<90:
        policy._enforce_action_deadline(controller.move_robot)
        observation=controller.get_observation()
        if observation is None:raise RuntimeError('Observable seat dwell lost observation')
        force=controller._force_magnitude(observation)
        tip,R=controller._tip_pose();relative=controller.Rp.T@(tip-controller.port_pos)
        error=np.rad2deg(np.arccos(np.clip((np.trace(np.asarray(rotation).T@R)-1)/2,-1,1)))
        if not np.isfinite(force) or force>8:
            controller._hold_tip(tip,rotation);reason='terminal_force_abort';break
        target=(controller._tcp_target(fixed,rotation) if connector=='sc' else policy._tcp_target_for_tip(fixed,rotation))
        if connector=='sc' and getattr(controller.config,'terminal_brake_enable',False) and not registered:
            from .sc_controller import next_sc_depth
            stamp=sim()
            previous_command=next_sc_depth(relative[2],
                relative[2] if previous_command is None else previous_command,
                stamp-previous_time,force,controller.config)
            previous_time=stamp
            governed=fixed+controller.Rp[:,2]*(previous_command-depth)
            target=controller._tcp_target(governed,rotation)
        stiffness=controller._seating_stiffness() if connector=='sc' else controller.STIFFNESS
        policy.set_pose_target(controller.move_robot,target,stiffness=stiffness,damping=controller.DAMPING)
        if checker.observe(sim(),relative,error,force):reason='observable_candidate_dwell_completed';break
        # A 20ms requested sleep plus TF/command latency repeatedly exceeded
        # the unchanged 30ms continuity bound. Poll with headroom instead of
        # broadening the acceptance gap or counting unsampled time as dwell.
        policy.sleep_for(.005)
    diagnostic={'connector':connector,'reason':reason,'estimated_geometry_last_reason':checker.last_reason,
                'sim_s':sim()-start,'wall_s':time.monotonic()-wall,'GT_or_scoring_event_used':False,
                'success_is_independently_verified':False,'sample_count':checker.samples,
                'maximum_sample_gap_s':checker.maximum_gap_s,'gap_resets':checker.gap_resets,
                'longest_valid_sample_span_s':checker.longest_valid_span_s,
                'maximum_accepted_gap_s':.03,'poll_sleep_s':.005}
    policy._observable_terminal_diagnostic=diagnostic
    policy.get_logger().info(f'[{connector}] OBSERVABLE_TERMINAL {diagnostic}')
    if reason=='terminal_force_abort':return 'hard_failure'
    return 'seated' if reason=='observable_candidate_dwell_completed' else 'stalled'
