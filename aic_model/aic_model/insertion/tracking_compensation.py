"""Bounded own-kinematics compensation for slow lateral tracking bias."""
import numpy as np


def bounded_axial_feedforward(lateral,axis,force_N,*,axial_bound_N=.25,total_bound_N=.5):
    if (axial_bound_N,total_bound_N) not in ((.25,.5),(1.,1.2),(1.5,1.75)):
        raise RuntimeError('Unknown tracking force bounds')
    if not np.isfinite([*lateral,*axis,force_N]).all() or abs(force_N)>axial_bound_N+1e-12:
        raise RuntimeError('Invalid/beyond-bound axial tracking force')
    combined=np.asarray(lateral)+np.asarray(axis)*force_N
    return combined*min(1.,total_bound_N/max(float(np.linalg.norm(combined)),1e-12))


class LateralTrackingCompensator:
    def __init__(self,axes):
        self.axes=np.asarray(axes,dtype=float).reshape(3,2)
        self.force_base=np.zeros(3);self.integral_base=np.zeros(3)
        self.last=None;self.trace=[];self.peak=0.

    def update(self,stamp,desired,measured,force_N,*,freeze=False):
        desired=np.asarray(desired,dtype=float);measured=np.asarray(measured,dtype=float)
        if not np.isfinite([stamp,*desired,*measured,force_N]).all() or force_N<0:
            raise RuntimeError('Invalid lateral tracking observation')
        if self.last is not None and stamp<self.last:
            raise RuntimeError('Lateral tracking clock moved backwards')
        dt=0. if self.last is None else min(.05,max(0.,stamp-self.last));self.last=stamp
        error=self.axes@(self.axes.T@(desired-measured))
        frozen=freeze or force_N>=1.
        if not frozen:
            candidate=self.integral_base+200.*error*dt
            self.integral_base=candidate*min(1.,.5/max(np.linalg.norm(candidate),1e-12))
        # Native seating stiffness is only 90 N/m laterally. The bounded
        # proportional term supplies prompt tracking during load changes,
        # near the separately qualified controller's 500 N/m stiffness.
        proportional=np.zeros(3) if freeze else 400.*error
        candidate=self.integral_base+proportional
        self.force_base=candidate*min(1.,.5/max(np.linalg.norm(candidate),1e-12))
        self.peak=max(self.peak,float(np.linalg.norm(self.force_base)))
        self.trace.append({'sim_s':float(stamp),'error_base_m':error.tolist(),
                           'force_base_N':self.force_base.tolist(),'observed_force_N':float(force_N),
                           'integral_force_base_N':self.integral_base.tolist(),
                           'proportional_force_base_N':proportional.tolist(),
                           'frozen_contact_or_probe':bool(frozen)})
        return self.force_base.copy()

    def diagnostics(self):
        return {'input':'own robot TCP target and measured position; raw wrist guard',
                'proportional_gain_N_per_m':400.,'integral_gain_N_per_m_s':200.,
                'norm_bound_N':.5,'contact_integral_freeze_N':1.,
                'disabled_by_ablation':getattr(self,'ablated',False),'peak_compensation_N':self.peak,'simulator_object_inputs':False,'trace':self.trace}


def install(controller):
    from .contract import quat_to_matrix
    tracker=LateralTrackingCompensator(controller.Rp[:,:2])
    controller.policy._lateral_tracking_compensator=tracker
    original=controller.move_robot
    def tracked_move(motion_update=None,joint_motion_update=None):
        if motion_update is not None:
            if motion_update.header.frame_id!='base_link':
                raise RuntimeError('Lateral tracking requires base_link pose command')
            if any(abs(x)>1e-12 for x in motion_update.wrench_feedback_gains_at_tip):
                raise RuntimeError('Lateral tracking requires zero native wrench feedback')
            point,quat=controller.policy._tcp()
            observation=controller.get_observation()
            if observation is None:raise RuntimeError('Lateral tracking lost force observation')
            force=controller._force_magnitude(observation)
            desired=np.array([getattr(motion_update.pose.position,k) for k in 'xyz'])
            stamp=controller.policy._parent_node.get_clock().now().nanoseconds*1e-9
            if getattr(controller,'_tracking_motion_stopped',False):
                extra=np.zeros(3)
            else:
                from .research_ablation import lateral_disabled
                tracker.ablated=lateral_disabled(controller.policy)
                extra=(np.zeros(3) if tracker.ablated else tracker.update(stamp,desired,point,force,
                                     freeze=getattr(controller,'_in_contact_probe',False)))
                path=np.asarray(getattr(controller,'_path_damping_force_base_N',np.zeros(3)))
                if path.shape!=(3,) or not np.isfinite(path).all() or np.linalg.norm(path)>.5+1e-12:
                    raise RuntimeError('Invalid free-path damping feedforward')
                extra=bounded_axial_feedforward(extra+path,controller.Rp[:,2],
                    float(getattr(controller,'_axial_tracking_force_N',0.)),
                    axial_bound_N=getattr(controller,'_axial_tracking_bound_N',.25),
                    total_bound_N=getattr(controller,'_combined_tracking_bound_N',.5))
            tcp_force=quat_to_matrix(quat).T@extra
            for k,value in zip('xyz',tcp_force):
                msg=motion_update.feedforward_wrench_at_tip.force
                setattr(msg,k,float(getattr(msg,k)+value))
        return original(motion_update=motion_update,joint_motion_update=joint_motion_update)
    controller.move_robot=tracked_move
