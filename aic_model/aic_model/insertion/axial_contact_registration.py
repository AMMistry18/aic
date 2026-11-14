"""Bounded SC shoulder registration from robot motion and wrist force.

An opt-in development component. A repeatable axial stop near the CAD seat is
an additional measurement, not simulator contact truth or an insertion verdict.
The independent evaluator retains the original seating/contact/force criteria.
"""
import numpy as np


def blocked_window(samples, *, window_s=.8):
    """Require continued commanded advance, little motion, and rising reaction."""
    if not samples:return None
    end=samples[-1][0];rows=np.asarray([s for s in samples if end-window_s-.03<=s[0]<=end])
    if len(rows)<20 or rows[-1,0]-rows[0,0]<window_s-.03:return None
    if not np.isfinite(rows).all():return None
    gaps=np.diff(rows[:,0])
    if np.any(gaps<=0) or np.max(gaps)>.05:return None
    advance=rows[-1,2]-rows[0,2];travel=np.ptp(rows[:,1])
    first=rows[rows[:,0]<=rows[0,0]+.2,3];last=rows[rows[:,0]>=end-.2,3]
    reaction=float(np.median(last)-np.median(first))
    lead=float(np.median(rows[-8:,2]-rows[-8:,1]))
    if advance<.0001 or travel>.000025 or lead<.00015 or reaction<.025:return None
    return dict(stop_depth_m=float(np.median(rows[-8:,1])),command_advance_m=float(advance),
                observed_travel_m=float(travel),reaction_increase_N=reaction,elastic_lead_m=lead)


def register_sc_shoulder(controller,rotation):
    policy=controller.policy;sim=lambda:policy._parent_node.get_clock().now().nanoseconds*1e-9
    start=sim();tip,_=controller._tip_pose();relative=controller.Rp.T@(tip-controller.port_pos)
    depth=float(relative[2]);axis=controller.Rp[:,2];lateral=relative[:2].copy()
    diagnostic=dict(reason='not_deep_candidate',GT_or_scoring_event_used=False,
                    positive_compression_signal='negative tared wrist force in port Z',
                    public_seat_depth_m=.01564,maximum_axial_correction_m=.0005,
                    motion_speed_m_s=.0002,repeat_stop_tolerance_m=.00005,stops=[])
    diagnostic['probe_samples']=[]
    policy._axial_contact_registration=diagnostic
    if depth<controller.config.seat_candidate_depth_m or np.linalg.norm(lateral)>.00025:return None
    command=float(getattr(controller,'_last_seat_command_depth',depth));last=sim()
    # Unload first: the previous trajectory may already carry almost the full
    # elastic lead, leaving no command range for a measurable loading slope.
    samples=[];stops=[];phase='initial_retract';retract_goal=depth-.0002;release_start=None
    initial_depth=depth
    diagnostic['reason']='contact_not_observed'
    while sim()-start<20:
        policy._enforce_action_deadline(controller.move_robot)
        stamp=sim();dt=max(0.,min(.04,stamp-last));last=stamp
        observation=controller.get_observation();force=controller._force_magnitude(observation)
        tip,_=controller._tip_pose();relative=controller.Rp.T@(tip-controller.port_pos)
        depth=float(relative[2]);f,_=controller._wrench_plug_frame(observation)
        if not np.isfinite([force,*f]).all() or force>=controller.config.force_abort_n:
            controller._hold_tip(tip,rotation);diagnostic['reason']='force_abort';return None
        if np.linalg.norm(relative[:2]-lateral)>.0002:
            controller._hold_tip(tip,rotation);diagnostic['reason']='lateral_motion';return None
        diagnostic['probe_samples'].append(dict(sim_s=stamp,phase=phase,estimated_depth_m=depth,
            applied_command_depth_m=command,force_port_N=f.tolist()))
        if phase in ('initial_retract','retract'):
            command=max(retract_goal,command-.0002*dt)
            reference=initial_depth if phase=='initial_retract' else stops[0]['stop_depth_m']
            if depth<=reference-.00012:
                release_start=depth;phase='press';samples=[]
        else:
            # Associate the measured reaction with the command already applied,
            # not the next setpoint that has yet to reach the controller.
            if not samples or stamp>samples[-1][0]:samples.append([stamp,depth,command,float(-f[2])])
            stop=blocked_window(samples)
            if stop:
                stops.append(stop);diagnostic['stops']=stops.copy()
                if len(stops)==1:
                    retract_goal=depth-.0002;phase='retract';samples=[]
                else:
                    difference=abs(stops[1]['stop_depth_m']-stops[0]['stop_depth_m'])
                    correction=.01564-stop['stop_depth_m']
                    diagnostic.update(repeat_stop_difference_m=difference,axial_correction_m=correction,
                                      released_distance_m=stops[0]['stop_depth_m']-release_start)
                    if difference>.00005 or abs(correction)>.0005:
                        controller._hold_tip(tip,rotation);diagnostic['reason']='inconsistent_contact_geometry';return None
                    # Update only the visually estimated grasp's axial component
                    # using the independently observed, repeatable CAD shoulder.
                    from .contract import quat_to_matrix
                    from .research_ablation import axial_correction
                    applied=axial_correction(policy,correction)
                    diagnostic.update(measured_axial_correction_m=correction,applied_axial_correction_m=applied)
                    tcp,quat=policy._tcp();translation,grasp_rotation=policy._sc_grasp_transform
                    policy._sc_grasp_transform=(np.asarray(translation)+quat_to_matrix(quat).T@(axis*applied),grasp_rotation)
                    body=getattr(policy,'_sc_body_diagnostic',None)
                    if body is not None and body.get('axial_measurement_pending'):
                        body.update(axial_measurement_pending=False,axial_contact_measurement_completed=True)
                    diagnostic.update(reason='repeatable_shoulder_registered',sim_s=sim()-start,
                                      elapsed_sim_s=sim()-start,physical_success_certified=False)
                    # Preserve physical TCP target and its gentle elastic preload
                    # across the change of estimated tip coordinates.
                    target=controller.port_pos+controller.Rp@np.r_[lateral,command+applied]
                    policy.get_logger().info(f'[sc] AXIAL_CONTACT_REGISTER {diagnostic}')
                    return target
            if phase=='press' and force<controller.config.seat_force_cap_n:
                command=min(command+.0002*dt,depth+.0003,.01564+.0005)
            if command>=.01564+.0005 and depth>=command-.000025:
                diagnostic['reason']='no_axial_stop_in_bounded_range';break
        target=controller.port_pos+controller.Rp@np.r_[lateral,command]
        policy.set_pose_target(controller.move_robot,controller._tcp_target(target,rotation),
                               stiffness=controller._seating_stiffness(),damping=controller.DAMPING)
        policy.sleep_for(.02)
    controller._hold_tip(tip,rotation)
    diagnostic.update(sim_s=sim()-start,phase=phase)
    policy.get_logger().warn(f'[sc] AXIAL_CONTACT_REGISTER {diagnostic}')
    return None
