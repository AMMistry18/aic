"""Bounded visual reacquisition from an already measured held-plug frame.

The action uses robot TCP, wrist force and the original visual grasp. It never
looks up simulator target/connector frames. A retreat may improve occlusion;
it does not by itself certify improved accuracy.
"""
import os
import time
import numpy as np


def enabled():
    return os.environ.get('AIC_INSERT_VISUAL_REACQUIRE','0') == '1'


def bounded_retreat(policy, get_observation, move_robot, connector, distance_m=.010):
    from geometry_msgs.msg import Point, Pose, Quaternion
    if not 0 < distance_m <= .010:
        raise ValueError('Visual retreat is limited to 10 mm per action')
    if connector == 'sc':
        from .sc_controller import sc_tip_from_tcp as tip_from_tcp
    elif connector == 'sfp':
        from .sfp_controller import sfp_tip_from_tcp as tip_from_tcp
    else:
        raise ValueError('Unsupported connector')
    # No constant-grasp fallback for any reacquisition movement.
    if getattr(policy,f'_{connector}_grasp_transform',None) is None:return False
    origin,quat=policy._tcp()
    origin=np.asarray(origin,dtype=float).copy()
    _,Rtip=tip_from_tcp(policy,origin,quat)
    direction=-np.asarray(Rtip[:,2],dtype=float)
    observation=get_observation()
    if observation is None:return False
    baseline=policy._wrench_vector(observation)[:3].copy()
    sim=lambda:policy._parent_node.get_clock().now().nanoseconds*1e-9
    start_sim=sim();start_wall=time.monotonic();stable_since=None
    correction=np.zeros(3);previous_sim=start_sim
    events=getattr(policy,'_visual_reacquisition_events',[])
    policy._visual_reacquisition_events=events
    event={'connector':connector,'start_sim_s':start_sim,'distance_m':distance_m,
           'direction_base':direction.tolist(),'GT_used':False,'outcome':'tracking_timeout'}
    events.append(event)
    while time.monotonic()-start_wall < 120 and sim()-start_sim < 8:
        policy._enforce_action_deadline(move_robot)
        observation=get_observation()
        if observation is None:raise RuntimeError('Visual retreat lost observation')
        force=float(np.linalg.norm(policy._wrench_vector(observation)[:3]-baseline))
        current,_=policy._tcp();current=np.asarray(current)
        displacement=current-origin
        elapsed=sim()-start_sim
        progress=min(distance_m,max(0.,elapsed)*.002)
        nominal=origin+direction*progress
        # A changing cable load leaves a small steady tracking error. Integrate
        # measured TCP error in free space, with at most 0.5 mm command offset
        # (0.4 N at this action's 800 N/m stiffness). No estimated target pose
        # or simulator state participates in this tracking compensation.
        dt=max(0.,min(.1,sim()-previous_sim));previous_sim=sim()
        correction+=(nominal-current)*dt
        norm=float(np.linalg.norm(correction))
        if norm>.0005:correction*=.0005/norm
        target=nominal+correction
        # Stop on contact or unexpected lateral movement, then hold measured TCP.
        perpendicular=displacement-direction*float(displacement@direction)
        abort=not np.isfinite(force) or force>3 or np.linalg.norm(perpendicular)>.002 or np.linalg.norm(displacement)>distance_m+.002
        if abort:target=current;event['outcome']='force_or_tracking_abort'
        pose=Pose(position=Point(x=float(target[0]),y=float(target[1]),z=float(target[2])),
                  orientation=Quaternion(w=float(quat[0]),x=float(quat[1]),y=float(quat[2]),z=float(quat[3])))
        policy.set_pose_target(move_robot,pose,stiffness=[800.,800.,800.,80.,80.,80.],damping=[80.,80.,80.,20.,20.,20.])
        if abort:break
        if progress>=distance_m and np.linalg.norm(current-(origin+direction*distance_m))<=.00025:
            stable_since=sim() if stable_since is None else stable_since
            if sim()-stable_since>=.3:
                event['outcome']='retreat_completed';break
        else:stable_since=None
        policy.sleep_for(.05)
    event.update(end_sim_s=sim(),wall_s=time.monotonic()-start_wall,tracking_correction_m=correction.tolist(),
                 actual_displacement_m=(np.asarray(policy._tcp()[0])-origin).tolist())
    policy.get_logger().info(f'[{connector}] VISUAL_REACQUIRE {event}')
    return event['outcome']=='retreat_completed'
