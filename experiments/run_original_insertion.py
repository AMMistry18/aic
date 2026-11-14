"""Execute the original learned perception and controller on a last-inch reset.

Only task identity is supplied by the harness. No scene poses, prescribed grasp
transform, or simulator contact information are read by this policy process.
The external recorder performs independent geometric/contact verification.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
import torch
from geometry_msgs.msg import WrenchStamped, Pose, Point, Quaternion
from rclpy.qos import qos_profile_sensor_data
from aic_control_interfaces.msg import MotionUpdate
from aic_model.aic_model import AicModel
from aic_task_interfaces.msg import Task
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation

ROOT = Path('/research')


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run=os.environ['AIC_RESEARCH_RUN_ID']
    connector=os.environ['AIC_RESEARCH_CONNECTOR']
    target=int(os.environ['AIC_RESEARCH_TARGET_INDEX'])
    slot=int(os.environ['AIC_RESEARCH_SFP_PORT'])
    out=ROOT/'.artifacts/kilted_handoff'/run
    out.mkdir(exist_ok=True)
    (out/'status.json').write_text(json.dumps({'active':True,'run':run}))
    started=time.monotonic()
    result={'run':run,'connector':connector,'method':os.environ.get('AIC_RESEARCH_METHOD','original_perception'),
            'ground_truth_policy_inputs':False,'prescribed_grasp_policy_input':False,
            'grasp_condition':'rigid_attachment_visually_estimated',
            'verified_success':None,'trace':[],'feedback':[],'command_count':0,
            'reason':'policy_configuration_failed','peak_tared_force_N':None,
            'outer_force_abort_N':float(os.environ.get('AIC_RESEARCH_FORCE_CEILING_N','8')),'outer_force_guard':'pre-command tared wrist force check, not a hard physical force bound',
            'refinement_strategy':os.environ.get('AIC_PLUG_POSE_REFINE_STRATEGY','fixed')}
    if result['outer_force_abort_N'] not in (8.,20.):raise ValueError('Unknown force protocol')
    result['sfp_motion_profile']=os.environ.get('AIC_SFP_MOTION_PROFILE','legacy36')
    result['harness_revision']=38
    result['angular_screen_diagnostic_ablation']=os.environ.get('AIC_SC_BODY_ANGULAR_SCREEN_ABLATION','0')=='1'
    result['counts_as_guarded_production_policy']=not result['angular_screen_diagnostic_ablation']
    result['sfp_conditional_contact_datum']=os.environ.get('AIC_SFP_CONTACT_DATUM','0')=='1'
    if result['sfp_conditional_contact_datum']:
        result['counts_as_guarded_production_policy']=False
    result['rim_occlusion_model']='white_plug_and_orange_cable'
    result['lateral_tracking_compensation_enabled']=os.environ.get('AIC_SC_LATERAL_TRACKING','0')=='1'
    result['body_translation_refinement_enabled']=os.environ.get('AIC_SC_BODY_TRANSLATION','0')=='1'
    result['terminal_contact_refinement_enabled']=os.environ.get('AIC_SC_TERMINAL_CONTACT_REGISTER','0')=='1'
    result['native_wrench_feedback_gain']=float(os.environ.get('AIC_INSERT_NATIVE_WRENCH_GAIN','.5'))
    result['rim_refinement_enabled']=os.environ.get('AIC_SC_RIM_REFINE','0')=='1'
    result['orientation_condition']=('upstream_aligned_handoff_assumed' if os.environ.get('AIC_SC_KEEP_HANDOFF_ORIENTATION','0')=='1'
                                     else 'visual_grasp_based_rotation_correction')
    result['rim_raster_offset_px']=float(os.environ.get('AIC_SC_RIM_RASTER_OFFSET_PX','0'))
    result['research_controller_parameters']={k:v for k,v in os.environ.items() if k.startswith(('RL_INSERT_SC_','RL_INSERT_V50_')) or k=='RL_INSERT_FORCE_ABORT_N'}
    result['sfp_full_port_orientation']=os.environ.get('AIC_SFP_FULL_PORT_ORIENTATION','0')=='1'
    result['physical_controller_timebase']='ROS simulation time; overall action deadline remains wall time'
    result['terminal_protocol']='estimated geometry/wrist-force dwell; simulator scoring events ignored'
    rclpy.init(args=['--ros-args','-p','policy:=aic_model.insertion.InsertionPolicy','-p','use_sim_time:=true'])
    node=AicModel()
    sensor=deque(maxlen=300)
    node.create_subscription(WrenchStamped,'/fts_broadcaster/wrench',lambda m:sensor.append(m),qos_profile_sensor_data)
    executor=MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    worker=threading.Thread(target=executor.spin,daemon=True)
    worker.start()
    sim=lambda:node.get_clock().now().nanoseconds*1e-9
    start_sim=sim()
    def save_inputs(label):
        observation=node.observation_callable()
        if observation is None:return
        from aic_model.insertion.sfp_controller import _plug_views_from_observation
        views=_plug_views_from_observation(node._policy,observation)
        folder=out/label;folder.mkdir(exist_ok=True)
        geometry=[]
        for v in views:
            cv2.imwrite(str(folder/f'{v.camera_name}_raw.png'),v.image_bgr)
            geometry.append({'camera':v.camera_name,'stamp_s':v.stamp_s,'K':v.K.tolist(),
                             'T_base_from_camera':v.T_world_from_camera.tolist(),'frame_id':v.frame_id})
        (folder/'camera_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    try:
        node.trigger_configure()
        if not getattr(node,'_policy',None):raise RuntimeError('Original policy did not configure')
        node._policy._research_ablation=os.environ.get('AIC_RESEARCH_ABLATION','full')
        from aic_model.insertion.research_ablation import mode
        result['research_ablation']=mode(node._policy)
        node.trigger_activate()
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            observation=node.observation_callable()
            if observation is not None and all(getattr(observation,f'{c}_image').width for c in ['left','center','right']):break
            time.sleep(.1)
        else:raise TimeoutError('No complete camera observation')
        node._policy._wait_for_transform('base_link','gripper/tcp',timeout_sec=10)
        # The real challenge engine tares the native controller after attaching
        # a cable. The last-inch reset bypasses that engine lifecycle. Without
        # this service the original helper's 0.5 wrench-feedback gain reacts to
        # the static payload (about 18 N), producing free-space drift.
        client=node.create_client(Trigger,'/aic_controller/tare_force_torque_sensor')
        if not client.wait_for_service(timeout_sec=10):raise RuntimeError('Controller tare service unavailable')
        future=client.call_async(Trigger.Request());deadline=time.monotonic()+10
        while not future.done() and time.monotonic()<deadline:time.sleep(.01)
        response=future.result() if future.done() else None
        result['controller_tare']={'success':bool(response and response.success),'message':response.message if response else 'timeout'}
        if not result['controller_tare']['success']:raise RuntimeError('Native controller tare failed')
        calibration_path=ROOT/'.artifacts/kilted_handoff/static_load_calibration.json'
        calibration=json.loads(calibration_path.read_text())
        load=np.array(calibration['calibrated_force_base_N'])
        if load.shape!=(3,) or not np.isfinite(load).all():raise ValueError('Invalid force-support calibration')
        result['load_calibration_sha256']=hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        result['load_calibration_contains_object_or_target_pose']=False
        result['shared_static_load_support_base_N']=load.tolist()
        initial_tcp=np.array(node._policy._tcp()[0])
        def save_rim_inputs(views,position,rotation,kind='rim_inputs'):
            if kind not in ['rim_inputs','body_inputs']:raise ValueError('Unknown visual capture kind')
            root=out/kind;root.mkdir(exist_ok=True)
            folder=root/f'{len(list(root.iterdir())):03d}';folder.mkdir()
            records=[]
            for v in views:
                cv2.imwrite(str(folder/f'{v["camera"]}.png'),v['image'])
                if 'tool_mask' in v:cv2.imwrite(str(folder/f'{v["camera"]}_tool_mask.png'),v['tool_mask'])
                records.append({k:(value.tolist() if isinstance(value,np.ndarray) else value)
                                for k,value in v.items() if k not in ['image','tool_mask']})
            (folder/'inputs.json').write_text(json.dumps({'views':records,'initial_position':position.tolist(),
                                                        'initial_rotation':rotation.tolist()},indent=2)+'\n')
        node._policy._rim_input_recorder=save_rim_inputs
        save_inputs('perception_before')
        task=Task();task.id=run;task.plug_type=connector;task.cable_name=f'cable_{target}'
        task.plug_name=f'{connector}_plug_{target}'
        task.target_module_name=f'sc_port_{target}' if connector=='sc' else f'nic_card_mount_{target}'
        task.port_name=f'sc_port_{target}' if connector=='sc' else f'sfp_port_{slot}'
        result['task']={'connector':connector,'target_module_name':task.target_module_name,'port_name':task.port_name}
        start_sim=sim();result['configuration_wall_s']=time.monotonic()-started
        result['action_start_sim_s']=start_sim
        tare=None
        def force_value(message):
            return np.array([getattr(message.wrench.force,k) for k in 'xyz'])
        def guarded_hold():
            # The environment's static load calibration uses controller/wrench
            # observations. It contains no simulator tip or target pose.
            pos,quat=node._policy._tcp()
            hold=MotionUpdate();hold.header.frame_id='base_link';hold.header.stamp=node.get_clock().now().to_msg()
            hold.pose=Pose(position=Point(x=float(pos[0]),y=float(pos[1]),z=float(pos[2])),
                           orientation=Quaternion(w=float(quat[0]),x=float(quat[1]),y=float(quat[2]),z=float(quat[3])))
            hold.trajectory_generation_mode.mode=2
            hold.target_stiffness=np.diag([500.,500.,500.,50.,50.,50.]).ravel().tolist()
            hold.target_damping=np.diag([45.,45.,45.,15.,15.,15.]).ravel().tolist()
            support=Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]]).as_matrix().T@np.asarray(load)
            for k,value in zip('xyz',support):setattr(hold.feedforward_wrench_at_tip.force,k,float(value))
            node.move_robot(motion_update=hold)
        def move_robot(motion_update=None,joint_motion_update=None):
            nonlocal tare
            if not sensor:raise RuntimeError('Outer force guard has no wrist observations')
            latest=sensor[-1]
            age=sim()-(latest.header.stamp.sec+latest.header.stamp.nanosec*1e-9)
            if age>.1:guarded_hold();raise RuntimeError('Outer force guard wrist data stale')
            if tare is None:
                if len(sensor)<10:raise RuntimeError('Outer force guard lacks stationary tare samples')
                tare=np.median([force_value(m) for m in list(sensor)[-10:]],axis=0)
                result['outer_force_tare_N']=tare.tolist()
            force=float(np.linalg.norm(force_value(latest)-tare))
            result['peak_tared_force_N']=max(result['peak_tared_force_N'] or 0.,force)
            if force>result['outer_force_abort_N']:
                result['outer_force_abort_triggered']=True;guarded_hold()
                raise RuntimeError(f"Outer research force guard exceeded {result['outer_force_abort_N']:g} N")
            if np.linalg.norm(node._policy._tcp()[0]-initial_tcp)>.08:
                result['outer_workspace_abort_triggered']=True;guarded_hold()
                raise RuntimeError('Last-inch policy exceeded 80 mm TCP workspace')
            result['command_count']+=1
            if motion_update is not None:
                # Native controller applies ff + gain*(ff-sensed_tared).
                # Divide the additive support by (1+gain) to preserve the
                # calibrated BASE force while retaining original force gains.
                _,quat=node._policy._tcp()
                w,x,y,z=quat;R=Rotation.from_quat([x,y,z,w]).as_matrix()
                support=R.T@load/(1+np.asarray(motion_update.wrench_feedback_gains_at_tip[:3]))
                for k,value in zip('xyz',support):
                    setattr(motion_update.feedforward_wrench_at_tip.force,k,
                            float(getattr(motion_update.feedforward_wrench_at_tip.force,k)+value))
            result['trace'].append({'sim_s':sim(),'phase':'original_policy_motion_command',
                'target_tcp':None if motion_update is None else [getattr(motion_update.pose.position,k) for k in 'xyz'],
                'feedback_gains':None if motion_update is None else list(motion_update.wrench_feedback_gains_at_tip)})
            return node.move_robot(motion_update,joint_motion_update)
        def feedback(message):
            result['feedback'].append({'sim_s':sim(),'message':str(message)})
            print(str(message),flush=True)
        success=bool(node._policy.insert_cable(task,node.observation_callable,move_robot,feedback))
        result['policy_returned_success']=success
        result['reason']='command_and_dwell_completed' if success else 'original_policy_returned_failure'
        if result.get('outer_force_abort_triggered'):result['reason']='outer_research_force_abort'
        for kind in ['sc','sfp']:
            est=getattr(node._policy,f'_{kind}_plug_estimator',None)
            if est:
                reason=getattr(est,'last_failure_reason',None)
                result[f'{kind}_plug_failure']=reason
                result[f'{kind}_refinement_diagnostics']=getattr(est,'last_refinement_diagnostics',None)
                if not success and kind==connector and reason:result['reason']='visual_grasp_failed:'+reason
        result['visual_grasp_available']=getattr(node._policy,f'_{connector}_grasp_transform',None) is not None
        result['visual_reacquisition_events']=getattr(node._policy,'_visual_reacquisition_events',[])
        result['observable_terminal_diagnostic']=getattr(node._policy,'_observable_terminal_diagnostic',None)
        result['axial_contact_registration']=getattr(node._policy,'_axial_contact_registration',None)
        result['sfp_fixture_diagnostic']=getattr(node._policy,'_sfp_fixture_diagnostic',None)
        result['sfp_fixture_failure']=getattr(node._policy,'_sfp_fixture_failure',None)
        result['sfp_contact_datum']=getattr(node._policy,'_sfp_contact_datum',None)
        result['sc_rim_diagnostics']=getattr(node._policy,'_sc_rim_diagnostics',[])
        result['sc_body_diagnostic']=getattr(node._policy,'_sc_body_diagnostic',None)
        result['sc_body_attempts']=getattr(node._policy,'_sc_body_attempts',[])
        tracker=getattr(node._policy,'_lateral_tracking_compensator',None)
        result['lateral_tracking_compensation']=None if tracker is None else tracker.diagnostics()
        result['contact_grasp_registration']=getattr(node._policy,'_contact_grasp_registration',None)
        result['contact_entry_search']=getattr(node._policy,'_contact_entry_search',None)
        result['contact_registration_trace']=getattr(node._policy,'_contact_registration_trace',[])
        wrist_cache=getattr(node._policy,'_wrist_sensor_cache',None)
        result['independent_wrist_sensor']=None if wrist_cache is None else wrist_cache.diagnostics()
        save_inputs('perception_after')
    except Exception as exc:
        result['reason']=f'original_policy_exception:{type(exc).__name__}:{exc}'
        import traceback;traceback.print_exc()
    finally:
        result.update(wall_s=time.monotonic()-started,sim_s=sim()-start_sim)
        weights=ROOT/'aic_example_policies/aic_example_policies/ros/weights'
        result['weight_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.glob('*.pt')}
        result['explicit_runtime_weight_overrides']={}
        for variable in ['AIC_SC_POSE_WEIGHTS','AIC_SC_PLUG_POSE_WEIGHTS','AIC_SFP_PLUG_POSE_WEIGHTS','AIC_NIC_WEIGHTS']:
            path=Path(os.environ.get(variable,''))
            if path.is_file():result['explicit_runtime_weight_overrides'][variable]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (out/'policy.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'status.json').write_text(json.dumps({'active':False,'run':run}))
        print(json.dumps({k:v for k,v in result.items() if k not in ['trace','weight_sha256']},indent=2),flush=True)
        node._tf_listener.executor.shutdown()
        node._tf_listener.dedicated_listener_thread.join(timeout=2)
        executor.shutdown(timeout_sec=2);node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
