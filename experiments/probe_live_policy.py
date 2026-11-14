"""Read-only deployed perception on live ROS images; no robot command or truth."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from aic_model.aic_model import AicModel
from aic_task_interfaces.msg import Task
from aic_model.insertion.sc_controller import perceive_sc_port_pose, configure_sc_plug_pose, prime_sc_plug_pose


def main():
    rclpy.init(args=['--ros-args','-p','policy:=aic_model.insertion.InsertionPolicy','-p','use_sim_time:=true'])
    node=AicModel()
    executor=MultiThreadedExecutor();executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    result={'purpose':'read_only_live_policy_perception','ground_truth_used':False,'robot_commands_sent':False}
    try:
        configured=node.trigger_configure()
        result['configure_transition']=str(configured)
        if not hasattr(node,'_policy') or node._policy is None:
            raise RuntimeError('Policy configuration failed')
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            obs=node.observation_callable()
            if obs is not None and obs.center_image.width and obs.left_image.width and obs.right_image.width:
                break
            time.sleep(.2)
        else:raise TimeoutError('No complete live observation')
        policy=node._policy
        views=policy._build_views(obs)
        result['views']={name:{'shape':list(bgr.shape),'intrinsics':K.tolist(),
            'image_sha256':hashlib.sha256(bgr.tobytes()).hexdigest()}
            for name,(bgr,K,T) in views.items()}
        result['sc_plug_estimator_configured']=bool(configure_sc_plug_pose(policy))
        def reject_motion(*args,**kwargs):
            raise RuntimeError('Read-only probe refused a robot command')
        result['visual_grasp_primed']=bool(prime_sc_plug_pose(policy,node.observation_callable,reject_motion))
        obs=node.observation_callable()
        from aic_model.insertion.sfp_controller import _plug_views_from_observation
        plug_views=_plug_views_from_observation(policy,obs)
        detections=policy._sc_plug_estimator.detect_views(plug_views)
        estimator=policy._sc_plug_estimator
        from aic_model.insertion.sfp_plug_pose import fuse_multiview_keypoints
        comparisons=[]
        for refine,pad in ((False,6.),(True,3.),(True,6.),(True,12.)):
            estimator.crop_refine=refine;estimator.crop_pad_scale=pad
            ds=estimator.detect_views(plug_views)
            entry={'crop_refine':refine,'pad':pad,'cameras':[d.camera_name for d in ds]}
            try:
                p,R,rmse,reproj,count,conf=fuse_multiview_keypoints(plug_views,ds,
                    local_keypoints_m=estimator.local_keypoints_m,
                    min_keypoint_confidence=estimator.min_keypoint_confidence)
                entry.update(position_base=p.tolist(),rotation_base=R.tolist(),
                             shape_rmse_mm=rmse*1000,reprojection_px=reproj)
            except Exception as exc:entry['failure']=str(exc)
            comparisons.append(entry)
        result['same_frame_pilot_crop_comparison']=comparisons
        estimator.crop_refine=True;estimator.crop_pad_scale=6.
        result['plug_detections']=[{'camera':d.camera_name,'keypoints_px':d.keypoints_px.tolist(),
            'confidence':d.box_confidence} for d in detections]
        folder=Path('/research/.artifacts/kilted_handoff/policy_images');folder.mkdir(exist_ok=True)
        for view in plug_views:
            image=view.image_bgr.copy()
            cv2.imwrite(str(folder/f'{view.camera_name}_raw.png'),image)
            for det in detections:
                if det.camera_name!=view.camera_name:continue
                for i,point in enumerate(det.keypoints_px):
                    uv=tuple(map(int,point));cv2.circle(image,uv,4,(0,0,255),-1)
                    cv2.putText(image,str(i),uv,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imwrite(str(folder/f'{view.camera_name}_detections.png'),image)
        (folder/'camera_geometry.json').write_text(json.dumps([
            {'name':v.camera_name,'K':v.K.tolist(),'T_base_from_camera':v.T_world_from_camera.tolist(),
             'stamp_s':v.stamp_s,'frame_id':v.frame_id} for v in plug_views],indent=2)+'\n')
        task=Task();task.plug_type='sc';task.target_module_name='sc_port_2';task.port_name='sc_port_2'
        pose=perceive_sc_port_pose(policy,task,obs)
        result['sc_port_pose']=None if pose is None else [np.asarray(value).tolist() for value in pose]
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    finally:
        path=Path('/research/.artifacts/kilted_handoff/live_policy_probe_primed.json')
        path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        node._tf_listener.executor.shutdown()
        node._tf_listener.dedicated_listener_thread.join(timeout=2)
        executor.shutdown();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
