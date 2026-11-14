"""Synchronized, read-only image/TF/actual simulated geometry diagnostic.

Truth is consumed here only, never passed to a policy or robot command.
"""
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage
from rosidl_runtime_py.convert import message_to_ordereddict
from aic_model.aic_model import AicModel
from aic_model.insertion.sc_controller import configure_sc_plug_pose
from aic_model.insertion.sfp_controller import _plug_views_from_observation
from aic_model.insertion.sfp_plug_pose import fuse_multiview_keypoints
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M as POINTS

OUT=Path('/research/.artifacts/kilted_handoff/projection_audit')

def matrix(transform):
    t=transform.transform if hasattr(transform,'transform') else transform
    M=np.eye(4);M[:3,3]=[t.translation.x,t.translation.y,t.translation.z]
    M[:3,:3]=Rotation.from_quat([t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w]).as_matrix()
    return M

def stamp(msg):
    h=msg.transforms[0].header if hasattr(msg,'transforms') else msg.header
    return h.stamp.sec+h.stamp.nanosec*1e-9

def main():
    OUT.mkdir(exist_ok=True)
    rclpy.init(args=['--ros-args','-p','policy:=aic_model.insertion.InsertionPolicy','-p','use_sim_time:=true'])
    node=AicModel();poses=[];robot=[]
    node.create_subscription(TFMessage,'/model/cable_2/pose',lambda m:poses.append(m),100)
    node.create_subscription(TFMessage,'/model/ur5e/pose',lambda m:robot.append(m),100)
    executor=MultiThreadedExecutor();executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    result={'purpose':'GT_diagnostic_only','commands_sent':0}
    try:
        node.trigger_configure();policy=node._policy
        if not configure_sc_plug_pose(policy):raise RuntimeError('No detector')
        end=time.monotonic()+25
        while time.monotonic()<end:
            obs=node.observation_callable()
            if obs is not None and poses and robot:
                images=[getattr(obs,f'{c}_image') for c in ('left','center','right')]
                times=[stamp(m) for m in images]
                if max(times)-min(times)<.001 and node.get_clock().now().nanoseconds/1e9-max(times)<.35:
                    time.sleep(.2);break
            time.sleep(.05)
        else:raise TimeoutError('No fresh synchronized observation and GT')
        views=_plug_views_from_observation(policy,obs)
        ds=policy._sc_plug_estimator.detect_views(views)
        base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',-np.pi).as_matrix();base[:3,3]=[-.2,.2,0]
        rows=[]
        for view in views:
            pose=min(poses,key=lambda m:abs(stamp(m)-view.stamp_s))
            robot_pose=min(robot,key=lambda m:abs(stamp(m)-view.stamp_s))
            frames={t.child_frame_id:matrix(t) for t in pose.transforms}
            tip=frames['cable_2']@frames['cable_2/sc_tip_link']
            exact=node._tf_buffer.lookup_transform(f'{view.camera_name}/optical','base_link',
                Time(seconds=view.stamp_s),Duration(seconds=2))
            cam_from_base=matrix(exact)
            info=getattr(obs,view.camera_name.replace('_camera','')+'_camera_info')
            row={'camera':view.camera_name,'stamp_s':view.stamp_s,'truth_delta_s':stamp(pose)-view.stamp_s,
                'robot_delta_s':stamp(robot_pose)-view.stamp_s,'K':view.K.tolist(),
                'camera_info':message_to_ordereddict(info),'T_camera_from_base_exact':cam_from_base.tolist(),
                'T_base_from_camera_latest':view.T_world_from_camera.tolist(),'tip_world':tip.tolist(),
                'robot_pose':message_to_ordereddict(robot_pose)}
            X=cam_from_base@np.linalg.inv(base)@tip@np.column_stack([POINTS,np.ones(8)]).T
            projected=view.K@X[:3];uv=(projected[:2]/projected[2]).T
            row['projected_px']=uv.tolist()
            d=next((d for d in ds if d.camera_name==view.camera_name),None)
            image=view.image_bgr.copy();cv2.imwrite(str(OUT/f'{view.camera_name}_raw.png'),image)
            for i,point in enumerate(uv):
                p=tuple(np.rint(point).astype(int));cv2.circle(image,p,4,(0,255,0),-1)
                cv2.putText(image,str(i),p,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
            if d is not None:
                row.update(predicted_px=d.keypoints_px.tolist(),confidence=d.keypoint_confidences.tolist(),
                    box_confidence=d.box_confidence,mean_gt_error_px=float(np.linalg.norm(d.keypoints_px-uv,axis=1).mean()))
                for i,point in enumerate(d.keypoints_px):
                    p=tuple(np.rint(point).astype(int));cv2.circle(image,p,5,(0,0,255),1)
                    cv2.putText(image,str(i),p,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            cv2.imwrite(str(OUT/f'{view.camera_name}_projection.png'),image)
            rows.append(row)
        result['views']=rows
        for crop,pad in ((False,6),(True,3),(True,6),(True,12)):
            est=policy._sc_plug_estimator;est.crop_refine=crop;est.crop_pad_scale=pad
            detections=est.detect_views(views)
            try:
                p,R,rmse,reproj,count,conf=fuse_multiview_keypoints(views,detections,local_keypoints_m=POINTS)
                result.setdefault('fusion',[]).append({'crop':crop,'pad':pad,'position':p.tolist(),
                    'rotation':R.tolist(),'shape_mm':rmse*1000,'reprojection_px':reproj})
            except Exception as exc:result.setdefault('fusion',[]).append({'crop':crop,'pad':pad,'error':str(exc)})
        (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'views':[{k:r[k] for k in ('camera','stamp_s','truth_delta_s','mean_gt_error_px') if k in r} for r in rows],
            'fusion':result['fusion']},indent=2))
    finally:
        node._tf_listener.executor.shutdown();node._tf_listener.dedicated_listener_thread.join(timeout=2)
        executor.shutdown();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
