"""Physical-mouth refinement from a nominal handoff, pixels and timestamped TF.

No simulator state. The nominal plane is supplied by the upstream handoff
contract; dark bores locate the physical opening within a bounded search ROI.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from aic_model.insertion.sc_visual_alignment import detect_sc_duplex_opening,fuse_sc_opening_hits,ray_to_plane
from aic_model.insertion.visual_gap import detect_dark_port_opening
from experiments.diagnostic_json import compact_diagnostics

def refine(observation,buffer,origin,Rp,connector,out):
    hits=[];diagnostics=[]
    quads=[]
    centers=(.00635,-.00635) if connector=='sc' else (0.,)
    width=.00971 if connector=='sc' else .0137
    height=.0081 if connector=='sc' else .0086
    for x in centers:
        local=np.array([[x+width/2,height/2,0],[x-width/2,height/2,0],
            [x-width/2,-height/2,0],[x+width/2,-height/2,0]])
        quads.append(origin+(Rp@local.T).T)
    points=np.array(quads).reshape(-1,3)
    for name in ('left','center','right'):
        msg=getattr(observation,f'{name}_image');info=getattr(observation,f'{name}_camera_info')
        diag={'camera':name};diagnostics.append(diag)
        try:
            tf=buffer.lookup_transform(f'{name}_camera/optical','base_link',Time.from_msg(msg.header.stamp))
            tr=tf.transform;T=np.eye(4);T[:3,3]=[tr.translation.x,tr.translation.y,tr.translation.z]
            T[:3,:3]=Rotation.from_quat([tr.rotation.x,tr.rotation.y,tr.rotation.z,tr.rotation.w]).as_matrix()
            K=np.array(info.k).reshape(3,3)
            xyz=(T@np.column_stack([points,np.ones(len(points))]).T)[:3]
            if np.any(xyz[2]<=0):raise ValueError('ROI behind camera')
            projected=K@xyz;uv=(projected[:2]/projected[2]).T.reshape(len(quads),4,2)
            image=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
            image=image.copy() if msg.encoding=='bgr8' else cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out/f'vision_{name}_raw.png'),image)
            if connector=='sc':det=detect_sc_duplex_opening(image,uv,diagnostics=diag)
            else:det=detect_dark_port_opening(image,uv[0],diagnostics=diag)
            if det is None:continue
            point=ray_to_plane(det.center_uv,K,T,plane_point=origin,plane_normal=Rp[:,2])
            if point is None:continue
            hits.append({'camera':name,'plane_point':point})
            diag.update(center_uv=det.center_uv.tolist(),plane_point=point.tolist())
            cv2.polylines(image,np.rint(uv).astype(np.int32),True,(0,255,0),1)
            cv2.circle(image,tuple(np.rint(det.center_uv).astype(int)),4,(0,0,255),-1)
            cv2.imwrite(str(out/f'vision_{name}.png'),image)
        except Exception as exc:diag['error']=f'{type(exc).__name__}: {exc}'
        finally:
            diag['camera']=name
            compact=compact_diagnostics(diag);diag.clear();diag.update(compact)
    estimate=fuse_sc_opening_hits(hits,origin_port_pos=origin,Rp=Rp,
        max_view_disagreement_m=.0006,max_total_offset_m=.002,allow_single_view=False)
    return (None if estimate is None else estimate.point_world),diagnostics
