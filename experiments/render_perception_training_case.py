"""Headless STATIC camera dataset capture; never counted as an insertion trial.

Original source meshes, calibrated wrist cameras and fixed robot FK are used.
Only training labels may access scene poses. Runtime learned policies do not
import this file. Split by entire randomized scene, not individual images.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'aic_model'),str(ROOT/'aic_example_policies')]
from experiments.lastinch_geometry import robot_fk
from aic_model.insertion.sc_plug_pose_geometry import SC_PLUG_LOCAL_KEYPOINTS_M,project_keypoints,padded_bbox,visibility_flags,format_yolo_pose_label as plug_label
from aic_example_policies.ros.sc_mouth_pose_geometry import LOCAL_SC_FRONT_MOUTH_KPS_M,format_yolo_pose_label as mouth_label


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--split',choices=['train','val','test'],required=True)
    ap.add_argument('--label-connector',choices=['sc','sfp'],default='sc')
    a=ap.parse_args();scene=a.scene.resolve();dataset=a.dataset.resolve();spec=json.loads((scene/'scene.json').read_text())
    if (scene/'training_capture.json').exists():raise RuntimeError('Preserve existing case; use a new identity')
    world=ET.parse(scene/'lastinch.sdf')
    # Generate the same source cable initialization as physical trials, then
    # freeze it for image-label collection. Do not substitute the cable's
    # authored rest shape, which can put the distal connector across the target.
    for model in world.getroot().find('world').findall('model'):
        static=model.find('static')
        if static is None:static=ET.SubElement(model,'static')
        static.text='true'
        for item in list(model):
            if item.tag in ('joint','plugin'):model.remove(item)
        links=model.findall('link')
        if links:
            for index,link in enumerate(links[1:]):
                joint=ET.SubElement(model,'joint',name=f'training_fixed_{index}',type='fixed')
                ET.SubElement(joint,'parent').text=links[0].get('name')
                ET.SubElement(joint,'child').text=link.get('name')
    for model in list(world.getroot().find('world').findall('model')):
        if model.get('name','').endswith('_camera') or model.get('name','') in ('overview_camera','board_camera','target_camera'):
            # External cameras are unnecessary for neural training; wrist cameras
            # are sensors inside the original robot model and remain intact.
            world.getroot().find('world').remove(model)
    for sensor in list(world.getroot().iter('sensor')):
        if sensor.get('type')=='camera':
            rate=sensor.find('update_rate')
            if rate is not None:rate.text='2'
    missing=[]
    for uri in world.getroot().iter('uri'):
        uri.text=uri.text.replace('/mnt/c/Users/anshu/College/Intrinsic/aic/aic',str(ROOT))
        if uri.text.startswith('/') and not Path(uri.text).exists():missing.append(uri.text)
    if missing:raise RuntimeError(f'Actual source asset files missing: {missing[:4]}')
    path=scene/'training_world.sdf';world.write(path,encoding='utf-8',xml_declaration=True)
    names=['left_camera','center_camera','right_camera']
    bridge=[]
    for name in names:
        for suffix,kind in [('image','Image'),('camera_info','CameraInfo')]:
            bridge.append({'ros_topic_name':f'/{name}/{suffix}','gz_topic_name':f'/{name}/{suffix}',
                'ros_type_name':f'sensor_msgs/msg/{kind}','gz_type_name':f'gz.msgs.{kind}',
                'direction':'GZ_TO_ROS','lazy':False})
    import yaml
    (scene/'training_bridge.yaml').write_text(yaml.safe_dump(bridge))
    os.environ.update(ROS_DOMAIN_ID='48',GZ_PARTITION='research_perception_training',LIBGL_ALWAYS_SOFTWARE='1',DISPLAY=':100')
    rclpy.init();node=Node('static_training_capture');images={};infos={}
    for name in names:
        node.create_subscription(Image,f'/{name}/image',lambda m,n=name:images.setdefault(n,m),qos_profile_sensor_data)
        node.create_subscription(CameraInfo,f'/{name}/camera_info',lambda m,n=name:infos.setdefault(n,m),qos_profile_sensor_data)
    processes=[];files=[];started=time.monotonic()
    try:
        for command,log in [(['Xvfb',':100','-screen','0','1280x1024x24','-nolisten','tcp'],'xvfb.log'),
                            (['ros2','run','ros_gz_bridge','parameter_bridge','--ros-args','-p',f'config_file:={scene/"training_bridge.yaml"}'],'bridge.log'),
                            (['gz','sim','-s','-r','--headless-rendering','-v','2',str(path)],'render.log')]:
            stream=(scene/log).open('wb');files.append(stream);processes.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+90
        while time.monotonic()<deadline and (len(images)<3 or len(infos)<3):
            if processes[-1].poll() is not None:raise RuntimeError('Gazebo exited; preserve and inspect render.log')
            rclpy.spin_once(node,timeout_sec=.1)
        if len(images)<3 or len(infos)<3:raise RuntimeError(f'Missing actual camera streams: images={list(images)}, info={list(infos)}')
        blue_counts={}
        for name,msg in images.items():
            raw=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
            bgr=raw if msg.encoding=='bgr8' else cv2.cvtColor(raw,cv2.COLOR_RGB2BGR)
            mask=cv2.inRange(cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV),np.array([90,80,60]),np.array([130,255,255]))
            blue_counts[name]=int(np.count_nonzero(mask))
        if a.label_connector=='sc' and sum(count>=300 for count in blue_counts.values())<2:
            raise RuntimeError(f'Camera integrity failed: expected SC mesh color absent: {blue_counts}')
        robot=ET.parse(ROOT/'.artifacts/lastinch_resources/ur5e.urdf').getroot();frames=robot_fk(robot,spec['robot_q'])
        record={'purpose':'static rendered TRAINING data, no physical insertion result','case_id':scene.name,
                'label_connector':a.label_connector,
                'scene_seed':spec['seed'],'split':a.split,'grasp':spec['reset']['grasp_provenance'],
                'camera_integrity_blue_counts':blue_counts,'cameras':{}}
        for name in names:
            msg=images[name];K=np.array(infos[name].k).reshape(3,3)
            raw=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
            bgr=raw if msg.encoding=='bgr8' else cv2.cvtColor(raw,cv2.COLOR_RGB2BGR)
            Tcam=np.linalg.inv(frames[f'{name}/optical'])
            label_sets=[
                ('sc_plug',SC_PLUG_LOCAL_KEYPOINTS_M,[np.array(spec['reset']['tip_world'])],plug_label),
                ('sc_mouth',LOCAL_SC_FRONT_MOUTH_KPS_M,[np.array(t['mouth']) for t in spec['targets'] if t['connector']=='sc'],mouth_label)]
            if a.label_connector=='sfp':
                from aic_model.insertion.sfp_plug_pose_geometry import SFP_PLUG_LOCAL_KEYPOINTS_M
                corners=np.array([[.00685,.0043,0],[-.00685,.0043,0],[-.00685,-.0043,0],[.00685,-.0043,0]])
                nic_points=[]
                for index in range(5):
                    ports=sorted([t for t in spec['targets'] if t['connector']=='sfp' and t['index']==index],key=lambda t:t['port'])
                    if len(ports)!=2:raise RuntimeError('Expected both source SFP slots for each NIC')
                    nic_points.append(np.concatenate([(np.array(t['mouth'])[:3,:3]@corners.T).T+np.array(t['mouth'])[:3,3] for t in ports]))
                # Identity poses because NIC labels already use world coordinates.
                label_sets=[('sfp_plug',SFP_PLUG_LOCAL_KEYPOINTS_M,[np.array(spec['reset']['tip_world'])],plug_label)]
                label_sets += [('sfp_nic',points,[np.eye(4)],plug_label) for points in nic_points]
            all_labels={}
            for kind,local,poses,formatter in label_sets:
                labels=[]
                for pose in poses:
                    pixels,front=project_keypoints(local,Tcam@pose,K)
                    flags=visibility_flags(pixels,front,msg.width,msg.height)
                    if np.count_nonzero(flags==2)<(4 if kind=='sfp_nic' else len(local)-1):continue
                    box=padded_bbox(pixels,front,msg.width,msg.height,padding=.28)
                    if box is None:continue
                    if kind=='sc_mouth':
                        # Do not teach a positive mouth detection through an
                        # entirely occluding plug/gripper. Label pose uses GT;
                        # this visibility screen uses only the rendered image.
                        x0,y0,x1,y1=np.rint(box).astype(int)
                        roi=bgr[max(0,y0):min(msg.height,y1+1),max(0,x0):min(msg.width,x1+1)]
                        blue=cv2.inRange(cv2.cvtColor(roi,cv2.COLOR_BGR2HSV),np.array([90,80,60]),np.array([130,255,255]))
                        if np.count_nonzero(blue)<max(15,blue.size*.08):continue
                    labels.append(formatter(box,pixels,flags,msg.width,msg.height))
                all_labels.setdefault(kind,[]).extend(labels)
            for kind,labels in all_labels.items():
                stem=f'{scene.name}_{name}'
                image_path=dataset/kind/'images'/a.split/f'{stem}.png';image_path.parent.mkdir(parents=True,exist_ok=True)
                label_path=dataset/kind/'labels'/a.split/f'{stem}.txt';label_path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(image_path),bgr);label_path.write_text('\n'.join(labels)+'\n')
            record['cameras'][name]={'K':K.tolist(),'T_world_from_camera':frames[f'{name}/optical'].tolist(),
                                    'image_stamp_s':msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                                    'width':msg.width,'height':msg.height,'encoding':msg.encoding,
                                    'pixel_std':float(np.std(bgr)),
                                    'labels_per_kind':{kind:len(labels) for kind,labels in all_labels.items()}}
            if np.std(bgr)<8:raise RuntimeError(f'Uninformative headless RGB stream: {name}')
        record.update(wall_s=time.monotonic()-started,world_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (scene/'training_capture.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'case':scene.name,'split':a.split,'cameras':len(images),'wall_s':record['wall_s']}))
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                # Pixels/calibration have already been copied. Avoid the
                # reproducible Ogre teardown heap failure in the isolated RGB
                # renderer; record deliberate termination instead of claiming
                # clean simulator shutdown. This is never a physics trial.
                if proc is processes[-1]:proc.kill()
                else:proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for stream in files:stream.close()
        node.destroy_node();rclpy.shutdown()
        if (scene/'training_capture.json').exists():
            record=json.loads((scene/'training_capture.json').read_text())
            record['process_returncodes_after_cleanup']=[proc.returncode for proc in processes]
            record['renderer_stop_protocol']='explicit isolated renderer termination after complete camera capture; avoids observed Ogre teardown heap error'
            record['renderer_teardown_clean']=processes[-1].returncode in (0,-2)
            (scene/'training_capture.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
