"""Run the generated last-inch scene headlessly and receive all actual cameras."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=150)
    a=p.parse_args()
    root=a.scene.resolve()
    if (root/'capture.json').exists():
        p.error('Existing captures must be preserved; use a fresh scene directory')
    os.environ['GZ_PARTITION']=f'aic_lastinch_{os.getpid()}'
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image as ImageMsg
    from gz.msgs10.scene_pb2 import Scene
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.msgs10.wrench_pb2 import Wrench
    from google.protobuf.json_format import MessageToDict
    from PIL import Image
    node=Node()
    records={name:[] for name in ('overview','board','target','left_camera','center_camera','right_camera')}
    lock=threading.Lock()
    callbacks=[]
    def make_callback(name):
        def receive(msg):
            with lock:
                if len(records[name])>=3: return
                path=root/'cameras'/name/f'{len(records[name]):03d}.png'
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.frombytes('RGB',(msg.width,msg.height),msg.data,'raw','RGB',msg.step).save(path)
                records[name].append({'file':str(path.relative_to(root)), 'width':msg.width,'height':msg.height,
                    'sim_time_s':msg.header.stamp.sec+msg.header.stamp.nsec/1e9,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return receive
    for name in records:
        cb=make_callback(name)
        callbacks.append(cb)
        topic=f'/{name}/image' if name.endswith('camera') else f'/lastinch/{name}/image'
        if not node.subscribe(ImageMsg,topic,cb): raise RuntimeError(topic)
    def scene_callback(msg):
        (root/'observed_scene.json').write_text(json.dumps(MessageToDict(msg),indent=2))
    node.subscribe(Scene,'/world/aic_lastinch/scene/info',scene_callback)
    robot_samples=[]
    def pose_callback(msg):
        robot_samples.append(MessageToDict(msg))
    node.subscribe(Pose_V,'/lastinch/robot/poses',pose_callback)
    trace_streams={name:[] for name in ('cable_poses','contacts','wrench')}
    def record_stream(name):
        def receive(msg): trace_streams[name].append(MessageToDict(msg))
        return receive
    for name,topic,kind in (('cable_poses','/lastinch/cable/poses',Pose_V),
                            ('contacts','/lastinch/contacts',Contacts),
                            ('wrench','/axia80_m20/wrench',Wrench)):
        cb=record_stream(name)
        callbacks.append(cb)
        node.subscribe(kind,topic,cb)
    env=os.environ.copy()
    env.update(EGL_PLATFORM='surfaceless',LIBGL_ALWAYS_SOFTWARE='1',QT_QPA_PLATFORM='offscreen')
    cmd=['gz','sim','-s','-r','--headless-rendering','-v','3',str(root/'lastinch.sdf')]
    start=time.monotonic()
    with (root/'gazebo.log').open('wb') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None and time.monotonic()-start<a.timeout:
            if all(len(rows)>=3 for rows in records.values()): break
            time.sleep(.1)
        if proc.poll() is None: proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    result={'command':cmd,'wall_time_s':time.monotonic()-start,'returncode':proc.returncode,
        'camera_frames':records,'all_six_cameras_received':all(len(r)>=3 for r in records.values()),
        'purpose':'last_inch_reset_integrity_not_policy_evaluation'}
    (root/'capture.json').write_text(json.dumps(result,indent=2))
    (root/'robot_poses.json').write_text(json.dumps(robot_samples))
    for name,rows in trace_streams.items():
        with (root/f'{name}.jsonl').open('w') as stream:
            for row in rows: stream.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='camera_frames'},indent=2))
    raise SystemExit(0 if result['all_six_cameras_received'] and proc.returncode==0 else 1)


if __name__=='__main__': main()
