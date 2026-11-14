"""Independent evaluator: actual cable pose, contact, fixture status and images."""
import gzip
from collections import Counter
import json
import os
import subprocess
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from ros_gz_interfaces.msg import Contacts
from geometry_msgs.msg import WrenchStamped
from rosidl_runtime_py.convert import message_to_ordereddict
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.contact_depth import penetration_m,resolve_contract,touching

OUT=Path('/research/.artifacts/kilted_handoff')

def main():
    run=os.environ.get('AIC_RESEARCH_RUN_ID','sc_known_grasp_001');path=OUT/run;path.mkdir(exist_ok=True)
    spec=json.loads((OUT/'scene.json').read_text());name=f"cable_{spec['reset']['target_index']}"
    convention=resolve_contract(spec)['depth_convention']
    connector=spec['connector'];tip_name='sc_tip_link' if connector=='sc' else 'sfp_tip_link'
    mouth=np.array(spec['reset']['mouth_world']);seat=np.array(spec['reset']['seat_world'])
    target_name=next(t['name'] for t in spec['targets'] if t['connector']==connector and t['index']==spec['reset']['target_index'] and t['port']==spec['reset']['port'])
    rclpy.init();node=Node('independent_seating_recorder');rows=[];contacts=[];grasp=[];cameras={};contact_seen=[];forces=[];audit=[];raw_forces=[]
    node.create_subscription(String,'/lastinch/contact_audit_status',lambda m:audit.append(json.loads(m.data)),100)
    node.create_subscription(WrenchStamped,'/fts_broadcaster/wrench',lambda m:forces.append(message_to_ordereddict(m)),qos_profile_sensor_data)
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    node.create_subscription(WrenchStamped,'/lastinch/wrist_raw',lambda m:raw_forces.append(message_to_ordereddict(m)),
        QoSProfile(depth=10000,reliability=ReliabilityPolicy.RELIABLE))
    def contact(m):
        data=message_to_ordereddict(m);contacts.append(data);contact_seen.append(time.monotonic())
    topics=['/lastinch/evaluator_contacts'] if 'contact_audit_source' in spec else spec.get('contact_diagnostic_ros_topics',['/research/contact_probe'])
    for topic in topics:node.create_subscription(Contacts,topic,contact,100)
    node.create_subscription(String,'/lastinch/grasp_state',lambda m:grasp.append(m.data),10)
    def pose(m):
        # Do not label an unobserved subscription state as an active fixture.
        # Bootstrap/hold is already complete; begin the evaluable pose stream
        # only once actual grasp and physics-health observations have arrived.
        if not grasp or ('contact_audit_source' in spec and not audit):return
        frames={}
        for t in m.transforms:
            v=t.transform;M=np.eye(4);M[:3,3]=[v.translation.x,v.translation.y,v.translation.z]
            M[:3,:3]=Rotation.from_quat([v.rotation.x,v.rotation.y,v.rotation.z,v.rotation.w]).as_matrix()
            frames[t.child_frame_id]=M
        if name not in frames or f'{name}/{tip_name}' not in frames:return
        T=frames[name]@frames[f'{name}/{tip_name}'];relative=np.linalg.inv(mouth)@T
        stamp=m.transforms[0].header.stamp;t=stamp.sec+stamp.nanosec/1e9
        recent=[c for msg in contacts[-len(topics)*3:] for c in msg['contacts']
            if abs((msg['header']['stamp']['sec']+msg['header']['stamp']['nanosec']/1e9)-t)<.03]
        requested=[c for c in recent if any(target_name.split('/')[0]+'::' in c[k]['name'] for k in ('collision1','collision2'))
            and any(f'{name}/{tip_name}' in c[k]['name'] or f"{name}::{'sc_plug_link' if connector=='sc' else 'sfp_module_link'}::" in c[k]['name'] for k in ('collision1','collision2'))]
        row={'sim_time_s':t,'depth_m':float(relative[2,3]),'lateral_error_m':float(np.linalg.norm(relative[:2,3])),
            'angular_error_deg':float(np.linalg.norm(Rotation.from_matrix(relative[:3,:3]).as_rotvec())*180/np.pi),
            'max_penetration_m':max([0.]+[penetration_m(d,convention) for c in recent for d in c['depths']]),
            'requested_contact':any(touching(c,convention) for c in requested),'contact_messages_received':len(contacts),
            'fixture_active':grasp[-1]!='grasp_attached_startup_fixture_removed','fixture_observed':True,'tip_world':T.tolist()}
        if not rows or t>rows[-1]['sim_time_s']:rows.append(row)
    node.create_subscription(TFMessage,f'/model/{name}/pose',pose,100)
    for camera in ('overview','target','board','left_camera','center_camera','right_camera'):
        def receive(m,camera=camera):
            stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
            if camera in cameras and stamp-cameras[camera]<1:return
            image=np.frombuffer(m.data,np.uint8).reshape(m.height,m.step)[:,:m.width*3].reshape(m.height,m.width,3)
            if m.encoding=='rgb8':image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path/f'{camera}_{stamp:.3f}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,90]);cameras[camera]=stamp
        topic=f'/{camera}/image' if camera.endswith('camera') else f'/lastinch/{camera}/image'
        node.create_subscription(Image,topic,receive,qos_profile_sensor_data)
    from experiments.recording_tail import RecordingTail
    tail=RecordingTail();tail_started=None;tail_complete=False;tail_seen=0
    start=time.monotonic();started=False;next_status_poll=0
    while time.monotonic()-start<1550:
        rclpy.spin_once(node,timeout_sec=.02)
        if time.monotonic()<next_status_poll:continue
        next_status_poll=time.monotonic()+.2
        try:active=json.loads((path/'status.json').read_text())['active']
        except (FileNotFoundError,json.JSONDecodeError):continue
        if active:started=True
        if started and not active:
            # Collect the same fixed coverage-only tail on success or failure.
            # No depth/contact truth selects a favorable stopping window.
            if tail_started is None:
                tail_started=time.monotonic();tail_seen=len(rows)
            for row in rows[tail_seen:]:tail_complete=tail.observe(row['sim_time_s']) or tail_complete
            tail_seen=len(rows)
            if not tail_complete and time.monotonic()-tail_started<45:continue
            # Freeze the physics, then drain transport. Otherwise faster pose
            # delivery can leave an unobserved contact tail at shutdown.
            subprocess.run(['gz','service','-s','/world/aic_lastinch/control',
                '--reqtype','gz.msgs.WorldControl','--reptype','gz.msgs.Boolean',
                '--timeout','5000','--req','pause: true'],check=True,capture_output=True,timeout=7)
            drain=time.monotonic()+3
            while time.monotonic()<drain:rclpy.spin_once(node,timeout_sec=.02)
            break
    with gzip.open(path/'truth_trace.json.gz','wt') as f:json.dump(rows,f)
    with gzip.open(path/'contacts.json.gz','wt') as f:json.dump(contacts,f)
    with gzip.open(path/'wrench.json.gz','wt') as f:json.dump(forces,f)
    with gzip.open(path/'wrench_raw.json.gz','wt') as f:json.dump(raw_forces,f)
    with gzip.open(path/'contact_audit_status.json.gz','wt') as f:json.dump(audit,f)
    pairs=Counter((c['collision1']['name'],c['collision2']['name']) for msg in contacts for c in msg['contacts'])
    result={'run':run,'samples':len(rows),'contact_messages':len(contacts),'fixture_states':sorted(set(grasp)),
        'raw_wrist_subscription_reliability':'RELIABLE','raw_wrist_subscription_depth':10000,
        'nominal_depth_m':float((np.linalg.inv(mouth)@seat)[2,3]),'target_name':target_name,
        'terminal':rows[-1] if rows else None,'max_penetration_m':max([0.]+[r['max_penetration_m'] for r in rows]),
        'requested_contact_samples':sum(r['requested_contact'] for r in rows),'camera_last_stamps':cameras}
    result['post_policy_recording_coverage']={**tail.diagnostics(),'completed':tail_complete,'wall_budget_s':45}
    result['contact_pair_counts']=[{'pair':list(k),'count':v} for k,v in pairs.most_common()]
    (path/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
