"""Read-only evaluator witness: synchronized full cable poses and contacts.

May attach to an already running recorder bridge. Never publishes commands,
changes the world, or supplies observations to an insertion policy. The capture
can cover only the latter part of a live trial; metadata makes that explicit.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import String
from rosidl_runtime_py.convert import message_to_ordereddict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--scene-dir',type=Path,default=Path('/research/.artifacts/kilted_handoff'))
    parser.add_argument('--max-wall-s',type=float,default=1000)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    scene_path=args.scene_dir/'scene.json';scene=json.loads(scene_path.read_text())
    cable=f"cable_{scene['reset']['target_index']}"
    metadata={'purpose':'read_only_evaluator_geometry_witness','run':args.run,
        'coverage':'late attachment; not the entire trial','policy_input':False,
        'scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pose_topic':f'/model/{cable}/pose','contact_topic':'/lastinch/evaluator_contacts',
        'pose_packets':0,'contact_packets':0,'audit_packets':0,'first_pose_sim_s':None,'last_pose_sim_s':None}
    rclpy.init();node=Node('independent_contact_geometry_witness')
    start=time.monotonic();last_save=0
    def save():
        metadata['elapsed_wall_s']=time.monotonic()-start
        (args.out/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with gzip.open(args.out/'poses.jsonl.gz','wt') as poses,gzip.open(args.out/'contacts.jsonl.gz','wt') as contacts,gzip.open(args.out/'audit.jsonl.gz','wt') as audits:
        def pose(message):
            if not message.transforms:return
            stamp=message.transforms[0].header.stamp;t=stamp.sec+stamp.nanosec/1e9
            poses.write(json.dumps(message_to_ordereddict(message),separators=(',',':'))+'\n')
            metadata['pose_packets']+=1
            if metadata['first_pose_sim_s'] is None:metadata['first_pose_sim_s']=t
            metadata['last_pose_sim_s']=t
        def contact(message):
            contacts.write(json.dumps(message_to_ordereddict(message),separators=(',',':'))+'\n')
            metadata['contact_packets']+=1
        def audit(message):
            audits.write(json.dumps(json.loads(message.data),separators=(',',':'))+'\n')
            metadata['audit_packets']+=1
        node.create_subscription(TFMessage,metadata['pose_topic'],pose,100)
        node.create_subscription(Contacts,metadata['contact_topic'],contact,100)
        node.create_subscription(String,'/lastinch/contact_audit_status',audit,100)
        save()
        try:
            while time.monotonic()-start<args.max_wall_s:
                rclpy.spin_once(node,timeout_sec=.02)
                if time.monotonic()-last_save<1:continue
                last_save=time.monotonic();poses.flush();contacts.flush();audits.flush();save()
                status_path=args.scene_dir/args.run/'status.json'
                if status_path.exists() and not json.loads(status_path.read_text())['active']:
                    metadata['stop_reason']='observed_policy_finished';break
            else:metadata['stop_reason']='witness_wall_timeout'
        finally:save();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
