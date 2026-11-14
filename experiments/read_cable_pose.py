"""Evaluator-only transport bridge sample of actual cable/link transforms."""
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rosidl_runtime_py.convert import message_to_ordereddict

rclpy.init();node=Node('evaluator_cable_pose_sample');samples=[]
node.create_subscription(TFMessage,'/model/cable_2/pose',lambda m:samples.append(message_to_ordereddict(m)),10)
start=time.monotonic()
while not samples and time.monotonic()-start<15:rclpy.spin_once(node,timeout_sec=.2)
Path('/research/.artifacts/kilted_handoff/cable_transforms.json').write_text(json.dumps(samples,indent=2)+'\n')
print('Transforms received:',len(samples[0]['transforms']) if samples else 0)
node.destroy_node();rclpy.shutdown()
