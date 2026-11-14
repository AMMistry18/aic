"""Read-only live delivery check; final qualification uses the complete recording."""
import json
import time
from pathlib import Path
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.qos import QoSProfile,ReliabilityPolicy

rclpy.init();node=rclpy.create_node('raw_wrist_delivery_diagnostic');messages=[]
node.create_subscription(WrenchStamped,'/lastinch/wrist_raw',messages.append,
    QoSProfile(depth=1000,reliability=ReliabilityPolicy.BEST_EFFORT))
deadline=time.monotonic()+30
while time.monotonic()<deadline and len(messages)<2000:rclpy.spin_once(node,timeout_sec=.1)
t=np.array([m.header.stamp.sec+m.header.stamp.nanosec/1e9 for m in messages]);dt=np.diff(t)
result=dict(samples=len(messages),mean_hz=None if len(t)<2 else float((len(t)-1)/(t[-1]-t[0])),
    maximum_gap_s=None if not len(dt) else float(dt.max()),frames=sorted({m.header.frame_id for m in messages}),
    complete_trial_audit=False,ground_truth_inputs=False)
Path('/research/.artifacts/forceaudit31_live_delivery.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));node.destroy_node();rclpy.shutdown()
