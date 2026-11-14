"""Independent live diagnostic only; never imported/read by a policy."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def main():
    root=Path('/research/.artifacts/kilted_handoff')
    spec=json.loads((root/'scene.json').read_text());name=f"cable_{spec['reset']['target_index']}"
    mouth=np.array(spec['reset']['mouth_world']);rows=[]
    rclpy.init();node=Node('independent_sfp_diagnostic_snapshot')
    def receive(msg):
        frames={}
        for t in msg.transforms:
            v=t.transform;M=np.eye(4);M[:3,3]=[v.translation.x,v.translation.y,v.translation.z]
            M[:3,:3]=Rotation.from_quat([v.rotation.x,v.rotation.y,v.rotation.z,v.rotation.w]).as_matrix();frames[t.child_frame_id]=M
        if name not in frames or f'{name}/sfp_tip_link' not in frames:return
        relative=np.linalg.inv(mouth)@frames[name]@frames[f'{name}/sfp_tip_link']
        t=msg.transforms[0].header.stamp
        rows.append(dict(sim_s=t.sec+t.nanosec/1e9,relative_mm=(relative[:3,3]*1000).tolist(),
                         angle_deg=float(np.rad2deg(Rotation.from_matrix(relative[:3,:3]).magnitude()))))
    node.create_subscription(TFMessage,f'/model/{name}/pose',receive,100)
    started=time.monotonic()
    while len(rows)<10 and time.monotonic()-started<12:rclpy.spin_once(node,timeout_sec=.05)
    node.destroy_node();rclpy.shutdown()
    result=dict(diagnostic_only=True,policy_input=False,rows=rows)
    out=Path('/research/.artifacts/sfp_visual33/live_diagnostics');out.mkdir(exist_ok=True)
    p=out/(f"snapshot_{rows[-1]['sim_s']:.4f}.json" if rows else 'no_live_pose.json')
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
