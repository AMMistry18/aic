"""Unpause only counts as successful once the observed clock advances."""
import json
import subprocess
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data


def resume(node,timeout=60):
    clocks=[]
    sub=node.create_subscription(Clock,'/clock',lambda m:clocks.append(m.clock.sec+m.clock.nanosec/1e9),qos_profile_sensor_data)
    start=time.monotonic()
    while not clocks and time.monotonic()-start<2:rclpy.spin_once(node,timeout_sec=.1)
    baseline=clocks[-1] if clocks else None
    attempts=0;next_request=0
    while time.monotonic()-start<timeout:
        if time.monotonic()>=next_request:
            subprocess.run(['gz','service','-s','/world/aic_lastinch/control',
                '--reqtype','gz.msgs.WorldControl','--reptype','gz.msgs.Boolean',
                '--timeout','5000','--req','pause: false'],check=True,capture_output=True,timeout=7)
            attempts+=1;next_request=time.monotonic()+2
        rclpy.spin_once(node,timeout_sec=.1)
        if baseline is None and clocks:baseline=clocks[-1]
        if clocks and baseline is not None and clocks[-1]>baseline+.01:break
    result={'baseline_sim_s':baseline,'last_sim_s':clocks[-1] if clocks else None,
            'unpause_requests':attempts,'clock_advancing':bool(clocks and baseline is not None and clocks[-1]>baseline+.01)}
    node.destroy_subscription(sub)
    return result


if __name__=='__main__':
    rclpy.init();node=Node('research_unpause_audit');result=resume(node)
    Path('/research/.artifacts/kilted_handoff/unpause_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));node.destroy_node();rclpy.shutdown()
