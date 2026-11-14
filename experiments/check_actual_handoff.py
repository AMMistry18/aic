"""Evaluator-only actual cable pose audit; never imported by the policy."""
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.convert import message_to_ordereddict
from simulation_interfaces.srv import GetEntities, GetEntityState
from scipy.spatial.transform import Rotation

OUT=Path('/research/.artifacts/kilted_handoff')


def main():
    rclpy.init();node=Node('evaluator_only_handoff_audit')
    listing=node.create_client(GetEntities,'/gz_server/get_entities')
    getter=node.create_client(GetEntityState,'/gz_server/get_entity_state')
    def call(client,request):
        if not client.wait_for_service(timeout_sec=10):raise TimeoutError(client.srv_name)
        future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=15)
        if future.result() is None:raise TimeoutError(client.srv_name)
        return future.result()
    response=call(listing,GetEntities.Request())
    data=message_to_ordereddict(response)
    (OUT/'entities.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
    rows={}
    # Entity names are inspected from the simulator, not assumed to resolve.
    for name in data.get('entities',[]):
        if isinstance(name,str) and ('sc_tip_link' in name or name=='cable_2'):
            request=GetEntityState.Request();request.entity=name
            rows[name]=message_to_ordereddict(call(getter,request))
    (OUT/'actual_handoff.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
