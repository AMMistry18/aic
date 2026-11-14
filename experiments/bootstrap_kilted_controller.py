"""Load controllers while paused; step only to finish their atomic activation."""
import json
import hashlib
from pathlib import Path
import subprocess
import time
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import LoadController, ConfigureController, SwitchController, ListControllers
from std_msgs.msg import String

OUT=Path('/research/.artifacts/kilted_handoff')


def main():
    rclpy.init();node=Node('research_paused_controller_bootstrap')
    grasp=[]
    node.create_subscription(String,'/lastinch/grasp_state',lambda m:grasp.append(m.data),10)
    def call(kind,name,request):
        client=node.create_client(kind,f'/controller_manager/{name}')
        if not client.wait_for_service(timeout_sec=40):raise TimeoutError(name)
        future=client.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=15)
        if future.result() is None:raise TimeoutError(name)
        return future.result()
    names=['joint_state_broadcaster','fts_broadcaster','aic_controller']
    existing={c.name:c.state for c in call(ListControllers,'list_controllers',ListControllers.Request()).controller}
    for name in names:
        if existing.get(name)=='active':continue
        if name in existing:raise RuntimeError(f'Unexpected existing controller state: {name}={existing[name]}')
        request=LoadController.Request();request.name=name
        if not call(LoadController,'load_controller',request).ok:raise RuntimeError(f'Load {name}')
        request=ConfigureController.Request();request.name=name
        if not call(ConfigureController,'configure_controller',request).ok:raise RuntimeError(f'Configure {name}')
    client=node.create_client(SwitchController,'/controller_manager/switch_controller')
    client.wait_for_service(timeout_sec=10)
    request=SwitchController.Request();request.activate_controllers=[n for n in names if existing.get(n)!='active'];request.strictness=2
    request.activate_asap=True;request.timeout.sec=120
    future=client.call_async(request) if request.activate_controllers else None
    steps=0
    activation_deadline=time.monotonic()+120
    while future is not None and not future.done() and time.monotonic()<activation_deadline:
        rclpy.spin_once(node,timeout_sec=.05)
        if future.done():break
        subprocess.run(['gz','service','-s','/world/aic_lastinch/control',
            '--reqtype','gz.msgs.WorldControl','--reptype','gz.msgs.Boolean',
            '--timeout','5000','--req','pause: true multi_step: 1'],check=True,capture_output=True,timeout=7)
        steps+=1
    if future is not None and (not future.done() or not future.result().ok):raise RuntimeError('Atomic activation did not finish')
    if future is None:
        # Revalidate an already active controller after an interrupted bootstrap.
        end=time.monotonic()+5
        while not grasp and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
    while (not grasp or grasp[-1]!='grasp_attached_startup_fixture_removed') and steps<25:
        subprocess.run(['gz','service','-s','/world/aic_lastinch/control',
            '--reqtype','gz.msgs.WorldControl','--reptype','gz.msgs.Boolean',
            '--timeout','5000','--req','pause: true multi_step: 1'],check=True,capture_output=True,timeout=7)
        steps+=1;rclpy.spin_once(node,timeout_sec=.1)
    if not grasp or grasp[-1]!='grasp_attached_startup_fixture_removed':raise RuntimeError('Grasp reset fixture did not release')
    listed=call(ListControllers,'list_controllers',ListControllers.Request())
    states={c.name:c.state for c in listed.controller}
    if any(states.get(name)!='active' for name in names):raise RuntimeError(states)
    from resume_kilted import resume
    clock=resume(node)
    if not clock['clock_advancing']:raise RuntimeError('Unpause accepted but simulation clock did not advance')
    result={'initially_paused':True,'activation_physics_steps':steps,'activation_timeout_s':120,'controllers':states,'ready':True,
            'grasp_state':grasp[-1],'clock':clock,
            'world_sha256':hashlib.sha256((OUT/'world.sdf').read_bytes()).hexdigest()}
    (OUT/'controller_bootstrap.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'controller_bootstrap.json').write_text(json.dumps({'ready':False,'error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n')
        raise
