"""One bounded real-policy SC pilot on the calibrated live diagnostic reset.

No scene/port/cable truth enters the policy. The setup load force is shared
through a research-only command adapter. This run is not a final ablation.
"""
import copy
import json
import hashlib
import os
from pathlib import Path
import threading
import time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from aic_model.aic_model import AicModel
from aic_task_interfaces.msg import Task

OUT=Path('/research/.artifacts/kilted_handoff')


class ResearchModel(AicModel):
    def __init__(self,bias):
        super().__init__();self.setup_load_bias=np.asarray(bias,float);self.commands=0

    def handle_motion_update(self,motion_update):
        from aic_model.insertion.contract import quat_to_matrix
        command=copy.deepcopy(motion_update)
        _,q=self._policy._tcp()
        force=quat_to_matrix(q).T@self.setup_load_bias
        command.feedforward_wrench_at_tip.force.x+=float(force[0])
        command.feedforward_wrench_at_tip.force.y+=float(force[1])
        command.feedforward_wrench_at_tip.force.z+=float(force[2])
        self.commands+=1
        return super().handle_motion_update(command)


def main():
    calibration=json.loads((OUT/'controller_hold_calibrated.json').read_text())
    if not calibration['gate_passed']:raise RuntimeError('Hold gate failed')
    world_hash=hashlib.sha256((OUT/'world.sdf').read_bytes()).hexdigest()
    bootstrap=json.loads((OUT/'controller_bootstrap.json').read_text())
    if calibration.get('world_sha256')!=world_hash or bootstrap.get('world_sha256')!=world_hash or not bootstrap.get('ready'):
        raise RuntimeError('Hold/bootstrap evidence does not match this world')
    reset=json.loads((OUT/'observed_reset_audit.json').read_text())
    if not reset.get('reset_gate_passed') or reset.get('world_sha256')!=world_hash:
        raise RuntimeError('Actual held-tip reset is not validated for this world')
    rclpy.init(args=['--ros-args','-p','policy:=aic_model.insertion.InsertionPolicy','-p','use_sim_time:=true'])
    node=ResearchModel(calibration['calibrated_force_base_N'])
    executor=MultiThreadedExecutor();executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    record={'purpose':'single_diagnostic_pilot_not_final_ablation','connector':'sc','target_index':2,
            'setup_load_bias_base_N':calibration['calibrated_force_base_N'],
            'ground_truth_used_by_policy':False,'verified_seating':None,
            'verification_status':'independent_contact_seating_recorder_not_validated'}
    record['world_sha256']=world_hash
    import torch
    record['cuda_available']=bool(torch.cuda.is_available())
    record['gpu_name']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    feedback=[]
    try:
        node.trigger_configure()
        if not hasattr(node,'_policy'):raise RuntimeError('Policy did not configure')
        # Warm SC outside the timed action, then require a fresh camera set.
        from aic_model.insertion.sc_controller import configure_sc_plug_pose
        from aic_model.insertion.sfp_controller import _observation_stamp_s
        if not configure_sc_plug_pose(node._policy):raise RuntimeError('SC model did not configure')
        node.trigger_activate()
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            obs=node.observation_callable()
            stamp=_observation_stamp_s(obs) if obs is not None else None
            if stamp is not None and 0<=node.get_clock().now().nanoseconds/1e9-stamp<=.35:break
            time.sleep(.05)
        else:raise TimeoutError('No camera observation within the unchanged 0.35 s age gate')
        task=Task();task.id='sc2_diagnostic_001';task.plug_type='sc';task.plug_name='sc_plug'
        task.cable_type='sfp_sc_cable';task.cable_name='cable_2';task.port_type='sc'
        task.port_name='sc_port_2';task.target_module_name='sc_port_2';task.time_limit=45
        start=time.monotonic()
        outcome=node._policy.insert_cable(task,node.observation_callable,node.move_robot,
            lambda message:feedback.append({'elapsed_wall_s':time.monotonic()-start,'message':str(message)}))
        record.update(policy_return=bool(outcome),wall_time_s=time.monotonic()-start,feedback=feedback,
                      motion_commands=node.commands)
    except Exception as exc:
        record['error']=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    finally:
        run_id=os.environ.get('AIC_RESEARCH_RUN_ID','sc_policy_pilot')
        if not run_id.replace('_','').isalnum():raise ValueError('Invalid run ID')
        (OUT/f'{run_id}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))
        node._tf_listener.executor.shutdown()
        node._tf_listener.dedicated_listener_thread.join(timeout=2)
        executor.shutdown();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
