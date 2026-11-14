"""Own-command diagnostic proxy for archived SFP waypoint failures.

The exception path commands a hold at its current estimated tip. For a final
waypoint command that had stopped changing, subtract that hold from the last
approach command. This is a timestamp-separated proxy, not a synchronous tip
measurement or an independently verified insertion verdict. No object/target
simulator transform is read.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=[]
    for batch in ['repair34/validation34v1','repair35/regression35v1']:
        state=json.loads((ROOT/'.artifacts'/batch/'progress.json').read_text())
        for result in state['results']:
            if result['connector']!='sfp':continue
            folder=ROOT/result['evidence_dir']/result['run']
            policy=json.loads((folder/'policy.json').read_text())
            datum=policy.get('sfp_contact_datum') or {}
            if datum.get('reason')!='SFP datum waypoint tracking timeout':continue
            commands=[r for r in policy['trace'] if r.get('phase')=='original_policy_motion_command']
            visual=policy['sfp_fixture_diagnostic']
            axes=Rotation.from_euler('z',visual['rim']['correction_mm_deg'][3],degrees=True).as_matrix()@visual['seed']['rotation']
            final,hold=commands[-2:]
            if len(commands)<12:continue
            variation=max(np.linalg.norm(np.array(c['target_tcp'])-final['target_tcp']) for c in commands[-12:-1])
            error=axes.T@(np.array(final['target_tcp'])-hold['target_tcp'])
            output.append(dict(run=result['run'],batch=batch,physical_success=result['verified_success'],
                component_reason=datum['reason'],phase=datum.get('phase'),
                final_command_variation_last_11_m=float(variation),
                proxy_command_minus_failure_hold_port_m=error.tolist(),
                command_hold_gap_sim_s=hold['sim_s']-final['sim_s'],
                force_peak_raw_N=result['high_rate_wrist_audit']['peak_tared_force_N'],
                limitation='Timestamp-separated TCP command/hold proxy; not exact synchronous tip error'))
    out=ROOT/'.artifacts/repair35/waypoint_diagnosis.json'
    out.write_text(json.dumps(dict(ground_truth_inputs=False,results=output),indent=2)+'\n',encoding='utf-8')
    for row in output:
        print(row['run'],np.round(np.array(row['proxy_command_minus_failure_hold_port_m'])*1000,6).tolist(),
              'variation_um',row['final_command_variation_last_11_m']*1e6)


if __name__=='__main__':main()
