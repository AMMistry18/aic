"""Post-trial error budget; truth is joined only after saved policy estimates.

This script never changes a verdict or supplies observations to the policy.
"""
import gzip
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT/'experiments/evidence/original_repair/persistent_screen32v1.json').read_text())
    base = np.eye(4); base[:3, :3] = np.diag([-1., -1., 1.]); base[:3, 3] = [-.2, .2, 0.]
    rows = []
    for run in summary['runs']:
        folder = ROOT/'.artifacts/lastinch_runs'/run['run']/run['run']
        policy = json.loads((folder/'policy.json').read_text())
        trace = json.load(gzip.open(folder/'truth_trace.json.gz', 'rt'))
        data = json.loads((folder/'body_inputs/000/inputs.json').read_text())
        stamp = max(v['stamp_s'] for v in data['views'])
        actual = min(trace, key=lambda r: abs(r['sim_time_s']-stamp))
        actual_tip = np.linalg.inv(base) @ np.array(actual['tip_world'])
        body = policy['sc_body_diagnostic']
        port = policy['sc_rim_diagnostics'][-1]
        scene = json.loads((folder.parent/'scene.json').read_text())
        truth_mouth = np.linalg.inv(base) @ np.array(scene['reset']['mouth_world'])
        error = np.array(body['position'])-actual_tip[:3, 3]
        port_error = np.array(port['position'])-truth_mouth[:3, 3]
        row = dict(case_id=run['case_id'], verified_success=run['verified_success'],
                   reason=run['reason'], raw_peak_N=run['raw_peak_N'],
                   body_accepted=body['accepted'], body_error_base_mm=(1000*error).tolist(),
                   final_rim_error_base_mm=(1000*port_error).tolist(),
                   relative_axial_bias_mm=float((truth_mouth[:3,:3].T@(error-port_error))[2]*1000),
                   body_loo_mm=(1000*np.array(body['leave_one_view_out_disagreement_m'])).tolist(),
                   estimated_terminal=policy.get('observable_terminal_diagnostic'),
                   truth_is_posthoc_only=True)
        rows.append(row)
    out = ROOT/'.artifacts/persistent32_diagnosis'; out.mkdir(exist_ok=True)
    (out/'error_budget.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    for row in rows: print(json.dumps(row))


if __name__ == '__main__': main()
