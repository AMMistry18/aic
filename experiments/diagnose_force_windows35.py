"""Offline force-sampling diagnosis, never a policy input or insertion verdict.

Uses recorded wrist sensor stamps, calibrated robot TCP orientation and visual
port axes. No simulator object/target/grasp transform enters this comparison.
Longer windows are hypotheses until prospective stationary-probe qualification.
"""
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def main():
    state=json.loads((ROOT/'.artifacts/repair34/validation34v1/progress.json').read_text());results=[]
    for result in state['results']:
        if result['case']['connector']!='sfp':continue
        folder=ROOT/result['evidence_dir'];policy=json.loads((folder/result['run']/'policy.json').read_text())
        datum=policy.get('sfp_contact_datum') or {};records=datum.get('lateral_registration') or []
        if not records:continue
        hold=json.loads((folder/'controller_hold_calibrated.json').read_text())
        Rtcp=Rotation.from_quat(hold['samples'][-1]['quaternion_xyzw']).as_matrix()
        visual=policy['sfp_fixture_diagnostic'];Rp=Rotation.from_euler('z',visual['rim']['correction_mm_deg'][3],degrees=True).as_matrix()@np.array(visual['seed']['rotation'])
        # The tracker records P_lateral @ (target_TCP - observed_TCP).
        # Pair it with its own outgoing command to reconstruct observed robot
        # lateral position. This does not use the simulator object trajectory.
        tracking=(policy.get('lateral_tracking_compensation') or {}).get('trace',[])
        commands=[r for r in policy['trace'] if r.get('phase')=='original_policy_motion_command']
        paired=len(tracking)==len(commands) and bool(tracking)
        pair_gap=max((abs(a['sim_s']-b['sim_s']) for a,b in zip(tracking,commands)),default=float('inf'))
        paired=paired and pair_gap<=.01
        robot_stamps=np.array([r['sim_s'] for r in tracking]) if paired else np.array([])
        robot_xy=np.array([Rp[:,:2].T@(np.array(b['target_tcp'])-np.array(a['error_base_m']))
                           for a,b in zip(tracking,commands)]) if paired else np.empty((0,2))
        messages=json.load(gzip.open(folder/result['run']/'wrench.json.gz','rt'))
        # Sensor-stamp deduplication, not controller loop timestamps.
        unique={m['header']['stamp']['sec']+m['header']['stamp']['nanosec']*1e-9:m for m in messages}
        stamps=np.array(sorted(unique));force=np.array([[unique[t]['wrench']['force'][k] for k in 'xyz'] for t in stamps])@(Rp.T@Rtcp).T
        axes=[]
        for record in records:
            axis=record['axis'];checks=record['plateau_checks'];ends=[checks[-2]['end_wrench_stamp_s'],checks[-1]['end_wrench_stamp_s']]
            comparisons=[]
            for window in [.2,.4,.6]:
                sides=[force[(stamps>=end-window)&(stamps<=end),axis] for end in ends]
                if min(map(len,sides))<5:continue
                delta=abs(sides[0].mean()-sides[1].mean());sem=3*np.hypot(*(s.std(ddof=1)/np.sqrt(len(s)) for s in sides))
                motion=[]
                for end in ends:
                    selected=(robot_stamps>=end-window)&(robot_stamps<=end)
                    t=robot_stamps[selected];xy=robot_xy[selected]
                    if len(t)<5:
                        motion.append(dict(qualified=False,reason='missing_paired_robot_observations'));continue
                    coverage=float(t[-1]-t[0]);t=t-t.mean()
                    velocity=np.sum(t[:,None]*(xy-xy.mean(0)),axis=0)/max(float(t@t),1e-12)
                    span=float(np.linalg.norm(np.ptp(xy,axis=0)))
                    motion.append(dict(qualified=bool(coverage>=window-.06 and np.linalg.norm(velocity)<=.0001),
                        samples=len(t),observed_span_s=coverage,lateral_velocity_m_s=velocity.tolist(),
                        lateral_peak_to_peak_m=span))
                correlations=[]
                for s in sides:
                    variance=float(np.sum((s-s.mean())**2))
                    correlations.append(None if variance==0 else float(np.sum((s[:-1]-s.mean())*(s[1:]-s.mean()))/variance))
                comparisons.append(dict(window_s=window,unique_sensor_samples=list(map(len,sides)),force_difference_N=float(delta),
                    force_difference_3sem_N=float(sem),passes_same_3sem_test=bool(delta>sem and delta>.05),
                    paired_own_robot_lateral_stationarity=motion,
                    own_robot_command_pair_max_gap_s=pair_gap if paired else None,
                    lag1_force_autocorrelation=correlations,
                    stationary_over_whole_window_proven=False,
                    limitation='Lateral robot stationarity only; no full tip orientation/axial history; SEM is descriptive, not calibrated significance'))
            axes.append(dict(axis=axis,runtime_observable=record['observable'],comparison=comparisons))
        results.append(dict(run=result['run'],axes=axes))
    out=ROOT/'.artifacts/repair35/force_window_diagnosis';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(dict(GT_used=False,prospective_validation_required=True,results=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
