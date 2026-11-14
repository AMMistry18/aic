"""Explain an archived lip rejection using policy observations only."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    run='dev_45003_axial35v1_original_perception_sfp_contact_datum_handoff'
    p=json.loads((ROOT/'.artifacts/lastinch_runs'/run/run/'policy.json').read_text())
    samples=p['sfp_contact_datum']['axial_probe_samples']
    s=np.array([[r['sim_s'],r['measured_depth_m'],r['applied_command_m'],-r['force_port_N'][2]] for r in samples])
    counts=np.zeros(6,int);candidates=[];windows=0
    for i in range(len(s)):
        end=s[i,0];r=s[(s[:,0]>=end-.83)&(s[:,0]<=end)]
        if len(r)<20 or r[-1,0]-r[0,0]<.77:continue
        windows+=1;gap=float(np.max(np.diff(r[:,0])))
        advance=float(r[-1,2]-r[0,2]);travel=float(np.ptp(r[:,1]));lead=float(np.median(r[-8:,2]-r[-8:,1]))
        reaction=float(np.median(r[r[:,0]>=end-.2,3])-np.median(r[r[:,0]<=r[0,0]+.2,3]))
        checks=[gap<=.05,advance>=.0001,travel<=.000025,lead>=.00015,reaction>=.025]
        counts[:5]+=checks;counts[5]+=all(checks)
        if all(checks[:4]):candidates.append(dict(end_sim_s=end,command_advance_m=advance,
            observed_travel_m=travel,lead_m=lead,reaction_increase_N=reaction,maximum_gap_s=gap))
    result=dict(run=run,simulator_truth_inputs=False,full_windows=windows,
        pass_counts=dict(zip(['gap','advance','travel','lead','reaction','all'],map(int,counts))),
        kinematically_blocked_but_reaction_rejected=candidates,
        proposed_repair='One bounded unload and slower reload; retain the complete contact test and require measured release. Not yet deployed.')
    (ROOT/'.artifacts/repair36/lip_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
