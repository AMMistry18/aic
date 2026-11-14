"""Offline truth comparison only; never supplies a policy input or correction."""
import bisect
import gzip
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
run='dev_45002_tracking36v1_original_perception_sfp_contact_datum_handoff'
folder=ROOT/'.artifacts/lastinch_runs'/run/run
policy=json.loads((folder/'policy.json').read_text())
truth=json.load(gzip.open(folder/'truth_trace.json.gz','rt'))
times=[r['sim_time_s'] for r in truth]
diagnostic=policy['sfp_contact_datum']


def comparison(samples,depth_key):
    rows=[]
    for s in samples:
        t=s['sim_s'];j=bisect.bisect_left(times,t)
        candidates=truth[max(0,j-1):j+1]
        r=min(candidates,key=lambda r:abs(r['sim_time_s']-t))
        if abs(r['sim_time_s']-t)>.02:continue
        rows.append((s[depth_key]-r['depth_m'])*1000)
    return dict(samples=len(rows),median_estimate_minus_independent_depth_mm=float(np.median(rows)),
                minimum_mm=float(np.min(rows)),maximum_mm=float(np.max(rows)))


before=[]
for offset in [.002,-.002]:
    samples=[s for s in diagnostic['axial_probe_samples'] if s['offset_m']==offset]
    last=samples[-1]['sim_s']
    before.append(dict(offset_m=offset,**comparison([s for s in samples if s['sim_s']>=last-.2],'measured_depth_m')))
travel=diagnostic['insertion_samples'];last=travel[-1]['sim_s']
after=comparison([s for s in travel if s['sim_s']>=last-1.],'estimated_depth_m')
result=dict(run=run,offline_ground_truth_diagnostic_only=True,policy_ground_truth_inputs=False,
            scope='relative depth estimate, not global mouth or grasp pose accuracy',
            rim_residual_px=policy['sfp_fixture_diagnostic']['rim']['residual_px'],
            relative_axial_correction_mm=diagnostic['axial_relative_datum_correction_m']*1000,
            before_datum_at_each_lip=before,after_datum_at_terminal=after)
(ROOT/'.artifacts/repair36/relative_depth_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
