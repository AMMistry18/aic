"""Prospective faster lateral travel; observation windows and guards unchanged."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.replay_registration47 import module,execute

def main():
    out=ROOT/'experiments/registration_development55';out.mkdir(exist_ok=True)
    if (out/'synthetic_replay.json').exists():raise FileExistsError('Preserve first replay')
    parent=ROOT/'experiments/candidates/registration47/contact_registration.py'
    path=ROOT/'experiments/candidates/registration55/contact_registration.py'
    path.parent.mkdir(exist_ok=True)
    source=parent.read_text()
    source=source.replace('length/.002','length/.004').replace('length/.010','length/.020')
    source=source.replace('peak_speed_bound_m_s=.002','peak_speed_bound_m_s=.004')
    source=source.replace('peak_acceleration_bound_m_s2=.010','peak_acceleration_bound_m_s2=.020')
    source=source.replace('measured-start quintic 47','measured-start quintic 55')
    path.write_text(source)
    methods={'baseline':module('base55',parent),'candidate':module('candidate55',path)}
    spans=json.loads((ROOT/'experiments/registration_development47/synthetic_replay.json').read_text())['development_spans']
    rows=[]
    for span in spans:
        for seed in range(5):
            rows.append(dict(case=span['case'],seed=47000+seed,
                results={k:execute(v,span['lo'],span['hi'],47000+seed) for k,v in methods.items()}))
    controls=[]
    for name,lo,hi,opts in [
        ('no_wall',[-.01,-.01],[.01,.01],{}),
        ('one_sided',[-.01,-.0005],[.0005,.0005],{}),
        ('entrapped',[-.0001,-.0001],[.0001,.0001],{}),
        ('stale_wrist',[-.0005,-.0005],[.0005,.0005],dict(stale=True)),
        ('overforce',[-.0005,-.0005],[.0005,.0005],dict(overforce=True))]:
        controls.append(dict(name=name,results={k:execute(v,lo,hi,47077,**opts) for k,v in methods.items()}))
    summary={k:dict(accepted=sum(r['results'][k]['observable'] for r in rows),attempted=len(rows),
                   median_synthetic_s=statistics.median(r['results'][k]['elapsed_s'] for r in rows),
                   maximum_pose_lead_m=max(r['results'][k]['maximum_pose_lead_m'] for r in rows)) for k in methods}
    passed=(all(r['results']['candidate']['observable'] for r in rows)
            and all(not v['observable'] for r in controls for v in r['results'].values())
            and summary['candidate']['maximum_pose_lead_m']<=.0005+1e-12)
    data=dict(scope='Synthetic controller logic only; no physical forces or insertion success claimed',
        final_cases_used=0,checks_passed=passed,physical_qualification=False,
        candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        summary=summary,negative_controls=controls,rows=rows)
    (out/'synthetic_replay.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(checks_passed=passed,summary=summary)))
    if not passed:raise RuntimeError('Candidate failed qualification')

if __name__=='__main__':main()
