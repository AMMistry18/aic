"""Controller-logic comparison on a synthetic bounded actuator with DEV spans.

Recorded own-robot plateau positions parameterize synthetic walls. This is not
Gazebo, a physical force model, or an insertion/robustness success estimate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class SyntheticActuator:
    def __init__(self,lo,hi,seed,*,stale=False,overforce=False):
        self.lo=np.r_[lo,-1.];self.hi=np.r_[hi,1.];self.point=np.zeros(3);self.target=self.point.copy()
        self.time=0.;self.force=np.zeros(3);self.rng=np.random.default_rng(seed);self.stale=stale;self.overforce=overforce
        self.max_lead=0.;self.commands=0
    def read(self):return self.point.copy(),np.eye(3),self.force.copy(),self.time
    def send(self,target,rotation):
        self.max_lead=max(self.max_lead,float(np.linalg.norm(target-self.point)));self.target=target.copy();self.commands+=1
    def sleep(self,dt):
        equilibrium=np.clip(self.target,self.lo,self.hi)
        self.point=equilibrium+(self.point-equilibrium)*np.exp(-dt/.035)
        self.time+=dt
        self.force=-500*(self.target-equilibrium)+self.rng.normal(0,.2,3)
        if self.overforce:self.force[0]=7.
    def stamp(self):return 0. if self.stale else max(0.,self.time-.006)


def execute(implementation,lo,hi,seed,**options):
    plant=SyntheticActuator(lo,hi,seed,**options)
    try:
        center,records=implementation.register_lateral(read=plant.read,send=plant.send,sleep=plant.sleep,
            clock=lambda:plant.time,check_deadline=lambda:None,axes=np.eye(3),command_dt=.02,
            axial_preload_m=0,plateau_window_s=.4,wrench_stamp=plant.stamp)
        return dict(observable=center is not None,elapsed_s=plant.time,
            center=None if center is None else center.tolist(),maximum_pose_lead_m=plant.max_lead,
            records=records)
    except RuntimeError as exc:return dict(observable=False,error=str(exc),elapsed_s=plant.time,maximum_pose_lead_m=plant.max_lead)


def main():
    out=ROOT/'experiments/registration_development47';out.mkdir(exist_ok=True)
    if (out/'synthetic_replay.json').exists():raise FileExistsError('Preserve experiment version')
    baseline_path=ROOT/'aic_model/aic_model/insertion/contact_registration.py'
    candidate_path=ROOT/'experiments/candidates/registration47/contact_registration.py'
    implementations=dict(baseline=module('registration_baseline47',baseline_path),candidate=module('registration_candidate47',candidate_path))
    ledger=json.loads((ROOT/'experiments/recovery_20261112/historical_batch_replay/progress.json').read_text());spans=[]
    for row in ledger['results']:
        if row['case']['connector']!='sfp' or row.get('status')!='completed':continue
        path=ROOT/row['evidence_dir'].replace('\\','/')/row['run']/'policy.json'
        policy=json.loads(path.read_text());records=(policy.get('sfp_contact_datum') or {}).get('lateral_registration',[])
        if len(records)!=2 or not all(r.get('observable') for r in records):continue
        spans.append(dict(case=row['case']['case_id'],source_policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            lo=[r['negative_m_force_N'][0] for r in records],hi=[r['positive_m_force_N'][0] for r in records]))
    rows=[];controls=[]
    for span in spans:
        for seed in range(5):
            results={name:execute(impl,span['lo'],span['hi'],47000+seed) for name,impl in implementations.items()}
            rows.append(dict(development_case=span['case'],noise_seed=47000+seed,results=results))
    for name,lo,hi,options in [
        ('no_wall',[-.01,-.01],[.01,.01],{}),('one_sided',[-.01,-.0005],[.0005,.0005],{}),
        ('entrapped',[-.0001,-.0001],[.0001,.0001],{}),('stale_wrist',[-.0005,-.0005],[.0005,.0005],dict(stale=True)),
        ('overforce',[-.0005,-.0005],[.0005,.0005],dict(overforce=True))]:
        results={key:execute(impl,lo,hi,47077,**options) for key,impl in implementations.items()}
        controls.append(dict(name=name,results=results,passed=not any(r['observable'] for r in results.values())))
    summary={}
    for name in implementations:
        accepted=[r['results'][name] for r in rows if r['results'][name]['observable']]
        summary[name]=dict(observable=len(accepted),attempted=len(rows),
            median_synthetic_registration_s=None if not accepted else statistics.median(r['elapsed_s'] for r in accepted),
            maximum_pose_lead_m=max(r['results'][name]['maximum_pose_lead_m'] for r in rows))
    result=dict(scope='Synthetic controller-logic check only; times/forces are not Gazebo or real insertion measurements',
        ground_truth_simulator_inputs=False,final_cases_used=0,geometry_thresholds_unchanged=True,
        candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),development_spans=spans,
        summary=summary,negative_controls=controls,rows=rows,
        controls_passed=all(r['passed'] for r in controls),physical_qualification=False)
    (out/'synthetic_replay.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(summary=summary,controls_passed=result['controls_passed'],control_count=len(controls))),flush=True)
    if not result['controls_passed']:raise RuntimeError('Synthetic negative control unexpectedly accepted')


if __name__=='__main__':main()
