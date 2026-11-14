"""Exercise actual mouth-stop logic on a synthetic unilateral axial actuator.

No Gazebo truth or final cases are loaded. The actuator has first-order own
robot tracking, a unilateral stop, and independent 0.2 N wrist noise. This is
a controller check, not evidence of insertion success or realistic peak force.
Waypoint travel is excluded from the simulated axial-search timing.
"""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'aic_model'))
from aic_model.insertion.lip_search import stationary_search_candidate


def implementation(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    methods=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='DatumController')
    selected=[n for n in methods.body if isinstance(n,ast.FunctionDef) and n.name in ('mouth_stop','_lip_force_plateau')]
    namespace=dict(np=np,kinematic_lip_candidate=stationary_search_candidate,__package__='aic_model.insertion')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return {name:namespace[name] for name in ('mouth_stop','_lip_force_plateau')}


class SyntheticAxialActuator:
    def __init__(self,wall,seed,mode='normal'):
        self.wall=wall;self.rng=np.random.default_rng(seed);self.mode=mode
        self.time=0.;self.position=-.0015;self.target=self.position;self.noise=0.
        self._axial_tracking_bound_N=1.5;self._axial_tracking_force_N=0.
        self._last_wrist_stamp_s=0.;self.Rp=np.eye(3);self.port_pos=np.zeros(3);self.held_rotation=np.eye(3)
        self.diagnostic=dict(axial_probe_samples=[]);self.max_lead=0.;self.offset=0.
        self.policy=SimpleNamespace(sleep_for=self.sleep)

    def clock(self):return self.time

    def line(self,goal):
        self.position=float(goal[2]);self.target=self.position;self.offset=float(goal[0])

    def read(self):
        self._last_wrist_stamp_s=0. if self.mode=='stale' else self.time
        reaction=0.
        if self.position>=self.wall-1e-12:
            reaction=max(0.,500*(self.target-self.wall)+self._axial_tracking_force_N)
        if self.mode=='drift':reaction+=.5*self.time
        if self.mode=='overforce':reaction=7.
        return np.array([self.offset,0.,self.position]),np.eye(3),np.array([0.,0.,-reaction-self.noise]),self.time

    def send(self,target,rotation):
        if np.linalg.norm(self.read()[2])>=6:raise RuntimeError('Synthetic independent 6N guard')
        self.target=float(target[2]);self.max_lead=max(self.max_lead,abs(self.target-self.position))

    def sleep(self,dt):
        count=max(1,round(dt/.002));step=dt/count
        for _ in range(count):
            velocity=(500*(self.target-self.position)+self._axial_tracking_force_N)/100
            self.position=min(self.wall,self.position+velocity*step)
            self.time+=step
        self.noise=float(self.rng.normal(0,.2))


def execute(methods,wall,seed,mode='normal'):
    cls=type('ActualMouthLogic',(SyntheticAxialActuator,),methods)
    plant=cls(wall,seed,mode)
    try:
        result=plant.mouth_stop(.002)
        return dict(accepted=True,axial_logic_s=plant.time,maximum_lead_m=plant.max_lead,
                    estimated_stop_m=result['stop_depth_m'],diagnostic=plant.diagnostic)
    except RuntimeError as exc:
        return dict(accepted=False,reason=str(exc),axial_logic_s=plant.time,maximum_lead_m=plant.max_lead)


def main():
    out=ROOT/'experiments/lip_development51';out.mkdir(exist_ok=True)
    target=out/'synthetic_replay.json'
    if target.exists():raise FileExistsError('Preserve first replay')
    paths=dict(baseline=ROOT/'aic_model/aic_model/insertion/sfp_contact_datum.py',
               candidate=ROOT/'experiments/candidates/lip51/sfp_contact_datum.py')
    methods={k:implementation(v) for k,v in paths.items()};rows=[];controls=[]
    for wall in [-.001,0.,.001,.002]:
        for seed in range(8):
            rows.append(dict(synthetic_stop_m=wall,seed=51000+seed,
                results={k:execute(v,wall,51000+seed) for k,v in methods.items()}))
    for name,wall,mode in [('no_lip',1.,'normal'),('stale_sensor',0.,'stale'),
                           ('monotonic_drift',0.,'drift'),('overforce',0.,'overforce')]:
        results={k:execute(v,wall,51017,mode) for k,v in methods.items()}
        controls.append(dict(name=name,results=results,rejected_by_both=not any(x['accepted'] for x in results.values())))
    summary={k:dict(accepted=sum(r['results'][k]['accepted'] for r in rows),attempted=len(rows),
        median_synthetic_axial_s=statistics.median(r['results'][k]['axial_logic_s'] for r in rows),
        maximum_lead_m=max(r['results'][k]['maximum_lead_m'] for r in rows)) for k in methods}
    checks_passed=(all(x['rejected_by_both'] for x in controls)
        and all(r['results']['candidate']['accepted']==r['results']['baseline']['accepted'] for r in rows)
        and all(v['maximum_lead_m']<=.0003000001 for r in rows for v in r['results'].values()))
    result=dict(scope='Synthetic controller logic only; waypoint travel excluded; no Gazebo insertion or force claim',
        final_cases_used=0,physical_qualification=False,checks_passed=checks_passed,summary=summary,
        source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},rows=rows,negative_controls=controls)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,checks_passed=checks_passed)))
    if not checks_passed:raise RuntimeError('Synthetic checks failed; do not install')


if __name__=='__main__':main()
