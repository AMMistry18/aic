"""Check admissibility against known feasible projections and arbitrary noise."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.candidates.efficiency48.epipolar_bound import mean_residual_lower_bound


def main():
    rng=np.random.default_rng(48001);checks=[];K=np.array([[900.,0,576],[0,900,512],[0,0,1.]])
    for index in range(1000):
        points=rng.uniform(-.1,.1,(8,3));points[:,2]+=.8;views=[];cost=0.;count=0
        for camera in range(2+index%2):
            T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(rng.normal(0,.1,3)).as_matrix();T[:3,3]=rng.normal(0,.08,3)
            q=(K@(T[:3,:3]@points.T+T[:3,3,None])).T;exact=q[:,:2]/q[:,2:]
            conf=rng.uniform(0,1,8);noise=rng.normal(0,rng.choice([.1,1,5,20]),(8,2));good=conf>=.2
            cost+=float(np.sum(abs(noise[good])*np.sqrt(conf[good,None])));count+=2*sum(good)
            views.append(dict(kps=exact+noise,kp_conf=conf,K=K,T=T))
        bound=mean_residual_lower_bound(views);feasible=cost/max(count,1)
        checks.append(dict(lower_bound_px=bound,known_feasible_cost_px=feasible,passed=bool(bound<=feasible+1e-8)))
    out=ROOT/'experiments/perception_efficiency48/admissibility_checks.json'
    result=dict(scope='Synthetic mathematical check, not insertion testing',checks=len(checks),
                passed=all(c['passed'] for c in checks),maximum_excess_px=max(c['lower_bound_px']-c['known_feasible_cost_px'] for c in checks))
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    if not result['passed']:raise RuntimeError('Lower bound exceeded a known feasible fit')


if __name__=='__main__':main()
