"""Fast analytical angular-capture diagnostic using pinned source collision boxes.

Exact separating-axis translation intervals are evaluated at fixed poses.
This is a CAD feasibility calculation, not a Gazebo contact or insertion test.
It tests whether a proposed two-depth wall-center measurement is observable.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import source_model,sdf_frames
from experiments.audit_sc_collision_fit import boxes


def collision_interval(a,b,direction,limit=.002):
    _,A,ha=a;_,B,hb=b
    axes=np.array([*A[:3,:3].T,*B[:3,:3].T,*[np.cross(x,y) for x in A[:3,:3].T for y in B[:3,:3].T]])
    norms=np.linalg.norm(axes,axis=1);axes=axes[norms>1e-8]/norms[norms>1e-8,None]
    radius=np.abs(axes@A[:3,:3])@ha+np.abs(axes@B[:3,:3])@hb
    distance=axes@(A[:3,3]-B[:3,3]);slope=axes@direction
    fixed=abs(slope)<1e-10
    # Strict positive overlap only. Cached touching manifolds are irrelevant.
    if np.any(abs(distance[fixed])>=radius[fixed]-1e-10):return None
    lo=(-radius[~fixed]-distance[~fixed])/slope[~fixed]
    hi=(radius[~fixed]-distance[~fixed])/slope[~fixed]
    lower=max(-limit,float(np.minimum(lo,hi).max(initial=-limit)))
    upper=min(limit,float(np.maximum(lo,hi).min(initial=limit)))
    return (lower,upper) if lower<upper else None


def free_channel(plug,port,direction,limit=.002):
    blocked=[v for a in plug for b in port if (v:=collision_interval(a,b,direction,limit)) is not None]
    merged=[]
    for lower,upper in sorted(blocked):
        if merged and lower<=merged[-1][1]:merged[-1][1]=max(upper,merged[-1][1])
        else:merged.append([lower,upper])
    free=[];last=-limit
    for lower,upper in merged:
        if lower>last:free.append([last,lower])
        last=max(last,upper)
    if last<limit:free.append([last,limit])
    # Do not select an external disconnected gap: require a bounded bore
    # interval whose midpoint remains within 1 mm of the entrance center.
    qualified=[v for v in free if v[0]>-limit+1e-9 and v[1]<limit-1e-9 and abs(np.mean(v))<.001]
    if len(qualified)!=1:return dict(observable=False,free_intervals_m=free)
    interval=qualified[0]
    return dict(observable=True,lower_m=interval[0],upper_m=interval[1],
                midpoint_m=float(np.mean(interval)),span_m=interval[1]-interval[0])


def main():
    port=source_model('NIC Card Mount 2');plug=source_model('SFP Module')
    pf=sdf_frames(port);sf=sdf_frames(plug)
    port_boxes=boxes(port.find("link[@name='nic_card_link']"),pf['nic_card_link'])
    entrance=pf['sfp_port_0_link_entrance'];rows=[]
    for axis in [0,1]:
        lateral_axis=1-axis
        for angle in [0.,.03,.1,.25,.5,1.,2.]:
            for sign in ([1] if angle==0 else [-1,1]):
                vector=np.zeros(3);vector[axis]=sign*angle
                R=Rotation.from_rotvec(np.deg2rad(vector)).as_matrix();measurements=[]
                for depth in [.002,.010]:
                    tip=entrance.copy();tip[:3,3]+=entrance[:3,2]*depth
                    tip[:3,:3]=entrance[:3,:3]@R;T=tip@np.linalg.inv(sf['sfp_tip_link'])
                    body=boxes(plug.find("link[@name='sfp_module_link']"),T)
                    body=[b for b in body if not b[0].startswith('port_collider_box')]
                    body+=boxes(plug.find("link[@name='sfp_tip_link']"),T@sf['sfp_tip_link'])
                    measured=free_channel(body,port_boxes,entrance[:3,lateral_axis]);measured['depth_m']=depth
                    measurements.append(measured)
                result=dict(tilt_axis=axis,angle_deg=sign*angle,lateral_axis=lateral_axis,depth_measurements=measurements)
                if all(m['observable'] for m in measurements):
                    delta=measurements[1]['midpoint_m']-measurements[0]['midpoint_m']
                    result.update(two_depth_center_shift_um=delta*1e6,
                        uniform_straight_channel_formula_angle_deg=float(np.rad2deg(np.arctan(2*delta/.008))))
                rows.append(result)
    noise=[]
    for sigma_um in [1,5,10,25]:
        noise.append(dict(hypothetical_independent_center_sigma_um=sigma_um,depth_separation_mm=8,
            angular_95_halfwidth_deg=float(np.rad2deg(1.96*2*np.sqrt(2)*sigma_um*1e-6/.008))))
    out=ROOT/'experiments/parallel_dev46/angular_contact_feasibility.json'
    out.write_text(json.dumps(dict(scope='Analytic pinned-source collision-box diagnostic only; no simulated contact/insertion claim',
        final_cases_read=0,noise_is_hypothetical_not_measured=True,noise_sensitivity=noise,rows=rows),indent=2))
    print(json.dumps(dict(cases=len(rows),both_depths_have_bore=sum(all(m['observable'] for m in r['depth_measurements']) for r in rows),
                         noise_sensitivity=noise)))


if __name__=='__main__':main()
