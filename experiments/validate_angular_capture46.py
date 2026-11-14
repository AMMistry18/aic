"""Cross-check analytical intervals with the existing independent scalar SAT."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import source_model,sdf_frames
from experiments.audit_sc_collision_fit import boxes,penetration


def main():
    folder=ROOT/'experiments/parallel_dev46';source=json.loads((folder/'angular_contact_feasibility.json').read_text())
    port=source_model('NIC Card Mount 2');plug=source_model('SFP Module');pf=sdf_frames(port);sf=sdf_frames(plug)
    port_boxes=boxes(port.find("link[@name='nic_card_link']"),pf['nic_card_link']);entrance=pf['sfp_port_0_link_entrance']
    rows=[]
    for row in source['rows']:
        if row['angle_deg'] not in (0.,2.):continue
        for interval in row['depth_measurements']:
            for label,offset,expected in [('inside',interval['midpoint_m'],False),
                    ('below_lower',interval['lower_m']-.000001,True),('above_upper',interval['upper_m']+.000001,True)]:
                vector=np.zeros(3);vector[row['tilt_axis']]=row['angle_deg']
                tip=entrance.copy();tip[:3,3]+=entrance[:3,2]*interval['depth_m']+entrance[:3,row['lateral_axis']]*offset
                tip[:3,:3]=entrance[:3,:3]@Rotation.from_rotvec(np.deg2rad(vector)).as_matrix()
                T=tip@np.linalg.inv(sf['sfp_tip_link'])
                body=boxes(plug.find("link[@name='sfp_module_link']"),T)
                body=[b for b in body if not b[0].startswith('port_collider_box')]
                body+=boxes(plug.find("link[@name='sfp_tip_link']"),T@sf['sfp_tip_link'])
                overlap=max(penetration(a,b) for a in body for b in port_boxes)
                rows.append(dict(axis=row['tilt_axis'],angle_deg=row['angle_deg'],depth_m=interval['depth_m'],
                    probe=label,maximum_overlap_m=float(overlap),expected_overlap=expected,
                    passed=bool((overlap>1e-8)==expected)))
    result=dict(scope='Independent scalar separating-axis verification of analytic intervals, not Gazebo validation',
                passed=all(r['passed'] for r in rows),checks=len(rows),rows=rows)
    (folder/'angular_contact_scalar_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if not result['passed']:raise RuntimeError('Analytical interval disagreement')


if __name__=='__main__':main()
