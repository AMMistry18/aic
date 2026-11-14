"""CAD-only local identifiability audit; not a physical angular controller.

Unknown tip translation and rotation are nuisance parameters. The observations
are opposed wall intervals at two depths, computed from the actual source
boxes rather than a uniform rectangular-channel approximation. Hypothetical
measurement noise is explicitly not a measured sensor accuracy.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import source_model,sdf_frames
from experiments.audit_sc_collision_fit import boxes
from experiments.probe_angular_capture46 import free_channel


class SourceGeometry:
    def __init__(self):
        self.port=source_model('NIC Card Mount 2');self.plug=source_model('SFP Module')
        pf=sdf_frames(self.port);sf=sdf_frames(self.plug)
        self.port_boxes=boxes(self.port.find("link[@name='nic_card_link']"),pf['nic_card_link'])
        self.entrance=pf['sfp_port_0_link_entrance'];self.tip=sf['sfp_tip_link']

    def observe(self,q):
        measurements=[]
        for depth in [.002,.010]:
            T=self.entrance.copy();T[:3,3]+=T[:3,:3]@(q[:3]*.001+np.array([0.,0.,depth]))
            T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(np.deg2rad(q[3:])).as_matrix()
            module=T@np.linalg.inv(self.tip)
            body=[b for b in boxes(self.plug.find("link[@name='sfp_module_link']"),module)
                  if not b[0].startswith('port_collider_box')]
            body+=boxes(self.plug.find("link[@name='sfp_tip_link']"),T)
            for axis in range(2):
                interval=free_channel(body,self.port_boxes,self.entrance[:3,axis])
                if not interval['observable']:raise ValueError('No unique bounded source bore interval')
                measurements += [1000*interval['lower_m'],1000*interval['upper_m']]
        return np.array(measurements)


def main():
    out=ROOT/'experiments/angular_development52';out.mkdir(exist_ok=True)
    target=out/'contact_angle_identifiability.json'
    if target.exists():raise FileExistsError('Preserve first audit')
    geometry=SourceGeometry();rows=[];began=time.monotonic()
    poses=[np.zeros(6),np.array([.2,-.2,.1,.5,0.,0.]),np.array([-.2,.1,-.1,0.,.5,0.]),
           np.array([0.,0.,0.,.3,-.3,.3]),np.array([0.,0.,0.,.3,-.3,-.3])]
    with threadpool_limits(limits=1):
        for q in poses:
            row=dict(parameters_mm_deg=q.tolist())
            try:
                base=geometry.observe(q);derivatives=[]
                for i in range(6):
                    delta=np.zeros(6);delta[i]=.001
                    derivatives.append((geometry.observe(q+delta)-geometry.observe(q-delta))/.002)
                J=np.array(derivatives).T;s=np.linalg.svd(J,compute_uv=False);rank=int(np.linalg.matrix_rank(J,tol=1e-6))
                row.update(observation_endpoints_mm=base.tolist(),jacobian_mm_per_mm_deg=J.tolist(),
                    singular_values_mixed_units=s.tolist(),rank=rank,parameter_count=6)
                if rank==6:
                    inverse=np.linalg.inv(J.T@J);row['hypothetical_independent_endpoint_noise']=[]
                    for sigma_um in [1.,5.,10.]:
                        sd=np.sqrt(np.diag(inverse))*(sigma_um*.001)
                        row['hypothetical_independent_endpoint_noise'].append(dict(sigma_um=sigma_um,
                            angular_95_marginal_halfwidth_deg=(1.96*sd[3:]).tolist(),
                            translation_95_marginal_halfwidth_mm=(1.96*sd[:3]).tolist()))
            except ValueError as exc:row['reason']=str(exc)
            rows.append(row)
            target.write_text(json.dumps(dict(scope='Analytic source CAD only; not Gazebo, contact sensing or runtime qualification',
                source_slot='NIC Card Mount 2, slot 0',depths_mm=[2,10],
                parameter_order=['tip_x_mm','tip_y_mm','tip_z_mm','rotvec_x_deg','rotvec_y_deg','rotvec_z_deg'],
                final_cases_used=0,hypothetical_noise_not_measured=True,controller_installed=False,rows=rows,
                elapsed_wall_s=time.monotonic()-began),indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k not in ['jacobian_mm_per_mm_deg','observation_endpoints_mm']}),flush=True)


if __name__=='__main__':main()
