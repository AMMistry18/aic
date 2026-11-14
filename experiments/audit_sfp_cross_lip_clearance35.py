"""Public CAD sweep for a proposed shorter inter-probe clearance path.

Geometry diagnosis only, not deployed, no live scene or grasp pose inputs.
The first measured lip could supply an axial reference for free traverse;
the second lip remains mandatory before accepting the relative datum.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import source_model,sdf_frames
from experiments.audit_sc_collision_fit import boxes
from experiments.audit_sfp_axial_datum33 import swept_interval


def main():
    cable_path=ROOT/'.artifacts/lastinch_resources/cable.sdf'
    cable=ET.parse(cable_path).getroot().find('model');cf=sdf_frames(cable)
    port=source_model('NIC Card Mount 2');pf=sdf_frames(port);mouth=pf['sfp_port_0_link_entrance']
    targets=[b for link in port.findall('link') for b in boxes(link,pf[link.get('name')])]
    angles=[(0.,0.,0.)]+list(itertools.product([-.03,.03],repeat=3));results=[]
    for clearance in [.0005,.00075,.001]:
        for y in [-.001,0.,.001]:
            for angle in angles:
                tip=mouth.copy();tip[:3,3]+=mouth[:3,:3]@np.array([-.002,y,.0000535-clearance])
                tip[:3,:3]=mouth[:3,:3]@Rotation.from_euler('xyz',angle,degrees=True).as_matrix()
                active=[b for link in cable.findall('link') if link.get('name') in ('sfp_module_link','sfp_tip_link')
                        for b in boxes(link,tip@np.linalg.inv(cf['sfp_tip_link'])@cf[link.get('name')])]
                hits=[]
                for a in active:
                    for b in targets:
                        interval=swept_interval(a,b,mouth[:3,0])
                        if interval and interval[1]>=0 and interval[0]<=.004:
                            hits.append(dict(plug_box=a[0],target_box=b[0],traverse_interval_m=interval))
                results.append(dict(clearance_from_public_lip_m=clearance,y_m=y,angular_components_deg=angle,
                                    collides=bool(hits),hits=hits[:5]))
    out=ROOT/'.artifacts/repair35/inter_probe_clearance';out.mkdir(exist_ok=True)
    report=dict(geometry_only_not_physics_qualification=True,policy_changed=False,
                public_lip_depth_m=.0000535,traverse_x_range_m=[-.002,.002],
                canonical_cable_sha256=hashlib.sha256(cable_path.read_bytes()).hexdigest(),results=results)
    (out/'public_sweep.json').write_text(json.dumps(report,indent=2)+'\n')
    for clearance in [.0005,.00075,.001]:
        rows=[r for r in results if r['clearance_from_public_lip_m']==clearance]
        print(json.dumps(dict(clearance_m=clearance,geometric_sweeps=len(rows),colliding=sum(r['collides'] for r in rows))))


if __name__=='__main__':main()
