"""Audit proposed 0.75 mm inter-probe clearance for all ten public NIC slots."""
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
    path=ROOT/'.artifacts/lastinch_resources/cable.sdf';cable=ET.parse(path).getroot().find('model');cf=sdf_frames(cable)
    groups={};members=[]
    for index in range(5):
        model=source_model(f'NIC Card Mount {index}');frames=sdf_frames(model)
        for slot in range(2):
            M=np.linalg.inv(frames[f'sfp_port_{slot}_link_entrance'])
            target=[b for link in model.findall('link') for b in boxes(link,M@frames[link.get('name')])]
            geometry=sorted(tuple(np.round(np.r_[b[1].ravel(),b[2]],9)) for b in target)
            key=hashlib.sha256(np.array(geometry).tobytes()).hexdigest()
            groups.setdefault(key,target)
            members.append(dict(nic=index,slot=slot,geometry_sha256=key,
                public_model_xml_sha256=hashlib.sha256(ET.tostring(model)).hexdigest()))
    results=[]
    for key,target in groups.items():
        for y in [-.001,0.,.001]:
            for angle in [(0.,0.,0.)]+list(itertools.product([-.1,.1],repeat=3)):
                tip=np.eye(4);tip[:3,3]=[-.002,y,.0000535-.00075]
                tip[:3,:3]=Rotation.from_euler('xyz',angle,degrees=True).as_matrix()
                active=[b for link in cable.findall('link') if link.get('name') in ('sfp_module_link','sfp_tip_link')
                    for b in boxes(link,tip@np.linalg.inv(cf['sfp_tip_link'])@cf[link.get('name')])]
                hits=[]
                for a in active:
                    for b in target:
                        interval=swept_interval(a,b,np.array([1.,0.,0.]))
                        if interval and interval[1]>=0 and interval[0]<=.004:
                            hits.append(dict(plug=a[0],target=b[0],interval_m=interval))
                results.append(dict(geometry_sha256=key,y_m=y,angular_components_deg=angle,collides=bool(hits),hits=hits[:5]))
        print(json.dumps(dict(geometry=key,completed_sweeps=sum(r['geometry_sha256']==key for r in results))),flush=True)
    out=ROOT/'.artifacts/repair36';out.mkdir(exist_ok=True)
    report=dict(geometry_only=True,live_scene_inputs=False,clearance_from_measured_lip_m=.00075,
        traverse_x_range_m=[-.002,.002],members=members,unique_geometries=len(groups),results=results,
        cable_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'cross_lip_clearance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(slots=len(members),unique_geometries=len(groups),sweeps=len(results),colliding=sum(r['collides'] for r in results))))


if __name__=='__main__':main()
