"""Public canonical-cable swept OBB contact geometry; no live scene poses."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import source_model,sdf_frames
from experiments.audit_sc_collision_fit import boxes


def swept_interval(a,b,direction):
    _,A,ha=a;_,B,hb=b
    axes=np.array([*A[:3,:3].T,*B[:3,:3].T,*[np.cross(x,y) for x in A[:3,:3].T for y in B[:3,:3].T]])
    norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-8]/norm[norm>1e-8,None]
    radius=abs(axes@A[:3,:3])@ha+abs(axes@B[:3,:3])@hb
    separation=axes@(A[:3,3]-B[:3,3]);speed=axes@direction
    fixed=abs(speed)<1e-9
    if np.any(abs(separation[fixed])>radius[fixed]+1e-12):return None
    lo=(-radius[~fixed]-separation[~fixed])/speed[~fixed]
    hi=(radius[~fixed]-separation[~fixed])/speed[~fixed]
    first=float(np.max(np.minimum(lo,hi)));last=float(np.min(np.maximum(lo,hi)))
    return (first,last) if first<=last+1e-12 else None


def main():
    cable_path=ROOT/'.artifacts/lastinch_resources/cable.sdf'
    cable=ET.parse(cable_path).getroot().find('model');cf=sdf_frames(cable)
    port=source_model('NIC Card Mount 2');pf=sdf_frames(port);mouth=pf['sfp_port_0_link_entrance']
    targets=[b for link in port.findall('link') for b in boxes(link,pf[link.get('name')])]
    rows=[]
    for x,y in [(x,y) for y in [0.,.001,.002] for x in [0.,-.004,-.002,-.001,.001,.002,.004]]:
        tip=mouth.copy();tip[:3,3]+=mouth[:3,:3]@np.array([x,y,0.])
        active=[b for link in cable.findall('link') if link.get('name') in ('sfp_module_link','sfp_tip_link')
                for b in boxes(link,tip@np.linalg.inv(cf['sfp_tip_link'])@cf[link.get('name')])]
        hits=[]
        for a in active:
            for b in targets:
                interval=swept_interval(a,b,mouth[:3,2])
                if interval and -.015<=interval[0]<=.015:
                    hits.append(dict(first_depth_m=interval[0],last_depth_m=interval[1],plug_box=a[0],target_box=b[0]))
        hits.sort(key=lambda z:z['first_depth_m'])
        rows.append(dict(offset_xy_mm=[x*1000,y*1000],first_contact=hits[:3]))
    result=dict(purpose=__doc__,canonical_cable_sha256=hashlib.sha256(cable_path.read_bytes()).hexdigest(),
                geometry_only_not_physical_contact_qualification=True,rows=rows)
    out=ROOT/'.artifacts/sfp_visual33/public_axial_datum.json';out.write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:print(row['offset_xy_mm'],row['first_contact'][:1])


if __name__=='__main__':main()
