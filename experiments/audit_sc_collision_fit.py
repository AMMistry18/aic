"""Source-geometry OBB clearance audit, diagnostic only (boxes, not cylinders)."""
import json
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.lastinch_geometry import source_model,sdf_frames,parse_pose

def boxes(link,T):
    out=[]
    for c in link.findall('collision'):
        size=c.findtext('geometry/box/size')
        if size:out.append((c.get('name'),T@parse_pose(c),np.fromstring(size,sep=' ')/2))
    return out

def penetration(a,b):
    _,A,ha=a;_,B,hb=b
    axes=[*A[:3,:3].T,*B[:3,:3].T,*[np.cross(x,y) for x in A[:3,:3].T for y in B[:3,:3].T]]
    overlaps=[]
    for axis in axes:
        norm=np.linalg.norm(axis)
        if norm<1e-8:continue
        axis=axis/norm
        ra=np.sum(np.abs(axis@A[:3,:3])*ha);rb=np.sum(np.abs(axis@B[:3,:3])*hb)
        overlaps.append(ra+rb-abs(axis@(A[:3,3]-B[:3,3])))
    return min(overlaps)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--connector',choices=['sc','sfp'],default='sc');args=parser.parse_args()
    sc=args.connector=='sc'
    port=source_model('SC Port' if sc else 'NIC Card Mount 2');plug=source_model('SC Plug' if sc else 'SFP Module')
    pf=sdf_frames(port);sf=sdf_frames(plug)
    portboxes=boxes(port.find("link[@name='sc_port_link']" if sc else "link[@name='nic_card_link']"),pf['sc_port_link' if sc else 'nic_card_link'])
    result=[]
    for depth in ((0,.005,.007,.010,.012,.01564) if sc else (0,.01,.02,.04,.0458)):
        tip=pf['sc_port_base_link_entrance' if sc else 'sfp_port_0_link_entrance'].copy();tip[:3,3]+=tip[:3,2]*depth
        T=tip@np.linalg.inv(sf['sc_tip_link' if sc else 'sfp_tip_link'])
        plugboxes=boxes(plug.find("link[@name='sc_plug_link']" if sc else "link[@name='sfp_module_link']"),T)
        if not sc:plugboxes=[b for b in plugboxes if not b[0].startswith('port_collider_box')]
        if not sc:plugboxes+=boxes(plug.find("link[@name='sfp_tip_link']"),T@sf['sfp_tip_link'])
        collisions=[]
        for a in plugboxes:
            for b in portboxes:
                overlap=penetration(a,b)
                if overlap>1e-7:collisions.append({'plug':a[0],'port':b[0],'minimum_separating_translation_mm':overlap*1000})
        result.append({'depth_mm':depth*1000,'box_overlaps':collisions})
    out=ROOT/f'.artifacts/kilted_handoff/{args.connector}_collision_fit.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
