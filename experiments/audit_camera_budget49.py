"""Prepare an uninstalled inspection-camera scheduling candidate, policy RGB intact."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'.artifacts/dev46_workspace/.artifacts/kilted_handoff/world.sdf'
    original=ET.parse(source).getroot();candidate=copy.deepcopy(original);rows=[]
    policy={'left_camera','center_camera','right_camera'};inspection={'overview','board','target'}
    for sensor in candidate.findall(".//sensor[@type='camera']"):
        name=sensor.get('name');old=float(sensor.findtext('update_rate'));width=int(sensor.findtext('camera/image/width'));height=int(sensor.findtext('camera/image/height'))
        new=1. if name in inspection else old
        if name not in policy|inspection:raise RuntimeError('Unexpected camera needs review')
        if name in inspection:sensor.find('update_rate').text='1'
        rows.append(dict(name=name,policy_input=name in policy,width=width,height=height,
                         before_hz=old,proposed_hz=new))
    restored=copy.deepcopy(candidate)
    for sensor in restored.findall(".//sensor[@type='camera']"):
        old=next(s for s in original.findall(".//sensor[@type='camera']") if s.get('name')==sensor.get('name'))
        sensor.find('update_rate').text=old.findtext('update_rate')
    identical=ET.tostring(restored)==ET.tostring(original)
    if not identical or len(rows)!=6 or any(r['before_hz']!=r['proposed_hz'] for r in rows if r['policy_input']):
        raise RuntimeError('A field besides the three inspection update rates changed')
    before=sum(r['width']*r['height']*r['before_hz'] for r in rows);after=sum(r['width']*r['height']*r['proposed_hz'] for r in rows)
    out=ROOT/'experiments/candidates/inspection49';out.mkdir(exist_ok=True)
    ET.ElementTree(candidate).write(out/'proposed_world.sdf',encoding='utf-8',xml_declaration=True)
    report=dict(scope='Uninstalled render-workload calculation, not a measured speedup',
        source_world_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),only_three_inspection_rates_differ=True,
        policy_images_geometry_calibration_force_physics_unchanged=True,
        before_Mpixels_per_sim_s=before/1e6,proposed_Mpixels_per_sim_s=after/1e6,
        pixel_workload_reduction_fraction=1-after/before,physical_and_runtime_qualification=False,cameras=rows)
    (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
