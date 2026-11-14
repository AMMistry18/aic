"""Stage exposure-TCP retention without mutating the running frozen study."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
name='aic_model/aic_model/insertion/sfp_fixture_refinement.py'
source=(ROOT/name).read_bytes();text=source.decode().replace('\r\n','\n')
old="    estimator=policy._sfp_plug_estimator;detections=estimator.detect_views(views)"
new="""    # Retain actual exposure-time robot poses before expensive inference/CAD
    # fitting can age them out of TF history. No current/latest-TF substitute.
    exposure_tcp={}
    for stamp in sorted({float(v.stamp_s) for v in views}):
        try:position,orientation=policy._tcp_at_stamp(stamp)
        except Exception as exc:
            raise ValueError(f'Exposure TCP unavailable at {stamp:.9f}s: {exc}') from exc
        exposure_tcp[stamp]=(np.asarray(position,dtype=float).copy(),np.asarray(orientation,dtype=float).copy())
    estimator=policy._sfp_plug_estimator;detections=estimator.detect_views(views)"""
assert text.count(old)==1;text=text.replace(old,new)
old="    tcp,quat=policy._tcp_at_stamp(measured.stamp_s)"
new="""    if float(measured.stamp_s) not in exposure_tcp:
        raise ValueError('Visual estimate stamp has no captured exposure TCP')
    tcp,quat=exposure_tcp[float(measured.stamp_s)]"""
assert text.count(old)==1;text=text.replace(old,new)
text=text.replace("        grasp_translation_from_RGB=True,", "        exposure_tcp_retained_before_inference=True,exposure_tcp_stamp_s=float(measured.stamp_s),\n        grasp_translation_from_RGB=True,")
out=ROOT/'.artifacts/repair40/exposure_candidate';out.mkdir(parents=True,exist_ok=True)
path=out/'sfp_fixture_refinement.py';path.write_bytes(text.encode())
(out/'candidate.json').write_text(json.dumps(dict(source=name,before_sha256=hashlib.sha256(source).hexdigest(),candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),installed=False),indent=2)+'\n')
print('Prepared exposure-time retention outside frozen production sources.')
