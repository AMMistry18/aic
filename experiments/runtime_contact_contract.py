"""Audit the exact pinned binary before accepting its contact-report semantics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

AUDITED_BINARY='7339266ce1a8074f0011f9786a1550dd518e50f645a0fe0c6b8ebbbf85ca265c'

def runtime_contract(backend=None):
    backend=backend or os.environ.get('AIC_RESEARCH_PHYSICS_BACKEND','pinned_float32')
    if backend not in ('pinned_float32','source_float64','source_float64_sharp','source_float64_damped'):raise ValueError('Unknown audited backend')
    root=Path('/ws_aic/install');header=(root/'include/gz/physics8/gz/physics/config.hh').read_text()
    version=re.search(r'#define GZ_PHYSICS_VERSION_FULL "([^"]+)"',header).group(1)
    plugin=root/f'lib/libgz-physics8-bullet-featherstone-plugin.so.{version}'
    expected=AUDITED_BINARY;precision=32;binary_audit=None
    if backend in ('source_float64','source_float64_sharp','source_float64_damped'):
        audit_name={'source_float64':'bullet_double_binary_audit.json','source_float64_sharp':'bullet_sharp_binary_audit.json','source_float64_damped':'bullet_damped_binary_audit.json'}[backend]
        binary_audit=json.loads((Path(__file__).parent/'configs'/audit_name).read_text())
        plugin=Path(binary_audit['plugin_path']);precision=64
        for name,sha in binary_audit['binary_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise RuntimeError(f'Audited binary changed: {name}')
        expected=binary_audit['binary_sha256'][str(plugin.resolve())]
    digest=hashlib.sha256(plugin.read_bytes()).hexdigest()
    if version!='8.3.0' or digest!=expected:raise RuntimeError(f'Contact semantics not audited for binary {version} {digest}')
    return {'engine':'bullet-featherstone','gz_physics_version':version,'plugin_sha256':digest,
        'backend':backend,'scalar_precision_bits':precision,'plugin_path':str(plugin.resolve()),
        'binary_audit':binary_audit,'depth_convention':'signed_distance_negative_penetration',
        'collision_identity_resolution':'link_only; 8.3.0 selects an arbitrary collision name within each link',
        'contact_wrench_semantics':'not a force vector: 8.3.0 reports scalar impulse in xyz; use independent wrist FT',
        'source_url':'https://github.com/gazebosim/gz-physics/blob/gz-physics8_8.3.0/bullet-featherstone/src/SimulationFeatures.cc',
        'audit_date':'2026-10-26' if precision==64 else '2026-10-25'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=runtime_contract();a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
